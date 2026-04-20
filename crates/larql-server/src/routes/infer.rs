//! POST /v1/infer — full forward pass with attention.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    #[serde(default = "default_top")]
    pub top: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
}

fn default_top() -> usize { 5 }
fn default_mode() -> String { "fast".into() }

/// Extract session ID from headers.
fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

fn run_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let has_gguf = model.gguf.is_some();
    if model.infer_disabled && !has_gguf {
        return Err(ServerError::InferenceUnavailable(
            "inference disabled (--no-infer)".into(),
        ));
    }

    // GGUF source supplies its own weights — skip the vindex check.
    if !has_gguf
        && !model.config.has_model_weights
        && model.config.extract_level != larql_vindex::ExtractLevel::Inference
        && model.config.extract_level != larql_vindex::ExtractLevel::All
    {
        return Err(ServerError::InferenceUnavailable(
            "vindex does not contain model weights. Rebuild with --include-weights".into(),
        ));
    }

    let encoding = model
        .tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();

    let is_compare = req.mode == "compare";
    let use_fast = req.mode == "fast";
    let use_walk = req.mode == "walk" || is_compare;
    let use_dense = req.mode == "dense" || is_compare;

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));

    // Fast path with GGUF weight source — bypasses vindex weight loading.
    // The GGUF pipeline holds attention/FFN weights mmap'd from disk; we
    // still hold the inference lock to serialize KV cache use.
    if use_fast {
        if let Some(gguf) = model.gguf.as_ref() {
            let backend = model.get_or_init_backend();
            let _guard = match model.inference_lock.lock() {
                Ok(g) => g,
                Err(p) => p.into_inner(),
            };
            backend.reset_kv_cache();
            let fast_start = std::time::Instant::now();
            let preds = gguf.predict_top_k(&token_ids, req.top, &**backend, &model.tokenizer);
            let fast_ms = fast_start.elapsed().as_secs_f64() * 1000.0;

            let predictions: Vec<serde_json::Value> = preds.iter().map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            }).collect();
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!("fast"));
            result.insert("source".into(), serde_json::json!("gguf"));
            result.insert("fast_ms".into(), serde_json::json!((fast_ms * 10.0).round() / 10.0));
            result.insert("knn_override".into(), serde_json::json!(false));
            let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
            result.insert("latency_ms".into(), serde_json::json!((latency_ms * 10.0).round() / 10.0));
            return Ok(serde_json::Value::Object(result));
        }
    }

    let weights = model
        .get_or_load_weights()
        .map_err(ServerError::InferenceUnavailable)?;

    if use_fast {
        // Fast path: Metal f32 forward + KNN overlay consult. Sub-second
        // prefill after the server's warmup pass. This is the route "ask"
        // in bench_interactive routes through. When `walk_only` is set, the
        // FFN backend swaps from dense WeightFfnGpu to the sparse Q4_0
        // WalkFfn (uses `interleaved_q4.bin` mmap), trading ~2× decode
        // speed for ~7 GB less RSS.
        let run_fast = |patched: &larql_vindex::PatchedVindex| {
            let backend = model.get_or_init_backend();
            // Empty CachedLayerGraph — we don't actually cache any layers
            // here; passing `build(&[])` would still invoke the FFN for the
            // first layer, which panics in walk-only mode (dropped FFN
            // tensors). `from_residuals` gives us an empty cache cheaply.
            let cache = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
            let knn_opt = if patched.knn_store.is_empty() {
                None
            } else { Some(&patched.knn_store) };

            let walk_ffn_opt = if model.walk_only {
                Some(larql_inference::WalkFfn::new_with_backend(
                    weights, patched.base(), 1024, &**backend,
                ))
            } else { None };
            let ffn_override: Option<&dyn larql_inference::ffn::FfnBackend> =
                walk_ffn_opt.as_ref().map(|w| w as &dyn larql_inference::ffn::FfnBackend);

            let _guard = match model.inference_lock.lock() {
                Ok(g) => g,
                Err(poisoned) => poisoned.into_inner(),
            };
            backend.reset_kv_cache();
            let fast_start = std::time::Instant::now();
            let pred = larql_inference::predict_honest_with_knn_ffn(
                weights, &model.tokenizer, &token_ids, req.top,
                patched.base(), &**backend, &cache,
                0..weights.num_layers, knn_opt, ffn_override,
            );
            let fast_ms = fast_start.elapsed().as_secs_f64() * 1000.0;
            (pred, fast_ms)
        };

        let (pred, fast_ms) = if let Some(sid) = session_id {
            let sessions = state.sessions.sessions_blocking_write();
            if let Some(session) = sessions.get(sid) {
                run_fast(&session.patched)
            } else {
                drop(sessions);
                let patched = model.patched.blocking_read();
                run_fast(&patched)
            }
        } else {
            let patched = model.patched.blocking_read();
            run_fast(&patched)
        };

        let predictions: Vec<serde_json::Value> = pred
            .predictions
            .iter()
            .map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            })
            .collect();
        let knn_override = pred.predictions.first()
            .map(|(s, _)| s.contains("KNN override"))
            .unwrap_or(false);

        result.insert("predictions".into(), serde_json::json!(predictions));
        result.insert("mode".into(), serde_json::json!("fast"));
        result.insert("fast_ms".into(), serde_json::json!((fast_ms * 10.0).round() / 10.0));
        result.insert("knn_override".into(), serde_json::json!(knn_override));

        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        result.insert("latency_ms".into(), serde_json::json!((latency_ms * 10.0).round() / 10.0));
        return Ok(serde_json::Value::Object(result));
    }

    // Helper: run walk inference against a PatchedVindex
    let run_walk = |patched: &larql_vindex::PatchedVindex| {
        let walk_ffn = larql_inference::WalkFfn::new(weights, patched, 8092);
        let walk_start = std::time::Instant::now();
        let pred = larql_inference::predict_with_ffn(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
            &walk_ffn,
        );
        let walk_ms = walk_start.elapsed().as_secs_f64() * 1000.0;
        (pred, walk_ms)
    };

    if use_walk {
        let (pred, walk_ms) = if let Some(sid) = session_id {
            // Session-scoped: use session's PatchedVindex
            let sessions = state.sessions.sessions_blocking_write();
            if let Some(session) = sessions.get(sid) {
                run_walk(&session.patched)
            } else {
                drop(sessions);
                let patched = model.patched.blocking_read();
                run_walk(&patched)
            }
        } else {
            let patched = model.patched.blocking_read();
            run_walk(&patched)
        };

        let predictions: Vec<serde_json::Value> = pred
            .predictions
            .iter()
            .map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            })
            .collect();

        if is_compare {
            result.insert("walk".into(), serde_json::json!(predictions));
            result.insert("walk_ms".into(), serde_json::json!((walk_ms * 10.0).round() / 10.0));
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!("walk"));
        }
    }

    if use_dense {
        let dense_start = std::time::Instant::now();
        let pred = larql_inference::predict(
            weights,
            &model.tokenizer,
            &token_ids,
            req.top,
        );
        let dense_ms = dense_start.elapsed().as_secs_f64() * 1000.0;

        let predictions: Vec<serde_json::Value> = pred
            .predictions
            .iter()
            .map(|(tok, prob)| {
                serde_json::json!({
                    "token": tok,
                    "probability": (*prob * 10000.0).round() / 10000.0,
                })
            })
            .collect();

        if is_compare {
            result.insert("dense".into(), serde_json::json!(predictions));
            result.insert("dense_ms".into(), serde_json::json!((dense_ms * 10.0).round() / 10.0));
        } else {
            result.insert("predictions".into(), serde_json::json!(predictions));
            result.insert("mode".into(), serde_json::json!("dense"));
        }
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    result.insert("latency_ms".into(), serde_json::json!((latency_ms * 10.0).round() / 10.0));

    Ok(serde_json::Value::Object(result))
}

pub async fn handle_infer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || run_infer(&state2, &model, &req, sid.as_deref()))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_infer_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || run_infer(&state2, &model, &req, sid.as_deref()))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
