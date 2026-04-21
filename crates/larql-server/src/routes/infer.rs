//! POST /v1/infer — single-token forward pass via the llama.cpp pipeline
//! plus an optional KNN overlay short-circuit.  The only mode is "fast";
//! the only weight source is `weights.gguf`, loaded through
//! `larql-llamacpp`.  Falls back to the legacy GGUF pipeline if the
//! llama.cpp load failed at startup.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::Deserialize;

use crate::error::ServerError;
use crate::llama_probe::{snapshot_layer, Mode};
use crate::state::{AppState, LoadedModel};

#[derive(Deserialize)]
pub struct InferRequest {
    pub prompt: String,
    #[serde(default = "default_top")]
    pub top: usize,
}

fn default_top() -> usize { 5 }

fn session_id(headers: &HeaderMap) -> Option<String> {
    headers.get("x-session-id").and_then(|v| v.to_str().ok()).map(|s| s.to_string())
}

/// Default probe layer: `num_layers - 8` (layer 26 for Gemma 3 4B),
/// matching the existing /v1/insert default.
fn probe_layer_of(model: &LoadedModel) -> u32 {
    model.config.num_layers.saturating_sub(8) as u32
}

/// Tensor we read for the KNN query.  `attn_post_norm` is the post-
/// attention residual pre-FFN — semantically equivalent to the old
/// `capture_residual_post_attn_norm` path.
fn probe_tensor_of(layer: u32) -> String {
    format!("attn_post_norm-{layer}")
}

fn run_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    if model.llama.is_some() {
        run_infer_llama(state, model, req, session_id)
    } else {
        Err(ServerError::InferenceUnavailable(
            "no weights.gguf in vindex dir — drop one in to enable inference".into(),
        ))
    }
}

/// New hot path: runs through `larql-llamacpp`, KNN query via the
/// server probe (one-shot, configured per-request).
fn run_infer_llama(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let pipe_mu = model.llama.as_ref().expect("llama present");

    let start = std::time::Instant::now();

    // --- Snapshot the right KnnStore for this request ------------------
    let probe_layer = probe_layer_of(model);
    let tensor = probe_tensor_of(probe_layer);
    const THRESHOLD: f32 = 0.75;

    let entries = if let Some(sid) = session_id {
        let sessions = state.sessions.sessions_blocking_write();
        match sessions.get(sid) {
            Some(session) => snapshot_layer(&session.patched.knn_store, probe_layer as usize),
            None => {
                let patched = model.patched.blocking_read();
                snapshot_layer(&patched.knn_store, probe_layer as usize)
            }
        }
    } else {
        let patched = model.patched.blocking_read();
        snapshot_layer(&patched.knn_store, probe_layer as usize)
    };

    // --- Configure probe for this request ------------------------------
    let entries_len = entries.len();
    {
        let mut s = model.probe_state.lock().unwrap();
        s.mode = Mode::KnnQuery {
            layer: probe_layer,
            tensor_name: tensor,
            threshold: THRESHOLD,
            entries,
            fired: false,
            forced: None,
            result: None,
            best_cosine: None,
        };
    }

    // --- Decode + top-k -----------------------------------------------
    let fast_start = std::time::Instant::now();
    let top_k = {
        let mut pipe = pipe_mu.lock().map_err(|e| {
            ServerError::Internal(format!("llama pipeline lock poisoned: {e}"))
        })?;
        pipe.reset_kv();
        pipe.prefill_and_top_k(&req.prompt, req.top)
            .map_err(|e| ServerError::Internal(format!("llama infer: {e}")))?
    };
    let fast_ms = fast_start.elapsed().as_secs_f64() * 1000.0;

    // --- Read probe result + reset mode --------------------------------
    let (knn_match, best_cos) = {
        let mut s = model.probe_state.lock().unwrap();
        let (m, c) = if let Mode::KnnQuery { result, best_cosine, .. } = &mut s.mode {
            (result.take(), best_cosine.take())
        } else {
            (None, None)
        };
        s.mode = Mode::Idle;
        (m, c)
    };

    // --- Build response ------------------------------------------------
    let pipe = pipe_mu.lock().unwrap();
    let predictions: Vec<serde_json::Value> = top_k
        .iter()
        .map(|(tid, p)| {
            let token = pipe.decode_token(*tid);
            serde_json::json!({
                "token": token,
                "probability": (p * 10000.0).round() / 10000.0,
            })
        })
        .collect();
    drop(pipe);

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));
    result.insert("predictions".into(), serde_json::json!(predictions));
    result.insert(
        "knn_override".into(),
        serde_json::json!(knn_match.as_ref().map(|m| &m.target_token)),
    );
    if let Some(m) = knn_match.as_ref() {
        result.insert(
            "knn_cosine".into(),
            serde_json::json!((m.cosine * 10000.0).round() / 10000.0),
        );
        result.insert("knn_entity".into(), serde_json::json!(m.entity));
        result.insert("knn_relation".into(), serde_json::json!(m.relation));
    } else if let Some(c) = best_cos {
        // Diagnostic: surface the best cosine even when we didn't override.
        result.insert(
            "knn_best_cosine".into(),
            serde_json::json!((c * 10000.0).round() / 10000.0),
        );
    }
    result.insert("knn_entries_scanned".into(), serde_json::json!(entries_len));
    result.insert(
        "fast_ms".into(),
        serde_json::json!((fast_ms * 10.0).round() / 10.0),
    );
    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    result.insert(
        "latency_ms".into(),
        serde_json::json!((latency_ms * 10.0).round() / 10.0),
    );
    result.insert("pipeline".into(), serde_json::json!("llama.cpp"));
    Ok(serde_json::Value::Object(result))
}

pub async fn handle_infer(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InferRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state.model(None)
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
    let model = state.model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || run_infer(&state2, &model, &req, sid.as_deref()))
        .await
        .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
