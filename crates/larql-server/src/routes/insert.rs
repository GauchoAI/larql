//! POST /v1/insert — constellation knowledge insertion.
//!
//! Full trace-guided multi-layer insert: forward pass to capture residuals,
//! use as gate vectors, write down vector overrides with target embedding.
//! Supports session isolation via X-Session-Id header.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::HeaderMap;
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::{AppState, LoadedModel};


#[derive(Deserialize)]
pub struct InsertRequest {
    pub entity: String,
    pub relation: String,
    pub target: String,
    #[serde(default)]
    pub layer: Option<usize>,
    #[serde(default = "default_alpha")]
    pub alpha: f32,
    #[serde(default = "default_confidence")]
    pub confidence: f32,
    /// `"constellation"` (default, legacy Architecture-A weight overlay via
    /// walk-FFN residual capture, ~75 s) or `"knn"` (Architecture-B KNN
    /// store via Metal f32 residual capture, ~400 ms).
    #[serde(default = "default_insert_mode")]
    pub mode: String,
    /// Optional raw prompt. When set, overrides the default template
    /// `"The {relation} of {entity} is"`. Use this to capture residuals
    /// from chat-formatted prompts so the KNN key matches the chat path.
    #[serde(default)]
    pub prompt: Option<String>,
    /// Value injection layer (chuk-lazurus two-layer architecture).
    /// When set, captures a SECOND residual at this layer and stores it
    /// as the value_vector for residual injection. The `layer` parameter
    /// becomes the query layer (for matching), and `value_layer` is where
    /// the value gets injected during inference.
    #[serde(default)]
    pub value_layer: Option<usize>,
}

fn default_alpha() -> f32 { 0.25 }
fn default_confidence() -> f32 { 0.9 }
fn default_insert_mode() -> String { "knn".into() }

/// Extract session ID from headers.
fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
}

/// Compute insert layers and residuals from a forward pass.
/// Needs only read access to the patched vindex.
fn compute_residuals(
    model: &LoadedModel,
    patched: &larql_vindex::PatchedVindex,
    req: &InsertRequest,
    insert_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    if !model.config.has_model_weights {
        return Vec::new();
    }

    let weights = match model.get_or_load_weights() {
        Ok(w) => w,
        Err(_) => return Vec::new(),
    };

    let prompt = format!("The {} of {} is",
        req.relation.replace(['-', '_'], " "), req.entity);
    let encoding = match model.tokenizer.encode(prompt.as_str(), true) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();

    let walk_ffn = larql_inference::vindex::WalkFfn::new_with_trace(weights, patched, 8092);
    let _result = larql_inference::predict_with_ffn(
        weights, &model.tokenizer, &token_ids, 1, &walk_ffn,
    );

    walk_ffn.take_residuals().into_iter()
        .filter(|(layer, _)| insert_layers.contains(layer))
        .collect()
}

/// Insert features into a PatchedVindex. Mutates patched in place.
fn apply_insert(
    model: &LoadedModel,
    patched: &mut larql_vindex::PatchedVindex,
    req: &InsertRequest,
    insert_layers: &[usize],
    residuals: &[(usize, Vec<f32>)],
) -> (usize, bool) {
    let hidden = model.embeddings.shape()[1];
    let mut inserted = 0usize;

    // Target embedding for down vector
    let target_encoding = match model.tokenizer.encode(req.target.as_str(), false) {
        Ok(e) => e,
        Err(_) => return (0, false),
    };
    let target_ids: Vec<u32> = target_encoding.get_ids().to_vec();
    let target_id = target_ids.first().copied().unwrap_or(0);

    let mut target_embed = vec![0.0f32; hidden];
    for &tok in &target_ids {
        let row = model.embeddings.row(tok as usize);
        for j in 0..hidden { target_embed[j] += row[j] * model.embed_scale; }
    }
    let n = target_ids.len().max(1) as f32;
    for v in &mut target_embed { *v /= n; }

    let use_constellation = !residuals.is_empty();

    for &layer in insert_layers {
        let feature = match patched.find_free_feature(layer) {
            Some(f) => f,
            None => continue,
        };

        // Gate vector: residual (constellation) or entity embedding (fallback)
        let gate_vec: Vec<f32> = if let Some((_, ref residual)) = residuals.iter().find(|(l, _)| *l == layer) {
            let mut gv = residual.clone();
            if let Some(gate_matrix) = patched.base().gate_vectors_at(layer) {
                let sample = gate_matrix.nrows().min(100);
                if sample > 0 {
                    let avg_norm: f32 = (0..sample)
                        .map(|i| gate_matrix.row(i).dot(&gate_matrix.row(i)).sqrt())
                        .sum::<f32>() / sample as f32;
                    let res_norm: f32 = gv.iter().map(|v| v * v).sum::<f32>().sqrt();
                    if res_norm > 1e-8 && avg_norm > 0.0 {
                        let scale = avg_norm / res_norm;
                        for v in &mut gv { *v *= scale; }
                    }
                }
            }
            gv
        } else {
            let enc = match model.tokenizer.encode(req.entity.as_str(), false) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let ids = enc.get_ids();
            let mut ev = vec![0.0f32; hidden];
            for &tok in ids {
                let row = model.embeddings.row(tok as usize);
                for j in 0..hidden { ev[j] += row[j] * model.embed_scale; }
            }
            let n = ids.len().max(1) as f32;
            for v in &mut ev { *v /= n; }
            let norm: f32 = ev.iter().map(|v| v * v).sum::<f32>().sqrt();
            if norm > 1e-8 { for v in &mut ev { *v /= norm; } }
            ev
        };

        let down_vec: Vec<f32> = target_embed.iter().map(|v| v * req.alpha).collect();

        let meta = larql_vindex::FeatureMeta {
            top_token: req.target.clone(),
            top_token_id: target_id,
            c_score: req.confidence,
            top_k: vec![larql_models::TopKEntry {
                token: req.target.clone(),
                token_id: target_id,
                logit: req.confidence,
            }],
        };

        patched.insert_feature(layer, feature, gate_vec, meta);
        patched.set_down_vector(layer, feature, down_vec);
        inserted += 1;
    }

    (inserted, use_constellation)
}

/// Architecture-B KNN-overlay insert via Metal f32 residual capture.
///
/// Runs `capture_residual_post_attn_norm` on the canonical prompt
/// `"The {relation words} of {entity} is"`, adds a single entry to
/// `patched.knn_store` at the install layer, returns timing + totals.
fn run_insert_knn(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let has_gguf = model.gguf.is_some();
    if !has_gguf && !model.config.has_model_weights {
        return Err(ServerError::InferenceUnavailable(
            "KNN insert requires model weights or weights.gguf".into(),
        ));
    }
    // Vindex weights only loaded if we'll need them (no GGUF source available).
    let weights_opt = if has_gguf {
        None
    } else {
        Some(model.get_or_load_weights().map_err(ServerError::InferenceUnavailable)?)
    };

    // Default install layer matches LQL and bench_interactive:
    //   num_layers - 8 (L26 for Gemma 3 4B).
    let install_layer: usize = req.layer.unwrap_or_else(||
        model.config.num_layers.saturating_sub(8));

    // Prompt: use custom prompt if provided (e.g. chat-formatted),
    // otherwise the canonical "The {relation} of {entity} is" template.
    let prompt = match &req.prompt {
        Some(p) => p.clone(),
        None => {
            let rel_words = req.relation.replace(['-', '_'], " ");
            format!("The {rel_words} of {} is", req.entity)
        }
    };
    let prompt_enc = model.tokenizer.encode(prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?;
    let token_ids: Vec<u32> = prompt_enc.get_ids().to_vec();

    // Target token id — tokenize " {target}" and take first id.
    let spaced_target = format!(" {}", req.target);
    let tgt_enc = model.tokenizer.encode(spaced_target.as_str(), false)
        .map_err(|e| ServerError::Internal(format!("tokenize target: {e}")))?;
    let target_id: u32 = tgt_enc.get_ids().first().copied().unwrap_or(0);

    // Capture residual(s). Use GGUF pipeline when available (matches the
    // GGUF infer path exactly); fall back to vindex Q4_K GPU capture.
    let (query_key, value_vec) = {
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };

        if let Some(gguf) = model.gguf.as_ref() {
            backend.reset_kv_cache();
            let qk = gguf.capture_residual_at_layer(&token_ids, install_layer, &**backend);
            let vv = if let Some(vl) = req.value_layer {
                backend.reset_kv_cache();
                gguf.capture_residual_at_layer(&token_ids, vl, &**backend)
            } else { None };
            backend.reset_kv_cache();
            (qk, vv)
        } else {
            let weights = weights_opt.unwrap();
            let patched = model.patched.blocking_read();
            backend.reset_kv_cache();
            let qk = larql_inference::capture_knn_key_gpu(
                weights, &token_ids, install_layer, patched.base(), &**backend,
            );
            let vv = if let Some(vl) = req.value_layer {
                backend.reset_kv_cache();
                larql_inference::capture_knn_key_gpu(
                    weights, &token_ids, vl, patched.base(), &**backend,
                )
            } else { None };
            backend.reset_kv_cache();
            (qk, vv)
        }
    };
    let query_key = query_key.ok_or_else(|| ServerError::Internal(
        "query residual capture failed".into(),
    ))?;

    // Store: value injection mode if value_layer set, else token override
    if let (Some(vv), Some(vl)) = (value_vec, req.value_layer) {
        if let Some(sid) = session_id {
            let mut sessions = state.sessions.sessions_blocking_write();
            let now = std::time::Instant::now();
            let session = sessions.entry(sid.to_string()).or_insert_with(|| {
                let base = model.patched.blocking_read();
                crate::session::SessionState::new(base.base().clone(), now)
            });
            session.touch(now);
            session.patched.knn_store.add_value_injection(
                install_layer, query_key, vv, vl,
                req.target.clone(), req.entity.clone(), req.relation.clone(),
                req.confidence,
            );
        } else {
            let mut patched = model.patched.blocking_write();
            patched.knn_store.add_value_injection(
                install_layer, query_key, vv, vl,
                req.target.clone(), req.entity.clone(), req.relation.clone(),
                req.confidence,
            );
        }
        let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
        return Ok(serde_json::json!({
            "entity": req.entity,
            "relation": req.relation,
            "target": req.target,
            "query_layer": install_layer,
            "value_layer": vl,
            "mode": "knn-inject",
            "session": session_id,
            "latency_ms": (latency_ms * 10.0).round() / 10.0,
        }));
    }

    // Token override mode (existing behavior)
    return Ok(store_knn_key(
        state, model, req, session_id, install_layer,
        query_key, target_id, start,
    ));
}

/// Store a KNN key in the session or global KNN store.
fn store_knn_key(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
    install_layer: usize,
    key: Vec<f32>,
    target_id: u32,
    start: std::time::Instant,
) -> serde_json::Value {
    // Store. Session-scoped if X-Session-Id present; else global.
    if let Some(sid) = session_id {
        let mut sessions = state.sessions.sessions_blocking_write();
        let now = std::time::Instant::now();
        let session = sessions.entry(sid.to_string()).or_insert_with(|| {
            let base = model.patched.blocking_read();
            crate::session::SessionState::new(base.base().clone(), now)
        });
        session.touch(now);
        session.patched.knn_store.add(
            install_layer, key, target_id,
            req.target.clone(), req.entity.clone(), req.relation.clone(),
            req.confidence,
        );
    } else {
        let mut patched = model.patched.blocking_write();
        patched.knn_store.add(
            install_layer, key, target_id,
            req.target.clone(), req.entity.clone(), req.relation.clone(),
            req.confidence,
        );
    }

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;
    serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "layer": install_layer,
        "mode": "knn",
        "session": session_id,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    })
}

fn run_insert(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let start = std::time::Instant::now();

    // KNN-overlay insert (Architecture B). Capture residual via Metal f32
    // and add to PatchedVindex.knn_store. ~400 ms vs ~75 s for
    // Architecture A walk-FFN capture below.
    if req.mode == "knn" {
        return run_insert_knn(state, model, req, session_id, start);
    }

    // Determine insert layers
    let last = model.config.num_layers.saturating_sub(1);
    let bands = model.config.layer_bands.clone()
        .or_else(|| larql_vindex::LayerBands::for_family(&model.config.family, model.config.num_layers))
        .unwrap_or(larql_vindex::LayerBands {
            syntax: (0, last),
            knowledge: (0, last),
            output: (0, last),
        });

    let insert_layers: Vec<usize> = if let Some(l) = req.layer {
        vec![l]
    } else {
        let mid = (bands.knowledge.0 + bands.knowledge.1) / 2;
        (mid..=bands.knowledge.1).collect()
    };

    let (inserted, use_constellation) = if let Some(sid) = session_id {
        // Session-scoped: read from session for residuals, write to session for insert
        let mut sessions = state.sessions.sessions_blocking_write();
        let now = std::time::Instant::now();

        let session = sessions
            .entry(sid.to_string())
            .or_insert_with(|| {
                let base = model.patched.blocking_read();
                crate::session::SessionState::new(base.base().clone(), now)
            });
        session.touch(now);

        let residuals = compute_residuals(model, &session.patched, req, &insert_layers);
        apply_insert(model, &mut session.patched, req, &insert_layers, &residuals)
    } else {
        // Global: read from global for residuals, write to global for insert
        let residuals = {
            let patched = model.patched.blocking_read();
            compute_residuals(model, &patched, req, &insert_layers)
        };

        let mut patched = model.patched.blocking_write();
        apply_insert(model, &mut patched, req, &insert_layers, &residuals)
    };

    let latency_ms = start.elapsed().as_secs_f64() * 1000.0;

    Ok(serde_json::json!({
        "entity": req.entity,
        "relation": req.relation,
        "target": req.target,
        "inserted": inserted,
        "mode": if use_constellation { "constellation" } else { "embedding" },
        "alpha": req.alpha,
        "session": session_id,
        "latency_ms": (latency_ms * 10.0).round() / 10.0,
    }))
}

pub async fn handle_insert(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_insert(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}

pub async fn handle_insert_multi(
    State(state): State<Arc<AppState>>,
    Path(model_id): Path<String>,
    headers: HeaderMap,
    Json(req): Json<InsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(Some(&model_id))
        .ok_or_else(|| ServerError::NotFound(format!("model '{}' not found", model_id)))?;
    let model = Arc::clone(model);
    let sid = session_id(&headers);
    let state2 = Arc::clone(&state);
    let result = tokio::task::spawn_blocking(move || {
        run_insert(&state2, &model, &req, sid.as_deref())
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;
    Ok(Json(result))
}
