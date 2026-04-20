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
    /// Layer to install the KNN entry at. Default: num_layers - 8 (L26 on
    /// Gemma 3 4B), the canonical knowledge layer.
    #[serde(default)]
    pub layer: Option<usize>,
    /// Confidence stored on the KNN entry; surfaced via DESCRIBE.
    #[serde(default = "default_confidence")]
    pub confidence: f32,
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

fn default_confidence() -> f32 { 0.9 }

/// Extract session ID from headers.
fn session_id(headers: &HeaderMap) -> Option<String> {
    headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string())
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
    let gguf = model.gguf.as_ref().ok_or_else(|| ServerError::InferenceUnavailable(
        "KNN insert requires weights.gguf in vindex dir".into(),
    ))?;

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

    // Capture residual(s) via the GGUF pipeline — matches the infer path.
    let (query_key, value_vec) = {
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(poisoned) => poisoned.into_inner(),
        };
        backend.reset_kv_cache();
        let qk = gguf.capture_residual_at_layer(&token_ids, install_layer, &**backend);
        let vv = if let Some(vl) = req.value_layer {
            backend.reset_kv_cache();
            gguf.capture_residual_at_layer(&token_ids, vl, &**backend)
        } else { None };
        backend.reset_kv_cache();
        (qk, vv)
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
    // Only KNN insert is supported. The legacy "constellation" walk-FFN
    // capture (~75 s, vindex weights) was deleted along with the rest of
    // the vindex inference path. Force-route everything through KNN.
    run_insert_knn(state, model, req, session_id, start)
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
