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
use crate::llama_probe::Mode;
use crate::state::{AppState, LoadedModel};

const KNN_PROBE_TENSOR_PREFIX: &str = "attn_post_norm-";

fn probe_tensor_for(layer: usize) -> String {
    format!("{KNN_PROBE_TENSOR_PREFIX}{layer}")
}


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
    if model.llama.is_none() {
        return Err(ServerError::InferenceUnavailable(
            "no weights.gguf in vindex dir — drop one in to enable inference".into(),
        ));
    }
    run_insert_llama(state, model, req, session_id, start)
}

/// Capture residual via llama.cpp's `cb_eval` probe, store as a KNN
/// entry.  Keys captured here are in the same numerical space as the
/// /v1/infer probe reads, so cosine hits 1.0 at query time.
fn run_insert_llama(
    state: &AppState,
    model: &LoadedModel,
    req: &InsertRequest,
    session_id: Option<&str>,
    start: std::time::Instant,
) -> Result<serde_json::Value, ServerError> {
    let pipe_mu = model.llama.as_ref().expect("llama present");

    let install_layer: usize = req
        .layer
        .unwrap_or_else(|| model.config.num_layers.saturating_sub(8));

    let prompt = match &req.prompt {
        Some(p) => p.clone(),
        None => {
            let rel_words = req.relation.replace(['-', '_'], " ");
            format!("The {rel_words} of {} is", req.entity)
        }
    };

    // Target token id via llama.cpp's tokenizer (same vocab as probe).
    let spaced_target = format!(" {}", req.target);
    let target_id: u32 = {
        let pipe = pipe_mu.lock().map_err(|e| {
            ServerError::Internal(format!("llama pipeline lock poisoned: {e}"))
        })?;
        pipe.token_id_of(&spaced_target).ok_or_else(|| {
            ServerError::Internal("failed to tokenize target".into())
        })? as u32
    };

    // Configure probe to capture, then prefill.
    {
        let mut s = model.probe_state.lock().unwrap();
        s.mode = Mode::Capture {
            layer: install_layer as u32,
            tensor_name: probe_tensor_for(install_layer),
            captured: None,
        };
    }

    let query_key = {
        let mut pipe = pipe_mu.lock().map_err(|e| {
            ServerError::Internal(format!("llama pipeline lock poisoned: {e}"))
        })?;
        pipe.reset_kv();
        pipe.prefill_and_top_k(&prompt, 1)
            .map_err(|e| ServerError::Internal(format!("llama prefill: {e}")))?;

        let mut s = model.probe_state.lock().unwrap();
        let capt = if let Mode::Capture { captured, .. } = &mut s.mode {
            captured.take()
        } else {
            None
        };
        s.mode = Mode::Idle;
        capt
    };
    let query_key = query_key.ok_or_else(|| {
        ServerError::Internal("residual not captured — tensor name mismatch?".into())
    })?;

    // Note: value-injection mode is not yet ported to the llama path.
    // Token-override insert only, matching the default old flow.
    if req.value_layer.is_some() {
        return Err(ServerError::BadRequest(
            "value_layer not yet supported on llama.cpp pipeline".into(),
        ));
    }

    Ok(store_knn_key(
        state, model, req, session_id, install_layer,
        query_key, target_id, start,
    ))
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
