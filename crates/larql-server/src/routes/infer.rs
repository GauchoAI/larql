//! POST /v1/infer — single-token forward pass via the GGUF pipeline
//! plus an optional KNN overlay short-circuit. The only mode is "fast";
//! the only weight source is `weights.gguf`.

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
}

fn default_top() -> usize { 5 }

fn session_id(headers: &HeaderMap) -> Option<String> {
    headers.get("x-session-id").and_then(|v| v.to_str().ok()).map(|s| s.to_string())
}

fn run_infer(
    state: &AppState,
    model: &LoadedModel,
    req: &InferRequest,
    session_id: Option<&str>,
) -> Result<serde_json::Value, ServerError> {
    let gguf = model.gguf.as_ref().ok_or_else(|| ServerError::InferenceUnavailable(
        "no weights.gguf in vindex dir — drop one in to enable inference".into(),
    ))?;

    let token_ids: Vec<u32> = model.tokenizer
        .encode(req.prompt.as_str(), true)
        .map_err(|e| ServerError::Internal(format!("tokenize error: {e}")))?
        .get_ids().to_vec();
    if token_ids.is_empty() {
        return Err(ServerError::BadRequest("empty prompt".into()));
    }

    let start = std::time::Instant::now();
    let backend = model.get_or_init_backend();

    let run = |patched: &larql_vindex::PatchedVindex| {
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();
        let knn_opt = if patched.knn_store.is_empty() { None } else { Some(&patched.knn_store) };
        let fast_start = std::time::Instant::now();
        let pred = gguf.predict_top_k_with_knn(
            &token_ids, req.top, &**backend, &model.tokenizer, knn_opt,
        );
        let fast_ms = fast_start.elapsed().as_secs_f64() * 1000.0;
        (pred, fast_ms)
    };

    let (pred, fast_ms) = if let Some(sid) = session_id {
        let sessions = state.sessions.sessions_blocking_write();
        if let Some(session) = sessions.get(sid) {
            run(&session.patched)
        } else {
            drop(sessions);
            let patched = model.patched.blocking_read();
            run(&patched)
        }
    } else {
        let patched = model.patched.blocking_read();
        run(&patched)
    };

    let predictions: Vec<serde_json::Value> = pred.predictions.iter().map(|(tok, prob)| {
        serde_json::json!({
            "token": tok,
            "probability": (*prob * 10000.0).round() / 10000.0,
        })
    }).collect();

    let mut result = serde_json::Map::new();
    result.insert("prompt".into(), serde_json::json!(req.prompt));
    result.insert("predictions".into(), serde_json::json!(predictions));
    result.insert("knn_override".into(), serde_json::json!(pred.knn_override));
    if let Some(c) = pred.knn_cosine {
        result.insert("knn_cosine".into(), serde_json::json!((c * 10000.0).round() / 10000.0));
    }
    result.insert("fast_ms".into(), serde_json::json!((fast_ms * 10.0).round() / 10.0));
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
