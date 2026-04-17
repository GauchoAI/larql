//! POST /v1/probe — capture L26 residual for a prompt, return as JSON.
//! Used for offline cosine experiments without KNN storage.

use std::sync::Arc;
use axum::Json;
use axum::extract::State;
use serde::Deserialize;
use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct ProbeRequest {
    pub prompt: String,
    #[serde(default = "default_layer")]
    pub layer: usize,
}

fn default_layer() -> usize { 26 }

pub async fn handle_probe(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ProbeRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);

    let result = tokio::task::spawn_blocking(move || {
        let weights = model.get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();

        let encoding = model.tokenizer.encode(req.prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();

        let patched = model.patched.blocking_read();
        let key = larql_inference::capture_knn_key_gpu(
            weights, &token_ids, req.layer, patched.base(), &**backend,
        );
        backend.reset_kv_cache();

        match key {
            Some(vec) => Ok(serde_json::json!({
                "layer": req.layer,
                "prompt_tokens": token_ids.len(),
                "dim": vec.len(),
                "vector": vec,
            })),
            None => Err(ServerError::Internal("probe failed".into())),
        }
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}
