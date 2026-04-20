//! POST /v1/probe — capture the pre-FFN-normalized residual at a layer
//! for a prompt, return as JSON. Used for offline cosine experiments
//! and KNN-store debugging without storing anything.

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
        let gguf = model.gguf.as_ref().ok_or_else(|| ServerError::InferenceUnavailable(
            "probe requires weights.gguf in vindex dir".into(),
        ))?;
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();

        let token_ids: Vec<u32> = model.tokenizer.encode(req.prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?
            .get_ids().to_vec();

        let key = gguf.capture_residual_at_layer(&token_ids, req.layer, &**backend);
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
