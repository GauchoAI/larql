//! POST /v1/probe — capture the residual at a layer via llama.cpp's
//! `cb_eval`, return as JSON.  Used for offline cosine experiments and
//! KNN-store debugging without storing anything.

use std::sync::Arc;
use axum::Json;
use axum::extract::State;
use serde::Deserialize;

use crate::error::ServerError;
use crate::llama_probe::Mode;
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
        let pipe_mu = model.llama.as_ref().ok_or_else(|| {
            ServerError::InferenceUnavailable(
                "probe requires weights.gguf in vindex dir (llama path)".into(),
            )
        })?;

        {
            let mut s = model.probe_state.lock().unwrap();
            s.mode = Mode::Capture {
                layer: req.layer as u32,
                tensor_name: format!("attn_post_norm-{}", req.layer),
                captured: None,
            };
        }

        let vec = {
            let mut pipe = pipe_mu.lock().map_err(|e| {
                ServerError::Internal(format!("llama pipeline lock poisoned: {e}"))
            })?;
            pipe.reset_kv();
            let _ = pipe.prefill_and_top_k(&req.prompt, 1).map_err(|e| {
                ServerError::Internal(format!("llama prefill: {e}"))
            })?;

            let mut s = model.probe_state.lock().unwrap();
            let captured = if let Mode::Capture { captured, .. } = &mut s.mode {
                captured.take()
            } else {
                None
            };
            s.mode = Mode::Idle;
            captured
        };

        let vec = vec.ok_or_else(|| {
            ServerError::Internal("probe failed: residual not captured".into())
        })?;
        let len = vec.len();

        Ok(serde_json::json!({
            "layer": req.layer,
            "dim": len,
            "vector": vec,
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}
