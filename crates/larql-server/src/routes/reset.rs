//! POST /v1/reset — wipe all KNN entries from the live store.
//! Purpose: give test harnesses a clean-slate button between runs.
//! Doesn't touch weights, sessions, or cache.

use std::sync::Arc;

use axum::Json;
use axum::extract::State;

use crate::error::ServerError;
use crate::state::AppState;

pub async fn handle_reset(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let mut patched = model.patched.write().await;
    let layers = patched.knn_store.layers();
    for layer in layers {
        if let Some(entries) = patched.knn_store.entries().get(&layer).cloned() {
            for e in entries {
                patched.knn_store.remove_by_entity_relation(&e.entity, &e.relation);
            }
        }
    }
    let remaining = patched.knn_store.len();
    Ok(Json(serde_json::json!({
        "status": "ok",
        "remaining_entries": remaining,
    })))
}
