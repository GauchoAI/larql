//! Router setup — maps URL paths to handlers.
//!
//! The hot path is GGUF: drop a `weights.gguf` into the vindex dir
//! and inference, insertion, and streaming all run through the same
//! GgufPipeline (Metal q8_0_gguf decode + Q8_0Gguf lm_head matvec +
//! KNN overlay at the configured probe layer).

pub mod describe;
pub mod generate_stream;
pub mod health;
pub mod infer;
pub mod insert;
pub mod models;
pub mod patches;
pub mod probe;
pub mod relations;
pub mod select;
pub mod stats;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post, delete};

use crate::state::AppState;

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        // Inference hot path (all GGUF)
        .route("/v1/infer", post(infer::handle_infer))
        .route("/v1/insert", post(insert::handle_insert))
        .route("/v1/generate", post(generate_stream::handle_generate_stream))
        // Vindex graph browsing (no inference)
        .route("/v1/describe", get(describe::handle_describe))
        .route("/v1/select", post(select::handle_select))
        .route("/v1/relations", get(relations::handle_relations))
        .route("/v1/probe", post(probe::handle_probe))
        // Patches
        .route("/v1/patches/apply", post(patches::handle_apply_patch))
        .route("/v1/patches", get(patches::handle_list_patches))
        .route("/v1/patches/{name}", delete(patches::handle_remove_patch))
        // Meta
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .route("/v1/stats", get(stats::handle_stats))
        .with_state(state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .route("/v1/{model_id}/describe", get(describe::handle_describe_multi))
        .route("/v1/{model_id}/select", post(select::handle_select_multi))
        .route("/v1/{model_id}/relations", get(relations::handle_relations_multi))
        .route("/v1/{model_id}/stats", get(stats::handle_stats_multi))
        .route("/v1/{model_id}/infer", post(infer::handle_infer_multi))
        .route("/v1/{model_id}/patches/apply", post(patches::handle_apply_patch_multi))
        .route("/v1/{model_id}/patches", get(patches::handle_list_patches_multi))
        .route("/v1/{model_id}/patches/{name}", delete(patches::handle_remove_patch_multi))
        .route("/v1/{model_id}/insert", post(insert::handle_insert_multi))
        .with_state(state)
}
