//! Router setup — maps URL paths to handlers.

pub mod chat_completions;
pub mod describe;
pub mod explain;
pub mod generate_stream;
pub mod health;
pub mod infer;
pub mod insert;
pub mod models;
pub mod patches;
pub mod relations;
pub mod select;
pub mod stats;
pub mod stream;
pub mod kv_cache;
pub mod kv_rag;
pub mod probe;
pub mod rag;
pub mod vec_inject;
pub mod walk;
pub mod walk_ffn;

use std::sync::Arc;

use axum::Router;
use axum::routing::{get, post, delete};

use crate::state::AppState;

/// Build the router for single-model serving.
pub fn single_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/describe", get(describe::handle_describe))
        .route("/v1/walk", get(walk::handle_walk))
        .route("/v1/select", post(select::handle_select))
        .route("/v1/relations", get(relations::handle_relations))
        .route("/v1/stats", get(stats::handle_stats))
        .route("/v1/infer", post(infer::handle_infer))
        .route("/v1/patches/apply", post(patches::handle_apply_patch))
        .route("/v1/patches", get(patches::handle_list_patches))
        .route("/v1/patches/{name}", delete(patches::handle_remove_patch))
        .route("/v1/walk-ffn", post(walk_ffn::handle_walk_ffn))
        .route("/v1/explain-infer", post(explain::handle_explain))
        .route("/v1/insert", post(insert::handle_insert))
        .route("/v1/stream", get(stream::handle_stream))
        .route("/v1/generate", post(generate_stream::handle_generate_stream))
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .route("/v1/chat/completions", post(chat_completions::handle_chat_completions))
        .route("/v1/probe", post(probe::handle_probe))
        .route("/v1/rag/insert", post(rag::handle_rag_insert))
        .route("/v1/rag/query", post(rag::handle_rag_query))
        .route("/v1/rag/insert-deep", post(rag::handle_rag_insert_deep))
        .route("/v1/rag/query-deep", post(rag::handle_rag_query_deep))
        .route("/v1/kv-rag/insert", post(kv_rag::handle_kv_rag_insert))
        .route("/v1/kv-rag/query", post(kv_rag::handle_kv_rag_query))
        .route("/v1/vec/insert", post(vec_inject::handle_vec_insert))
        .route("/v1/vec/query", post(vec_inject::handle_vec_query))
        .route("/v1/kv/precompute", post(kv_cache::handle_precompute))
        .route("/v1/kv/stats", get(kv_cache::handle_kv_stats))
        .with_state(state)
}

/// Build the router for multi-model serving.
pub fn multi_model_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(health::handle_health))
        .route("/v1/models", get(models::handle_models))
        .route("/v1/{model_id}/describe", get(describe::handle_describe_multi))
        .route("/v1/{model_id}/walk", get(walk::handle_walk_multi))
        .route("/v1/{model_id}/select", post(select::handle_select_multi))
        .route("/v1/{model_id}/relations", get(relations::handle_relations_multi))
        .route("/v1/{model_id}/stats", get(stats::handle_stats_multi))
        .route("/v1/{model_id}/infer", post(infer::handle_infer_multi))
        .route("/v1/{model_id}/patches/apply", post(patches::handle_apply_patch_multi))
        .route("/v1/{model_id}/patches", get(patches::handle_list_patches_multi))
        .route("/v1/{model_id}/patches/{name}", delete(patches::handle_remove_patch_multi))
        .route("/v1/{model_id}/explain-infer", post(explain::handle_explain_multi))
        .route("/v1/{model_id}/insert", post(insert::handle_insert_multi))
        .with_state(state)
}
