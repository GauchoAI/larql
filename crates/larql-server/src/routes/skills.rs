//! Skill-registry routes.
//!
//!   GET    /v1/skills            — list every registered skill
//!   POST   /v1/skills/register   — upsert a skill (wizard / make_skill use this)
//!   POST   /v1/skills/{name}/used  — mark a skill as just-used (success/failure)
//!   DELETE /v1/skills/{name}     — drop a skill from the registry
//!
//! The filesystem (`~/.larql/skills/<name>/`) remains the source of
//! truth for executable content.  This DB is the catalog on top.

use std::sync::Arc;

use axum::Json;
use axum::extract::{Path, State};
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Deserialize;

use crate::state::AppState;

#[derive(Deserialize)]
pub struct RegisterRequest {
    pub name: String,
    #[serde(default)]
    pub description: String,
    #[serde(default)]
    pub keywords: String,
    #[serde(default = "default_runtime")]
    pub runtime: String,
    #[serde(default = "default_source")]
    pub source: String,
    #[serde(default = "default_scope")]
    pub scope: String,
}

fn default_runtime() -> String { "host".into() }
fn default_source()  -> String { "manual".into() }
fn default_scope()   -> String { "global".into() }

#[derive(Deserialize)]
pub struct UsedRequest {
    #[serde(default = "default_success")]
    pub success: bool,
}
fn default_success() -> bool { true }

pub async fn handle_list(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    state.bump_requests();
    match state.skill_registry.list() {
        Ok(entries) => (StatusCode::OK, Json(serde_json::json!({"skills": entries}))).into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

pub async fn handle_register(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RegisterRequest>,
) -> impl IntoResponse {
    state.bump_requests();
    match state.skill_registry.upsert(
        &req.name,
        &req.description,
        &req.keywords,
        &req.runtime,
        &req.source,
        &req.scope,
    ) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({"name": req.name, "registered": true})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

pub async fn handle_used(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
    Json(req): Json<UsedRequest>,
) -> impl IntoResponse {
    state.bump_requests();
    match state.skill_registry.mark_used(&name, req.success) {
        Ok(()) => (
            StatusCode::OK,
            Json(serde_json::json!({"name": name, "success": req.success})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}

pub async fn handle_delete(
    State(state): State<Arc<AppState>>,
    Path(name): Path<String>,
) -> impl IntoResponse {
    state.bump_requests();
    match state.skill_registry.delete(&name) {
        Ok(true) => (StatusCode::OK, Json(serde_json::json!({"name": name, "deleted": true}))).into_response(),
        Ok(false) => (
            StatusCode::NOT_FOUND,
            Json(serde_json::json!({"name": name, "deleted": false})),
        )
            .into_response(),
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({"error": e})),
        )
            .into_response(),
    }
}
