//! Session log routes — list, fetch, delete.
//!
//! Backed by `chat_log`'s on-disk JSONL files.  Lets clients (TUI, REPL,
//! curl) discover and resume past conversations.

use axum::extract::Path;
use axum::Json;
use axum::http::StatusCode;
use axum::response::IntoResponse;
use serde::Deserialize;

use crate::chat_log;

/// GET /v1/sessions — list all sessions, sorted newest first.
pub async fn handle_list_sessions() -> Json<serde_json::Value> {
    let sessions: Vec<serde_json::Value> = chat_log::list_sessions()
        .into_iter()
        .map(|s| {
            serde_json::json!({
                "id": s.id,
                "turns": s.turns,
                "last_modified": s.last_modified,
                "preview": s.preview,
            })
        })
        .collect();
    Json(serde_json::json!({"sessions": sessions}))
}

/// GET /v1/sessions/:id — full conversation history.
pub async fn handle_get_session(Path(id): Path<String>) -> Json<serde_json::Value> {
    let turns: Vec<serde_json::Value> = chat_log::load_session(&id)
        .into_iter()
        .map(|t| {
            serde_json::json!({
                "ts": t.ts,
                "role": t.role,
                "content": t.content,
            })
        })
        .collect();
    Json(serde_json::json!({
        "id": id,
        "turns": turns,
    }))
}

/// DELETE /v1/sessions/:id — wipe the file.
pub async fn handle_delete_session(Path(id): Path<String>) -> impl IntoResponse {
    let existed = chat_log::delete_session(&id);
    let body = Json(serde_json::json!({"id": id, "deleted": existed}));
    if existed {
        (StatusCode::OK, body)
    } else {
        (StatusCode::NOT_FOUND, body)
    }
}

#[derive(Deserialize)]
pub struct AppendTurnRequest {
    pub role: String,
    pub content: String,
    /// Optional structured metadata (e.g. `{"tool_ms": 82310}`).
    #[serde(default)]
    pub meta: Option<serde_json::Value>,
}

/// POST /v1/sessions/:id/log — append an arbitrary turn (used by the
/// TUI to persist `tool_use` / `tool_result` so they re-render on
/// resume).  `role` may be anything; the client decides the schema.
pub async fn handle_append_turn(
    Path(id): Path<String>,
    Json(req): Json<AppendTurnRequest>,
) -> Json<serde_json::Value> {
    if let Some(meta) = req.meta {
        chat_log::append_turn_with_meta(&id, &req.role, &req.content, meta);
    } else {
        chat_log::append_turn(&id, &req.role, &req.content);
    }
    Json(serde_json::json!({"id": id, "appended": true, "role": req.role}))
}
