//! Per-session conversation log on disk.  Append-only JSONL files
//! under `~/.larql/sessions/<id>.jsonl`.
//!
//! The server appends one entry per role-turn after each
//! `/v1/chat/completions` request that carries `X-Session-Id`.  Reload
//! is a flat read of the file, no in-memory state to maintain.

use std::fs::{self, OpenOptions};
use std::io::{BufRead, BufReader, Write};
use std::path::PathBuf;
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Process-wide write lock so concurrent `/v1/chat/completions`,
/// `/v1/sessions/.../log`, and the server's own end-of-stream append
/// don't interleave bytes (we observed `}{` on disk under load).
static WRITE_LOCK: Mutex<()> = Mutex::new(());

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LoggedTurn {
    pub ts: u64,
    pub role: String,
    pub content: String,
    /// Optional, free-form metadata: timings, token counts, tool ids.
    /// Read by view_session to surface profiling info next to each turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub meta: Option<serde_json::Value>,
}

#[derive(Debug, Serialize)]
pub struct SessionSummary {
    pub id: String,
    pub turns: usize,
    pub last_modified: u64,
    pub preview: String,
}

fn root_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    PathBuf::from(home).join(".larql/sessions")
}

fn sanitize(id: &str) -> String {
    id.chars()
        .map(|c| match c {
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect()
}

fn path_for(id: &str) -> PathBuf {
    root_dir().join(format!("{}.jsonl", sanitize(id)))
}

fn now_unix() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0)
}

/// Append one turn to a session's log.  Creates the file (and parent
/// dir) on demand.  Errors are logged at trace level — chat shouldn't
/// fail just because logging hiccupped.
///
/// Two safety nets:
///  1. A process-wide `WRITE_LOCK` serializes appends so concurrent
///     callers don't interleave bytes (e.g. lose newlines).
///  2. We refuse to append a turn that's identical (role + content) to
///     the most recent persisted turn — drops the obvious duplicates
///     produced by tool-follow-up chat requests that resend the same
///     user message.
pub fn append_turn(session_id: &str, role: &str, content: &str) {
    if session_id.is_empty() || content.is_empty() {
        return;
    }
    // Skip the synthetic "Tool result: …" pseudo-user follow-ups the
    // TUI inserts to feed tool output back to the model.  We log the
    // genuine `tool_result` row separately, so persisting this fake
    // user message only pollutes the resume view.
    if role == "user" && content.starts_with("Tool result:") {
        return;
    }
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());

    let dir = root_dir();
    if let Err(e) = fs::create_dir_all(&dir) {
        tracing::warn!("chat_log: mkdir {} failed: {e}", dir.display());
        return;
    }
    let path = path_for(session_id);

    // Cheap dedupe: read the tail and skip if we'd repeat the last
    // role+content verbatim.  `last_turn` opens the file ourselves so
    // we don't double-lock.
    if let Some(prev) = last_turn_unlocked(&path) {
        if prev.role == role && prev.content == content {
            return;
        }
    }

    let entry = LoggedTurn {
        ts: now_unix(),
        role: role.to_string(),
        content: content.to_string(),
        meta: None,
    };
    let line = match serde_json::to_string(&entry) {
        Ok(s) => s,
        Err(e) => {
            tracing::warn!("chat_log: encode failed: {e}");
            return;
        }
    };
    match OpenOptions::new().create(true).append(true).open(&path) {
        Ok(mut f) => {
            if let Err(e) = writeln!(f, "{line}") {
                tracing::warn!("chat_log: write {} failed: {e}", path.display());
            }
        }
        Err(e) => tracing::warn!("chat_log: open {} failed: {e}", path.display()),
    }
}

/// Append a turn with optional structured metadata (timings etc).
/// No dedupe — caller controls what gets stored.
pub fn append_turn_with_meta(
    session_id: &str,
    role: &str,
    content: &str,
    meta: serde_json::Value,
) {
    if session_id.is_empty() {
        return;
    }
    let _guard = WRITE_LOCK.lock().unwrap_or_else(|e| e.into_inner());
    let dir = root_dir();
    if fs::create_dir_all(&dir).is_err() {
        return;
    }
    let path = path_for(session_id);
    let entry = LoggedTurn {
        ts: now_unix(),
        role: role.to_string(),
        content: content.to_string(),
        meta: Some(meta),
    };
    let line = match serde_json::to_string(&entry) {
        Ok(s) => s,
        Err(_) => return,
    };
    if let Ok(mut f) = OpenOptions::new().create(true).append(true).open(&path) {
        let _ = writeln!(f, "{line}");
    }
}

/// Read just the last JSON line of a session file.  Used by the
/// dedupe guard in `append_turn`; not exposed publicly.
fn last_turn_unlocked(path: &std::path::Path) -> Option<LoggedTurn> {
    let f = std::fs::File::open(path).ok()?;
    let mut last: Option<String> = None;
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if !line.trim().is_empty() {
            last = Some(line);
        }
    }
    serde_json::from_str(&last?).ok()
}

/// Load a session's full history.  Returns an empty vec if missing.
pub fn load_session(session_id: &str) -> Vec<LoggedTurn> {
    let path = path_for(session_id);
    let f = match std::fs::File::open(&path) {
        Ok(f) => f,
        Err(_) => return Vec::new(),
    };
    let mut out = Vec::new();
    for line in BufReader::new(f).lines().map_while(Result::ok) {
        if let Ok(t) = serde_json::from_str::<LoggedTurn>(&line) {
            out.push(t);
        }
    }
    out
}

/// List every session in the directory with a one-line preview from
/// the first turn.  Sorted by last_modified descending.
pub fn list_sessions() -> Vec<SessionSummary> {
    let dir = root_dir();
    let entries = match fs::read_dir(&dir) {
        Ok(e) => e,
        Err(_) => return Vec::new(),
    };
    let mut out: Vec<SessionSummary> = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        let stem = match path.file_stem().and_then(|s| s.to_str()) {
            Some(s) => s.to_string(),
            None => continue,
        };
        if path.extension().and_then(|s| s.to_str()) != Some("jsonl") {
            continue;
        }
        let last_modified = entry
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.duration_since(UNIX_EPOCH).ok())
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let turns = load_session(&stem);
        let preview = turns
            .iter()
            .find(|t| t.role == "user")
            .map(|t| {
                let mut p: String = t.content.chars().take(80).collect();
                if t.content.chars().count() > 80 {
                    p.push_str("…");
                }
                p
            })
            .unwrap_or_default();
        out.push(SessionSummary {
            id: stem,
            turns: turns.len(),
            last_modified,
            preview,
        });
    }
    out.sort_by(|a, b| b.last_modified.cmp(&a.last_modified));
    out
}

/// Delete a session log entirely.  Returns true if the file existed.
pub fn delete_session(session_id: &str) -> bool {
    let path = path_for(session_id);
    if path.exists() {
        let _ = fs::remove_file(&path);
        true
    } else {
        false
    }
}
