//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

use std::io;

mod skill_router;
use skill_router::{build_index, load_skills, route, Skill};

use crossterm::event::{
    self, DisableMouseCapture, EnableMouseCapture, Event as CEvent, KeyCode, KeyEventKind,
    KeyModifiers, MouseEventKind,
};
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use crossterm::execute;
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use ratatui::Terminal;

use serde::Serialize;

// ── Types ────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Message {
    User(String),
    Assistant(String),
    System(String),
    ToolUse { tool: String, detail: String },
    /// Text the agent should consume on the next turn (extracted from
    /// ```` ```summary``` ```` blocks emitted by skill tool.sh).
    ToolResult { summary: String },
    /// Render-only payload (charts, etc).  NOT sent back to the model
    /// — it lives only for the human reading the TUI.
    ToolRender { content: String },
}

enum StreamEvent {
    Token(String),
    Done,
    Error(String),
}

struct AppState {
    input: String,
    cursor: usize,
    messages: Vec<Message>,
    status: String,
    is_generating: bool,
    server_url: String,
    session_id: Option<String>,
    /// How many wrapped rows above the bottom we are scrolled.  0 means
    /// pinned to the latest message (auto-scroll on new tokens).  >0
    /// freezes the view.  End / Esc snaps back to 0.
    scroll_offset: u16,
    /// All available skills (loaded once at startup from
    /// `~/.larql/skills/` and `./.skills/`).  The routed primer pulls
    /// from this on every chat send.
    skills: Vec<Skill>,
    /// Diagnostic: the routing decision for the most recent prompt.
    /// Surfaced to view_session via the chat metadata.
    last_route: Option<(String, f32)>,
}

impl AppState {
    fn new(server_url: &str) -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            messages: vec![Message::System(
                "larql — LLM as a Database. Type questions, use tools.".into()
            )],
            status: format!("connecting to {server_url}..."),
            is_generating: false,
            server_url: server_url.to_string(),
            session_id: None,
            scroll_offset: 0,
            skills: Vec::new(),
            last_route: None,
        }
    }

    /// Build the chat messages — full conversation history (user +
    /// assistant turns).  The server applies the model's chat template
    /// to format them as a single prompt.  Skips empty assistant
    /// placeholders (the in-flight reply).
    fn build_chat_messages(&self) -> Vec<ChatMsg> {
        let mut msgs = Vec::new();
        for m in &self.messages {
            match m {
                Message::User(text) if !text.is_empty() => {
                    msgs.push(ChatMsg { role: "user".into(), content: text.clone() });
                }
                Message::Assistant(text) if !text.is_empty() => {
                    msgs.push(ChatMsg { role: "assistant".into(), content: text.clone() });
                }
                _ => {}
            }
        }
        msgs
    }
}

// ── HTTP Backend ─────────────────────────────────────────────────────────

#[derive(Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMsg>,
    stream: bool,
}

#[derive(Serialize, Clone)]
struct ChatMsg {
    role: String,
    content: String,
}

/// Send chat messages to the server, streaming tokens via `tx`.
async fn chat_stream(
    url: &str,
    messages: Vec<ChatMsg>,
    tx: &tokio::sync::mpsc::Sender<StreamEvent>,
    session_id: Option<&str>,
) -> Result<(), String> {
    let client = reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()
        .unwrap_or_else(|_| reqwest::Client::new());

    let req = ChatRequest {
        model: "gemma-3-4b".into(),
        messages,
        stream: true,
    };

    let mut builder = client.post(format!("{url}/v1/chat/completions")).json(&req);
    if let Some(sid) = session_id {
        builder = builder.header("X-Session-Id", sid);
    }
    let resp = builder
        .send()
        .await
        .map_err(|e| format!("connection failed: {e}"))?;

    if !resp.status().is_success() {
        return Err(format!("server error: {}", resp.status()));
    }

    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        buf.push_str(&text);

        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].to_string();
            buf = buf[newline_pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() || line.starts_with(':') { continue; }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    return Ok(());
                }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        let _ = tx.send(StreamEvent::Token(content.to_string())).await;
                    }
                }
            }
        }
    }
    Ok(())
}

fn spawn_chat(
    url: String,
    messages: Vec<ChatMsg>,
    tx: tokio::sync::mpsc::Sender<StreamEvent>,
    session_id: Option<String>,
) {
    tokio::spawn(async move {
        match chat_stream(&url, messages, &tx, session_id.as_deref()).await {
            Ok(()) => { let _ = tx.send(StreamEvent::Done).await; }
            Err(e) => { let _ = tx.send(StreamEvent::Error(e)).await; }
        }
    });
}

/// Fetch a session's full history from the server.  Returns turns in
/// the order they were appended.  Empty vec on miss or error.
async fn fetch_session_history(
    url: &str,
    session_id: &str,
) -> Vec<(String, String)> {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_secs(5))
        .build()
    {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };
    let resp = match client
        .get(format!("{url}/v1/sessions/{session_id}"))
        .send()
        .await
    {
        Ok(r) => r,
        Err(_) => return Vec::new(),
    };
    if !resp.status().is_success() {
        return Vec::new();
    }
    let body: serde_json::Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    body["turns"]
        .as_array()
        .map(|arr| {
            arr.iter()
                .filter_map(|t| {
                    let role = t["role"].as_str()?.to_string();
                    let content = t["content"].as_str()?.to_string();
                    Some((role, content))
                })
                .collect()
        })
        .unwrap_or_default()
}

/// Persist an arbitrary turn (typically `tool_use` or `tool_result`)
/// to the server-side session log so it shows up on resume.
fn append_turn_to_session(
    server_url: &str,
    session_id: Option<&str>,
    role: &str,
    content: &str,
) {
    append_turn_to_session_with_meta(server_url, session_id, role, content, None);
}

/// Variant that attaches a JSON `meta` blob — used to record per-tool
/// timing (`tool_ms`) so view_session can surface "[stats took 82s]".
fn append_turn_to_session_with_meta(
    server_url: &str,
    session_id: Option<&str>,
    role: &str,
    content: &str,
    meta: Option<serde_json::Value>,
) {
    let sid = match session_id {
        Some(s) => s.to_string(),
        None => return,
    };
    let url = format!("{server_url}/v1/sessions/{sid}/log");
    let mut body = serde_json::json!({"role": role, "content": content});
    if let Some(m) = meta {
        body["meta"] = m;
    }
    let payload = serde_json::to_string(&body).unwrap_or_default();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let _ = client
            .post(&url)
            .header("content-type", "application/json")
            .body(payload)
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
    });
}

// ── Skills ────────────────────────────────────────────────────────────────

fn home_dir() -> std::path::PathBuf {
    std::env::var("HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}

/// Start the shared `larql-skill-runtime` Debian container if it
/// isn't running.  Used by skills with `runtime: container` in their
/// frontmatter (the wizard's output) so Gemma's Linux-trained shell
/// works end-to-end on macOS too.  Idempotent + best-effort.
fn ensure_runtime_container() -> std::io::Result<()> {
    let alive = std::process::Command::new("docker")
        .args(["inspect", "-f", "{{.State.Running}}", "larql-skill-runtime"])
        .output()?;
    if alive.status.success()
        && String::from_utf8_lossy(&alive.stdout).trim() == "true"
    {
        return Ok(());
    }
    let _ = std::process::Command::new("docker")
        .args(["rm", "-f", "larql-skill-runtime"])
        .output();
    let host_skills = home_dir().join(".larql/skills");
    let _ = std::fs::create_dir_all(&host_skills);
    let mount = format!("{}:{}", host_skills.display(), host_skills.display());
    let _ = std::process::Command::new("docker")
        .args([
            "run", "-d", "--name", "larql-skill-runtime", "--rm",
            "-v", &mount,
            "debian:bookworm-slim", "bash", "-c", "sleep infinity",
        ])
        .output()?;
    let _ = std::process::Command::new("docker")
        .args([
            "exec", "larql-skill-runtime", "bash", "-c",
            "apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
             coreutils procps net-tools iproute2 dnsutils curl wget jq grep sed gawk findutils \
             git ca-certificates file unzip xz-utils >/dev/null 2>&1",
        ])
        .output()?;
    Ok(())
}

/// Derive a stable session id from the current working directory so a
/// fresh `larql` launch in the same folder resumes the prior chat.
/// e.g. `/Users/miguel/Desktop/llm-as-a-database/larql` →
/// `Users-miguel-Desktop-llm-as-a-database-larql`.
fn default_session_id_from_cwd() -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| std::path::PathBuf::from("/"));
    let raw: String = cwd
        .to_string_lossy()
        .chars()
        .map(|c| match c {
            '/' => '-',
            'a'..='z' | 'A'..='Z' | '0'..='9' | '-' | '_' | '.' => c,
            _ => '_',
        })
        .collect();
    let trimmed = raw.trim_start_matches('-').to_string();
    if trimmed.is_empty() {
        "root".into()
    } else {
        trimmed
    }
}

fn execute_skill_tool(
    text: &str,
    messages: &mut Vec<Message>,
    server_url: &str,
    session_id: Option<&str>,
    skills: &[Skill],
) -> Option<String> {
    let open = "```tool";
    let start = text.find(open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    let tool_call = after[..close].trim();

    let parts: Vec<&str> = tool_call.splitn(2, char::is_whitespace).collect();
    let skill_name = parts.first()?;
    let skill_args = parts.get(1).unwrap_or(&"");

    let skills_dirs = vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        home_dir().join(".larql/skills"),
    ];
    let mut tool_path = None;
    for dir in &skills_dirs {
        let candidate = dir.join(skill_name).join("tool.sh");
        if candidate.exists() { tool_path = Some(candidate); break; }
    }

    let path = match tool_path {
        Some(p) => p,
        None => {
            // The model called a skill we don't have.  Surface a note
            // to the user AND kick off the wizard async so the skill
            // exists by the next conversational turn.  We use the
            // most-recent user message as the wizard's prompt.
            let latest_user: String = messages
                .iter()
                .rev()
                .find_map(|m| if let Message::User(t) = m { Some(t.clone()) } else { None })
                .unwrap_or_default();
            let note = format!(
                "skill `{skill_name}` not installed.  Building it in the background — try again \
                 in ~10-20 seconds and it should be ready."
            );
            messages.push(Message::System(note.clone()));
            append_turn_to_session_with_meta(
                server_url,
                session_id,
                "wizard",
                &format!("starting background build of `{skill_name}`"),
                Some(serde_json::json!({
                    "skill": skill_name,
                    "reason": "not_installed",
                    "from_prompt": latest_user.chars().take(120).collect::<String>(),
                })),
            );
            spawn_background_wizard(skill_name, &latest_user, server_url);
            return None;
        }
    };
    let detail: String = skill_args.chars().take(70).collect();
    messages.push(Message::ToolUse {
        tool: format!("{skill_name}"),
        detail: detail.clone(),
    });
    let tool_use_payload = if detail.is_empty() {
        skill_name.to_string()
    } else {
        format!("{skill_name} {detail}")
    };
    let tool_started = std::time::Instant::now();

    // Decide where to run.  Wizard-built skills declare
    // `runtime: container` in frontmatter and run inside the same
    // Debian sandbox the wizard validated them in (Linux + GNU
    // coreutils, exactly what Gemma was trained on).  Legacy skills
    // default to `host` and run natively.
    let runtime = skills
        .iter()
        .find(|s| s.name == *skill_name)
        .map(|s| s.runtime.clone())
        .unwrap_or_else(|| "host".into());

    use std::io::Write as _;
    let mut command = if runtime == "container" {
        let _ = ensure_runtime_container();
        let mut c = std::process::Command::new("docker");
        c.args(["exec", "-i", "larql-skill-runtime", "bash"]);
        c.arg(&path);
        c.args(skill_args.split_whitespace());
        c
    } else {
        let mut c = std::process::Command::new("bash");
        c.arg(&path);
        c.args(skill_args.split_whitespace());
        c
    };
    let mut child = match command
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
    {
        Ok(c) => c,
        Err(e) => {
            let err = format!("tool spawn error: {e}");
            messages.push(Message::System(err.clone()));
            append_turn_to_session(server_url, session_id, "tool_error", &err);
            return None;
        }
    };
    if let Some(mut stdin) = child.stdin.take() {
        let _ = stdin.write_all(skill_args.as_bytes());
        // Drop closes the pipe → EOF for the child reading stdin.
    }
    match child.wait_with_output()
    {
        Ok(output) => {
            let tool_ms = tool_started.elapsed().as_millis() as u64;
            // Persist the tool invocation with how long it took, so
            // view_session can show "[stats took 82s]" inline.
            append_turn_to_session_with_meta(
                server_url,
                session_id,
                "tool_use",
                &tool_use_payload,
                Some(serde_json::json!({"tool_ms": tool_ms})),
            );
            let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

            if let Some(summary) = extract_block(&tool_output, "summary") {
                messages.push(Message::ToolResult { summary: summary.clone() });
                append_turn_to_session_with_meta(
                    server_url, session_id, "tool_result", &summary,
                    Some(serde_json::json!({"bytes": summary.len()})),
                );

                if let Some(chart) = extract_block(&tool_output, "chartjs") {
                    let chart_md = format!("```chartjs\n{chart}\n```");
                    messages.push(Message::ToolRender { content: chart_md.clone() });
                    append_turn_to_session(server_url, session_id, "tool_render", &chart_md);
                }
                mark_skill_used(server_url, skill_name, true);
                return Some(summary);
            }
            mark_skill_used(server_url, skill_name, false);
            None
        }
        Err(e) => {
            let tool_ms = tool_started.elapsed().as_millis() as u64;
            append_turn_to_session_with_meta(
                server_url, session_id, "tool_use", &tool_use_payload,
                Some(serde_json::json!({"tool_ms": tool_ms, "spawn_failed": true})),
            );
            let err = format!("tool error: {e}");
            messages.push(Message::System(err.clone()));
            append_turn_to_session(server_url, session_id, "tool_error", &err);
            mark_skill_used(server_url, skill_name, false);
            None
        }
    }
}

/// Path of the on-disk session-summary cache for `session_id`.  The
/// summary is generated in the background once the live session has
/// drifted past a threshold and is read by build_chat_messages_with_system
/// so long sessions don't develop amnesia when older turns are dropped
/// to fit the context budget.
fn session_summary_path(session_id: &str) -> std::path::PathBuf {
    let safe: String = session_id
        .chars()
        .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
        .collect();
    home_dir().join(".larql/sessions").join(format!("{safe}.summary"))
}

#[derive(serde::Serialize, serde::Deserialize, Default)]
struct SessionSummary {
    /// Number of turns covered (so we know when it's stale).
    pub turns_covered: usize,
    /// Last turn ts covered (defines the boundary — newer turns aren't
    /// part of the summary).
    pub up_to_ts: u64,
    /// 3-5 line natural-language summary of what was discussed.
    pub text: String,
}

/// Read the cached summary for this session if any.
fn load_session_summary(session_id: &str) -> Option<SessionSummary> {
    let path = session_summary_path(session_id);
    let raw = std::fs::read_to_string(&path).ok()?;
    serde_json::from_str(&raw).ok()
}

/// Spawn a background task that:
///   1. Reads the full session JSONL via /v1/sessions/<id>.
///   2. Asks /v1/chat/completions to summarise it in 3-5 bullets.
///   3. Writes the result to ~/.larql/sessions/<id>.summary so the
///      *next* chat call can use it as a synthetic system message.
///
/// Cheap to call repeatedly — short-circuits if the on-disk summary
/// already covers a turns-count close to the live one.
fn maybe_spawn_summarizer(server_url: &str, session_id: &str, current_turn_count: usize) {
    const RESUMMARIZE_EVERY: usize = 8;  // re-summarise every 8 new turns
    const MIN_TURNS: usize = 12;         // don't bother on short sessions
    if current_turn_count < MIN_TURNS {
        return;
    }
    let existing = load_session_summary(session_id);
    if let Some(s) = &existing {
        if current_turn_count.saturating_sub(s.turns_covered) < RESUMMARIZE_EVERY {
            return;
        }
    }
    let url_chat = format!("{server_url}/v1/chat/completions");
    let url_sess = format!("{server_url}/v1/sessions/{session_id}");
    let path = session_summary_path(session_id);
    let _ = std::fs::create_dir_all(path.parent().unwrap_or(std::path::Path::new(".")));
    let session_id_owned = session_id.to_string();
    tokio::spawn(async move {
        let client = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(60))
            .build()
            .unwrap_or_default();
        let resp = match client.get(&url_sess).send().await {
            Ok(r) => r,
            Err(_) => return,
        };
        let body: serde_json::Value = match resp.json().await {
            Ok(v) => v,
            Err(_) => return,
        };
        let turns = match body["turns"].as_array() {
            Some(t) => t,
            None => return,
        };
        let mut transcript = String::new();
        let mut max_ts = 0u64;
        for t in turns {
            let role = t["role"].as_str().unwrap_or("");
            let content = t["content"].as_str().unwrap_or("");
            if matches!(role, "user" | "assistant") && !content.is_empty() {
                transcript.push_str(role);
                transcript.push_str(": ");
                transcript.push_str(&content.chars().take(800).collect::<String>());
                transcript.push_str("\n\n");
            }
            if let Some(ts) = t["ts"].as_u64() { max_ts = max_ts.max(ts); }
        }
        if transcript.trim().is_empty() {
            return;
        }
        let prompt = format!(
            "Summarise the following chat history in 3-5 short bullet points capturing the key \
             facts established (user identity, preferences, recurring topics, decisions, names, \
             numbers).  Be terse — one bullet per fact.  Do not invent.  Reply with the bullets \
             only, no preamble.\n\n{transcript}"
        );
        let body = serde_json::json!({
            "messages": [{"role":"user", "content": prompt}],
            "stream": true,
            "max_tokens": 400,
            "temperature": 0.0,
        });
        let resp = match client.post(&url_chat)
            .header("content-type", "application/json")
            .body(body.to_string()).send().await
        { Ok(r) => r, Err(_) => return };
        let bytes = match resp.bytes().await { Ok(b) => b, Err(_) => return };
        let raw = String::from_utf8_lossy(&bytes).into_owned();
        let mut text = String::new();
        for line in raw.lines() {
            if let Some(data) = line.strip_prefix("data: ") {
                if data == "[DONE]" { break; }
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(c) = v["choices"][0]["delta"]["content"].as_str() {
                        text.push_str(c);
                    }
                }
            }
        }
        if text.trim().is_empty() { return; }
        let summary = SessionSummary {
            turns_covered: turns.len(),
            up_to_ts: max_ts,
            text: text.trim().to_string(),
        };
        if let Ok(json) = serde_json::to_string_pretty(&summary) {
            let _ = std::fs::write(&path, json);
        }
        let _ = session_id_owned;
    });
}

/// Peek at an assistant response for a ```tool``` block without
/// running anything.  Returns (name, detail) if found.  Used to
/// render an immediate "⚡ tool (running…)" marker before
/// `execute_skill_tool` blocks the event loop on the bash subprocess.
fn preview_tool_call(text: &str) -> Option<(String, String)> {
    let open = "```tool";
    let start = text.find(open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    let body = after[..close].trim();
    let mut parts = body.splitn(2, char::is_whitespace);
    let name = parts.next()?.to_string();
    let detail: String = parts.next().unwrap_or("").chars().take(70).collect();
    Some((name, detail))
}

/// Scan an assistant turn for ```fact``` blocks and POST each to
/// /v1/insert so the captured residual gets stored in the KnnStore at
/// L26.  This closes the LARQL loop: future prompts whose residual
/// is near `prompt_for_capture` will trigger the override.
///
/// Fact format the `annotate` skill teaches the model to emit:
///   ```fact
///   key: <short lookup key>
///   value: <the fact>
///   source: user | derived | tool
///   ```
/// Synchronously POST every fact in `response` to the global
/// KnnStore via /v1/insert.  Captures the residual at L26 of
/// `prompt_for_capture` (typically the user's most recent message).
/// Future prompts whose residual is near that capture point will hit
/// the override and emit `<value>` instead of whatever the model
/// would have said.
///
/// Synchronous (not tokio::spawn) so headless `/quit` can't drop the
/// task on the floor before it actually fires.  Each fact costs
/// ~70 ms; usually we have 1-3 per turn, so total <300 ms.
async fn ingest_facts_to_knn(
    server_url: &str,
    prompt_for_capture: &str,
    response: &str,
) {
    if prompt_for_capture.trim().is_empty() {
        return;
    }
    let facts = extract_facts(response);
    if facts.is_empty() {
        return;
    }
    let url = format!("{server_url}/v1/insert");
    let client = reqwest::Client::new();
    for f in facts {
        let body = serde_json::json!({
            "entity":     f.key,
            "relation":   "fact",
            "target":     f.value,
            "prompt":     prompt_for_capture,
            "confidence": 0.95,
        });
        let _ = client
            .post(&url)
            .header("content-type", "application/json")
            .body(body.to_string())
            .timeout(std::time::Duration::from_secs(10))
            .send()
            .await;
    }
}

#[derive(Debug)]
struct ExtractedFact {
    key: String,
    value: String,
}

fn extract_facts(text: &str) -> Vec<ExtractedFact> {
    let mut out = Vec::new();
    let mut cursor = 0;
    while let Some(rel) = text[cursor..].find("```fact") {
        let block_start = cursor + rel + "```fact".len();
        // Skip optional language-tag suffix until newline.
        let after = &text[block_start..];
        let nl = match after.find('\n') { Some(n) => n + 1, None => break };
        let body_start = block_start + nl;
        let close = match text[body_start..].find("```") { Some(c) => c, None => break };
        let body = &text[body_start..body_start + close];
        let mut key = String::new();
        let mut value = String::new();
        for line in body.lines() {
            if let Some(v) = line.strip_prefix("key:") {
                key = v.trim().to_string();
            } else if let Some(v) = line.strip_prefix("value:") {
                value = v.trim().to_string();
            }
        }
        if !key.is_empty() && !value.is_empty() {
            out.push(ExtractedFact { key, value });
        }
        cursor = body_start + close + 3;
    }
    out
}

/// Fork the skill_wizard binary in the background.  The TUI keeps
/// chatting while the wizard works; on the user's next prompt the
/// skills directory will be re-scanned and the new skill will appear
/// in routing.
fn spawn_background_wizard(skill_name: &str, prompt: &str, server_url: &str) {
    // Locate the binary next to the larql workspace.  Fall back to
    // PATH if not found at the expected location.
    let candidate = std::path::PathBuf::from(
        "/Users/miguel_lemos/Desktop/llm-as-a-database/larql/target/release/examples/skill_wizard",
    );
    let bin = if candidate.exists() {
        candidate.into_os_string()
    } else {
        std::ffi::OsString::from("skill_wizard")
    };
    let name = skill_name.to_string();
    let prompt = if prompt.is_empty() {
        format!("create the {name} skill")
    } else {
        prompt.to_string()
    };
    let server = server_url.to_string();
    // Log the build to /tmp so we can see what happened after the
    // user's session ends.  Each wizard invocation gets its own file.
    let log_path = std::env::temp_dir().join(format!("larql_wizard_{name}.log"));
    std::thread::spawn(move || {
        let log = match std::fs::File::create(&log_path) {
            Ok(f) => f,
            Err(_) => return,
        };
        let log2 = match log.try_clone() { Ok(f) => f, Err(_) => return };
        let _ = std::process::Command::new(&bin)
            .args(["build", "--prompt", &prompt, "--name", &name])
            .env("LARQL_SERVER", &server)
            .stdout(std::process::Stdio::from(log))
            .stderr(std::process::Stdio::from(log2))
            .stdin(std::process::Stdio::null())
            .spawn()
            .map(|mut c| c.wait());
    });
}

/// POST /v1/skills/{name}/used so the server's SQLite catalog tracks
/// which skills are actually getting used and which are failing.
/// Fire-and-forget — never blocks the chat loop.
fn mark_skill_used(server_url: &str, name: &str, success: bool) {
    let url = format!("{server_url}/v1/skills/{name}/used");
    let body = serde_json::json!({"success": success}).to_string();
    tokio::spawn(async move {
        let client = reqwest::Client::new();
        let _ = client
            .post(&url)
            .header("content-type", "application/json")
            .body(body)
            .timeout(std::time::Duration::from_secs(2))
            .send()
            .await;
    });
}

fn extract_block(text: &str, lang: &str) -> Option<String> {
    let open = format!("```{lang}");
    let start = text.find(&open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    Some(after[..close].trim().to_string())
}

// ── Drawing ──────────────────────────────────────────────────────────────

fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, state: &AppState) {
    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),
                Constraint::Length(3),
                Constraint::Length(1),
            ])
            .split(f.area());

        draw_messages(f, state, chunks[0]);
        draw_input(f, state, chunks[1]);
        draw_status(f, state, chunks[2]);
    }).ok();
}

fn draw_messages(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let mut lines: Vec<Line> = Vec::new();

    for msg in &state.messages {
        match msg {
            Message::User(text) => {
                lines.push(Line::from(vec![
                    Span::styled("❯ ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)),
                    Span::styled(text.as_str(), Style::default().fg(Color::White).add_modifier(Modifier::BOLD)),
                ]));
                lines.push(Line::from(""));
            }
            Message::Assistant(text) => {
                lines.extend(gc_markdown::render_markdown(text, gc_markdown::Theme::Dark));
                lines.push(Line::from(""));
            }
            Message::System(text) => {
                lines.push(Line::from(Span::styled(
                    format!("  {text}"), Style::default().fg(Color::DarkGray).add_modifier(Modifier::ITALIC),
                )));
                lines.push(Line::from(""));
            }
            Message::ToolUse { tool, detail } => {
                lines.push(Line::from(vec![
                    Span::styled("  ⚡ ", Style::default().fg(Color::Magenta)),
                    Span::styled(tool.as_str(), Style::default().fg(Color::Magenta).add_modifier(Modifier::BOLD)),
                    Span::styled(format!(" {detail}"), Style::default().fg(Color::DarkGray)),
                ]));
            }
            Message::ToolResult { summary } => {
                lines.extend(gc_markdown::render_markdown(summary, gc_markdown::Theme::Dark));
                lines.push(Line::from(""));
            }
            Message::ToolRender { content } => {
                lines.extend(gc_markdown::render_markdown(content, gc_markdown::Theme::Dark));
                lines.push(Line::from(""));
            }
        }
    }

    // Pre-wrap into a flat Vec<Line> where each entry is exactly one
    // terminal row — no surprises from Paragraph's internal wrap. This
    // makes scrolling deterministic: scroll offset == row offset.
    let inner_width = area.width.saturating_sub(2) as usize;
    let wrapped: Vec<Line<'static>> = lines
        .into_iter()
        .flat_map(|line| wrap_line_to_rows(line, inner_width))
        .collect();

    let visible = area.height.saturating_sub(2) as usize;
    let total = wrapped.len();
    let scroll_off = state.scroll_offset as usize;
    let end = total.saturating_sub(scroll_off);
    let start = end.saturating_sub(visible);
    let view: Vec<Line<'static>> = wrapped[start..end].to_vec();

    let title_text = if state.scroll_offset > 0 {
        format!(
            " larql · scrolled +{} (End to follow) [{}/{}] ",
            state.scroll_offset, end, total
        )
    } else {
        format!(" larql [{}/{}] ", end, total)
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(title_text, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    // No wrap: each entry already fits.  Trim=false preserves indents.
    let para = Paragraph::new(view).block(block);
    f.render_widget(para, area);
}

/// Split a `Line` into one or more lines that each fit in `width`
/// terminal columns.  Uses unicode-width so emojis and CJK wide chars
/// are counted correctly.  Style is preserved per fragment.
fn wrap_line_to_rows(line: Line<'_>, width: usize) -> Vec<Line<'static>> {
    use unicode_width::UnicodeWidthChar as _;
    if width == 0 {
        return vec![Line::from("")];
    }
    let total_w: usize = line
        .spans
        .iter()
        .map(|s| unicode_width::UnicodeWidthStr::width(s.content.as_ref()))
        .sum();
    if total_w <= width {
        // Fast path — already fits.  Clone spans into 'static.
        let spans: Vec<Span<'static>> = line
            .spans
            .into_iter()
            .map(|s| Span::styled(s.content.into_owned(), s.style))
            .collect();
        return vec![Line::from(spans)];
    }

    let mut out: Vec<Line<'static>> = Vec::new();
    let mut cur: Vec<Span<'static>> = Vec::new();
    let mut cur_w: usize = 0;
    for span in line.spans {
        let style = span.style;
        let mut buf = String::new();
        let mut buf_w = 0usize;
        for ch in span.content.chars() {
            let cw = unicode_width::UnicodeWidthChar::width(ch).unwrap_or(0);
            if cur_w + buf_w + cw > width {
                if !buf.is_empty() {
                    cur.push(Span::styled(std::mem::take(&mut buf), style));
                    buf_w = 0;
                }
                if !cur.is_empty() {
                    out.push(Line::from(std::mem::take(&mut cur)));
                    cur_w = 0;
                }
            }
            buf.push(ch);
            buf_w += cw;
        }
        if !buf.is_empty() {
            cur.push(Span::styled(buf, style));
            cur_w += buf_w;
        }
    }
    if !cur.is_empty() {
        out.push(Line::from(cur));
    }
    if out.is_empty() {
        out.push(Line::from(""));
    }
    out
}

/// Helper used by key handlers — clamp scroll_offset so we don't
/// scroll past the top of the buffer.  Caller passes the current
/// rendered area size to know the visible page height.
fn clamp_scroll(state: &mut AppState, msg_area_height: u16) {
    let max_back = max_scroll(state, msg_area_height);
    if state.scroll_offset > max_back {
        state.scroll_offset = max_back;
    }
}

fn max_scroll(state: &AppState, msg_area_height: u16) -> u16 {
    let visible = msg_area_height.saturating_sub(2);
    // Approximate wrapped height the same way draw_messages does, just
    // with an unknown width.  Using a wide-ish default (120) is fine —
    // worst case we let the user scroll a few lines past the top, which
    // ratatui clamps anyway.
    let inner_width: usize = 120;
    let total: usize = state
        .messages
        .iter()
        .flat_map(message_estimated_rows)
        .map(|content_len| if content_len == 0 { 1 } else { content_len.div_ceil(inner_width) })
        .sum();
    let bottom_scroll: u16 = total
        .saturating_sub(visible as usize)
        .try_into()
        .unwrap_or(u16::MAX);
    bottom_scroll
}

fn message_estimated_rows(msg: &Message) -> Vec<usize> {
    match msg {
        Message::User(t)
        | Message::Assistant(t)
        | Message::System(t)
        | Message::ToolResult { summary: t }
        | Message::ToolRender { content: t } => {
            // Char count (not byte len) for accurate column estimation
            // when the content includes multi-byte UTF-8 (emoji etc).
            let mut v: Vec<usize> = t.lines().map(|l| l.chars().count()).collect();
            v.push(0);
            v
        }
        Message::ToolUse { tool, detail } => {
            vec![tool.chars().count() + detail.chars().count() + 4]
        }
    }
}

fn draw_input(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let (style, text) = if state.is_generating {
        (Style::default().fg(Color::DarkGray), "  generating...".to_string())
    } else if state.input.is_empty() {
        (Style::default().fg(Color::DarkGray), "  Type a question...".to_string())
    } else {
        (Style::default().fg(Color::White), format!("  {}", state.input))
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if state.is_generating { Color::DarkGray } else { Color::Cyan }));
    let para = Paragraph::new(Line::from(Span::styled(text, style))).block(block);
    f.render_widget(para, area);

    if !state.is_generating {
        f.set_cursor_position((area.x + 3 + state.cursor as u16, area.y + 1));
    }
}

fn draw_status(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let status = format!(" {} ", state.status);
    let para = Paragraph::new(Line::from(Span::styled(
        status, Style::default().fg(Color::White).bg(Color::DarkGray),
    )));
    f.render_widget(para, area);
}

// ── Session Loading ──────────────────────────────────────────────────────

/// Load a Claude Code session (.jsonl) or TUI session into RAG facts.
/// Searches ~/.claude/projects/*/SESSION_ID.jsonl for Claude sessions.
async fn load_session_into_rag(server_url: &str, session_id: &str) -> Result<usize, String> {
    // Find the session file
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let claude_dir = std::path::PathBuf::from(&home).join(".claude/projects");

    let mut session_path = None;
    if let Ok(entries) = std::fs::read_dir(&claude_dir) {
        for entry in entries.flatten() {
            let candidate = entry.path().join(format!("{session_id}.jsonl"));
            if candidate.exists() {
                session_path = Some(candidate);
                break;
            }
        }
    }
    // Also check direct path (if user passed a file path)
    if session_path.is_none() {
        let direct = std::path::PathBuf::from(session_id);
        if direct.exists() {
            session_path = Some(direct);
        }
    }

    let path = session_path.ok_or_else(|| format!("session {session_id} not found"))?;

    // Parse JSONL and extract facts
    let content = std::fs::read_to_string(&path)
        .map_err(|e| format!("read session: {e}"))?;

    let client = reqwest::Client::new();
    let mut inserted = 0usize;

    for line in content.lines() {
        let obj: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v, Err(_) => continue,
        };

        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let ts = obj.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        let ts_short = &ts[..ts.len().min(19)];

        // Only insert assistant messages — they contain the actual knowledge.
        // User messages are short/rambling and pollute the embedding space.
        let fact = match entry_type {
            "assistant" => {
                let blocks = obj.pointer("/message/content")
                    .and_then(|v| v.as_array());
                if let Some(blocks) = blocks {
                    let text: String = blocks.iter()
                        .filter_map(|b| {
                            if b.get("type")?.as_str()? == "text" {
                                b.get("text")?.as_str().map(|s| s.to_string())
                            } else { None }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if text.len() > 50 {
                        Some(format!("[{}] {}", ts_short, &text[..text.char_indices().take(300).last().map(|(i,c)| i + c.len_utf8()).unwrap_or(0)]))
                    } else { None }
                } else { None }
            }
            _ => None,
        };

        if let Some(fact_text) = fact {
            let resp = client.post(format!("{server_url}/v1/rag/insert"))
                .json(&serde_json::json!({
                    "fact": fact_text,
                    "category": "session",
                }))
                .timeout(std::time::Duration::from_secs(5))
                .send().await;
            if resp.is_ok() { inserted += 1; }
        }
    }

    Ok(inserted)
}

// ── Headless mode ───────────────────────────────────────────────────────
//
// Drives the same `spawn_chat` / `build_chat_messages` flow as the TUI,
// but reads prompts from stdin and writes streamed tokens to stdout.
// Used for scripted multi-turn verification of the model + harness.
//
// Per-turn protocol: a line of stdin = one user message.  On the same
// stdout, the model's reply is streamed token-by-token, terminated by a
// trailing `\n`.  History accumulates in `AppState.messages` exactly as
// the TUI would maintain it, so context (and the server-side KNN
// overlay) behaves identically.
async fn run_headless(server_url: &str, session_id: Option<&str>) -> io::Result<()> {
    use std::io::Write;
    use tokio::io::{AsyncBufReadExt, BufReader};

    let mut state = AppState::new(server_url);
    state.session_id = session_id.map(|s| s.to_string());
    state.skills = load_skills(&skill_dirs());

    // Resume from server log if a session id was passed.
    if let Some(sid) = session_id {
        let history = fetch_session_history(server_url, sid).await;
        if !history.is_empty() {
            for (role, content) in &history {
                match role.as_str() {
                    "user" => state.messages.push(Message::User(content.clone())),
                    "assistant" => state.messages.push(Message::Assistant(content.clone())),
                    "tool_use" => {
                        // Persisted as "<name> <args>" — split for the
                        // pretty-printed marker.
                        let mut parts = content.splitn(2, char::is_whitespace);
                        let tool = parts.next().unwrap_or("").to_string();
                        let detail = parts.next().unwrap_or("").to_string();
                        state.messages.push(Message::ToolUse { tool, detail });
                    }
                    "tool_result" => state.messages.push(Message::ToolResult {
                        summary: content.clone(),
                    }),
                    "tool_render" => state.messages.push(Message::ToolRender {
                        content: content.clone(),
                    }),
                    "tool_error" => state.messages.push(Message::System(content.clone())),
                    _ => {}
                }
            }
            eprintln!("[headless] resumed session {sid} ({} prior turns)", history.len());
        } else {
            eprintln!("[headless] new session {sid}");
        }
    }

    // Health check.
    match reqwest::Client::new()
        .get(format!("{server_url}/v1/health"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            eprintln!("[headless] connected · {server_url}");
        }
        _ => {
            eprintln!("[headless] cannot reach {server_url}");
            return Ok(());
        }
    }
    eprintln!("[headless] type a prompt; Ctrl-D or /quit to exit; /reset to clear history");
    eprintln!();

    let stdin = tokio::io::stdin();
    let mut stdin = BufReader::new(stdin).lines();

    loop {
        eprint!("you ❯ ");
        let _ = io::stderr().flush();
        let line = match stdin.next_line().await {
            Ok(Some(l)) => l,
            Ok(None) => break,
            Err(e) => {
                eprintln!("[headless] read error: {e}");
                break;
            }
        };
        let trimmed = line.trim().to_string();
        if trimmed.is_empty() {
            continue;
        }
        if trimmed == "/quit" || trimmed == "/exit" {
            break;
        }
        if trimmed == "/reset" {
            state.messages.retain(|m| matches!(m, Message::System(_)));
            eprintln!("[headless] history cleared");
            continue;
        }

        state.messages.push(Message::User(trimmed));
        run_one_turn(&mut state, server_url, /*tool_depth=*/ 0).await;
    }

    eprintln!("[headless] bye");
    Ok(())
}

/// Discover all skill directories the TUI cares about.
fn skill_dirs() -> Vec<std::path::PathBuf> {
    vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        home_dir().join(".larql/skills"),
    ]
}

/// Compose the system primer for one chat turn:
///   * a tiny header listing every skill name the model could invoke,
///   * the body of every `always: true` skill (e.g. `annotate`),
///   * the body of the *one* skill that the TF-IDF router picked for
///     the user's latest message — or nothing if no match cleared the
///     confidence floor.
///
/// This keeps the prefill small and roughly constant regardless of how
/// many skills are installed.  Returns Some(primer, route) where route
/// is the (skill_name, confidence) chosen, or None.
fn routed_primer(
    skills: &[Skill],
    latest_user: Option<&str>,
) -> Option<(String, Option<(String, f32)>)> {
    if skills.is_empty() {
        return None;
    }
    let always: Vec<&Skill> = skills.iter().filter(|s| s.always).collect();
    let routable: Vec<&Skill> = skills.iter().filter(|s| !s.always).collect();
    let routable_names: Vec<&str> = routable.iter().map(|s| s.name.as_str()).collect();

    let header = format!(
        "You have access to {} skills: {}.\n\
         When the user wants one, emit a fenced code block with \
         language `tool`, e.g.:\n\n```tool\nlist /tmp\n```\n\n\
         The system runs the matching `~/.larql/skills/<name>/tool.sh` \
         and feeds the result back to you for commentary.\n",
        routable_names.len(),
        routable_names.join(", ")
    );

    let mut primer = header;
    for s in &always {
        primer.push_str("\n---\n\n");
        primer.push_str(s.body.trim());
        primer.push('\n');
    }

    let mut chosen: Option<(String, f32)> = None;
    if let Some(prompt) = latest_user {
        let idx = build_index(skills);
        if let Some((i, conf)) = route(prompt, &idx) {
            chosen = Some((skills[i].name.clone(), conf));
            primer.push_str("\n---\n\n");
            primer.push_str(skills[i].body.trim());
            primer.push('\n');
        }
    }

    Some((primer, chosen))
}

/// Execute one chat turn: stream the reply, then if it contains a
/// `tool` block run it and recursively follow up.  Bounded by
/// `tool_depth` to stop infinite loops.
fn run_one_turn<'a>(
    state: &'a mut AppState,
    server_url: &'a str,
    tool_depth: usize,
) -> std::pin::Pin<Box<dyn std::future::Future<Output = ()> + 'a>> {
    Box::pin(async move {
        use std::io::Write;
        const MAX_TOOL_DEPTH: usize = 4;

        state.messages.push(Message::Assistant(String::new()));

        let chat_msgs = build_chat_messages_with_system(state);
        let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);
        spawn_chat(server_url.to_string(), chat_msgs, ev_tx, state.session_id.clone());

        eprint!("bot ❯ ");
        let _ = io::stderr().flush();
        loop {
            match ev_rx.recv().await {
                Some(StreamEvent::Token(tok)) => {
                    if let Some(Message::Assistant(ref mut text)) = state.messages.last_mut() {
                        text.push_str(&tok);
                    }
                    print!("{tok}");
                    let _ = io::stdout().flush();
                }
                Some(StreamEvent::Done) | None => break,
                Some(StreamEvent::Error(e)) => {
                    eprintln!("\n[headless] error: {e}");
                    return;
                }
            }
        }
        println!();

        // Drop empty assistant turn (server error etc.).
        if let Some(Message::Assistant(text)) = state.messages.last() {
            if text.is_empty() {
                state.messages.pop();
                return;
            }
        }

        // Tool execution.
        if tool_depth >= MAX_TOOL_DEPTH {
            return;
        }
        let response_text = match state.messages.last() {
            Some(Message::Assistant(t)) => t.clone(),
            _ => return,
        };

        // Close the KNN loop: every fact the model just emitted gets
        // pushed into the live KnnStore via /v1/insert, captured at
        // the user's most-recent message position.  Future semantically
        // similar prompts will hit the override at L26.
        let latest_user_for_facts: String = state
            .messages
            .iter()
            .rev()
            .find_map(|m| if let Message::User(t) = m { Some(t.clone()) } else { None })
            .unwrap_or_default();
        ingest_facts_to_knn(server_url, &latest_user_for_facts, &response_text).await;

        // Long-session summarisation: kick off in the background
        // every few turns so the next chat can use a cached summary
        // instead of letting older turns silently fall off the
        // byte-budget cliff.
        if let Some(sid) = state.session_id.as_deref() {
            let turn_count = state
                .messages
                .iter()
                .filter(|m| matches!(m, Message::User(_) | Message::Assistant(_)))
                .count();
            maybe_spawn_summarizer(server_url, sid, turn_count);
        }

        let sid = state.session_id.clone();
        if let Some(summary) = execute_skill_tool(&response_text, &mut state.messages, server_url, sid.as_deref(), &state.skills) {
            // Surface tool flow on stdout so it's visible when scripted.
            if let Some(Message::ToolUse { tool, detail }) =
                state.messages.iter().rev().find(|m| matches!(m, Message::ToolUse { .. }))
            {
                eprintln!("[tool] {tool} {detail}");
            }
            eprintln!("[tool result]");
            for line in summary.lines().take(40) {
                eprintln!("  {line}");
            }
            // Follow-up turn — `build_chat_messages_with_system` now
            // converts the just-pushed `ToolResult` into a synthetic
            // user-role turn, so the model sees the tool output and
            // can write a commentary reply.
            let _ = summary;
            run_one_turn(state, server_url, tool_depth + 1).await;
        }
    })
}

/// Slice a UTF-8 string by byte range without splitting a codepoint.
fn utf8_safe_slice(s: &str, mut start: usize, mut end: usize) -> &str {
    start = start.min(s.len());
    end = end.min(s.len());
    while start < end && !s.is_char_boundary(start) {
        start += 1;
    }
    while end > start && !s.is_char_boundary(end) {
        end -= 1;
    }
    &s[start..end]
}

/// Build chat messages for the model.
///
/// Rules:
/// 1. System primer is composed dynamically per turn: the
///    `always`-tagged skills + the *one* skill the TF-IDF router
///    chooses for the latest user message.  Plus any free-form
///    `Message::System` the AppState carries (e.g. status banners).
/// 2. A `ToolResult` is "live" only when it's the most recent
///    interaction (no `Assistant` came after it).  Once the model has
///    answered for it, the result is dropped — replaying it on every
///    later turn pollutes context and breaks role alternation.
/// 3. Consecutive user turns are coalesced into one (Gemma's chat
///    template expects strict user/assistant alternation).
/// 4. The tail is then bounded by a byte budget so prefill fits in
///    n_ctx; older turns are dropped first.
fn build_chat_messages_with_system(state: &AppState) -> Vec<ChatMsg> {
    const MAX_TAIL_BYTES: usize = 16_000;

    let mut live_start = state.messages.len();
    for (i, m) in state.messages.iter().enumerate().rev() {
        if matches!(m, Message::Assistant(t) if !t.is_empty()) {
            live_start = i + 1;
            break;
        }
        live_start = i;
    }

    // Latest user message — drives the router.
    let latest_user: Option<&str> = state
        .messages
        .iter()
        .rev()
        .find_map(|m| if let Message::User(t) = m { (!t.is_empty()).then_some(t.as_str()) } else { None });

    // Cheap reload — picks up skills the background wizard installed
    // since the last chat send.  Only re-parses skill.md headers, not
    // tool.sh bodies, so it stays sub-millisecond even for many skills.
    // (state is &AppState; we need interior mutability, so just read
    // freshly without storing back — the router uses the result for
    // this one call.)
    let live_skills: Vec<Skill> = {
        let scanned = load_skills(&skill_dirs());
        if scanned.len() != state.skills.len() { scanned } else { state.skills.clone() }
    };

    let mut system_msgs: Vec<ChatMsg> = Vec::new();

    // Inject the cached session summary (if any) BEFORE the skill
    // primer so the model sees long-conversation context that has
    // since aged out of the byte-budget tail.
    if let Some(sid) = state.session_id.as_deref() {
        if let Some(summary) = load_session_summary(sid) {
            if !summary.text.trim().is_empty() {
                system_msgs.push(ChatMsg {
                    role: "system".into(),
                    content: format!(
                        "Earlier conversation summary (the {} oldest turns have been summarised \
                         to keep context small):\n\n{}",
                        summary.turns_covered, summary.text
                    ),
                });
            }
        }
    }

    if let Some((primer, route_decision)) = routed_primer(&live_skills, latest_user) {
        system_msgs.push(ChatMsg {
            role: "system".into(),
            content: primer,
        });
        // Persist the routing decision so view_session can audit it
        // ("[router=clock conf=0.78]" or "no_match" if the router
        // didn't choose).  Fire & forget — never block the chat call.
        let mut meta = serde_json::Map::new();
        let content = match &route_decision {
            Some((name, conf)) => {
                meta.insert("skill".into(), serde_json::Value::String(name.clone()));
                meta.insert(
                    "confidence".into(),
                    serde_json::Value::from(((*conf) * 10000.0).round() / 10000.0),
                );
                format!("{name} (conf={:.2})", conf)
            }
            None => "no_match".to_string(),
        };
        meta.insert(
            "n_skills".into(),
            serde_json::Value::from(state.skills.len()),
        );
        if let Some(p) = latest_user {
            meta.insert(
                "prompt_chars".into(),
                serde_json::Value::from(p.chars().count()),
            );
        }
        append_turn_to_session_with_meta(
            &state.server_url,
            state.session_id.as_deref(),
            "router",
            &content,
            Some(serde_json::Value::Object(meta)),
        );
    }

    let mut all: Vec<ChatMsg> = Vec::new();
    for (i, m) in state.messages.iter().enumerate() {
        match m {
            Message::System(text) if !text.is_empty() => {
                // Free-form System banners (the welcome line, "Server
                // connected.", etc.) — append after the routed primer.
                system_msgs.push(ChatMsg {
                    role: "system".into(),
                    content: text.clone(),
                });
            }
            Message::User(text) if !text.is_empty() => {
                all.push(ChatMsg {
                    role: "user".into(),
                    content: text.clone(),
                });
            }
            Message::Assistant(text) if !text.is_empty() => {
                all.push(ChatMsg {
                    role: "assistant".into(),
                    content: text.clone(),
                });
            }
            Message::ToolResult { summary } if !summary.is_empty() && i >= live_start => {
                // Live tool result — needs a model response.
                const HEAD: usize = 4000;
                const TAIL: usize = 1500;
                let body = if summary.len() > HEAD + TAIL + 64 {
                    let head = utf8_safe_slice(summary, 0, HEAD);
                    let tail = utf8_safe_slice(summary, summary.len() - TAIL, summary.len());
                    format!(
                        "{head}\n…(truncated, {} bytes omitted)…\n{tail}",
                        summary.len() - HEAD - TAIL
                    )
                } else {
                    summary.clone()
                };
                all.push(ChatMsg {
                    role: "user".into(),
                    content: format!(
                        "Tool result:\n{body}\n\nPlease summarise this concisely for me."
                    ),
                });
            }
            _ => {}
        }
    }

    // Coalesce consecutive same-role messages (chat template expects
    // strict alternation; multiple users in a row get merged).
    let mut coalesced: Vec<ChatMsg> = Vec::new();
    for m in all {
        match coalesced.last_mut() {
            Some(prev) if prev.role == m.role => {
                prev.content.push_str("\n\n");
                prev.content.push_str(&m.content);
            }
            _ => coalesced.push(m),
        }
    }

    // Walk newest→oldest, keep until budget is exhausted.
    let mut tail: Vec<ChatMsg> = Vec::new();
    let mut budget = MAX_TAIL_BYTES;
    for m in coalesced.into_iter().rev() {
        if m.content.len() > budget && !tail.is_empty() {
            break;
        }
        budget = budget.saturating_sub(m.content.len());
        tail.push(m);
    }
    tail.reverse();

    let mut out = system_msgs;
    out.extend(tail);
    out
}

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> io::Result<()> {
    let server_url = std::env::var("LARQL_SERVER")
        .unwrap_or_else(|_| "http://localhost:3000".into());

    // Session selection rules:
    //   * default              → derive id from cwd; resume if it exists.
    //   * --new-session        → derive id from cwd, but wipe any prior
    //                            log so this run starts fresh.
    //   * --session <id>       → use the explicit id, resume if exists.
    //   * --session <id> --new-session → use explicit id but wipe first.
    let args: Vec<String> = std::env::args().collect();
    let explicit_session = args
        .iter()
        .position(|a| a == "--session")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let new_session = args.iter().any(|a| a == "--new-session");
    let headless = args.iter().any(|a| a == "--headless");

    let session_id = explicit_session.unwrap_or_else(default_session_id_from_cwd);

    if new_session {
        // Best-effort wipe of the existing log so this conversation
        // starts clean.  Non-fatal if it doesn't exist.
        let _ = reqwest::Client::new()
            .delete(format!("{server_url}/v1/sessions/{session_id}"))
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
    }

    if headless {
        return run_headless(&server_url, Some(&session_id)).await;
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen, EnableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new(&server_url);
    state.session_id = Some(session_id.clone());
    // Skills are loaded once; per-turn the chat builder picks ONE
    // (TF-IDF router) plus all `always: true` skills (annotate) so the
    // prefill stays small regardless of how many skills exist.
    state.skills = load_skills(&skill_dirs());
    draw(&mut terminal, &state);

    // Check server health
    match reqwest::Client::new().get(format!("{server_url}/v1/health"))
        .timeout(std::time::Duration::from_secs(3))
        .send().await
    {
        Ok(r) if r.status().is_success() => {
            state.status = format!("connected · {server_url}");
            state.messages.push(Message::System("Server connected.".into()));
        }
        _ => {
            state.status = "server not reachable — start larql-server first".into();
            state.messages.push(Message::System(
                format!("Cannot reach {server_url}. Start the server:\n  cargo run --release -p larql-server -- /path/to/vindex")
            ));
        }
    }
    draw(&mut terminal, &state);

    // Always attempt to resume — session id is either cwd-derived or
    // explicit.  Subsequent chats carry X-Session-Id so new turns
    // append to the same log.
    {
        let sid = &session_id;
        state.status = format!("loading session {sid}...");
        draw(&mut terminal, &state);

        let history = fetch_session_history(&server_url, sid).await;
        if history.is_empty() {
            state.messages.push(Message::System(
                format!("Session '{sid}' — starting fresh.")
            ));
        } else {
            for (role, content) in &history {
                match role.as_str() {
                    "user" => state.messages.push(Message::User(content.clone())),
                    "assistant" => state.messages.push(Message::Assistant(content.clone())),
                    "tool_use" => {
                        // Persisted as "<name> <args>" — split for the
                        // pretty-printed marker.
                        let mut parts = content.splitn(2, char::is_whitespace);
                        let tool = parts.next().unwrap_or("").to_string();
                        let detail = parts.next().unwrap_or("").to_string();
                        state.messages.push(Message::ToolUse { tool, detail });
                    }
                    "tool_result" => state.messages.push(Message::ToolResult {
                        summary: content.clone(),
                    }),
                    "tool_render" => state.messages.push(Message::ToolRender {
                        content: content.clone(),
                    }),
                    "tool_error" => state.messages.push(Message::System(content.clone())),
                    _ => {}
                }
            }
            state.status = format!("connected · {server_url} · session {sid} ({} turns)", history.len());
            state.messages.push(Message::System(
                format!("Resumed session '{sid}' with {} turns.", history.len())
            ));
        }
        draw(&mut terminal, &state);
    }

    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);
    let mut tool_depth: usize = 0; // prevent infinite tool execution chains

    loop {
        let mut new_output = false;
        while let Ok(ev) = ev_rx.try_recv() {
            match ev {
                StreamEvent::Token(tok) => {
                    if let Some(Message::Assistant(ref mut text)) = state.messages.last_mut() {
                        text.push_str(&tok);
                    } else {
                        state.messages.push(Message::Assistant(tok));
                    }
                    new_output = true;
                }
                StreamEvent::Done => {
                    let response_text = match state.messages.last() {
                        Some(Message::Assistant(text)) => text.clone(),
                        _ => String::new(),
                    };

                    // Close the KNN loop: every `fact` block the model
                    // just emitted gets stashed in the live KnnStore
                    // via /v1/insert.  Captured at the user's latest
                    // message position so future similar prompts hit
                    // the L26 override.
                    let latest_user: String = state.messages.iter().rev()
                        .find_map(|m| if let Message::User(t) = m { Some(t.clone()) } else { None })
                        .unwrap_or_default();
                    {
                        let url = state.server_url.clone();
                        let prompt = latest_user.clone();
                        let resp = response_text.clone();
                        let h = tokio::runtime::Handle::current();
                        let _ = h.block_on(async {
                            ingest_facts_to_knn(&url, &prompt, &resp).await;
                        });
                    }
                    // Long-session summariser tick.
                    if let Some(sid) = state.session_id.as_deref() {
                        let turn_count = state
                            .messages
                            .iter()
                            .filter(|m| matches!(m, Message::User(_) | Message::Assistant(_)))
                            .count();
                        maybe_spawn_summarizer(&state.server_url, sid, turn_count);
                    }

                    // Only execute tools on first response, not follow-ups
                    if tool_depth == 0 {
                        let sid = state.session_id.clone();
                        let server_url_cl = state.server_url.clone();
                        // Pre-render the "⚡ tool" marker BEFORE
                        // execute_skill_tool blocks on bash.  Without
                        // this, slow tools (8s `stats`, 7s
                        // `find_large`) freeze the screen with no
                        // feedback.  We push a placeholder ToolUse
                        // here, draw, then let execute_skill_tool
                        // overwrite it (its own ToolUse push is a
                        // duplicate but the dedupe in append_turn
                        // catches it).
                        if let Some((preview_name, preview_detail)) =
                            preview_tool_call(&response_text)
                        {
                            state.messages.push(Message::ToolUse {
                                tool: format!("{preview_name} (running…)"),
                                detail: preview_detail,
                            });
                            draw(&mut terminal, &state);
                        }
                        if let Some(_summary) = execute_skill_tool(&response_text, &mut state.messages, &server_url_cl, sid.as_deref(), &state.skills) {
                            tool_depth += 1;
                            state.messages.push(Message::Assistant(String::new()));
                            let chat_msgs = build_chat_messages_with_system(&state);
                            spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone(), state.session_id.clone());
                        } else {
                            state.is_generating = false;
                            tool_depth = 0;
                        }
                    } else {
                        state.is_generating = false;
                        tool_depth = 0;
                    }
                    new_output = true;
                }
                StreamEvent::Error(e) => {
                    state.messages.push(Message::System(format!("Error: {e}")));
                    state.is_generating = false;
                    new_output = true;
                }
            }
        }

        if new_output {
            draw(&mut terminal, &state);
        }

        if event::poll(std::time::Duration::from_millis(30))? {
            let evt = event::read()?;
            // ── Mouse wheel scroll ──
            if let CEvent::Mouse(m) = evt {
                let msg_h = terminal.size().map(|s| s.height.saturating_sub(4)).unwrap_or(20);
                match m.kind {
                    MouseEventKind::ScrollUp => {
                        state.scroll_offset = state.scroll_offset.saturating_add(3);
                        clamp_scroll(&mut state, msg_h);
                        draw(&mut terminal, &state);
                    }
                    MouseEventKind::ScrollDown => {
                        state.scroll_offset = state.scroll_offset.saturating_sub(3);
                        draw(&mut terminal, &state);
                    }
                    _ => {}
                }
                continue;
            }
            if let CEvent::Key(key) = evt {
                if key.kind != KeyEventKind::Press { continue; }
                match key.code {
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Enter if !state.is_generating => {
                        let input = state.input.trim().to_string();
                        if input.is_empty() { continue; }
                        state.input.clear();
                        state.cursor = 0;

                        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
                            break;
                        }

                        state.messages.push(Message::User(input));
                        state.is_generating = true;
                        state.messages.push(Message::Assistant(String::new()));
                        draw(&mut terminal, &state);

                        // Send full conversation history (includes
                        // system primer for skill awareness).
                        let chat_msgs = build_chat_messages_with_system(&state);
                        spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone(), state.session_id.clone());
                    }
                    KeyCode::Char(c) if !state.is_generating => {
                        state.input.insert(state.cursor, c);
                        state.cursor += 1;
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Backspace if !state.is_generating && state.cursor > 0 => {
                        state.cursor -= 1;
                        state.input.remove(state.cursor);
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Left if state.cursor > 0 => { state.cursor -= 1; draw(&mut terminal, &state); }
                    KeyCode::Right if state.cursor < state.input.len() => { state.cursor += 1; draw(&mut terminal, &state); }
                    // ── Scroll keys ──
                    KeyCode::Up => {
                        let msg_h = terminal.size().map(|s| s.height.saturating_sub(4)).unwrap_or(20);
                        state.scroll_offset = state.scroll_offset.saturating_add(1);
                        clamp_scroll(&mut state, msg_h);
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Down => {
                        state.scroll_offset = state.scroll_offset.saturating_sub(1);
                        draw(&mut terminal, &state);
                    }
                    KeyCode::PageUp => {
                        let msg_h = terminal.size().map(|s| s.height.saturating_sub(4)).unwrap_or(20);
                        let page = msg_h.saturating_sub(2).max(1);
                        state.scroll_offset = state.scroll_offset.saturating_add(page);
                        clamp_scroll(&mut state, msg_h);
                        draw(&mut terminal, &state);
                    }
                    KeyCode::PageDown => {
                        let msg_h = terminal.size().map(|s| s.height.saturating_sub(4)).unwrap_or(20);
                        let page = msg_h.saturating_sub(2).max(1);
                        state.scroll_offset = state.scroll_offset.saturating_sub(page);
                        draw(&mut terminal, &state);
                    }
                    KeyCode::End | KeyCode::Esc => {
                        state.scroll_offset = 0;
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Home => {
                        let msg_h = terminal.size().map(|s| s.height.saturating_sub(4)).unwrap_or(20);
                        state.scroll_offset = max_scroll(&state, msg_h);
                        draw(&mut terminal, &state);
                    }
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), DisableMouseCapture, LeaveAlternateScreen)?;
    Ok(())
}
