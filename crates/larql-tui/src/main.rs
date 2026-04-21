//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

use std::io;

mod skill_router;
mod workflows;
use skill_router::{build_index, load_skills, route, Skill};
use workflows::{
    extract_plan_blocks, extract_status_updates, StepState, Workflow, WorkflowState,
    WorkflowStore,
};

use crossterm::event::{
    self, DisableMouseCapture, Event as CEvent, KeyCode, KeyEventKind,
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
    /// Inverse of ToolRender: goes to the model (as a system role)
    /// but is NOT rendered in the chat bubble.  Used for internal
    /// control prompts like the auto-continue nudge — the user
    /// shouldn't see "auto-continue: previous tool succeeded…".
    HiddenSystem(String),
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
    /// Live planner/worker workflow state.  Loaded on startup from
    /// `~/.larql/workflows.json`, mutated by `plan` + `status` blocks
    /// extracted from each model response, rendered in the sidebar.
    workflows: WorkflowStore,
    /// User intent: should the plans surface be available?  Toggled
    /// with Ctrl+B.  Even when true, narrow terminals downgrade
    /// from side-by-side to tabbed layout.
    sidebar_visible: bool,
    /// In tabbed layout (narrow terminal), which view is foregrounded.
    active_tab: ActiveTab,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ActiveTab {
    Chat,
    Plans,
}

/// Layout decision recomputed on every frame from the current
/// terminal size.  Centralising this makes the resize behaviour
/// trivial: the next draw picks the right mode for the new size.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum LayoutMode {
    /// Wide terminal: chat on the left, sidebar of width `sidebar_w`
    /// on the right.
    SideBySide { sidebar_w: u16 },
    /// Narrow terminal but plans exist: tabbed view, only one panel
    /// rendered at a time, tab bar at the bottom.
    Tabs,
    /// No plans (or plans hidden via Ctrl+B and terminal narrow):
    /// chat-only, no tab bar.
    ChatOnly,
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
            // Workflows load after session_id is set in main; start
            // empty so the sidebar doesn't briefly show stale plans
            // from another session before we know which file to read.
            workflows: WorkflowStore::default(),
            sidebar_visible: true,
            active_tab: ActiveTab::Chat,
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
    // Tokenise skill_args with full shell-word semantics so quoted
    // strings, heredoc-ish payloads, and embedded whitespace survive
    // intact.  `split_whitespace` would tear `python3 -c "x  y"` into
    // 4 tokens, destroying every multi-space indent inside the
    // quoted source.  shell_words fails only on truly malformed
    // input (unclosed quote) — fall back to whitespace split there.
    let argv = shell_words::split(&skill_args)
        .unwrap_or_else(|_| skill_args.split_whitespace().map(str::to_string).collect());
    let mut command = if runtime == "container" {
        let _ = ensure_runtime_container();
        let mut c = std::process::Command::new("docker");
        c.args(["exec", "-i", "larql-skill-runtime", "bash"]);
        c.arg(&path);
        c.args(&argv);
        c
    } else {
        let mut c = std::process::Command::new("bash");
        c.arg(&path);
        c.args(&argv);
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
/// Path of the on-disk workflow store, namespaced by session.  Each
/// session has its own plans — starting a fresh conversation gets
/// you an empty sidebar without affecting any other session, and
/// resuming a session brings back its plans.
fn workflows_path(session_id: Option<&str>) -> std::path::PathBuf {
    match session_id {
        Some(sid) => {
            let safe: String = sid
                .chars()
                .map(|c| if c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '.') { c } else { '_' })
                .collect();
            home_dir()
                .join(".larql/sessions")
                .join(format!("{safe}.workflows.json"))
        }
        // Default (no session set yet) falls back to a global file
        // so the field is never empty in tests / pre-init draws.
        None => home_dir().join(".larql/workflows.json"),
    }
}

/// Pull plan/status blocks from `response`, apply to `store`, persist
/// to the per-session path.  Returns true when the sidebar needs a redraw.
fn apply_workflow_annotations(
    store: &mut WorkflowStore,
    response: &str,
    session_id: Option<&str>,
) -> bool {
    let mut changed = false;
    for wf in extract_plan_blocks(response) {
        store.upsert(wf);
        changed = true;
    }
    for upd in extract_status_updates(response) {
        if store.apply_status(&upd) {
            changed = true;
        }
    }
    if changed {
        let _ = store.save(&workflows_path(session_id));
    }
    changed
}

/// Why we want to auto-continue.  Drives the prompt the loop
/// injects so the model knows what's expected.
#[derive(Clone, Copy, PartialEq, Eq)]
enum ContinueReason {
    /// Workflow has pending/active steps — push the model to
    /// emit the next ```tool``` block.
    PendingSteps,
    /// Last response was meta-blocks only (```fact```, ```status```,
    /// ```plan``` — all hidden from the user).  User saw nothing;
    /// ask the model to wrap up with a real prose summary.
    NoVisibleResponse,
}

/// Decide whether the agent loop should re-spawn chat after a model
/// turn that didn't emit a tool.  We continue when:
///  * we haven't exhausted the chain budget,
///  * AND either:
///     - there's an active workflow with non-done steps (fires even
///       at tool_depth == 0 — catches the common "model emits plan
///       then says 'let's do step 1' without emitting the tool"
///       stall that used to need a manual nudge), OR
///     - we're inside an active agent loop AND the model's last
///       response had nothing visible to the user (only meta-blocks),
///       so they need a closing summary.
fn should_auto_continue(
    state: &AppState,
    tool_depth: usize,
    max_depth: usize,
) -> Option<ContinueReason> {
    if tool_depth >= max_depth {
        return None;
    }
    let has_pending = state.workflows.workflows.iter().any(|w| {
        w.state == WorkflowState::Active
            && w.steps
                .iter()
                .any(|s| matches!(s.state, StepState::Pending | StepState::Active))
    });
    if has_pending {
        return Some(ContinueReason::PendingSteps);
    }
    // NoVisibleResponse: only while mid-loop.  Otherwise a plain
    // chat turn ("hi there") would loop forever asking for a
    // summary of nothing.
    if tool_depth > 0 {
        let last_assistant: Option<&str> = state.messages.iter().rev().find_map(|m| {
            if let Message::Assistant(t) = m { (!t.is_empty()).then_some(t.as_str()) } else { None }
        });
        if let Some(text) = last_assistant {
            if strip_meta_blocks(text).trim().is_empty() {
                return Some(ContinueReason::NoVisibleResponse);
            }
        }
    }
    None
}

/// The step the just-run tool likely closed: only returns a step
/// currently marked `active` (not merely pending) so we can safely
/// distinguish it from the NEXT step.  Returns None when nothing is
/// explicitly active — in that case we skip the "close previous"
/// reminder because we can't be sure which step the tool addressed.
fn previously_active_step(state: &AppState) -> Option<String> {
    let mut active: Vec<&Workflow> = state
        .workflows
        .workflows
        .iter()
        .filter(|w| w.state == WorkflowState::Active)
        .collect();
    active.sort_by_key(|w| std::cmp::Reverse(w.ts));
    for w in active {
        if let Some(s) = w.steps.iter().find(|s| s.state == StepState::Active) {
            return Some(s.description.clone());
        }
    }
    None
}

/// Description of the next non-done step in the most recently
/// updated active workflow.  Used in the auto-continue prompt so
/// the model knows what to do next.
fn next_pending_step(state: &AppState) -> Option<String> {
    let mut active: Vec<&Workflow> = state
        .workflows
        .workflows
        .iter()
        .filter(|w| w.state == WorkflowState::Active)
        .collect();
    active.sort_by_key(|w| std::cmp::Reverse(w.ts));
    for w in active {
        for s in &w.steps {
            if !matches!(s.state, StepState::Done | StepState::Failed) {
                return Some(s.description.clone());
            }
        }
    }
    None
}

/// True when the last assistant message, after stripping meta
/// blocks, has nothing the user can see.  Used by the fallback
/// summariser as the "rescue" trigger.
fn last_assistant_is_hidden(state: &AppState) -> bool {
    let last = state.messages.iter().rev().find_map(|m| {
        if let Message::Assistant(t) = m { (!t.is_empty()).then_some(t.as_str()) } else { None }
    });
    match last {
        Some(t) => strip_meta_blocks(t).trim().is_empty(),
        None => false,
    }
}

/// Collect the latest tool invocations + their one-line outcomes so
/// the auto-continue wrap-up prompt has concrete material to
/// summarise, and the fallback synthesiser has something to show.
/// Walks backward until it hits a User message (current turn only).
fn tool_results_recap(state: &AppState) -> String {
    let mut pairs: Vec<(String, String)> = Vec::new(); // (tool, first-line of result)
    let mut pending_tool: Option<String> = None;
    for m in state.messages.iter().rev() {
        match m {
            Message::User(_) => break,
            Message::ToolResult { summary } => {
                // First non-empty, non-italic line is the concrete output.
                let head = summary
                    .lines()
                    .map(|l| l.trim())
                    .find(|l| !l.is_empty() && !l.starts_with('*'))
                    .unwrap_or("")
                    .to_string();
                pending_tool = Some(head);
            }
            Message::ToolUse { tool, detail } => {
                let outcome = pending_tool.take().unwrap_or_else(|| "(no output)".into());
                pairs.push((format!("{tool} {detail}").trim().to_string(), outcome));
            }
            _ => {}
        }
    }
    pairs.reverse();
    if pairs.is_empty() {
        return String::new();
    }
    pairs
        .into_iter()
        .map(|(cmd, out)| format!("- `{cmd}` → {out}"))
        .collect::<Vec<_>>()
        .join("\n")
}

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

/// Hide ```fact / ```status / ```plan / ```tool fenced blocks from
/// `text` — they're consumed by the annotation, workflow and tool
/// pipelines respectively, and shouldn't render as raw code in the
/// chat bubble.  Partial blocks (still streaming, no close fence
/// yet) are also hidden so we don't flicker raw markdown while
/// generating.
fn strip_meta_blocks(text: &str) -> String {
    const HIDDEN: &[&str] = &["fact", "status", "plan", "tool"];
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0usize;
    while cursor < text.len() {
        // Find the earliest opening fence we want to hide.
        let mut next: Option<(usize, usize)> = None; // (open_start, header_end)
        for &b in HIDDEN {
            let needle = format!("```{b}");
            if let Some(pos) = text[cursor..].find(&needle) {
                let abs = cursor + pos;
                let header_end = abs + needle.len();
                if next.map_or(true, |(p, _)| abs < p) {
                    next = Some((abs, header_end));
                }
            }
        }
        match next {
            None => {
                out.push_str(&text[cursor..]);
                break;
            }
            Some((open_start, header_end)) => {
                out.push_str(&text[cursor..open_start]);
                // Skip past the header line (```fact[\n] or trailing tag).
                let after = &text[header_end..];
                let nl = match after.find('\n') {
                    Some(n) => n + 1,
                    None => break,
                };
                let body_start = header_end + nl;
                // Find the closing fence; if absent we're still
                // streaming — hide everything from open_start onward.
                let close = match text[body_start..].find("```") {
                    Some(c) => c,
                    None => break,
                };
                cursor = body_start + close + 3;
                // Eat one trailing newline so we don't leave a gap.
                if cursor < text.len() && text.as_bytes()[cursor] == b'\n' {
                    cursor += 1;
                }
            }
        }
    }
    // Collapse runs of blank lines to at most one.
    let mut cleaned = String::with_capacity(out.len());
    let mut prev_blank = false;
    for line in out.lines() {
        let blank = line.trim().is_empty();
        if blank && prev_blank {
            continue;
        }
        cleaned.push_str(line);
        cleaned.push('\n');
        prev_blank = blank;
    }
    cleaned.trim_end_matches('\n').to_string()
}

/// Find the body of a fenced code block tagged ```<lang>.  Walks
/// line by line and tracks fence depth so a NESTED ``` inside the
/// block (e.g. a ```terminal subblock inside a ```summary) doesn't
/// silently close the outer fence — that bug previously truncated
/// the run-skill summary at "(no output)" and the model never
/// received the hint about how to actually run the file.
fn extract_block(text: &str, lang: &str) -> Option<String> {
    let open_tag = format!("```{lang}");
    let start = text.find(&open_tag)?;
    // Skip past the opening fence header line.
    let after_tag = &text[start + open_tag.len()..];
    let nl = after_tag.find('\n')?;
    let body_start = start + open_tag.len() + nl + 1;
    // Walk line by line; ``` opens or closes a level.  Start at
    // depth 1 (we're inside the outer block).  When depth returns
    // to 0 the matching close is found.
    let mut depth: usize = 1;
    let mut idx = body_start;
    let bytes = text.as_bytes();
    while idx < bytes.len() {
        let line_end = text[idx..]
            .find('\n')
            .map(|p| idx + p)
            .unwrap_or(bytes.len());
        let line = text[idx..line_end].trim_start();
        if line.starts_with("```") {
            // Pure ``` (possibly trailing whitespace) closes; ```<tag>
            // opens a sub-block.
            let after = &line[3..].trim();
            if after.is_empty() {
                depth -= 1;
                if depth == 0 {
                    return Some(text[body_start..idx].trim_end().to_string());
                }
            } else {
                depth += 1;
            }
        }
        idx = line_end + 1;
    }
    // Unclosed — return whatever we have rather than None so the
    // caller still surfaces something to the user.
    Some(text[body_start..].trim_end().to_string())
}

// ── Drawing ──────────────────────────────────────────────────────────────

/// Minimum widths for the side-by-side layout.  Below this we
/// downgrade to tabs so neither panel is unreadable.
const CHAT_MIN_WIDTH: u16 = 40;
const SIDEBAR_MIN_WIDTH: u16 = 28;

/// Decide the layout for the current frame purely from the current
/// terminal area + state.  Pure function — easy to test, called on
/// every draw so resize is automatic.
fn compute_layout(state: &AppState, area: Rect) -> LayoutMode {
    if state.workflows.workflows.is_empty() {
        return LayoutMode::ChatOnly;
    }
    if !state.sidebar_visible {
        // User explicitly hid the sidebar.  Stay full-screen chat.
        return LayoutMode::ChatOnly;
    }
    let want_sidebar = sidebar_target_width(state, area.width);
    if want_sidebar >= SIDEBAR_MIN_WIDTH
        && area.width >= CHAT_MIN_WIDTH + want_sidebar
    {
        LayoutMode::SideBySide { sidebar_w: want_sidebar }
    } else {
        LayoutMode::Tabs
    }
}

fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, state: &AppState) {
    terminal.draw(|f| {
        let area = f.area();
        let mode = compute_layout(state, area);
        match mode {
            LayoutMode::SideBySide { sidebar_w } => {
                let cols = Layout::default()
                    .direction(Direction::Horizontal)
                    .constraints([Constraint::Min(CHAT_MIN_WIDTH), Constraint::Length(sidebar_w)])
                    .split(area);
                draw_chat_panel(f, state, cols[0], mode);
                draw_sidebar(f, state, cols[1]);
            }
            LayoutMode::Tabs => match state.active_tab {
                ActiveTab::Chat => draw_chat_panel(f, state, area, mode),
                ActiveTab::Plans => draw_plans_panel(f, state, area),
            },
            LayoutMode::ChatOnly => {
                draw_chat_panel(f, state, area, mode);
            }
        }
    }).ok();
}

/// Draw the chat panel (messages + input + status) into `area`.
/// In Tabs mode the status line is replaced by a tab bar.
fn draw_chat_panel(f: &mut ratatui::Frame, state: &AppState, area: Rect, mode: LayoutMode) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(5),
            Constraint::Length(3),
            Constraint::Length(1),
        ])
        .split(area);

    draw_messages(f, state, chunks[0]);
    draw_input(f, state, chunks[1]);
    if matches!(mode, LayoutMode::Tabs) {
        draw_tab_bar(f, state, chunks[2]);
    } else {
        draw_status(f, state, chunks[2]);
    }
}

/// Plans-as-full-panel: same content as the sidebar but rendered in
/// the whole window.  Used in Tabs mode when the user presses `2`.
fn draw_plans_panel(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(5), Constraint::Length(1)])
        .split(area);
    draw_sidebar(f, state, chunks[0]);
    draw_tab_bar(f, state, chunks[1]);
}

/// Render the bottom tab bar shown only in Tabs (narrow) layout.
/// Number keys 1/2 switch tabs.  The active tab is highlighted.
fn draw_tab_bar(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let active = state.active_tab;
    let make = |label: &str, this: ActiveTab| {
        let is_active = this == active;
        let style = if is_active {
            Style::default().fg(Color::Black).bg(Color::Cyan).add_modifier(Modifier::BOLD)
        } else {
            Style::default().fg(Color::White).bg(Color::DarkGray)
        };
        Span::styled(format!(" {label} "), style)
    };
    let plan_count = state.workflows.workflows.len();
    let plans_label = if plan_count > 0 {
        format!("2 plans [{plan_count}]")
    } else {
        "2 plans".to_string()
    };
    let line = Line::from(vec![
        make("1 chat", ActiveTab::Chat),
        Span::raw(" "),
        make(&plans_label, ActiveTab::Plans),
        Span::styled("  ·  press 1 / 2 to switch  ·  Ctrl-C to quit",
            Style::default().fg(Color::DarkGray)),
    ]);
    let para = Paragraph::new(line).style(Style::default().bg(Color::Reset));
    f.render_widget(para, area);
}

/// Pick a sidebar width that fits the longest label without wrapping
/// when the terminal is wide enough.  Clamps to [28, 45% of terminal
/// width] so the chat panel always has room.  Returns 0 when we
/// shouldn't render the sidebar at all (no workflows or terminal too
/// narrow).
fn sidebar_target_width(state: &AppState, terminal_width: u16) -> u16 {
    use unicode_width::UnicodeWidthStr;
    if state.workflows.workflows.is_empty() {
        return 0;
    }
    // Step prefix: "├─● ⚡ N. " — 9 cells.  Borders add 2.  Output
    // continuation prefix is "│   ↪ " — 6 cells.  Use the bigger.
    const PREFIX: usize = 11;
    let longest_label = state
        .workflows
        .workflows
        .iter()
        .flat_map(|w| {
            let head = w.name.as_str();
            let head_width = UnicodeWidthStr::width(head)
                + format!("  [{}/{}]", w.steps.len(), w.steps.len()).len();
            std::iter::once(head_width)
                .chain(w.steps.iter().map(|s| UnicodeWidthStr::width(s.description.as_str()) + 4))
                .chain(
                    w.steps
                        .iter()
                        .filter_map(|s| s.output.as_deref())
                        .map(|o| UnicodeWidthStr::width(o) + 6),
                )
        })
        .max()
        .unwrap_or(20);
    let cap = (terminal_width as usize * 45 / 100).max(28);
    let target = (longest_label + PREFIX).clamp(28, cap);
    target as u16
}

/// Render the workflow store as a gitk-style branch graph: each
/// workflow is its own column of dots-on-a-pipe (`●` nodes connected
/// by `│` segments), workflows stack vertically.  Step state
/// determines the dot colour + the trailing icon (✓ ⚡ ⏳ ✗).
///
/// Long labels are word-wrapped (no mid-word breaks, no ellipsis);
/// continuation rows render the workflow's pipe so the column stays
/// visually connected.
fn draw_sidebar(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let mut lines: Vec<Line<'static>> = Vec::new();

    let dim = Style::default().fg(Color::DarkGray);
    let head = Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD);
    let muted = Style::default().fg(Color::Gray);
    let white = Style::default().fg(Color::White);

    let inner_w = area.width.saturating_sub(2) as usize;
    if inner_w == 0 {
        return;
    }

    // Newest workflows on top so the active stuff is always visible.
    let mut wfs: Vec<&Workflow> = state.workflows.workflows.iter().collect();
    wfs.sort_by_key(|w| std::cmp::Reverse(w.ts));

    for (idx, wf) in wfs.iter().enumerate() {
        if idx > 0 {
            // Gap between workflows — gitk-ish divergence marker.
            lines.push(Line::from(Span::styled("│", dim)));
            lines.push(Line::from(Span::styled("◇", dim)));
            lines.push(Line::from(Span::styled("│", dim)));
        }

        // Workflow head row: bold dot + name + [done/total].  Wrap the
        // name itself if it's too long, with continuation rows
        // indented under the name.
        let (head_dot_style, head_state_label, head_state_style) = match wf.state {
            WorkflowState::Active => ("●", "active", Style::default().fg(Color::Yellow)),
            WorkflowState::Done => ("●", "done", Style::default().fg(Color::Green)),
            WorkflowState::Cancelled => ("●", "cancelled", Style::default().fg(Color::Red)),
        };
        let done_count = wf.steps.iter().filter(|s| s.state == StepState::Done).count();
        let total = wf.steps.len();
        let count_str = format!("  [{done_count}/{total}]");
        let head_prefix_w = 2; // "● "
        let count_w = unicode_width::UnicodeWidthStr::width(count_str.as_str());
        let name_budget = inner_w.saturating_sub(head_prefix_w + count_w).max(1);
        let name_rows = wrap_words(&wf.name, name_budget);
        for (i, row) in name_rows.iter().enumerate() {
            let mut spans: Vec<Span<'static>> = Vec::new();
            if i == 0 {
                spans.push(Span::styled(format!("{head_dot_style} "), head_state_style));
            } else {
                spans.push(Span::styled("  ".to_string(), dim));
            }
            spans.push(Span::styled(row.clone(), head));
            if i + 1 == name_rows.len() {
                spans.push(Span::styled(count_str.clone(), muted));
            }
            lines.push(Line::from(spans));
        }
        lines.push(Line::from(vec![
            Span::styled("│ ".to_string(), dim),
            Span::styled(head_state_label.to_string(), head_state_style),
        ]));

        // Each step: a dot on the workflow's pipe + numbered label,
        // word-wrapped onto continuation rows under the description.
        for (i, step) in wf.steps.iter().enumerate() {
            let (icon, dot_style) = match step.state {
                StepState::Done => ("✓", Style::default().fg(Color::Green)),
                StepState::Active => ("⚡", Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD)),
                StepState::Pending => ("⏳", Style::default().fg(Color::DarkGray)),
                StepState::Failed => ("✗", Style::default().fg(Color::Red)),
            };
            let n = i + 1;
            // Prefix layout: "├─● <icon> N. <text>"
            //                 ^^   ^^^^^ ^^^
            //                  2     4    3-4 (depending on N digits)
            let n_str = format!("{n}. ");
            let prefix_w = 2 + 4 + unicode_width::UnicodeWidthStr::width(n_str.as_str());
            let cont_indent = " ".repeat(prefix_w.saturating_sub(2));
            let text_budget = inner_w.saturating_sub(prefix_w).max(1);
            let rows = wrap_words(&step.description, text_budget);
            for (j, row) in rows.iter().enumerate() {
                let mut spans: Vec<Span<'static>> = Vec::new();
                if j == 0 {
                    spans.push(Span::styled("├─".to_string(), dim));
                    spans.push(Span::styled(format!("● {icon} "), dot_style));
                    spans.push(Span::styled(n_str.clone(), muted));
                } else {
                    spans.push(Span::styled("│ ".to_string(), dim));
                    spans.push(Span::styled(cont_indent.clone(), dim));
                }
                spans.push(Span::styled(row.clone(), white));
                lines.push(Line::from(spans));
            }
            if let Some(out) = &step.output {
                let out_prefix_w: usize = 6; // "│   ↪ "
                let out_cont_indent = " ".repeat(out_prefix_w.saturating_sub(2));
                let out_budget = inner_w.saturating_sub(out_prefix_w).max(1);
                let out_rows = wrap_words(out, out_budget);
                for (j, row) in out_rows.iter().enumerate() {
                    let mut spans: Vec<Span<'static>> = Vec::new();
                    if j == 0 {
                        spans.push(Span::styled("│   ".to_string(), dim));
                        spans.push(Span::styled("↪ ".to_string(), muted));
                    } else {
                        spans.push(Span::styled("│ ".to_string(), dim));
                        spans.push(Span::styled(out_cont_indent.clone(), dim));
                    }
                    spans.push(Span::styled(row.clone(), dim));
                    lines.push(Line::from(spans));
                }
            }
        }
    }

    let title_text = format!(" plans · {} active ", state.workflows.workflows.iter()
        .filter(|w| w.state == WorkflowState::Active).count());
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(title_text, head));

    let para = Paragraph::new(lines).block(block);
    f.render_widget(para, area);
}

/// Greedy word-wrap: split `text` into rows of at most `width`
/// terminal cells, breaking only on whitespace.  Words that exceed
/// `width` on their own (rare) fall back to a hard char split.
fn wrap_words(text: &str, width: usize) -> Vec<String> {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    if width == 0 {
        return vec![text.to_string()];
    }
    let mut rows: Vec<String> = Vec::new();
    let mut cur = String::new();
    let mut cur_w = 0usize;
    for word in text.split_whitespace() {
        let w_w = UnicodeWidthStr::width(word);
        if cur.is_empty() {
            if w_w <= width {
                cur.push_str(word);
                cur_w = w_w;
            } else {
                // Hard-split a too-long word into multiple rows.
                let mut chunk = String::new();
                let mut chunk_w = 0usize;
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if chunk_w + cw > width && !chunk.is_empty() {
                        rows.push(std::mem::take(&mut chunk));
                        chunk_w = 0;
                    }
                    chunk.push(ch);
                    chunk_w += cw;
                }
                cur = chunk;
                cur_w = chunk_w;
            }
        } else if cur_w + 1 + w_w <= width {
            cur.push(' ');
            cur.push_str(word);
            cur_w += 1 + w_w;
        } else {
            rows.push(std::mem::take(&mut cur));
            cur_w = 0;
            if w_w <= width {
                cur.push_str(word);
                cur_w = w_w;
            } else {
                let mut chunk = String::new();
                let mut chunk_w = 0usize;
                for ch in word.chars() {
                    let cw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if chunk_w + cw > width && !chunk.is_empty() {
                        rows.push(std::mem::take(&mut chunk));
                        chunk_w = 0;
                    }
                    chunk.push(ch);
                    chunk_w += cw;
                }
                cur = chunk;
                cur_w = chunk_w;
            }
        }
    }
    if !cur.is_empty() {
        rows.push(cur);
    }
    if rows.is_empty() {
        rows.push(String::new());
    }
    rows
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
                // The annotate skill teaches the model to emit
                // ```fact```, ```status```, ```plan``` and ```tool```
                // blocks for downstream processing; those are NOT for
                // human reading.  Strip them before rendering so the
                // chat bubble shows only prose.  Tool execution surfaces
                // separately as Message::ToolUse + Message::ToolResult.
                let cleaned = strip_meta_blocks(text);
                lines.extend(gc_markdown::render_markdown(&cleaned, gc_markdown::Theme::Dark));
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
            // Hidden system messages are control-channel only — they
            // go to the model when building the next chat request,
            // but the user shouldn't see prompts like
            // "auto-continue: previous tool succeeded…".
            Message::HiddenSystem(_) => {}
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
        Message::HiddenSystem(_) => Vec::new(),
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
    let plans_hint = if state.workflows.workflows.is_empty() {
        ""
    } else if state.sidebar_visible {
        "  ·  Ctrl-B: hide plans"
    } else {
        "  ·  Ctrl-B: show plans"
    };
    let copy_hint = "  ·  Ctrl-Y: copy mode";
    let status = format!(" {}{}{}  ", state.status, plans_hint, copy_hint);
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

        // Plan/status blocks: update the live workflow store so the
        // sidebar reflects the model's planning + per-step progress.
        apply_workflow_annotations(&mut state.workflows, &response_text, state.session_id.as_deref());

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
            Message::HiddenSystem(text) if !text.is_empty() => {
                // Internal control prompts (auto-continue nudges) —
                // hidden from the user but sent to the model as a
                // system role so they actually steer behaviour.
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
    //   * default                  → derive id from cwd; resume if it exists.
    //   * --new (alias --new-session) → derive id from cwd, but wipe any
    //                                prior log AND clear the live KNN
    //                                override store so this run starts
    //                                with no inherited overrides.
    //   * --session <id>           → use the explicit id, resume if exists.
    //   * --session <id> --new     → use explicit id but wipe first.
    let args: Vec<String> = std::env::args().collect();
    let explicit_session = args
        .iter()
        .position(|a| a == "--session")
        .and_then(|i| args.get(i + 1))
        .cloned();
    let new_session = args.iter().any(|a| a == "--new" || a == "--new-session");
    let headless = args.iter().any(|a| a == "--headless");

    let session_id = explicit_session.unwrap_or_else(default_session_id_from_cwd);

    if new_session {
        // Best-effort wipe of the existing session log so this
        // conversation starts clean.  Non-fatal if it doesn't exist.
        let client = reqwest::Client::new();
        let _ = client
            .delete(format!("{server_url}/v1/sessions/{session_id}"))
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
        // Also clear the in-memory KNN override store so polluted
        // entries from prior sessions don't intercept new prompts.
        // Server-wide, which matches user intent for --new.
        let _ = client
            .post(format!("{server_url}/v1/reset"))
            .timeout(std::time::Duration::from_secs(3))
            .send()
            .await;
        // `--new` on the cwd-derived session still resolves to the
        // SAME session_id as prior runs (cwd is unchanged), so the
        // per-session workflows file needs to be wiped to match the
        // user's "fresh slate" intent.  Other sessions' workflow
        // files are untouched — namespacing still applies.
        let _ = std::fs::remove_file(workflows_path(Some(&session_id)));
    }

    if headless {
        return run_headless(&server_url, Some(&session_id)).await;
    }

    // Install a panic hook that restores the terminal BEFORE the
    // panic message is printed.  Without this, an unexpected panic
    // (e.g. tokio block_on misuse) leaves the terminal in raw mode
    // with mouse-tracking enabled — the parent shell then prints raw
    // SGR mouse reports ("35;63;13M…") on every mouse move.
    let original_hook = std::panic::take_hook();
    std::panic::set_hook(Box::new(move |info| {
        let _ = disable_raw_mode();
        let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
        original_hook(info);
    }));

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    // Mouse capture starts disabled so native click-and-drag text
    // selection works.  Belt-and-suspenders: explicitly send a
    // DisableMouseCapture FIRST in case a prior crash left the
    // terminal stuck in mouse-reporting mode (no-op when already
    // disabled).  Ctrl-T toggles capture on for scroll-wheel users.
    execute!(stdout, EnterAlternateScreen, DisableMouseCapture)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new(&server_url);
    state.session_id = Some(session_id.clone());
    // Now that we know the session, load its per-session workflow
    // store.  New session id → file doesn't exist → empty sidebar.
    state.workflows = WorkflowStore::load(&workflows_path(Some(&session_id)));
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
    // Two separate caps:
    //   * tool_depth: how many ACTUAL tool runs this user turn (cap
    //     prevents infinite write/read loops).
    //   * continue_depth: how many consecutive auto-continue prompts
    //     without the model emitting a tool.  Reset to 0 every time
    //     a tool actually runs so a long "tool → nudge → tool → nudge"
    //     dance has room, but pure "nudge → nudge → nudge" can't
    //     spin forever.
    let mut tool_depth: usize = 0;
    let mut continue_depth: usize = 0;

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
                        // Interactive loop runs inside the tokio runtime,
                        // so calling Handle::current().block_on() here
                        // panics ("Cannot start a runtime from within a
                        // runtime").  Just spawn — the insert is
                        // fire-and-forget and the runtime stays alive
                        // for the whole chat session.
                        let url = state.server_url.clone();
                        let prompt = latest_user.clone();
                        let resp = response_text.clone();
                        tokio::spawn(async move {
                            ingest_facts_to_knn(&url, &prompt, &resp).await;
                        });
                    }
                    // Plan/status: same sweep, sidebar reflects new state.
                    apply_workflow_annotations(&mut state.workflows, &response_text, state.session_id.as_deref());
                    // Long-session summariser tick.
                    if let Some(sid) = state.session_id.as_deref() {
                        let turn_count = state
                            .messages
                            .iter()
                            .filter(|m| matches!(m, Message::User(_) | Message::Assistant(_)))
                            .count();
                        maybe_spawn_summarizer(&state.server_url, sid, turn_count);
                    }

                    // Allow a small chain of tool calls per user turn:
                    // model emits tool → result → follow-up → maybe
                    // another tool → result → ... up to MAX_TOOL_DEPTH.
                    // Separately cap CONSECUTIVE auto-continues so a
                    // final prose wrap-up always has room even after
                    // a long chain.
                    const MAX_TOOL_DEPTH: usize = 6;
                    const MAX_CONSEC_CONTINUES: usize = 2;
                    if tool_depth < MAX_TOOL_DEPTH {
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
                        // Push a "(running…)" placeholder so slow
                        // tools have visible feedback BEFORE the bash
                        // call returns.  Remember its index so we can
                        // delete it once execute_skill_tool appends
                        // the real ToolUse line — otherwise both the
                        // spinner and the final marker render.
                        let placeholder_idx = preview_tool_call(&response_text)
                            .map(|(preview_name, preview_detail)| {
                                let idx = state.messages.len();
                                state.messages.push(Message::ToolUse {
                                    tool: format!("{preview_name} (running…)"),
                                    detail: preview_detail,
                                });
                                draw(&mut terminal, &state);
                                idx
                            });
                        let exec_result = execute_skill_tool(&response_text, &mut state.messages, &server_url_cl, sid.as_deref(), &state.skills);
                        if let Some(idx) = placeholder_idx {
                            // Remove the spinner row only if it's still
                            // a "(running…)" ToolUse at that index.
                            if idx < state.messages.len() {
                                let is_placeholder = matches!(
                                    &state.messages[idx],
                                    Message::ToolUse { tool, .. } if tool.ends_with(" (running…)")
                                );
                                if is_placeholder {
                                    state.messages.remove(idx);
                                }
                            }
                        }
                        if let Some(_summary) = exec_result {
                            tool_depth += 1;
                            continue_depth = 0; // real tool ran, reset nudge counter
                            state.messages.push(Message::Assistant(String::new()));
                            let chat_msgs = build_chat_messages_with_system(&state);
                            spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone(), state.session_id.clone());
                        } else if let Some(reason) = (continue_depth < MAX_CONSEC_CONTINUES)
                            .then(|| should_auto_continue(&state, tool_depth, MAX_TOOL_DEPTH))
                            .flatten()
                        {
                            // Model finished without emitting a tool.
                            // Either it stalled mid-plan ("let me try X"
                            // with no tool block) — push to continue —
                            // or its whole response was meta-blocks
                            // (fact/status hidden from the user) — push
                            // for a real prose summary.  Counts toward
                            // the continue budget, not the tool budget.
                            continue_depth += 1;
                            let prompt = match reason {
                                ContinueReason::PendingSteps => {
                                    let next = next_pending_step(&state)
                                        .unwrap_or_else(|| "the next step".to_string());
                                    let prev = previously_active_step(&state);
                                    let prev_line = match prev {
                                        Some(p) => format!(
                                            "If your previous tool completed step \"{p}\" \
                                             of the plan, emit a ```status``` block \
                                             marking it done BEFORE the next tool. "
                                        ),
                                        None => String::new(),
                                    };
                                    format!(
                                        "auto-continue: previous tool succeeded. {prev_line}\
                                         Then continue your active plan — emit the \
                                         next ```tool``` block now (next step: {next})."
                                    )
                                }
                                ContinueReason::NoVisibleResponse => {
                                    // Include the concrete tool results so the
                                    // model has something to summarise.  Then
                                    // forbid any meta-block so the reply can't
                                    // come back hidden again.
                                    let recap = tool_results_recap(&state);
                                    format!(
                                        "auto-continue: the task is done.  Final results:\n\
                                         {recap}\n\n\
                                         Now write ONE or TWO sentences in plain English \
                                         for the user, describing what you did and the \
                                         result.  DO NOT emit any ```fact```, ```status```, \
                                         ```plan```, or ```tool``` blocks — they are \
                                         hidden from the user.  Plain prose only."
                                    )
                                }
                            };
                            state.messages.push(Message::HiddenSystem(prompt));
                            state.messages.push(Message::Assistant(String::new()));
                            let chat_msgs = build_chat_messages_with_system(&state);
                            spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone(), state.session_id.clone());
                        } else {
                            // No more auto-continues allowed AND the last
                            // reply may still be all-meta (user saw
                            // nothing).  Synthesise a minimal summary
                            // from tool results as a last-resort so the
                            // turn doesn't end in silence.
                            if last_assistant_is_hidden(&state) {
                                let recap = tool_results_recap(&state);
                                if !recap.trim().is_empty() {
                                    // Replace the empty/meta-only reply
                                    // with a visible synthesized summary.
                                    if let Some(Message::Assistant(t)) = state.messages.last_mut() {
                                        *t = format!(
                                            "Done. Here's what happened:\n\n{recap}"
                                        );
                                    } else {
                                        state.messages.push(Message::Assistant(format!(
                                            "Done. Here's what happened:\n\n{recap}"
                                        )));
                                    }
                                }
                            }
                            state.is_generating = false;
                            tool_depth = 0;
                            continue_depth = 0;
                        }
                    } else {
                        state.is_generating = false;
                        tool_depth = 0;
                        continue_depth = 0;
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
            // ── Resize ── recompute layout immediately so the
            // sidebar re-flows and chat re-wraps for the new size.
            if let CEvent::Resize(_, _) = evt {
                draw(&mut terminal, &state);
                continue;
            }
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
                    KeyCode::Char('y') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        // Copy mode: drop the alt screen entirely so
                        // the chat transcript ends up in normal
                        // terminal scrollback where click-and-drag +
                        // ⌘C just works.  Returns to the chat on
                        // Enter.  Some terminals stubbornly block
                        // selection inside the alt-screen even
                        // without app-side mouse capture; this is
                        // the universal escape hatch.
                        let _ = execute!(io::stdout(), DisableMouseCapture, LeaveAlternateScreen);
                        let _ = disable_raw_mode();
                        // Print transcript to scrollback in plain
                        // text (no ratatui styling — copy-friendly).
                        println!();
                        println!("─── COPY MODE ─── select with mouse + ⌘C, then press Enter to return ───");
                        println!();
                        for m in &state.messages {
                            match m {
                                Message::User(t) => println!("> {t}\n"),
                                Message::Assistant(t) => {
                                    let cleaned = strip_meta_blocks(t);
                                    if !cleaned.trim().is_empty() {
                                        println!("{cleaned}\n");
                                    }
                                }
                                Message::ToolUse { tool, detail } => {
                                    println!("⚡ {tool} {detail}");
                                }
                                Message::ToolResult { summary } => {
                                    println!("{summary}\n");
                                }
                                Message::ToolRender { content } => {
                                    println!("{content}\n");
                                }
                                Message::System(t) => println!("({t})"),
                                Message::HiddenSystem(_) => {}
                            }
                        }
                        println!();
                        println!("─── press Enter to return to chat ───");
                        let mut _line = String::new();
                        let _ = std::io::stdin().read_line(&mut _line);
                        // Restore TUI state.
                        let _ = enable_raw_mode();
                        let mut so = io::stdout();
                        let _ = execute!(so, EnterAlternateScreen);
                        let _ = terminal.clear();
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.sidebar_visible = !state.sidebar_visible;
                        // If the user re-shows the sidebar in narrow
                        // mode, default-foreground the plans tab so
                        // they actually see something.
                        if state.sidebar_visible {
                            let area = terminal.size().unwrap_or(ratatui::layout::Size::new(80, 24));
                            let r = Rect { x: 0, y: 0, width: area.width, height: area.height };
                            if matches!(compute_layout(&state, r), LayoutMode::Tabs) {
                                state.active_tab = ActiveTab::Plans;
                            }
                        }
                        draw(&mut terminal, &state);
                    }
                    // Number keys switch tabs when (a) we're in tabs
                    // layout and (b) input is empty so we don't steal
                    // a literal "1" the user is typing.
                    KeyCode::Char('1') | KeyCode::Char('2')
                        if !state.is_generating && state.input.is_empty() =>
                    {
                        let area = terminal.size().unwrap_or(ratatui::layout::Size::new(80, 24));
                        let r = Rect { x: 0, y: 0, width: area.width, height: area.height };
                        if matches!(compute_layout(&state, r), LayoutMode::Tabs) {
                            state.active_tab = if matches!(key.code, KeyCode::Char('1')) {
                                ActiveTab::Chat
                            } else {
                                ActiveTab::Plans
                            };
                            draw(&mut terminal, &state);
                        } else if let KeyCode::Char(c) = key.code {
                            // Wide layout: treat as ordinary text.
                            state.input.insert(state.cursor, c);
                            state.cursor += 1;
                            draw(&mut terminal, &state);
                        }
                    }
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

#[cfg(test)]
mod layout_tests {
    use super::*;

    fn state_with_workflows(n: usize) -> AppState {
        let mut s = AppState::new("http://localhost:0");
        // AppState::new loads ~/.larql/workflows.json from disk; clear
        // it so tests don't depend on dev fixtures.
        s.workflows.workflows.clear();
        for i in 0..n {
            s.workflows.workflows.push(Workflow {
                name: format!("flow {i}"),
                state: WorkflowState::Active,
                steps: vec![workflows::Step {
                    description: "step".into(),
                    state: StepState::Pending,
                    output: None,
                }],
                ts: 0,
            });
        }
        s
    }

    fn rect(w: u16) -> Rect {
        Rect { x: 0, y: 0, width: w, height: 30 }
    }

    #[test]
    fn no_workflows_means_chat_only() {
        let s = state_with_workflows(0);
        assert_eq!(compute_layout(&s, rect(200)), LayoutMode::ChatOnly);
        assert_eq!(compute_layout(&s, rect(40)), LayoutMode::ChatOnly);
    }

    #[test]
    fn user_hidden_means_chat_only_even_when_wide() {
        let mut s = state_with_workflows(2);
        s.sidebar_visible = false;
        assert_eq!(compute_layout(&s, rect(200)), LayoutMode::ChatOnly);
    }

    #[test]
    fn wide_terminal_picks_side_by_side() {
        let s = state_with_workflows(2);
        let m = compute_layout(&s, rect(200));
        assert!(matches!(m, LayoutMode::SideBySide { .. }));
    }

    #[test]
    fn narrow_terminal_falls_back_to_tabs() {
        let s = state_with_workflows(2);
        // 60 cols: chat_min(40) + sidebar_min(28) = 68 > 60.
        assert_eq!(compute_layout(&s, rect(60)), LayoutMode::Tabs);
    }

    #[test]
    fn boundary_precisely_at_threshold() {
        let s = state_with_workflows(2);
        // Need chat_min(40) + at least sidebar_min(28) → 68.
        // At 67 we should still be in tabs.
        assert_eq!(compute_layout(&s, rect(67)), LayoutMode::Tabs);
        // At 100 we should be side-by-side.
        assert!(matches!(compute_layout(&s, rect(100)), LayoutMode::SideBySide { .. }));
    }
}

#[cfg(test)]
mod extract_block_tests {
    use super::extract_block;

    #[test]
    fn extracts_simple() {
        let s = "```summary\nhello world\n```";
        assert_eq!(extract_block(s, "summary").as_deref(), Some("hello world"));
    }

    #[test]
    fn handles_nested_fence_without_truncating() {
        // The bug we're regression-testing: a ```terminal block nested
        // inside ```summary used to close the outer fence at the inner
        // opener, dropping everything after.
        let s = "```summary\nhello\n\n```terminal\nfoo\nbar\n```\n```";
        let body = extract_block(s, "summary").expect("extracted");
        assert!(body.contains("hello"));
        assert!(body.contains("foo"));
        assert!(body.contains("bar"));
    }

    #[test]
    fn returns_none_when_tag_missing() {
        assert!(extract_block("no fence here", "summary").is_none());
    }

    #[test]
    fn handles_unclosed_block_gracefully() {
        // Should not panic; should return whatever body we have.
        let s = "```summary\nstreaming…";
        let body = extract_block(s, "summary").expect("partial body");
        assert!(body.contains("streaming"));
    }
}

#[cfg(test)]
mod tokenize_tests {
    /// The legacy `split_whitespace` path destroyed every multi-space
    /// run inside a quoted argument; shell_words preserves them.
    /// These tests pin the new behaviour so we don't regress.
    #[test]
    fn shell_words_preserves_quoted_indentation() {
        let input = "python3 -c \"def f():\n    return 1\nprint(f())\"";
        let argv = shell_words::split(input).expect("tokenise");
        assert_eq!(argv.len(), 3);
        assert_eq!(argv[0], "python3");
        assert_eq!(argv[1], "-c");
        // The 4-space indent before `return` MUST survive.
        assert!(argv[2].contains("    return 1"));
        assert!(argv[2].contains("\n"));
    }

    #[test]
    fn shell_words_split_whitespace_diverge_on_quotes() {
        let input = "ls \"path with spaces\"";
        let argv = shell_words::split(input).unwrap();
        assert_eq!(argv, vec!["ls", "path with spaces"]);
        // For comparison, the legacy path would mistokenise:
        let legacy: Vec<String> = input.split_whitespace().map(str::to_string).collect();
        assert_eq!(legacy, vec!["ls", "\"path", "with", "spaces\""]);
    }

    #[test]
    fn shell_words_falls_back_on_unclosed_quote() {
        // We choose to fall back to whitespace split rather than fail
        // the tool call — the user gets *something* even from a half
        // emitted command.  This test pins the unwrap_or_else branch.
        let bad = "ls \"unclosed";
        assert!(shell_words::split(bad).is_err());
        let fallback: Vec<String> = bad.split_whitespace().map(str::to_string).collect();
        assert_eq!(fallback, vec!["ls", "\"unclosed"]);
    }
}

#[cfg(test)]
mod strip_tests {
    use super::strip_meta_blocks;

    #[test]
    fn hides_fact_block_keeps_prose() {
        let s = "Here is the answer.\n\n```fact\nkey: foo\nvalue: bar\n```\n\nMore prose.";
        let out = strip_meta_blocks(s);
        assert!(!out.contains("```fact"));
        assert!(!out.contains("key: foo"));
        assert!(out.contains("Here is the answer"));
        assert!(out.contains("More prose"));
    }

    #[test]
    fn hides_status_plan_tool_blocks() {
        let s = "Doing it.\n\
            ```status\ntask: x\nstate: active\n```\n\
            ```plan\nworkflow: x\nstep: a\n```\n\
            ```tool\nrun ls\n```\n\
            Done.";
        let out = strip_meta_blocks(s);
        for needle in ["```status", "```plan", "```tool", "task: x", "workflow: x", "run ls"] {
            assert!(!out.contains(needle), "should not contain {needle:?}\nGot: {out:?}");
        }
        assert!(out.contains("Doing it"));
        assert!(out.contains("Done"));
    }

    #[test]
    fn keeps_normal_code_blocks() {
        let s = "Look:\n```python\nprint('hi')\n```\nDone.";
        let out = strip_meta_blocks(s);
        assert!(out.contains("```python"));
        assert!(out.contains("print('hi')"));
    }

    #[test]
    fn hides_unfinished_block_during_streaming() {
        // Mid-stream: opening fence present, closing fence not yet.
        let s = "Reasoning…\n```fact\nkey: foo";
        let out = strip_meta_blocks(s);
        assert!(out.contains("Reasoning"));
        assert!(!out.contains("```fact"));
        assert!(!out.contains("key: foo"));
    }
}

#[cfg(test)]
mod sidebar_tests {
    use super::wrap_words;

    #[test]
    fn fits_when_short() {
        assert_eq!(wrap_words("hello world", 20), vec!["hello world"]);
    }

    #[test]
    fn breaks_only_on_whitespace() {
        let rows = wrap_words("implement Redis client with consistent hashing", 18);
        // No row should end mid-word.
        for row in &rows {
            for ch in row.chars().rev() {
                if ch == ' ' { panic!("trailing space in row {row:?}"); }
                break;
            }
            assert!(row.len() <= 18 + 1, "row too wide: {row:?}");
        }
        let joined = rows.join(" ");
        assert_eq!(joined, "implement Redis client with consistent hashing");
    }

    #[test]
    fn no_ellipsis_in_long_word() {
        // A pathological identifier longer than width gets hard-split,
        // but still no ellipsis character.
        let rows = wrap_words("supercalifragilisticexpialidocious", 10);
        for row in &rows {
            assert!(!row.contains('…'));
        }
        assert_eq!(rows.join(""), "supercalifragilisticexpialidocious");
    }

    #[test]
    fn empty_text_yields_one_empty_row() {
        let rows = wrap_words("", 10);
        assert_eq!(rows, vec![String::new()]);
    }
}
