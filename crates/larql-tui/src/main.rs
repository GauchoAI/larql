//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

use std::io;

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
    let sid = match session_id {
        Some(s) => s.to_string(),
        None => return,
    };
    let url = format!("{server_url}/v1/sessions/{sid}/log");
    let body = serde_json::json!({"role": role, "content": content});
    let payload = serde_json::to_string(&body).unwrap_or_default();
    // Non-blocking: fire & forget, don't slow down the chat loop.
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

    let path = tool_path?;
    let detail: String = skill_args.chars().take(70).collect();
    messages.push(Message::ToolUse {
        tool: format!("{skill_name}"),
        detail: detail.clone(),
    });
    // Persist the invocation as `tool_use`.  Content is `<name> <args>`.
    let tool_use_payload = if detail.is_empty() {
        skill_name.to_string()
    } else {
        format!("{skill_name} {detail}")
    };
    append_turn_to_session(server_url, session_id, "tool_use", &tool_use_payload);

    match std::process::Command::new("bash").arg(&path)
        .args(skill_args.split_whitespace()).output()
    {
        Ok(output) => {
            let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

            if let Some(summary) = extract_block(&tool_output, "summary") {
                messages.push(Message::ToolResult { summary: summary.clone() });
                append_turn_to_session(server_url, session_id, "tool_result", &summary);

                if let Some(chart) = extract_block(&tool_output, "chartjs") {
                    let chart_md = format!("```chartjs\n{chart}\n```");
                    messages.push(Message::ToolRender { content: chart_md.clone() });
                    // Persist as a separate role so resume replays the
                    // chart in-place but doesn't feed it to the model.
                    append_turn_to_session(server_url, session_id, "tool_render", &chart_md);
                }
                return Some(summary);
            }
            None
        }
        Err(e) => {
            let err = format!("tool error: {e}");
            messages.push(Message::System(err.clone()));
            append_turn_to_session(server_url, session_id, "tool_error", &err);
            None
        }
    }
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

    // Scroll: estimate wrapped line count.  Auto-pin to the bottom
    // when `state.scroll_offset == 0`; otherwise back up by that many
    // rows so the user can read older messages.
    let inner_width = area.width.saturating_sub(2) as usize;
    let wrapped_height: usize = lines.iter().map(|line| {
        let content_len: usize = line.spans.iter().map(|s| s.content.len()).sum();
        if content_len == 0 { 1 } else { content_len.div_ceil(inner_width.max(1)) }
    }).sum();
    let visible = area.height.saturating_sub(2) as usize;
    let bottom_scroll: u16 = if wrapped_height > visible {
        (wrapped_height - visible) as u16
    } else {
        0
    };
    let scroll = bottom_scroll.saturating_sub(state.scroll_offset);

    let title_text = if state.scroll_offset > 0 {
        format!(" larql · scrolled +{} (End to follow) ", state.scroll_offset)
    } else {
        " larql ".into()
    };
    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(title_text, Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false }).scroll((scroll, 0));
    f.render_widget(para, area);
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
            // One blank trailer line per message.
            let mut v: Vec<usize> = t.lines().map(str::len).collect();
            v.push(0);
            v
        }
        Message::ToolUse { tool, detail } => vec![tool.len() + detail.len() + 4],
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

    // Inject skill instructions as a system primer so the model knows
    // it can drive tools by emitting ```tool <name> <args>``` blocks.
    if let Some(primer) = build_skill_primer() {
        state.messages.push(Message::System(primer));
    }

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

/// Build a single system primer string from every skill.md found in
/// `~/.larql/skills/` and `./.skills/`.  Concatenated so the model sees
/// the full list of tools it can invoke.
fn build_skill_primer() -> Option<String> {
    let mut sections: Vec<String> = Vec::new();
    let dirs = vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        home_dir().join(".larql/skills"),
    ];
    for dir in &dirs {
        if let Ok(entries) = std::fs::read_dir(dir) {
            for entry in entries.flatten() {
                let p = entry.path().join("skill.md");
                if p.exists() {
                    if let Ok(text) = std::fs::read_to_string(&p) {
                        sections.push(text.trim().to_string());
                    }
                }
            }
        }
    }
    if sections.is_empty() {
        return None;
    }
    Some(format!(
        "You have the following skills available. To use one, emit a fenced code block with language `tool`, e.g.:\n\n```tool\nlist /tmp\n```\n\nThe system intercepts these blocks, runs the matching `~/.larql/skills/<name>/tool.sh`, and feeds the summary back as a follow-up turn for you to comment on.\n\n{}",
        sections.join("\n\n---\n\n")
    ))
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
        let sid = state.session_id.clone();
        if let Some(summary) = execute_skill_tool(&response_text, &mut state.messages, server_url, sid.as_deref()) {
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

/// Build chat messages including system primers as a separate `system`
/// role and feeding any `ToolResult` back to the model as a synthetic
/// user turn.  Without that round-trip the model sees its own
/// `\`\`\`tool …\`\`\`` block but no output, so it just re-emits the
/// same call instead of writing a summary.
///
/// The server's chat-log dedupe drops user turns whose content starts
/// with `"Tool result:"` so this re-feed doesn't pollute the on-disk
/// session log.
fn build_chat_messages_with_system(state: &AppState) -> Vec<ChatMsg> {
    let mut msgs: Vec<ChatMsg> = Vec::new();
    for m in &state.messages {
        match m {
            Message::System(text) if !text.is_empty() => {
                msgs.push(ChatMsg { role: "system".into(), content: text.clone() });
            }
            Message::User(text) if !text.is_empty() => {
                msgs.push(ChatMsg { role: "user".into(), content: text.clone() });
            }
            Message::Assistant(text) if !text.is_empty() => {
                msgs.push(ChatMsg { role: "assistant".into(), content: text.clone() });
            }
            Message::ToolResult { summary } if !summary.is_empty() => {
                // Cap the body but preserve the tail — most skill
                // summaries put the bottom-line ("Total: N items ...")
                // at the end, so head-only truncation loses the
                // information the model needs to reply.
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
                msgs.push(ChatMsg {
                    role: "user".into(),
                    content: format!(
                        "Tool result:\n{body}\n\nPlease summarise this concisely for me."
                    ),
                });
            }
            _ => {}
        }
    }
    msgs
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
    // Inject the skill primer so Gemma knows which tools it can call
    // (`list`, `run`, `git`, …).  The TUI's `execute_skill_tool` parses
    // any ```tool``` block the model emits and runs the matching shell
    // script in `~/.larql/skills/<name>/`.
    if let Some(primer) = build_skill_primer() {
        state.messages.push(Message::System(primer));
    }
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

                    // Only execute tools on first response, not follow-ups
                    if tool_depth == 0 {
                        let sid = state.session_id.clone();
                        let server_url_cl = state.server_url.clone();
                        if let Some(_summary) = execute_skill_tool(&response_text, &mut state.messages, &server_url_cl, sid.as_deref()) {
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
