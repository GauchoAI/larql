//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

use std::io;

use crossterm::event::{self, Event as CEvent, KeyCode, KeyModifiers, KeyEventKind};
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
    ToolResult { summary: String },
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
        }
    }

    /// Build the OpenAI messages array from conversation history.
    fn build_chat_messages(&self) -> Vec<ChatMsg> {
        let mut msgs = Vec::new();

        for msg in &self.messages {
            match msg {
                Message::User(text) => {
                    msgs.push(ChatMsg { role: "user".into(), content: text.clone() });
                }
                Message::Assistant(text) if !text.is_empty() => {
                    msgs.push(ChatMsg { role: "assistant".into(), content: text.clone() });
                }
                Message::ToolResult { summary } => {
                    msgs.push(ChatMsg {
                        role: "user".into(),
                        content: format!("[Tool output]\n{summary}"),
                    });
                }
                _ => {}
            }
        }

        // Cap context to avoid unbounded prefill growth
        let max_msgs = 12;
        let start = msgs.len().saturating_sub(max_msgs);
        msgs.drain(..start);
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

    let resp = client.post(format!("{url}/v1/chat/completions"))
        .json(&req)
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

fn spawn_chat(url: String, messages: Vec<ChatMsg>, tx: tokio::sync::mpsc::Sender<StreamEvent>) {
    tokio::spawn(async move {
        match chat_stream(&url, messages, &tx).await {
            Ok(()) => { let _ = tx.send(StreamEvent::Done).await; }
            Err(e) => { let _ = tx.send(StreamEvent::Error(e)).await; }
        }
    });
}

// ── Skills ────────────────────────────────────────────────────────────────

fn home_dir() -> std::path::PathBuf {
    std::env::var("HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}

fn execute_skill_tool(text: &str, messages: &mut Vec<Message>) -> Option<String> {
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
    messages.push(Message::ToolUse {
        tool: format!("{skill_name}"),
        detail: skill_args.chars().take(70).collect(),
    });

    match std::process::Command::new("bash").arg(&path)
        .args(skill_args.split_whitespace()).output()
    {
        Ok(output) => {
            let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

            if let Some(summary) = extract_block(&tool_output, "summary") {
                messages.push(Message::ToolResult { summary: summary.clone() });
                if let Some(chart) = extract_block(&tool_output, "chartjs") {
                    let chart_md = format!("```chartjs\n{chart}\n```");
                    messages.push(Message::ToolResult { summary: chart_md });
                }
                return Some(summary);
            }
            None
        }
        Err(e) => {
            messages.push(Message::System(format!("tool error: {e}")));
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
        }
    }

    // Scroll: estimate wrapped line count for proper auto-scroll.
    // Each Line can wrap across multiple screen rows. Use the inner
    // width (area minus borders) to approximate the wrapped height.
    let inner_width = area.width.saturating_sub(2) as usize;
    let wrapped_height: usize = lines.iter().map(|line| {
        let content_len: usize = line.spans.iter().map(|s| s.content.len()).sum();
        if content_len == 0 { 1 } else { content_len.div_ceil(inner_width.max(1)) }
    }).sum();
    let visible = area.height.saturating_sub(2) as usize;
    let scroll = if wrapped_height > visible { (wrapped_height - visible) as u16 } else { 0 };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" larql ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    let para = Paragraph::new(lines).block(block).wrap(Wrap { trim: false }).scroll((scroll, 0));
    f.render_widget(para, area);
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

// ── Main ─────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> io::Result<()> {
    let server_url = std::env::var("LARQL_SERVER")
        .unwrap_or_else(|_| "http://localhost:3000".into());

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new(&server_url);
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

    let (ev_tx, mut ev_rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);

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

                    if let Some(_summary) = execute_skill_tool(&response_text, &mut state.messages) {
                        // Tool executed. The ToolResult is now in messages.
                        // Send full conversation history so model has context.
                        state.messages.push(Message::Assistant(String::new()));
                        let chat_msgs = state.build_chat_messages();
                        spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone());
                    } else {
                        state.is_generating = false;
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
            if let CEvent::Key(key) = event::read()? {
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

                        // Send full conversation history
                        let chat_msgs = state.build_chat_messages();
                        spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone());
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
                    _ => {}
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
