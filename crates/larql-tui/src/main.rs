//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

use std::io;
use std::time::Instant;

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

use serde::{Deserialize, Serialize};

// ── Types ────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Message {
    User(String),
    Assistant(String),
    System(String),
    ToolUse { tool: String, detail: String },
    ToolResult { summary: String },
    Metrics { tok_s: f64, tokens: usize },
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
                "larql — LLM as a Database. Connect to server, type questions, use skills.".into()
            )],
            status: format!("connecting to {server_url}..."),
            is_generating: false,
            server_url: server_url.to_string(),
        }
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

/// Send a chat request to the server and collect the streamed response.
/// Returns the full assistant response text.
async fn chat_stream(
    url: &str,
    user_msg: &str,
    tx: &tokio::sync::mpsc::Sender<String>,
) -> Result<(), String> {
    let client = reqwest::Client::new();
    let req = ChatRequest {
        model: "gemma-3-4b".into(),
        messages: vec![ChatMsg { role: "user".into(), content: user_msg.into() }],
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

    // Read SSE stream
    use futures::StreamExt;
    let mut stream = resp.bytes_stream();
    let mut buf = String::new();

    while let Some(chunk) = stream.next().await {
        let chunk = chunk.map_err(|e| format!("stream error: {e}"))?;
        let text = String::from_utf8_lossy(&chunk);
        buf.push_str(&text);

        // Process complete SSE lines
        while let Some(newline_pos) = buf.find('\n') {
            let line = buf[..newline_pos].to_string();
            buf = buf[newline_pos + 1..].to_string();

            let line = line.trim();
            if line.is_empty() || line.starts_with(':') { continue; }

            if let Some(data) = line.strip_prefix("data: ") {
                if data.trim() == "[DONE]" {
                    return Ok(());
                }
                // Parse OpenAI delta format
                if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                    if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                        let _ = tx.send(content.to_string()).await;
                    }
                }
            }
        }
    }
    Ok(())
}

// ── Skills ────────────────────────────────────────────────────────────────

fn home_dir() -> std::path::PathBuf {
    std::env::var("HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}

fn match_skills(input: &str) -> String {
    let input_lower = input.to_lowercase();
    let mut context = String::new();

    let skills_dirs = vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        home_dir().join(".larql/skills"),
    ];

    for dir in &skills_dirs {
        if !dir.is_dir() { continue; }
        let Ok(entries) = std::fs::read_dir(dir) else { continue; };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }
            let skill_name = path.file_name().unwrap_or_default()
                .to_string_lossy().to_lowercase();

            let matches = input_lower.contains(&skill_name) || match skill_name.as_str() {
                "list" => ["list", "files", "directory", "folder", "ls", "what's here", "show me"]
                    .iter().any(|k| input_lower.contains(k)),
                "search" => ["search", "find", "grep", "look for", "where is"]
                    .iter().any(|k| input_lower.contains(k)),
                "stats" => ["stats", "statistics", "metrics", "how many", "count", "overview"]
                    .iter().any(|k| input_lower.contains(k)),
                "du" => ["disk", "space", "storage", "size", "how big", "usage"]
                    .iter().any(|k| input_lower.contains(k)),
                "git" => ["git", "commit", "branch", "changes", "diff", "status"]
                    .iter().any(|k| input_lower.contains(k)),
                _ => false,
            };

            if matches {
                let skill_md = path.join("skill.md");
                if let Ok(content) = std::fs::read_to_string(&skill_md) {
                    if !context.is_empty() { context.push_str(" "); }
                    context.push_str(&format!("[Skill: {}] {}", skill_name,
                        content.replace('\n', " ")));
                }
            }
        }
    }
    context
}

fn execute_skill_tool(text: &str, messages: &mut Vec<Message>) -> Option<String> {
    // Look for ```tool skill_name args```
    let open = "```tool";
    let start = text.find(open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    let tool_call = after[..close].trim();

    let parts: Vec<&str> = tool_call.splitn(2, char::is_whitespace).collect();
    let skill_name = parts.first()?;
    let skill_args = parts.get(1).unwrap_or(&"");

    // Find tool.sh
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
        tool: format!("skill:{skill_name}"),
        detail: format!("{skill_args}"),
    });

    match std::process::Command::new("bash").arg(&path)
        .args(skill_args.split_whitespace()).output()
    {
        Ok(output) => {
            let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

            // Route blocks
            if let Some(summary) = extract_block(&tool_output, "summary") {
                messages.push(Message::ToolResult { summary: summary.clone() });
                // chartjs goes to TUI rendering
                if let Some(chart) = extract_block(&tool_output, "chartjs") {
                    messages.push(Message::ToolUse {
                        tool: "chart".into(), detail: chart,
                    });
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
                // Use gc-markdown for rich rendering
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
                    Span::styled(format!(" {}", &detail[..detail.len().min(70)]), Style::default().fg(Color::DarkGray)),
                ]));
            }
            Message::ToolResult { summary } => {
                // Render tool result as markdown too
                lines.extend(gc_markdown::render_markdown(summary, gc_markdown::Theme::Dark));
                lines.push(Line::from(""));
            }
            Message::Metrics { tok_s, tokens } => {
                lines.push(Line::from(Span::styled(
                    format!("  ↳ {tok_s:.1} tok/s · {tokens} tokens"),
                    Style::default().fg(Color::DarkGray),
                )));
                lines.push(Line::from(""));
            }
        }
    }

    let visible = area.height.saturating_sub(2) as usize;
    let scroll = if lines.len() > visible { (lines.len() - visible) as u16 } else { 0 };

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
                format!("Cannot reach {server_url}. Start the server:\n  cargo run --release --features metal -p larql-server")
            ));
        }
    }
    draw(&mut terminal, &state);

    // ── Event loop ──
    let (token_tx, mut token_rx) = tokio::sync::mpsc::channel::<String>(256);

    loop {
        // Poll incoming tokens from streaming response
        let mut new_output = false;
        while let Ok(token) = token_rx.try_recv() {
            if let Some(Message::Assistant(ref mut text)) = state.messages.last_mut() {
                text.push_str(&token);
            } else {
                state.messages.push(Message::Assistant(token));
            }
            new_output = true;
        }

        if new_output {
            draw(&mut terminal, &state);
        }

        // Poll keyboard
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

                        state.messages.push(Message::User(input.clone()));
                        state.is_generating = true;
                        draw(&mut terminal, &state);

                        // Inject skills if matched
                        let skill_context = match_skills(&input);
                        let full_msg = if !skill_context.is_empty() {
                            format!("{skill_context} --- User request: {input}")
                        } else {
                            input
                        };

                        // Send async request
                        let url = state.server_url.clone();
                        let tx = token_tx.clone();
                        let is_gen_tx = token_tx.clone(); // signal completion
                        tokio::spawn(async move {
                            match chat_stream(&url, &full_msg, &tx).await {
                                Ok(()) => { let _ = tx.send("\n__DONE__".to_string()).await; }
                                Err(e) => { let _ = tx.send(format!("\n__ERROR__:{e}")).await; }
                            }
                        });
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

        // Check for completion signal
        if state.is_generating {
            if let Some(Message::Assistant(ref text)) = state.messages.last() {
                if text.ends_with("\n__DONE__") || text.contains("\n__ERROR__:") {
                    // Clean up signal markers
                    if let Some(Message::Assistant(ref mut text)) = state.messages.last_mut() {
                        if let Some(pos) = text.find("\n__DONE__") {
                            text.truncate(pos);
                        }
                        if let Some(pos) = text.find("\n__ERROR__:") {
                            let err = text[pos + 11..].to_string();
                            text.truncate(pos);
                            state.messages.push(Message::System(format!("Error: {err}")));
                        }
                    }

                    // Check for tool calls in the response
                    let response_text = if let Some(Message::Assistant(ref text)) = state.messages.last() {
                        text.clone()
                    } else { String::new() };

                    if let Some(summary) = execute_skill_tool(&response_text, &mut state.messages) {
                        // Feed summary back to model
                        let url = state.server_url.clone();
                        let tx = token_tx.clone();
                        let feedback = format!("Tool output: {} --- Briefly comment on the results.", summary.replace('\n', " | "));
                        state.messages.push(Message::Assistant(String::new()));
                        tokio::spawn(async move {
                            match chat_stream(&url, &feedback, &tx).await {
                                Ok(()) => { let _ = tx.send("\n__DONE__".to_string()).await; }
                                Err(e) => { let _ = tx.send(format!("\n__ERROR__:{e}")).await; }
                            }
                        });
                        // Stay in generating state for the follow-up
                    } else {
                        state.is_generating = false;
                    }
                    draw(&mut terminal, &state);
                }
            }
        }
    }

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}
