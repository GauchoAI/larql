//! larql TUI — interactive terminal interface for LLM-as-a-Database.
//!
//! Ratatui chat interface with streaming output, tool use (file writing),
//! KNN overlay indicators, and tok/s metrics. Connects to larql-server
//! via subprocess (bench_interactive) for the decode pipeline.

use std::io;
use std::process::{Command, Stdio};
use std::sync::{Arc, Mutex};
use std::io::{BufRead, BufReader, Write as IoWrite};
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
use ratatui::widgets::{Block, Borders, Paragraph, Wrap, Clear};
use ratatui::Terminal;

// ── Types ────────────────────────────────────────────────────────────────

#[derive(Clone)]
enum Message {
    User(String),
    Assistant(String),
    System(String),
    ToolUse { tool: String, detail: String },
    Metrics { tok_s: f64, prefill_ms: f64 },
}

struct AppState {
    input: String,
    cursor: usize,
    messages: Vec<Message>,
    status: String,
    is_generating: bool,
    tok_s: f64,
    total_tokens: usize,
    knn_entries: usize,
    last_prompt: String,      // for echo stripping
    echo_stripped: bool,       // whether we've stripped the echo for this query
    assistant_buf: String,     // accumulate streaming tokens
}

impl AppState {
    fn new() -> Self {
        Self {
            input: String::new(),
            cursor: 0,
            messages: vec![Message::System(
                "larql — LLM as a Database. Type a question, or INSERT <entity> <relation> <target>.".into()
            )],
            status: "connecting...".into(),
            is_generating: false,
            tok_s: 0.0,
            total_tokens: 0,
            knn_entries: 0,
            last_prompt: String::new(),
            echo_stripped: false,
            assistant_buf: String::new(),
        }
    }
}

// ── Backend (subprocess) ─────────────────────────────────────────────────

struct Backend {
    stdin: std::process::ChildStdin,
    stdout_rx: std::sync::mpsc::Receiver<String>,
    stderr_rx: std::sync::mpsc::Receiver<String>,
}

impl Backend {
    fn spawn() -> io::Result<Self> {
        // Find the bench_interactive binary
        let bin = find_binary()?;
        let model = std::env::var("LARQL_MODEL")
            .unwrap_or_else(|_| "/Users/miguel_lemos/Desktop/gemma-3-4b-it".into());
        let vindex = std::env::var("LARQL_VINDEX")
            .unwrap_or_else(|_| "/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex".into());

        let mut child = Command::new(&bin)
            .args(["--model", &model, "--vindex", &vindex, "--walk-only", "--no-warmup"])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;

        let stdin = child.stdin.take().unwrap();
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
        let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

        // Read stdout CHAR-BY-CHAR for real-time streaming.
        // Accumulate into chunks separated by small time gaps or newlines.
        std::thread::spawn(move || {
            use std::io::Read;
            let mut reader = BufReader::new(stdout);
            let mut buf = [0u8; 1];
            let mut line_buf = String::new();
            while reader.read(&mut buf).unwrap_or(0) == 1 {
                let c = buf[0] as char;
                line_buf.push(c);
                // Send on newline or when we have accumulated text
                if c == '\n' || line_buf.len() > 80 {
                    let _ = stdout_tx.send(line_buf.clone());
                    line_buf.clear();
                }
            }
            if !line_buf.is_empty() {
                let _ = stdout_tx.send(line_buf);
            }
        });

        // Read stderr in background
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr);
            for line in reader.lines() {
                if let Ok(line) = line {
                    let _ = stderr_tx.send(line);
                }
            }
        });

        Ok(Self { stdin, stdout_rx, stderr_rx })
    }

    fn send(&mut self, cmd: &str) -> io::Result<()> {
        writeln!(self.stdin, "{}", cmd)?;
        self.stdin.flush()
    }

    fn poll_stdout(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.stdout_rx.try_recv() {
            lines.push(line);
        }
        lines
    }

    fn poll_stderr(&self) -> Vec<String> {
        let mut lines = Vec::new();
        while let Ok(line) = self.stderr_rx.try_recv() {
            lines.push(line);
        }
        lines
    }
}

fn find_binary() -> io::Result<String> {
    // Look relative to the TUI binary, or in common locations
    let candidates = [
        "target/release/examples/bench_interactive",
        "../target/release/examples/bench_interactive",
        "larql/target/release/examples/bench_interactive",
    ];
    for c in &candidates {
        if std::path::Path::new(c).exists() {
            return Ok(c.to_string());
        }
    }
    Err(io::Error::new(io::ErrorKind::NotFound,
        "bench_interactive not found. Build with: cargo build --release --features metal -p larql-inference --example bench_interactive"))
}

// ── Drawing ──────────────────────────────────────────────────────────────

fn draw(terminal: &mut Terminal<CrosstermBackend<io::Stdout>>, state: &AppState) {
    terminal.draw(|f| {
        let chunks = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(5),      // messages
                Constraint::Length(3),    // input
                Constraint::Length(1),    // status bar
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
                for line in text.lines() {
                    if line.starts_with("```") {
                        lines.push(Line::from(Span::styled(line, Style::default().fg(Color::DarkGray))));
                    } else if line.starts_with('#') {
                        lines.push(Line::from(Span::styled(line, Style::default().fg(Color::Yellow).add_modifier(Modifier::BOLD))));
                    } else {
                        lines.push(Line::from(Span::styled(line, Style::default().fg(Color::Gray))));
                    }
                }
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
            Message::Metrics { tok_s, prefill_ms } => {
                lines.push(Line::from(Span::styled(
                    format!("  ↳ {tok_s:.1} tok/s · prefill {prefill_ms:.0}ms"),
                    Style::default().fg(Color::DarkGray),
                )));
                lines.push(Line::from(""));
            }
        }
    }

    // Auto-scroll to bottom
    let visible_height = area.height.saturating_sub(2) as usize;
    let scroll = if lines.len() > visible_height {
        (lines.len() - visible_height) as u16
    } else { 0 };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(Color::DarkGray))
        .title(Span::styled(" larql ", Style::default().fg(Color::Cyan).add_modifier(Modifier::BOLD)));

    let para = Paragraph::new(lines)
        .block(block)
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));

    f.render_widget(para, area);
}

fn draw_input(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let style = if state.is_generating {
        Style::default().fg(Color::DarkGray)
    } else {
        Style::default().fg(Color::White)
    };

    let input_text = if state.is_generating {
        "  generating...".to_string()
    } else if state.input.is_empty() {
        "  Type a question, or INSERT <entity> <relation> <target>...".to_string()
    } else {
        format!("  {}", state.input)
    };

    let block = Block::default()
        .borders(Borders::ALL)
        .border_style(Style::default().fg(if state.is_generating { Color::DarkGray } else { Color::Cyan }))
        .title(Span::styled(" input ", Style::default().fg(Color::Cyan)));

    let para = Paragraph::new(Line::from(Span::styled(input_text, style)))
        .block(block);

    f.render_widget(para, area);

    // Cursor
    if !state.is_generating {
        f.set_cursor_position((area.x + 3 + state.cursor as u16, area.y + 1));
    }
}

fn draw_status(f: &mut ratatui::Frame, state: &AppState, area: Rect) {
    let left = format!(" {} ", state.status);
    let right = if state.tok_s > 0.0 {
        format!(" {:.1} tok/s │ {} tokens │ {} KNN entries ", state.tok_s, state.total_tokens, state.knn_entries)
    } else {
        format!(" {} tokens │ {} KNN entries ", state.total_tokens, state.knn_entries)
    };

    let width = area.width as usize;
    let pad = width.saturating_sub(left.len() + right.len());
    let status_line = format!("{}{:pad$}{}", left, "", right, pad = pad);

    let para = Paragraph::new(Line::from(Span::styled(
        status_line, Style::default().fg(Color::White).bg(Color::DarkGray),
    )));
    f.render_widget(para, area);
}

// ── Main loop ────────────────────────────────────────────────────────────

fn main() -> io::Result<()> {
    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new();
    let mut be = Backend::spawn()?;
    state.status = "loading model...".into();
    draw(&mut terminal, &state);

    // Wait for "[ready]" from stderr
    let ready_deadline = Instant::now() + std::time::Duration::from_secs(60);
    loop {
        for line in be.poll_stderr() {
            if line.contains("[ready]") && line.contains("backend=") {
                let backend_name = if line.contains("metal") { "metal (GPU)" } else { "cpu" };
                state.status = format!("ready · {backend_name}");
                state.messages.push(Message::System(format!("Model loaded. Backend: {backend_name}")));
            }
            if line.contains("interleaved_q4k_real") {
                state.messages.push(Message::System("GPU decode enabled (Q4_K real)".into()));
            }
        }
        draw(&mut terminal, &state);
        if state.status.starts_with("ready") { break; }
        if Instant::now() > ready_deadline {
            state.status = "timeout waiting for model".into();
            break;
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
    }

    // Main event loop
    loop {
        // Poll backend output
        let mut new_output = false;
        for line in be.poll_stdout() {
            process_stdout_line(&line, &mut state);
            new_output = true;
        }
        for line in be.poll_stderr() {
            process_stderr_line(&line, &mut state);
            new_output = true;
        }

        if new_output {
            draw(&mut terminal, &state);
        }

        // Poll keyboard
        if event::poll(std::time::Duration::from_millis(50))? {
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

                        // Parse command
                        if input.to_lowercase().starts_with("insert ") {
                            let parts: Vec<&str> = input[7..].splitn(3, ' ').collect();
                            if parts.len() == 3 {
                                state.messages.push(Message::ToolUse {
                                    tool: "INSERT".into(),
                                    detail: format!("{} —[{}]→ {}", parts[0], parts[1], parts[2]),
                                });
                                be.send(&format!("insert {} {} {}", parts[0], parts[1], parts[2]))?;
                                state.is_generating = true;
                            }
                        } else if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
                            break;
                        } else {
                            state.messages.push(Message::User(input.clone()));
                            // Use the `chat` command which wraps in Gemma 3
                            // chat template internally. No echo of prompt.
                            state.last_prompt = input.clone();
                            state.echo_stripped = true; // chat cmd doesn't echo
                            state.assistant_buf.clear();
                            be.send(&format!("chat {input}"))?;
                            state.is_generating = true;
                            state.tok_s = 0.0;
                        }
                        draw(&mut terminal, &state);
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
                    KeyCode::Left if state.cursor > 0 => {
                        state.cursor -= 1;
                        draw(&mut terminal, &state);
                    }
                    KeyCode::Right if state.cursor < state.input.len() => {
                        state.cursor += 1;
                        draw(&mut terminal, &state);
                    }
                    _ => {}
                }
            }
        }
    }

    // Cleanup
    be.send("quit").ok();
    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    Ok(())
}

fn process_stdout_line(line: &str, state: &mut AppState) {
    let trimmed = line.trim_end();
    if trimmed.is_empty() || trimmed == ">" || trimmed == "> " { return; }

    // Strip leading "> " prompt marker
    let content = if trimmed.starts_with("> ") { &trimmed[2..] } else { trimmed };

    // Timing line — finalize the current response
    if content.contains("tok/s") && (content.contains("prefill") || content.contains("decode")) {
        // Flush accumulated assistant text
        if !state.assistant_buf.is_empty() {
            let text = state.assistant_buf.clone();
            state.messages.push(Message::Assistant(text));
            state.assistant_buf.clear();

            // Check for code blocks in the response — show as tool use
            if let Some(last) = state.messages.last() {
                if let Message::Assistant(ref text) = last {
                    if text.contains("```") {
                        state.messages.push(Message::ToolUse {
                            tool: "code".into(),
                            detail: "code block generated (copy to use)".into(),
                        });
                    }
                }
            }
        }
        if let Some(tok_s) = extract_tok_s(content) {
            state.tok_s = tok_s;
            if let Some(prefill) = extract_prefill_ms(content) {
                state.messages.push(Message::Metrics { tok_s, prefill_ms: prefill });
            }
        }
        state.is_generating = false;
        return;
    }

    // KNN override
    if content.contains("KNN override") {
        state.messages.push(Message::ToolUse {
            tool: "⚡ KNN override".into(),
            detail: content.to_string(),
        });
        state.is_generating = false;
        return;
    }

    // INSERT result
    if content.contains("inserted:") {
        state.messages.push(Message::System(content.to_string()));
        state.knn_entries += 1;
        state.is_generating = false;
        return;
    }

    if content.contains("KNN overlay now:") || content.contains("KNN override, no decode") {
        if content.contains("no decode") { state.is_generating = false; }
        return;
    }

    // Skip metadata lines
    if content.starts_with("[insert]") || content.starts_with("[spec]") { return; }

    // ── Echo stripping ──
    // bench_interactive prints the prompt text before the model output.
    // We strip it by checking if the accumulated text starts with the prompt.
    if !state.echo_stripped && !state.last_prompt.is_empty() {
        state.assistant_buf.push_str(content);
        // Check if we've accumulated enough to compare with the prompt
        let clean = state.assistant_buf.trim();
        if clean.starts_with(&state.last_prompt) || state.last_prompt.starts_with(clean) {
            // Still accumulating the echo OR echo matches exactly
            if state.assistant_buf.len() >= state.last_prompt.len() {
                // Strip the echo prefix
                let after_echo = state.assistant_buf[state.last_prompt.len()..].trim_start().to_string();
                state.assistant_buf = after_echo;
                state.echo_stripped = true;
            }
            return;
        }
        // Content doesn't match the prompt — not an echo, treat as real output
        state.echo_stripped = true;
    } else {
        state.assistant_buf.push_str(content);
    }

    // Update the live assistant message for streaming feel
    let display = state.assistant_buf.clone();
    if !display.trim().is_empty() {
        // Remove previous live assistant message and replace
        if let Some(Message::Assistant(_)) = state.messages.last() {
            state.messages.pop();
        }
        state.messages.push(Message::Assistant(display));
        state.total_tokens += 1;
    }
}

fn process_stderr_line(line: &str, state: &mut AppState) {
    if line.contains("knn-gpu-prefill") && line.contains("cos=") {
        // KNN probe fired during prefill
        if let Some(cos_start) = line.find("cos=") {
            let cos_str = &line[cos_start + 4..];
            let cos_end = cos_str.find(' ').unwrap_or(cos_str.len());
            state.status = format!("ready · KNN probe cos={}", &cos_str[..cos_end]);
        }
    }
}

fn extract_tok_s(line: &str) -> Option<f64> {
    let re_pos = line.find("tok/s")?;
    let before = &line[..re_pos];
    let num_start = before.rfind(|c: char| !c.is_ascii_digit() && c != '.')? + 1;
    before[num_start..].parse().ok()
}

fn extract_prefill_ms(line: &str) -> Option<f64> {
    let pos = line.find("prefill:")?;
    let after = &line[pos + 8..].trim_start();
    let end = after.find("ms")?;
    after[..end].trim().parse().ok()
}
