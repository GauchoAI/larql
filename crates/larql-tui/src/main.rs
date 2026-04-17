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
use std::fs::OpenOptions;
#[cfg(unix)]
use std::os::unix::io::{FromRawFd, IntoRawFd};

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
    last_output_time: Instant, // detect stalled generation
    pending_tool_result: Option<String>, // bash output to feed back to model
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
            last_output_time: Instant::now(),
            pending_tool_result: None,
        }
    }
}

// ── Backend (subprocess) ─────────────────────────────────────────────────

struct Backend {
    child: Option<std::process::Child>, // None for daemon mode
    stdin: Box<dyn IoWrite + Send>,
    stdout_rx: std::sync::mpsc::Receiver<String>,
    stderr_rx: std::sync::mpsc::Receiver<String>,
    log: Arc<Mutex<Option<std::fs::File>>>,
}

impl Drop for Backend {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
            let _ = child.wait();
        }
    }
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

        let stdin_handle: Box<dyn IoWrite + Send> = Box::new(child.stdin.take().unwrap());
        let stdout = child.stdout.take().unwrap();
        let stderr = child.stderr.take().unwrap();

        let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
        let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

        // Log file for debugging
        let log_path = std::env::temp_dir().join("larql-tui.log");
        let log_file = Arc::new(Mutex::new(
            OpenOptions::new().create(true).truncate(true).write(true).open(&log_path).ok()
        ));
        let log_stdout = Arc::clone(&log_file);
        let log_stderr = Arc::clone(&log_file);
        eprintln!("[larql-tui] log: {}", log_path.display());

        // Read stdout CHAR-BY-CHAR with timeout flush for partial lines.
        // The "> " prompt has no newline — must flush on timeout.
        std::thread::spawn(move || {
            use std::io::Read;
            use std::time::{Duration, Instant};
            // Set non-blocking reads via polling
            let mut reader = stdout;
            let mut buf = [0u8; 256];
            let mut line_buf = String::new();
            let mut last_read = Instant::now();
            loop {
                // Try to read available bytes
                match reader.read(&mut buf) {
                    Ok(0) => break, // EOF
                    Ok(n) => {
                        for &b in &buf[..n] {
                            let c = b as char;
                            line_buf.push(c);
                            if c == '\n' {
                                let _ = stdout_tx.send(line_buf.clone());
                                line_buf.clear();
                            }
                        }
                        last_read = Instant::now();
                        // Flush partial line immediately if it looks like a prompt
                        if line_buf.trim() == ">" || line_buf.trim() == "> " {
                            let _ = stdout_tx.send(line_buf.clone());
                            line_buf.clear();
                        }
                        // Log
                        if let Ok(mut guard) = log_stdout.lock() {
                            if let Some(ref mut f) = *guard {
                                use std::io::Write;
                                let _ = write!(f, "[stdout] {}", std::str::from_utf8(&buf[..n]).unwrap_or("?"));
                            }
                        }
                    }
                    Err(_) => break,
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
                    if let Ok(mut guard) = log_stderr.lock() {
                        if let Some(ref mut f) = *guard {
                            use std::io::Write;
                            let _ = writeln!(f, "[stderr] {}", line);
                        }
                    }
                    let _ = stderr_tx.send(line);
                }
            }
        });

        Ok(Self { child: Some(child), stdin: stdin_handle, stdout_rx, stderr_rx, log: Arc::clone(&log_file) })
    }

    /// Connect to an already-running daemon via FIFOs.
    fn connect_daemon() -> io::Result<Self> {
        use std::fs::File;

        let log_path = std::env::temp_dir().join("larql-tui.log");
        let log_file = Arc::new(Mutex::new(
            OpenOptions::new().create(true).truncate(true).write(true).open(&log_path).ok()
        ));
        let log_stdout = Arc::clone(&log_file);
        let log_stderr = Arc::clone(&log_file);

        // Open FIFOs — stdin is write-only, stdout/stderr are read-only
        let stdin_file = OpenOptions::new().write(true).open(DAEMON_FIFO_IN)?;
        let stdout_file = File::open(DAEMON_FIFO_OUT)?;
        let stderr_file = File::open(DAEMON_FIFO_ERR)?;

        let (stdout_tx, stdout_rx) = std::sync::mpsc::channel();
        let (stderr_tx, stderr_rx) = std::sync::mpsc::channel();

        // Stdout reader thread
        std::thread::spawn(move || {
            use std::io::Read;
            let mut reader = stdout_file;
            let mut buf = [0u8; 256];
            let mut line_buf = String::new();
            loop {
                match reader.read(&mut buf) {
                    Ok(0) => break,
                    Ok(n) => {
                        for &b in &buf[..n] {
                            let c = b as char;
                            line_buf.push(c);
                            if c == '\n' {
                                let _ = stdout_tx.send(line_buf.clone());
                                line_buf.clear();
                            }
                        }
                        if line_buf.trim() == ">" || line_buf.trim() == "> " {
                            let _ = stdout_tx.send(line_buf.clone());
                            line_buf.clear();
                        }
                        if let Ok(mut guard) = log_stdout.lock() {
                            if let Some(ref mut f) = *guard {
                                use std::io::Write;
                                let _ = write!(f, "[stdout] {}", std::str::from_utf8(&buf[..n]).unwrap_or("?"));
                            }
                        }
                    }
                    Err(_) => break,
                }
            }
        });

        // Stderr reader thread
        std::thread::spawn(move || {
            let reader = BufReader::new(stderr_file);
            for line in reader.lines() {
                if let Ok(line) = line {
                    if let Ok(mut guard) = log_stderr.lock() {
                        if let Some(ref mut f) = *guard {
                            use std::io::Write;
                            let _ = writeln!(f, "[stderr] {}", line);
                        }
                    }
                    let _ = stderr_tx.send(line);
                }
            }
        });

        Ok(Self {
            child: None, // daemon mode — don't own the process
            stdin: Box::new(stdin_file),
            stdout_rx,
            stderr_rx,
            log: log_file,
        })
    }

    fn send(&mut self, cmd: &str) -> io::Result<()> {
        self.log_msg(&format!("[SEND] {cmd}"));
        writeln!(self.stdin, "{}", cmd)?;
        self.stdin.flush()
    }

    fn log_msg(&self, msg: &str) {
        if let Ok(mut guard) = self.log.lock() {
            if let Some(ref mut f) = *guard {
                use std::io::Write;
                let _ = writeln!(f, "{msg}");
                let _ = f.flush();
            }
        }
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
    // Resolve relative to the TUI binary's own location (not CWD).
    let exe = std::env::current_exe().unwrap_or_default();
    let exe_dir = exe.parent().unwrap_or(std::path::Path::new("."));
    // The TUI binary is at target/release/larql, bench_interactive is at
    // target/release/examples/bench_interactive — same target dir.
    let sibling = exe_dir.join("examples/bench_interactive");
    if sibling.exists() {
        return Ok(sibling.to_string_lossy().into_owned());
    }
    // Fallback: search relative to CWD
    let candidates = [
        "target/release/examples/bench_interactive",
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

const DAEMON_PID_FILE: &str = "/tmp/larql-daemon.pid";
const DAEMON_FIFO_IN: &str = "/tmp/larql-daemon.stdin";
const DAEMON_FIFO_OUT: &str = "/tmp/larql-daemon.stdout";
const DAEMON_FIFO_ERR: &str = "/tmp/larql-daemon.stderr";

fn is_daemon_running() -> bool {
    if let Ok(pid_str) = std::fs::read_to_string(DAEMON_PID_FILE) {
        if let Ok(pid) = pid_str.trim().parse::<u32>() {
            let status = std::process::Command::new("kill")
                .args(["-0", &pid.to_string()])
                .output();
            if status.map(|o| o.status.success()).unwrap_or(false) {
                return true;
            }
        }
    }
    // Clean up stale FIFOs if daemon is not actually running
    let _ = std::fs::remove_file(DAEMON_PID_FILE);
    let _ = std::fs::remove_file(DAEMON_FIFO_IN);
    let _ = std::fs::remove_file(DAEMON_FIFO_OUT);
    let _ = std::fs::remove_file(DAEMON_FIFO_ERR);
    false
}

fn start_daemon() -> io::Result<()> {
    let bin = find_binary()?;
    let model = std::env::var("LARQL_MODEL")
        .unwrap_or_else(|_| "/Users/miguel_lemos/Desktop/gemma-3-4b-it".into());
    let vindex = std::env::var("LARQL_VINDEX")
        .unwrap_or_else(|_| "/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex".into());

    // Create FIFOs (remove old ones first)
    for fifo in &[DAEMON_FIFO_IN, DAEMON_FIFO_OUT, DAEMON_FIFO_ERR] {
        let _ = std::fs::remove_file(fifo);
        let status = std::process::Command::new("mkfifo").arg(fifo).status()?;
        if !status.success() {
            return Err(io::Error::new(io::ErrorKind::Other, format!("mkfifo failed for {fifo}")));
        }
    }

    // Start bench_interactive with FIFOs, backgrounded
    let child = std::process::Command::new("sh")
        .arg("-c")
        .arg(format!(
            "{bin} --model {model} --vindex {vindex} --walk-only --no-warmup \
             < {DAEMON_FIFO_IN} > {DAEMON_FIFO_OUT} 2> {DAEMON_FIFO_ERR} &\n\
             echo $!"
        ))
        .output()?;

    let pid = String::from_utf8_lossy(&child.stdout).trim().to_string();
    std::fs::write(DAEMON_PID_FILE, &pid)?;
    eprintln!("[larql] daemon started (PID {pid})");
    Ok(())
}

fn main() -> io::Result<()> {
    // Handle --daemon flag: just start daemon and exit
    let args: Vec<String> = std::env::args().collect();
    if args.iter().any(|a| a == "--daemon") {
        if is_daemon_running() {
            eprintln!("[larql] daemon already running");
            return Ok(());
        }
        start_daemon()?;
        eprintln!("[larql] daemon started. Run `larql` to connect.");
        return Ok(());
    }

    if args.iter().any(|a| a == "--stop") {
        if let Ok(pid_str) = std::fs::read_to_string(DAEMON_PID_FILE) {
            let _ = std::process::Command::new("kill").arg(pid_str.trim()).output();
            let _ = std::fs::remove_file(DAEMON_PID_FILE);
            eprintln!("[larql] daemon stopped");
        }
        return Ok(());
    }

    // Terminal setup
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new();

    // Spawn backend subprocess directly (daemon mode via --daemon is separate)
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

        // Timeout: if generating and no output for 3 seconds, assume done
        if state.is_generating && state.last_output_time.elapsed().as_secs() >= 30 {
            tui_log("[STATE] timeout → is_generating = false");
            if !state.assistant_buf.is_empty() {
                let buf_copy = state.assistant_buf.clone();
                if let Some(output) = execute_tool_calls(&buf_copy, &mut state.messages) {
                    state.pending_tool_result = Some(output);
                }
                state.assistant_buf.clear();
            }
            state.is_generating = false;
            new_output = true;
        }

        // Tool-result feedback: feed summary back to model (no visible user message)
        if let Some(result) = state.pending_tool_result.take() {
            let flat = result.replace('\n', " | ");
            let feedback = format!("Tool output: {flat} --- Based on this, provide a brief commentary.");
            // No user message shown — this is internal
            state.echo_stripped = true;
            state.assistant_buf.clear();
            state.last_output_time = Instant::now();
            be.send(&format!("chat {feedback}"))?;
            state.is_generating = true;
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
                                state.last_output_time = Instant::now();
                            }
                        } else if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
                            break;
                        } else {
                            state.messages.push(Message::User(input.clone()));
                            state.last_prompt = input.clone();
                            state.echo_stripped = true;
                            state.assistant_buf.clear();

                            // Auto-inject matching skills as single-line chat
                            let skill_context = match_skills(&input);
                            if !skill_context.is_empty() {
                                let flat_skill = skill_context.replace('\n', " ");
                                be.send(&format!("chat {flat_skill} --- User request: {input}"))?;
                            } else {
                                be.send(&format!("chat {input}"))?;
                            }
                            state.is_generating = true;
                            state.last_output_time = Instant::now(); // reset timeout clock
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

/// Scan .skills/ folders for skills matching the user's input.
/// Checks: ./.skills/ (local) and ~/.skills/ (global).
/// Returns concatenated skill.md content for all matching skills.
fn match_skills(input: &str) -> String {
    let input_lower = input.to_lowercase();
    let mut context = String::new();

    // Skills: ~/.larql/skills/ (global) and ./.skills/ (project-local override)
    let skills_dirs = vec![
        std::env::current_dir().unwrap_or_default().join(".skills"),
        dirs_fallback().join(".larql/skills"),
    ];

    for dir in &skills_dirs {
        if !dir.is_dir() { continue; }
        let Ok(entries) = std::fs::read_dir(dir) else { continue; };
        for entry in entries.flatten() {
            let path = entry.path();
            if !path.is_dir() { continue; }
            let skill_name = path.file_name().unwrap_or_default()
                .to_string_lossy().to_lowercase();

            // Match: skill name appears in input, or common synonyms
            let matches = input_lower.contains(&skill_name) || match skill_name.as_str() {
                "list" => input_lower.contains("list") || input_lower.contains("files")
                    || input_lower.contains("directory") || input_lower.contains("folder")
                    || input_lower.contains("ls"),
                "search" => input_lower.contains("search") || input_lower.contains("find")
                    || input_lower.contains("grep") || input_lower.contains("look for"),
                "stats" => input_lower.contains("stats") || input_lower.contains("statistics")
                    || input_lower.contains("metrics") || input_lower.contains("how many")
                    || input_lower.contains("count"),
                _ => false,
            };

            if matches {
                let skill_md = path.join("skill.md");
                if let Ok(content) = std::fs::read_to_string(&skill_md) {
                    if !context.is_empty() { context.push_str("\n\n"); }
                    context.push_str(&format!("[Skill: {}]\n{}", skill_name, content));
                    tui_log(&format!("[SKILL] matched '{}' for input: {}", skill_name, &input[..input.len().min(40)]));
                }
            }
        }
    }
    context
}

fn dirs_fallback() -> std::path::PathBuf {
    std::env::var("HOME").map(std::path::PathBuf::from)
        .unwrap_or_else(|_| std::path::PathBuf::from("/tmp"))
}

fn tui_log(msg: &str) {
    let path = std::env::temp_dir().join("larql-tui.log");
    if let Ok(mut f) = OpenOptions::new().append(true).open(&path) {
        let _ = writeln!(f, "{msg}");
    }
}

fn process_stdout_line(line: &str, state: &mut AppState) {
    state.last_output_time = Instant::now();
    // Preserve newlines! Only trim carriage returns, not \n.
    let trimmed = line.trim_end_matches('\r');
    if trimmed.is_empty() { return; }
    tui_log(&format!("[PROCESS] is_gen={} trimmed={:?}", state.is_generating, &trimmed[..trimmed.len().min(80)]));
    // The "> " prompt from bench_interactive means generation is done
    if trimmed.trim() == ">" || trimmed.trim() == "> " {
        if state.is_generating {
            tui_log("[STATE] prompt detected → is_generating = false");
            // Finalize
            if !state.assistant_buf.is_empty() {
                let buf_copy = state.assistant_buf.clone();
                if let Some(output) = execute_tool_calls(&buf_copy, &mut state.messages) {
                    state.pending_tool_result = Some(output);
                }
                state.assistant_buf.clear();
            }
            state.is_generating = false;
        }
        return;
    }

    // Strip leading "> " prompt marker
    let content = if trimmed.starts_with("> ") { &trimmed[2..] } else { trimmed };

    // Timing line — finalize the current response
    if content.contains("tok/s") && (content.contains("prefill") || content.contains("decode")) {
        // Execute tool calls on the accumulated text
        let mut bash_output: Option<String> = None;
        if !state.assistant_buf.is_empty() {
            let buf_copy = state.assistant_buf.clone();
            bash_output = execute_tool_calls(&buf_copy, &mut state.messages);
            state.assistant_buf.clear();
        }
        if let Some(tok_s) = extract_tok_s(content) {
            state.tok_s = tok_s;
            if let Some(prefill) = extract_prefill_ms(content) {
                state.messages.push(Message::Metrics { tok_s, prefill_ms: prefill });
            }
        }
        state.is_generating = false;
        // If a bash tool was executed, queue the output for the model
        if let Some(output) = bash_output {
            state.pending_tool_result = Some(output);
        }
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

/// Execute tool calls found in the response. Returns bash output if a bash
/// block was executed (for feeding back to the model).
fn execute_tool_calls(text: &str, messages: &mut Vec<Message>) -> Option<String> {
    let mut bash_result: Option<String> = None;

    // ── Skill tool calls: ```tool\nname args\n``` ──
    // The model outputs a ```tool``` block with the skill name and args.
    // We find the skill's tool.sh, execute it, parse the structured output,
    // route each block type: summary → LLM, chartjs → TUI, raw → log.
    if let Some(tool_block) = extract_fenced_block(text, "tool") {
        let parts: Vec<&str> = tool_block.trim().splitn(2, char::is_whitespace).collect();
        let skill_name = parts.first().unwrap_or(&"");
        let skill_args = parts.get(1).unwrap_or(&"");

        // Find the tool executable
        let skills_dirs = vec![
            std::env::current_dir().unwrap_or_default().join(".skills"),
            dirs_fallback().join(".larql/skills"),
        ];
        let mut tool_path = None;
        for dir in &skills_dirs {
            let candidate = dir.join(skill_name).join("tool.sh");
            if candidate.exists() {
                tool_path = Some(candidate);
                break;
            }
        }

        if let Some(path) = tool_path {
            messages.push(Message::ToolUse {
                tool: format!("skill:{skill_name}"),
                detail: format!("running tool.sh {skill_args}"),
            });

            match std::process::Command::new("bash")
                .arg(&path)
                .args(skill_args.split_whitespace())
                .output()
            {
                Ok(output) => {
                    let tool_output = String::from_utf8_lossy(&output.stdout).to_string();

                    // Route each block type
                    if let Some(raw) = extract_fenced_block(&tool_output, "raw") {
                        tui_log(&format!("[raw] {raw}"));
                        // raw goes to log only, not shown in TUI
                    }
                    if let Some(summary) = extract_fenced_block(&tool_output, "summary") {
                        messages.push(Message::System(summary.clone()));
                        bash_result = Some(summary); // feed back to LLM
                    }
                    if let Some(chart) = extract_fenced_block(&tool_output, "chartjs") {
                        messages.push(Message::ToolUse {
                            tool: "chartjs".into(),
                            detail: chart[..chart.len().min(80)].to_string(),
                        });
                        // TUI would render this as a chart widget
                    }
                }
                Err(e) => {
                    messages.push(Message::System(format!("skill error: {e}")));
                }
            }
        } else {
            messages.push(Message::System(format!("skill '{skill_name}' not found")));
        }
    }

    // Parse <tool>write_file</tool> <path>...</path> <content>...</content>
    if let Some(tool_start) = text.find("<tool>") {
        let after_tool = &text[tool_start + 6..];
        if let Some(tool_end) = after_tool.find("</tool>") {
            let tool_name = after_tool[..tool_end].trim();

            if tool_name == "write_file" {
                if let (Some(path), Some(content)) = (
                    extract_tag(text, "path"),
                    extract_tag(text, "content"),
                ) {
                    // Write the file
                    let workspace = std::env::current_dir().unwrap_or_default();
                    let full_path = workspace.join(&path);
                    if let Some(parent) = full_path.parent() {
                        std::fs::create_dir_all(parent).ok();
                    }
                    match std::fs::write(&full_path, &content) {
                        Ok(()) => {
                            messages.push(Message::ToolUse {
                                tool: "write_file".into(),
                                detail: format!("✓ wrote {} ({} bytes)", full_path.display(), content.len()),
                            });
                        }
                        Err(e) => {
                            messages.push(Message::ToolUse {
                                tool: "write_file".into(),
                                detail: format!("✗ failed: {e}"),
                            });
                        }
                    }
                }
            } else if tool_name == "bash" {
                if let Some(command) = extract_tag(text, "command") {
                    messages.push(Message::ToolUse {
                        tool: "bash".into(),
                        detail: format!("$ {command}"),
                    });
                    // Execute the command
                    match std::process::Command::new("sh")
                        .arg("-c")
                        .arg(&command)
                        .output()
                    {
                        Ok(output) => {
                            let stdout = String::from_utf8_lossy(&output.stdout);
                            let stderr = String::from_utf8_lossy(&output.stderr);
                            let mut result = String::new();
                            if !stdout.is_empty() {
                                result.push_str(&stdout);
                            }
                            if !stderr.is_empty() {
                                if !result.is_empty() { result.push('\n'); }
                                result.push_str(&stderr);
                            }
                            if result.is_empty() {
                                result = "(no output)".into();
                            }
                            // Truncate long output
                            if result.len() > 500 {
                                result.truncate(500);
                                result.push_str("\n...(truncated)");
                            }
                            messages.push(Message::System(result));
                        }
                        Err(e) => {
                            messages.push(Message::System(format!("bash error: {e}")));
                        }
                    }
                }
            }
        }
    }

    // Auto-detect and execute/save markdown code blocks
    let mut in_block = false;
    let mut lang = String::new();
    let mut code = String::new();
    // bash_result already set by skill tool call above (or None)
    for line in text.lines() {
        if line.trim_start().starts_with("```") && !in_block {
            in_block = true;
            lang = line.trim_start().trim_start_matches('`').trim().to_string();
            code.clear();
        } else if line.trim_start().starts_with("```") && in_block {
            in_block = false;
            if !code.trim().is_empty() {
                // Bash blocks: EXECUTE and return output for model feedback
                if lang == "bash" || lang == "sh" {
                    let cmd = code.trim();
                    messages.push(Message::ToolUse {
                        tool: "bash".into(),
                        detail: format!("$ {}", cmd.lines().next().unwrap_or(cmd)),
                    });
                    match std::process::Command::new("sh").arg("-c").arg(cmd).output() {
                        Ok(output) => {
                            let mut result = String::from_utf8_lossy(&output.stdout).to_string();
                            let err = String::from_utf8_lossy(&output.stderr);
                            if !err.is_empty() {
                                if !result.is_empty() { result.push('\n'); }
                                result.push_str(&err);
                            }
                            if result.is_empty() { result = "(no output)".into(); }
                            if result.len() > 1000 {
                                result.truncate(1000);
                                result.push_str("\n...(truncated)");
                            }
                            messages.push(Message::System(result.clone()));
                            bash_result = Some(result);
                        }
                        Err(e) => {
                            messages.push(Message::System(format!("error: {e}")));
                        }
                    }
                } else {
                    // Non-bash: save to file
                    let ext = match lang.as_str() {
                        "python" | "py" => "py",
                        "javascript" | "js" => "js",
                        "rust" | "rs" => "rs",
                        "html" => "html",
                        "css" => "css",
                        _ => "txt",
                    };
                    let filename = format!("output.{ext}");
                    let workspace = std::env::current_dir().unwrap_or_default();
                    let path = workspace.join(&filename);
                    match std::fs::write(&path, &code) {
                        Ok(()) => {
                            messages.push(Message::ToolUse {
                                tool: "write_file".into(),
                                detail: format!("✓ saved {} ({} bytes)", path.display(), code.len()),
                            });
                        }
                        Err(e) => {
                            messages.push(Message::ToolUse {
                                tool: "write_file".into(),
                                detail: format!("✗ {e}"),
                            });
                        }
                    }
                }
            }
        } else if in_block {
            code.push_str(line);
            code.push('\n');
        }
    }
    bash_result
}

/// Extract content from a fenced code block with a specific language tag.
/// Handles both multi-line (```tool\ncontent\n```) and single-line (```tool content```)
fn extract_fenced_block(text: &str, lang: &str) -> Option<String> {
    let open = format!("```{lang}");
    let start_pos = text.find(&open)?;
    let after_open = &text[start_pos + open.len()..];

    // Find the closing ``` FIRST (before deciding about newlines)
    let close_pos = after_open.find("```")?;
    let content = after_open[..close_pos].trim();

    // If content starts with newline, skip it
    let content = content.trim_start_matches('\n');
    if content.is_empty() { return None; }
    Some(content.to_string())
}

fn extract_tag(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{tag}>");
    let close = format!("</{tag}>");
    let start = text.find(&open)? + open.len();
    let end = text.find(&close)?;
    if end > start {
        Some(text[start..end].trim().to_string())
    } else {
        None
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
