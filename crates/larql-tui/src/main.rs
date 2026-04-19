//! larql TUI — ratatui terminal interface powered by HTTP API.
//!
//! Connects to larql-server at /v1/chat/completions (OpenAI format).
//! Server runs separately — start once, TUI connects instantly.
//! Skills loaded from ~/.larql/skills/ and ./.skills/

mod annotations;
mod app;
mod draw;
mod events;
mod headless;
mod log;
mod session;
mod skills;
mod stream;
mod workflows;

use std::io;

use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use app::{AppState, Message, StreamEvent};
use draw::draw;

#[tokio::main]
async fn main() -> io::Result<()> {
    let server_url =
        std::env::var("LARQL_SERVER").unwrap_or_else(|_| "http://localhost:3000".into());

    // Parse arguments
    let args: Vec<String> = std::env::args().collect();
    let session_id = args
        .iter()
        .position(|a| a == "--session")
        .and_then(|i| args.get(i + 1))
        .cloned();

    let is_headless = args.contains(&"--headless".to_string());
    let scenario_path = args
        .iter()
        .position(|a| a == "--scenario")
        .and_then(|i| args.get(i + 1))
        .cloned();

    // Headless mode: stdin -> server -> stdout, no TUI
    if is_headless {
        return headless::run_headless(&server_url).await;
    }

    // Scenario mode: run test scenarios from JSON file
    if let Some(ref path) = scenario_path {
        return headless::run_scenario(&server_url, path).await;
    }

    enable_raw_mode()?;
    let mut stdout = io::stdout();
    execute!(stdout, EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    let mut state = AppState::new(&server_url);
    draw(&mut terminal, &state);

    // Check server health
    match reqwest::Client::new()
        .get(format!("{server_url}/v1/health"))
        .timeout(std::time::Duration::from_secs(3))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => {
            state.status = format!("connected · {server_url}");
            state.messages.push(Message::System("Server connected.".into()));
        }
        _ => {
            state.status = "server not reachable — start larql-server first".into();
            state.messages.push(Message::System(format!(
                "Cannot reach {server_url}. Start the server:\n  cargo run --release -p larql-server -- /path/to/vindex"
            )));
        }
    }
    draw(&mut terminal, &state);

    // Insert stored facts into model via KNN overlay
    let facts_path = app::home_dir().join(".larql/facts.jsonl");
    if facts_path.exists() {
        state.status = "inserting facts into model...".into();
        draw(&mut terminal, &state);
        state.insert_facts_into_model().await;
        let n = std::fs::read_to_string(&facts_path)
            .map(|c| c.lines().count()).unwrap_or(0);
        if n > 0 {
            state.messages.push(Message::System(format!("{n} facts inserted into model")));
            state.status = format!("connected · {server_url} · {n} facts loaded");
        }
    }
    draw(&mut terminal, &state);

    // Load session into RAG if --session provided
    if let Some(ref sid) = session_id {
        state.status = format!("loading session {sid}...");
        state
            .messages
            .push(Message::System(format!("Loading session: {sid}")));
        draw(&mut terminal, &state);

        match session::load_session_into_rag(&server_url, sid).await {
            Ok(n) => {
                state.status = format!("connected · {server_url} · {n} facts loaded");
                state.messages.push(Message::System(format!(
                    "Loaded {n} facts from session. Ask me anything about the conversation."
                )));
            }
            Err(e) => {
                state
                    .messages
                    .push(Message::System(format!("Session load failed: {e}")));
            }
        }
        draw(&mut terminal, &state);
    }

    let (ev_tx, ev_rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);

    events::run_event_loop(&mut terminal, &mut state, ev_tx, ev_rx).await?;

    disable_raw_mode()?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)?;
    terminal.show_cursor()?;

    Ok(())
}
