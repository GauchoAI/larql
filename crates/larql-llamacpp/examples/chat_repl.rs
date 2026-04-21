//! Tiny REPL that talks to a running larql-server's
//! `/v1/chat/completions` endpoint.  Maintains conversation history
//! locally so each new prompt sees the full context.
//!
//!   LARQL_SERVER=http://localhost:3000 \
//!     cargo run --release -p larql-llamacpp --example chat_repl
//!
//! Type a message, press enter; Ctrl-D or "/quit" to exit.

use std::io::{self, BufRead, Read, Write};
use std::time::Duration;

fn main() {
    let url = std::env::var("LARQL_SERVER")
        .unwrap_or_else(|_| "http://localhost:3000".into());
    eprintln!("[chat_repl] connected to {url}");
    eprintln!("[chat_repl] type a message; Ctrl-D or /quit to exit");
    eprintln!();

    let client = std::sync::Arc::new(
        ureq::AgentBuilder::new()
            .timeout_read(Duration::from_secs(120))
            .timeout_connect(Duration::from_secs(5))
            .build(),
    );

    // Conversation history as JSON-ready (role, content) pairs.
    let mut history: Vec<(String, String)> = Vec::new();

    let stdin = io::stdin();
    loop {
        eprint!("you ❯ ");
        let _ = io::stderr().flush();
        let mut line = String::new();
        match stdin.lock().read_line(&mut line) {
            Ok(0) => {
                eprintln!();
                break;
            }
            Ok(_) => {}
            Err(e) => {
                eprintln!("[chat_repl] read error: {e}");
                break;
            }
        }
        let user = line.trim();
        if user.is_empty() {
            continue;
        }
        if user == "/quit" || user == "/exit" {
            break;
        }
        if user == "/reset" {
            history.clear();
            eprintln!("[chat_repl] history cleared");
            continue;
        }

        history.push(("user".into(), user.to_string()));

        // Build OpenAI request body.
        let messages: Vec<serde_json::Value> = history
            .iter()
            .map(|(r, c)| serde_json::json!({"role": r, "content": c}))
            .collect();
        let body = serde_json::json!({
            "model": "gemma",
            "messages": messages,
            "stream": true,
            "max_tokens": 256,
        });

        eprint!("bot ❯ ");
        let _ = io::stderr().flush();

        let resp = match client
            .post(&format!("{url}/v1/chat/completions"))
            .set("content-type", "application/json")
            .send_string(&serde_json::to_string(&body).unwrap())
        {
            Ok(r) => r,
            Err(e) => {
                eprintln!("\n[chat_repl] request failed: {e}");
                history.pop();
                continue;
            }
        };

        let mut reader = resp.into_reader();
        let mut buf = Vec::with_capacity(8192);
        let mut chunk = [0u8; 1024];
        let mut assistant_text = String::new();
        let mut leftover = String::new();

        loop {
            match reader.read(&mut chunk) {
                Ok(0) => break,
                Ok(n) => {
                    buf.extend_from_slice(&chunk[..n]);
                    let s = String::from_utf8_lossy(&buf).into_owned();
                    buf.clear();
                    leftover.push_str(&s);
                }
                Err(e) => {
                    eprintln!("\n[chat_repl] read error: {e}");
                    break;
                }
            }
            while let Some(nl) = leftover.find('\n') {
                let raw_line = leftover[..nl].trim_end_matches('\r').to_string();
                leftover = leftover[nl + 1..].to_string();
                let line = raw_line.trim();
                if line.is_empty() || line.starts_with(':') {
                    continue;
                }
                if let Some(data) = line.strip_prefix("data: ") {
                    if data.trim() == "[DONE]" {
                        leftover.clear();
                        break;
                    }
                    if let Ok(v) = serde_json::from_str::<serde_json::Value>(data) {
                        if let Some(content) = v["choices"][0]["delta"]["content"].as_str() {
                            print!("{content}");
                            let _ = io::stdout().flush();
                            assistant_text.push_str(content);
                        }
                    }
                }
            }
        }
        println!();

        if !assistant_text.is_empty() {
            history.push(("assistant".into(), assistant_text));
        } else {
            history.pop();
        }
    }
    eprintln!("[chat_repl] bye");
}
