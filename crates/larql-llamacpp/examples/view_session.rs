//! Session viewer — prints the conversation transcript that the TUI
//! would render, without ratatui.  Lets the operator (or an agent
//! debugging the TUI) inspect "what the user is currently seeing"
//! purely via stdout.
//!
//! Usage:
//!   cargo run --release -p larql-llamacpp --example view_session
//!   cargo run --release -p larql-llamacpp --example view_session -- <session-id>
//!   cargo run --release -p larql-llamacpp --example view_session -- --follow
//!
//! With no id, derives the session id from the cwd the way the TUI does
//! so `cd /path && view_session` always shows the right log.  --follow
//! `tail -f`s the file so new turns scroll past as they arrive.

use std::io::{Read, Seek, SeekFrom, Write};
use std::path::PathBuf;
use std::time::Duration;

fn home_dir() -> PathBuf {
    std::env::var("HOME").map(PathBuf::from).unwrap_or_else(|_| PathBuf::from("/tmp"))
}

fn default_session_id_from_cwd() -> String {
    let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("/"));
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
    if trimmed.is_empty() { "root".into() } else { trimmed }
}

fn session_path(id: &str) -> PathBuf {
    home_dir().join(".larql/sessions").join(format!("{id}.jsonl"))
}

/// Pretty-print one JSONL turn as the TUI would visually present it.
fn render_turn(line: &str, w: &mut impl Write) {
    let v: serde_json::Value = match serde_json::from_str(line) {
        Ok(v) => v,
        Err(_) => return,
    };
    let role = v["role"].as_str().unwrap_or("?");
    let content = v["content"].as_str().unwrap_or("");
    let ts = v["ts"].as_u64().unwrap_or(0);
    let when = if ts > 0 {
        let d = std::time::UNIX_EPOCH + std::time::Duration::from_secs(ts);
        // Coarse local-ish hh:mm:ss.
        let secs = d.duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs();
        format!("{:02}:{:02}:{:02}", (secs / 3600 - 4) % 24, (secs / 60) % 60, secs % 60)
    } else {
        "--:--:--".into()
    };

    let tag = match role {
        "user" => "you ❯".to_string(),
        "assistant" => "bot ❯".to_string(),
        "system" => "  [system]".to_string(),
        "tool_use" => "  ⚡ tool".to_string(),
        "tool_result" => "  └─ result".to_string(),
        "tool_render" => "  └─ render".to_string(),
        "tool_error" => "  └─ error".to_string(),
        "router" => "  ↪ router".to_string(),
        other => format!("  [{other}]"),
    };
    let meta_str = format_meta(&v["meta"]);
    let _ = writeln!(w, "[{when}] {tag}{meta_str}");
    for line in content.lines() {
        let _ = writeln!(w, "          {line}");
    }
    let _ = writeln!(w);
}

/// Pretty-print a meta blob as a `[k=v · k=v]` annotation suffix.
fn format_meta(meta: &serde_json::Value) -> String {
    if !meta.is_object() {
        return String::new();
    }
    let mut parts: Vec<String> = Vec::new();
    let dur = |ms: f64| -> String {
        if ms >= 1000.0 {
            format!("{:.2}s", ms / 1000.0)
        } else {
            format!("{:.0}ms", ms)
        }
    };
    if let Some(prefill) = meta["prefill_ms"].as_f64() {
        parts.push(format!("prefill={}", dur(prefill)));
    }
    if let Some(first) = meta["first_token_ms"].as_f64() {
        parts.push(format!("first_token={}", dur(first)));
    }
    if let Some(decode) = meta["decode_ms"].as_f64() {
        parts.push(format!("decode={}", dur(decode)));
    }
    if let Some(n) = meta["decoded_tokens"].as_u64() {
        parts.push(format!("tokens={n}"));
    }
    if let Some(tps) = meta["tok_per_sec"].as_f64() {
        if tps > 0.0 {
            parts.push(format!("{tps:.1}t/s"));
        }
    }
    if let Some(p_msgs) = meta["prompt_msgs"].as_u64() {
        if let Some(p_b) = meta["prompt_bytes"].as_u64() {
            parts.push(format!("prompt={p_msgs}msgs/{p_b}B"));
        }
    }
    if let Some(tool_ms) = meta["tool_ms"].as_u64() {
        parts.push(format!("tool={}", dur(tool_ms as f64)));
    }
    if let Some(b) = meta["bytes"].as_u64() {
        parts.push(format!("bytes={b}"));
    }
    if meta["knn_override"].as_bool().unwrap_or(false) {
        parts.push("knn=hit".into());
    }
    if let Some(skill) = meta["skill"].as_str() {
        if let Some(c) = meta["confidence"].as_f64() {
            parts.push(format!("skill={skill} conf={c:.2}"));
        } else {
            parts.push(format!("skill={skill}"));
        }
    }
    if let Some(n) = meta["n_skills"].as_u64() {
        parts.push(format!("of {n} skills"));
    }
    if parts.is_empty() {
        String::new()
    } else {
        format!("  [{}]", parts.join(" · "))
    }
}

fn print_all(path: &std::path::Path) -> std::io::Result<u64> {
    let data = std::fs::read_to_string(path)?;
    let mut stdout = std::io::stdout().lock();
    let _ = writeln!(stdout, "── {} ──", path.display());
    for line in data.lines() {
        if !line.trim().is_empty() {
            render_turn(line, &mut stdout);
        }
    }
    Ok(data.len() as u64)
}

fn follow(path: &std::path::Path, mut offset: u64) {
    use std::io::BufRead;
    eprintln!("\n── following {} ── (Ctrl-C to stop)\n", path.display());
    let mut buf = String::new();
    loop {
        let f = match std::fs::File::open(path) {
            Ok(f) => f,
            Err(_) => {
                std::thread::sleep(Duration::from_millis(500));
                continue;
            }
        };
        let mut f = f;
        let len = f.metadata().map(|m| m.len()).unwrap_or(0);
        if len < offset {
            // File was truncated (e.g. /v1/reset).  Restart.
            offset = 0;
        }
        if len > offset {
            f.seek(SeekFrom::Start(offset)).ok();
            buf.clear();
            f.read_to_string(&mut buf).ok();
            offset = len;
            let mut stdout = std::io::stdout().lock();
            for line in buf.lines() {
                if !line.trim().is_empty() {
                    render_turn(line, &mut stdout);
                }
            }
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let follow_mode = args.iter().any(|a| a == "--follow" || a == "-f");
    let id = args
        .iter()
        .find(|a| !a.starts_with('-'))
        .cloned()
        .unwrap_or_else(default_session_id_from_cwd);

    let path = session_path(&id);
    if !path.exists() {
        eprintln!("[view_session] no session at {}", path.display());
        eprintln!("[view_session] sessions on disk:");
        if let Ok(entries) = std::fs::read_dir(home_dir().join(".larql/sessions")) {
            for e in entries.flatten() {
                eprintln!("  {}", e.file_name().to_string_lossy());
            }
        }
        std::process::exit(1);
    }

    let offset = match print_all(&path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("[view_session] read {} failed: {e}", path.display());
            std::process::exit(1);
        }
    };

    if follow_mode {
        follow(&path, offset);
    }
}
