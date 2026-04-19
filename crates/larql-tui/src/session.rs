/// Load a Claude Code session (.jsonl) or TUI session into RAG facts.
/// Searches ~/.claude/projects/*/SESSION_ID.jsonl for Claude sessions.
pub async fn load_session_into_rag(server_url: &str, session_id: &str) -> Result<usize, String> {
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
    let content =
        std::fs::read_to_string(&path).map_err(|e| format!("read session: {e}"))?;

    let client = reqwest::Client::new();
    let mut inserted = 0usize;

    for line in content.lines() {
        let obj: serde_json::Value = match serde_json::from_str(line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let entry_type = obj.get("type").and_then(|v| v.as_str()).unwrap_or("");
        let ts = obj.get("timestamp").and_then(|v| v.as_str()).unwrap_or("");
        let ts_short = &ts[..ts.len().min(19)];

        // Only insert assistant messages — they contain the actual knowledge.
        // User messages are short/rambling and pollute the embedding space.
        let fact = match entry_type {
            "assistant" => {
                let blocks = obj.pointer("/message/content").and_then(|v| v.as_array());
                if let Some(blocks) = blocks {
                    let text: String = blocks
                        .iter()
                        .filter_map(|b| {
                            if b.get("type")?.as_str()? == "text" {
                                b.get("text")?.as_str().map(|s| s.to_string())
                            } else {
                                None
                            }
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    if text.len() > 50 {
                        Some(format!(
                            "[{}] {}",
                            ts_short,
                            &text[..text
                                .char_indices()
                                .take(300)
                                .last()
                                .map(|(i, c)| i + c.len_utf8())
                                .unwrap_or(0)]
                        ))
                    } else {
                        None
                    }
                } else {
                    None
                }
            }
            _ => None,
        };

        if let Some(fact_text) = fact {
            let resp = client
                .post(format!("{server_url}/v1/rag/insert"))
                .json(&serde_json::json!({
                    "fact": fact_text,
                    "category": "session",
                }))
                .timeout(std::time::Duration::from_secs(5))
                .send()
                .await;
            if resp.is_ok() {
                inserted += 1;
            }
        }
    }

    Ok(inserted)
}
