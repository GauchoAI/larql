use crate::app::{chrono_now, home_dir};
use crate::workflows::{StepState, Workflow, WorkflowStore};

pub fn extract_block(text: &str, lang: &str) -> Option<String> {
    let open = format!("```{lang}");
    let start = text.find(&open)?;
    let after = &text[start + open.len()..];
    let close = after.find("```")?;
    Some(after[..close].trim().to_string())
}

/// Extract ALL blocks of a given type (there may be multiple per response).
pub fn extract_all_blocks(text: &str, lang: &str) -> Vec<String> {
    let open = format!("```{lang}");
    let mut blocks = Vec::new();
    let mut search = text;
    while let Some(start) = search.find(&open) {
        let after = &search[start + open.len()..];
        if let Some(close) = after.find("```") {
            blocks.push(after[..close].trim().to_string());
            search = &after[close + 3..];
        } else {
            break;
        }
    }
    blocks
}

/// Extract self-annotated fact, status, and plan blocks from model output.
/// Persists facts to ~/.larql/facts.jsonl and status to ~/.larql/workflows.json.
/// Also strips the blocks from the displayed text.
pub fn process_annotations(
    text: &str,
    server_url: &str,
    workflows: &mut Vec<Workflow>,
    workflow_store: &WorkflowStore,
) -> String {
    let facts = extract_all_blocks(text, "fact");
    let statuses = extract_all_blocks(text, "status");
    let plans = extract_all_blocks(text, "plan");

    // Persist facts to local JSONL file
    if !facts.is_empty() {
        let facts_path = home_dir().join(".larql/facts.jsonl");
        if let Ok(mut f) = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&facts_path)
        {
            use std::io::Write;
            for fact_block in &facts {
                let mut key = String::new();
                let mut value = String::new();
                let mut source = String::from("derived");
                for line in fact_block.lines() {
                    let line = line.trim();
                    if let Some(rest) = line.strip_prefix("key:") {
                        key = rest.trim().to_string();
                    } else if let Some(rest) = line.strip_prefix("value:") {
                        value = rest.trim().to_string();
                    } else if let Some(rest) = line.strip_prefix("source:") {
                        source = rest.trim().to_string();
                    }
                }
                if !key.is_empty() && !value.is_empty() {
                    let entry = serde_json::json!({
                        "key": key, "value": value, "source": source,
                        "ts": chrono_now(),
                    });
                    let _ = writeln!(f, "{}", entry);
                    // Also POST to server RAG store
                    let url = server_url.to_string();
                    let fact = format!("{key}: {value}");
                    tokio::spawn(async move {
                        let _ = reqwest::Client::new()
                            .post(format!("{url}/v1/rag/insert"))
                            .json(&serde_json::json!({
                                "fact": fact,
                                "category": "fact",
                            }))
                            .send()
                            .await;
                    });
                }
            }
        }
    }

    // Persist workflow status
    if !statuses.is_empty() {
        for status_block in &statuses {
            let mut task = String::new();
            let mut state = String::new();
            let mut detail = String::new();
            for line in status_block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("task:") {
                    task = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("state:") {
                    state = rest.trim().to_string();
                } else if let Some(rest) = line.strip_prefix("detail:") {
                    detail = rest.trim().to_string();
                }
            }
            if !task.is_empty() {
                let step_state = match state.as_str() {
                    "active" => StepState::Active,
                    "done" => StepState::Done,
                    "blocked" => StepState::Blocked,
                    _ => StepState::Pending,
                };
                WorkflowStore::upsert_flat(workflows, &task, step_state, &detail);
            }
        }
        workflow_store.save(workflows);
    }

    // Process plan blocks
    if !plans.is_empty() {
        for plan_block in &plans {
            let mut name = String::new();
            let mut steps = Vec::new();
            for line in plan_block.lines() {
                let line = line.trim();
                if let Some(rest) = line.strip_prefix("workflow:") {
                    name = rest.trim().to_string();
                } else if line.starts_with(|c: char| c.is_ascii_digit()) {
                    if let Some(dot_pos) = line.find(". ") {
                        let desc = line[dot_pos + 2..].trim();
                        // Strip [pending] or [done] suffix if present
                        let desc = desc
                            .trim_end_matches(|c: char| c == ']')
                            .rsplit_once('[')
                            .map(|(d, _)| d.trim())
                            .unwrap_or(desc);
                        steps.push(desc.to_string());
                    }
                }
            }
            if !name.is_empty() && !steps.is_empty() {
                let wf = workflow_store.create_from_plan(&name, steps);
                workflows.push(wf);
                workflow_store.save(workflows);
            }
        }
    }

    // Strip annotation blocks from displayed text
    let mut cleaned = text.to_string();
    for lang in &["fact", "status", "plan"] {
        let open = format!("```{lang}");
        loop {
            if let Some(start) = cleaned.find(&open) {
                let after = &cleaned[start + open.len()..];
                if let Some(close) = after.find("```") {
                    let end = start + open.len() + close + 3;
                    cleaned.replace_range(start..end, "");
                    continue;
                }
            }
            break;
        }
    }
    cleaned.trim().to_string()
}
