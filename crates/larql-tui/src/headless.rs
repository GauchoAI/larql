use std::io;

use tokio::io::{AsyncBufReadExt, BufReader};

use crate::annotations;
use crate::app::{AppState, Message, StreamEvent};
use crate::stream::spawn_chat;

/// Headless mode: stdin -> server -> stdout. No TUI.
pub async fn run_headless(server_url: &str) -> io::Result<()> {
    let mut state = AppState::new(server_url);

    // Insert stored facts into model via KNN overlay
    state.insert_facts_into_model().await;

    let stdin = BufReader::new(tokio::io::stdin());
    let mut lines = stdin.lines();

    while let Ok(Some(input)) = lines.next_line().await {
        if input.is_empty() {
            continue;
        }

        state.logger.log("user", &input, None);
        state.messages.push(Message::User(input.clone()));

        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);
        let chat_msgs = state.build_chat_messages();
        spawn_chat(server_url.to_string(), chat_msgs, tx);

        let mut response = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Token(t) => response.push_str(&t),
                StreamEvent::Done => break,
                StreamEvent::Error(e) => {
                    eprintln!("error: {e}");
                    break;
                }
            }
        }

        let cleaned = annotations::process_annotations(
            &response,
            server_url,
            &mut state.workflows,
            &state.workflow_store,
        );
        state.logger.log("assistant", &response, None);

        println!("{cleaned}");

        state.messages.push(Message::Assistant(cleaned));
    }
    Ok(())
}

/// Scenario mode: run test scenarios from a JSON file.
pub async fn run_scenario(server_url: &str, scenario_path: &str) -> io::Result<()> {
    let content = std::fs::read_to_string(scenario_path)
        .map_err(|e| io::Error::new(io::ErrorKind::NotFound, format!("scenario file: {e}")))?;

    let scenarios: Vec<serde_json::Value> = serde_json::from_str(&content)
        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, format!("parse scenario: {e}")))?;

    let mut state = AppState::new(server_url);
    let mut passed = 0usize;
    let mut failed = 0usize;
    let total = scenarios.len();

    println!("Running {} scenario steps against {server_url}", total);
    println!();

    let client = reqwest::Client::new();

    for (i, scenario) in scenarios.iter().enumerate() {
        // Run setup commands (INSERT facts for KNN override tests)
        if let Some(setup) = scenario["setup"].as_array() {
            for cmd in setup {
                if let Some(insert) = cmd.get("insert") {
                    let resp = client
                        .post(format!("{server_url}/v1/insert"))
                        .json(insert)
                        .timeout(std::time::Duration::from_secs(10))
                        .send()
                        .await;
                    // Wait for insert to complete (KV cache needs to be ready)
                    tokio::time::sleep(std::time::Duration::from_millis(100)).await;
                    match resp {
                        Ok(r) if r.status().is_success() => {
                            let entity = insert["entity"].as_str().unwrap_or("?");
                            let target = insert["target"].as_str().unwrap_or("?");
                            println!("  setup: INSERT {entity} → {target}");
                        }
                        Ok(r) => eprintln!("  setup: INSERT failed: {}", r.status()),
                        Err(e) => eprintln!("  setup: INSERT error: {e}"),
                    }
                }
            }
            println!();
        }

        let input = scenario["input"].as_str().unwrap_or("");
        if input.is_empty() {
            println!("  SKIP step {} -- no input", i + 1);
            continue;
        }

        // Reset KV cache between scenario steps for deterministic matching
        let _ = client
            .post(format!("{server_url}/v1/kv/precompute"))
            .json(&serde_json::json!({"text": ""}))
            .timeout(std::time::Duration::from_secs(5))
            .send()
            .await;

        state.messages.push(Message::User(input.to_string()));

        let (tx, mut rx) = tokio::sync::mpsc::channel::<StreamEvent>(256);
        let chat_msgs = state.build_chat_messages();
        spawn_chat(server_url.to_string(), chat_msgs, tx);

        let mut response = String::new();
        while let Some(ev) = rx.recv().await {
            match ev {
                StreamEvent::Token(t) => response.push_str(&t),
                StreamEvent::Done => break,
                StreamEvent::Error(e) => {
                    eprintln!("    error: {e}");
                    break;
                }
            }
        }

        let cleaned = annotations::process_annotations(
            &response,
            server_url,
            &mut state.workflows,
            &state.workflow_store,
        );
        state.messages.push(Message::Assistant(cleaned.clone()));

        // Check assertions
        let expect = &scenario["expect"];
        let mut step_ok = true;
        let mut details = Vec::new();

        // Check "contains" assertion
        if let Some(needle) = expect["contains"].as_str() {
            if cleaned.to_lowercase().contains(&needle.to_lowercase()) {
                details.push(format!("  contains \"{needle}\": PASS"));
            } else {
                details.push(format!("  contains \"{needle}\": FAIL (not found)"));
                step_ok = false;
            }
        }

        // Check "not_contains" assertions (any key starting with "not_contains")
        if let Some(obj) = expect.as_object() {
            for (key, val) in obj {
                if key.starts_with("not_contains") {
                    if let Some(needle) = val.as_str() {
                        if cleaned.to_lowercase().contains(&needle.to_lowercase()) {
                            details.push(format!("  not_contains \"{needle}\": FAIL (found!)"));
                            step_ok = false;
                        } else {
                            details.push(format!("  not_contains \"{needle}\": PASS"));
                        }
                    }
                }
            }
        }

        // Check "facts" assertions
        if let Some(fact_checks) = expect["facts"].as_array() {
            let facts_path = crate::app::home_dir().join(".larql/facts.jsonl");
            let facts_content = std::fs::read_to_string(&facts_path).unwrap_or_default();
            for fc in fact_checks {
                let key_contains = fc["key_contains"].as_str().unwrap_or("");
                let value_contains = fc["value_contains"].as_str().unwrap_or("");
                let found = facts_content.lines().any(|line| {
                    let lower = line.to_lowercase();
                    (key_contains.is_empty()
                        || lower.contains(&key_contains.to_lowercase()))
                        && (value_contains.is_empty()
                            || lower.contains(&value_contains.to_lowercase()))
                });
                if found {
                    details.push(format!(
                        "  fact key~\"{key_contains}\" val~\"{value_contains}\": PASS"
                    ));
                } else {
                    details.push(format!(
                        "  fact key~\"{key_contains}\" val~\"{value_contains}\": FAIL"
                    ));
                    step_ok = false;
                }
            }
        }

        // Check "workflows" assertions
        if let Some(wf_checks) = expect["workflows"].as_array() {
            for wc in wf_checks {
                let name_contains = wc["name_contains"].as_str().unwrap_or("");
                let min_steps = wc["min_steps"].as_u64().unwrap_or(0) as usize;
                let found = state.workflows.iter().any(|w| {
                    let name_ok = name_contains.is_empty()
                        || w.name
                            .to_lowercase()
                            .contains(&name_contains.to_lowercase());
                    let steps_ok = w.steps.len() >= min_steps;
                    name_ok && steps_ok
                });
                if found {
                    details.push(format!(
                        "  workflow~\"{name_contains}\" (>={min_steps} steps): PASS"
                    ));
                } else {
                    details.push(format!(
                        "  workflow~\"{name_contains}\" (>={min_steps} steps): FAIL"
                    ));
                    step_ok = false;
                }
            }
        }

        let status = if step_ok { "PASS" } else { "FAIL" };
        if step_ok {
            passed += 1;
        } else {
            failed += 1;
        }

        println!(
            "[{status}] Step {}: \"{}\"",
            i + 1,
            input.chars().take(60).collect::<String>()
        );
        for d in &details {
            println!("{d}");
        }

        // Show truncated response for context
        let preview: String = cleaned.chars().take(100).collect();
        println!(
            "    response: \"{preview}{}\"",
            if cleaned.len() > 100 { "..." } else { "" }
        );
        println!();
    }

    println!("===================================");
    println!("Results: {passed}/{total} passed, {failed} failed");

    if failed > 0 {
        std::process::exit(1);
    }
    Ok(())
}
