use std::io;

use crossterm::event::{self, Event as CEvent, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::backend::CrosstermBackend;
use ratatui::Terminal;

use crate::annotations::process_annotations;
use crate::app::{AppState, Message, StreamEvent};
use crate::draw::draw;
use crate::skills::execute_skill_tool;
use crate::stream::spawn_chat;

pub async fn run_event_loop(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    state: &mut AppState,
    ev_tx: tokio::sync::mpsc::Sender<StreamEvent>,
    mut ev_rx: tokio::sync::mpsc::Receiver<StreamEvent>,
) -> io::Result<()> {
    let mut tool_depth: usize = 0; // prevent infinite tool execution chains

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

                    // Extract and persist self-annotated fact/status/plan blocks
                    let cleaned = process_annotations(
                        &response_text,
                        &state.server_url,
                        &mut state.workflows,
                        &state.workflow_store,
                    );
                    if cleaned != response_text {
                        // Replace the assistant message with the cleaned version
                        if let Some(Message::Assistant(ref mut text)) = state.messages.last_mut() {
                            *text = cleaned;
                        }
                    }

                    // Log assistant response
                    state.logger.log("assistant", &response_text, None);

                    // Only execute tools on first response, not follow-ups
                    if tool_depth == 0 {
                        if let Some(_summary) =
                            execute_skill_tool(&response_text, &mut state.messages)
                        {
                            tool_depth += 1;
                            state.messages.push(Message::Assistant(String::new()));
                            let chat_msgs = state.build_chat_messages();
                            spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone());
                        } else {
                            state.is_generating = false;
                            tool_depth = 0;
                        }
                    } else {
                        state.is_generating = false;
                        tool_depth = 0;

                        // Auto-INSERT conversation turn into RAG store.
                        // Both user message and assistant response become
                        // retrievable facts — no context window growth.
                        let url = state.server_url.clone();
                        let user_msg = state
                            .messages
                            .iter()
                            .rev()
                            .find_map(|m| match m {
                                Message::User(t) => Some(t.clone()),
                                _ => None,
                            })
                            .unwrap_or_default();
                        let asst_msg = response_text.chars().take(500).collect::<String>();
                        tokio::spawn(async move {
                            let client = reqwest::Client::new();
                            // INSERT user message
                            if !user_msg.is_empty() {
                                let _ = client
                                    .post(format!("{url}/v1/rag/insert"))
                                    .json(&serde_json::json!({
                                        "fact": format!("User asked: {}", user_msg.chars().take(300).collect::<String>()),
                                        "category": "conversation",
                                    }))
                                    .send()
                                    .await;
                            }
                            // INSERT assistant response
                            if !asst_msg.is_empty() {
                                let _ = client
                                    .post(format!("{url}/v1/rag/insert"))
                                    .json(&serde_json::json!({
                                        "fact": format!("Assistant answered: {}", asst_msg.chars().take(300).collect::<String>()),
                                        "category": "conversation",
                                    }))
                                    .send()
                                    .await;
                            }
                        });
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
            draw(terminal, state);
        }

        if event::poll(std::time::Duration::from_millis(30))? {
            if let CEvent::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char('b') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                        state.sidebar_visible = !state.sidebar_visible;
                        draw(terminal, state);
                    }
                    KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Char('q') if key.modifiers.contains(KeyModifiers::CONTROL) => break,
                    KeyCode::Enter if !state.is_generating => {
                        let input = state.input.trim().to_string();
                        if input.is_empty() {
                            continue;
                        }
                        state.input.clear();
                        state.cursor = 0;

                        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
                            break;
                        }

                        // Log user input
                        state.logger.log("user", &input, None);

                        state.messages.push(Message::User(input));
                        state.is_generating = true;
                        state.messages.push(Message::Assistant(String::new()));
                        draw(terminal, state);

                        // Send full conversation history
                        let chat_msgs = state.build_chat_messages();
                        spawn_chat(state.server_url.clone(), chat_msgs, ev_tx.clone());
                    }
                    KeyCode::Char(c) if !state.is_generating => {
                        state.input.insert(state.cursor, c);
                        state.cursor += 1;
                        draw(terminal, state);
                    }
                    KeyCode::Backspace if !state.is_generating && state.cursor > 0 => {
                        state.cursor -= 1;
                        state.input.remove(state.cursor);
                        draw(terminal, state);
                    }
                    KeyCode::Left if state.cursor > 0 => {
                        state.cursor -= 1;
                        draw(terminal, state);
                    }
                    KeyCode::Right if state.cursor < state.input.len() => {
                        state.cursor += 1;
                        draw(terminal, state);
                    }
                    _ => {}
                }
            }
        }
    }

    Ok(())
}
