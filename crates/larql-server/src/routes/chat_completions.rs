//! POST /v1/chat/completions — OpenAI-compatible streaming endpoint.
//!
//! Accepts `{model, messages: [{role, content}], stream: bool}` and
//! streams back SSE chunks shaped like
//! `data: {"choices":[{"delta":{"content":"..."}}]}` followed by
//! `data: [DONE]`.  Routes through the llama.cpp pipeline + chat
//! template.

use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;

use axum::Json;
use axum::extract::State;
use axum::http::HeaderMap;
use axum::response::IntoResponse;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures::stream::{self, Stream};
use serde::Deserialize;

use crate::chat_log;
use crate::error::ServerError;
use crate::llama_probe::{snapshot_layer, Mode};
use crate::state::AppState;

#[derive(Deserialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct ChatCompletionRequest {
    #[serde(default)]
    pub model: Option<String>,
    pub messages: Vec<ChatMessage>,
    #[serde(default)]
    pub stream: bool,
    #[serde(default = "default_max_tokens")]
    pub max_tokens: usize,
    #[serde(default)]
    pub temperature: Option<f32>,
    #[serde(default)]
    pub top_p: Option<f32>,
    #[serde(default)]
    pub seed: Option<u64>,
}

fn default_max_tokens() -> usize {
    256
}

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    headers: HeaderMap,
    Json(req): Json<ChatCompletionRequest>,
) -> Result<impl IntoResponse, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);

    if model.llama.is_none() {
        return Err(ServerError::InferenceUnavailable(
            "no weights.gguf in vindex dir — drop one in to enable inference".into(),
        ));
    }

    let session_id = headers
        .get("x-session-id")
        .and_then(|v| v.to_str().ok())
        .map(|s| s.to_string());

    // Diagnostic: log the size + role sequence of every chat request
    // so we can see what the TUI actually sends.
    let role_summary: String = req
        .messages
        .iter()
        .map(|m| format!("{}({})", &m.role[..3.min(m.role.len())], m.content.len()))
        .collect::<Vec<_>>()
        .join(" ");
    let total_bytes: usize = req.messages.iter().map(|m| m.content.len()).sum();
    tracing::info!(
        "chat: session={} msgs={} bytes={} roles=[{}]",
        session_id.as_deref().unwrap_or("-"),
        req.messages.len(),
        total_bytes,
        role_summary,
    );

    // Persist the user's latest turn before we start generating, so the
    // session is still partially captured if the client disconnects
    // mid-stream.  Only the last `user` turn is logged — system primers
    // and earlier user turns are already on disk from prior requests.
    if let Some(sid) = session_id.as_deref() {
        if let Some(last_user) = req.messages.iter().rfind(|m| m.role == "user") {
            chat_log::append_turn(sid, "user", &last_user.content);
        }
    }

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(64);
    let model_cl = Arc::clone(&model);
    let session_log_id = session_id.clone();
    tokio::task::spawn_blocking(move || {
        run_chat(&model_cl, req, &tx, session_log_id.as_deref());
    });

    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|msg| (msg, rx))
    });
    let sse: Sse<_> = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok(sse)
}

fn run_chat(
    model: &Arc<crate::state::LoadedModel>,
    req: ChatCompletionRequest,
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
    session_id: Option<&str>,
) {
    let req_start = Instant::now();
    let prompt_msgs = req.messages.len();
    let prompt_bytes: usize = req.messages.iter().map(|m| m.content.len()).sum();

    // Accumulator for what we eventually persist as the assistant turn.
    let mut assistant_buf = String::new();
    // Per-turn timing the handler updates as it progresses.  Logged
    // alongside the assistant turn so view_session can display
    // "[prefill 412ms · first 178ms · decode 1.2s · 87 tok @ 72 t/s]".
    #[allow(unused_assignments)]
    let mut timing = ChatTiming {
        prompt_msgs,
        prompt_bytes,
        prefill_ms: 0.0,
        first_token_ms: 0.0,
        decode_ms: 0.0,
        decoded_tokens: 0,
        knn_override: false,
    };
    let pipe_mu = model.llama.as_ref().expect("llama present");

    // Build prompt from messages via the model's chat template.
    let messages: Vec<(String, String)> = req
        .messages
        .iter()
        .map(|m| (m.role.clone(), m.content.clone()))
        .collect();

    let prompt = {
        let pipe = match pipe_mu.lock() {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.blocking_send(Ok(sse_err(&format!("pipeline lock: {e}"))));
                return;
            }
        };
        match pipe.apply_chat_template(&messages) {
            Ok(p) => p,
            Err(e) => {
                let _ = tx.blocking_send(Ok(sse_err(&format!("chat template: {e}"))));
                return;
            }
        }
    };

    // Snapshot KNN entries at the install layer for one-shot override.
    let install_layer = model.config.num_layers.saturating_sub(8);
    let tensor = format!("attn_post_norm-{install_layer}");
    // Higher threshold than /v1/infer because chat sessions tend to
    // accumulate many fact-derived KNN entries and 0.75 starts firing
    // false-positive overrides on unrelated prompts.  0.90 gives us
    // crisp recall on real matches without the spam.
    const THRESHOLD: f32 = 0.90;

    let patched = model.patched.blocking_read();
    let entries = snapshot_layer(&patched.knn_store, install_layer);
    drop(patched);

    {
        let mut s = model.probe_state.lock().unwrap();
        s.mode = Mode::KnnQuery {
            layer: install_layer as u32,
            tensor_name: tensor,
            threshold: THRESHOLD,
            entries,
            fired: false,
            forced: None,
            result: None,
            best_cosine: None,
        };
    }

    // Sampling.
    let mut sampler = larql_llamacpp::Sampler::new(larql_llamacpp::SamplingConfig {
        temperature: req.temperature.unwrap_or(0.0),
        top_p: req.top_p.unwrap_or(1.0),
        top_k: 0,
        seed: req.seed,
    });

    let mut pipe = match pipe_mu.lock() {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.blocking_send(Ok(sse_err(&format!("pipeline lock: {e}"))));
            return;
        }
    };
    pipe.reset_kv();

    let prefill_start = Instant::now();
    let n_prefill = match pipe.prefill(&prompt) {
        Ok(n) => n,
        Err(e) => {
            let _ = tx.blocking_send(Ok(sse_err(&format!("prefill: {e}"))));
            return;
        }
    };
    timing.prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;
    if n_prefill == 0 {
        let _ = tx.blocking_send(Ok(sse_err("empty prompt")));
        return;
    }

    // Honour KNN overlay: if the probe matched, emit the override and stop.
    let knn_hit = {
        let mut s = model.probe_state.lock().unwrap();
        let hit = if let Mode::KnnQuery { result, .. } = &mut s.mode {
            result.take()
        } else {
            None
        };
        s.mode = Mode::Idle;
        hit
    };
    if let Some(m) = knn_hit {
        let label = format!(
            "{} (KNN override, cos={:.2}, L{install_layer})",
            m.target_token, m.cosine
        );
        let _ = tx.blocking_send(Ok(sse_delta(&label)));
        let _ = tx.blocking_send(Ok(sse_done_openai()));
        timing.knn_override = true;
        timing.first_token_ms = req_start.elapsed().as_secs_f64() * 1000.0;
        log_assistant_with_timing(&label, session_id, &timing);
        return;
    }

    let first_preds = pipe.top_k_at((n_prefill - 1) as i32, 64);
    let mut next: i32 = match sampler.sample(&first_preds) {
        Some(t) => t as i32,
        None => {
            let _ = tx.blocking_send(Ok(sse_done_openai()));
            return;
        }
    };

    // Stop tokens: EOS plus Gemma's <end_of_turn> (id 106 on the Gemma vocab).
    let stop_ids: [i32; 2] = [1, 106];

    timing.first_token_ms = req_start.elapsed().as_secs_f64() * 1000.0;
    let decode_start = Instant::now();
    let first_tok = pipe.decode_token(next);
    if !first_tok.is_empty() {
        assistant_buf.push_str(&first_tok);
        timing.decoded_tokens += 1;
        if tx.blocking_send(Ok(sse_delta(&first_tok))).is_err() {
            timing.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            log_assistant_with_timing(&assistant_buf, session_id, &timing);
            return;
        }
    }

    for _ in 1..req.max_tokens {
        if stop_ids.contains(&next) {
            break;
        }
        if let Err(e) = pipe.feed(next) {
            let _ = tx.blocking_send(Ok(sse_err(&format!("decode: {e}"))));
            return;
        }
        let preds = pipe.top_k_at(0, 64);
        next = match sampler.sample(&preds) {
            Some(t) if stop_ids.contains(&(t as i32)) => break,
            Some(t) => t as i32,
            None => break,
        };
        let tok = pipe.decode_token(next);
        if tok.is_empty() {
            continue;
        }
        assistant_buf.push_str(&tok);
        timing.decoded_tokens += 1;
        if tx.blocking_send(Ok(sse_delta(&tok))).is_err() {
            timing.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
            log_assistant_with_timing(&assistant_buf, session_id, &timing);
            return;
        }
    }

    let _ = tx.blocking_send(Ok(sse_done_openai()));
    timing.decode_ms = decode_start.elapsed().as_secs_f64() * 1000.0;
    log_assistant_with_timing(&assistant_buf, session_id, &timing);
}

#[derive(Clone, Debug)]
struct ChatTiming {
    prompt_msgs: usize,
    prompt_bytes: usize,
    prefill_ms: f64,
    first_token_ms: f64,
    decode_ms: f64,
    decoded_tokens: usize,
    knn_override: bool,
}

fn log_assistant_with_timing(text: &str, session_id: Option<&str>, t: &ChatTiming) {
    let sid = match session_id {
        Some(s) => s,
        None => return,
    };
    if text.is_empty() {
        return;
    }
    let tps = if t.decode_ms > 0.0 {
        (t.decoded_tokens as f64) * 1000.0 / t.decode_ms
    } else {
        0.0
    };
    let meta = serde_json::json!({
        "kind": "chat",
        "prompt_msgs": t.prompt_msgs,
        "prompt_bytes": t.prompt_bytes,
        "prefill_ms": (t.prefill_ms * 10.0).round() / 10.0,
        "first_token_ms": (t.first_token_ms * 10.0).round() / 10.0,
        "decode_ms": (t.decode_ms * 10.0).round() / 10.0,
        "decoded_tokens": t.decoded_tokens,
        "tok_per_sec": (tps * 10.0).round() / 10.0,
        "knn_override": t.knn_override,
    });
    chat_log::append_turn_with_meta(sid, "assistant", text, meta);
    tracing::info!(
        "chat_done: prompt_msgs={} prompt_bytes={} prefill={:.0}ms first_token={:.0}ms decode={:.0}ms tokens={} {:.1}tok/s knn={}",
        t.prompt_msgs, t.prompt_bytes,
        t.prefill_ms, t.first_token_ms, t.decode_ms,
        t.decoded_tokens, tps, t.knn_override
    );
}

/// Build an OpenAI-compatible delta SSE chunk for a single token.
fn sse_delta(content: &str) -> Event {
    Event::default().data(
        serde_json::to_string(&serde_json::json!({
            "choices": [
                {"delta": {"content": content}, "index": 0, "finish_reason": null}
            ]
        }))
        .unwrap_or_default(),
    )
}

/// Final SSE chunk: per spec, `data: [DONE]`.
fn sse_done_openai() -> Event {
    Event::default().data("[DONE]")
}

fn sse_err(msg: &str) -> Event {
    Event::default().event("error").json_data(serde_json::json!({"error": msg}))
        .unwrap_or_else(|_| Event::default().data(msg))
}

#[allow(dead_code)]
fn _stream_type_check<S: Stream<Item = Result<Event, Infallible>>>() {}
