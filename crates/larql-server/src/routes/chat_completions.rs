//! POST /v1/chat/completions — OpenAI-compatible chat API with SSE streaming.
//!
//! Drop-in replacement for the OpenAI chat API. Any client that speaks
//! OpenAI format (ChatGPT apps, VSCode extensions, gaucho-code) can
//! connect to larql as a backend.
//!
//! Request:
//!   {"model": "gemma-3-4b", "messages": [{"role": "user", "content": "hello"}], "stream": true}
//!
//! Streaming response:
//!   data: {"id":"chatcmpl-1","object":"chat.completion.chunk","choices":[{"delta":{"content":"Hello"}}]}
//!   ...
//!   data: [DONE]

use std::sync::Arc;
use std::time::Instant;
use std::convert::Infallible;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use futures::stream::{self, Stream};
use serde::{Deserialize, Serialize};

use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct ChatRequest {
    #[serde(default)]
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(default = "default_true")]
    pub stream: bool,
    #[serde(default = "default_temperature")]
    pub temperature: f32,
    #[serde(default = "default_one")]
    pub top_p: f32,
    #[serde(default)]
    pub max_tokens: Option<usize>,
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

fn default_true() -> bool { true }
fn default_one() -> f32 { 1.0 }
fn default_temperature() -> f32 { 0.7 }

/// Convert OpenAI messages to Gemma 3 chat template.
fn messages_to_gemma3_prompt(messages: &[ChatMessage]) -> String {
    let system = "You are a local coding assistant running directly on the user's machine. \
        You have full access to their filesystem and can run bash commands. \
        Always give complete answers with working code.";

    let mut prompt = format!("<start_of_turn>system\n{system}<end_of_turn>\n");

    for msg in messages {
        match msg.role.as_str() {
            "system" => {
                // Override default system prompt
                prompt = format!("<start_of_turn>system\n{}<end_of_turn>\n", msg.content);
            }
            "user" => {
                prompt.push_str(&format!("<start_of_turn>user\n{}<end_of_turn>\n", msg.content));
            }
            "assistant" => {
                prompt.push_str(&format!("<start_of_turn>model\n{}<end_of_turn>\n", msg.content));
            }
            _ => {}
        }
    }
    prompt.push_str("<start_of_turn>model\n");
    prompt
}

pub async fn handle_chat_completions(
    State(state): State<Arc<AppState>>,
    Json(req): Json<ChatRequest>,
) -> Result<impl IntoResponse, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);

    if model.infer_disabled {
        return Err(ServerError::InferenceUnavailable("inference disabled".into()));
    }

    let max_tokens = req.max_tokens.unwrap_or(4096);

    // RAG: retrieve matching facts and inject as context
    let user_msg = req.messages.last().map(|m| m.content.as_str()).unwrap_or("");
    let rag_context = super::rag::retrieve_context(&state, &model, user_msg, 0.55);
    let chat_prompt = if let Some(ref ctx) = rag_context {
        tracing::info!("[chat] RAG injecting {} chars of context", ctx.len());
        // Inject RAG context as a system-level addition
        let mut msgs = req.messages.clone();
        if let Some(last) = msgs.last_mut() {
            last.content = format!("{ctx}\n\n{}", last.content);
        }
        messages_to_gemma3_prompt(&msgs)
    } else {
        messages_to_gemma3_prompt(&req.messages)
    };
    let request_id = format!("chatcmpl-{}", std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_millis());

    let prompt_preview: String = req.messages.last()
        .map(|m| m.content.chars().take(80).collect())
        .unwrap_or_default();
    tracing::info!("[chat] {} max_tokens={} prompt=\"{}{}\"",
        request_id, max_tokens, prompt_preview,
        if req.messages.last().map(|m| m.content.len()).unwrap_or(0) > 80 { "..." } else { "" });

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let model_cl = Arc::clone(&model);
    let req_id = request_id.clone();
    tokio::task::spawn_blocking(move || {
        let weights = match model_cl.get_or_load_weights() {
            Ok(w) => w,
            Err(e) => {
                let _ = tx.blocking_send(Ok(oai_error(&format!("weights: {e}"))));
                return;
            }
        };

        let encoding = match model_cl.tokenizer.encode(chat_prompt.as_str(), true) {
            Ok(e) => e,
            Err(e) => {
                let _ = tx.blocking_send(Ok(oai_error(&format!("tokenize: {e}"))));
                return;
            }
        };
        let ids: Vec<u32> = encoding.get_ids().to_vec();
        if ids.is_empty() {
            let _ = tx.blocking_send(Ok(oai_error("empty prompt")));
            return;
        }

        let backend = model_cl.get_or_init_backend();
        let _guard = match model_cl.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();

        let cache = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
        let patched = model_cl.patched.blocking_read();
        let knn_opt = if patched.knn_store.is_empty() { None } else { Some(&patched.knn_store) };

        let walk_ffn_opt = if model_cl.walk_only {
            Some(larql_inference::WalkFfn::new_with_backend(weights, patched.base(), 1024, &**backend))
        } else { None };
        let ffn_override: Option<&dyn larql_inference::ffn::FfnBackend> =
            walk_ffn_opt.as_ref().map(|w| w as &dyn larql_inference::ffn::FfnBackend);

        let mut sampler = larql_inference::sampling::Sampler::new(
            larql_inference::sampling::SamplingConfig {
                temperature: req.temperature,
                top_p: req.top_p,
                top_k: 0,
                seed: None,
            },
        );

        // Prefill
        let prefill = larql_inference::predict_honest_with_knn_ffn(
            weights, &model_cl.tokenizer, &ids, 20,
            patched.base(), &**backend, &cache,
            0..weights.num_layers, knn_opt, ffn_override,
        );

        // KNN override
        let first_label = prefill.predictions.first().map(|(s,_)| s.as_str()).unwrap_or("");
        if first_label.contains("KNN override") {
            let label = prefill.predictions.first().map(|(s,_)| s.clone()).unwrap_or_default();
            let _ = tx.blocking_send(Ok(oai_chunk(&req_id, &label)));
            let _ = tx.blocking_send(Ok(oai_done()));
            return;
        }

        let mut next = match sampler.sample(&prefill.raw_predictions) {
            Some(tid) => tid,
            None => {
                let _ = tx.blocking_send(Ok(oai_done()));
                return;
            }
        };
        let first_tok = model_cl.tokenizer.decode(&[next], true).unwrap_or_default();
        if tx.blocking_send(Ok(oai_chunk(&req_id, &first_tok))).is_err() { return; }

        // Decode loop
        let t_decode = std::time::Instant::now();
        let mut token_count = 1usize; // first token already sent
        for _step in 1..max_tokens {
            let input = vec![next];
            let r = larql_inference::predict_honest_with_knn_ffn(
                weights, &model_cl.tokenizer, &input, 20,
                patched.base(), &**backend, &cache,
                0..weights.num_layers, knn_opt, ffn_override,
            );

            // Multi-token KNN override: when the predict returns a full
            // target string (e.g. "tool list ."), tokenize it and force-
            // inject ALL tokens. Each forced token is fed through decode
            // to maintain the KV cache.
            if let Some(ref knn_target) = r.knn_override {
                let target_enc = model_cl.tokenizer.encode(
                    format!(" {}", knn_target).as_str(), false
                ).ok();
                let target_ids: Vec<u32> = target_enc
                    .map(|e| e.get_ids().to_vec())
                    .unwrap_or_default();
                tracing::info!("[chat] {} KNN multi-token override: {:?} ({} tokens)",
                    req_id, knn_target, target_ids.len());
                for &forced_tid in &target_ids {
                    let tok = model_cl.tokenizer.decode(&[forced_tid], true).unwrap_or_default();
                    if tx.blocking_send(Ok(oai_chunk(&req_id, &tok))).is_err() {
                        tracing::warn!("[chat] {} client disconnected during KNN inject", req_id);
                        return;
                    }
                    // Feed forced token through decode to update KV cache
                    let _r = larql_inference::predict_honest_with_knn_ffn(
                        weights, &model_cl.tokenizer, &[forced_tid], 1,
                        patched.base(), &**backend, &cache,
                        0..weights.num_layers, knn_opt, ffn_override,
                    );
                    next = forced_tid;
                    token_count += 1;
                }
                continue;
            }

            match sampler.sample(&r.raw_predictions) {
                Some(tid) => {
                    let tok_raw = model_cl.tokenizer.decode(&[tid], false).unwrap_or_default();
                    if tok_raw.contains("<end_of_turn>") || tok_raw.contains("<eos>")
                        || tok_raw.contains("</s>") || tid <= 1 {
                        break;
                    }
                    let tok = model_cl.tokenizer.decode(&[tid], true).unwrap_or_default();
                    if tx.blocking_send(Ok(oai_chunk(&req_id, &tok))).is_err() {
                        tracing::warn!("[chat] {} client disconnected at token {}", req_id, token_count);
                        return;
                    }
                    next = tid;
                    token_count += 1;
                }
                None => break,
            }
        }
        let decode_s = t_decode.elapsed().as_secs_f64();
        let tok_s = if decode_s > 0.0 { token_count as f64 / decode_s } else { 0.0 };
        tracing::info!("[chat] {} done: {} tokens in {:.1}s ({:.1} tok/s)",
            req_id, token_count, decode_s, tok_s);
        let _ = tx.blocking_send(Ok(oai_done()));
    });

    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|msg| (msg, rx))
    });
    let sse: Sse<_> = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok(sse)
}

fn oai_chunk(id: &str, content: &str) -> Event {
    let data = serde_json::json!({
        "id": id,
        "object": "chat.completion.chunk",
        "created": std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH).unwrap_or_default().as_secs(),
        "model": "gemma-3-4b",
        "choices": [{
            "index": 0,
            "delta": { "content": content },
            "finish_reason": serde_json::Value::Null,
        }]
    });
    Event::default().data(data.to_string())
}

fn oai_done() -> Event {
    Event::default().data("[DONE]")
}

fn oai_error(msg: &str) -> Event {
    Event::default().data(serde_json::json!({"error": {"message": msg}}).to_string())
}
