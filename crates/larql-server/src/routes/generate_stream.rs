//! POST /v1/generate — Server-Sent Events token streaming.
//!
//! Client POSTs JSON:
//!   {"prompt": "...", "n_tokens": 100, "mode": "fast",
//!    "temperature": 0.8, "top_p": 0.9, "top_k": 0, "seed": 42}
//!
//! Server streams:
//!   event: token
//!   data: {"token": "Hello", "tid": 9876, "step": 0}
//!   ...
//!   event: done
//!   data: {"stopped_on": "eos", "decoded_tokens": 12, "prefill_ms": 180.3, "avg_decode_ms": 95.2}
//!
//! Client can abort the HTTP connection to stop generation mid-stream.

use std::sync::Arc;
use std::time::Instant;
use std::convert::Infallible;

use axum::Json;
use axum::extract::State;
use axum::response::sse::{Event, KeepAlive, Sse};
use axum::response::IntoResponse;
use futures::stream::{self, Stream};
use serde::Deserialize;

use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
#[allow(dead_code)]
pub struct GenerateRequest {
    pub prompt: String,
    #[serde(default = "default_n")]
    pub n_tokens: usize,
    #[serde(default = "default_mode")]
    pub mode: String,
    #[serde(default)]
    pub temperature: f32,
    #[serde(default = "default_one")]
    pub top_p: f32,
    #[serde(default)]
    pub top_k: usize,
    #[serde(default)]
    pub seed: Option<u64>,
    #[serde(default)]
    pub stop_ids: Option<Vec<u32>>,
}

fn default_n() -> usize { 100 }
fn default_mode() -> String { "fast".into() }
fn default_one() -> f32 { 1.0 }

pub async fn handle_generate_stream(
    State(state): State<Arc<AppState>>,
    Json(req): Json<GenerateRequest>,
) -> Result<impl IntoResponse, ServerError> {
    state.bump_requests();
    let model = state
        .model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);

    // Require GGUF — the only weight source.
    if model.gguf.is_none() {
        return Err(ServerError::InferenceUnavailable(
            "no weights.gguf in vindex dir — drop one in to enable inference".into(),
        ));
    }

    let stop_ids: Vec<u32> = req.stop_ids.clone().unwrap_or_else(|| vec![1u32, 106]);

    // Produce tokens in a blocking task, forward each one as an SSE event.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);

    let model_cl = Arc::clone(&model);
    tokio::task::spawn_blocking(move || {
        run_gguf_generate(&model_cl, &req, &stop_ids, &tx);
    });

    // Consume the channel into an SSE stream.
    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|msg| (msg, rx))
    });
    let sse: Sse<_> = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok(sse)
}

/// SSE generate loop driven by GgufPipeline. Bypasses vindex weights
/// entirely; consults vindex KNN store for fact overlay only.
fn run_gguf_generate(
    model: &Arc<crate::state::LoadedModel>,
    req: &GenerateRequest,
    stop_ids: &[u32],
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    use larql_inference::gguf_pipeline::DecodeSession;

    let gguf = match model.gguf.as_ref() {
        Some(g) => g,
        None => {
            let _ = tx.blocking_send(Ok(sse_err("no GGUF source")));
            return;
        }
    };

    let encoding = match model.tokenizer.encode(req.prompt.as_str(), true) {
        Ok(e) => e,
        Err(e) => {
            let _ = tx.blocking_send(Ok(sse_err(&format!("tokenize: {e}"))));
            return;
        }
    };
    let mut ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() {
        let _ = tx.blocking_send(Ok(sse_err("empty prompt")));
        return;
    }

    let backend = model.get_or_init_backend();
    let _guard = match model.inference_lock.lock() {
        Ok(g) => g,
        Err(p) => p.into_inner(),
    };
    backend.reset_kv_cache();

    let patched = model.patched.blocking_read();
    let knn_store_opt = if patched.knn_store.is_empty() { None } else { Some(&patched.knn_store) };
    let knn_probe_layer: Option<usize> = knn_store_opt
        .and_then(|s| s.layers().into_iter().next());
    const KNN_COSINE_THRESHOLD: f32 = 0.75;

    let mut sampler = larql_inference::sampling::Sampler::new(
        larql_inference::sampling::SamplingConfig {
            temperature: req.temperature,
            top_p: req.top_p,
            top_k: req.top_k,
            seed: req.seed,
        },
    );

    let session = DecodeSession::new(gguf);

    // Prefill: decode each prompt token. Probe KNN at the LAST prompt token.
    let prefill_start = Instant::now();
    let last_idx = ids.len() - 1;
    let mut last_h: Option<Vec<f32>> = None;
    let mut last_probe: Option<Vec<f32>> = None;
    for (i, &tid) in ids.iter().enumerate() {
        let probe_for = if i == last_idx { knn_probe_layer } else { None };
        let (h, probe) = session.step(tid, probe_for, &**backend);
        last_h = Some(h);
        if i == last_idx { last_probe = probe; }
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // KNN override short-circuit.
    if let (Some(probe), Some(store), Some(pl)) = (last_probe.as_ref(), knn_store_opt, knn_probe_layer) {
        let normed = gguf.apply_pre_ffn_norm_for_layer(probe, pl);
        if let Some((entry, cos)) = store.query_top1(pl, &normed) {
            if cos > KNN_COSINE_THRESHOLD {
                let label = format!("{} (KNN override, cos={:.2}, L{})", entry.target_token, cos, pl);
                let _ = tx.blocking_send(Ok(sse_token(&label, 0, 0)));
                let _ = tx.blocking_send(Ok(sse_done("knn_override", 0, prefill_ms, 0.0)));
                return;
            }
        }
    }

    // First sampled token from prefill state.
    let raw = session.finalize_for_sampler(last_h.as_ref().unwrap(), 64, &**backend);
    let mut next = match sampler.sample(&raw) {
        Some(t) => t,
        None => {
            let _ = tx.blocking_send(Ok(sse_done("empty_predictions", 0, prefill_ms, 0.0)));
            return;
        }
    };
    let first_tok = model.tokenizer.decode(&[next], true).unwrap_or_default();
    if tx.blocking_send(Ok(sse_token(&first_tok, next, 0))).is_err() { return; }
    ids.push(next);

    // Decode loop: one token per step, emit SSE.
    let mut per: Vec<f64> = Vec::with_capacity(req.n_tokens);
    let mut stopped_on = "n_tokens_reached";
    for step in 1..req.n_tokens {
        let t = Instant::now();
        let (h, _) = session.step(next, None, &**backend);
        let raw = session.finalize_for_sampler(&h, 64, &**backend);
        per.push(t.elapsed().as_secs_f64() * 1000.0);
        match sampler.sample(&raw) {
            Some(tid) if stop_ids.contains(&tid) => { stopped_on = "eos"; break; }
            Some(tid) => {
                let tok = model.tokenizer.decode(&[tid], true).unwrap_or_default();
                if tx.blocking_send(Ok(sse_token(&tok, tid, step))).is_err() { return; }
                next = tid;
            }
            None => { stopped_on = "empty"; break; }
        }
    }
    let avg = if !per.is_empty() { per.iter().sum::<f64>() / per.len() as f64 } else { 0.0 };
    let _ = tx.blocking_send(Ok(sse_done(stopped_on, per.len() + 1, prefill_ms, avg)));
}

fn sse_token(token: &str, tid: u32, step: usize) -> Event {
    Event::default().event("token").json_data(serde_json::json!({
        "token": token, "tid": tid, "step": step,
    })).unwrap_or_else(|_| Event::default().data("{}"))
}

fn sse_done(stopped_on: &str, decoded: usize, prefill_ms: f64, avg_decode_ms: f64) -> Event {
    Event::default().event("done").json_data(serde_json::json!({
        "stopped_on": stopped_on,
        "decoded_tokens": decoded,
        "prefill_ms": (prefill_ms * 10.0).round() / 10.0,
        "avg_decode_ms": (avg_decode_ms * 10.0).round() / 10.0,
    })).unwrap_or_else(|_| Event::default().data("{}"))
}

fn sse_err(msg: &str) -> Event {
    Event::default().event("error").json_data(serde_json::json!({"error": msg}))
        .unwrap_or_else(|_| Event::default().data(msg))
}

/// Ensure the `Stream<Item = Result<Event, Infallible>>` we assemble typechecks
/// against the trait bounds axum expects.
#[allow(dead_code)]
fn _sse_type_check<S: Stream<Item = Result<Event, Infallible>>>() {}
