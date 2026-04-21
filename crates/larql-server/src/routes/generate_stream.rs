//! POST /v1/generate — Server-Sent Events token streaming.
//!
//! Routes through the llama.cpp pipeline when available, falling back
//! to the legacy GGUF pipeline otherwise.  KNN overlay short-circuits
//! the response before sampling begins: if the probe matches the last
//! prompt residual with cosine above threshold, the server emits the
//! override token and closes the stream.

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
use crate::llama_probe::{snapshot_layer, Mode};
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

    if model.llama.is_none() {
        return Err(ServerError::InferenceUnavailable(
            "no weights.gguf in vindex dir — drop one in to enable inference".into(),
        ));
    }

    let stop_ids: Vec<u32> = req.stop_ids.clone().unwrap_or_else(|| vec![1u32, 106]);

    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, Infallible>>(32);
    let model_cl = Arc::clone(&model);
    tokio::task::spawn_blocking(move || {
        run_llama_generate(&model_cl, &req, &stop_ids, &tx);
    });

    let stream = stream::unfold(rx, |mut rx| async move {
        rx.recv().await.map(|msg| (msg, rx))
    });
    let sse: Sse<_> = Sse::new(stream).keep_alive(KeepAlive::default());
    Ok(sse)
}

/// New hot path: generate via larql-llamacpp, one-shot KNN override at
/// prefill, stream tokens via SSE.
fn run_llama_generate(
    model: &Arc<crate::state::LoadedModel>,
    req: &GenerateRequest,
    stop_ids: &[u32],
    tx: &tokio::sync::mpsc::Sender<Result<Event, Infallible>>,
) {
    let pipe_mu = model.llama.as_ref().expect("llama present");

    // Snapshot KNN entries at the install layer for this request.
    let install_layer = model.config.num_layers.saturating_sub(8);
    let tensor = format!("attn_post_norm-{install_layer}");
    const THRESHOLD: f32 = 0.75;

    let patched = model.patched.blocking_read();
    let entries = snapshot_layer(&patched.knn_store, install_layer);
    drop(patched);

    // Configure probe: KnnQuery, one-shot at the probe layer.
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

    let mut pipe = match pipe_mu.lock() {
        Ok(p) => p,
        Err(e) => {
            let _ = tx.blocking_send(Ok(sse_err(&format!("pipeline lock poisoned: {e}"))));
            return;
        }
    };
    pipe.reset_kv();

    // Prefill.
    let prefill_start = Instant::now();
    let n_prefill = match pipe.prefill(&req.prompt) {
        Ok(n) => n,
        Err(e) => {
            let _ = tx.blocking_send(Ok(sse_err(&format!("prefill: {e}"))));
            return;
        }
    };
    if n_prefill == 0 {
        let _ = tx.blocking_send(Ok(sse_err("empty prompt")));
        return;
    }
    let prefill_ms = prefill_start.elapsed().as_secs_f64() * 1000.0;

    // Check for KNN override from probe.
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
        let _ = tx.blocking_send(Ok(sse_token(&label, 0, 0)));
        let _ = tx.blocking_send(Ok(sse_done("knn_override", 0, prefill_ms, 0.0)));
        return;
    }

    // Sampling setup — share larql-inference's Sampler.
    let mut sampler = larql_llamacpp::Sampler::new(larql_llamacpp::SamplingConfig {
        temperature: req.temperature,
        top_p: req.top_p,
        top_k: req.top_k,
        seed: req.seed,
    });

    // First sample from the last prefill position.
    let first_preds = pipe.top_k_at((n_prefill - 1) as i32, 64);
    let mut next: i32 = match sampler.sample(&first_preds) {
        Some(t) => t as i32,
        None => {
            let _ = tx.blocking_send(Ok(sse_done("empty_predictions", 0, prefill_ms, 0.0)));
            return;
        }
    };
    let first_tok = pipe.decode_token(next);
    if tx.blocking_send(Ok(sse_token(&first_tok, next as u32, 0))).is_err() {
        return;
    }

    // Decode loop.
    let mut per: Vec<f64> = Vec::with_capacity(req.n_tokens);
    let mut stopped_on = "n_tokens_reached";
    for step in 1..req.n_tokens {
        let t = Instant::now();
        if let Err(e) = pipe.feed(next) {
            let _ = tx.blocking_send(Ok(sse_err(&format!("decode: {e}"))));
            return;
        }
        let preds = pipe.top_k_at(0, 64);
        per.push(t.elapsed().as_secs_f64() * 1000.0);

        match sampler.sample(&preds) {
            Some(tid) if stop_ids.contains(&tid) => {
                stopped_on = "eos";
                break;
            }
            Some(tid) => {
                let tok = pipe.decode_token(tid as i32);
                if tx
                    .blocking_send(Ok(sse_token(&tok, tid, step)))
                    .is_err()
                {
                    return;
                }
                next = tid as i32;
            }
            None => {
                stopped_on = "empty";
                break;
            }
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

#[allow(dead_code)]
fn _sse_type_check<S: Stream<Item = Result<Event, Infallible>>>() {}
