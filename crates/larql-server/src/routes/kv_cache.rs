//! KV cache precompute + replay.
//!
//! POST /v1/kv/precompute — prefill text, save KV state to memory
//! POST /v1/kv/stats — show saved KV state info
//!
//! The chat_completions handler auto-restores saved KV before each query,
//! so the model attends to precomputed context without re-prefilling.

use std::sync::Arc;
use axum::Json;
use axum::extract::State;
use serde::Deserialize;
use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct PrecomputeRequest {
    /// Text to prefill and cache (conversation history)
    pub text: String,
}

/// Saved KV cache state — all layers' K and V vectors.
pub struct SavedKvState {
    /// Per-layer (k_flat, v_flat, seq_len)
    pub layers: Vec<(Vec<f32>, Vec<f32>, usize)>,
    pub num_layers: usize,
    pub total_tokens: usize,
}

impl SavedKvState {
    pub fn size_bytes(&self) -> usize {
        self.layers.iter().map(|(k, v, _)| (k.len() + v.len()) * 4).sum()
    }
}

pub struct KvCacheStore {
    pub saved: std::sync::RwLock<Option<SavedKvState>>,
}

impl KvCacheStore {
    pub fn new() -> Self {
        Self { saved: std::sync::RwLock::new(None) }
    }
}

pub async fn handle_precompute(
    State(state): State<Arc<AppState>>,
    Json(req): Json<PrecomputeRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let text = req.text;

    let result = tokio::task::spawn_blocking(move || {
        let weights = model.get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g, Err(p) => p.into_inner(),
        };

        // Wrap in Gemma 3 chat template so KV matches inference format.
        // The text becomes the system message content.
        let system = "You are a helpful assistant. Answer questions based on the conversation history provided.";
        let chat_prompt = format!(
            "<start_of_turn>system\n{system}\n\nConversation history:\n{text}<end_of_turn>\n"
        );
        let encoding = model.tokenizer.encode(chat_prompt.as_str(), true)
            .map_err(|e| ServerError::Internal(format!("tokenize: {e}")))?;
        let token_ids: Vec<u32> = encoding.get_ids().to_vec();
        let num_tokens = token_ids.len();

        tracing::info!("[kv] precomputing {} tokens...", num_tokens);
        let t0 = std::time::Instant::now();

        // Prefill via predict (populates KV cache)
        let patched = model.patched.blocking_read();
        let cache = larql_inference::CachedLayerGraph::from_residuals(Vec::new());
        let walk_ffn = if model.walk_only {
            Some(larql_inference::WalkFfn::new_with_backend(
                weights, patched.base(), 1024, &**backend,
            ))
        } else { None };
        let ffn_override: Option<&dyn larql_inference::ffn::FfnBackend> =
            walk_ffn.as_ref().map(|w| w as &dyn larql_inference::ffn::FfnBackend);

        backend.reset_kv_cache();
        let _result = larql_inference::predict_honest_with_knn_ffn(
            weights, &model.tokenizer, &token_ids, 1,
            patched.base(), &**backend, &cache,
            0..weights.num_layers, None, ffn_override,
        );

        let prefill_s = t0.elapsed().as_secs_f64();
        tracing::info!("[kv] prefill done: {} tokens in {:.1}s ({:.0} tok/s)",
            num_tokens, prefill_s, num_tokens as f64 / prefill_s);

        // Save KV state from all layers
        let mut saved_layers = Vec::new();
        for layer in 0..weights.num_layers {
            if let Some((k, v, seq_len)) = backend.debug_read_kv_layer(layer) {
                saved_layers.push((k, v, seq_len));
            }
        }

        let saved = SavedKvState {
            num_layers: saved_layers.len(),
            total_tokens: num_tokens,
            layers: saved_layers,
        };
        let size_mb = saved.size_bytes() as f64 / 1024.0 / 1024.0;

        tracing::info!("[kv] saved {} layers, {:.1} MB", saved.num_layers, size_mb);

        // Store in the KV cache store
        *state.kv_cache_store.saved.write().unwrap() = Some(saved);

        backend.reset_kv_cache();

        Ok::<_, ServerError>(serde_json::json!({
            "status": "ok",
            "tokens": num_tokens,
            "layers": weights.num_layers,
            "prefill_seconds": (prefill_s * 10.0).round() / 10.0,
            "saved_mb": (size_mb * 10.0).round() / 10.0,
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}

pub async fn handle_kv_stats(
    State(state): State<Arc<AppState>>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let saved = state.kv_cache_store.saved.read().unwrap();
    match saved.as_ref() {
        Some(s) => Ok(Json(serde_json::json!({
            "status": "loaded",
            "tokens": s.total_tokens,
            "layers": s.num_layers,
            "size_mb": (s.size_bytes() as f64 / 1024.0 / 1024.0 * 10.0).round() / 10.0,
        }))),
        None => Ok(Json(serde_json::json!({"status": "empty"}))),
    }
}

/// Restore saved KV state into the backend's KV cache.
/// Called before each chat completion if saved state exists.
pub fn restore_kv_cache(
    backend: &dyn larql_inference::ComputeBackend,
    saved: &SavedKvState,
    num_kv_heads: usize,
    head_dim: usize,
) {
    backend.reset_kv_cache();
    for (layer, (k, v, seq_len)) in saved.layers.iter().enumerate() {
        backend.populate_kv_layer(layer, k, v, *seq_len, num_kv_heads, head_dim);
    }
}
