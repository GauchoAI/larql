//! RAG fact store — embedding-based retrieval for novel facts.
//!
//! POST /v1/rag/insert — store a fact with its embedding
//! POST /v1/rag/query  — find matching facts for a query
//!
//! Uses the model's token embedding matrix for sentence embeddings
//! (mean of token embeddings). Fast: no forward pass needed.

use std::sync::Arc;
use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use crate::error::ServerError;
use crate::state::AppState;

#[derive(Deserialize)]
pub struct RagInsertRequest {
    /// The fact text to store (e.g. "The user's name is Miguel")
    pub fact: String,
    /// Optional category for filtering
    #[serde(default)]
    pub category: String,
}

#[derive(Deserialize)]
pub struct RagQueryRequest {
    /// Query text
    pub query: String,
    /// Max results
    #[serde(default = "default_top_k")]
    pub top_k: usize,
    /// Minimum cosine threshold
    #[serde(default = "default_threshold")]
    pub threshold: f32,
}

fn default_top_k() -> usize { 3 }
fn default_threshold() -> f32 { 0.5 }

#[derive(Clone, Serialize)]
pub struct RagEntry {
    pub fact: String,
    pub category: String,
    pub embedding: Vec<f32>,
}

/// Thread-safe RAG store. Lives on AppState.
pub struct RagStore {
    entries: std::sync::RwLock<Vec<RagEntry>>,
}

impl RagStore {
    pub fn new() -> Self {
        Self { entries: std::sync::RwLock::new(Vec::new()) }
    }

    pub fn insert(&self, entry: RagEntry) {
        self.entries.write().unwrap().push(entry);
    }

    pub fn query(&self, query_embed: &[f32], top_k: usize, threshold: f32) -> Vec<(RagEntry, f32)> {
        let entries = self.entries.read().unwrap();
        let q_norm = l2_norm(query_embed);

        let mut scored: Vec<(RagEntry, f32)> = entries.iter()
            .map(|e| {
                let cos = cosine(&q_norm, &e.embedding);
                (e.clone(), cos)
            })
            .filter(|(_, cos)| *cos >= threshold)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }
}

/// Compute sentence embedding using the model's hidden state at a middle
/// layer (L12). Runs a forward pass to get contextual representations,
/// then averages the last position's hidden state. Much richer than
/// mean-of-token-embeddings because it captures semantic relationships.
///
/// Falls back to mean-of-token-embeddings when backend is unavailable.
pub fn sentence_embedding(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    embeddings: &larql_vindex::ndarray::Array2<f32>,
    text: &str,
) -> Option<Vec<f32>> {
    sentence_embedding_token_mean(tokenizer, embeddings, text)
}

/// Rich embedding via model forward pass. Captures at the specified layer.
/// Returns the hidden state at the last token position — contextual, not bag-of-words.
pub fn sentence_embedding_model(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    weights: &larql_models::ModelWeights,
    index: &larql_vindex::VectorIndex,
    backend: &dyn larql_compute::ComputeBackend,
    text: &str,
    layer: usize,
) -> Option<Vec<f32>> {
    let encoding = tokenizer.encode(text, true).ok()?;
    let ids: Vec<u32> = encoding.get_ids().to_vec();
    if ids.is_empty() { return None; }

    // Use the probe endpoint's capture — runs GPU prefill,
    // returns pre_ffn_norm(h_post_attn) at the probe layer.
    let vec = larql_inference::capture_knn_key_gpu(
        weights, &ids, layer, index, backend,
    )?;

    Some(l2_norm(&vec))
}

/// Fallback: mean of token embeddings (no forward pass).
fn sentence_embedding_token_mean(
    tokenizer: &larql_vindex::tokenizers::Tokenizer,
    embeddings: &larql_vindex::ndarray::Array2<f32>,
    text: &str,
) -> Option<Vec<f32>> {
    let encoding = tokenizer.encode(text, false).ok()?;
    let ids = encoding.get_ids();
    if ids.is_empty() { return None; }

    let dim = embeddings.shape()[1];
    let mut mean = vec![0.0f32; dim];
    let mut count = 0usize;

    for &id in ids {
        let id = id as usize;
        if id < embeddings.shape()[0] {
            let row = embeddings.row(id);
            for j in 0..dim {
                mean[j] += row[j];
            }
            count += 1;
        }
    }

    if count == 0 { return None; }
    let inv = 1.0 / count as f32;
    for v in &mut mean { *v *= inv; }

    Some(l2_norm(&mean))
}

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-12 { return vec![0.0; v.len()]; }
    v.iter().map(|x| x / norm).collect()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ── Handlers ──

/// Deep INSERT — uses model forward pass for embedding (slower but richer).
pub async fn handle_rag_insert_deep(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagInsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let fact = req.fact.clone();
    let category = req.category.clone();

    let result = tokio::task::spawn_blocking(move || {
        let weights = model.get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();

        let patched = model.patched.blocking_read();
        let embed = sentence_embedding_model(
            &model.tokenizer, weights, patched.base(), &**backend,
            &fact, 12, // L12 — middle layer, semantic but not output-specific
        ).ok_or_else(|| ServerError::Internal("model embedding failed".into()))?;

        backend.reset_kv_cache();

        state.rag_store.insert(RagEntry {
            fact: fact.clone(),
            category: category.clone(),
            embedding: embed,
        });

        Ok::<_, ServerError>(serde_json::json!({
            "status": "ok",
            "fact": fact,
            "method": "deep-L12",
            "total_facts": state.rag_store.len(),
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}

/// Deep QUERY — uses model forward pass for query embedding.
pub async fn handle_rag_query_deep(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagQueryRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let query = req.query.clone();
    let top_k = req.top_k;
    let threshold = req.threshold;

    let result = tokio::task::spawn_blocking(move || {
        let weights = model.get_or_load_weights()
            .map_err(ServerError::InferenceUnavailable)?;
        let backend = model.get_or_init_backend();
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g,
            Err(p) => p.into_inner(),
        };
        backend.reset_kv_cache();

        let patched = model.patched.blocking_read();
        let query_embed = sentence_embedding_model(
            &model.tokenizer, weights, patched.base(), &**backend,
            &query, 12,
        ).ok_or_else(|| ServerError::Internal("model embedding failed".into()))?;

        backend.reset_kv_cache();

        let results = state.rag_store.query(&query_embed, top_k, threshold);

        Ok::<_, ServerError>(serde_json::json!({
            "query": query,
            "method": "deep-L12",
            "results": results.iter().map(|(e, cos)| serde_json::json!({
                "fact": e.fact,
                "category": e.category,
                "cosine": (cos * 1000.0).round() / 1000.0,
            })).collect::<Vec<_>>(),
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}

pub async fn handle_rag_insert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagInsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let embed = sentence_embedding(&model.tokenizer, &model.embeddings, &req.fact)
        .ok_or_else(|| ServerError::Internal("embedding failed".into()))?;

    state.rag_store.insert(RagEntry {
        fact: req.fact.clone(),
        category: req.category.clone(),
        embedding: embed,
    });

    Ok(Json(serde_json::json!({
        "status": "ok",
        "fact": req.fact,
        "total_facts": state.rag_store.len(),
    })))
}

pub async fn handle_rag_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<RagQueryRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;

    let query_embed = sentence_embedding(&model.tokenizer, &model.embeddings, &req.query)
        .ok_or_else(|| ServerError::Internal("embedding failed".into()))?;

    let results = state.rag_store.query(&query_embed, req.top_k, req.threshold);

    Ok(Json(serde_json::json!({
        "query": req.query,
        "results": results.iter().map(|(e, cos)| serde_json::json!({
            "fact": e.fact,
            "category": e.category,
            "cosine": (cos * 1000.0).round() / 1000.0,
        })).collect::<Vec<_>>(),
    })))
}

/// Retrieve matching facts for a prompt, return as context string.
/// Called by chat_completions to inject RAG context.
pub fn retrieve_context(
    state: &AppState,
    model: &crate::state::LoadedModel,
    user_msg: &str,
    threshold: f32,
) -> Option<String> {
    if state.rag_store.len() == 0 { return None; }

    let query_embed = sentence_embedding(&model.tokenizer, &model.embeddings, user_msg)?;
    let results = state.rag_store.query(&query_embed, 3, threshold);

    if results.is_empty() { return None; }

    let facts: Vec<String> = results.iter()
        .map(|(e, _cos)| {
            // Cap each fact to ~200 chars, strip code fences
            let clean: String = e.fact.replace("```", "")
                .chars().take(200).collect();
            format!("- {clean}")
        })
        .collect();

    Some(format!("Context:\n{}", facts.join("\n")))
}
