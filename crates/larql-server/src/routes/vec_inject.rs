//! Vec_inject — chuk-lazurus style fact injection via residual stream.
//!
//! 12 bytes per fact: token_id + coefficient.
//! Retrieval via Q·K cosine at L29 H4 (Gemma 3 4B copy head).
//! Injection at L30: h += coef × (embed / ‖embed‖²).
//!
//! POST /v1/vec/insert — store a fact with K vector + coefficient
//! POST /v1/vec/query  — Q·K retrieval

use std::sync::Arc;
use axum::Json;
use axum::extract::State;
use serde::{Deserialize, Serialize};
use crate::error::ServerError;
use crate::state::AppState;

/// Copy head config for Gemma 3 4B (from calibrate_arch.py ablation)
const RETRIEVAL_LAYER: usize = 29;
const QUERY_HEAD: usize = 4;      // 8 Q heads, this is the copy head
const KV_HEAD: usize = 2;         // query_head 4 → kv_head 2 (GQA 8/4)
#[allow(dead_code)]
const INJECTION_LAYER: usize = 30;
const HEAD_DIM: usize = 256;

#[derive(Deserialize)]
pub struct VecInsertRequest {
    /// The fact text (e.g., "The server runs on port 3000")
    pub fact: String,
    /// The answer token (e.g., "3000") — the key entity to inject
    pub answer: String,
}

#[derive(Deserialize)]
pub struct VecQueryRequest {
    pub query: String,
    #[serde(default = "default_top_k")]
    pub top_k: usize,
}

fn default_top_k() -> usize { 5 }

#[derive(Clone, Serialize)]
pub struct VecEntry {
    pub fact: String,
    pub answer: String,
    pub token_id: u32,
    pub coefficient: f32,
    pub k_vector: Vec<f32>, // K at L29 H4, head_dim floats, L2-normalized
}

pub struct VecStore {
    entries: std::sync::RwLock<Vec<VecEntry>>,
}

impl VecStore {
    pub fn new() -> Self {
        Self { entries: std::sync::RwLock::new(Vec::new()) }
    }

    pub fn insert(&self, entry: VecEntry) {
        self.entries.write().unwrap().push(entry);
    }

    pub fn query(&self, q_vector: &[f32], top_k: usize) -> Vec<(VecEntry, f32)> {
        let entries = self.entries.read().unwrap();
        if entries.is_empty() { return Vec::new(); }

        let q_norm = l2_norm(q_vector);
        let mut scored: Vec<(usize, f32)> = entries.iter().enumerate()
            .map(|(i, e)| (i, dot(&q_norm, &e.k_vector)))
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);
        scored.into_iter().map(|(i, s)| (entries[i].clone(), s)).collect()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap().len()
    }

    /// Get all entries for vec_inject at inference time
    #[allow(dead_code)]
    pub fn get_inject_entries(&self, q_vector: &[f32], threshold: f32) -> Vec<(u32, f32)> {
        let entries = self.entries.read().unwrap();
        if entries.is_empty() { return Vec::new(); }

        let q_norm = l2_norm(q_vector);
        let scores: Vec<f32> = entries.iter().map(|e| dot(&q_norm, &e.k_vector)).collect();
        let mean_score: f32 = scores.iter().sum::<f32>() / scores.len() as f32;
        let adaptive_threshold = threshold.max(mean_score * 2.0);

        entries.iter().zip(scores.iter())
            .filter(|(_, &s)| s > adaptive_threshold)
            .map(|(e, _)| (e.token_id, e.coefficient))
            .collect()
    }
}

fn l2_norm(v: &[f32]) -> Vec<f32> {
    let n: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if n < 1e-12 { return vec![0.0; v.len()]; }
    v.iter().map(|x| x / n).collect()
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Extract K vector at L29 H4 using f32 weight projections (not Q4_K KV cache).
/// Runs GPU pipeline for L0-28, then f32 K projection at L29.
fn extract_k_and_coef(
    model: &crate::state::LoadedModel,
    text: &str,
    answer_token: &str,
) -> Option<(Vec<f32>, u32, f32)> {
    let weights = model.get_or_load_weights().ok()?;
    let backend = model.get_or_init_backend();

    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();

    let encoding = model.tokenizer.encode(text, true).ok()?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() { return None; }

    // Get answer token ID
    let answer_enc = model.tokenizer.encode(format!(" {answer_token}").as_str(), false).ok()?;
    let answer_id: u32 = *answer_enc.get_ids().first()?;

    let patched = model.patched.blocking_read();
    let gate_index: &dyn larql_vindex::GateIndex = patched.base();

    // GPU pipeline for L0-28 (Q4_K, fast)
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_real_mmap_ref() {
        (Some(mmap), true)
    } else { return None; };
    let q4_ffn = q4_ffn?;
    let intermediate = gate_index.num_features(0);
    if !gate_index.attn_q4k_layer_data(0).is_some() || intermediate == 0 { return None; }

    let q4_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 148
    } else { intermediate * hidden / 32 * 18 };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else { larql_compute::QuantFormat::Q4_0 };
    let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
        weights, patched.base(), 0..weights.num_layers,
        q4_ffn, q4_per_matrix, ffn_format,
    );
    if layers.is_empty() { return None; }

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    backend.reset_kv_cache();
    let embeds = larql_inference::forward::embed_tokens_pub(weights, &token_ids);

    // Prefill all tokens via GPU, capturing probe at RETRIEVAL_LAYER
    let mut last_hidden = vec![0.0f32; hidden];
    for p in 0..token_ids.len() {
        let x: Vec<f32> = embeds.row(p).to_vec();
        // Probe at RETRIEVAL_LAYER - 1 to get h BEFORE L29's attention
        // (output of L28 = input to L29). Then we apply L29's input norm + K/Q proj.
        let probe = if p == token_ids.len() - 1 { Some(RETRIEVAL_LAYER - 1) } else { None };
        if let Some((result, probe_h)) = backend.decode_token_with_probe(
            &layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            probe,
        ) {
            last_hidden = result;
            if let Some(ph) = probe_h {
                // We got h_post_attn at L29. Now do the f32 K projection.
                let h_attn = larql_inference::ndarray::Array2::from_shape_vec(
                    (1, hidden), ph).unwrap();

                // Apply input norm at L29
                let h_norm = larql_inference::forward::apply_norm(
                    weights, &h_attn, &arch.input_layernorm_key(RETRIEVAL_LAYER), norm_offset);

                // K projection using f32 weights (NOT Q4_K)
                let wk = weights.tensors.get(&arch.attn_k_key(RETRIEVAL_LAYER))?;
                let k_full = larql_compute::dot_proj_gpu(&h_norm, wk, Some(&**backend));

                // QK norm if present
                let nkv = arch.num_kv_heads_for_layer(RETRIEVAL_LAYER);
                let hd = arch.head_dim_for_layer(RETRIEVAL_LAYER);
                let qk_off = if arch.qk_norm_weight_offset() != 0.0 {
                    arch.qk_norm_weight_offset()
                } else { norm_offset };
                let k_normed = match arch.attn_k_norm_key(RETRIEVAL_LAYER)
                    .and_then(|k| weights.vectors.get(&k))
                {
                    Some(w) => larql_inference::residual::rms_norm_heads(&k_full, w, nkv, hd, qk_off),
                    None => k_full,
                };

                // Extract KV_HEAD's slice (no RoPE — raw semantic K)
                let head_start = KV_HEAD * HEAD_DIM;
                let k_head: Vec<f32> = k_normed.row(0)
                    .as_slice().unwrap()[head_start..head_start + HEAD_DIM].to_vec();

                // Coefficient: dot(h_L30, embed(answer_token))
                // We need h at L30. The GPU pipeline computed through all layers,
                // so last_hidden is h at the final layer. Use it as approximation
                // for h at L30 (L30 is near-final in a 34-layer model).
                let embed_scale = arch.embed_scale() as f32;
                let embed_row = model.embeddings.row(answer_id as usize);
                let mut embed_vec = vec![0.0f32; hidden.min(embed_row.len())];
                for j in 0..embed_vec.len() { embed_vec[j] = embed_row[j] * embed_scale; }
                let coef: f32 = last_hidden.iter().zip(embed_vec.iter())
                    .map(|(h, e)| h * e).sum();

                backend.reset_kv_cache();
                return Some((l2_norm(&k_head), answer_id, coef));
            }
        }
    }

    backend.reset_kv_cache();
    None
}

/// Extract Q vector at L29 H4 for a query (same projection, Q head slice)
fn extract_q(
    model: &crate::state::LoadedModel,
    text: &str,
) -> Option<Vec<f32>> {
    let weights = model.get_or_load_weights().ok()?;
    let backend = model.get_or_init_backend();
    let arch = &*weights.arch;
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();

    let encoding = model.tokenizer.encode(text, true).ok()?;
    let token_ids: Vec<u32> = encoding.get_ids().to_vec();
    if token_ids.is_empty() { return None; }

    let patched = model.patched.blocking_read();
    let gate_index: &dyn larql_vindex::GateIndex = patched.base();

    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_real_mmap_ref() {
        (Some(mmap), true)
    } else { return None; };
    let q4_ffn = q4_ffn?;
    let intermediate = gate_index.num_features(0);
    if !gate_index.attn_q4k_layer_data(0).is_some() || intermediate == 0 { return None; }

    let q4_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 148
    } else { intermediate * hidden / 32 * 18 };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else { larql_compute::QuantFormat::Q4_0 };
    let layers = larql_inference::layer_graph::pipeline_layer::build_pipeline_layers(
        weights, patched.base(), 0..weights.num_layers,
        q4_ffn, q4_per_matrix, ffn_format,
    );
    if layers.is_empty() { return None; }

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    backend.reset_kv_cache();
    let embeds = larql_inference::forward::embed_tokens_pub(weights, &token_ids);

    for p in 0..token_ids.len() {
        let x: Vec<f32> = embeds.row(p).to_vec();
        // Probe at RETRIEVAL_LAYER - 1 to get h BEFORE L29's attention
        // (output of L28 = input to L29). Then we apply L29's input norm + K/Q proj.
        let probe = if p == token_ids.len() - 1 { Some(RETRIEVAL_LAYER - 1) } else { None };
        if let Some((_result, probe_h)) = backend.decode_token_with_probe(
            &layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            probe,
        ) {
            if let Some(ph) = probe_h {
                let h_attn = larql_inference::ndarray::Array2::from_shape_vec(
                    (1, hidden), ph).unwrap();
                let h_norm = larql_inference::forward::apply_norm(
                    weights, &h_attn, &arch.input_layernorm_key(RETRIEVAL_LAYER), norm_offset);

                // Q projection using f32 weights
                let wq = weights.tensors.get(&arch.attn_q_key(RETRIEVAL_LAYER))?;
                let q_full = larql_compute::dot_proj_gpu(&h_norm, wq, Some(&**backend));

                // QK norm
                let nq = arch.num_q_heads_for_layer(RETRIEVAL_LAYER);
                let hd = arch.head_dim_for_layer(RETRIEVAL_LAYER);
                let qk_off = if arch.qk_norm_weight_offset() != 0.0 {
                    arch.qk_norm_weight_offset()
                } else { norm_offset };
                let q_normed = match arch.attn_q_norm_key(RETRIEVAL_LAYER)
                    .and_then(|k| weights.vectors.get(&k))
                {
                    Some(w) => larql_inference::residual::rms_norm_heads(&q_full, w, nq, hd, qk_off),
                    None => q_full,
                };

                // Extract QUERY_HEAD's slice
                let head_start = QUERY_HEAD * HEAD_DIM;
                let q_head: Vec<f32> = q_normed.row(0)
                    .as_slice().unwrap()[head_start..head_start + HEAD_DIM].to_vec();

                backend.reset_kv_cache();
                return Some(l2_norm(&q_head));
            }
        }
    }

    backend.reset_kv_cache();
    None
}

// ── Handlers ──

pub async fn handle_vec_insert(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VecInsertRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let fact = req.fact.clone();
    let answer = req.answer.clone();

    let result = tokio::task::spawn_blocking(move || {
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g, Err(p) => p.into_inner(),
        };

        let (k_vec, token_id, coef) = extract_k_and_coef(&model, &fact, &answer)
            .ok_or_else(|| ServerError::Internal("K extraction failed".into()))?;

        state.vec_store.insert(VecEntry {
            fact: fact.clone(),
            answer: answer.clone(),
            token_id,
            coefficient: coef,
            k_vector: k_vec,
        });

        Ok::<_, ServerError>(serde_json::json!({
            "status": "ok",
            "fact": fact,
            "answer": answer,
            "token_id": token_id,
            "coefficient": coef,
            "layer": RETRIEVAL_LAYER,
            "head": QUERY_HEAD,
            "total_facts": state.vec_store.len(),
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}

pub async fn handle_vec_query(
    State(state): State<Arc<AppState>>,
    Json(req): Json<VecQueryRequest>,
) -> Result<Json<serde_json::Value>, ServerError> {
    let model = state.model(None)
        .ok_or_else(|| ServerError::NotFound("no model loaded".into()))?;
    let model = Arc::clone(model);
    let query = req.query.clone();
    let top_k = req.top_k;

    let result = tokio::task::spawn_blocking(move || {
        let _guard = match model.inference_lock.lock() {
            Ok(g) => g, Err(p) => p.into_inner(),
        };

        let q_vec = extract_q(&model, &query)
            .ok_or_else(|| ServerError::Internal("Q extraction failed".into()))?;

        let results = state.vec_store.query(&q_vec, top_k);

        Ok::<_, ServerError>(serde_json::json!({
            "query": query,
            "layer": RETRIEVAL_LAYER,
            "head": QUERY_HEAD,
            "results": results.iter().map(|(e, score)| serde_json::json!({
                "fact": e.fact,
                "answer": e.answer,
                "token_id": e.token_id,
                "coefficient": e.coefficient,
                "score": (score * 1000.0).round() / 1000.0,
            })).collect::<Vec<_>>(),
        }))
    })
    .await
    .map_err(|e| ServerError::Internal(e.to_string()))??;

    Ok(Json(result))
}
