//! Prediction entry points — the functions external code calls
//!
//! All GPU pipeline layer construction goes through `pipeline_layer::build_pipeline_layers()`.
//! Logits computation goes through `logits::finalize_logits()`.
//! KV cache prefill goes through `prefill::prefill_with_kv()`.
//! Token generation goes through `generate::generate()`.

use ndarray::Array2;

use larql_compute::ComputeBackend;
use crate::model::ModelWeights;
use super::{LayerGraph, DenseLayerGraph, CachedLayerGraph};

// Re-export moved functions for backward compatibility.
pub use super::prefill::prefill_with_kv;
pub use super::logits::finalize_logits;
pub use super::generate::{generate, GenerateResult};

// Alias for internal callers.
use super::prefill::prefill_kv_cache_cpu;

/// Run a full forward pass using vindex logits (KNN against lm_head mmap).
/// Replaces the 231ms dense logits matmul with a ~1ms KNN lookup.
pub fn predict_with_graph_vindex_logits(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: &larql_vindex::VectorIndex,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    // Final norm
    let norm_offset = weights.arch.norm_weight_offset();
    let h_final = crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);

    // Vindex logits: KNN against lm_head mmap
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    // Get raw scores from KNN (dot products against lm_head)
    let hits = index.lm_head_knn(&last_row, top_k);

    // Apply scaling, softcap, softmax over top-K
    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();

    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions, raw_predictions: Vec::new() }
}

/// Run a full forward pass using a LayerGraph for per-layer routing.
/// This is the generic layer loop — embedding → layers → logits.
pub fn predict_with_graph(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
) -> crate::forward::PredictResult {
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);

    for layer in 0..weights.num_layers {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => h = output.residual,
            None => break,
        }
    }

    crate::forward::logits_to_predictions_pub(weights, &h, tokenizer, top_k)
}

/// Two-pass split pipeline: attention on CPU, FFN batched on Metal GPU.
///
/// Pass 1: Run attention for all layers with attention-only residual stream (CPU).
///          The FFN contribution is deferred — attention sees approximate residuals.
///          This is valid because attention is template-fixed (~99% identical across entities).
///
/// Pass 2: Compute FFN for all layers in one Metal command buffer (8.5ms for 21 layers).
///          Uses the post-attention residuals from pass 1 as FFN inputs.
///
/// Pass 3: Add FFN outputs to the final residual and compute logits via vindex KNN.
///
/// Target: 55ms attention + 8.5ms FFN + 5ms logits = 68ms → 15 tok/s
#[allow(clippy::too_many_arguments)]
pub fn predict_split_pass(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let arch = &*weights.arch;
    let norm_offset = arch.norm_weight_offset();

    // ── Pass 0: Cached layers (0ms) ──
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // ── Pass 1: Attention-only for walk layers (CPU BLAS) ──
    // Store post-attention residuals and FFN-normed inputs.
    let mut post_attn_residuals: Vec<Array2<f32>> = Vec::new();
    let mut ffn_inputs: Vec<Vec<f32>> = Vec::new();

    for layer in layer_range.clone() {
        // Run attention only (CPU BLAS, no FFN)
        let (h_post_attn, _attn_proj, _) =
            crate::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                .unwrap_or_else(|| {
                    // Fallback: identity (shouldn't happen with valid weights)
                    (h.clone(), h.clone(), None)
                });

        // Compute pre-FFN norm (this is the FFN input)
        let pre_ffn_key = if arch.has_post_norms() {
            arch.pre_feedforward_layernorm_key(layer)
        } else {
            Some(arch.post_attention_layernorm_key(layer))
        };
        let h_ffn = match pre_ffn_key {
            Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
            None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
        };

        // Store last-token FFN input for GPU batch
        let last_row = h_ffn.row(seq_len - 1);
        ffn_inputs.push(last_row.to_vec());
        post_attn_residuals.push(h_post_attn.clone());

        // Continue with attention-only residual (approximate — no FFN contribution)
        h = h_post_attn;
    }

    // ── Pass 2: Batch FFN on Metal GPU ──
    let num_walk_layers = layer_range.len();

    // Try batched Q4 FFN via multi_layer_q4_ffn
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let ffn_outputs = if gate_index.has_interleaved_q4() && backend.has_q4() {
        if let Some(q4_mmap) = gate_index.interleaved_q4_mmap_ref() {
            let intermediate = gate_index.num_features(layer_range.start);
            if intermediate > 0 {
                let q4_bytes_per_matrix = intermediate * hidden / 32 * 18;
                let q4_bytes_per_layer = q4_bytes_per_matrix * 3;

                // Collect Q4 data slices for all walk layers
                let layers_q4: Vec<(&[u8], &[u8], &[u8])> = layer_range.clone()
                    .map(|layer| {
                        let start = layer * q4_bytes_per_layer;
                        let gate = &q4_mmap[start..start + q4_bytes_per_matrix];
                        let up = &q4_mmap[start + q4_bytes_per_matrix..start + 2 * q4_bytes_per_matrix];
                        let down = &q4_mmap[start + 2 * q4_bytes_per_matrix..start + 3 * q4_bytes_per_matrix];
                        (gate, up, down)
                    })
                    .collect();

                // Use the first FFN input as the batch input
                // multi_layer_q4_ffn chains layers: out of layer N → input of layer N+1
                let x = &ffn_inputs[0];
                backend.multi_layer_q4_ffn(&layers_q4, x, intermediate, hidden)
            } else {
                None
            }
        } else {
            None
        }
    } else {
        None
    };

    // ── Pass 3: Combine ──
    // If GPU batch succeeded, use the final FFN output.
    // Otherwise, fall back to per-layer CPU FFN.
    if let Some(ffn_final) = ffn_outputs {
        // The multi_layer_q4_ffn returns the final residual after all FFN layers.
        // Add it to the last post-attention residual.
        let last_post_attn = &post_attn_residuals[num_walk_layers - 1];
        let mut h_final = last_post_attn.clone();
        let mut last_row_mut = h_final.row_mut(seq_len - 1);
        for j in 0..hidden {
            // The FFN output is the chained result — use it as the last-token residual
            last_row_mut[j] = ffn_final[j];
        }
        h = h_final;
    } else {
        // Fallback: run FFN per-layer on CPU
        h = crate::forward::embed_tokens_pub(weights, token_ids);
        for layer in 0..layer_range.start {
            if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range.clone() {
            let dense = DenseLayerGraph {
                ffn: &walk_ffn, backend: None,
                capture_activation: false, capture_attention: false,
            };
            if let Some(output) = dense.forward_layer(weights, &h, layer) {
                h = output.residual;
            }
        }
    }

    // Final norm + vindex logits
    let h_final = crate::forward::apply_norm(weights, &h, weights.arch.final_norm_key(), norm_offset);
    let last_row = h_final.row(seq_len - 1).to_owned();

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let hits = index.lm_head_knn(&last_row, top_k);
    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();
    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions, raw_predictions: Vec::new() }
}

/// Split pass using cached attention residuals — exact output at GPU speed.
///
/// Uses `AttentionCache` (built from one exact run) to skip all attention
/// computation. Batches FFN on Metal GPU in one command buffer.
///
/// Target: 0ms attention + 8.5ms FFN + 5ms logits = ~14ms → 71 tok/s
pub fn predict_split_cached(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    attn_cache: &super::AttentionCache,
    _layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Zero-copy: borrow the cached residual, don't clone.
    // Final norm produces a new array (unavoidable), but the input is borrowed.
    let h_final = crate::forward::apply_norm(
        weights, &attn_cache.final_residual, weights.arch.final_norm_key(), norm_offset,
    );
    let seq_len = h_final.shape()[0];
    let last_row = h_final.row(seq_len - 1).to_owned();

    // GPU Q4 logits when available (1ms), else CPU BLAS (10ms)
    let hits = index.lm_head_knn_backend(&last_row, top_k, backend);

    let logits_scale = weights.arch.logits_scaling();
    let final_softcap = weights.arch.final_logit_softcapping();
    let inv_scale = 1.0 / logits_scale;

    let scaled: Vec<(u32, f32)> = hits.iter().map(|&(tid, score)| {
        let mut logit = score * inv_scale;
        if let Some(cap) = final_softcap {
            logit = (logit / cap).tanh() * cap;
        }
        (tid, logit)
    }).collect();

    let max_logit = scaled.iter().map(|(_, l)| *l).fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f64 = scaled.iter().map(|(_, l)| ((*l - max_logit) as f64).exp()).sum();
    let predictions = scaled.iter()
        .filter_map(|&(tid, logit)| {
            let prob = ((logit - max_logit) as f64).exp() / exp_sum;
            tokenizer.decode(&[tid], true).ok()
                .map(|s| (s.trim().to_string(), prob))
        })
        .collect();

    crate::forward::PredictResult { predictions, raw_predictions: Vec::new() }
}

/// Honest production pipeline: real computation, no over-caching.
///
/// - L0-12: cached (template-fixed, proven at 0.999 cosine — legitimate)
/// - L13-33: interleaved attention (CPU BLAS) + FFN (GPU Q4 via compute crate)
/// - Logits: GPU Q4 matvec against lm_head (1ms)
///
/// Capture the pre-FFN residual at `target_layer` for the LAST token of
/// `token_ids`, using the same Gemma 3 f32 + Metal matmul path as
/// `predict_honest` prefill. The returned vector is in the exact stage
/// (`apply_norm(h_post_attn, pre_feedforward_layernorm_key)`) that
/// `WalkFfn::forward` receives — so it's suitable as a key for
/// `KnnStore::add`, matching what LQL INSERT produces via the walk path.
///
/// Expected cost: ~1 s for a ~8-token prompt on M4 Pro (Metal f32), vs
/// ~75 s for the equivalent LQL walk-FFN capture.
///
/// Returns `None` if the model's architecture isn't post-norm (this
/// helper is Gemma 3 specific) or if the target_layer is out of range.
pub fn capture_residual_post_attn_norm(
    weights: &ModelWeights,
    token_ids: &[u32],
    target_layer: usize,
    backend: &dyn ComputeBackend,
) -> Option<Vec<f32>> {
    capture_residual_post_attn_norm_ffn(weights, token_ids, target_layer, backend, None)
}

/// Same as `capture_residual_post_attn_norm` but with an FFN backend override.
/// Walk-only callers must pass `Some(&WalkFfn)` because the default path
/// (`WeightFfnGpu`) reads FFN tensors that were dropped to save RAM.
pub fn capture_residual_post_attn_norm_ffn(
    weights: &ModelWeights,
    token_ids: &[u32],
    target_layer: usize,
    backend: &dyn ComputeBackend,
    ffn_override: Option<&dyn crate::ffn::FfnBackend>,
) -> Option<Vec<f32>> {
    let arch = &*weights.arch;
    if !arch.has_post_norms() { return None; }
    if target_layer >= weights.num_layers { return None; }

    let norm_offset = arch.norm_weight_offset();

    // Embedding → attention + FFN per layer, matching the Gemma 3 prefill
    // branch of predict_honest. Uses override (walk) when provided; falls back
    // to WeightFfnGpu for dense capture.
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    backend.reset_kv_cache();
    let gpu_ffn = crate::ffn::WeightFfnGpu { weights, backend };
    let ffn: &dyn crate::ffn::FfnBackend = match ffn_override {
        Some(f) => f,
        None => &gpu_ffn,
    };

    for abs_layer in 0..=target_layer {
        let (h_post_attn, k_rope, v) =
            crate::attention::gpu::run_attention_with_kv_backend(
                weights, &h, abs_layer, Some(backend),
            )?;

        // Populate KV cache so subsequent decode calls see the prompt context.
        if backend.has_kv_cache() {
            let seq_len = h.shape()[0];
            let k_flat = k_rope.as_slice().unwrap_or(&[]);
            let v_flat = v.as_slice().unwrap_or(&[]);
            backend.populate_kv_layer(abs_layer, k_flat, v_flat,
                seq_len, weights.num_kv_heads, weights.head_dim);
        }

        if abs_layer == target_layer {
            // Match the exact stage WalkFfn captures: pre_ffn_norm(h_post_attn),
            // last token. See the "cos=0.23 vs 1.00" note in findings.md.
            let pre_ffn_key = arch.pre_feedforward_layernorm_key(abs_layer);
            let h_ffn = match pre_ffn_key {
                Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
            };
            let last = h_ffn.shape()[0] - 1;
            return Some(h_ffn.row(last).to_vec());
        }

        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, abs_layer, ffn, false);
        h = h_out;
    }
    None
}

/// Every entity-dependent layer is computed. No approximate residuals.
#[allow(clippy::too_many_arguments)]
pub fn predict_honest(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
) -> crate::forward::PredictResult {
    predict_honest_with_knn(
        weights, tokenizer, token_ids, top_k, index, backend, cached_layers, layer_range, None,
    )
}

/// Same as `predict_honest` but with an optional KNN overlay store.
///
/// When `knn_store` is `Some` and the Gemma 3 single-token fast path is active,
/// after each layer's attention step we query the store at that layer with the
/// post-attention residual. On a cosine hit above `KNN_COSINE_THRESHOLD` the
/// remaining layers are skipped and the override target is returned as top-1.
/// Other paths are unchanged (no KNN consult) — they remain the slow/debug
/// paths.
#[allow(clippy::too_many_arguments)]
pub fn predict_honest_with_knn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    knn_store: Option<&larql_vindex::KnnStore>,
) -> crate::forward::PredictResult {
    predict_honest_with_knn_ffn(
        weights, tokenizer, token_ids, top_k, index, backend, cached_layers,
        layer_range, knn_store, None,
    )
}

/// Like `predict_honest_with_knn`, but with an optional FFN override so callers
/// can swap the dense `WeightFfnGpu` for `WalkFfn` (Q4_0 sparse walk). This is
/// the entry point for the Gemma 3 walk-path that gets Metal KV-cached attention
/// alongside the sparse walk FFN — closes most of the speed gap between walk
/// and dense at 10× less RAM.
#[allow(clippy::too_many_arguments)]
pub fn predict_honest_with_knn_ffn(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    cached_layers: &CachedLayerGraph,
    layer_range: std::ops::Range<usize>,
    knn_store: Option<&larql_vindex::KnnStore>,
    ffn_override: Option<&dyn crate::ffn::FfnBackend>,
) -> crate::forward::PredictResult {
    let norm_offset = weights.arch.norm_weight_offset();

    // Pass 0: cached layers (legitimate — template-fixed)
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    for layer in 0..layer_range.start {
        if let Some(output) = cached_layers.forward_layer(weights, &h, layer) {
            h = output.residual;
        }
    }

    // GPU pipeline: decode (seq=1) uses decode_token/full_pipeline_q4,
    // prefill (seq>1) uses prefill_q4 for GPU-accelerated multi-position inference.
    let seq_len = h.shape()[0];
    let used_gpu = if backend.has_q4() {
        let gate_index: &dyn larql_vindex::GateIndex = index;
        // Prefer Q4_K FFN (Ollama-compatible) over Q4_0.
        // Q4_K_real mmap also qualifies: it's never read by the pre-norm
        // quant pipelines (those use q4_ffn_mmap for decode_token /
        // prefill_q4), but it makes used_gpu=true so the Gemma 3 post-norm
        // branches fire and pick up ffn_override. Without this, walk-only
        // with only Q4_K_real loaded falls to the CPU fallback which has no
        // KV cache between decode calls.
        // For pre-norm models the mmap is used as FFN weights by the quant
        // pipelines. For Gemma 3 (post_norm) we only need *some* mmap so
        // the post-norm branches fire (they pick up ffn_override for the
        // FFN itself) — any mmap works as a marker. Order: prefer real
        // quant sources so pre-norm models get their proper weights.
        // Prefer true Q4_K (interleaved_q4k_real.bin) over Q6_K
        // (interleaved_q4k.bin) for FFN weights. Gemma 3's post_ffn_norm has
        // weight max ≈ 304, which amplifies Q6_K quantisation noise to the
        // point where cos(GPU, CPU) = 0.84 per layer, compounding to garbage
        // across 34 layers. Q4_K (surprisingly) avoids this because its
        // block structure preserves the distribution post_ffn_norm needs.
        // Walk-FFN already uses Q4_K via q4k_ffn_full and works correctly.
        let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_real_mmap_ref() {
            (Some(mmap), true)
        } else if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
            (Some(mmap), true)
        } else if let Some(mmap) = gate_index.interleaved_q4_mmap_ref() {
            (Some(mmap), false)
        } else if weights.arch.has_post_norms() && ffn_override.is_some() {
            // Post-norm Gemma 3 + explicit FFN override (e.g. `--walk-only` with
            // LARQL_WALK_FORMAT=f32 loading only interleaved.bin). No quant mmap
            // exists, but we still want the KV-cached Gemma 3 branches — the
            // override handles every FFN call. Passing an empty byte slice here
            // is safe because `build_pipeline_layers` and decode_token/
            // prefill_q4 don't fire for post-norm models.
            (Some(&[][..]), true)
        } else {
            (None, false)
        };
        let has_q4k = index.attn_q4k_layer_data(layer_range.start).is_some();
        let has_q8 = index.attn_q8_layer_data(layer_range.start).is_some();

        if let Some(q4_ffn_mmap) = q4_ffn {
            let intermediate = gate_index.num_features(layer_range.start);
            let hidden = weights.hidden_size;
            if intermediate > 0 && (has_q4k || has_q8) {
                // Q4_K: 148B/256vals, Q4_0: 18B/32vals
                let q4_ffn_per_matrix = if ffn_is_q4k {
                    (intermediate * hidden).div_ceil(256) * 148
                } else {
                    intermediate * hidden / 32 * 18
                };
                // q4_ffn_per_layer computed inside build_pipeline_layers
                let ffn_format = if ffn_is_q4k { larql_compute::QuantFormat::Q4_K } else { larql_compute::QuantFormat::Q4_0 };
                let arch = &*weights.arch;

                // build_pipeline_layers does fixed byte-offset slicing into
                // q4_ffn_mmap. It's only read by decode_token / full_pipeline_q4
                // / prefill_q4 which never run for post-norm Gemma 3 models.
                // Skip the slicing entirely when we're on a post-norm model —
                // otherwise an empty/dummy mmap slice (the "walk-only with
                // override" case) would panic at layer offsetting.
                let force_quant_early = std::env::var("LARQL_FORCE_QUANT_DECODE")
                    .ok().as_deref() == Some("1");
                let layers: Vec<larql_compute::FullPipelineLayer> =
                    if arch.has_post_norms() && !force_quant_early {
                        Vec::new()
                    } else {
                        super::pipeline_layer::build_pipeline_layers(
                            weights, index, layer_range.clone(),
                            q4_ffn_mmap, q4_ffn_per_matrix, ffn_format,
                        )
                    };

                // GPU pipeline uses uniform dims (sliding layer defaults). Models with
                // per-layer variation (Gemma 4) route through CPU via has_post_norms().
                let q_dim = weights.num_q_heads * weights.head_dim;
                let kv_dim = weights.num_kv_heads * weights.head_dim;
                let rope = arch.rope_base_for_layer(layer_range.start) as f32;
                let softcap = arch.attn_logit_softcapping().unwrap_or(0.0);
                let qk_norm = arch.attn_q_norm_key(layer_range.start).is_some();

                // Gemma 3 (and any post-norm model): route single-token decode
                // through the same CPU-math + backend-matmul loop the multi-token
                // prefill uses below. This bypasses the Q4_K / Q4_KF quantised
                // decode_token path — necessary for Gemma 3 4B where the model
                // is too quant-sensitive to stay on-token at Q4/Q6 precision,
                // but the f32-weights + Metal-matmul path keeps fidelity and
                // still gets GPU speed on every matmul. Opt out with
                // LARQL_FORCE_QUANT_DECODE=1 if you need the quant path for
                // benchmarking or debugging.
                let force_quant = std::env::var("LARQL_FORCE_QUANT_DECODE")
                    .ok().as_deref() == Some("1");
                if seq_len == 1 && arch.has_post_norms() && !force_quant {
                    // Single-token Gemma 3 decode via f32 attention + Metal matmul
                    // + (either dense or walk) FFN. Reads past K/V from the Metal
                    // KV cache and appends the current token so subsequent calls
                    // see the full accumulated context.
                    use crate::ffn::WeightFfnGpu;
                    let dense_ffn = WeightFfnGpu { weights, backend };
                    let ffn: &dyn crate::ffn::FfnBackend = match ffn_override {
                        Some(f) => f,
                        None => &dense_ffn,
                    };
                    let mut h_cpu = h.clone();
                    let trace = std::env::var("LARQL_TRACE_LAYERS").ok().as_deref() == Some("1");
                    let trace_decode = std::env::var("LARQL_TRACE_DECODE").ok().as_deref() == Some("1");
                    let measure = trace_decode || crate::perf::is_enabled();
                    let mut t_attn_total_us = 0u128;
                    let mut t_ffn_total_us = 0u128;
                    let mut t_knn_total_us = 0u128;
                    // KNN overlay layers — precompute set (small, typically 1 layer).
                    let knn_layers: Vec<usize> = knn_store.map(|s| s.layers()).unwrap_or_default();
                    const KNN_COSINE_THRESHOLD: f32 = 0.75;
                    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                        let t_a = std::time::Instant::now();
                        let (h_post_attn, _past_len_after) =
                            crate::attention::gpu::run_attention_kv_cached_f32_opt(
                                weights, &h_cpu, abs_layer, rel_idx, backend,
                                Some(index as &dyn larql_vindex::GateIndex))
                                .unwrap();
                        if measure { t_attn_total_us += t_a.elapsed().as_micros(); }

                        // KNN overlay consult at the same residual stage the walk
                        // path captured: pre_ffn_norm(h_post_attn).
                        if !knn_layers.is_empty() && knn_layers.contains(&abs_layer) {
                            if let Some(store) = knn_store {
                                let pre_ffn_key = if arch.has_post_norms() {
                                    arch.pre_feedforward_layernorm_key(abs_layer)
                                } else {
                                    Some(arch.post_attention_layernorm_key(abs_layer))
                                };
                                let h_ffn = match pre_ffn_key {
                                    Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                                    None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
                                };
                                let residual: Vec<f32> = h_ffn.row(0).to_vec();
                                if let Some((entry, cosine)) = store.query_top1(abs_layer, &residual) {
                                    if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                                        eprintln!("[knn-decode L{}] top1={} cos={:.4} (threshold={})",
                                            abs_layer, entry.target_token, cosine, KNN_COSINE_THRESHOLD);
                                    }
                                    if cosine > KNN_COSINE_THRESHOLD {
                                        let label = format!(
                                            "{} (KNN override, cos={:.2}, L{})",
                                            entry.target_token, cosine, abs_layer,
                                        );
                                        return crate::forward::PredictResult {
                                            predictions: vec![(label, cosine as f64)],
                                            raw_predictions: vec![(entry.target_id, cosine, cosine as f64)],
                                        };
                                    }
                                }
                            }
                        }

                        let t_f = std::time::Instant::now();
                        let (h_out, _) = crate::forward::run_ffn(
                            weights, &h_post_attn, abs_layer, ffn, false);
                        if measure { t_ffn_total_us += t_f.elapsed().as_micros(); }
                        h_cpu = h_out;
                        if trace {
                            let pa_amax = h_post_attn.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                            let amax = h_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                            let nnan = h_cpu.iter().filter(|v| !v.is_finite()).count();
                            eprintln!("[kvdec-L{abs_layer:02}] h_post_attn={pa_amax:.2} h_out={amax:.2} nnan={nnan}");
                        }
                    }
                    h = h_cpu;
                    let t_lm = std::time::Instant::now();
                    let result = finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset);
                    let lm_us = t_lm.elapsed().as_micros();
                    if trace_decode {
                        eprintln!("[trace_decode] attn={:.2}ms ffn={:.2}ms knn_overlay={:.2}ms lm_head={:.2}ms",
                            t_attn_total_us as f64 / 1000.0,
                            t_ffn_total_us as f64 / 1000.0,
                            t_knn_total_us as f64 / 1000.0,
                            lm_us as f64 / 1000.0,
                        );
                    }
                    crate::perf::record("decode.attn",        t_attn_total_us);
                    crate::perf::record("decode.ffn",         t_ffn_total_us);
                    crate::perf::record("decode.knn_overlay", t_knn_total_us);
                    crate::perf::record("decode.lm_head",     lm_us);
                    return result;
                }
                if seq_len == 1 {
                    // Decode path (seq=1): try KV-cached decode first, then full_pipeline
                    let x: Vec<f32> = h.row(0).to_vec();
                    let trace_nan = std::env::var("LARQL_TRACE_NAN").ok().as_deref() == Some("1");

                    if trace_nan {
                        let in_nans = x.iter().filter(|v| !v.is_finite()).count();
                        let in_min = x.iter().copied().fold(f32::INFINITY, f32::min);
                        let in_max = x.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        eprintln!("[nan] decode_token INPUT: {} non-finite, range [{:.3}, {:.3}]", in_nans, in_min, in_max);
                    }

                    // KNN probe: if knn_store has entries, probe h_post_attn at the
                    // overlay layer. GPU runs the FULL 34-layer pipeline in one cmd
                    // buffer; the probe is a single residual_copy dispatch (~0.01 ms).
                    // After decode, CPU checks KNN on the probed residual. If cosine
                    // > threshold, override the prediction. Otherwise use the GPU result.
                    // Zero speed penalty — GPU never breaks to CPU mid-pipeline.
                    let knn_probe_layer: Option<usize> = knn_store
                        .and_then(|s| s.layers().into_iter().next());
                    const KNN_COSINE_THRESHOLD_GPU: f32 = 0.75;

                    if let Some((result, probe_h)) = backend.decode_token_with_probe(
                        &layers, &x, hidden, intermediate, q_dim, kv_dim,
                        weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
                        knn_probe_layer,
                    ) {
                        if trace_nan {
                            let n = result.iter().filter(|v| !v.is_finite()).count();
                            let mn = result.iter().copied().fold(f32::INFINITY, f32::min);
                            let mx = result.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                            eprintln!("[nan] decode_token OUTPUT: {} non-finite of {}, range [{:.3}, {:.3}]", n, result.len(), mn, mx);
                        }

                        // KNN overlay check on probed residual
                        if let (Some(probe), Some(store), Some(pl)) = (&probe_h, knn_store, knn_probe_layer) {
                            // Apply pre_ffn_norm to probe (same as CPU path's KNN check point)
                            let probe_arr = ndarray::Array2::from_shape_vec((1, hidden), probe.clone()).unwrap();
                            let pre_ffn_key = if arch.has_post_norms() {
                                arch.pre_feedforward_layernorm_key(pl)
                            } else {
                                Some(arch.post_attention_layernorm_key(pl))
                            };
                            let h_ffn = match pre_ffn_key {
                                Some(key) => crate::forward::apply_norm(weights, &probe_arr, &key, norm_offset),
                                None => crate::residual::rms_norm(&probe_arr, None, norm_offset),
                            };
                            let residual: Vec<f32> = h_ffn.row(0).to_vec();
                            if let Some((entry, cosine)) = store.query_top1(pl, &residual) {
                                if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                                    eprintln!("[knn-gpu-probe L{}] top1={} cos={:.4} (threshold={})",
                                        pl, entry.target_token, cosine, KNN_COSINE_THRESHOLD_GPU);
                                }
                                if cosine > KNN_COSINE_THRESHOLD_GPU {
                                    let label = format!(
                                        "{} (KNN override, cos={:.2}, L{}, GPU probe)",
                                        entry.target_token, cosine, pl,
                                    );
                                    return crate::forward::PredictResult {
                                        predictions: vec![(label, cosine as f64)],
                                        raw_predictions: vec![(entry.target_id, cosine, cosine as f64)],
                                    };
                                }
                            }
                        }

                        let mut row = h.row_mut(0);
                        for j in 0..hidden { row[j] = result[j]; }
                        if trace_nan {
                            let h_nans = h.iter().filter(|v| !v.is_finite()).count();
                            eprintln!("[nan] h-matrix before finalize_logits: {} non-finite", h_nans);
                        }
                        return finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset);
                    } else if trace_nan {
                        eprintln!("[nan] decode_token returned None");
                    }

                    if let Some(result) = backend.full_pipeline_q4(
                        &layers, &x, hidden, intermediate, q_dim, kv_dim,
                        1, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
                        rope, qk_norm, softcap,
                    ) {
                        let mut row = h.row_mut(0);
                        for j in 0..hidden { row[j] = result[j]; }
                        true
                    } else { false }
                } else if !arch.has_post_norms() {
                    // Prefill path (seq>1): GPU Q4 pipeline for pre-norm models.
                    // Post-norm (Gemma 3): dispatch_full_pipeline's post-norm handling
                    // produces NaN (different code path from decode_token which works).
                    // Falls through to CPU prefill below — 800ms one-time cost, correct.
                    let x: Vec<f32> = h.as_slice().unwrap_or(&[]).to_vec();

                    if let Some(result) = backend.prefill_q4(
                        &layers, &x, hidden, intermediate, q_dim, kv_dim,
                        seq_len, weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
                        rope, qk_norm, softcap,
                    ) {
                        // GPU prefill may return all positions (seq_len * hidden)
                        // or just the last position (hidden). Either way, copy
                        // the final position into h for the next decode step.
                        let n_positions = result.len() / hidden;
                        if n_positions >= seq_len {
                            for s in 0..seq_len {
                                let mut row = h.row_mut(s);
                                for j in 0..hidden { row[j] = result[s * hidden + j]; }
                            }
                        } else {
                            // Last position only — sufficient for decode continuation.
                            let last = result.len().saturating_sub(hidden);
                            let mut row = h.row_mut(seq_len - 1);
                            for j in 0..hidden { row[j] = result[last + j]; }
                        }

                        // GPU prefill already populated the KV cache internally via
                        // Metal RoPE + kv_cache_append. Skip CPU re-population when
                        // FFN weights aren't loaded (SKIP_FFN_LOAD walk-only mode).
                        // For non-walk-only, CPU re-population gives f32 precision.
                        if !force_quant_early {
                            prefill_kv_cache_cpu(weights, token_ids, index, backend, &layer_range);
                        }

                        true
                    } else { false }
                } else {
                    // Post-norm models (Gemma3): CPU prefill (correct) → GPU logits (fast)
                    // CPU handles post-norms correctly. Use CPU hidden state, GPU for logits only.
                    // KV cache populated for future decode_token calls (token generation).
                    backend.reset_kv_cache();

                    // Prefill FFN: swap the slow CPU walk (all ~348K features per
                    // layer) for the f32-weights + Metal-matmul FFN that the
                    // single-token decode branch already uses. Correctness is
                    // identical (f32 weights, same math) but per-layer FFN drops
                    // from ~2 s CPU to ~15 ms Metal. Opt out with
                    // LARQL_PREFILL_WALK_FFN=1 to restore the CPU walk (useful
                    // for side-by-side parity checks).
                    let use_walk_prefill = std::env::var("LARQL_PREFILL_WALK_FFN")
                        .ok().as_deref() == Some("1");
                    let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
                    let gpu_ffn = crate::ffn::WeightFfnGpu { weights, backend };
                    // When caller supplied ffn_override (e.g. a top_k-capped WalkFfn),
                    // use it instead of walk_ffn / gpu_ffn.
                    let override_ffn = ffn_override;
                    let mut h_cpu = h.clone();
                    let trace_layers = std::env::var("LARQL_TRACE_LAYERS").ok().as_deref() == Some("1");
                    if trace_layers {
                        let amax = h_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                        eprintln!("[trace-cpu] pre-L0 h amax={amax:.2}");
                    }
                    // KNN overlay layer set — precompute (small, typically [26]).
                    let knn_layers_prefill: Vec<usize> =
                        knn_store.map(|s| s.layers()).unwrap_or_default();
                    const KNN_COSINE_THRESHOLD: f32 = 0.75;
                    let trace_time = std::env::var("LARQL_TRACE_PREFILL_TIME")
                        .ok().as_deref() == Some("1");
                    let mut t_attn_total = 0u128;
                    let mut t_kv_total = 0u128;
                    let mut t_ffn_total = 0u128;
                    let mut t_knn_total = 0u128;
                    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                        let t_layer = std::time::Instant::now();
                        let t_a = std::time::Instant::now();
                        let (h_post_attn, k_rope, v) =
                            crate::attention::gpu::run_attention_with_kv_backend_opt(
                                weights, &h_cpu, abs_layer, Some(backend),
                                Some(index as &dyn larql_vindex::GateIndex))
                                .unwrap();
                        let attn_us = t_a.elapsed().as_micros();
                        t_attn_total += attn_us;

                        if backend.has_kv_cache() {
                            let t_kv = std::time::Instant::now();
                            let k_flat = k_rope.as_slice().unwrap_or(&[]);
                            let v_flat = v.as_slice().unwrap_or(&[]);
                            backend.populate_kv_layer(rel_idx, k_flat, v_flat,
                                seq_len, weights.num_kv_heads, weights.head_dim);
                            t_kv_total += t_kv.elapsed().as_micros();
                            // LARQL_VERIFY_KV=1: for layer 0, read cache back and compare first 8 floats.
                            if rel_idx == 0 && std::env::var("LARQL_VERIFY_KV").ok().as_deref() == Some("1") {
                                if let Some((k_back, v_back, n)) = backend.debug_read_kv_layer(0) {
                                    let kmatch8: Vec<bool> = (0..8.min(k_flat.len().min(k_back.len())))
                                        .map(|i| (k_flat[i] - k_back[i]).abs() < 1e-5).collect();
                                    let vmatch8: Vec<bool> = (0..8.min(v_flat.len().min(v_back.len())))
                                        .map(|i| (v_flat[i] - v_back[i]).abs() < 1e-5).collect();
                                    eprintln!("[verify_kv L0] cache_len={n} k_in.len={} k_back.len={} v_in.len={} v_back.len={}",
                                        k_flat.len(), k_back.len(), v_flat.len(), v_back.len());
                                    eprintln!("  k_in  [0..8]: {:?}", &k_flat[..8.min(k_flat.len())]);
                                    eprintln!("  k_back[0..8]: {:?}", &k_back[..8.min(k_back.len())]);
                                    eprintln!("  k_match[0..8]: {:?}", kmatch8);
                                    eprintln!("  v_in  [0..8]: {:?}", &v_flat[..8.min(v_flat.len())]);
                                    eprintln!("  v_back[0..8]: {:?}", &v_back[..8.min(v_back.len())]);
                                    eprintln!("  v_match[0..8]: {:?}", vmatch8);
                                    // Count total mismatches (>1e-5 diff)
                                    let k_mismatch = k_flat.iter().zip(k_back.iter())
                                        .filter(|(a, b)| (*a - *b).abs() >= 1e-5).count();
                                    let v_mismatch = v_flat.iter().zip(v_back.iter())
                                        .filter(|(a, b)| (*a - *b).abs() >= 1e-5).count();
                                    eprintln!("  k_mismatch_total={k_mismatch}  v_mismatch_total={v_mismatch}");
                                }
                            }
                        }

                        let t_k = std::time::Instant::now();
                        // KNN overlay consult at the same residual stage the walk
                        // path captured: pre_ffn_norm(h_post_attn). WalkFfn.forward
                        // receives `h_ffn` (post-pre_ffn_norm, pre-FFN), and that's
                        // what the INSERT stored as the key.
                        if !knn_layers_prefill.is_empty() && knn_layers_prefill.contains(&abs_layer) {
                            if let Some(store) = knn_store {
                                let arch = &*weights.arch;
                                let pre_ffn_key = if arch.has_post_norms() {
                                    arch.pre_feedforward_layernorm_key(abs_layer)
                                } else {
                                    Some(arch.post_attention_layernorm_key(abs_layer))
                                };
                                let h_ffn = match pre_ffn_key {
                                    Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                                    None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
                                };
                                let last = h_ffn.shape()[0] - 1;
                                let residual: Vec<f32> = h_ffn.row(last).to_vec();
                                if let Some((entry, cosine)) = store.query_top1(abs_layer, &residual) {
                                    if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                                        eprintln!("[knn-prefill L{}] top1={} cos={:.4} (threshold={})",
                                            abs_layer, entry.target_token, cosine, KNN_COSINE_THRESHOLD);
                                    }
                                    if cosine > KNN_COSINE_THRESHOLD {
                                        let label = format!(
                                            "{} (KNN override, cos={:.2}, L{})",
                                            entry.target_token, cosine, abs_layer,
                                        );
                                        return crate::forward::PredictResult {
                                            predictions: vec![(label, cosine as f64)],
                                            raw_predictions: vec![(entry.target_id, cosine, cosine as f64)],
                                        };
                                    }
                                }
                            }
                        }

                        t_knn_total += t_k.elapsed().as_micros();

                        let t_f = std::time::Instant::now();
                        let (h_out, _) = if let Some(f) = override_ffn {
                            crate::forward::run_ffn(weights, &h_post_attn, abs_layer, f, false)
                        } else if use_walk_prefill {
                            crate::forward::run_ffn(weights, &h_post_attn, abs_layer, &walk_ffn, false)
                        } else {
                            crate::forward::run_ffn(weights, &h_post_attn, abs_layer, &gpu_ffn, false)
                        };
                        t_ffn_total += t_f.elapsed().as_micros();
                        h_cpu = h_out;
                        if trace_layers {
                            let pa_amax = h_post_attn.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                            let amax = h_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
                            eprintln!("[trace-cpu] L{abs_layer:02} h_post_attn amax={pa_amax:.2}  h_out amax={amax:.2}");
                        }
                        if trace_time {
                            let layer_ms = t_layer.elapsed().as_secs_f64() * 1000.0;
                            eprintln!("[time L{abs_layer:02}] layer={layer_ms:.1}ms (attn={attn_us}µs)");
                        }
                    }
                    if trace_time {
                        eprintln!("[time totals] attn={:.0}ms kv={:.0}ms ffn={:.0}ms knn={:.0}ms",
                            t_attn_total as f64/1000.0, t_kv_total as f64/1000.0,
                            t_ffn_total as f64/1000.0, t_knn_total as f64/1000.0);
                    }

                    // Use correct CPU hidden state, finalize with GPU logits
                    h = h_cpu;
                    return finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset);
                }
            } else { false }
        } else { false }
    } else { false };

    // CPU fallback: interleaved attention + FFN (for prefill or when GPU not available)
    // Only fires when the Gemma 3 post-norm branch above couldn't run (no Q4 mmap).
    // Works for pre-norm models; post-norm Gemma 3 should have entered the post-norm
    // branches via the Q4_K_real mmap acceptance above.
    if !used_gpu {
        let walk_ffn = crate::vindex::WalkFfn::new_unlimited(weights, index);
        for layer in layer_range {
            let (h_post_attn, _, _) =
                crate::attention::run_attention_block_gpu(weights, &h, layer, false, None)
                    .unwrap();
            let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, layer, &walk_ffn, false);
            h = h_out;
        }
    }

    finalize_logits(weights, tokenizer, &h, top_k, index, backend, norm_offset)
}

// generate(), GenerateResult, and softmax_prob moved to generate.rs and logits.rs

/// Optimized predict: uses vindex logits when lm_head is loaded, falls back to full matmul.
///
/// This is the production entry point. It:
/// 1. Runs embedding → layer loop via LayerGraph
/// 2. Uses vindex lm_head KNN if available (eliminates 226ms logits matmul)
/// 3. Falls back to full vocab matmul if no lm_head loaded
pub fn predict_pipeline(
    weights: &ModelWeights,
    tokenizer: &tokenizers::Tokenizer,
    token_ids: &[u32],
    top_k: usize,
    graph: &dyn LayerGraph,
    index: Option<&larql_vindex::VectorIndex>,
) -> crate::forward::PredictResult {
    // Use vindex logits if lm_head is loaded
    if let Some(idx) = index {
        if idx.has_lm_head() {
            return predict_with_graph_vindex_logits(weights, tokenizer, token_ids, top_k, graph, idx);
        }
    }
    // Fallback: full vocab matmul
    predict_with_graph(weights, tokenizer, token_ids, top_k, graph)
}

/// Run a full forward pass with tracing (residuals + activations + attention).
pub fn trace_with_graph(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
    graph: &dyn LayerGraph,
) -> crate::forward::TraceResult {
    let seq_len = token_ids.len();
    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    let mut results = Vec::new();
    let mut activations = Vec::new();
    let mut attention_captures = Vec::new();

    for layer in 0..=max_layer.min(weights.num_layers - 1) {
        match graph.forward_layer(weights, &h, layer) {
            Some(output) => {
                h = output.residual;

                if capture_layers.contains(&layer) {
                    let last_row = h.row(seq_len - 1);
                    results.push((layer, last_row.to_vec()));

                    if let Some(act) = output.activation {
                        let act_row = act.row(seq_len - 1);
                        let mut indexed: Vec<(usize, f32)> = act_row.iter().copied().enumerate().collect();
                        indexed.sort_unstable_by(|a, b| b.1.abs().partial_cmp(&a.1.abs()).unwrap());
                        indexed.truncate(200);
                        activations.push((layer, indexed));
                    }

                    if let Some(attn) = output.attention {
                        attention_captures.push(crate::forward::LayerAttentionCapture {
                            layer,
                            weights: attn,
                        });
                    }
                }
            }
            None => break,
        }
    }

    crate::forward::TraceResult {
        residuals: results,
        activations,
        attention: attention_captures,
    }
}
