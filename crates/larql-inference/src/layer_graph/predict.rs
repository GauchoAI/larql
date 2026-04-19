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

// Removed from predict.rs (only used by bench examples, not server):
// - predict_with_graph_vindex_logits (57 lines) — LayerGraph + vindex logits
// - predict_with_graph (32 lines) — LayerGraph generic loop
// - predict_split_pass (164 lines) — two-pass split attn/FFN pipeline
// - predict_split_cached (65 lines) — cached attention + GPU FFN batch

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
    let arch = &*weights.arch;
    if !arch.has_post_norms() { return None; }
    if target_layer >= weights.num_layers { return None; }
    let norm_offset = arch.norm_weight_offset();
    let mut h = crate::forward::embed_tokens_pub(weights, token_ids);
    backend.reset_kv_cache();
    let gpu_ffn = crate::ffn::WeightFfnGpu { weights, backend };
    for abs_layer in 0..=target_layer {
        let (h_post_attn, k_rope, v) =
            crate::attention::gpu::run_attention_with_kv_backend(
                weights, &h, abs_layer, Some(backend),
            )?;
        if backend.has_kv_cache() {
            let seq_len = h.shape()[0];
            let k_flat = k_rope.as_slice().unwrap_or(&[]);
            let v_flat = v.as_slice().unwrap_or(&[]);
            backend.populate_kv_layer(abs_layer, k_flat, v_flat,
                seq_len, weights.num_kv_heads, weights.head_dim);
        }
        if abs_layer == target_layer {
            let pre_ffn_key = arch.pre_feedforward_layernorm_key(abs_layer);
            let h_ffn = match pre_ffn_key {
                Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
            };
            let last = h_ffn.shape()[0] - 1;
            return Some(h_ffn.row(last).to_vec());
        }
        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, abs_layer, &gpu_ffn, false);
        h = h_out;
    }
    None
}

/// Capture the normed probe residual matching the EXACT inference path:
/// Q4_K decode_token for prefill (populates KV cache), then per-layer
/// path for the last token (matching how inference does value-injection
/// decode). This ensures KV cache entries and residuals are identical.
pub fn capture_knn_key_perlayer(
    weights: &ModelWeights,
    token_ids: &[u32],
    probe_layer: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
    ffn_override: Option<&dyn crate::ffn::FfnBackend>,
) -> Option<Vec<f32>> {
    let arch = &*weights.arch;
    if !arch.has_post_norms() { return None; }
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();

    // Step 1: Prefill via Q4_K decode_token (same as inference prefill).
    // This populates the Metal KV cache with Q4_K-computed K/V entries.
    // Use capture_knn_key_gpu's pipeline setup but DON'T capture the probe —
    // just populate the KV cache.
    let gate_index: &dyn larql_vindex::GateIndex = index;
    let intermediate = gate_index.num_features(0);
    let has_q4k = gate_index.attn_q4k_layer_data(0).is_some();
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_real_mmap_ref() {
        (Some(mmap), true)
    } else if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else if let Some(mmap) = gate_index.interleaved_q4_mmap_ref() {
        (Some(mmap), false)
    } else {
        return None;
    };
    let q4_ffn = q4_ffn?;
    if !has_q4k || intermediate == 0 { return None; }

    let q4_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 148
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let layers = super::pipeline_layer::build_pipeline_layers(
        weights, index, 0..weights.num_layers,
        q4_ffn, q4_per_matrix, ffn_format,
    );
    if layers.is_empty() { return None; }

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    backend.reset_kv_cache();
    let embeds = crate::forward::embed_tokens_pub(weights, token_ids);
    let seq_len = token_ids.len();

    // Prefill: Q4_K decode_token for all tokens (populates KV cache)
    for p in 0..seq_len {
        let x: Vec<f32> = embeds.row(p).to_vec();
        backend.decode_token_with_probe(
            &layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            None, // no probe during prefill
        );
    }

    // Step 2: Run ONE more decode step through the per-layer path to
    // capture the residual. Use the SAME function as inference decode.
    // The KV cache now has Q4_K entries from prefill — per-layer reads them.
    use crate::ffn::WeightFfnGpu;
    let dense_ffn = WeightFfnGpu { weights, backend };
    let ffn: &dyn crate::ffn::FfnBackend = match ffn_override {
        Some(f) => f,
        None => &dense_ffn,
    };

    // The "next token" after prefill — use the embedding of the last prompt token
    // as input (this mirrors how inference starts the decode loop after prefill).
    let last_embed: Vec<f32> = embeds.row(seq_len - 1).to_vec();
    let mut h_tok = ndarray::Array2::from_shape_vec((1, hidden), last_embed).unwrap();

    // Actually — after Q4_K prefill, the model already computed the output for
    // all prefill tokens. The NEXT decode step would use the sampled first token.
    // But we don't have a sampled token. Instead, we capture the residual
    // at the LAST prefill position by running the per-layer path WITH the
    // Q4_K-populated KV cache. The KV cache already has seq_len entries.
    // We remove the last entry so the per-layer path re-processes position
    // seq_len-1 with its own attention, reading K/V[0..seq_len-2] from cache.
    backend.rollback_kv_cache(1); // remove last position from Q4_K prefill

    let last_x = embeds.row(seq_len - 1).to_vec();
    let mut h_decode = ndarray::Array2::from_shape_vec((1, hidden), last_x).unwrap();

    for (rel_idx, abs_layer) in (0..weights.num_layers).enumerate() {
        let (h_post_attn, _) =
            crate::attention::gpu::run_attention_kv_cached_f32_opt(
                weights, &h_decode, abs_layer, rel_idx, backend,
                Some(index as &dyn larql_vindex::GateIndex),
            )?;

        if abs_layer == probe_layer {
            let pre_ffn_key = if arch.has_post_norms() {
                arch.pre_feedforward_layernorm_key(abs_layer)
            } else {
                Some(arch.post_attention_layernorm_key(abs_layer))
            };
            let h_ffn = match pre_ffn_key {
                Some(key) => crate::forward::apply_norm(weights, &h_post_attn, &key, norm_offset),
                None => crate::residual::rms_norm(&h_post_attn, None, norm_offset),
            };
            backend.reset_kv_cache();
            return Some(h_ffn.row(0).to_vec());
        }

        let (h_out, _) = crate::forward::run_ffn(weights, &h_post_attn, abs_layer, ffn, false);
        h_decode = h_out;
    }

    backend.reset_kv_cache();
    None
}

/// Capture the normed probe residual at `probe_layer` using the SAME GPU
/// path as `predict_honest_with_knn_ffn` (Q4_K decode_token_with_probe).
/// Returns the pre_ffn_norm(h_post_attn) vector — identical to what the
/// KNN overlay checks during inference. Use for INSERT so stored keys
/// match the inference path exactly.
///
/// Resets the KV cache, runs GPU sequential prefill through all prompt
/// tokens, captures the probe on the last token.
pub fn capture_knn_key_gpu(
    weights: &ModelWeights,
    token_ids: &[u32],
    probe_layer: usize,
    index: &larql_vindex::VectorIndex,
    backend: &dyn ComputeBackend,
) -> Option<Vec<f32>> {
    let arch = &*weights.arch;
    if !arch.has_post_norms() { return None; }
    let hidden = weights.hidden_size;
    let norm_offset = arch.norm_weight_offset();

    let gate_index: &dyn larql_vindex::GateIndex = index;
    let intermediate = gate_index.num_features(0);
    let has_q4k = gate_index.attn_q4k_layer_data(0).is_some();

    // Find Q4_K FFN mmap — same priority as predict_honest
    let (q4_ffn, ffn_is_q4k) = if let Some(mmap) = gate_index.interleaved_q4k_real_mmap_ref() {
        (Some(mmap), true)
    } else if let Some(mmap) = gate_index.interleaved_q4k_mmap_ref() {
        (Some(mmap), true)
    } else if let Some(mmap) = gate_index.interleaved_q4_mmap_ref() {
        (Some(mmap), false)
    } else {
        return None;
    };
    let q4_ffn = q4_ffn?;
    if !has_q4k || intermediate == 0 { return None; }

    let q4_per_matrix = if ffn_is_q4k {
        (intermediate * hidden).div_ceil(256) * 148
    } else {
        intermediate * hidden / 32 * 18
    };
    let ffn_format = if ffn_is_q4k {
        larql_compute::QuantFormat::Q4_K
    } else {
        larql_compute::QuantFormat::Q4_0
    };
    let layers = super::pipeline_layer::build_pipeline_layers(
        weights, index, 0..weights.num_layers,
        q4_ffn, q4_per_matrix, ffn_format,
    );
    if layers.is_empty() { return None; }

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;
    let rope = arch.rope_base_for_layer(0) as f32;

    backend.reset_kv_cache();
    let embeds = crate::forward::embed_tokens_pub(weights, token_ids);
    let seq_len = token_ids.len();

    for p in 0..seq_len {
        let x: Vec<f32> = embeds.row(p).to_vec();
        let is_last = p == seq_len - 1;
        let probe = if is_last { Some(probe_layer) } else { None };
        if let Some((_result, probe_h)) = backend.decode_token_with_probe(
            &layers, &x, hidden, intermediate, q_dim, kv_dim,
            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
            probe,
        ) {
            if let Some(ph) = probe_h {
                // Apply pre_ffn_norm — same norm the KNN check uses
                let probe_arr = ndarray::Array2::from_shape_vec((1, hidden), ph).unwrap();
                let pre_ffn_key = if arch.has_post_norms() {
                    arch.pre_feedforward_layernorm_key(probe_layer)
                } else {
                    Some(arch.post_attention_layernorm_key(probe_layer))
                };
                let h_ffn = match pre_ffn_key {
                    Some(key) => crate::forward::apply_norm(weights, &probe_arr, &key, norm_offset),
                    None => crate::residual::rms_norm(&probe_arr, None, norm_offset),
                };
                backend.reset_kv_cache();
                return Some(h_ffn.row(0).to_vec());
            }
        }
    }
    backend.reset_kv_cache();
    None
}

// Removed: capture_residual_post_attn_norm_ffn (59 lines) — only used by bench examples

// Removed: predict_honest (24 lines) — thin wrapper around predict_honest_with_knn_ffn
// Removed: predict_honest_with_knn (23 lines) — thin wrapper around predict_honest_with_knn_ffn

/// Production inference entry point. Runs all layers with real computation,
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
    let trace_path = std::env::var("LARQL_TRACE_PATH").ok().as_deref() == Some("1");
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
                // Auto-detect GPU decode: when both Q4_K attention weights AND
                // Q4_K real FFN weights exist, use GPU decode by default.
                // Override with LARQL_CPU_DECODE=1 to force CPU path.
                let cpu_forced = std::env::var("LARQL_CPU_DECODE")
                    .ok().as_deref() == Some("1");
                let force_quant_env = std::env::var("LARQL_FORCE_QUANT_DECODE")
                    .ok().as_deref() == Some("1");
                let force_quant_early = force_quant_env
                    || (!cpu_forced && has_q4k && gate_index.interleaved_q4k_real_mmap_ref().is_some());
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
                let force_quant = force_quant_early;
                // Force per-layer path when value-injection KNN entries exist,
                // since decode_token can't inject mid-pipeline.
                let has_value_inject = knn_store.map(|s|
                    s.entries().values().any(|entries|
                        entries.iter().any(|e| e.value_vector.is_some())
                    )
                ).unwrap_or(false);
                if seq_len == 1 && arch.has_post_norms() && (!force_quant || has_value_inject) {
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
                    let knn_layers: Vec<usize> = knn_store.map(|s| s.layers()).unwrap_or_default();
                    const KNN_COSINE_THRESHOLD: f32 = 0.75;
                    // Pending value injection: (value_vector, value_layer, label)
                    let mut pending_inject: Option<(Vec<f32>, usize, String)> = None;
                    for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                        let t_a = std::time::Instant::now();
                        let (mut h_post_attn, _past_len_after) =
                            crate::attention::gpu::run_attention_kv_cached_f32_opt(
                                weights, &h_cpu, abs_layer, rel_idx, backend,
                                Some(index as &dyn larql_vindex::GateIndex))
                                .unwrap();
                        if measure { t_attn_total_us += t_a.elapsed().as_micros(); }

                        // Value injection: blend stored vector into residual at value_layer
                        if let Some((ref vv, vl, ref label)) = pending_inject {
                            if abs_layer == vl {
                                let hidden = h_post_attn.shape()[1];
                                let mut row = h_post_attn.row_mut(0);
                                for j in 0..hidden.min(vv.len()) {
                                    row[j] = 0.5 * row[j] + 0.5 * vv[j]; // blend=0.5
                                }
                                if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                                    eprintln!("[knn-inject L{}] value injected for: {}", vl, label);
                                }
                                pending_inject = None;
                            }
                        }

                        // KNN overlay: query matching + token override / value injection setup
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
                                        eprintln!("[knn-decode L{}] top1={} cos={:.4} (threshold={}) mode={}",
                                            abs_layer, entry.target_token, cosine, KNN_COSINE_THRESHOLD,
                                            if entry.value_vector.is_some() { "inject" } else { "override" });
                                    }
                                    if cosine > KNN_COSINE_THRESHOLD {
                                        // Value injection mode: schedule injection at value_layer
                                        if let (Some(vv), Some(vl)) = (&entry.value_vector, entry.value_layer) {
                                            pending_inject = Some((
                                                vv.clone(), vl, entry.target_token.clone(),
                                            ));
                                            // Don't return — continue the layer loop
                                        } else {
                                            // Token override mode (existing)
                                            let label = format!(
                                                "{} (KNN override, cos={:.2}, L{})",
                                                entry.target_token, cosine, abs_layer,
                                            );
                                            return crate::forward::PredictResult {
                                                predictions: vec![(label, cosine as f64)],
                                                raw_predictions: vec![(entry.target_id, cosine, cosine as f64)],
                                                knn_override: Some(entry.target_token.clone()), h_final: None,
                                            };
                                        }
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
                    // Decode-time probe threshold: keep high (0.75) to avoid
                    // false positives during generation. The prefill KNN check
                    // (separate path) uses a lower threshold for Q4_K captures.
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
                                        knn_override: Some(entry.target_token.clone()), h_final: None,
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
                } else if force_quant_early && arch.has_post_norms() {
                    // GPU prefill for post-norm models (Gemma 3).
                    // Skip reset if KV cache already has precomputed context.
                    let (_, _, existing_len) = backend.debug_read_kv_layer(0)
                        .unwrap_or((Vec::new(), Vec::new(), 0));
                    if existing_len == 0 {
                        backend.reset_kv_cache();
                    }
                    let embeds = crate::forward::embed_tokens_pub(weights, token_ids);
                    let t_prefill = std::time::Instant::now();
                    let mut last_h = vec![0.0f32; hidden];
                    let knn_probe_layer_prefill: Option<usize> = knn_store
                        .and_then(|s| s.layers().into_iter().next());
                    let mut prefill_probe: Option<Vec<f32>> = None;

                    // Batch prefill: process all tokens through all layers in
                    // one Metal command buffer. ~24ms total vs ~24ms × seq_len
                    // for sequential. Falls back to sequential if batch fails.
                    let batch_ok = if seq_len > 1 {
                        let x_batch: Vec<f32> = embeds.as_slice().unwrap_or(&[]).to_vec();
                        if let Some(h_out) = backend.decode_token_batch(
                            &layers, &x_batch, seq_len,
                            hidden, intermediate, q_dim, kv_dim,
                            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
                        ) {
                            // Extract last position's hidden state
                            let last_off = (seq_len - 1) * hidden;
                            last_h = h_out[last_off..last_off + hidden].to_vec();
                            let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
                            if std::env::var("LARQL_TRACE_PREFILL_TIME").ok().as_deref() == Some("1")
                                || seq_len > 5 {
                                eprintln!("[gpu-batch-prefill] {seq_len} tokens in {prefill_ms:.0}ms ({:.1}ms/tok, existing_kv={existing_len})",
                                    prefill_ms / seq_len as f64);
                            }
                            true
                        } else { false }
                    } else { false };

                    // Sequential fallback (single token or batch unavailable)
                    if !batch_ok {
                    for p in 0..seq_len {
                        let x: Vec<f32> = embeds.row(p).to_vec();
                        let is_last = p == seq_len - 1;
                        let probe = if is_last { knn_probe_layer_prefill } else { None };
                        if let Some((result, ph)) = backend.decode_token_with_probe(
                            &layers, &x, hidden, intermediate, q_dim, kv_dim,
                            weights.num_q_heads, weights.num_kv_heads, weights.head_dim, rope,
                            probe,
                        ) {
                            last_h = result;
                            if ph.is_some() { prefill_probe = ph; }
                        }
                    }
                    }
                    // KNN override check on prefill probe
                    if let (Some(probe), Some(store), Some(pl)) = (&prefill_probe, knn_store, knn_probe_layer_prefill) {
                        let probe_arr = ndarray::Array2::from_shape_vec((1, hidden), probe.clone()).unwrap();
                        let pre_ffn_key = if arch.has_post_norms() {
                            arch.pre_feedforward_layernorm_key(pl)
                        } else { Some(arch.post_attention_layernorm_key(pl)) };
                        let h_ffn = match pre_ffn_key {
                            Some(key) => crate::forward::apply_norm(weights, &probe_arr, &key, norm_offset),
                            None => probe_arr,
                        };
                        let residual: Vec<f32> = h_ffn.row(0).to_vec();
                        const KNN_PREFILL_THRESHOLD: f32 = 0.70;
                        if let Some((entry, cosine)) = store.query_top1(pl, &residual) {
                            if std::env::var("LARQL_TRACE_KNN").ok().as_deref() == Some("1") {
                                eprintln!("[knn-gpu-prefill L{}] top1={} cos={:.4} (threshold={})",
                                    pl, entry.target_token, cosine, KNN_PREFILL_THRESHOLD);
                            }
                            if cosine > KNN_PREFILL_THRESHOLD {
                                let label = format!(
                                    "{} (KNN override, cos={:.2}, L{}, GPU prefill)",
                                    entry.target_token, cosine, pl,
                                );
                                return crate::forward::PredictResult {
                                    predictions: vec![(label, cosine as f64)],
                                    raw_predictions: vec![(entry.target_id, cosine, cosine as f64)],
                                    knn_override: Some(entry.target_token.clone()), h_final: None,
                                };
                            }
                        }
                    }
                    let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
                    if std::env::var("LARQL_TRACE_PREFILL_TIME").ok().as_deref() == Some("1") {
                        eprintln!("[gpu-seq-prefill] {seq_len} tokens in {prefill_ms:.0}ms ({:.1}ms/tok)",
                            prefill_ms / seq_len as f64);
                    }
                    // Copy final h into the h matrix for finalize_logits
                    let mut row = h.row_mut(seq_len - 1);
                    for j in 0..hidden { row[j] = last_h[j]; }
                    true
                } else {
                    // Post-norm models (Gemma3): CPU prefill (correct) → GPU logits (fast)
                    // Skip reset if KV cache already has precomputed context.
                    let (_, _, existing_len) = backend.debug_read_kv_layer(0)
                        .unwrap_or((Vec::new(), Vec::new(), 0));
                    if existing_len == 0 {
                        backend.reset_kv_cache();
                    }

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

                    // When KV cache has precomputed context (from KV replay),
                    // process each user token one at a time through KV-cached
                    // attention. This correctly reads past K/V and appends at
                    // position [existing_len + i], preserving the precomputed
                    // context. The multi-token `run_attention_with_kv_backend_opt`
                    // ignores the KV cache entirely and would overwrite it.
                    if existing_len > 0 {
                        use crate::ffn::WeightFfnGpu;
                        let dense_ffn = WeightFfnGpu { weights, backend };
                        let ffn: &dyn crate::ffn::FfnBackend = match ffn_override {
                            Some(f) => f,
                            None => &dense_ffn,
                        };
                        let embeds = crate::forward::embed_tokens_pub(weights, token_ids);
                        let t_prefill = std::time::Instant::now();
                        eprintln!("[kv-replay-prefill] processing {} user tokens with {} precomputed KV entries",
                            seq_len, existing_len);
                        for p in 0..seq_len {
                            let h_tok = embeds.slice(ndarray::s![p..p+1, ..]).to_owned();
                            for (rel_idx, abs_layer) in layer_range.clone().enumerate() {
                                let h_in = if rel_idx == 0 { &h_tok } else { &h_cpu };
                                let (h_post_attn, _past_len) =
                                    crate::attention::gpu::run_attention_kv_cached_f32_opt(
                                        weights, h_in, abs_layer, rel_idx, backend,
                                        Some(index as &dyn larql_vindex::GateIndex))
                                        .unwrap();
                                let (h_out, _) = crate::forward::run_ffn(
                                    weights, &h_post_attn, abs_layer, ffn, false);
                                h_cpu = h_out;
                            }
                        }
                        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1000.0;
                        eprintln!("[kv-replay-prefill] {} tokens in {:.0}ms ({:.1}ms/tok)",
                            seq_len, prefill_ms, prefill_ms / seq_len as f64);
                        let hidden = h.shape()[1];
                        let mut row = h.row_mut(seq_len - 1);
                        for j in 0..hidden { row[j] = h_cpu[[0, j]]; }
                        true
                    } else {
                    // Standard multi-token prefill (no precomputed KV).
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
                                            knn_override: Some(entry.target_token.clone()), h_final: None,
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
                    } // end of else (standard prefill, no precomputed KV)
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

// Removed from predict.rs (only used by bench examples, not server):
// - predict_with_graph_vindex_logits (57 lines) — LayerGraph + vindex logits
// - predict_with_graph (32 lines) — LayerGraph generic loop
// - predict_split_pass (164 lines) — two-pass split attn/FFN pipeline
// - predict_split_cached (65 lines) — cached attention + GPU FFN batch
// - predict_honest (24 lines) — wrapper for predict_honest_with_knn_ffn
// - predict_honest_with_knn (23 lines) — wrapper for predict_honest_with_knn_ffn
// - capture_residual_post_attn_norm_ffn (59 lines) — inlined into capture_residual_post_attn_norm
// - predict_pipeline (19 lines) — vindex logits wrapper
// - trace_with_graph (48 lines) — residual+activation tracing
