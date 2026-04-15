//! GPU-accelerated attention — routes projections through ComputeBackend.
//!
//! Falls back to CPU BLAS when backend is None.
//! Also includes Q4 quantized attention projection and KV-capture attention.

use ndarray::Array2;
use super::AttentionWeights;
use super::rope::apply_rope_partial;
use super::gqa::gqa_attention_with_weights;

/// GPU-accelerated attention block. Same as `run_attention_block` but routes
/// Q/K/V/O projections through the ComputeBackend (Metal, CUDA, or CPU).
pub fn run_attention_block_gpu(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    capture_attention: bool,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, Array2<f32>, Option<AttentionWeights>)> {
    use larql_compute::dot_proj_gpu;
    use crate::forward::add_bias;
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let head_dim = arch.head_dim_for_layer(layer);
    let num_q = arch.num_q_heads_for_layer(layer);
    let num_kv = arch.num_kv_heads_for_layer(layer);
    let reps = num_q / num_kv;
    let scale = if arch.attention_multiplier() != 1.0 {
        arch.attention_multiplier() as f64
    } else {
        arch.attention_scale_for_layer(layer)
    };
    let seq_len = h.shape()[0];
    let norm_offset = arch.norm_weight_offset();

    let h_norm = crate::forward::apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_offset);

    let w_q = weights.tensors.get(&arch.attn_q_key(layer))?;
    let w_k = weights.tensors.get(&arch.attn_k_key(layer)).unwrap();
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let w_v = if v_from_k { w_k } else { weights.tensors.get(&arch.attn_v_key(layer)).unwrap() };
    let w_o = weights.tensors.get(&arch.attn_o_key(layer)).unwrap();

    let mut q_full = dot_proj_gpu(&h_norm, w_q, backend);
    let mut k_full = dot_proj_gpu(&h_norm, w_k, backend);
    let mut v_full = dot_proj_gpu(&h_norm, w_v, backend);

    if let Some(bias) = arch.attn_q_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut q_full, bias);
    }
    if let Some(bias) = arch.attn_k_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut k_full, bias);
    }
    if let Some(bias) = arch.attn_v_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut v_full, bias);
    }

    if arch.has_v_norm() {
        v_full = rms_norm_heads_no_weight(&v_full, num_kv, head_dim);
    }

    let qk_offset = weights.arch.qk_norm_weight_offset();
    let qk_norm_off = if qk_offset != 0.0 { qk_offset } else { norm_offset };
    let q_normed = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim, qk_norm_off),
        None => q_full,
    };
    let k_normed = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim, qk_norm_off),
        None => k_full,
    };

    let layer_rope_base = arch.rope_base_for_layer(layer);
    let rotary_frac = arch.rotary_fraction_for_layer(layer);
    let q_rope = apply_rope_partial(&q_normed, num_q, head_dim, layer_rope_base, rotary_frac);
    let k_rope = apply_rope_partial(&k_normed, num_kv, head_dim, layer_rope_base, rotary_frac);

    let softcap = arch.attn_logit_softcapping();
    let (attn_out, attn_weights) = gqa_attention_with_weights(
        &q_rope, &k_rope, &v_full, num_q, head_dim, reps, scale, seq_len,
        capture_attention, softcap,
    );

    let mut attn_projected = dot_proj_gpu(&attn_out, w_o, backend);
    if let Some(bias) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) {
        add_bias(&mut attn_projected, bias);
    }

    let res_mult = arch.residual_multiplier();
    let h_post_attn = if arch.has_post_norms() {
        let normed = crate::forward::apply_norm(
            weights, &attn_projected, &arch.post_attention_layernorm_key(layer), norm_offset,
        );
        if res_mult != 1.0 { h + &(&normed * res_mult) } else { h + &normed }
    } else if res_mult != 1.0 {
        h + &(&attn_projected * res_mult)
    } else {
        h + &attn_projected
    };

    Some((h_post_attn, attn_projected, attn_weights))
}

/// Run attention and return K (post-RoPE) and V for KV cache population.
/// Accepts optional ComputeBackend for GPU-accelerated projections.
pub fn run_attention_with_kv(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    run_attention_with_kv_backend(weights, h, layer, None)
}

/// Run attention with optional compute backend for accelerated projections.
pub fn run_attention_with_kv_backend(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>)> {
    use crate::forward::{apply_norm, add_bias};
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};

    let arch = &*weights.arch;
    let hd = arch.head_dim_for_layer(layer);
    let nq = arch.num_q_heads_for_layer(layer);
    let nkv = arch.num_kv_heads_for_layer(layer);
    let reps = nq / nkv;
    let scale = if arch.attention_multiplier() != 1.0 { arch.attention_multiplier() as f64 } else { arch.attention_scale_for_layer(layer) };
    let seq_len = h.shape()[0];
    let norm_off = arch.norm_weight_offset();

    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_off);
    let wq = weights.tensors.get(&arch.attn_q_key(layer))?;
    let wk = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let wv = if v_from_k { wk } else { weights.tensors.get(&arch.attn_v_key(layer))? };
    let wo = weights.tensors.get(&arch.attn_o_key(layer))?;

    let (mut q, mut k, mut v) = (
        larql_compute::dot_proj_gpu(&h_norm, wq, backend),
        larql_compute::dot_proj_gpu(&h_norm, wk, backend),
        larql_compute::dot_proj_gpu(&h_norm, wv, backend),
    );
    for (proj, bias_fn) in [(&mut q, arch.attn_q_bias_key(layer) as Option<String>),
                             (&mut k, arch.attn_k_bias_key(layer)),
                             (&mut v, arch.attn_v_bias_key(layer))] {
        if let Some(b) = bias_fn.and_then(|key| weights.vectors.get(&key)) { add_bias(proj, b); }
    }

    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, nkv, hd);
    }

    let qk_off = if arch.qk_norm_weight_offset() != 0.0 { arch.qk_norm_weight_offset() } else { norm_off };
    let q = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&q, w, nq, hd, qk_off), None => q,
    };
    let k = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&k, w, nkv, hd, qk_off), None => k,
    };

    let rb = arch.rope_base_for_layer(layer);
    let rf = arch.rotary_fraction_for_layer(layer);
    let q_r = apply_rope_partial(&q, nq, hd, rb, rf);
    let k_r = apply_rope_partial(&k, nkv, hd, rb, rf);

    let (attn_out, _) = gqa_attention_with_weights(
        &q_r, &k_r, &v, nq, hd, reps, scale, seq_len, false, arch.attn_logit_softcapping());
    let mut o = larql_compute::dot_proj_gpu(&attn_out, wo, backend);
    if let Some(b) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut o, b); }

    let rm = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let n = apply_norm(weights, &o, &arch.post_attention_layernorm_key(layer), norm_off);
        if rm != 1.0 { h + &(&n * rm) } else { h + &n }
    } else if rm != 1.0 { h + &(&o * rm) } else { h + &o };

    Some((h_out, k_r, v))
}

/// Single-token attention that reads past K/V from the backend's KV cache,
/// applies RoPE at the correct position, appends the current K/V, and
/// re-populates the cache. All projections run through `backend.matmul_transb`
/// on f32 weights — no quantisation in the hot path. This is the decode-time
/// counterpart to `run_attention_with_kv_backend` for post-norm models where
/// the quant path loses too much precision to stay on-token.
///
/// Returns `(h_out, past_len_after)` where `h_out` is the post-attention
/// residual and the Metal KV cache for this layer now holds past_len+1 entries.
pub fn run_attention_kv_cached_f32(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    rel_layer: usize,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<(Array2<f32>, usize)> {
    use crate::forward::{apply_norm, add_bias};
    use crate::residual::{rms_norm_heads, rms_norm_heads_no_weight};
    use ndarray::s;

    let arch = &*weights.arch;
    let hd = arch.head_dim_for_layer(layer);
    let nq = arch.num_q_heads_for_layer(layer);
    let nkv = arch.num_kv_heads_for_layer(layer);
    let reps = nq / nkv;
    let scale = if arch.attention_multiplier() != 1.0 { arch.attention_multiplier() as f64 } else { arch.attention_scale_for_layer(layer) };
    let norm_off = arch.norm_weight_offset();
    assert_eq!(h.shape()[0], 1, "run_attention_kv_cached_f32 is single-token");

    // Read past K/V from Metal KV cache (may be empty on first call).
    let (k_past_flat, v_past_flat, past_len) =
        backend.debug_read_kv_layer(rel_layer).unwrap_or((Vec::new(), Vec::new(), 0));

    // Input norm + QKV projection on current token.
    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_off);
    let wq = weights.tensors.get(&arch.attn_q_key(layer))?;
    let wk = weights.tensors.get(&arch.attn_k_key(layer))?;
    let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
    let wv = if v_from_k { wk } else { weights.tensors.get(&arch.attn_v_key(layer))? };
    let wo = weights.tensors.get(&arch.attn_o_key(layer))?;
    let mut q = larql_compute::dot_proj_gpu(&h_norm, wq, Some(backend));
    let mut k = larql_compute::dot_proj_gpu(&h_norm, wk, Some(backend));
    let mut v = larql_compute::dot_proj_gpu(&h_norm, wv, Some(backend));
    for (proj, bias_fn) in [(&mut q, arch.attn_q_bias_key(layer) as Option<String>),
                             (&mut k, arch.attn_k_bias_key(layer)),
                             (&mut v, arch.attn_v_bias_key(layer))] {
        if let Some(b) = bias_fn.and_then(|key| weights.vectors.get(&key)) { add_bias(proj, b); }
    }
    if arch.has_v_norm() {
        v = rms_norm_heads_no_weight(&v, nkv, hd);
    }

    // QK-norm per head.
    let qk_off = if arch.qk_norm_weight_offset() != 0.0 { arch.qk_norm_weight_offset() } else { norm_off };
    let q = match arch.attn_q_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&q, w, nq, hd, qk_off), None => q,
    };
    let k = match arch.attn_k_norm_key(layer).and_then(|k| weights.vectors.get(&k)) {
        Some(w) => rms_norm_heads(&k, w, nkv, hd, qk_off), None => k,
    };

    // RoPE at absolute position = past_len.
    let rb = arch.rope_base_for_layer(layer);
    let rf = arch.rotary_fraction_for_layer(layer);
    let q_r = super::apply_rope_partial_at_pos(&q, nq, hd, rb, rf, past_len);
    let k_cur = super::apply_rope_partial_at_pos(&k, nkv, hd, rb, rf, past_len);

    // Assemble K_full = [K_past; K_cur], V_full = [V_past; V_cur] at shape
    // [past_len+1, kv_dim].
    let kv_dim = nkv * hd;
    let t_full = past_len + 1;
    let mut k_full = Array2::<f32>::zeros((t_full, kv_dim));
    let mut v_full = Array2::<f32>::zeros((t_full, kv_dim));
    if past_len > 0 {
        // k_past_flat is already shape [past_len, kv_dim] row-major.
        let k_past_arr = Array2::from_shape_vec((past_len, kv_dim), k_past_flat).unwrap();
        let v_past_arr = Array2::from_shape_vec((past_len, kv_dim), v_past_flat).unwrap();
        k_full.slice_mut(s![0..past_len, ..]).assign(&k_past_arr);
        v_full.slice_mut(s![0..past_len, ..]).assign(&v_past_arr);
    }
    k_full.slice_mut(s![past_len..t_full, ..]).assign(&k_cur);
    v_full.slice_mut(s![past_len..t_full, ..]).assign(&v);

    // Single-query attention: Q has 1 position at absolute pos = past_len;
    // K/V have t_full positions. Attend over all t_full past+current tokens.
    // `gqa_attention_with_weights` assumes Q and K share positions (causal_len
    // = qi + 1 per Q position), so doesn't work here — just do it inline.
    let scale_f32 = scale as f32;
    let mut attn_out = Array2::<f32>::zeros((1, nq * hd));
    let mut scores = vec![0.0f32; t_full];
    for h_idx in 0..nq {
        let kv_h = h_idx / reps;
        let q_off = h_idx * hd;
        let kv_off = kv_h * hd;
        // Score[j] = Q · K[j] * scale
        for j in 0..t_full {
            let mut s = 0.0f32;
            for d in 0..hd {
                s += q_r[[0, q_off + d]] * k_full[[j, kv_off + d]];
            }
            scores[j] = s * scale_f32;
        }
        if let Some(cap) = arch.attn_logit_softcapping() {
            for s in scores.iter_mut() { *s = (*s / cap).tanh() * cap; }
        }
        // Softmax
        let max_s = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let mut sum_e = 0.0f32;
        for s in scores.iter_mut() { *s = (*s - max_s).exp(); sum_e += *s; }
        let inv_sum = 1.0 / sum_e;
        for s in scores.iter_mut() { *s *= inv_sum; }
        // Weighted V sum
        for d in 0..hd {
            let mut acc = 0.0f32;
            for j in 0..t_full {
                acc += scores[j] * v_full[[j, kv_off + d]];
            }
            attn_out[[0, q_off + d]] = acc;
        }
    }
    let mut o = larql_compute::dot_proj_gpu(&attn_out, wo, Some(backend));
    if let Some(b) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut o, b); }

    // Residual + post_attn_norm.
    let rm = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let n = apply_norm(weights, &o, &arch.post_attention_layernorm_key(layer), norm_off);
        if rm != 1.0 { h + &(&n * rm) } else { h + &n }
    } else if rm != 1.0 { h + &(&o * rm) } else { h + &o };

    // Write the updated full K/V back to the Metal KV cache (length = past_len+1).
    let k_flat = k_full.as_slice().unwrap_or(&[]);
    let v_flat = v_full.as_slice().unwrap_or(&[]);
    backend.populate_kv_layer(rel_layer, k_flat, v_flat, t_full, nkv, hd);

    Some((h_out, t_full))
}

/// Q4 attention projection: single projection via Q4 matvec through ComputeBackend.
/// Returns [seq_len, out_dim] f32 result, or None if backend doesn't support Q4.
pub fn q4_attention_proj(
    h: &Array2<f32>,
    q4_data: &[u8],
    num_rows: usize,
    hidden: usize,
    backend: &dyn larql_compute::ComputeBackend,
) -> Option<Array2<f32>> {
    if !backend.has_q4() { return None; }
    let seq_len = h.shape()[0];
    let mut out = Array2::<f32>::zeros((seq_len, num_rows));

    for s in 0..seq_len {
        let x_row = h.row(s);
        let x_slice = x_row.as_slice()?;
        let (q8_x, q8_scales) = larql_compute::cpu::q4::quantize_to_q8(x_slice);
        let scores = backend.q4_matvec(q4_data, &q8_x, &q8_scales, num_rows, hidden)?;
        let mut out_row = out.row_mut(s);
        for j in 0..num_rows { out_row[j] = scores[j]; }
    }
    Some(out)
}
