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
    run_attention_with_kv_backend_opt(weights, h, layer, backend, None)
}

/// Variant with optional attn_q4k index. When LARQL_ATTN_Q6K=1 and the index
/// exposes Q6_K bytes for this layer, QKV/O projections dispatch per-position
/// through `q6k_matvec`. This is the prefill analog of
/// `run_attention_kv_cached_f32_opt`.
pub fn run_attention_with_kv_backend_opt(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    backend: Option<&dyn larql_compute::ComputeBackend>,
    attn_q4k_index: Option<&dyn larql_vindex::GateIndex>,
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

    let use_q6k_attn = std::env::var("LARQL_ATTN_Q6K").ok().as_deref() == Some("1");
    let q6k_data = if use_q6k_attn {
        attn_q4k_index.and_then(|idx| idx.attn_q4k_layer_data(layer))
    } else { None };

    let hidden = h_norm.shape()[1];
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;

    let (mut q, mut k, mut v);
    let wo_q6k: Option<(&[u8], &str)>;
    if let (Some(arr4), Some(be)) = (q6k_data, backend) {
        let (q_bytes, _) = arr4[0];
        let (k_bytes, _) = arr4[1];
        let (v_bytes, _) = arr4[2];
        wo_q6k = Some(arr4[3]);
        // Per-position q6k dispatch. For seq_len > 1 this is 3 × seq_len
        // dispatches vs one batched f32 matmul — slower, but necessary
        // when owned f32 attn weights were dropped for RAM savings.
        q = ndarray::Array2::<f32>::zeros((seq_len, q_dim));
        k = ndarray::Array2::<f32>::zeros((seq_len, kv_dim));
        v = ndarray::Array2::<f32>::zeros((seq_len, kv_dim));
        for s in 0..seq_len {
            let x_row: Vec<f32> = h_norm.row(s).to_vec();
            let q_v = be.q6k_matvec(q_bytes, &x_row, q_dim, hidden)?;
            let k_v = be.q6k_matvec(k_bytes, &x_row, kv_dim, hidden)?;
            let v_v = be.q6k_matvec(v_bytes, &x_row, kv_dim, hidden)?;
            for j in 0..q_dim { q[[s, j]] = q_v[j]; }
            for j in 0..kv_dim { k[[s, j]] = k_v[j]; v[[s, j]] = v_v[j]; }
        }
    } else {
        wo_q6k = None;
        let wq = weights.tensors.get(&arch.attn_q_key(layer))?;
        let wk = weights.tensors.get(&arch.attn_k_key(layer))?;
        let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
        let wv = if v_from_k { wk } else { weights.tensors.get(&arch.attn_v_key(layer))? };
        q = larql_compute::dot_proj_gpu(&h_norm, wq, backend);
        k = larql_compute::dot_proj_gpu(&h_norm, wk, backend);
        v = larql_compute::dot_proj_gpu(&h_norm, wv, backend);
    }
    let wo = weights.tensors.get(&arch.attn_o_key(layer));
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

    // Idea 2: Metal fused multi-token attention. Measured crossover (Gemma 3 4B,
    // 34 layers, 8 q-heads) is ~48-64 tokens: below that, shader-launch overhead
    // exceeds the CPU softmax cost; above, per-token compute outweighs dispatch
    // overhead (75-80 ms saved at 66-128 tokens). Default threshold = 48 tokens;
    // override via LARQL_GPU_PREFILL_ATTN env var:
    //   unset / "auto" / "1" → seq_len >= threshold enables GPU
    //   "0"                  → always CPU
    //   "force"              → always GPU (ignores threshold)
    //   numeric N            → use N as threshold
    let softcap_opt = arch.attn_logit_softcapping();
    let gpu_attn_env = std::env::var("LARQL_GPU_PREFILL_ATTN").unwrap_or_default();
    let use_gpu_attn = backend.is_some() && seq_len > 1 && match gpu_attn_env.as_str() {
        "0" => false,
        "force" => true,
        "" | "auto" | "1" => seq_len >= 48,
        s => s.parse::<usize>().map(|t| seq_len >= t).unwrap_or(seq_len >= 48),
    };
    let attn_out = if use_gpu_attn {
        let be = backend.unwrap();
        let q_c = if q_r.is_standard_layout() { q_r.clone() } else { q_r.to_owned() };
        let k_c = if k_r.is_standard_layout() { k_r.clone() } else { k_r.to_owned() };
        let v_c = if v.is_standard_layout() { v.clone() } else { v.to_owned() };
        let q_flat = q_c.as_slice().expect("q contiguous");
        let k_flat = k_c.as_slice().expect("k contiguous");
        let v_flat = v_c.as_slice().expect("v contiguous");
        match be.fused_attention_prefill(
            q_flat, k_flat, v_flat,
            seq_len, nq, nkv, hd,
            scale as f32, softcap_opt.unwrap_or(0.0),
        ) {
            Some(out) => Array2::from_shape_vec((seq_len, q_dim), out).ok()?,
            None => gqa_attention_with_weights(&q_r, &k_r, &v, nq, hd, reps, scale, seq_len, false, softcap_opt).0,
        }
    } else {
        gqa_attention_with_weights(&q_r, &k_r, &v, nq, hd, reps, scale, seq_len, false, softcap_opt).0
    };
    let mut o = if let (Some((wo_bytes, _)), Some(be)) = (wo_q6k, backend) {
        let mut out = ndarray::Array2::<f32>::zeros((seq_len, hidden));
        for s in 0..seq_len {
            let a_row: Vec<f32> = attn_out.row(s).to_vec();
            let o_v = be.q6k_matvec(wo_bytes, &a_row, hidden, q_dim)?;
            for j in 0..hidden { out[[s, j]] = o_v[j]; }
        }
        out
    } else {
        larql_compute::dot_proj_gpu(&attn_out, wo.as_ref().unwrap(), backend)
    };
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
    run_attention_kv_cached_f32_opt(weights, h, layer, rel_layer, backend, None)
}

/// Variant of `run_attention_kv_cached_f32` with optional Q6_K/Q4_K attention
/// weights source. When `attn_q4k_index` is provided AND the env var
/// `LARQL_ATTN_Q6K=1` is set, QKV/O projections dispatch through
/// `backend.q6k_matvec` on the mmap'd Q6_K bytes instead of the owned f32
/// `weights.tensors`. Saves ~1 GB of owned RAM at the cost of one shader
/// dispatch per projection (instead of one fused dense matmul).
pub fn run_attention_kv_cached_f32_opt(
    weights: &crate::model::ModelWeights,
    h: &Array2<f32>,
    layer: usize,
    rel_layer: usize,
    backend: &dyn larql_compute::ComputeBackend,
    attn_q4k_index: Option<&dyn larql_vindex::GateIndex>,
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

    let trace_attn = std::env::var("LARQL_TRACE_ATTN").ok().as_deref() == Some("1");
    let measure_attn = trace_attn || crate::perf::is_enabled();

    // Read past K/V from Metal KV cache (may be empty on first call).
    let t_kv_read = std::time::Instant::now();
    let (k_past_flat, v_past_flat, past_len) =
        backend.debug_read_kv_layer(rel_layer).unwrap_or((Vec::new(), Vec::new(), 0));
    let kv_read_us = if measure_attn { t_kv_read.elapsed().as_micros() } else { 0 };

    // Input norm + QKV projection on current token.
    let t_norm = std::time::Instant::now();
    let h_norm = apply_norm(weights, h, &arch.input_layernorm_key(layer), norm_off);
    let norm_us = if measure_attn { t_norm.elapsed().as_micros() } else { 0 };

    // Use quantized attention (Q4_K/Q6_K) when available — matches the GPU
    // decode_token path so KNN residuals are compatible. Fall back to f32
    // when no quantized weights exist. Opt OUT with LARQL_ATTN_F32=1.
    let force_f32 = std::env::var("LARQL_ATTN_F32").ok().as_deref() == Some("1");
    let q6k_data = if force_f32 { None } else {
        attn_q4k_index.and_then(|idx| idx.attn_q4k_layer_data(layer))
    };

    let hidden = h_norm.shape()[1];
    let q_dim = nq * hd;
    let kv_dim = nkv * hd;

    let (mut q, mut k, mut v);
    let wo_q6k: Option<(&[u8], &str)>;
    if let Some(arr4) = q6k_data {
        // arr4 = [(bytes, fmt); 4] for Q, K, V, O projections (Q6_K on our vindex)
        let (q_bytes, _q_fmt) = arr4[0];
        let (k_bytes, _k_fmt) = arr4[1];
        let (v_bytes, _v_fmt) = arr4[2];
        wo_q6k = Some(arr4[3]);
        let h_flat: Vec<f32> = h_norm.row(0).to_vec();
        // q6k_matvec(W[N,K], x[K]) → out[N]. N = q_dim or kv_dim, K = hidden.
        let q_vec = backend.q6k_matvec(q_bytes, &h_flat, q_dim, hidden)?;
        let k_vec = backend.q6k_matvec(k_bytes, &h_flat, kv_dim, hidden)?;
        let v_vec = backend.q6k_matvec(v_bytes, &h_flat, kv_dim, hidden)?;
        q = Array2::from_shape_vec((1, q_dim), q_vec).ok()?;
        k = Array2::from_shape_vec((1, kv_dim), k_vec).ok()?;
        v = Array2::from_shape_vec((1, kv_dim), v_vec).ok()?;
    } else {
        wo_q6k = None;
        let wq = weights.tensors.get(&arch.attn_q_key(layer))?;
        let wk = weights.tensors.get(&arch.attn_k_key(layer))?;
        let v_from_k = !weights.tensors.contains_key(&arch.attn_v_key(layer));
        let wv = if v_from_k { wk } else { weights.tensors.get(&arch.attn_v_key(layer))? };
        // Try fused QKV: one Metal command buffer with three matmul encoders
        // sharing the same input h_norm. Falls back to three separate calls if
        // the backend doesn't implement it (or if shapes are too small).
        if let Some((q_o, k_o, v_o)) = backend.matmul_transb_triple_share_a(
            h_norm.view(), wq.view(), wk.view(), wv.view(),
        ) {
            q = q_o; k = k_o; v = v_o;
        } else {
            q = larql_compute::dot_proj_gpu(&h_norm, wq, Some(backend));
            k = larql_compute::dot_proj_gpu(&h_norm, wk, Some(backend));
            v = larql_compute::dot_proj_gpu(&h_norm, wv, Some(backend));
        }
    }
    let qkv_us = if measure_attn { t_norm.elapsed().as_micros() - norm_us } else { 0 };
    let t_softmax = std::time::Instant::now();
    let wo = weights.tensors.get(&arch.attn_o_key(layer));
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
    let t_rope = std::time::Instant::now();
    let rb = arch.rope_base_for_layer(layer);
    let rf = arch.rotary_fraction_for_layer(layer);
    let q_r = super::apply_rope_partial_at_pos(&q, nq, hd, rb, rf, past_len);
    let k_cur = super::apply_rope_partial_at_pos(&k, nkv, hd, rb, rf, past_len);
    crate::perf::record("attn.rope", t_rope.elapsed().as_micros());

    // Assemble K_full = [K_past; K_cur], V_full = [V_past; V_cur] at shape
    // [past_len+1, kv_dim].
    let t_kv_assembly = std::time::Instant::now();
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
    crate::perf::record("attn.kv_assembly", t_kv_assembly.elapsed().as_micros());

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
    // LARQL_XSA: Exclusive Self-Attention post-fix (arXiv paper "2-line fix").
    // For each Q head, remove the component of attn_out that lies along the
    // CURRENT TOKEN's value vector at the head's KV group.
    //   y_h ← y_h − ((y_h · v_cur_h) / ‖v_cur_h‖²) · v_cur_h
    // Gemma 3 was NOT trained with XSA, so expect accuracy regression — this
    // is the curiosity test the user asked for.
    if std::env::var("LARQL_XSA").ok().as_deref() == Some("1") {
        let t_cur = past_len; // v_full[past_len, :] is the current token's row
        for h_idx in 0..nq {
            let kv_h = h_idx / reps;
            let q_off = h_idx * hd;
            let kv_off = kv_h * hd;
            let mut v_norm_sq = 0.0f32;
            for d in 0..hd {
                let v_d = v_full[[t_cur, kv_off + d]];
                v_norm_sq += v_d * v_d;
            }
            if v_norm_sq < 1e-12 { continue; }
            let mut dot = 0.0f32;
            for d in 0..hd {
                dot += attn_out[[0, q_off + d]] * v_full[[t_cur, kv_off + d]];
            }
            let alpha = dot / v_norm_sq;
            for d in 0..hd {
                attn_out[[0, q_off + d]] -= alpha * v_full[[t_cur, kv_off + d]];
            }
        }
    }

    let softmax_us = if measure_attn { t_softmax.elapsed().as_micros() } else { 0 };

    // O projection via Q6_K mmap when available; else owned f32 tensor.
    let t_o = std::time::Instant::now();
    let mut o = if let Some((wo_bytes, _fmt)) = wo_q6k {
        let attn_flat: Vec<f32> = attn_out.row(0).to_vec();
        let o_vec = backend.q6k_matvec(wo_bytes, &attn_flat, hidden, q_dim)?;
        Array2::from_shape_vec((1, hidden), o_vec).ok()?
    } else {
        larql_compute::dot_proj_gpu(&attn_out, wo.as_ref().unwrap(), Some(backend))
    };
    if let Some(b) = arch.attn_o_bias_key(layer).and_then(|k| weights.vectors.get(&k)) { add_bias(&mut o, b); }

    // Residual + post_attn_norm.
    let rm = arch.residual_multiplier();
    let h_out = if arch.has_post_norms() {
        let n = apply_norm(weights, &o, &arch.post_attention_layernorm_key(layer), norm_off);
        if rm != 1.0 { h + &(&n * rm) } else { h + &n }
    } else if rm != 1.0 { h + &(&o * rm) } else { h + &o };

    let o_us = if measure_attn { t_o.elapsed().as_micros() } else { 0 };

    // Write the updated full K/V back to the Metal KV cache (length = past_len+1).
    let t_kv_write = std::time::Instant::now();
    let k_flat = k_full.as_slice().unwrap_or(&[]);
    let v_flat = v_full.as_slice().unwrap_or(&[]);
    backend.populate_kv_layer(rel_layer, k_flat, v_flat, t_full, nkv, hd);
    let kv_write_us = if measure_attn { t_kv_write.elapsed().as_micros() } else { 0 };

    if trace_attn {
        eprintln!("[attn L{layer:02}] kv_read={:.2}ms norm={:.2}ms qkv={:.2}ms softmax={:.2}ms o={:.2}ms kv_write={:.2}ms",
            kv_read_us as f64 / 1000.0,
            norm_us as f64 / 1000.0,
            qkv_us as f64 / 1000.0,
            softmax_us as f64 / 1000.0,
            o_us as f64 / 1000.0,
            kv_write_us as f64 / 1000.0,
        );
    }
    crate::perf::record("attn.kv_read",  kv_read_us);
    crate::perf::record("attn.qkv_proj", qkv_us);
    crate::perf::record("attn.softmax",  softmax_us);
    crate::perf::record("attn.o_proj",   o_us);
    crate::perf::record("attn.kv_write", kv_write_us);

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
