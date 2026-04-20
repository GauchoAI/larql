//! P17/F Phase 2 — bisect decode_token vs CPU reference at layer 0 with REAL prefill input.
//!
//! Loads Gemma 3 4B, runs prefill via prefill_with_kv (which populates the
//! Metal KV cache and returns h after all 34 layers), captures the LAST-TOKEN
//! h post-prefill at layer 0 (by re-running prefill on the prompt with stop=0),
//! then runs ONE call to decode_token (single-layer) and compares to a CPU
//! reference (run_attention_kv_cached_f32_opt + Gemma 3 post-norm chain).
//!
//! Reports element-wise divergence: where does GPU first differ from CPU?

extern crate blas_src;

use ndarray::Array2;

use larql_compute::{
    ComputeBackend, FullPipelineLayer, NormType, FfnType,
    QuantWeight, QuantFormat, Activation,
};
use larql_inference::{InferenceModel, default_backend};
use larql_inference::layer_graph::predict::prefill_with_kv;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn stats(name: &str, v: &[f32]) {
    let n_nan = v.iter().filter(|x| x.is_nan()).count();
    let n_inf = v.iter().filter(|x| x.is_infinite()).count();
    let mn = v.iter().filter(|x| x.is_finite()).copied().fold(f32::INFINITY, f32::min);
    let mx = v.iter().filter(|x| x.is_finite()).copied().fold(f32::NEG_INFINITY, f32::max);
    let amax = v.iter().filter(|x| x.is_finite()).map(|x| x.abs()).fold(0.0f32, f32::max);
    println!("  {name:>20}: len={} nan={} inf={} range [{:.3}, {:.3}] amax={:.3}",
        v.len(), n_nan, n_inf, mn, mx, amax);
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32; let mut na = 0.0f32; let mut nb = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        if x.is_finite() && y.is_finite() {
            dot += x * y;
            na += x * x;
            nb += y * y;
        }
    }
    dot / (na.sqrt() * nb.sqrt()).max(1e-30)
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter())
        .filter(|(x, y)| x.is_finite() && y.is_finite())
        .map(|(x, y)| (x - y).abs())
        .fold(0.0f32, f32::max)
}

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut layer = 0usize;
    let mut prompt = String::from("What is the capital of France? The answer is");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--layer"  => { i += 1; layer = args[i].parse()?; }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--use-real-q4k" => {} // just a flag, handled below
            _ => {}
        }
        i += 1;
    }

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  P17/F Phase 2 — decode_token L{layer:02} bisect (real input)         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  Model: {model_ref}");
    println!("  Vindex: {vindex_path}");
    println!("  Prompt: \"{prompt}\"");
    println!();

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let arch = &*weights.arch;
    let be = default_backend();

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    // Load BOTH interleaved files; preference via --use-real-q4k flag.
    let _ = index.load_interleaved_q4k_real(std::path::Path::new(&vindex_path));

    let token_ids = tokenizer.encode(&prompt[..], true)?.get_ids().to_vec();
    println!("  Tokens: {} (last 5: {:?})", token_ids.len(),
        &token_ids[token_ids.len().saturating_sub(5)..]);
    println!();

    // ── Step 1. Run prefill on layers 0..layer to get h entering target layer ──
    // We want h_in_to_target_layer = the residual stream right BEFORE target layer.
    // prefill_with_kv runs through layer_range — we use 0..layer so it stops
    // BEFORE target_layer; the returned h is what target_layer's input would be.
    println!("Step 1: prefill 0..{layer} to position h at target layer entry");
    be.reset_kv_cache();
    let h_pre_target = if layer == 0 {
        // Just embed
        larql_inference::forward::embed_tokens_pub(weights, &token_ids)
    } else {
        prefill_with_kv(weights, &token_ids, &index, &*be, 0..layer)
    };
    let last_h: Vec<f32> = h_pre_target.row(h_pre_target.shape()[0] - 1).to_vec();
    stats("h_in (last token)", &last_h);
    println!();

    // For decode_token semantics, the KV cache must already include past_len = seq_len-1
    // (all tokens up to but not including the new token to decode). For layer >= 1
    // prefill_with_kv populated layers 0..layer-1 with full sequence. For layer==0
    // we need to populate just layer 0 KV from the embedding. But we want to TEST
    // layer L's decode_token with the same input the production CPU path sees,
    // so really we want a single-layer comparison: hand last_h to both paths,
    // measure their layer-L output.

    // ── Step 2. CPU reference for layer `layer` ──
    println!("Step 2: CPU reference (run_attention_kv_cached_f32_opt + post-norm chain)");
    // For the CPU reference we need to run the full Gemma 3 layer L:
    //   h_post_attn = run_attention_kv_cached_f32_opt(weights, h, L, rel_idx, backend, gate_index)
    //   normed_o = post_attn_norm(h_post_attn - h)         [Gemma 3 post-norm]
    //   ... wait — actually run_attention_kv_cached_f32_opt RETURNS h_post_attn
    //   already including residual + post-norm. Then FFN runs on it.
    //   For bisect we want JUST the attention contribution to compare against
    //   decode_token's attention encoder chain.
    // For now, run the full layer (attn + FFN) and compare end-to-end.
    let h_in_array = Array2::from_shape_vec((1, last_h.len()), last_h.clone())?;
    let cpu_h_post_attn = larql_inference::attention::gpu::run_attention_kv_cached_f32_opt(
        weights, &h_in_array, layer, layer, &*be,
        Some(&index as &dyn larql_vindex::GateIndex),
    ).ok_or("CPU attn failed")?.0;
    let cpu_h_post_attn_vec: Vec<f32> = cpu_h_post_attn.row(0).to_vec();
    stats("CPU h_post_attn", &cpu_h_post_attn_vec);
    println!();

    // ── Step 3. GPU decode_token for ONLY this one layer ──
    println!("Step 3: GPU decode_token (single layer)");
    // Build a one-layer FullPipelineLayer for layer L.
    let q4k_layer = match index.attn_q4k_layer_data(layer) {
        Some(d) => d,
        None => return Err("attn_q4k_layer_data unavailable; load_attn_q4k must succeed".into()),
    };
    let to_format = |s: &str| -> QuantFormat {
        match s {
            "Q6_K" => QuantFormat::Q6_K,
            "Q4_KF" | "Q4_K_GGUF" => QuantFormat::Q4_KF,
            _ => QuantFormat::Q4_K,
        }
    };
    // FFN weights from interleaved_q4k.bin — check via gate_layer_f32 path? We
    // have walk-only mmap. For decode_token we need raw quant bytes.
    // Just look up via index for now — if not available, FFN-side path won't fire
    // but attention-side will, which is what we're bisecting.
    let gi: &dyn larql_vindex::GateIndex = &index;
    let inter_mmap_data: Option<&[u8]> = gi.interleaved_q4k_real_mmap_ref();
    if inter_mmap_data.is_none() {
        eprintln!("WARNING: interleaved_q4k_real.bin not loaded; FFN side will be empty");
    }
    let hidden = weights.hidden_size;
    let inter = weights.intermediate_size;
    let q4k_per_matrix = (inter * hidden).div_ceil(256) * 148;
    let q4kf_per_matrix = (inter * hidden).div_ceil(256) * 144;
    let q6k_per_matrix = (inter * hidden).div_ceil(256) * 210;
    let inter_mmap = inter_mmap_data.unwrap_or(&[]);
    let per_layer = if !inter_mmap.is_empty() {
        inter_mmap.len() / 34
    } else { 0 };
    let (gate_per, up_per, down_per, gate_fmt, up_fmt, down_fmt) =
        if per_layer == 3 * q6k_per_matrix {
            (q6k_per_matrix, q6k_per_matrix, q6k_per_matrix,
             QuantFormat::Q6_K, QuantFormat::Q6_K, QuantFormat::Q6_K)
        } else if per_layer == 3 * q4k_per_matrix {
            (q4k_per_matrix, q4k_per_matrix, q4k_per_matrix,
             QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
        } else if per_layer == 2 * q4kf_per_matrix + q6k_per_matrix {
            (q4kf_per_matrix, q4kf_per_matrix, q6k_per_matrix,
             QuantFormat::Q4_KF, QuantFormat::Q4_KF, QuantFormat::Q6_K)
        } else if per_layer == 2 * q4k_per_matrix + q6k_per_matrix {
            (q4k_per_matrix, q4k_per_matrix, q6k_per_matrix,
             QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q6_K)
        } else {
            (0, 0, 0, QuantFormat::Q4_K, QuantFormat::Q4_K, QuantFormat::Q4_K)
        };
    let layer_offset = layer * (gate_per + up_per + down_per);
    let gate_bytes = if !inter_mmap.is_empty() {
        &inter_mmap[layer_offset..layer_offset + gate_per]
    } else { &[][..] };
    let up_bytes = if !inter_mmap.is_empty() {
        &inter_mmap[layer_offset + gate_per..layer_offset + gate_per + up_per]
    } else { &[][..] };
    let down_bytes = if !inter_mmap.is_empty() {
        &inter_mmap[layer_offset + gate_per + up_per..layer_offset + gate_per + up_per + down_per]
    } else { &[][..] };

    // Norm vectors from model
    let input_norm: Vec<f32> = weights.vectors.get(&arch.input_layernorm_key(layer))
        .cloned().unwrap_or_default();
    let post_attn_norm: Vec<f32> = weights.vectors.get(&arch.post_attention_layernorm_key(layer))
        .cloned().unwrap_or_default();
    let pre_ffn_norm: Option<Vec<f32>> = arch.pre_feedforward_layernorm_key(layer)
        .and_then(|k| weights.vectors.get(&k)).cloned();
    let post_ffn_norm: Option<Vec<f32>> = arch.post_feedforward_layernorm_key(layer)
        .and_then(|k| weights.vectors.get(&k)).cloned();
    let q_norm: Option<Vec<f32>> = arch.attn_q_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k)).cloned();
    let k_norm: Option<Vec<f32>> = arch.attn_k_norm_key(layer)
        .and_then(|k| weights.vectors.get(&k)).cloned();

    println!("  norm magnitudes: input_max={:.2} post_attn_max={:.2} pre_ffn_max={:.2} post_ffn_max={:.2}",
        input_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        post_attn_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        pre_ffn_norm.as_ref().map(|v| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)).unwrap_or(0.0),
        post_ffn_norm.as_ref().map(|v| v.iter().map(|x| x.abs()).fold(0.0f32, f32::max)).unwrap_or(0.0),
    );

    let q_dim = weights.num_q_heads * weights.head_dim;
    let kv_dim = weights.num_kv_heads * weights.head_dim;

    let gpu_layer = FullPipelineLayer {
        wq: QuantWeight { data: q4k_layer[0].0, scales: None, format: to_format(q4k_layer[0].1) },
        wk: QuantWeight { data: q4k_layer[1].0, scales: None, format: to_format(q4k_layer[1].1) },
        wv: QuantWeight { data: q4k_layer[2].0, scales: None, format: to_format(q4k_layer[2].1) },
        wo: QuantWeight { data: q4k_layer[3].0, scales: None, format: to_format(q4k_layer[3].1) },
        gate: QuantWeight { data: gate_bytes, scales: None, format: gate_fmt },
        up:   QuantWeight { data: up_bytes,   scales: None, format: up_fmt },
        down: QuantWeight { data: down_bytes, scales: None, format: down_fmt },
        input_norm: &input_norm,
        post_attn_norm: &post_attn_norm,
        pre_ffn_norm: pre_ffn_norm.as_deref(),
        post_ffn_norm: post_ffn_norm.as_deref(),
        q_norm_weight: q_norm.as_deref(),
        k_norm_weight: k_norm.as_deref(),
        norm_offset: arch.norm_weight_offset(),
        qk_norm_offset: arch.qk_norm_weight_offset(),
        eps: arch.norm_eps(),
        has_post_norms: arch.has_post_norms(),
        norm_type: NormType::RmsNorm,
        ffn_type: FfnType::Gated,
        activation: Activation::GeluTanh,
        attn_scale: 1.0 / (weights.head_dim as f32).sqrt(),
        head_dim: weights.head_dim,
        num_q_heads: weights.num_q_heads,
        num_kv_heads: weights.num_kv_heads,
        rope_base: arch.rope_base_for_layer(layer) as f32,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: arch.has_v_norm(),
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        ffn_up_bias: None,
        ffn_down_bias: None, softcap: 0.0,
        router_weight: None,
        expert_gate_up: None,
        expert_down: None,
        expert_down_scale: None,
        is_moe_layer: false,
        num_experts: 0,
        num_active_experts: 0,
        expert_intermediate: 0,
    };

    // For layer L>0, the KV cache index inside MetalBackend is rel_idx, not abs.
    // prefill_with_kv with range 0..layer populated KV cache slots 0..layer.
    // decode_token here uses layer index 0 in its array → reads/writes KV slot 0.
    // That conflicts with the prefilled state for L>0. For correctness on L>0
    // we'd need to pass a sliced KV cache. For now, run only L=0 to validate.
    if layer != 0 {
        eprintln!("WARNING: layer>0 not supported here yet — KV cache slot mapping needed.");
    }

    let result = be.decode_token(
        std::slice::from_ref(&gpu_layer), &last_h, hidden, inter, q_dim, kv_dim,
        weights.num_q_heads, weights.num_kv_heads, weights.head_dim,
        arch.rope_base_for_layer(layer) as f32,
    );
    let gpu_out = result.ok_or("decode_token returned None")?;
    stats("GPU decode_token L0 out", &gpu_out);
    println!();

    // ── Step 4. Compare ──
    println!("Step 4: GPU vs CPU comparison");
    println!("  cosine(GPU, CPU h_post_attn)  = {:.4}", cosine(&gpu_out, &cpu_h_post_attn_vec));
    println!("  max|Δ|(GPU, CPU h_post_attn)  = {:.4e}", max_abs_diff(&gpu_out, &cpu_h_post_attn_vec));
    println!();

    // ── Step 4b. Run CPU FFN to produce full layer output ──
    println!("Step 4b: CPU full layer reference (attn + FFN)");
    let dense_ffn = larql_inference::ffn::WeightFfnGpu { weights, backend: &*be };
    let (cpu_full_out, _) = larql_inference::forward::run_ffn(
        weights, &cpu_h_post_attn, layer, &dense_ffn, false,
    );
    let cpu_full_vec: Vec<f32> = cpu_full_out.row(0).to_vec();
    stats("CPU full L0 (attn+FFN)", &cpu_full_vec);
    println!("  cosine(GPU full, CPU full) = {:.4}", cosine(&gpu_out, &cpu_full_vec));
    println!("  max|Δ|(GPU full, CPU full) = {:.4e}", max_abs_diff(&gpu_out, &cpu_full_vec));
    println!();

    // ── Step 4c. Compare FFN input: GPU ffn_norm_out vs CPU pre_ffn_norm ──
    println!("Step 4c: FFN input comparison");
    let norm_offset = arch.norm_weight_offset();
    let cpu_ffn_input = {
        let pre_ffn_key = arch.pre_feedforward_layernorm_key(layer);
        match pre_ffn_key {
            Some(key) => larql_inference::forward::apply_norm(weights, &cpu_h_post_attn, &key, norm_offset),
            None => larql_inference::forward::apply_norm(weights, &cpu_h_post_attn,
                &arch.post_attention_layernorm_key(layer), norm_offset),
        }
    };
    let cpu_ffn_input_vec: Vec<f32> = cpu_ffn_input.row(0).to_vec();
    stats("CPU pre_ffn_norm(h_pa)", &cpu_ffn_input_vec);
    // GPU ffn_norm_out from READBACK — we need to capture it. For now, compute
    // GPU's expected ffn_norm_out from its h_post_attn (which we proved = CPU's).
    // If residual_norm shader is correct, GPU ffn_norm_out = pre_ffn_norm(GPU h_post_attn).
    // Since GPU h_post_attn ≈ CPU h_post_attn (cos 1.0), CPU pre_ffn_norm should match.
    println!("  (GPU ffn_norm_out can only be checked via LARQL_READBACK=1 stderr dumps.)");
    println!("  CPU ffn_input amax={:.3}; GPU ffn_norm_out amax from readback was 34.56",
        cpu_ffn_input_vec.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    println!();

    // ── Step 5. Isolate normed_o = h_post_attn - h_in ──
    // Both paths compute h_post_attn = h_buf + normed_o. Subtracting h_buf
    // gives the post_attn_norm contribution alone, which is what we want
    // to validate (it's the suspect dispatch in metal/decode.rs).
    let cpu_normed_o: Vec<f32> = cpu_h_post_attn_vec.iter().zip(last_h.iter())
        .map(|(c, h)| c - h).collect();
    let gpu_normed_o: Vec<f32> = gpu_out.iter().zip(last_h.iter())
        .map(|(g, h)| g - h).collect();
    println!("Step 5: isolate normed_o (= h_post_attn - h_in)");
    stats("CPU normed_o", &cpu_normed_o);
    stats("GPU normed_o", &gpu_normed_o);
    println!("  cosine(GPU, CPU normed_o) = {:.4}", cosine(&gpu_normed_o, &cpu_normed_o));
    println!("  max|Δ|(GPU, CPU normed_o) = {:.4e}", max_abs_diff(&gpu_normed_o, &cpu_normed_o));
    println!();

    // CPU went through attention only; GPU went through attention + FFN. So a
    // perfect match isn't expected. For pure attention bisect we'd need a
    // SKIP_FFN-equivalent inside this test. Run again with env hint:
    println!("Notes:");
    println!("  - CPU result = attn-only (run_attention_kv_cached_f32_opt).");
    println!("  - GPU result = attn + FFN (full decode_token).");
    println!("  - For attn-only GPU comparison, set LARQL_SKIP_FFN=1 and re-run.");
    println!("  - The first divergence point should be inside attention if the");
    println!("    SKIP_FFN run still differs by more than O(1e-3) from CPU.");

    Ok(())
}

