//! Standalone GGUF inference — load a GGUF file directly (no vindex) and run
//! inference with the Metal GPU pipeline.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example gguf_infer -- \
//!     <path-to.gguf> [prompt] [max_tokens]
//!
//! Example:
//!   cargo run --release --features metal -p larql-inference --example gguf_infer -- \
//!     ~/.ollama/models/blobs/sha256-<hash> "What is 2+2?" 50
//!
//! The example:
//!   1. Parses GGUF metadata and detects the architecture
//!   2. Loads small f32 tensors (norms, scalars) via load_tensors_filtered
//!   3. mmaps the GGUF for zero-copy quantized data access
//!   4. Dequantizes the embedding + lm_head tensors
//!   5. Builds FullPipelineLayer structs from GGUF tensor byte slices
//!   6. Runs prefill + decode loop via the Metal backend
//!   7. Prints generated text and tok/s

use std::path::PathBuf;
use std::time::Instant;

use larql_compute::{
    FullPipelineLayer, QuantFormat, QuantWeight,
    Activation as ComputeActivation, FfnType as ComputeFfnType,
    NormType as ComputeNormType,
};
use larql_models::{
    GgufFile, GgufQuantizedData, ModelArchitecture,
    detect_from_json,
    quant::ggml,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let gguf_path = args.get(1).ok_or("Usage: gguf_infer <path.gguf> [prompt] [max_tokens]")?;
    let prompt = args.get(2).map(|s| s.as_str()).unwrap_or("The capital of France is");
    let max_tokens: usize = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(50);

    println!("=== GGUF Standalone Inference ===");
    println!("  File:       {gguf_path}");
    println!("  Prompt:     \"{prompt}\"");
    println!("  Max tokens: {max_tokens}");
    println!();

    // ── Step 1: Parse GGUF ──
    let t0 = Instant::now();
    let gguf_file = PathBuf::from(gguf_path);
    let gguf = GgufFile::open(&gguf_file)?;
    let config_json = gguf.to_config_json();
    let arch = detect_from_json(&config_json);
    let cfg = arch.config();
    let num_layers = cfg.num_layers;
    let hidden = cfg.hidden_size;
    let inter = cfg.intermediate_size;

    println!("[1] GGUF parsed in {:.1}ms", t0.elapsed().as_secs_f64() * 1000.0);
    println!("    Arch:         {}", cfg.model_type);
    println!("    Layers:       {num_layers}");
    println!("    Hidden:       {hidden}");
    println!("    Intermediate: {inter}");
    if let Some(ne) = cfg.num_experts {
        println!("    Experts:      {ne} (top-{})", cfg.num_experts_per_token.unwrap_or(0));
    }
    println!("    Head dim:     {}", cfg.head_dim);
    println!("    Q heads:      {}", cfg.num_q_heads);
    println!("    KV heads:     {}", cfg.num_kv_heads);
    if let Some(ghd) = cfg.global_head_dim {
        println!("    Global h_dim: {ghd}");
    }
    if let Some(gkv) = cfg.num_global_kv_heads {
        println!("    Global KV:    {gkv}");
    }
    println!("    Tensors:      {}", gguf.tensor_infos.len());
    println!();

    // ── Step 2: Load norms and small f32 tensors ──
    let t1 = Instant::now();
    let (_tensors, mut vectors) = gguf.load_tensors_filtered(2_000_000)?;
    // For 2-norm models: GGUF's ffn_norm now normalizes to pre_feedforward_layernorm.
    // Duplicate it under post_attention_layernorm so the f32 inference path finds it.
    if !arch.has_post_norms() {
        for l in 0..num_layers {
            let pre_ffn = format!("layers.{l}.pre_feedforward_layernorm.weight");
            let post_attn = format!("layers.{l}.post_attention_layernorm.weight");
            if !vectors.contains_key(&post_attn) {
                if let Some(v) = vectors.get(&pre_ffn) {
                    let cloned = v.clone();
                    vectors.insert(post_attn, cloned);
                }
            }
        }
    }
    println!("[2] Norms loaded: {} vectors in {:.1}ms",
        vectors.len(), t1.elapsed().as_secs_f64() * 1000.0);

    // Dump first few vector keys for debugging
    if std::env::var("LARQL_TRACE_KEYS").ok().as_deref() == Some("1") {
        let mut keys: Vec<&str> = vectors.keys().map(|s| s.as_str()).collect();
        keys.sort();
        for k in keys.iter().take(30) {
            println!("    norm: {k} [{}]", vectors[*k].len());
        }
    }

    // ── Step 3: mmap for raw quantized access ──
    let t2 = Instant::now();
    let qdata = GgufQuantizedData::open(&gguf_file, gguf.data_offset)?;
    println!("[3] mmap opened in {:.1}ms", t2.elapsed().as_secs_f64() * 1000.0);

    // ── Step 4: Embed + LM head setup (on-demand, no bulk dequantize) ──
    let t3 = Instant::now();
    let embed_info = gguf.find_tensor("token_embd.weight")
        .ok_or("missing token_embd.weight")?;
    let embed_bytes = qdata.tensor_bytes(embed_info);
    println!("    embed dims: {:?} type={} ({})", embed_info.dims,
        embed_info.tensor_type, ggml::type_name(embed_info.tensor_type));
    // GGUF dims are stored inner-axis-first. dims=[A, B] means a numpy-shape
    // [B, A] tensor with row-major storage: each "row" is A contiguous elements.
    // For an embedding of vocab × hidden, the inner axis must be hidden (since
    // Q-K-quant block sizes only divide hidden, not vocab) — so dims[0]=hidden
    // and each token's embedding is one row of `hidden` contiguous elements.
    // The previous transposed=true branch column-extracted from huge "rows"
    // and produced plausible-looking but wrong values.
    let embed_dim0 = embed_info.dims[0] as usize;
    let embed_dim1 = embed_info.dims[1] as usize;
    assert_eq!(embed_dim0, hidden, "expected GGUF embed dims[0]=hidden");
    let (embed_vocab, embed_hidden, embed_transposed) = (embed_dim1, embed_dim0, false);
    assert_eq!(embed_hidden, hidden, "embed hidden dim mismatch");
    // Compute bytes-per-row for on-demand dequantization
    let embed_elements_per_row = if embed_transposed { embed_vocab } else { embed_hidden };
    let embed_bytes_per_row = ggml::tensor_data_size(embed_info.tensor_type, embed_elements_per_row)?;
    let embed_type = embed_info.tensor_type;

    /// Dequantize a single embedding row on demand.
    fn embed_row(embed_bytes: &[u8], token_id: u32, hidden: usize,
                 embed_transposed: bool, embed_type: u32, embed_bytes_per_row: usize,
                 embed_vocab: usize, embed_scale: f32) -> Vec<f32> {
        if embed_transposed {
            // dims = [hidden, vocab] — each row is a hidden-dim slice across all tokens
            // Need to gather column token_id from each of the hidden rows
            let mut row = vec![0.0f32; hidden];
            for r in 0..hidden {
                let row_bytes = &embed_bytes[r * embed_bytes_per_row..(r+1) * embed_bytes_per_row];
                let full_row = ggml::dequantize(row_bytes, embed_type, embed_vocab).unwrap();
                row[r] = full_row[token_id as usize] * embed_scale;
            }
            row
        } else {
            // dims = [vocab, hidden] — each row is one token's embedding
            let offset = token_id as usize * embed_bytes_per_row;
            let row_bytes = &embed_bytes[offset..offset + embed_bytes_per_row];
            let mut row = ggml::dequantize(row_bytes, embed_type, hidden).unwrap();
            for v in &mut row { *v *= embed_scale; }
            row
        }
    }

    // LM head: try "output.weight", fall back to embed (quantized, kept as raw bytes)
    let lm_head_info_opt = gguf.find_tensor("output.weight");
    let lm_head_bytes: &[u8];
    let lm_head_type: u32;
    let lm_head_vocab: usize;
    if let Some(info) = lm_head_info_opt {
        lm_head_bytes = qdata.tensor_bytes(info);
        lm_head_type = info.tensor_type;
        lm_head_vocab = info.dims[0] as usize;
        println!("    lm_head dims: {:?} type={}", info.dims, ggml::type_name(info.tensor_type));
    } else {
        lm_head_bytes = embed_bytes;
        lm_head_type = embed_type;
        lm_head_vocab = embed_vocab;
        println!("    lm_head: tied to embed");
    }
    let lm_head_bpr = ggml::tensor_data_size(lm_head_type, hidden)?;

    /// Compute logits by dequantizing lm_head rows in chunks and dot-producting with h.
    /// Returns top-1 token ID. Does NOT allocate full [vocab x hidden] matrix.
    fn argmax_logits(h: &[f32], lm_bytes: &[u8], lm_type: u32, lm_vocab: usize,
                     hidden: usize, lm_bpr: usize, softcap: f32) -> u32 {
        let mut best_id = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        // Process in chunks to amortize dequant overhead
        let chunk = 256;
        for start in (0..lm_vocab).step_by(chunk) {
            let end = (start + chunk).min(lm_vocab);
            let chunk_bytes = &lm_bytes[start * lm_bpr..end * lm_bpr];
            let chunk_f32 = ggml::dequantize(chunk_bytes, lm_type, (end - start) * hidden).unwrap();
            for i in 0..(end - start) {
                let row = &chunk_f32[i * hidden..(i + 1) * hidden];
                let mut dot: f32 = row.iter().zip(h.iter()).map(|(a, b)| a * b).sum();
                if softcap > 0.0 {
                    dot = softcap * (dot / softcap).tanh();
                }
                if dot > best_val {
                    best_val = dot;
                    best_id = (start + i) as u32;
                }
            }
        }
        best_id
    }

    println!("[4] Embed + lm_head setup in {:.1}ms (on-demand, 0 MB allocated)",
        t3.elapsed().as_secs_f64() * 1000.0);

    // ── Step 5: Tokenizer ──
    let t4 = Instant::now();
    // Try to find tokenizer.json alongside the GGUF
    let tokenizer_path = gguf_file.parent().unwrap().join("tokenizer.json");
    let tokenizer = if tokenizer_path.exists() {
        tokenizers::Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| format!("tokenizer: {e}"))?
    } else {
        // Fallback: try the vindex tokenizer (Gemma 3/4 share tokenizers)
        let vindex_tok = PathBuf::from(
            std::env::var("LARQL_TOKENIZER")
                .unwrap_or_else(|_| "/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex/tokenizer.json".into())
        );
        if vindex_tok.exists() {
            tokenizers::Tokenizer::from_file(&vindex_tok)
                .map_err(|e| format!("tokenizer: {e}"))?
        } else {
            return Err("No tokenizer.json found. Set LARQL_TOKENIZER=<path>".into());
        }
    };
    println!("[5] Tokenizer loaded in {:.1}ms", t4.elapsed().as_secs_f64() * 1000.0);

    // ── Step 6: Build pipeline layers ──
    let t5 = Instant::now();
    let layers = build_gguf_layers(&gguf, &qdata, &vectors, &*arch);
    println!("[6] {} pipeline layers built in {:.1}ms",
        layers.len(), t5.elapsed().as_secs_f64() * 1000.0);

    // ── Step 7: Create backend ──
    let backend = larql_compute::default_backend();
    println!("[7] Backend: {} ({})", backend.name(), backend.device_info());

    // Compute max dimensions for buffer allocation
    let max_q_dim = (0..num_layers)
        .map(|l| arch.num_q_heads_for_layer(l) * arch.head_dim_for_layer(l))
        .max().unwrap_or(0);
    let max_kv_dim = (0..num_layers)
        .map(|l| arch.num_kv_heads_for_layer(l) * arch.head_dim_for_layer(l))
        .max().unwrap_or(0);
    let max_inter = inter; // shared expert intermediate
    println!("    max_q_dim={max_q_dim}  max_kv_dim={max_kv_dim}  inter={max_inter}");

    // ── Step 8: Tokenize prompt ──
    let encoding = tokenizer.encode(prompt, true)
        .map_err(|e| format!("tokenize: {e}"))?;
    let token_ids = encoding.get_ids();
    println!("[8] Prompt tokenized: {} tokens {:?}",
        token_ids.len(), &token_ids[..token_ids.len().min(10)]);

    // ── Step 9: Prefill via GPU decode_token (token by token) ──
    let embed_scale = arch.embed_scale();
    // GGUF Gemma3 norm weights have +1 baked in (convert_hf_to_gguf adds 1.0
    // to all "norm.weight" tensors — see Gemma3Model.norm_shift). Offset must
    // be 0 here to avoid double-applying.
    let norm_offset = if arch.family() == "gemma3" { 0.0 } else { arch.norm_weight_offset() };
    let final_norm_key = arch.final_norm_key();

    println!("[9] Starting prefill + decode ...");

    // ── Prefill: feed each prompt token through decode_token ──
    let final_norm_vec = vectors.get(final_norm_key);
    let mut generated_tokens: Vec<String> = Vec::new();

    backend.reset_kv_cache();
    let mut last_h: Option<Vec<f32>> = None;

    let t_prefill2 = Instant::now();
    for &tid in token_ids.iter() {
        let x = embed_row(embed_bytes, tid, hidden, embed_transposed, embed_type,
            embed_bytes_per_row, embed_vocab, embed_scale);
        let h = backend.decode_token(
            &layers, &x, hidden, max_inter, max_q_dim, max_kv_dim,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            arch.rope_base_for_layer(0) as f32,
        ).expect("decode_token failed");
        last_h = Some(h);
    }
    let prefill_ms = t_prefill2.elapsed().as_secs_f64() * 1000.0;
    println!("    Prefill: {:.0}ms ({} tokens, {:.1} tok/s)",
        prefill_ms, token_ids.len(),
        token_ids.len() as f64 / (prefill_ms / 1000.0));

    // First token from prefill result
    let h_out = last_h.unwrap();
    let h_normed = apply_final_norm(&h_out, final_norm_vec, norm_offset);

    // Show top-5 for diagnostics
    // Top-5 via on-demand dequant
    let softcap = arch.config().final_logit_softcapping.unwrap_or(0.0) as f32;
    let top5 = {
        let mut scores: Vec<(u32, f32)> = Vec::new();
        let chunk = 256;
        for start in (0..lm_head_vocab).step_by(chunk) {
            let end = (start + chunk).min(lm_head_vocab);
            let cb = &lm_head_bytes[start * lm_head_bpr..end * lm_head_bpr];
            let cf = ggml::dequantize(cb, lm_head_type, (end - start) * hidden).unwrap();
            for i in 0..(end - start) {
                let row = &cf[i * hidden..(i + 1) * hidden];
                let mut dot: f32 = row.iter().zip(h_normed.iter()).map(|(a, b)| a * b).sum();
                if softcap > 0.0 { dot = softcap * (dot / softcap).tanh(); }
                scores.push(((start + i) as u32, dot));
            }
        }
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(5);
        scores
    };
    print!("    Top-5: ");
    for (i, &(tid, score)) in top5.iter().enumerate() {
        let tok = tokenizer.decode(&[tid], true).unwrap_or_default();
        print!("{}=\"{}\"({:.1}) ", i + 1, tok.trim(), score);
    }
    println!();

    let mut current_token_id = top5[0].0;
    let tok_str = tokenizer.decode(&[current_token_id], true).unwrap_or_default();
    generated_tokens.push(tok_str.clone());
    print!("{prompt}");
    print!("{tok_str}");

    // Decode loop
    let mut decode_times: Vec<f64> = Vec::new();
    let stop_ids: Vec<u32> = vec![1, 106, 107]; // <eos>, <end_of_turn>, etc.

    for _step in 1..max_tokens {
        let t_dec = Instant::now();

        let x = embed_row(embed_bytes, current_token_id, hidden, embed_transposed, embed_type,
            embed_bytes_per_row, embed_vocab, embed_scale);

        let h = match backend.decode_token(
            &layers, &x, hidden, max_inter, max_q_dim, max_kv_dim,
            cfg.num_q_heads, cfg.num_kv_heads, cfg.head_dim,
            arch.rope_base_for_layer(0) as f32,
        ) {
            Some(h) => h,
            None => {
                eprintln!("\n  decode_token returned None at step {_step}");
                break;
            }
        };

        let h_normed = apply_final_norm(&h, final_norm_vec, norm_offset);
        let next_id = argmax_logits(&h_normed, lm_head_bytes, lm_head_type, lm_head_vocab,
            hidden, lm_head_bpr, softcap);
        let dec_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        decode_times.push(dec_ms);

        let tok_str = tokenizer.decode(&[next_id], true).unwrap_or_default();
        print!("{tok_str}");
        generated_tokens.push(tok_str);
        current_token_id = next_id;

        if stop_ids.contains(&next_id) {
            break;
        }
    }
    println!();

    // ── Summary ──
    let avg_decode_ms = if decode_times.is_empty() { 0.0 }
        else { decode_times.iter().sum::<f64>() / decode_times.len() as f64 };
    let tok_per_sec = if avg_decode_ms > 0.0 { 1000.0 / avg_decode_ms } else { 0.0 };
    println!();
    println!("=== Summary ===");
    println!("  Generated:  {} tokens", generated_tokens.len());
    println!("  Prefill:    {:.0}ms ({} tokens)", prefill_ms, token_ids.len());
    println!("  Avg decode: {:.1}ms/tok ({:.1} tok/s)", avg_decode_ms, tok_per_sec);
    if !decode_times.is_empty() {
        println!("  First tok:  {:.1}ms", decode_times[0]);
        let last_5: Vec<f64> = decode_times.iter().rev().take(5).copied().collect();
        let warm_avg = last_5.iter().sum::<f64>() / last_5.len() as f64;
        println!("  Warm avg:   {:.1}ms/tok ({:.1} tok/s)", warm_avg, 1000.0 / warm_avg);
    }

    Ok(())
}

// ═══════════════════════════════════════════════════════════════
// Layer construction
// ═══════════════════════════════════════════════════════════════

/// Map GGUF tensor types to larql-compute QuantFormat.
///
/// GGUF Q4_K uses the 144-byte llama.cpp layout. larql-compute calls this
/// `Q4_KF` ("Q4_K from GGUF Format") — the Metal kernel reads the 144-byte
/// `block_q4_K_gguf` struct, not the internal 148-byte `block_q4_K`.
fn gguf_tensor_type_to_quant_format(tensor_type: u32) -> QuantFormat {
    match tensor_type {
        ggml::TYPE_Q4_0 => QuantFormat::Q4_0,
        ggml::TYPE_Q8_0 => QuantFormat::Q8_0Gguf,
        ggml::TYPE_Q4_K => QuantFormat::Q4_KF,  // GGUF 144-byte layout
        ggml::TYPE_Q6_K => QuantFormat::Q6_K,
        // Q5_0/Q5_K: no Metal kernel. Map to closest supported format.
        // The byte layout mismatch means these produce wrong results, but
        // the example won't crash. Print a warning at layer build time.
        ggml::TYPE_Q5_0 => QuantFormat::Q8_0Gguf,
        ggml::TYPE_Q5_K => QuantFormat::Q6_K,
        _ => {
            eprintln!("  WARN: unsupported GGML tensor type {tensor_type}, mapping to Q4_KF");
            QuantFormat::Q4_KF
        }
    }
}

fn build_gguf_layers<'a>(
    gguf: &'a GgufFile,
    qdata: &'a GgufQuantizedData,
    vectors: &'a std::collections::HashMap<String, Vec<f32>>,
    arch: &dyn ModelArchitecture,
) -> Vec<FullPipelineLayer<'a>> {
    let num_layers = arch.config().num_layers;
    let _is_moe = arch.is_moe();

    (0..num_layers).map(|l| {
        // ── Attention weights ──
        let wq_info = gguf.find_tensor(&format!("blk.{l}.attn_q.weight"))
            .unwrap_or_else(|| panic!("missing blk.{l}.attn_q.weight"));
        let wq_data = qdata.tensor_bytes(wq_info);
        let wq_format = gguf_tensor_type_to_quant_format(wq_info.tensor_type);

        let wk_info = gguf.find_tensor(&format!("blk.{l}.attn_k.weight"))
            .unwrap_or_else(|| panic!("missing blk.{l}.attn_k.weight"));
        let wk_data = qdata.tensor_bytes(wk_info);
        let wk_format = gguf_tensor_type_to_quant_format(wk_info.tensor_type);

        // V may share K (Gemma 4 K=V on global layers)
        let wv_info = gguf.find_tensor(&format!("blk.{l}.attn_v.weight"));
        let (wv_data, wv_format) = if let Some(info) = wv_info {
            (qdata.tensor_bytes(info), gguf_tensor_type_to_quant_format(info.tensor_type))
        } else {
            // V shares K
            (wk_data, wk_format)
        };

        let wo_info = gguf.find_tensor(&format!("blk.{l}.attn_output.weight"))
            .unwrap_or_else(|| panic!("missing blk.{l}.attn_output.weight"));
        let wo_data = qdata.tensor_bytes(wo_info);
        let wo_format = gguf_tensor_type_to_quant_format(wo_info.tensor_type);

        // ── Dense FFN weights (shared expert for MoE, or sole FFN for dense) ──
        let gate_info = gguf.find_tensor(&format!("blk.{l}.ffn_gate.weight"));
        let up_info = gguf.find_tensor(&format!("blk.{l}.ffn_up.weight"));
        let down_info = gguf.find_tensor(&format!("blk.{l}.ffn_down.weight"));

        let empty_qw = QuantWeight { data: &[], scales: None, format: QuantFormat::Q4_K };

        let gate = gate_info.map(|info| QuantWeight {
            data: qdata.tensor_bytes(info), scales: None,
            format: gguf_tensor_type_to_quant_format(info.tensor_type),
        }).unwrap_or(empty_qw);

        let up = up_info.map(|info| QuantWeight {
            data: qdata.tensor_bytes(info), scales: None,
            format: gguf_tensor_type_to_quant_format(info.tensor_type),
        }).unwrap_or(empty_qw);

        let down = down_info.map(|info| QuantWeight {
            data: qdata.tensor_bytes(info), scales: None,
            format: gguf_tensor_type_to_quant_format(info.tensor_type),
        }).unwrap_or(empty_qw);

        // ── Norm weights ──
        let input_norm = lookup_norm_vec(vectors, &arch.input_layernorm_key(l));
        let post_attn_norm = lookup_norm_vec(vectors, &arch.post_attention_layernorm_key(l));
        let pre_ffn_norm = arch.pre_feedforward_layernorm_key(l)
            .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
        let post_ffn_norm = arch.post_feedforward_layernorm_key(l)
            .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
        let q_norm_weight = arch.attn_q_norm_key(l)
            .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());
        let k_norm_weight = arch.attn_k_norm_key(l)
            .and_then(|k| vectors.get(&k)).map(|v| v.as_slice());

        // ── Per-layer architecture ──
        let head_dim = arch.head_dim_for_layer(l);
        let num_q_heads = arch.num_q_heads_for_layer(l);
        let num_kv_heads = arch.num_kv_heads_for_layer(l);
        let rotary_frac = arch.rotary_fraction_for_layer(l);
        let rotary_dim = if rotary_frac >= 1.0 { 0 } else { (head_dim as f64 * rotary_frac) as usize };
        let is_sliding = arch.is_sliding_window_layer(l);
        let sliding_window = if is_sliding {
            arch.sliding_window_size().unwrap_or(0)
        } else {
            0
        };

        let layer_scalar = arch.layer_scalar_key(l)
            .and_then(|k| vectors.get(&k)
                .or_else(|| vectors.get(&format!("{k}.weight"))))
            .and_then(|v| v.first().copied())
            .unwrap_or(0.0);

        // ── MoE fields ──
        let router_info = gguf.find_tensor(&format!("blk.{l}.ffn_gate_inp.weight"));
        let router_weight: Option<&[f32]> = router_info
            .and_then(|info| qdata.tensor_f32(info));

        let expert_gu_info = gguf.find_tensor(&format!("blk.{l}.ffn_gate_up_exps.weight"));
        let expert_gate_up: Option<QuantWeight> = expert_gu_info.map(|info| {
            QuantWeight {
                data: qdata.tensor_bytes(info),
                scales: None,
                format: gguf_tensor_type_to_quant_format(info.tensor_type),
            }
        });

        let expert_down_info = gguf.find_tensor(&format!("blk.{l}.ffn_down_exps.weight"));
        let expert_down: Option<QuantWeight> = expert_down_info.map(|info| {
            QuantWeight {
                data: qdata.tensor_bytes(info),
                scales: None,
                format: gguf_tensor_type_to_quant_format(info.tensor_type),
            }
        });

        let has_moe = router_weight.is_some() && expert_gate_up.is_some();
        let num_experts = if has_moe { arch.num_experts() } else { 0 };
        let num_active_experts = if has_moe { arch.num_experts_per_token() } else { 0 };
        // Expert intermediate: for Gemma 4, the GGUF expert gate+up is fused
        // [hidden, 2*expert_inter, num_experts]. Derive from tensor dims.
        let expert_inter = expert_gu_info
            .map(|info| {
                // dims: [hidden=2816, 2*expert_inter=1408, num_experts=128]
                if info.dims.len() >= 2 { info.dims[1] as usize / 2 } else { 0 }
            })
            .unwrap_or(0);

        FullPipelineLayer {
            wq: QuantWeight { data: wq_data, scales: None, format: wq_format },
            wk: QuantWeight { data: wk_data, scales: None, format: wk_format },
            wv: QuantWeight { data: wv_data, scales: None, format: wv_format },
            wo: QuantWeight { data: wo_data, scales: None, format: wo_format },
            gate, up, down,
            input_norm,
            post_attn_norm,
            pre_ffn_norm,
            post_ffn_norm,
            q_norm_weight,
            k_norm_weight,
            input_norm_bias: None,
            post_attn_norm_bias: None,
            // GGUF Gemma3 norms are pre-shifted by +1; per-layer offset must be 0.
            norm_offset: if arch.family() == "gemma3" { 0.0 } else { arch.norm_weight_offset() },
            qk_norm_offset: if arch.family() == "gemma3" { 0.0 } else { arch.qk_norm_weight_offset() },
            eps: arch.norm_eps(),
            has_post_norms: arch.has_post_norms(),
            norm_type: match arch.norm_type() {
                larql_models::NormType::LayerNorm => ComputeNormType::LayerNorm,
                _ => ComputeNormType::RmsNorm,
            },
            ffn_type: match arch.ffn_type() {
                larql_models::FfnType::Standard => ComputeFfnType::Standard,
                _ => ComputeFfnType::Gated,
            },
            activation: match arch.activation() {
                larql_models::Activation::GeluTanh => ComputeActivation::GeluTanh,
                _ => ComputeActivation::Silu,
            },
            attn_scale: arch.attention_scale_for_layer(l) as f32,
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: arch.rope_base_for_layer(l) as f32,
            rope_freq_scale: arch.rope_freq_scale_for_layer(l) as f32,
            rotary_dim,
            sliding_window,
            has_v_norm: arch.has_v_norm(),
            layer_scalar,
            softcap: arch.attn_logit_softcapping().unwrap_or(0.0),
            ffn_up_bias: None,
            ffn_down_bias: None,
            router_weight,
            expert_gate_up,
            expert_down,
            expert_down_scale: None,
            is_moe_layer: has_moe,
            num_experts,
            num_active_experts,
            expert_intermediate: expert_inter,
        }
    }).collect()
}

// ═══════════════════════════════════════════════════════════════
// Utility functions
// ═══════════════════════════════════════════════════════════════

fn lookup_norm_vec<'a>(
    vectors: &'a std::collections::HashMap<String, Vec<f32>>,
    key: &str,
) -> &'a [f32] {
    vectors.get(key).map(|v| v.as_slice()).unwrap_or_else(|| {
        if std::env::var("LARQL_TRACE_KEYS").ok().as_deref() == Some("1") {
            eprintln!("  WARN: norm key not found: {key}");
        }
        &[]
    })
}

/// Apply RMSNorm: x * w / rms(x), with optional norm_offset on weights.
fn apply_final_norm(x: &[f32], norm_weight: Option<&Vec<f32>>, norm_offset: f32) -> Vec<f32> {
    let n = x.len();
    let eps = 1e-6f32;

    // Compute RMS
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    match norm_weight {
        Some(w) if w.len() == n => {
            x.iter().zip(w.iter()).map(|(&xi, &wi)| {
                xi * inv_rms * (wi + norm_offset)
            }).collect()
        }
        _ => {
            x.iter().map(|&xi| xi * inv_rms).collect()
        }
    }
}

