//! End-to-end decode tok/s using primitives directly (skipping the
//! predict_honest_with_knn_ffn dispatcher which kept hitting the prefill
//! branch). Prefills KV via `prefill_with_kv`, then loops single-token
//! decode: embed → per-layer (run_attention_kv_cached_f32_opt + walk FFN)
//! → lm_head_knn → next token.

extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;

use larql_inference::{InferenceModel, default_backend, ffn::FfnBackend};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use larql_compute::ComputeBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut prompt = String::from("The capital of France is");
    let mut decode_tokens: usize = 10;
    let mut ef_search: usize = 64;
    let mut top_k: usize = 64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--tokens" => { i += 1; decode_tokens = args[i].parse()?; }
            "--ef"     => { i += 1; ef_search = args[i].parse()?; }
            "--top-k"  => { i += 1; top_k = args[i].parse()?; }
            _ => {}
        }
        i += 1;
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    let be = default_backend();

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;
    index.load_down_features(std::path::Path::new(&vindex_path))?;
    index.load_up_features(std::path::Path::new(&vindex_path))?;
    let _ = index.load_lm_head(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q8(std::path::Path::new(&vindex_path));

    let enc = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  End-to-end decode (direct primitives)                    ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  prompt: {:?} ({} tokens)", prompt, prompt_ids.len());
    println!();

    let dense_ffn_helper = larql_inference::WeightFfn { weights };

    let run = |label: &str, walk: &WalkFfn| -> (f64, f64, Vec<String>) {
        let norm_offset = weights.arch.norm_weight_offset();

        // ── Prefill via predict_honest_with_knn_ffn (same path as bench_hnsw_gate)
        // This populates the KV cache AND returns the first predicted token.
        // We do this instead of hand-rolling prefill to avoid subtle wiring bugs.
        be.reset_kv_cache();
        let cache = larql_inference::CachedLayerGraph::build(weights, &prompt_ids, &[], &dense_ffn_helper);
        let t_pref = Instant::now();
        let pref_r = larql_inference::predict_honest_with_knn_ffn(
            weights, tokenizer, &prompt_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(walk),
        );
        let prefill_ms = t_pref.elapsed().as_secs_f64() * 1000.0;
        let mut h = larql_inference::forward::embed_tokens_pub(weights, &prompt_ids);
        // We don't actually need h for decode because the KV cache is populated
        // and predict_honest_with_knn_ffn already returned the first token.
        let _ = &h;

        // First token comes from the prefill's predict result
        let mut current_id = pref_r.raw_predictions.first().map(|(id, _, _)| *id).unwrap_or(0);
        let first_tok = tokenizer.decode(&[current_id], true).unwrap_or_default();
        eprintln!("    [{label}] prefill first-token id={} string={:?}", current_id, first_tok);
        let mut emitted: Vec<String> = vec![first_tok];
        let _ = (norm_offset,);

        // ── Decode loop: single-token forward with KV cache ──
        let t_dec = Instant::now();
        let mut t_attn_total = 0u128;
        let mut t_ffn_total = 0u128;
        for _ in 1..decode_tokens {
            let h_tok = larql_inference::forward::embed_tokens_pub(weights, &[current_id]);
            let mut h_cpu = h_tok;
            for (rel_idx, abs_layer) in (0..num_layers).enumerate() {
                let t_a = Instant::now();
                let (h_post_attn, _) =
                    larql_inference::attention::gpu::run_attention_kv_cached_f32_opt(
                        weights, &h_cpu, abs_layer, rel_idx, &*be,
                        Some(&index as &dyn larql_vindex::GateIndex),
                    ).unwrap();
                t_attn_total += t_a.elapsed().as_micros();
                let t_f = Instant::now();
                h_cpu = larql_inference::forward::run_ffn(weights, &h_post_attn, abs_layer, walk, false).0;
                t_ffn_total += t_f.elapsed().as_micros();
            }
            let h_final = larql_inference::forward::apply_norm(
                weights, &h_cpu, weights.arch.final_norm_key(), norm_offset);
            let h_1d = h_final.row(0).to_owned();
            let hits = index.lm_head_knn_backend(&h_1d, 1, &*be);
            current_id = hits.first().map(|&(tid, _)| tid).unwrap_or(0);
            emitted.push(tokenizer.decode(&[current_id], true).unwrap_or_default());
        }
        let decode_ms = t_dec.elapsed().as_secs_f64() * 1000.0;
        let per_tok = decode_ms / (decode_tokens - 1) as f64;
        let tok_per_sec = 1000.0 / per_tok;
        let attn_ms = t_attn_total as f64 / 1000.0 / (decode_tokens - 1) as f64;
        let ffn_ms = t_ffn_total as f64 / 1000.0 / (decode_tokens - 1) as f64;
        println!("  {label:<30}  prefill {prefill_ms:>6.0}ms  decode {per_tok:>6.1}ms/tok  (attn={attn_ms:.1}ms, ffn={ffn_ms:.1}ms)  = {tok_per_sec:>5.1} tok/s");
        println!("     emitted: {:?}", emitted.join(""));
        (per_tok, tok_per_sec, emitted)
    };

    // Helper shim: WalkFfn needs run_ffn-style call; we fake it by computing
    // pre_ffn_norm + walk + post_ffn_norm + residual ourselves. Simpler: just
    // use `larql_inference::forward::run_ffn` helper.
    //
    // (We can't call run_ffn directly in a closure easily; the struct wraps
    // the FFN backend. So let's use it via a trait-object path.)
    //
    // The run_ffn call internally does: pre_ffn_norm(h_post_attn) → ffn.forward
    // → post_ffn_norm → residual add. That's what we want.

    // BF unlimited (reference)
    index.disable_hnsw();
    let walk_bf_unlim = WalkFfn::new_unlimited(weights, &index);
    let (bf_ms, bf_tps, bf_toks) = run("BF unlimited", &walk_bf_unlim);

    // BF at top_k (same truncation as HNSW will use — isolates truncation from ANN error)
    index.disable_hnsw();
    let walk_bf_topk = WalkFfn::new(weights, &index, top_k);
    let (_bf2_ms, _bf2_tps, bf2_toks) = run(&format!("BF top_k={top_k} (exact)"), &walk_bf_topk);

    // HNSW top_k
    index.enable_hnsw(ef_search);
    let walk_hnsw = WalkFfn::new(weights, &index, top_k);
    let (hn_ms, hn_tps, hn_toks) = run(&format!("HNSW top_k={top_k} ef={ef_search}"), &walk_hnsw);

    let _ = bf2_toks;

    println!();
    println!("  speedup:   {:.2}× decode   ({:.1} → {:.1} tok/s)", bf_ms / hn_ms, bf_tps, hn_tps);
    let agree = bf_toks.iter().zip(hn_toks.iter()).take_while(|(a, b)| a == b).count();
    println!("  agreement: first {} of {} match", agree, decode_tokens);
    Ok(())
}
