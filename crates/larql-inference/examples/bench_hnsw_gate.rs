//! HNSW gate scoring benchmark: brute-force vs HNSW graph search.
//!
//! Enables the existing HNSW index on VectorIndex, then compares per-prediction
//! timing and token agreement between brute-force and HNSW paths.

extern crate blas_src;

use std::time::Instant;

use larql_inference::{
    predict_honest_with_knn_ffn, CachedLayerGraph, InferenceModel, default_backend,
};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "The capital of Germany is",
    "The capital of Japan is",
    "The largest planet in our solar system is",
    "The chemical symbol for gold is",
    "The Mona Lisa was painted by",
    "Python is a programming",
    "Water boils at",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut ef_search: usize = 64;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--ef"     => { i += 1; ef_search = args[i].parse()?; }
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
    let _ = index.load_interleaved_q4(std::path::Path::new(&vindex_path));

    let encs: Vec<(String, Vec<u32>)> = PROMPTS.iter()
        .map(|p| (p.to_string(), tokenizer.encode(*p, true).expect("enc").get_ids().to_vec()))
        .collect();
    let dense_ffn = larql_inference::WeightFfn { weights };
    let cache = CachedLayerGraph::build(weights, &encs[0].1, &[], &dense_ffn);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  HNSW gate scoring: brute-force vs graph search           ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  ef_search = {ef_search}");
    println!();

    // ── Brute force baseline ──
    index.disable_hnsw();
    let walk_bf = WalkFfn::new_unlimited(weights, &index);
    // Warmup
    for (_, token_ids) in &encs {
        let _ = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk_bf),
        );
    }
    let mut bf_tokens: Vec<String> = Vec::new();
    let t0 = Instant::now();
    for (_, token_ids) in &encs {
        let r = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk_bf),
        );
        bf_tokens.push(r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default());
    }
    let bf_ms = t0.elapsed().as_secs_f64() * 1000.0 / encs.len() as f64;

    // ── HNSW + low top_k ──
    index.enable_hnsw(ef_search);
    eprintln!("[building HNSW indices for all layers...]");
    let walk_hnsw = WalkFfn::new(weights, &index, 64);
    // Warmup (builds HNSW lazily on first call per layer)
    let build_start = Instant::now();
    for (_, token_ids) in encs.iter().take(1) {
        let _ = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk_hnsw),
        );
    }
    let build_ms = build_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[HNSW built in {build_ms:.0}ms]");
    // Second warmup for stable timing
    for (_, token_ids) in &encs {
        let _ = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk_hnsw),
        );
    }

    let mut hnsw_tokens: Vec<String> = Vec::new();
    let t0 = Instant::now();
    for (_, token_ids) in &encs {
        let r = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk_hnsw),
        );
        hnsw_tokens.push(r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default());
    }
    let hnsw_ms = t0.elapsed().as_secs_f64() * 1000.0 / encs.len() as f64;

    // Report
    let mut agree = 0;
    for ((prompt, _), (bf_tok, hnsw_tok)) in encs.iter().zip(bf_tokens.iter().zip(hnsw_tokens.iter())) {
        let ok = bf_tok == hnsw_tok;
        if ok { agree += 1; }
        println!("  {prompt:<44} bf={bf_tok:<10} hnsw={hnsw_tok:<10} {}", if ok { "✓" } else { "✗" });
    }
    println!();
    println!("  brute-force:  {bf_ms:.0} ms/prediction");
    println!("  HNSW (ef={ef_search}): {hnsw_ms:.0} ms/prediction");
    println!("  speedup:      {:.2}×", bf_ms / hnsw_ms);
    println!("  agreement:    {agree}/{} ({:.0}%)", encs.len(), agree as f64 * 100.0 / encs.len() as f64);
    Ok(())
}
