//! Does early-exit at layer L predict the same token as the full 34-layer forward?
//!
//! If yes at some L < num_layers, we have a cheap draft candidate for Idea 4.

extern crate blas_src;

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
    "The Great Wall of China is located in",
    "Python is a programming",
    "The Mona Lisa was painted by",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
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
    let _ = index.load_interleaved(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q8(std::path::Path::new(&vindex_path));

    let walk = WalkFfn::new_unlimited(weights, &index);
    let dense_ffn = larql_inference::WeightFfn { weights };
    let cached: Vec<usize> = vec![];

    // Full-forward baseline
    let encs: Vec<(String, Vec<u32>)> = PROMPTS.iter()
        .map(|p| (p.to_string(), tokenizer.encode(*p, true).expect("enc").get_ids().to_vec()))
        .collect();
    let cache = CachedLayerGraph::build(weights, &encs[0].1, &cached, &dense_ffn);

    let full_tokens: Vec<String> = encs.iter().map(|(_, token_ids)| {
        let r = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk),
        );
        r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default()
    }).collect();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Early-exit agreement: layers 0..L vs full 34-layer forward ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    print!("  {:<44}", "prompt");
    let ls: Vec<usize> = vec![18, 22, 26, 28, 30, 32, 34];
    for &l in &ls { print!("  L0..{l:<3}"); }
    println!();

    let mut agree_counts = vec![0usize; ls.len()];
    for ((prompt, token_ids), full_tok) in encs.iter().zip(full_tokens.iter()) {
        print!("  {:<44} ({:<6})", prompt, full_tok);
        for (i, &end) in ls.iter().enumerate() {
            let r = predict_honest_with_knn_ffn(
                weights, tokenizer, token_ids, 1, &index, &*be, &cache,
                0..end, None, Some(&walk),
            );
            let tok = r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
            let ok = &tok == full_tok;
            if ok { agree_counts[i] += 1; }
            print!("  {}", if ok { "  ✓    " } else { "  ✗    " });
        }
        println!();
    }
    println!();
    print!("  {:<44} agreement:", "");
    for (i, &end) in ls.iter().enumerate() {
        let pct = agree_counts[i] as f64 * 100.0 / encs.len() as f64;
        print!("  {pct:>5.0}%");
        let _ = end;
    }
    println!();

    // ── Attention-only draft (HighwayFfn = zero FFN) ──
    println!();
    println!("  ═══ Attention-only draft (HighwayFfn, no FFN) ═══");
    let highway = larql_inference::HighwayFfn;
    let mut highway_agree = 0usize;
    for ((prompt, token_ids), full_tok) in encs.iter().zip(full_tokens.iter()) {
        let r = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&highway),
        );
        let tok = r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
        let ok = &tok == full_tok;
        if ok { highway_agree += 1; }
        println!("  {prompt:<44} verify={:<8} highway={:<8} {}",
            full_tok, tok, if ok { "✓" } else { "✗" });
    }
    let pct = highway_agree as f64 * 100.0 / encs.len() as f64;
    println!("  Highway agreement: {pct:.0}% ({highway_agree}/{} prompts)", encs.len());
    Ok(())
}
