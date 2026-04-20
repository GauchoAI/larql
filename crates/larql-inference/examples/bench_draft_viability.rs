//! Idea 4 viability probe: does a low-top_k walk agree often enough with
//! high-top_k walk to be useful as a specdec draft?
//!
//! For each prompt, we predict the next token at verify-quality (top_k=8192)
//! and at several draft candidates (top_k in {64, 128, 256, 1024}), and report
//! whether top-1 matches. No multi-token generation; this is a one-shot
//! acceptance-rate proxy. If even the single-step match rate is <40%, full
//! specdec won't amortize the extra verify pass.

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
    "2 + 2 equals",
    "The largest planet in our solar system is",
    "Shakespeare wrote the play",
    "The chemical symbol for gold is",
    "The speed of light is approximately",
    "The Great Wall of China is located in",
    "Python is a programming",
    "Water boils at",
    "The Mona Lisa was painted by",
];

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut draft_ks: Vec<usize> = vec![64, 128, 256, 512, 1024, 2048];
    let verify_k: usize = 8192;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--draft-ks" => { i += 1; draft_ks = args[i].split(',').filter_map(|s| s.parse().ok()).collect(); }
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
    let _ = index.load_lm_head(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q8(std::path::Path::new(&vindex_path));

    let verify_walk = WalkFfn::new_unlimited(weights, &index);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Idea 4 — draft-vs-verify top-1 agreement on {} prompts  ║", PROMPTS.len());
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!();
    println!("  verify top_k = {verify_k} (unlimited)");
    println!("  draft top_k candidates: {draft_ks:?}");
    println!();

    // Cache the prompt encodings
    let encodings: Vec<(String, Vec<u32>)> = PROMPTS.iter()
        .map(|p| {
            let enc = tokenizer.encode(*p, true).expect("encode");
            (p.to_string(), enc.get_ids().to_vec())
        })
        .collect();

    // Build a dummy CachedLayerGraph for the entire range (empty cache means everything recomputes)
    let dense_ffn = larql_inference::WeightFfn { weights };
    let cached: Vec<usize> = vec![]; // no caching — full recompute for both paths
    let cache = CachedLayerGraph::build(weights, &encodings[0].1, &cached, &dense_ffn);

    // Step 1: verify predictions on all prompts
    println!("  {:<44} {:<16}", "prompt", "verify top-1");
    let mut verify_tokens: Vec<u32> = Vec::new();
    for (prompt, token_ids) in &encodings {
        let r = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&verify_walk),
        );
        let tok_id = r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
        let tok_str = tok_id.clone();
        // Try to convert back to id via tokenizer for match
        let tok_id_num: u32 = tokenizer.encode(tok_str.as_str(), false)
            .ok().and_then(|e| e.get_ids().first().copied()).unwrap_or(u32::MAX);
        verify_tokens.push(tok_id_num);
        println!("  {prompt:<44} {:<16}", format!("{:?}", tok_str));
    }
    println!();

    // Step 2: sweep draft top_k values, measure agreement rate + timing
    println!("  {:<8} {:<10} {:<10} {:<40}", "top_k", "agree%", "ms/pred", "mismatches (verify → draft)");
    // Warmup verify first so the top_k=8192 row isn't cold
    {
        let (_, token_ids) = &encodings[0];
        let _ = predict_honest_with_knn_ffn(
            weights, tokenizer, token_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&verify_walk),
        );
    }
    // Time verify once for the baseline
    let verify_ms = {
        let t0 = std::time::Instant::now();
        for (_, token_ids) in &encodings {
            let _ = predict_honest_with_knn_ffn(
                weights, tokenizer, token_ids, 1, &index, &*be, &cache,
                0..num_layers, None, Some(&verify_walk),
            );
        }
        t0.elapsed().as_secs_f64() * 1000.0 / encodings.len() as f64
    };
    println!("  {verify_k:<8} {:<9.1}% {verify_ms:<9.1}ms  (verify baseline)", 100.0);
    for &k in &draft_ks {
        let draft_walk = WalkFfn::new(weights, &index, k);
        // Warmup
        {
            let (_, token_ids) = &encodings[0];
            let _ = predict_honest_with_knn_ffn(
                weights, tokenizer, token_ids, 1, &index, &*be, &cache,
                0..num_layers, None, Some(&draft_walk),
            );
        }
        let mut agreeing = 0usize;
        let mut disagreements: Vec<String> = Vec::new();
        let t0 = std::time::Instant::now();
        for ((prompt, token_ids), verify_tok) in encodings.iter().zip(verify_tokens.iter()) {
            let r = predict_honest_with_knn_ffn(
                weights, tokenizer, token_ids, 1, &index, &*be, &cache,
                0..num_layers, None, Some(&draft_walk),
            );
            let tok_str = r.predictions.first().map(|(t, _)| t.clone()).unwrap_or_default();
            let tok_id_num: u32 = tokenizer.encode(tok_str.as_str(), false)
                .ok().and_then(|e| e.get_ids().first().copied()).unwrap_or(u32::MAX);
            if tok_id_num == *verify_tok {
                agreeing += 1;
            } else {
                let verify_str = tokenizer.decode(&[*verify_tok], true).unwrap_or_default();
                disagreements.push(format!("[{prompt:.30}: {:?}→{:?}]", verify_str.trim(), tok_str.trim()));
            }
        }
        let draft_ms = t0.elapsed().as_secs_f64() * 1000.0 / encodings.len() as f64;
        let rate = agreeing as f64 * 100.0 / encodings.len() as f64;
        let mism_preview: String = disagreements.iter().take(2).cloned().collect::<Vec<_>>().join(", ");
        println!("  {k:<8} {rate:<9.1}% {draft_ms:<9.1}ms  {mism_preview:<40}");
    }
    println!();
    println!("  Guideline: specdec wins iff (draft_ms × k_draft + verify_ms) < verify_ms × accepted+1.");
    println!("  I.e. at 100% acceptance and k_draft=4, need draft_ms < 0.75 × verify_ms.");
    Ok(())
}
