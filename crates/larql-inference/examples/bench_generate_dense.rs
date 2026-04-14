//! Bench multi-token generation via the dense `predict_honest` path.
//! No Q4 quantization required — loads the model once, loops predict-next-token
//! with token appending, and reports tokens/second.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference --example bench_generate_dense -- \
//!     --model /path/to/gemma-3-4b-it \
//!     --vindex /path/to/gemma3-4b.vindex \
//!     --prompt "def fibonacci(n):" \
//!     --tokens 80

use std::time::Instant;

use larql_inference::{
    InferenceModel, CachedLayerGraph, default_backend,
};
use larql_inference::ffn::WeightFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = std::path::PathBuf::from("output/gemma3-4b-v2.vindex");
    let mut model_ref = String::from("google/gemma-3-4b-it");
    let mut prompt = String::from("The capital of France is");
    let mut n_tokens: usize = 20;
    let mut i = 1;
    while i < args.len() {
        if args[i] == "--vindex" { i += 1; vindex_path = std::path::PathBuf::from(&args[i]); }
        else if args[i] == "--model" { i += 1; model_ref = args[i].clone(); }
        else if args[i] == "--prompt" { i += 1; prompt = args[i].clone(); }
        else if args[i] == "--tokens" { i += 1; n_tokens = args[i].parse().unwrap_or(20); }
        i += 1;
    }

    println!("Loading model...");
    let load0 = Instant::now();
    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let _ = index.load_lm_head(&vindex_path);
    println!("  Loaded in {:.2}s", load0.elapsed().as_secs_f64());

    let gpu_be = default_backend();
    let dense_ffn = WeightFfn { weights };
    // Empty cached_layers so the prefill populates KV cache for ALL layers,
    // enabling decode_token (seq=1) for subsequent tokens.
    let cached_layers: Vec<usize> = Vec::new();

    let encoding = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
    let mut token_ids: Vec<u32> = encoding.get_ids().to_vec();
    let prompt_len = token_ids.len();

    println!();
    println!("  Prompt: {:?} ({} tokens)", prompt, prompt_len);
    println!("  Backend: {}", gpu_be.name());
    let cache_range = if cached_layers.is_empty() {
        "none (full prefill)".to_string()
    } else {
        format!("0-{} cached, {}-{} compute", cached_layers.last().unwrap(), cached_layers.last().unwrap() + 1, num_layers - 1)
    };
    println!("  Layers: {num_layers} ({cache_range})");
    println!("  Generating {} tokens via predict_honest (KV-cached decode)...", n_tokens);
    println!();

    let gen_start = Instant::now();
    let mut per_tok_ms: Vec<f64> = Vec::with_capacity(n_tokens);
    let mut generated_text = String::new();

    use std::io::Write as _;
    print!("{}", prompt);
    std::io::stdout().flush().ok();

    // Cache over the (empty) cached_layers — only built once, reused across steps.
    let cache = CachedLayerGraph::build(weights, &token_ids, &cached_layers, &dense_ffn);

    // Call 1: full prompt — prefill populates KV for all layers.
    // Call 2+: just the new token — seq=1 decode path uses the KV cache.
    let mut input_tokens: Vec<u32> = token_ids.clone();

    for step in 0..n_tokens {
        let t0 = Instant::now();

        let r = larql_inference::layer_graph::predict::predict_honest(
            weights, tokenizer, &input_tokens, 1,
            &index, &*gpu_be, &cache, (cached_layers.len())..num_layers,
        );

        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        per_tok_ms.push(ms);

        if let Some(&(tid, _logit, _prob)) = r.raw_predictions.first() {
            let tok_text = r.predictions.first().map(|(s, _)| s.clone()).unwrap_or_default();
            generated_text.push_str(&tok_text);
            token_ids.push(tid);
            print!("{}", tok_text);
            std::io::stdout().flush().ok();
            let phase = if step == 0 { "prefill" } else { "decode" };
            eprintln!("  [{} step {:>3}  tid={:>6}  {:>7.1}ms  {:>5.2} tok/s]",
                phase, step + 1, tid, ms, 1000.0 / ms);

            // Subsequent steps: feed ONLY the new token (seq=1), KV cache has the history.
            input_tokens = vec![tid];
        } else {
            eprintln!("  [step {:>3}: empty prediction, stopping]", step + 1);
            break;
        }
    }
    println!();

    let total_ms = gen_start.elapsed().as_secs_f64() * 1000.0;
    let avg_ms = if !per_tok_ms.is_empty() { per_tok_ms.iter().sum::<f64>() / per_tok_ms.len() as f64 } else { 0.0 };
    let tok_per_s = if avg_ms > 0.0 { 1000.0 / avg_ms } else { 0.0 };

    println!();
    println!("  Generated:   {:?}", generated_text);
    println!("  Tokens:      {}", per_tok_ms.len());
    println!("  Total:       {:.1}s", total_ms / 1000.0);
    println!("  Average:     {:.1}ms/tok  ({:.2} tok/s)", avg_ms, tok_per_s);

    Ok(())
}
