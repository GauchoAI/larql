//! Idea 2 bench: time `run_attention_with_kv_backend_opt` across all layers for
//! a short prompt, with GPU fused attention on vs off. Measures the prefill
//! attention cost in isolation — no FFN, no embedding, no logits.
//!
//! Usage:
//!   cargo run --release --features metal -p larql-inference \
//!     --example bench_prefill_attn -- --model /path/to/gemma-3-4b-it
extern crate blas_src;

use std::time::Instant;

use larql_inference::{InferenceModel, default_backend};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut prompt = String::from("The capital of France is");
    let mut runs: usize = 3;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--runs"   => { i += 1; runs = args[i].parse().unwrap_or(3); }
            _ => {}
        }
        i += 1;
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;

    let be = default_backend();

    let enc = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
    let token_ids: Vec<u32> = enc.get_ids().to_vec();

    println!("╔═══════════════════════════════════════════════════════╗");
    println!("║  Idea 2 — Metal prefill attention (fused)             ║");
    println!("╚═══════════════════════════════════════════════════════╝");
    println!("  prompt: {prompt:?}  ({} tokens)", token_ids.len());
    println!("  layers: {num_layers}");
    println!("  backend: {}", be.name());
    println!();

    let embed_once = larql_inference::forward::embed_tokens_pub(weights, &token_ids);

    let bench = |gpu_on: bool, label: &str| -> f64 {
        // Toggle via env var read inside the call site.
        if gpu_on {
            std::env::remove_var("LARQL_GPU_PREFILL_ATTN");
        } else {
            std::env::set_var("LARQL_GPU_PREFILL_ATTN", "0");
        }
        // Warmup
        let _ = run_all_layers(weights, &embed_once, num_layers, &*be);
        // Measure
        let mut best = f64::INFINITY;
        let mut avg = 0.0;
        for _ in 0..runs {
            let t0 = Instant::now();
            let _ = run_all_layers(weights, &embed_once, num_layers, &*be);
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            if ms < best { best = ms; }
            avg += ms;
        }
        avg /= runs as f64;
        println!("  {label:<28}  avg {avg:>7.1} ms   best {best:>7.1} ms");
        avg
    };

    // Correctness: compute final hidden state on both paths, report cosine.
    std::env::set_var("LARQL_GPU_PREFILL_ATTN", "0");
    let h_cpu = run_all_layers(weights, &embed_once, num_layers, &*be);
    std::env::set_var("LARQL_GPU_PREFILL_ATTN", "force");
    let h_gpu = run_all_layers(weights, &embed_once, num_layers, &*be);
    let a = h_cpu.as_slice().unwrap();
    let b = h_gpu.as_slice().unwrap();
    let mut dot = 0.0f64; let mut na = 0.0f64; let mut nb = 0.0f64;
    let mut max_abs = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x as f64 * y as f64;
        na += x as f64 * x as f64;
        nb += y as f64 * y as f64;
        let d = (x - y).abs();
        if d > max_abs { max_abs = d; }
    }
    let cos = dot / (na.sqrt() * nb.sqrt());
    println!("  correctness: cos(CPU, GPU final h) = {cos:.6}   max|Δ| = {max_abs:.4e}");
    println!();

    println!("  Timing `run_attention_with_kv_backend_opt` across all {num_layers} layers:");
    let cpu_ms = bench(false, "CPU attention (baseline)");
    let gpu_ms = bench(true,  "GPU fused attention");
    let speedup = cpu_ms / gpu_ms;
    let delta = cpu_ms - gpu_ms;
    println!();
    println!("  speedup: {speedup:.2}x   savings: {delta:.1} ms/prefill");
    Ok(())
}

fn run_all_layers(
    weights: &larql_inference::ModelWeights,
    h_embed: &ndarray::Array2<f32>,
    num_layers: usize,
    backend: &dyn larql_compute::ComputeBackend,
) -> ndarray::Array2<f32> {
    let mut h = h_embed.clone();
    for layer in 0..num_layers {
        let (h_post_attn, _k, _v) = larql_inference::attention::gpu::run_attention_with_kv_backend(
            weights, &h, layer, Some(backend),
        ).expect("attention succeeds");
        // Skip FFN — we're only timing the attention section.
        // Residual is already applied inside run_attention_with_kv_backend,
        // so `h_post_attn` is a valid next-layer input for attention math.
        h = h_post_attn;
    }
    h
}
