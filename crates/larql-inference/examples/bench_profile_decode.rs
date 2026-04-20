//! P1: profile the production walk-only decode path.
//!
//! Mirrors larql-server's `--walk-only` decode loop exactly:
//!   WalkFfn::new_with_backend(weights, &index, 1024, &backend)
//!   predict_honest_with_knn_ffn(...)
//!
//! Run with LARQL_TRACE_DECODE=1 to get per-layer attn/ffn/lm_head ms breakdown
//! per decode step.

extern crate blas_src;

use std::time::Instant;

use larql_inference::{
    predict_honest_with_knn_ffn, CachedLayerGraph, InferenceModel, default_backend,
};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use larql_compute::ComputeBackend;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut prompt = String::from("The capital of France is");
    let mut decode_tokens: usize = 6;
    let mut walk_top_k: usize = 1024;  // production default
    let mut report_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--prompt" => { i += 1; prompt = args[i].clone(); }
            "--tokens" => { i += 1; decode_tokens = args[i].parse()?; }
            "--top-k"  => { i += 1; walk_top_k = args[i].parse()?; }
            "--report" => { i += 1; report_path = Some(args[i].clone()); }
            _ => {}
        }
        i += 1;
    }

    // Enable perf accumulator if a report path was given (or via env var).
    if report_path.is_some() || std::env::var("LARQL_PERF_RECORD").ok().as_deref() == Some("1") {
        larql_inference::perf::enable();
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let num_layers = weights.num_layers;
    let be = default_backend();

    // Production loads
    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;
    index.load_down_features(std::path::Path::new(&vindex_path))?;
    index.load_up_features(std::path::Path::new(&vindex_path))?;
    let _ = index.load_lm_head(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q8(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved_q4k_real(std::path::Path::new(&vindex_path));

    let enc = tokenizer.encode(prompt.as_str(), true).map_err(|e| format!("{e}"))?;
    let prompt_ids: Vec<u32> = enc.get_ids().to_vec();

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  P1: production walk-only decode profile                  ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  prompt: {:?} ({} tokens)", prompt, prompt_ids.len());
    println!("  WalkFfn::new_with_backend(weights, &index, {walk_top_k}, &backend)");
    println!();

    // Production WalkFfn
    let walk = WalkFfn::new_with_backend(weights, &index, walk_top_k, &*be);
    let cache = CachedLayerGraph::from_residuals(Vec::new());

    // Warmup (fault in mmaps, JIT shaders, decode f16 if any)
    for _ in 0..2 {
        be.reset_kv_cache();
        let _ = predict_honest_with_knn_ffn(
            weights, tokenizer, &prompt_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk),
        );
    }

    let n_runs: usize = std::env::var("LARQL_BENCH_RUNS").ok().and_then(|s| s.parse().ok()).unwrap_or(3);
    let mut prefill_runs: Vec<f64> = Vec::with_capacity(n_runs);
    let mut decode_runs: Vec<f64> = Vec::with_capacity(n_runs);
    let mut last_emitted: Vec<String> = Vec::new();

    // Reset accumulator after warmup so only the timed runs contribute.
    // Disable during prefill so per-component stats reflect ONLY single-token
    // decode work — multi-token prefill has very different per-call cost shape.
    larql_inference::perf::reset();
    let mut per_step_decode_ms: Vec<f64> = Vec::with_capacity(n_runs * decode_tokens);

    for _run in 0..n_runs {
        be.reset_kv_cache();
        larql_inference::perf::disable();
        let t_pref = Instant::now();
        let pref = predict_honest_with_knn_ffn(
            weights, tokenizer, &prompt_ids, 1, &index, &*be, &cache,
            0..num_layers, None, Some(&walk),
        );
        prefill_runs.push(t_pref.elapsed().as_secs_f64() * 1000.0);
        larql_inference::perf::enable();

        let mut current = pref.raw_predictions.first().map(|(id, _, _)| *id).unwrap_or(0);
        let mut emitted: Vec<String> = vec![tokenizer.decode(&[current], true).unwrap_or_default()];
        let t_dec = Instant::now();
        for _ in 1..decode_tokens {
            let t_step = Instant::now();
            let r = predict_honest_with_knn_ffn(
                weights, tokenizer, &[current], 1, &index, &*be, &cache,
                0..num_layers, None, Some(&walk),
            );
            per_step_decode_ms.push(t_step.elapsed().as_secs_f64() * 1000.0);
            current = r.raw_predictions.first().map(|(id, _, _)| *id).unwrap_or(0);
            emitted.push(tokenizer.decode(&[current], true).unwrap_or_default());
        }
        decode_runs.push(t_dec.elapsed().as_secs_f64() * 1000.0 / (decode_tokens - 1) as f64);
        last_emitted = emitted;
    }

    let stats = |v: &[f64]| {
        let mean = v.iter().sum::<f64>() / v.len() as f64;
        let std = (v.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / v.len() as f64).sqrt();
        let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
        let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        (mean, std, min, max)
    };
    let (pf_m, pf_s, pf_mn, pf_mx) = stats(&prefill_runs);
    let (dc_m, dc_s, dc_mn, dc_mx) = stats(&decode_runs);

    println!("\n  ── Multi-run stats ({n_runs} runs × prefill + {decode_tokens}-decode) ──");
    println!("  prefill:   {pf_m:>6.0} ± {pf_s:>4.0} ms      (min {pf_mn:>5.0}, max {pf_mx:>5.0})");
    println!("  decode:    {dc_m:>6.1} ± {dc_s:>4.1} ms/tok  (min {dc_mn:>5.1}, max {dc_mx:>5.1})");
    println!("  tok/s:     {:.2}                  (range {:.2} - {:.2})",
        1000.0 / dc_m, 1000.0 / dc_mx, 1000.0 / dc_mn);
    println!("  generated: {:?}", last_emitted.join(""));

    if let Some(path) = report_path.as_ref() {
        let generated = last_emitted.join("");
        let meta = larql_inference::perf::ReportMeta {
            model: &model_ref,
            vindex: &vindex_path,
            prompt: &prompt,
            prompt_tokens: prompt_ids.len(),
            decode_tokens,
            n_runs,
            backend: be.name(),
            generated: &generated,
            decode_tok_s_mean: 1000.0 / dc_m,
            decode_ms_mean: dc_m,
            decode_ms_std: dc_s,
            decode_ms_min: dc_mn,
            decode_ms_max: dc_mx,
            prefill_ms_mean: pf_m,
            prefill_ms_std: pf_s,
            prefill_ms_min: pf_mn,
            prefill_ms_max: pf_mx,
            per_step_decode_ms: &per_step_decode_ms,
        };
        let md = larql_inference::perf::report_markdown(&meta);
        std::fs::write(path, md)?;
        println!("\n  perf report written to: {path}");
    }
    Ok(())
}
