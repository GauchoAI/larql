//! P16(c) — HNSW gate-row recall study.
//!
//! For each (layer, K, ef_search), measures recall of HNSW top-K vs brute-force
//! top-K against real residual queries captured from a forward pass.
//! Decision criterion: if recall@K >= 99% holds for K~4096 across all 34 layers
//! at HNSW query time below the gate matvec time it would replace, the
//! sparse-walk path is viable. Otherwise it is not.

extern crate blas_src;

use std::collections::HashSet;
use std::time::Instant;

use ndarray::Array1;

use larql_inference::InferenceModel;
use larql_inference::forward::trace::capture_residuals;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "The capital of Germany is",
    "Photosynthesis is the process by which plants",
    "The chemical symbol for gold is",
    "Albert Einstein developed the theory of",
    "Water boils at one hundred degrees",
];

const LAYERS: &[usize] = &[0, 5, 10, 15, 20, 25, 30, 33];
const KS:     &[usize] = &[1024, 2048, 4096];
const EFS:    &[usize] = &[128, 256, 512, 1024, 2048];
const REPS_PER_QUERY: usize = 3;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut report_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--report" => { i += 1; report_path = Some(args[i].clone()); }
            _ => {}
        }
        i += 1;
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;

    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║  P16(c) — HNSW gate-row recall study                         ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!("  vindex: {vindex_path}");
    println!("  prompts: {}    layers: {:?}", PROMPTS.len(), LAYERS);
    println!("  Ks: {:?}    efs: {:?}", KS, EFS);
    println!();

    // 1. Capture real residuals at each studied layer for each prompt.
    println!("Step 1: capture residuals via real forward passes ({} prompts)…", PROMPTS.len());
    let t_cap = Instant::now();
    let mut residuals_per_prompt: Vec<Vec<(usize, Array1<f32>)>> = Vec::new();
    for (i, p) in PROMPTS.iter().enumerate() {
        let token_ids = tokenizer.encode(*p, true).expect("enc").get_ids().to_vec();
        let captured = capture_residuals(weights, &token_ids, LAYERS);
        let arrayed: Vec<(usize, Array1<f32>)> = captured.into_iter()
            .map(|(l, v)| (l, Array1::from_vec(v))).collect();
        println!("  prompt {i}: {} layer residuals captured (toks={})",
            arrayed.len(), token_ids.len());
        residuals_per_prompt.push(arrayed);
    }
    println!("  capture took {:.2}s", t_cap.elapsed().as_secs_f64());
    println!();

    // 2. Brute-force top-K per (prompt, layer, K) for ground truth.
    println!("Step 2: brute-force ground-truth top-K…");
    index.disable_hnsw();
    let max_k = *KS.iter().max().unwrap();
    let t_bf = Instant::now();
    // bf_results[prompt_idx][layer_idx] = bf top-max_k as Vec<usize>, plus query time.
    let mut bf_results: Vec<Vec<(Vec<usize>, f64)>> = Vec::new();
    for (pi, layer_residuals) in residuals_per_prompt.iter().enumerate() {
        let mut per_layer: Vec<(Vec<usize>, f64)> = Vec::new();
        for (li, &target_layer) in LAYERS.iter().enumerate() {
            let (_, ref residual) = layer_residuals[li];
            // Warm
            let _ = index.gate_knn(target_layer, residual, max_k);
            let mut elapsed = 0.0f64;
            let mut last: Vec<usize> = Vec::new();
            for _ in 0..REPS_PER_QUERY {
                let t = Instant::now();
                let bf = index.gate_knn(target_layer, residual, max_k);
                elapsed += t.elapsed().as_secs_f64() * 1000.0;
                last = bf.into_iter().map(|(f, _)| f).collect();
            }
            elapsed /= REPS_PER_QUERY as f64;
            per_layer.push((last, elapsed));
            println!("  prompt {pi} layer {target_layer}: bf top-{max_k} ({:.2} ms)", elapsed);
        }
        bf_results.push(per_layer);
    }
    println!("  brute-force pass took {:.2}s", t_bf.elapsed().as_secs_f64());
    println!();

    // 3. HNSW sweep for each (ef, K, layer): measure recall + time.
    println!("Step 3: HNSW sweep — ef × K × layer × prompts…");
    // result[ef_idx][k_idx][layer_idx] = (mean_recall, mean_query_ms)
    let mut sweep: Vec<Vec<Vec<(f64, f64)>>> = vec![
        vec![vec![(0.0, 0.0); LAYERS.len()]; KS.len()];
        EFS.len()
    ];

    for (ei, &ef) in EFS.iter().enumerate() {
        index.enable_hnsw(ef);
        // Warm HNSW build for each studied layer (build is lazy on first query).
        for &target_layer in LAYERS {
            if let Some((_, ref r0)) = residuals_per_prompt[0].iter().find(|(l, _)| *l == target_layer) {
                let _ = index.gate_knn(target_layer, r0, *KS.last().unwrap());
            }
        }

        for (ki, &k) in KS.iter().enumerate() {
            for (li, &target_layer) in LAYERS.iter().enumerate() {
                let mut total_recall = 0.0f64;
                let mut total_ms = 0.0f64;
                let mut samples = 0usize;
                for (pi, layer_residuals) in residuals_per_prompt.iter().enumerate() {
                    let (_, ref residual) = layer_residuals[li];
                    let bf_topk: HashSet<usize> = bf_results[pi][li].0
                        .iter().take(k).copied().collect();
                    // Warm
                    let _ = index.gate_knn(target_layer, residual, k);
                    let mut last_recall = 0.0f64;
                    let mut elapsed = 0.0f64;
                    for _ in 0..REPS_PER_QUERY {
                        let t = Instant::now();
                        let hnsw = index.gate_knn(target_layer, residual, k);
                        elapsed += t.elapsed().as_secs_f64() * 1000.0;
                        let hnsw_set: HashSet<usize> = hnsw.into_iter()
                            .map(|(f, _)| f).collect();
                        let overlap = bf_topk.intersection(&hnsw_set).count();
                        last_recall = overlap as f64 / k as f64;
                    }
                    elapsed /= REPS_PER_QUERY as f64;
                    total_recall += last_recall;
                    total_ms += elapsed;
                    samples += 1;
                }
                sweep[ei][ki][li] = (
                    total_recall / samples as f64,
                    total_ms / samples as f64,
                );
            }
            println!("  ef={ef:>4}  K={k:>5}: mean recall across {} layers = {:.2}%, mean qtime = {:.2} ms",
                LAYERS.len(),
                100.0 * sweep[ei][ki].iter().map(|(r, _)| r).sum::<f64>() / LAYERS.len() as f64,
                sweep[ei][ki].iter().map(|(_, t)| t).sum::<f64>() / LAYERS.len() as f64,
            );
        }
    }

    // 4. Markdown report.
    let mut md = String::new();
    md.push_str("# P16(c) — HNSW gate-row recall study\n\n");
    md.push_str(&format!("Vindex: `{vindex_path}`\n"));
    md.push_str(&format!("Prompts: {}\nLayers studied: {:?}\nKs: {:?}\nef_search values: {:?}\nReps per query: {}\n\n",
        PROMPTS.len(), LAYERS, KS, EFS, REPS_PER_QUERY));

    // Brute-force per-layer baseline timing
    md.push_str("## Brute-force baseline (full gate matvec at top-K=4096)\n\n");
    md.push_str("| Layer | mean BF query time (ms) |\n|---:|---:|\n");
    for (li, &lyr) in LAYERS.iter().enumerate() {
        let mean_bf_ms: f64 = bf_results.iter()
            .map(|p| p[li].1).sum::<f64>() / bf_results.len() as f64;
        md.push_str(&format!("| {lyr} | {:.2} |\n", mean_bf_ms));
    }
    md.push_str("\nThis is the cost the HNSW path would replace.\n\n");

    // Recall summary tables — one per K
    for (ki, &k) in KS.iter().enumerate() {
        md.push_str(&format!("## Recall@{k} as a function of (ef_search, layer)\n\n"));
        md.push_str("| ef \\ layer ");
        for &lyr in LAYERS { md.push_str(&format!("| L{lyr:02} ")); }
        md.push_str("| **mean** |\n|---:");
        for _ in 0..LAYERS.len() { md.push_str("|---:"); }
        md.push_str("|---:|\n");
        for (ei, &ef) in EFS.iter().enumerate() {
            md.push_str(&format!("| {ef} "));
            let mut sum = 0.0;
            for li in 0..LAYERS.len() {
                let (r, _) = sweep[ei][ki][li];
                md.push_str(&format!("| {:.1} % ", 100.0 * r));
                sum += r;
            }
            md.push_str(&format!("| **{:.1} %** |\n", 100.0 * sum / LAYERS.len() as f64));
        }
        md.push_str("\n");
        md.push_str(&format!("### HNSW query time @K={k} (ms, mean across layers)\n\n"));
        md.push_str("| ef | mean qtime (ms) | speedup vs BF |\n|---:|---:|---:|\n");
        for (ei, &ef) in EFS.iter().enumerate() {
            let mean_q: f64 = sweep[ei][ki].iter().map(|(_, t)| t).sum::<f64>() / LAYERS.len() as f64;
            let mean_bf: f64 = bf_results.iter().enumerate()
                .map(|(pi, _)| bf_results[pi].iter().map(|(_, t)| t).sum::<f64>())
                .sum::<f64>() / (bf_results.len() * LAYERS.len()) as f64;
            md.push_str(&format!("| {ef} | {:.2} | {:.2}× |\n", mean_q, mean_bf / mean_q));
        }
        md.push_str("\n");
    }

    md.push_str("## Decision criterion\n\n");
    md.push_str("- **Ship** if mean recall@4096 >= 99 % AND HNSW qtime < 50 % of BF qtime at the chosen ef.\n");
    md.push_str("- **Try alternatives** (LSH, IVF) if recall ceiling < 95 % even at ef=2048.\n");
    md.push_str("- **Drop entirely** if recall < 90 % at ef=2048: graph search isn't extracting the structure.\n\n");

    if let Some(path) = report_path {
        std::fs::write(&path, &md)?;
        println!();
        println!("Report written to: {path}");
    } else {
        println!();
        println!("{md}");
    }

    Ok(())
}
