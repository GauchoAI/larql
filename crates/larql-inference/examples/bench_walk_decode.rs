//! Decode-speed bench: how fast is the walk FFN per decode token,
//! BF vs HNSW? Simulates the decode loop: 1-token forward per step,
//! just through walk FFN layers (L13-33).

extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;

use larql_inference::{InferenceModel, ffn::FfnBackend};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut ef_search: usize = 64;
    let mut tokens: usize = 20;
    let mut layer_start: usize = 13;
    let mut layer_end: usize = 34;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--ef"     => { i += 1; ef_search = args[i].parse()?; }
            "--tokens" => { i += 1; tokens = args[i].parse()?; }
            "--layer-start" => { i += 1; layer_start = args[i].parse()?; }
            "--layer-end" => { i += 1; layer_end = args[i].parse()?; }
            _ => {}
        }
        i += 1;
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let hidden = weights.hidden_size;

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;
    index.load_down_features(std::path::Path::new(&vindex_path))?;
    index.load_up_features(std::path::Path::new(&vindex_path))?;

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Walk decode micro-bench: L{layer_start}-L{layer_end} ({} layers)           ║", layer_end - layer_start);
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  {tokens} decode tokens × {} layers = {} walk FFN calls", layer_end-layer_start, tokens*(layer_end-layer_start));
    println!();

    let x = Array2::<f32>::from_elem((1, hidden), 0.01);

    let configs: Vec<(&str, bool, usize)> = vec![
        ("brute-force top_k=8192",     false, 8192),
        ("brute-force unlimited",      false, 0),
        ("HNSW top_k=64  ef=64",        true, 64),
        ("HNSW top_k=256 ef=256",       true, 256),
        ("HNSW top_k=1024 ef=1024",     true, 1024),
    ];

    for (label, use_hnsw, top_k) in configs {
        if use_hnsw { index.enable_hnsw(ef_search.max(top_k)); }
        else { index.disable_hnsw(); }

        let walk = if top_k == 0 { WalkFfn::new_unlimited(weights, &index) }
                   else { WalkFfn::new(weights, &index, top_k) };

        // Warmup all layers (triggers HNSW build, gate decode, mmap faulting)
        for _ in 0..2 {
            for l in layer_start..layer_end { let _ = walk.forward(l, &x); }
        }

        let t0 = Instant::now();
        for _ in 0..tokens {
            for l in layer_start..layer_end { let _ = walk.forward(l, &x); }
        }
        let total_ms = t0.elapsed().as_secs_f64() * 1000.0;
        let per_token = total_ms / tokens as f64;
        let tok_per_sec = 1000.0 / per_token;
        println!("  {label:<32} {per_token:>7.1} ms/tok   {tok_per_sec:>5.1} tok/s");
    }
    Ok(())
}
