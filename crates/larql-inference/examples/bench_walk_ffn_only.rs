//! Micro-bench: isolate walk FFN cost. No attention, no embedding, no prefill.
//! Just calls WalkFfn::forward repeatedly on synthetic hidden states.

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
    let mut layer: usize = 26;
    let mut iters: usize = 50;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--ef"     => { i += 1; ef_search = args[i].parse()?; }
            "--layer"  => { i += 1; layer = args[i].parse()?; }
            "--iters"  => { i += 1; iters = args[i].parse()?; }
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

    // Synthetic single-token hidden state
    let x = Array2::<f32>::from_elem((1, hidden), 0.01);

    println!("╔═══════════════════════════════════════════════════════════╗");
    println!("║  Walk FFN micro-bench at layer {layer}                          ║");
    println!("╚═══════════════════════════════════════════════════════════╝");
    println!("  hidden={hidden}, intermediate={}, iters={iters}", index.num_features(layer));
    println!();

    let configs = vec![
        ("brute-force unlimited", false, 0usize),
        ("brute-force top_k=64", false, 64),
        ("brute-force top_k=256", false, 256),
        ("HNSW top_k=64 (ef=64)", true, 64),
        ("HNSW top_k=256 (ef=256)", true, 256),
    ];

    for (label, use_hnsw, top_k) in configs {
        if use_hnsw { index.enable_hnsw(ef_search.max(top_k)); }
        else { index.disable_hnsw(); }

        let walk = if top_k == 0 {
            WalkFfn::new_unlimited(weights, &index)
        } else {
            WalkFfn::new(weights, &index, top_k)
        };

        // Warmup (triggers HNSW build on first call)
        for _ in 0..2 { let _ = walk.forward(layer, &x); }

        let t0 = Instant::now();
        for _ in 0..iters { let _ = walk.forward(layer, &x); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        println!("  {label:<28} {ms:>8.2} ms/call");
    }
    Ok(())
}
