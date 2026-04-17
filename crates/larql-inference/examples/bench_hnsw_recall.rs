//! Does HNSW return the same top-K features as brute-force?

extern crate blas_src;

use ndarray::Array1;

use larql_inference::InferenceModel;
use larql_vindex::{GateIndex, SilentLoadCallbacks, VectorIndex};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut top_k: usize = 64;
    let mut ef: usize = 64;
    let mut layer: usize = 26;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--top-k" => { i += 1; top_k = args[i].parse()?; }
            "--ef" => { i += 1; ef = args[i].parse()?; }
            "--layer" => { i += 1; layer = args[i].parse()?; }
            _ => {}
        }
        i += 1;
    }

    let _model = InferenceModel::load("/Users/miguel_lemos/Desktop/gemma-3-4b-it")?;

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;

    let hidden = index.hidden_size;
    let mut q = vec![0.0f32; hidden];
    let mut state: u64 = 42;
    for v in q.iter_mut() {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        *v = ((state >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0;
    }
    let residual = Array1::from_vec(q);

    println!("Layer {layer}, top_k={top_k}, ef={ef}");
    println!();

    // Brute force (HNSW disabled)
    index.disable_hnsw();
    let bf = index.gate_knn(layer, &residual, top_k);
    println!("BF top-{top_k} (brute-force):");
    for (i, (feat, score)) in bf.iter().take(10).enumerate() {
        println!("  #{i:2} feat={feat:<5} score={score:>+8.3}");
    }

    // HNSW
    index.enable_hnsw(ef);
    // Warm
    let _ = index.gate_knn(layer, &residual, top_k);
    let hnsw = index.gate_knn(layer, &residual, top_k);
    println!();
    println!("HNSW top-{top_k}:");
    for (i, (feat, score)) in hnsw.iter().take(10).enumerate() {
        println!("  #{i:2} feat={feat:<5} score={score:>+8.3}");
    }

    // Compute overlap
    let bf_set: std::collections::HashSet<usize> = bf.iter().map(|(f, _)| *f).collect();
    let hnsw_set: std::collections::HashSet<usize> = hnsw.iter().map(|(f, _)| *f).collect();
    let overlap = bf_set.intersection(&hnsw_set).count();
    println!();
    println!("Top-{top_k} overlap: {overlap}/{top_k} ({:.1}%)", overlap as f64 * 100.0 / top_k as f64);

    // Top-1 agreement
    let bf_top1 = bf.first().map(|(f, _)| *f);
    let hnsw_top1 = hnsw.first().map(|(f, _)| *f);
    println!("Top-1 BF={bf_top1:?}  HNSW={hnsw_top1:?}  match={}", bf_top1 == hnsw_top1);

    // Top-K magnitude agreement (are the HNSW scores CLOSE to BF's scores for the same features?)
    let bf_scores: std::collections::HashMap<usize, f32> = bf.iter().map(|(f, s)| (*f, *s)).collect();
    let mut max_diff = 0.0f32;
    for (feat, hnsw_score) in &hnsw {
        if let Some(bf_score) = bf_scores.get(feat) {
            let d = (hnsw_score - bf_score).abs();
            if d > max_diff { max_diff = d; }
        }
    }
    println!("Max score delta (common features): {max_diff:.6}");
    Ok(())
}
