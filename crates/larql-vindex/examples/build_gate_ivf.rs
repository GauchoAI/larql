//! Build IVF (Inverted File Index) for gate vectors.
//!
//! Clusters gate vectors per layer using k-means, producing:
//!   gate_ivf.bin — centroids + feature→cluster assignments
//!
//! At query time, score C centroids instead of N features, then only
//! score features in the top-P clusters. Reduces gate scoring from
//! O(N × hidden) to O(C × hidden + P × avg_cluster_size × hidden).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_gate_ivf -- \
//!     /path/to/gemma3-4b.vindex --clusters 64

use std::io::Write;
use std::path::PathBuf;

use larql_vindex::{SilentLoadCallbacks, VectorIndex};
use ndarray::{Array2, ArrayView1, s};

const MAGIC: u32 = 0x49564647; // "GVFI" — Gate Vector File Index
const VERSION: u32 = 1;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let vindex_path = PathBuf::from(args.get(1).expect("usage: build_gate_ivf <vindex_dir> [--clusters N]"));
    let mut num_clusters: usize = 64;
    let mut i = 2;
    while i < args.len() {
        if args[i] == "--clusters" { i += 1; num_clusters = args[i].parse()?; }
        i += 1;
    }

    let mut cb = SilentLoadCallbacks;
    let index = VectorIndex::load_vindex(&vindex_path, &mut cb)?;
    let hidden = index.hidden_size;

    let out_path = vindex_path.join("gate_ivf.bin");
    let mut file = std::fs::File::create(&out_path)?;

    // Count layers with valid gate data
    let mut layer_ids: Vec<usize> = Vec::new();
    for layer in 0..64 {
        let n = index.num_features(layer);
        if n > 0 {
            if let Some(d) = index.gate_layer_f32(layer) {
                if d.len() == n * hidden { layer_ids.push(layer); }
            }
        }
    }
    let num_layers = layer_ids.len();

    eprintln!("[gate_ivf] {} layers, {} clusters, hidden={}", num_layers, num_clusters, hidden);

    // Header
    file.write_all(&MAGIC.to_le_bytes())?;
    file.write_all(&VERSION.to_le_bytes())?;
    file.write_all(&(num_layers as u32).to_le_bytes())?;
    file.write_all(&(num_clusters as u32).to_le_bytes())?;
    file.write_all(&(hidden as u32).to_le_bytes())?;

    for &layer in &layer_ids {
        let n = index.num_features(layer);
        let gate_data = index.gate_layer_f32(layer).unwrap();
        let gate = Array2::from_shape_vec((n, hidden), gate_data)?;

        eprintln!("[L{layer:02}] {n} features → {num_clusters} clusters...");

        let (centroids, assignments) = kmeans(&gate, num_clusters, 20);

        // Verify
        let max_cluster = *assignments.iter().max().unwrap_or(&0) as usize;
        assert!(max_cluster < num_clusters, "cluster id out of range");

        // Write layer record: layer_id(u32), num_features(u32),
        //   centroids[C × hidden × f32], assignments[N × u16]
        file.write_all(&(layer as u32).to_le_bytes())?;
        file.write_all(&(n as u32).to_le_bytes())?;
        // Centroids
        for val in centroids.iter() {
            file.write_all(&val.to_le_bytes())?;
        }
        // Assignments
        for &a in &assignments {
            file.write_all(&a.to_le_bytes())?;
        }

        // Stats
        let mut sizes = vec![0usize; num_clusters];
        for &a in &assignments { sizes[a as usize] += 1; }
        let avg = n as f64 / num_clusters as f64;
        let max_sz = *sizes.iter().max().unwrap_or(&0);
        let min_sz = *sizes.iter().min().unwrap_or(&0);
        eprintln!("  cluster sizes: min={min_sz} avg={avg:.0} max={max_sz}");
    }

    let file_size = file.metadata()?.len();
    eprintln!("[gate_ivf] wrote {} ({:.1} MB)", out_path.display(), file_size as f64 / 1e6);
    Ok(())
}

/// K-means clustering. Returns (centroids [C, hidden], assignments [N] as u16).
fn kmeans(data: &Array2<f32>, k: usize, max_iter: usize) -> (Array2<f32>, Vec<u16>) {
    let n = data.shape()[0];
    let d = data.shape()[1];
    let k = k.min(n);

    // K-means++ initialization
    let mut centroids = Array2::<f32>::zeros((k, d));
    let mut rng_state: u64 = 42;
    let mut next_rng = || -> usize {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        (rng_state >> 33) as usize
    };

    // First centroid: random
    let first = next_rng() % n;
    centroids.row_mut(0).assign(&data.row(first));

    // Remaining: proportional to distance from nearest centroid
    let mut min_dists = vec![f32::INFINITY; n];
    for c in 1..k {
        // Update min distances
        for i in 0..n {
            let d2 = l2_sq(&data.row(i), &centroids.row(c - 1));
            if d2 < min_dists[i] { min_dists[i] = d2; }
        }
        let total: f64 = min_dists.iter().map(|&d| d as f64).sum();
        let threshold = (next_rng() as f64 / u32::MAX as f64) * total;
        let mut cumulative = 0.0f64;
        let mut chosen = 0;
        for (i, &d) in min_dists.iter().enumerate() {
            cumulative += d as f64;
            if cumulative >= threshold { chosen = i; break; }
        }
        centroids.row_mut(c).assign(&data.row(chosen));
    }

    // Iterate
    let mut assignments = vec![0u16; n];
    for iter in 0..max_iter {
        // Assign
        let mut changed = 0usize;
        for i in 0..n {
            let mut best_c = 0u16;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let d2 = l2_sq(&data.row(i), &centroids.row(c));
                if d2 < best_d { best_d = d2; best_c = c as u16; }
            }
            if assignments[i] != best_c { changed += 1; }
            assignments[i] = best_c;
        }

        // Recompute centroids
        let mut sums = Array2::<f64>::zeros((k, d));
        let mut counts = vec![0usize; k];
        for i in 0..n {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for j in 0..d { sums[[c, j]] += data[[i, j]] as f64; }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..d { centroids[[c, j]] = (sums[[c, j]] / counts[c] as f64) as f32; }
            }
        }

        if iter > 0 && changed == 0 {
            eprintln!("  converged at iter {iter}");
            break;
        }
    }

    (centroids, assignments)
}

fn l2_sq(a: &ArrayView1<f32>, b: &ArrayView1<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| { let d = x - y; d * d }).sum()
}
