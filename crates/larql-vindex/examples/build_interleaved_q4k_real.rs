//! Build a REAL Q4_K interleaved file (`interleaved_q4k_real.bin`).
//!
//! Historical note: the existing `interleaved_q4k.bin` contains **Q6_K**
//! data despite its name (build_q4k_weights.rs uses quantize_q6_k for all
//! FFN matrices; 2.1 GB / 210-byte blocks confirms). This script emits a
//! genuine Q4_K file using `quantize_q4_k_gguf` (148 bytes / 256 values,
//! llama.cpp-compatible layout).
//!
//! Format per layer: gate (Q4_K) | up (Q4_K) | down (Q4_K), all three
//! with the same block layout `interleaved * hidden / 256 * 148` bytes.
//! Matches the layout assumption in `walk_ffn_q4k_interleaved`.
//!
//! Source: `interleaved.bin` (f32, 10.2 GB).
//! Output: `interleaved_q4k_real.bin` (~1.5 GB).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_interleaved_q4k_real -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

// 148-byte Ollama layout — matches the existing `q4k_matvec` Metal shader
// (Q4K_BLOCK_SIZE = 148 in crates/larql-compute/src/metal/shaders/q4k_matvec.rs).
// The 144-byte GGUF layout has its own shaders (q4kf_proj, q4kf_ffn_gate_up)
// but reusing the simpler `q4k_matvec` trait method is the straight path here.
use larql_compute::cpu::ops::q4_common::quantize_q4_k;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_interleaved_q4k_real <vindex_dir>"); std::process::exit(1); });
    let dir = Path::new(&dir);

    let config_path = dir.join("index.json");
    let config: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(&config_path)?)?;
    let num_layers = config["num_layers"].as_u64().unwrap() as usize;
    let hidden = config["hidden_size"].as_u64().unwrap() as usize;
    let inter = config["intermediate_size"].as_u64()
        .unwrap_or_else(|| config["num_features_per_layer"].as_u64().unwrap()) as usize;

    let floats_per_matrix = inter * hidden;
    let bytes_per_matrix_f32 = floats_per_matrix * 4;
    let bytes_per_layer_f32 = bytes_per_matrix_f32 * 3;
    let bytes_per_matrix_q4k = floats_per_matrix / 256 * 148; // Ollama Q4_K
    let bytes_per_layer_q4k = bytes_per_matrix_q4k * 3;

    println!("=== Build TRUE Q4_K interleaved vindex ===\n");
    println!("Layers: {num_layers}, hidden: {hidden}, intermediate: {inter}");
    println!("Per matrix: f32 {:.1} MB → Q4_K {:.1} MB  ({:.1}× smaller)",
        bytes_per_matrix_f32 as f64 / 1e6,
        bytes_per_matrix_q4k as f64 / 1e6,
        bytes_per_matrix_f32 as f64 / bytes_per_matrix_q4k as f64);
    println!("Total out: {:.2} GB\n", (bytes_per_layer_q4k * num_layers) as f64 / 1e9);

    let src_path = dir.join("interleaved.bin");
    let src_file = std::fs::File::open(&src_path)
        .map_err(|e| format!("open {} failed: {e}", src_path.display()))?;
    let src = unsafe { memmap2::Mmap::map(&src_file)? };
    assert_eq!(src.len(), bytes_per_layer_f32 * num_layers);

    let out_path = dir.join("interleaved_q4k_real.bin");
    let mut out = std::io::BufWriter::with_capacity(
        16 * 1024 * 1024,
        std::fs::File::create(&out_path)?,
    );

    let t0 = Instant::now();
    let mut total_bytes: u64 = 0;

    for layer in 0..num_layers {
        let layer_start_f32 = layer * bytes_per_layer_f32;
        for (i, name) in ["gate", "up", "down"].iter().enumerate() {
            let off = layer_start_f32 + i * bytes_per_matrix_f32;
            let f32_slice = unsafe {
                let ptr = src[off..off + bytes_per_matrix_f32].as_ptr() as *const f32;
                std::slice::from_raw_parts(ptr, floats_per_matrix)
            };

            // IMPORTANT: for `down`, transpose [intermediate, hidden] →
            // [hidden, intermediate] before quantizing. This lets inference
            // use the Metal `q4k_matvec` kernel with shape [hidden,
            // intermediate] × activation[intermediate] → out[hidden], which
            // is exactly activation @ down. Avoids needing a separate
            // Metal q4k_vecmat kernel for the down direction.
            let q = if *name == "down" {
                let mut transposed = vec![0.0f32; floats_per_matrix];
                for r in 0..inter {
                    for c in 0..hidden {
                        // src is row-major [inter, hidden]:
                        //   src[r * hidden + c]
                        // dst is row-major [hidden, inter]:
                        //   dst[c * inter + r]
                        transposed[c * inter + r] = f32_slice[r * hidden + c];
                    }
                }
                quantize_q4_k(&transposed)
            } else {
                quantize_q4_k(f32_slice)
            };
            assert_eq!(q.len(), bytes_per_matrix_q4k,
                "L{layer} {name}: Q4_K size mismatch ({} vs {})", q.len(), bytes_per_matrix_q4k);
            out.write_all(&q)?;
            total_bytes += q.len() as u64;
        }
        if layer % 5 == 0 || layer == num_layers - 1 {
            let elapsed = t0.elapsed().as_secs_f64();
            let pct = (layer + 1) as f64 / num_layers as f64 * 100.0;
            println!("  L{layer:02}/{num_layers}  {:.1}%  elapsed {:.1}s", pct, elapsed);
        }
    }

    out.flush()?;
    println!("\nWrote: {} ({:.2} GB in {:.1}s)",
        out_path.display(),
        total_bytes as f64 / 1e9,
        t0.elapsed().as_secs_f64());
    Ok(())
}
