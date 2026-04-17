//! Build `interleaved_q4.bin` — Q4_0 quantized gate+up+down, one contiguous
//! block per layer. Format matches what `WalkFfn::walk_ffn_q4_interleaved`
//! expects: `intermediate * hidden / 32 * 18` bytes per matrix, three
//! matrices per layer (gate, up, down).
//!
//! Source: `interleaved.bin` (f32, 10.2 GB on Gemma 3 4B).
//! Output: `interleaved_q4.bin` (~1.8 GB).
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_interleaved_q4 -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

use larql_compute::cpu::ops::q4_common::quantize_q4_0;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .unwrap_or_else(|| { eprintln!("Usage: build_interleaved_q4 <vindex_dir>"); std::process::exit(1); });
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
    let bytes_per_matrix_q4 = floats_per_matrix / 32 * 18;
    let bytes_per_layer_q4 = bytes_per_matrix_q4 * 3;

    println!("=== Build Q4_0 interleaved vindex ===\n");
    println!("Layers: {num_layers}, hidden: {hidden}, intermediate: {inter}");
    println!("Per matrix: f32 {:.1} MB → Q4_0 {:.1} MB  ({:.1}× smaller)",
        bytes_per_matrix_f32 as f64 / 1e6,
        bytes_per_matrix_q4 as f64 / 1e6,
        bytes_per_matrix_f32 as f64 / bytes_per_matrix_q4 as f64);
    println!("Total out: {:.2} GB\n", (bytes_per_layer_q4 * num_layers) as f64 / 1e9);

    let src_path = dir.join("interleaved.bin");
    let src_file = std::fs::File::open(&src_path)
        .map_err(|e| format!("open {} failed: {e}", src_path.display()))?;
    let src = unsafe { memmap2::Mmap::map(&src_file)? };
    assert_eq!(src.len(), bytes_per_layer_f32 * num_layers,
        "interleaved.bin size mismatch: expected {}, got {}",
        bytes_per_layer_f32 * num_layers, src.len());

    let out_path = dir.join("interleaved_q4.bin");
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
            let q = quantize_q4_0(f32_slice);
            assert_eq!(q.len(), bytes_per_matrix_q4,
                "L{layer} {name}: Q4_0 size mismatch ({} vs {})", q.len(), bytes_per_matrix_q4);
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
    let elapsed = t0.elapsed();
    println!("\nWrote: {} ({:.2} GB in {:.1}s)",
        out_path.display(),
        total_bytes as f64 / 1e9,
        elapsed.as_secs_f64());
    Ok(())
}
