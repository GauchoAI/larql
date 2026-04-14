//! Build lm_head.bin (f32) from embeddings.bin (f16) for tied-embedding models
//! like Gemma, where the LM head shares weights with the token embedding table.
//!
//! Usage:
//!   cargo run --release -p larql-vindex --example build_lm_head_from_embed -- <vindex_dir>

use std::io::Write;
use std::path::Path;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let dir = std::env::args().nth(1)
        .ok_or("Usage: build_lm_head_from_embed <vindex_dir>")?;
    let dir = Path::new(&dir);

    let src = dir.join("embeddings.bin");
    let dst = dir.join("lm_head.bin");

    if !src.exists() { return Err("embeddings.bin not found".into()); }
    if dst.exists() {
        println!("lm_head.bin already exists. Overwriting.");
    }

    let config: serde_json::Value = serde_json::from_str(
        &std::fs::read_to_string(dir.join("index.json"))?
    )?;
    let dtype = config.get("dtype").and_then(|v| v.as_str()).unwrap_or("f32");
    let hidden_size = config["hidden_size"].as_u64().unwrap() as usize;

    println!("Source: {}", src.display());
    println!("dtype: {dtype}, hidden: {hidden_size}");

    let t0 = Instant::now();
    let file = std::fs::File::open(&src)?;
    let mmap = unsafe { memmap2::Mmap::map(&file)? };

    let f32_data: Vec<f32> = match dtype {
        "f16" => larql_models::quant::half::decode_f16(&mmap),
        "f32" => {
            let n = mmap.len() / 4;
            unsafe { std::slice::from_raw_parts(mmap.as_ptr() as *const f32, n) }.to_vec()
        }
        other => return Err(format!("unsupported dtype: {other}").into()),
    };

    let vocab = f32_data.len() / hidden_size;
    println!("vocab_size: {vocab}");
    println!("writing {} floats ({:.1} MB)", f32_data.len(), (f32_data.len() * 4) as f64 / 1e6);

    let f32_bytes: &[u8] = unsafe {
        std::slice::from_raw_parts(
            f32_data.as_ptr() as *const u8,
            f32_data.len() * 4,
        )
    };
    let mut w = std::io::BufWriter::new(std::fs::File::create(&dst)?);
    w.write_all(f32_bytes)?;
    w.flush()?;

    println!("Wrote {} in {:.1}s", dst.display(), t0.elapsed().as_secs_f64());
    Ok(())
}
