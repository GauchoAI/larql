//! Throughput check: load a GGUF, generate N tokens greedily, print tok/s.
//!
//!   cargo run --release -p larql-llamacpp --example bench -- \
//!     /tmp/gemma3-4b-stock-q8_0.gguf "The capital of Australia is " 128

use larql_llamacpp::{GenerateConfig, LlamaPipeline};
use std::path::PathBuf;
use std::time::Instant;

fn main() {
    let mut args = std::env::args().skip(1);
    let gguf: PathBuf = args
        .next()
        .unwrap_or_else(|| "/tmp/gemma3-4b-stock-q8_0.gguf".into())
        .into();
    let prompt = args
        .next()
        .unwrap_or_else(|| "The capital of Australia is ".into());
    let n_gen: usize = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    println!("Loading {gguf:?}...");
    let t0 = Instant::now();
    let mut pipe = LlamaPipeline::load(&gguf, 2048).expect("load failed");
    println!("  loaded in {:.1} ms (n_layer={}, n_embd={})",
        t0.elapsed().as_secs_f64() * 1000.0, pipe.n_layer(), pipe.n_embd());

    let cfg = GenerateConfig {
        max_tokens: n_gen,
        stop_at_eos: false,
    };
    let t1 = Instant::now();
    let out = pipe.generate(&prompt, &cfg).expect("generate failed");
    let dt = t1.elapsed().as_secs_f64();
    println!();
    println!("{}{}", prompt, out);
    println!();
    println!("{} tokens in {:.3}s = {:.2} tok/s", n_gen, dt, n_gen as f64 / dt);
}
