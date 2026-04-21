//! End-to-end probe check: zero `l_out-26` every decode step and verify
//! the output diverges from baseline.  Mirrors the throughput-spike
//! validation through the new public API.
//!
//!   cargo run --release -p larql-llamacpp --example probe_zero

use larql_llamacpp::{GenerateConfig, LlamaPipeline, ProbeHandler, ProbeNode};
use std::path::PathBuf;

struct ZeroOutL26 {
    hits: usize,
}

impl ProbeHandler for ZeroOutL26 {
    fn wants(&self, node: &ProbeNode<'_>) -> bool {
        node.name == "l_out-26"
    }
    fn observe(&mut self, _node: &ProbeNode<'_>, data: &[f32]) -> Option<Vec<f32>> {
        self.hits += 1;
        Some(vec![0.0; data.len()])
    }
}

fn main() {
    let gguf: PathBuf = std::env::var("LLAMA_GGUF")
        .unwrap_or_else(|_| "/tmp/gemma3-4b-stock-q8_0.gguf".into())
        .into();
    let prompt = "The capital of Australia is ";
    let cfg = GenerateConfig { max_tokens: 12, stop_at_eos: false };

    println!("--- BASELINE (no probe) ---");
    let mut a = LlamaPipeline::load(&gguf, 1024).expect("load");
    let out_a = a.generate(prompt, &cfg).expect("gen");
    println!("{prompt}{out_a}");
    drop(a);

    println!();
    println!("--- OVERRIDE (zero l_out-26) ---");
    let handler: Box<dyn ProbeHandler> = Box::new(ZeroOutL26 { hits: 0 });
    let mut b = LlamaPipeline::load_with_probe(&gguf, 1024, handler).expect("load");
    let out_b = b.generate(prompt, &cfg).expect("gen");
    println!("{prompt}{out_b}");

    println!();
    if out_a != out_b {
        println!("VERDICT: probe override changes output. ✓");
    } else {
        println!("VERDICT: outputs identical — probe did not take effect.");
    }
}
