//! Token generation benchmarks: simulates actual decode-time inference.
//!
//! Tests the production case: seq=1 per token with KV cache,
//! vs seq=6 without cache. Shows the multiplier from KV caching.
//!
//! Usage:
//!   cargo run --release -p larql-compute --features metal --example bench_generation

extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;
use larql_compute::{ComputeBackend, default_backend, cpu_backend};
use larql_compute::cpu::q4;

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
}

fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0);
    let n = data.len() / 32;
    let mut out = Vec::with_capacity(n * 18);
    for i in 0..n {
        let blk = &data[i * 32..(i + 1) * 32];
        let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 { sign as u16 }
            else if exp >= 31 + 127 - 15 { (sign | 0x7C00) as u16 }
            else if exp <= -15 + 127 { sign as u16 }
            else { (sign | (((exp - 127 + 15) as u32) << 10) | (mant >> 13)) as u16 };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((blk[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((blk[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

struct Timer { n: usize }
impl Timer {
    fn run<F: FnMut()>(&self, name: &str, mut f: F) -> f64 {
        f();
        let t0 = Instant::now();
        for _ in 0..self.n { f(); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / self.n as f64;
        let tps = 1000.0 / ms;
        println!("  {name:55} {ms:>7.2}ms  ({tps:>5.1} tok/s)");
        ms
    }
}

fn main() {
    let hidden = 2560;
    let inter = 10240;
    let head_dim = 320;
    let num_q = 8;
    let num_kv = 4;
    let kv_dim = num_kv * head_dim;
    let cpu = cpu_backend();
    let t = Timer { n: 10 };

    println!("=== Token Generation Benchmarks ===");
    println!("Simulating decode: seq=1 per token (KV cached)\n");

    // Build 21 layers of Q4 data
    let mut layers_q4: Vec<(Vec<u8>, Vec<u8>, Vec<u8>)> = Vec::new();
    for l in 0..21u64 {
        let g: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 1e7) * 0.0001).cos() as f32).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 2e7) * 0.0002).sin() as f32).collect();
        let d: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 3e7) * 0.0003).cos() as f32).collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = d[r * hidden + c]; } }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    // Build attention weights for 21 layers
    let attn_wq: Vec<Vec<f32>> = (0..21).map(|l| (0..hidden * hidden).map(|i| ((i + l * 1000) as f32 * 0.0001).cos()).collect()).collect();
    let attn_wk: Vec<Vec<f32>> = (0..21).map(|l| (0..kv_dim * hidden).map(|i| ((i + l * 2000) as f32 * 0.0002).sin()).collect()).collect();
    let attn_wv: Vec<Vec<f32>> = (0..21).map(|l| (0..kv_dim * hidden).map(|i| ((i + l * 3000) as f32 * 0.0003).cos()).collect()).collect();
    let attn_wo: Vec<Vec<f32>> = (0..21).map(|l| (0..hidden * hidden).map(|i| ((i + l * 4000) as f32 * 0.0004).sin()).collect()).collect();

    // ── 1. Prefill (seq=6, no KV cache) ──
    println!("--- 1. Prefill: seq=6, 21 layers (no KV cache) ---\n");

    t.run("CPU f32 prefill (seq=6, 4 attn proj × 21 layers)", || {
        let h = synth(6, hidden, 42);
        for l in 0..21 {
            let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
            let _ = cpu.matmul_transb(h.view(), wq.view());
            let _ = cpu.matmul_transb(h.view(), wq.view());
            let _ = cpu.matmul_transb(h.view(), wq.view());
            let _ = cpu.matmul_transb(h.view(), wq.view());
        }
    });

    // ── 2. Decode: seq=1 with KV cache (CPU) ──
    println!("\n--- 2. Decode: seq=1 per token, 21 layers (KV cached) ---\n");

    // CPU Q4 decode (seq=1)
    t.run("CPU C Q4 decode (seq=1, FFN only, 21 layers)", || {
        let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
            let g = q4::q4_matvec(gate_q4, &h, inter, hidden);
            let u = q4::q4_matvec(up_q4, &h, inter, hidden);
            let mut act = vec![0.0f32; inter];
            for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
            h = q4::q4_matvec(down_t_q4, &act, hidden, inter);
        }
    });

    // CPU f32 BLAS decode (seq=1, attention only — 4 projections)
    t.run("CPU f32 decode (seq=1, attn 4 proj only, 21 layers)", || {
        let h = synth(1, hidden, 42);
        for l in 0..21 {
            let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
            let wk = Array2::from_shape_vec((kv_dim, hidden), attn_wk[l].clone()).unwrap();
            let wv = Array2::from_shape_vec((kv_dim, hidden), attn_wv[l].clone()).unwrap();
            let wo = Array2::from_shape_vec((hidden, hidden), attn_wo[l].clone()).unwrap();
            let _ = cpu.matmul_transb(h.view(), wq.view());
            let _ = cpu.matmul_transb(h.view(), wk.view());
            let _ = cpu.matmul_transb(h.view(), wv.view());
            // O proj after attention: [1, hidden] @ [hidden, hidden]^T
            let _ = cpu.matmul_transb(h.view(), wo.view());
        }
    });

    // CPU full decode (seq=1, attn + FFN)
    t.run("CPU full decode (seq=1, attn + Q4 FFN, 21 layers)", || {
        let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        for l in 0..21 {
            // Attention: 4 projections (simulate)
            let h_arr = Array2::from_shape_vec((1, hidden), h.clone()).unwrap();
            let wq = Array2::from_shape_vec((hidden, hidden), attn_wq[l].clone()).unwrap();
            let _ = cpu.matmul_transb(h_arr.view(), wq.view());
            let _ = cpu.matmul_transb(h_arr.view(), wq.view()); // K reuses wq for simplicity
            let _ = cpu.matmul_transb(h_arr.view(), wq.view()); // V
            let _ = cpu.matmul_transb(h_arr.view(), wq.view()); // O
            // FFN: Q4
            let (gate_q4, up_q4, down_t_q4) = &layers_q4[l];
            let g = q4::q4_matvec(gate_q4, &h, inter, hidden);
            let u = q4::q4_matvec(up_q4, &h, inter, hidden);
            let mut act = vec![0.0f32; inter];
            for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
            h = q4::q4_matvec(down_t_q4, &act, hidden, inter);
        }
    });

    // ── 3. Metal decode (seq=1) ──
    println!("\n--- 3. Metal decode: seq=1, 21 layers ---\n");
    #[cfg(feature = "metal")]
    {
        if let Some(ref metal) = larql_compute::metal::MetalBackend::new() {
            // Metal full layer at seq=1
            t.run("Metal full layer (seq=1, 21 layers, 1 cmd/layer)", || {
                let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
                // We can't easily chain layers without reading back, so benchmark one layer
                // and multiply
                for l in 0..21 {
                    let (gate_q4, up_q4, down_t_q4) = &layers_q4[l];
                    let _ = metal.full_layer_direct(
                        &attn_wq[l], &attn_wk[l], &attn_wv[l], &attn_wo[l],
                        gate_q4, up_q4, down_t_q4,
                        &x, 1, hidden, num_q, num_kv, head_dim, inter,
                        1.0 / (head_dim as f32).sqrt(),
                    );
                }
            });

            // Metal Q4 FFN only at seq=1
            t.run("Metal Q4 FFN only (seq=1, 21 layers)", || {
                let mut h: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
                for (gate_q4, up_q4, down_t_q4) in &layers_q4 {
                    let (q8, sc) = q4::quantize_to_q8(&h);
                    let g = metal.q4_matvec_direct(gate_q4, &q8, &sc, inter, hidden);
                    let u = metal.q4_matvec_direct(up_q4, &q8, &sc, inter, hidden);
                    let mut act = vec![0.0f32; inter];
                    for i in 0..inter { act[i] = (g[i] / (1.0 + (-g[i]).exp())) * u[i]; }
                    h = metal.q4_f32_matvec_direct(down_t_q4, &act, hidden, inter);
                }
            });
        }
    }
    #[cfg(not(feature = "metal"))]
    println!("  (Metal not enabled)");

    // ── 4. Comparison summary ──
    println!("\n--- 4. Summary ---\n");
    println!("  Ollama (Q4 Metal, KV cache):     ~10ms/token → ~100 tok/s");
    println!("  LARQL target with Metal + KV:     ~25ms/token → ~40 tok/s");
    println!("  LARQL current (f32, no KV):      ~220ms/token → ~4.5 tok/s");

    println!("\n=== Done ===");
}
