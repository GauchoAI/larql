//! Per-operation standalone benchmarks — CPU and Metal side by side.
//!
//! Every operation benchmarked individually at representative sizes.
//! Run with:
//!   cargo run --release -p larql-compute --example bench_shaders                  # CPU only
//!   cargo run --release -p larql-compute --features metal --example bench_shaders # CPU + Metal

extern crate blas_src;

use std::time::Instant;
use larql_compute::cpu::q4;

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
        println!("  {name:50} {ms:>7.3}ms");
        ms
    }
}

fn main() {
    let t = Timer { n: 20 };
    let hidden = 2560;
    let inter = 10240;

    let cpu = larql_compute::cpu_backend();

    println!("=== Per-Operation Benchmarks (CPU + Metal) ===\n");

    // ── sgemm ──
    println!("--- f32 matmul (C = A × B) ---");
    {
        let a = ndarray::Array2::from_shape_fn((6, hidden), |_| 0.01f32);
        let b = ndarray::Array2::from_shape_fn((hidden, hidden), |_| 0.01f32);
        t.run("CPU BLAS [6,2560] × [2560,2560]", || { let _ = cpu.matmul(a.view(), b.view()); });
    }

    // ── sgemm_transb ──
    println!("\n--- f32 matmul_transb (C = A × B^T) ---");
    {
        let a = ndarray::Array2::from_shape_fn((6, hidden), |_| 0.01f32);
        let b = ndarray::Array2::from_shape_fn((inter, hidden), |_| 0.01f32);
        t.run("CPU BLAS [6,2560] × [10240,2560]^T", || { let _ = cpu.matmul_transb(a.view(), b.view()); });
    }

    // ── q4_matvec (CPU) ──
    println!("\n--- Q4 matvec (CPU C kernel) ---");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        t.run("CPU C kernel [10240,2560] × x[2560]", || {
            let _ = larql_compute::cpu::ops::q4_matvec::dispatch(&q4_data, &x, inter, hidden);
        });
    }

    // ── q4_vecmat (CPU) ──
    println!("\n--- Q4 vecmat (CPU C kernel) ---");
    {
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
        let q4_data = quantize_q4_0(&matrix);
        let act: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
        t.run("CPU C kernel act[10240] × Q4[10240,2560]", || {
            let _ = larql_compute::cpu::ops::q4_vecmat::dispatch(&act, &q4_data, inter, hidden);
        });
    }

    // ── geglu (CPU) ──
    println!("\n--- GEGLU (CPU) ---");
    {
        let gate: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
        let up: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.002).cos()).collect();
        t.run("CPU geglu silu (10240 elements)", || {
            let _ = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);
        });
    }

    // ── attention (CPU) ──
    println!("\n--- Causal attention (CPU) ---");
    {
        let dim = 320;
        let seq = 6;
        let q = vec![0.01f32; seq * dim];
        let k = vec![0.01f32; seq * dim];
        let v = vec![0.01f32; seq * dim];
        t.run("CPU causal attention (seq=6, dim=320)", || {
            let _ = larql_compute::cpu::ops::attention::causal_attention(&q, &k, &v, seq, dim, 1.0 / (dim as f32).sqrt());
        });
        let q1 = vec![0.01f32; dim];
        let k1 = vec![0.01f32; dim];
        let v1 = vec![0.01f32; dim];
        t.run("CPU causal attention (seq=1, dim=320)", || {
            let _ = larql_compute::cpu::ops::attention::causal_attention(&q1, &k1, &v1, 1, dim, 1.0 / (dim as f32).sqrt());
        });
    }

    // ── Q8 quantize (CPU) ──
    println!("\n--- Q8 quantize (CPU) ---");
    {
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
        t.run("CPU quantize_to_q8 (2560 elements)", || {
            let _ = q4::quantize_to_q8(&x);
        });
    }

    // ── Metal shaders ──
    #[cfg(feature = "metal")]
    {
        use larql_compute::ComputeBackend;

        let metal = match larql_compute::metal::MetalBackend::new() {
            Some(m) => m,
            None => { println!("\nMetal not available"); return; }
        };

        println!("\n--- Metal: f32 matmul ---");
        {
            let a = ndarray::Array2::from_shape_fn((6, hidden), |_| 0.01f32);
            let b = ndarray::Array2::from_shape_fn((hidden, hidden), |_| 0.01f32);
            t.run("Metal [6,2560] × [2560,2560]", || { let _ = metal.matmul(a.view(), b.view()); });
        }

        println!("\n--- Metal: f32 matmul_transb ---");
        {
            let a = ndarray::Array2::from_shape_fn((6, hidden), |_| 0.01f32);
            let b = ndarray::Array2::from_shape_fn((inter, hidden), |_| 0.01f32);
            t.run("Metal [6,2560] × [10240,2560]^T", || { let _ = metal.matmul_transb(a.view(), b.view()); });
        }

        // ── q4_matvec ──
        println!("\n--- q4_matvec (Q4×Q8, simdgroup optimised) ---");
        {
            let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let q4 = quantize_q4_0(&matrix);
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            let (q8, sc) = q4::quantize_to_q8(&x);
            t.run("Metal [10240,2560] × Q8[2560]", || {
                let _ = metal.q4_matvec_direct(&q4, &q8, &sc, inter, hidden);
            });
        }

        // ── q4_vecmat ──
        println!("\n--- q4_vecmat (scatter-accumulate) ---");
        {
            let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let q4 = quantize_q4_0(&matrix);
            let act: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { 1.0 } else { 0.0 }).collect();
            t.run("Metal act[10240] × Q4[10240,2560]", || {
                let _ = metal.q4_vecmat_direct(&act, &q4, inter, hidden);
            });
        }

        // ── q4_f32_matvec ──
        println!("\n--- q4_f32_matvec (transposed down) ---");
        {
            let matrix: Vec<f32> = (0..hidden * inter).map(|i| (i as f32 * 0.0001).cos()).collect();
            let q4 = quantize_q4_0(&matrix);
            let act: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
            t.run("Metal Q4[2560,10240] × f32[10240]", || {
                let _ = metal.q4_f32_matvec_direct(&q4, &act, hidden, inter);
            });
        }

        // ── geglu ──
        println!("\n--- geglu_silu (element-wise) ---");
        {
            // GEGLU is inside the multi-layer pipeline, not directly exposed.
            // Benchmark via a single-layer multi_layer_ffn minus the gate/up/down cost.
            let gate: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
            let up: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.002).cos()).collect();
            // CPU reference for geglu timing
            t.run("CPU geglu silu (10240 elements)", || {
                let mut out = vec![0.0f32; inter];
                for i in 0..inter {
                    let g = gate[i];
                    out[i] = (g / (1.0 + (-g).exp())) * up[i];
                }
                std::hint::black_box(&out);
            });
            println!("  (Metal geglu runs inside multi-layer pipeline, not standalone)");
        }

        // ── quantize_q8 ──
        println!("\n--- quantize_q8 (f32 → Q8) ---");
        {
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            t.run("CPU quantize_to_q8 (2560 elements)", || {
                let _ = q4::quantize_to_q8(&x);
            });
            println!("  (Metal Q8 quantize runs inside multi-layer pipeline)");
        }

        // ── causal_attention ──
        println!("\n--- causal_attention (basic, seq=6) ---");
        {
            let head_dim = 320;
            let seq = 6;
            // Benchmark via full_layer which includes attention
            let wq: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let wk: Vec<f32> = (0..512 * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
            let wv: Vec<f32> = (0..512 * hidden).map(|i| (i as f32 * 0.0003).cos()).collect();
            let wo: Vec<f32> = (0..hidden * hidden).map(|i| (i as f32 * 0.0004).sin()).collect();
            let gq4 = quantize_q4_0(&(0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect::<Vec<_>>());
            let uq4 = quantize_q4_0(&(0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect::<Vec<_>>());
            let dq4 = quantize_q4_0(&(0..hidden * inter).map(|i| (i as f32 * 0.0003).cos()).collect::<Vec<_>>());
            let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.001).sin()).collect();

            t.run("Metal full_layer (attn+FFN, seq=6)", || {
                let _ = metal.full_layer_direct(
                    &wq, &wk, &wv, &wo, &gq4, &uq4, &dq4,
                    &x, seq, hidden, 8, 4, head_dim, inter, 1.0 / (head_dim as f32).sqrt(),
                );
            });
            t.run("Metal full_layer (attn+FFN, seq=1)", || {
                let _ = metal.full_layer_direct(
                    &wq, &wk, &wv, &wo, &gq4, &uq4, &dq4,
                    &x[..hidden], 1, hidden, 8, 4, head_dim, inter, 1.0 / (head_dim as f32).sqrt(),
                );
            });
        }

        // ── pair_batch ──
        println!("\n--- pair_batch (gate+up × 6 positions) ---");
        {
            let gf: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
            let uf: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
            let gq4 = quantize_q4_0(&gf);
            let uq4 = quantize_q4_0(&uf);
            let x: Vec<f32> = (0..6 * hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            t.run("Metal pair_batch (6 pos)", || {
                let _ = metal.q4_matvec_pair_batch_direct(&gq4, &uq4, &x, 6, inter, hidden);
            });
        }

        // ── multi_layer_ffn ──
        println!("\n--- multi_layer_ffn (21 layers, 1 cmd buffer) ---");
        {
            let mut layers = Vec::new();
            for l in 0..21u64 {
                let g: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 1e7) * 0.0001).cos() as f32).collect();
                let u: Vec<f32> = (0..inter * hidden).map(|i| ((i as f64 + l as f64 * 2e7) * 0.0002).sin() as f32).collect();
                let mut dt = vec![0.0f32; hidden * inter];
                for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c) as f64 * 0.0003).cos() as f32; } }
                layers.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
            }
            let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
            t.run("Metal 21-layer Q4 FFN (1 cmd buffer)", || {
                let _ = metal.multi_layer_q4_ffn(&layers, &x, inter, hidden);
            });
        }
    }

    #[cfg(not(feature = "metal"))]
    println!("Metal not enabled. Run with --features metal");

    println!("\n=== Done ===");
}
