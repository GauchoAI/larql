//! Microbench: 3 separate matmul_transb calls vs 1 call on stacked QKV weights.
//!
//! Gemma 3 4B attention shapes at decode (m=1):
//!   Q: [1, 2560] × [2048, 2560]^T
//!   K: [1, 2560] × [1024, 2560]^T
//!   V: [1, 2560] × [1024, 2560]^T
//!
//! Stacked: [1, 2560] × [4096, 2560]^T → split into q/k/v along axis 1.
//!
//! If stacked is meaningfully faster at m=1, we ship P10 (one BLAS call per
//! layer's QKV block). If not, we drop P10 and look elsewhere.

extern crate blas_src;

use std::time::Instant;
use ndarray::{Array2, s};
use larql_compute::{CpuBackend, ComputeBackend};

fn rand_mat(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32 / u32::MAX as f32) * 2.0 - 1.0
    })
}

fn main() {
    let hidden = 2560;
    let q_dim = 2048;
    let kv_dim = 1024;
    let cpu = CpuBackend;

    let h = rand_mat(1, hidden, 1);
    let wq = rand_mat(q_dim, hidden, 2);
    let wk = rand_mat(kv_dim, hidden, 3);
    let wv = rand_mat(kv_dim, hidden, 4);
    let mut w_stacked = Array2::<f32>::zeros((q_dim + 2 * kv_dim, hidden));
    w_stacked.slice_mut(s![0..q_dim, ..]).assign(&wq);
    w_stacked.slice_mut(s![q_dim..q_dim + kv_dim, ..]).assign(&wk);
    w_stacked.slice_mut(s![q_dim + kv_dim.., ..]).assign(&wv);

    let iters = 1000;

    // Warmup
    for _ in 0..200 {
        let _ = cpu.matmul_transb(h.view(), wq.view());
        let _ = cpu.matmul_transb(h.view(), wk.view());
        let _ = cpu.matmul_transb(h.view(), wv.view());
        let _ = cpu.matmul_transb(h.view(), w_stacked.view());
    }

    // 3-call path
    let mut total_3 = std::time::Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        let q = cpu.matmul_transb(h.view(), wq.view());
        let k = cpu.matmul_transb(h.view(), wk.view());
        let v = cpu.matmul_transb(h.view(), wv.view());
        total_3 += t.elapsed();
        std::hint::black_box((q, k, v));
    }

    // 1-call stacked
    let mut total_1 = std::time::Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        let qkv = cpu.matmul_transb(h.view(), w_stacked.view());
        let q = qkv.slice(s![.., 0..q_dim]).to_owned();
        let k = qkv.slice(s![.., q_dim..q_dim + kv_dim]).to_owned();
        let v = qkv.slice(s![.., q_dim + kv_dim..]).to_owned();
        total_1 += t.elapsed();
        std::hint::black_box((q, k, v));
    }

    // 1-call stacked WITHOUT the owned-copy (pretend we use views)
    let mut total_1_view = std::time::Duration::ZERO;
    for _ in 0..iters {
        let t = Instant::now();
        let qkv = cpu.matmul_transb(h.view(), w_stacked.view());
        let q = qkv.slice(s![.., 0..q_dim]);
        let k = qkv.slice(s![.., q_dim..q_dim + kv_dim]);
        let v = qkv.slice(s![.., q_dim + kv_dim..]);
        total_1_view += t.elapsed();
        std::hint::black_box((q.to_owned(), k.to_owned(), v.to_owned())); // forces use
    }

    let us_3 = total_3.as_secs_f64() * 1e6 / iters as f64;
    let us_1 = total_1.as_secs_f64() * 1e6 / iters as f64;
    let us_1v = total_1_view.as_secs_f64() * 1e6 / iters as f64;

    println!("QKV projection microbench (Gemma 3 4B decode shapes)");
    println!("  h: [1, {hidden}], q: [{q_dim}, {hidden}], k,v: [{kv_dim}, {hidden}]");
    println!("  stacked: [{}, {hidden}]", q_dim + 2 * kv_dim);
    println!();
    println!("  3 separate matmul_transb calls : {:.2} µs/iter", us_3);
    println!("  1 stacked (+ owned split)      : {:.2} µs/iter", us_1);
    println!("  1 stacked (+ view split)       : {:.2} µs/iter", us_1v);
    println!();
    println!("  Savings 3→1 (owned): {:.2} µs = {:.1} %",
        us_3 - us_1, 100.0 * (us_3 - us_1) / us_3);
    println!("  Savings 3→1 (view):  {:.2} µs = {:.1} %",
        us_3 - us_1v, 100.0 * (us_3 - us_1v) / us_3);
    println!();
    println!("  Per-token projection (× 34 layers):");
    println!("    3-call total   : {:.2} ms/tok", us_3 * 34.0 / 1000.0);
    println!("    stacked total  : {:.2} ms/tok", us_1v * 34.0 / 1000.0);
    println!("    saving         : {:.2} ms/tok", (us_3 - us_1v) * 34.0 / 1000.0);
    println!();
    let decode_ms = 60.0; // approx post-S1
    println!("  At decode ≈ {decode_ms} ms/tok, projected speedup: {:.1} %",
        100.0 * (us_3 - us_1v) * 34.0 / 1000.0 / decode_ms);
}
