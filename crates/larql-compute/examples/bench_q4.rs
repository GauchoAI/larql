//! Three-way Q4 benchmark: BLAS f32 vs C Q4 kernel vs Metal Q4 shader.
//!
//! Usage:
//!   cargo run --release -p larql-compute --example bench_q4
//!   cargo run --release -p larql-compute --features metal --example bench_q4

extern crate blas_src;

use std::time::Instant;
use ndarray::Array2;
use larql_compute::{ComputeBackend, default_backend, cpu_backend};
use larql_compute::cpu::q4;

fn main() {
    let hidden = 2560;
    let intermediate = 10240;
    let n = 20;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..intermediate * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu = cpu_backend();
    let default = default_backend();

    println!("=== Q4 Benchmark ===");
    println!("Matrix: [{intermediate}, {hidden}] = {:.1}MB f32 → {:.1}MB Q4_0",
        (intermediate * hidden * 4) as f64 / 1e6, q4_data.len() as f64 / 1e6);
    println!("CPU: {}", cpu.name());
    println!("Default: {}\n", default.name());

    // 1. BLAS f32 gemv
    {
        let mat = ndarray::ArrayView2::from_shape((intermediate, hidden), &matrix).unwrap();
        let xv = ndarray::Array1::from_vec(x.clone());
        let _ = mat.dot(&xv);
        let t0 = Instant::now();
        for _ in 0..n { let _ = mat.dot(&xv); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = (intermediate * hidden * 4) as f64 / ms / 1e6;
        println!("  BLAS f32 gemv:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            (intermediate * hidden * 4) as f64 / 1e6);
    }

    // 2. C Q4 kernel (via CPU backend)
    {
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
        let _ = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  CPU Q4 kernel:     {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            q4_data.len() as f64 / 1e6);
    }

    // 3. Default backend Q4 (Metal if available)
    if default.has_q4() && default.name() != cpu.name() {
        let (q8_x, q8_scales) = q4::quantize_to_q8(&x);
        let _ = default.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden);
        let t0 = Instant::now();
        for _ in 0..n { let _ = default.q4_matvec(&q4_data, &q8_x, &q8_scales, intermediate, hidden); }
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / n as f64;
        let gbps = q4_data.len() as f64 / ms / 1e6;
        println!("  {} Q4:  {ms:>6.2}ms  ({gbps:>5.1} GB/s on {:.1}MB)",
            default.name(), q4_data.len() as f64 / 1e6);
    }

    println!("\n=== Done ===");
}

fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0);
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
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
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}
