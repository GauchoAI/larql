//! Correctness tests: verify all backends produce matching output.

extern crate blas_src;

use ndarray::Array2;
use larql_compute::{ComputeBackend, cpu_backend};

fn synth_matrix(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut state = seed;
    let data: Vec<f32> = (0..rows * cols)
        .map(|_| {
            state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
        })
        .collect();
    Array2::from_shape_vec((rows, cols), data).unwrap()
}

fn max_diff(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn cpu_matmul_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(2560, 2560, 43);
    let expected = a.dot(&b);
    let result = cpu.matmul(a.view(), b.view());
    assert!(max_diff(&expected, &result) < 1e-5, "matmul mismatch");
}

#[test]
fn cpu_matmul_transb_matches_ndarray() {
    let cpu = cpu_backend();
    let a = synth_matrix(6, 2560, 42);
    let b = synth_matrix(10240, 2560, 43);
    let expected = a.dot(&b.t());
    let result = cpu.matmul_transb(a.view(), b.view());
    assert!(max_diff(&expected, &result) < 1e-5, "matmul_transb mismatch");
}

#[test]
fn cpu_has_q4() {
    let cpu = cpu_backend();
    assert!(cpu.has_q4(), "CPU backend should support Q4");
}

#[test]
fn cpu_q4_matvec_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256; // small for test speed
    let rows = 128;
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();

    // Quantize matrix to Q4
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu = cpu_backend();
    let result = cpu.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "Q4 matvec should produce nonzero output");
}

#[test]
fn cpu_q4_vecmat_nonzero() {
    use larql_compute::cpu::q4;

    let hidden = 256;
    let inter = 128;
    let activation: Vec<f32> = (0..inter).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
    let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "Q4 vecmat should produce nonzero output");
}

#[test]
fn default_backend_has_name() {
    let be = larql_compute::default_backend();
    assert!(!be.name().is_empty());
}

// ── Helper: Q4 quantization for tests ──

fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len() % 32 == 0);
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        // f16 encode scale
        let scale_f16 = f32_to_f16(scale);
        out.extend_from_slice(&scale_f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

fn f32_to_f16(v: f32) -> u16 {
    let bits = v.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    if exp == 0 { return sign as u16; }
    if exp == 255 { return (sign | 0x7C00 | (mant >> 13) as u32) as u16; }
    let new_exp = exp - 127 + 15;
    if new_exp >= 31 { return (sign | 0x7C00) as u16; }
    if new_exp <= 0 { return sign as u16; }
    (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16
}
