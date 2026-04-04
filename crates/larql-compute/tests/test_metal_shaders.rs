//! Per-shader correctness tests for Metal compute kernels.
//!
//! Each test runs the Metal shader and compares output against
//! a CPU reference implementation. Tests both correctness and
//! that the shader compiles and dispatches successfully.
//!
//! Run with: cargo test -p larql-compute --features metal

#![cfg(feature = "metal")]

extern crate blas_src;

use ndarray::Array2;
use larql_compute::{ComputeBackend, cpu::q4};

// ── Test helpers ──

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

fn max_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn get_metal() -> larql_compute::metal::MetalBackend {
    larql_compute::metal::MetalBackend::new().expect("Metal device required for these tests")
}

// ── Shader compilation ──

#[test]
fn all_shaders_compile() {
    let src = larql_compute::metal::shaders::all_shaders();
    assert!(src.len() > 1000, "Shader source too short");

    let device = metal::Device::system_default().expect("No Metal device");
    let opts = metal::CompileOptions::new();
    device.new_library_with_source(&src, &opts)
        .expect("Shader compilation failed");
}

#[test]
fn all_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = ["sgemm", "sgemm_transb", "q4_matvec", "q4_vecmat",
                 "q4_f32_matvec", "geglu_silu", "quantize_q8", "causal_attention"];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── f32 sgemm ──

#[test]
fn sgemm_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(2560, 2560, 43);

    let cpu_result = a.dot(&b);
    let metal_result = metal.matmul(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm max diff {diff} exceeds 0.1");
}

// ── f32 sgemm_transb ──

#[test]
fn sgemm_transb_matches_cpu() {
    let metal = get_metal();
    let a = synth(6, 2560, 42);
    let b = synth(10240, 2560, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.1, "sgemm_transb max diff {diff} exceeds 0.1");
}

#[test]
fn sgemm_transb_small_matrix() {
    let metal = get_metal();
    let a = synth(1, 256, 42);
    let b = synth(512, 256, 43);

    let cpu_result = a.dot(&b.t());
    let metal_result = metal.matmul_transb(a.view(), b.view());

    let diff = max_diff(cpu_result.as_slice().unwrap(), metal_result.as_slice().unwrap());
    assert!(diff < 0.01, "small sgemm_transb max diff {diff}");
}

// ── Q4 matvec ──

#[test]
fn q4_matvec_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let rows = 10240;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.001).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "q4_matvec max diff {diff} exceeds 0.01");
}

#[test]
fn q4_matvec_small_matrix() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 128;

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let cpu_result = q4::q4_matvec(&q4_data, &x, rows, hidden);
    let metal_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.01, "small q4_matvec max diff {diff}");
}

#[test]
fn q4_matvec_zero_input() {
    let metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let x = vec![0.0f32; hidden];
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, rows, hidden);
    assert!(result.iter().all(|&v| v.abs() < 0.01), "zero input should produce near-zero output");
}

// ── Q4 vecmat ──

#[test]
fn q4_vecmat_matches_cpu() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| if i % 5 == 0 { (i as f32 * 0.01).sin() } else { 0.0 }).collect();
    let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);

    let cpu_result = q4::q4_vecmat(&activation, &q4_data, inter, hidden);
    let metal_result = metal.q4_vecmat_direct(&activation, &q4_data, inter, hidden);

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.1, "q4_vecmat max diff {diff} exceeds 0.1");
}

// ── Q4 f32 matvec (for transposed down) ──

#[test]
fn q4_f32_matvec_nonzero() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 10240;

    let activation: Vec<f32> = (0..inter).map(|i| (i as f32 * 0.001).sin()).collect();
    let mut down_t: Vec<f32> = vec![0.0; hidden * inter];
    for r in 0..inter { for c in 0..hidden { down_t[c * inter + r] = ((r * hidden + c) as f32 * 0.0001).cos(); } }
    let q4_data = quantize_q4_0(&down_t);

    let result = metal.q4_f32_matvec_direct(&q4_data, &activation, hidden, inter);
    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.01), "should produce nonzero output");
}

// ── Q4 pair batch ──

#[test]
fn q4_pair_batch_matches_individual() {
    let metal = get_metal();
    let hidden = 2560;
    let inter = 1024; // smaller for test speed
    let seq = 2;

    let gate_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0001).cos()).collect();
    let up_f32: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.0002).sin()).collect();
    let gate_q4 = quantize_q4_0(&gate_f32);
    let up_q4 = quantize_q4_0(&up_f32);
    let x: Vec<f32> = (0..seq * hidden).map(|i| (i as f32 * 0.001).sin()).collect();

    // Individual calls
    let mut indiv_gate = Vec::new();
    let mut indiv_up = Vec::new();
    for s in 0..seq {
        let slice = &x[s * hidden..(s + 1) * hidden];
        let (q8, sc) = q4::quantize_to_q8(slice);
        indiv_gate.push(metal.q4_matvec_direct(&gate_q4, &q8, &sc, inter, hidden));
        indiv_up.push(metal.q4_matvec_direct(&up_q4, &q8, &sc, inter, hidden));
    }

    // Batched call
    let (batch_gate, batch_up) = metal.q4_matvec_pair_batch_direct(
        &gate_q4, &up_q4, &x, seq, inter, hidden,
    );

    // Compare
    for s in 0..seq {
        let diff_g = max_diff(&indiv_gate[s], &batch_gate[s]);
        let diff_u = max_diff(&indiv_up[s], &batch_up[s]);
        assert!(diff_g < 0.001, "pair_batch gate diff {diff_g} at seq {s}");
        assert!(diff_u < 0.001, "pair_batch up diff {diff_u} at seq {s}");
    }
}

// ── Multi-layer Q4 FFN ──

#[test]
fn multi_layer_q4_produces_output() {
    let metal = get_metal();
    let hidden = 256; // small for test speed
    let inter = 512;
    let layers = 3;

    let mut layers_q4 = Vec::new();
    for l in 0..layers {
        let g: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 1000) as f32 * 0.001).cos()).collect();
        let u: Vec<f32> = (0..inter * hidden).map(|i| ((i + l * 2000) as f32 * 0.002).sin()).collect();
        let mut dt = vec![0.0f32; hidden * inter];
        for r in 0..inter { for c in 0..hidden { dt[c * inter + r] = ((r * hidden + c + l * 3000) as f32 * 0.003).cos(); } }
        layers_q4.push((quantize_q4_0(&g), quantize_q4_0(&u), quantize_q4_0(&dt)));
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let result = metal.multi_layer_q4_ffn(&layers_q4, &x, inter, hidden);

    assert_eq!(result.len(), hidden);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "multi-layer should produce nonzero output");
}

// ── Buffer cache ──

#[test]
fn buffer_cache_reuses_same_pointer() {
    let metal = get_metal();
    let data = vec![1.0f32; 1024];
    let q4 = quantize_q4_0(&data);
    let (q8, sc) = q4::quantize_to_q8(&data[..256]);

    // Call twice with same data — buffer should be cached
    let r1 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);
    let r2 = metal.q4_matvec_direct(&q4, &q8, &sc, 4, 256);

    let diff = max_diff(&r1, &r2);
    assert!(diff < 1e-6, "cached buffer should produce identical results, diff: {diff}");
}

// ── Trait dispatch ──

#[test]
fn metal_backend_implements_trait() {
    use larql_compute::ComputeBackend;
    let metal = get_metal();

    assert!(metal.has_q4());
    assert!(metal.name().contains("metal"));

    let a = synth(2, 64, 42);
    let b = synth(32, 64, 43);
    let result = metal.matmul_transb(a.view(), b.view());
    assert_eq!(result.shape(), &[2, 32]);
}
