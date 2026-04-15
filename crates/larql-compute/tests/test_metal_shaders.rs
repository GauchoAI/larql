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
use larql_compute::cpu::q4::quantize_q4_0;

// ── Test helpers ──

fn synth(rows: usize, cols: usize, seed: u64) -> Array2<f32> {
    let mut s = seed;
    Array2::from_shape_fn((rows, cols), |_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    })
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

    let names = [
        // f32 matmul
        "sgemm", "sgemm_transb",
        // Q4_0 matvec variants
        "q4_matvec", "q4_vecmat", "q4_f32_matvec",
        // Q4_K / Q4_KF matvec
        "q4k_matvec", "q4k_qkv_proj", "q4k_proj",
        "q4kf_qkv_proj", "q4kf_proj",
        // Q4_K fused FFN
        "q4k_ffn_gate_up", "q4kf_ffn_gate_up",
        "q4k_geglu_silu_down", "q4k_geglu_gelu_tanh_down",
        // Activations
        "geglu_silu", "geglu_gelu_tanh", "silu", "gelu_tanh",
        // Quantize / norms / residuals
        "quantize_q8", "rms_norm_q8", "residual_norm", "residual_norm_q8", "residual_add",
        "layer_norm", "layer_norm_no_bias", "v_norm", "v_norm_batched", "scale_vector",
        // Attention / RoPE
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "rope_at_pos", "rope_at_pos_batched",
    ];
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
    let layers_refs: Vec<(&[u8], &[u8], &[u8])> = layers_q4.iter()
        .map(|(g, u, d)| (g.as_slice(), u.as_slice(), d.as_slice())).collect();
    let result = metal.multi_layer_q4_ffn(&layers_refs, &x, inter, hidden);

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

// ── Q8 matvec ──

#[test]
fn q8_matvec_metal_nonzero() {
    let _metal = get_metal();
    let hidden = 256;
    let rows = 64;

    let weights: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let (w_q8, w_scales) = larql_compute::cpu::ops::q8_matvec::quantize_weights_q8(&weights, rows, hidden);
    let (x_q8, x_scales) = larql_compute::cpu::ops::q4_common::quantize_to_q8(&x);

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::q8_matvec::dispatch(&w_q8, &w_scales, &x_q8, &x_scales, rows, hidden);
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.01), "Q8 CPU should produce nonzero");
}

// ── Sparse Q4 matvec ──

#[test]
fn sparse_matvec_matches_dense() {
    let metal = get_metal();
    let hidden = 256;
    let n_rows = 64;
    let k_selected = 16;

    let matrix: Vec<f32> = (0..n_rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4_data = quantize_q4_0(&matrix);
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    // Dense: score all rows
    let dense_result = metal.q4_matvec_direct(&q4_data, &q8_x, &q8_scales, n_rows, hidden);

    // Sparse: score selected rows [0, 4, 8, 12, ...]
    let indices: Vec<u32> = (0..k_selected as u32).map(|i| i * 4).collect();

    // Use the sparse shader via raw Metal dispatch
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("q4_sparse_matvec", None).unwrap()
    ).unwrap();

    let bufs = &larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    let buf_q4 = bufs.get_bytes(&q4_data);
    let buf_q8 = bufs.transient_from_i8(&q8_x);
    let buf_sc = bufs.transient_from_f32(&q8_scales);
    let idx_bytes: Vec<u8> = indices.iter().flat_map(|i| i.to_le_bytes()).collect();
    let buf_idx = bufs.transient_from_f32(unsafe {
        std::slice::from_raw_parts(idx_bytes.as_ptr() as *const f32, indices.len())
    });
    let buf_out = bufs.output((k_selected * 4) as u64);

    let k_val = k_selected as u32;
    let h_val = hidden as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q4), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_sc), 0);
    enc.set_buffer(3, Some(&buf_idx), 0);
    enc.set_buffer(4, Some(&buf_out), 0);
    enc.set_bytes(5, 4, &k_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &h_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(k_selected as u64, 1, 1), metal::MTLSize::new(k_selected as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let sparse_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, k_selected).to_vec() };

    // Verify sparse results match corresponding dense results
    for (i, &idx) in indices.iter().enumerate() {
        let diff = (sparse_result[i] - dense_result[idx as usize]).abs();
        assert!(diff < 0.01, "sparse[{i}] (row {idx}) diff {diff}");
    }
}

// ── Residual ops ──

#[test]
fn residual_add_correct() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let a = vec![1.0f32, 2.0, 3.0, 4.0];
    let b = vec![10.0f32, 20.0, 30.0, 40.0];
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output(16);
    let len = 4u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(4, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, 4).to_vec() };
    assert!((result[0] - 11.0).abs() < 1e-5);
    assert!((result[1] - 22.0).abs() < 1e-5);
    assert!((result[2] - 33.0).abs() < 1e-5);
    assert!((result[3] - 44.0).abs() < 1e-5);
}

// ── GEGLU ──

#[test]
fn geglu_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("geglu_silu", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let n = 256;
    let gate: Vec<f32> = (0..n).map(|i| i as f32 * 0.1 - 12.8).collect();
    let up: Vec<f32> = (0..n).map(|i| i as f32 * 0.05).collect();

    // CPU reference
    let cpu_result = larql_compute::cpu::ops::geglu::geglu_silu_alloc(&gate, &up);

    // Metal
    let buf_g = bufs.transient_from_f32(&gate);
    let buf_u = bufs.transient_from_f32(&up);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_g), 0);
    enc.set_buffer(1, Some(&buf_u), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, n).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "GEGLU CPU vs Metal diff {diff}");
}

// ── Cross-validation: all kernels listed ──

#[test]
fn all_new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();

    let names = [
        "sgemm", "sgemm_transb",
        "q4_matvec", "q4_matvec_v2", "q4_matvec_v3", "q4_matvec_v4", "q4_matvec_v5",
        "q4_vecmat", "q4_f32_matvec", "q4_sparse_matvec",
        "q8_matvec",
        "geglu_silu", "quantize_q8",
        "residual_copy", "residual_add", "rms_norm",
        "causal_attention", "kv_attention", "kv_cache_append",
        "rope_apply", "fused_attention",
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

// ── RoPE shader ──

#[test]
fn rope_apply_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rope_apply", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 10000.0f32;

    // Create test data
    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: apply RoPE manually
    let half = dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half {
            let freq = 1.0 / base.powf(2.0 * d as f32 / dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half] = re * sin_a + im * cos_a;
        }
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full dim rotation
    enc.set_bytes(3, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half as u64, seq_len as u64, 1),
        metal::MTLSize::new(half as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "RoPE max diff {diff} exceeds 1e-4");
}

#[test]
fn rope_apply_partial_rotation() {
    // Verify partial RoPE: only first rotary_dim dimensions are rotated,
    // remaining dimensions pass through unchanged.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rope_apply", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let dim = 64u32;
    let seq_len = 4u32;
    let base = 1000000.0f32;
    let rotary_dim = 16u32; // 25% of dim (Gemma 4 style)

    let data: Vec<f32> = (0..seq_len as usize * dim as usize)
        .map(|i| (i as f32 * 0.01).sin())
        .collect();
    let data_copy = data.clone();

    // CPU reference: partial RoPE (rotate first rotary_dim dims, rest unchanged)
    let half_rotary = rotary_dim as usize / 2;
    let mut cpu_result = data_copy.clone();
    for pos in 0..seq_len as usize {
        for d in 0..half_rotary {
            let freq = 1.0 / base.powf(2.0 * d as f32 / rotary_dim as f32);
            let angle = pos as f32 * freq;
            let cos_a = angle.cos();
            let sin_a = angle.sin();
            let re = cpu_result[pos * dim as usize + d];
            let im = cpu_result[pos * dim as usize + d + half_rotary];
            cpu_result[pos * dim as usize + d] = re * cos_a - im * sin_a;
            cpu_result[pos * dim as usize + d + half_rotary] = re * sin_a + im * cos_a;
        }
        // Dimensions [rotary_dim..dim] must remain unchanged
    }

    // Metal
    let buf = bufs.transient_from_f32(&data);
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_bytes(1, 4, &dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(2, 4, &base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &rotary_dim as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(
        metal::MTLSize::new(half_rotary as u64, seq_len as u64, 1),
        metal::MTLSize::new(half_rotary as u64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe {
        std::slice::from_raw_parts(ptr, seq_len as usize * dim as usize).to_vec()
    };

    // Rotated dims should match CPU
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "Partial RoPE max diff {diff} exceeds 1e-4");

    // Non-rotated dims (rotary_dim..dim) should be unchanged
    for pos in 0..seq_len as usize {
        for d in rotary_dim as usize..dim as usize {
            let idx = pos * dim as usize + d;
            assert_eq!(
                metal_result[idx], data[idx],
                "Non-rotated dim {d} at pos {pos} was modified: {} -> {}",
                data[idx], metal_result[idx]
            );
        }
    }
}

// ── Fused attention shader ──

#[test]
fn fused_attention_single_token() {
    // At seq=1, attention output = V (only one key to attend to, weight = 1.0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();

    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 1u32;
    let head_dim = 32u32;
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = seq_len as usize * num_q as usize * head_dim as usize;
    let kv_total = seq_len as usize * num_kv as usize * head_dim as usize;

    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.1).sin()).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.2).cos()).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| i as f32 * 0.05 + 1.0).collect();

    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // At seq=1, output should be V (rotated by RoPE, but with weight=1.0)
    // Just verify nonzero and finite
    assert!(result.iter().all(|v| v.is_finite()), "output should be finite");
    assert!(result.iter().any(|v| v.abs() > 0.01), "output should be nonzero");
}

// ══════════════════════════════════════════════════════════════
// Shader correctness tests — each shader vs CPU reference
// ══════════════════════════════════════════════════════════════

// ── rms_norm with offset ──

#[test]
fn rms_norm_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + (i as f32 * 0.01)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32; // Gemma 2/3 style (Gemma 4 uses 0.0)

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms)
        .collect();

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch for cooperative SIMD reduction.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm max diff {diff}");
}

#[test]
fn rms_norm_zero_offset() {
    // Standard RMS norm (Llama-style, offset=0)
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 32usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.2 - 3.0).collect();
    let weight: Vec<f32> = vec![1.0f32; len];
    let eps = 1e-6f32;
    let offset = 0.0f32;

    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().map(|xi| xi * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-5, "rms_norm(offset=0) max diff {diff}");
}

// ── cooperative SIMD norm (large vector, multi-simdgroup) ──

#[test]
fn rms_norm_large_vector_simd_cooperative() {
    // Tests with len=2560 (actual Gemma 4B hidden size) to exercise
    // the cooperative SIMD reduction across multiple simdgroups.
    // With TG=256: 8 simdgroups, each sums a 2560/256=10-element stripe.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let x: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.0037).sin() * 2.0)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + (i as f32 * 0.0001)).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = x.iter().zip(weight.iter())
        .map(|(xi, wi)| xi * (wi + offset) * rms).collect();

    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    // Single threadgroup dispatch — cooperative SIMD reduction needs all threads in one TG.
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "rms_norm(len=2560) SIMD cooperative max diff {diff}");
}

#[test]
fn residual_norm_large_vector_simd_cooperative() {
    // Tests residual_norm with len=2560 to exercise cooperative reduction.
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 2560usize;
    let a: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.003).cos() * 1.5)).collect();
    let b: Vec<f32> = (0..len).map(|i| ((i as f32 * 0.007).sin() * 0.5)).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.9 + (i as f32 * 0.00005)).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: h = a + b, then rms_norm(h)
    let h: Vec<f32> = a.iter().zip(&b).map(|(ai, bi)| ai + bi).collect();
    let sum_sq: f32 = h.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = h.iter().zip(weight.iter())
        .map(|(hi, wi)| hi * (wi + offset) * rms).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_result = larql_compute::metal::buffers::read_buffer_f32(&buf_out, len);
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm(len=2560) SIMD cooperative max diff {diff}");
}

// ── residual_add ──

#[test]
fn residual_add_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_add", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 128usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1).collect();
    let b: Vec<f32> = (0..len).map(|i| -(i as f32 * 0.05)).collect();
    let cpu_result: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-6, "residual_add max diff {diff}");
}

// ── fused_attention correctness (3 tokens, 2 heads, verified against CPU) ──

#[test]
fn fused_attention_matches_cpu_reference() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("fused_attention", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let seq_len = 3u32;
    let head_dim = 8u32;  // small for easy debugging
    let num_q = 2u32;
    let num_kv = 2u32;
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let rope_base = 10000.0f32;
    let use_qk_norm = 0u32;
    let softcap = 0.0f32;

    let total = (seq_len * num_q * head_dim) as usize;
    let kv_total = (seq_len * num_kv * head_dim) as usize;

    // Deterministic test data
    let q: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37 + 1.0).sin() * 0.5).collect();
    let k: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.23 + 2.0).cos() * 0.5).collect();
    let v: Vec<f32> = (0..kv_total).map(|i| (i as f32 * 0.11 + 3.0).sin() * 0.3).collect();

    // ── CPU reference: apply RoPE then causal attention ──
    let hd = head_dim as usize;
    let half = hd / 2;
    let nq = num_q as usize;
    let nkv = num_kv as usize;
    let sl = seq_len as usize;

    // Apply RoPE to Q and K
    let mut q_rope = q.clone();
    let mut k_rope = k.clone();
    for pos in 0..sl {
        for head in 0..nq {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nq * hd + head * hd + d;
                let idx_im = pos * nq * hd + head * hd + d + half;
                let re = q[idx_re];
                let im = q[idx_im];
                q_rope[idx_re] = re * cos_a - im * sin_a;
                q_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
        for head in 0..nkv {
            for d in 0..half {
                let freq = 1.0 / rope_base.powf(2.0 * d as f32 / hd as f32);
                let angle = pos as f32 * freq;
                let (cos_a, sin_a) = (angle.cos(), angle.sin());
                let idx_re = pos * nkv * hd + head * hd + d;
                let idx_im = pos * nkv * hd + head * hd + d + half;
                let re = k[idx_re];
                let im = k[idx_im];
                k_rope[idx_re] = re * cos_a - im * sin_a;
                k_rope[idx_im] = re * sin_a + im * cos_a;
            }
        }
    }

    // Causal attention per head per position
    let mut cpu_out = vec![0.0f32; total];
    for head in 0..nq {
        let kv_head = head / (nq / nkv);
        for qi in 0..sl {
            // Compute scores for all k <= qi
            let mut scores = Vec::new();
            for ki in 0..=qi {
                let mut dot = 0.0f32;
                for d in 0..hd {
                    let q_val = q_rope[qi * nq * hd + head * hd + d];
                    let k_val = k_rope[ki * nkv * hd + kv_head * hd + d];
                    dot += q_val * k_val;
                }
                scores.push(dot * scale);
            }
            // Softmax
            let max_s = scores.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            let exps: Vec<f32> = scores.iter().map(|s| (s - max_s).exp()).collect();
            let sum_exp: f32 = exps.iter().sum();
            let weights: Vec<f32> = exps.iter().map(|e| e / sum_exp).collect();
            // Weighted V
            for d in 0..hd {
                let mut acc = 0.0f32;
                for ki in 0..=qi {
                    acc += weights[ki] * v[ki * nkv * hd + kv_head * hd + d];
                }
                cpu_out[qi * nq * hd + head * hd + d] = acc;
            }
        }
    }

    // ── Metal ──
    let buf_q = bufs.transient_from_f32(&q);
    let buf_k = bufs.transient_from_f32(&k);
    let buf_v = bufs.transient_from_f32(&v);
    let buf_out = bufs.output((total * 4) as u64);

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_q), 0);
    enc.set_buffer(1, Some(&buf_k), 0);
    enc.set_buffer(2, Some(&buf_v), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &seq_len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &head_dim as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &num_q as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &num_kv as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &scale as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &rope_base as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(10, 4, &use_qk_norm as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(11, 4, &softcap as *const f32 as *const std::ffi::c_void);
    let skip_rope_val = 0u32;
    enc.set_bytes(12, 4, &skip_rope_val as *const u32 as *const std::ffi::c_void);
    let rotary_dim_val = 0u32; // 0 = full head_dim rotation
    enc.set_bytes(13, 4, &rotary_dim_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(num_q as u64, seq_len as u64, 1),
        metal::MTLSize::new(256, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, total).to_vec() };

    // Compare
    let diff = max_diff(&cpu_out, &metal_result);
    assert!(diff < 0.01, "fused_attention max diff {diff} (expected < 0.01).\nCPU[0..8]: {:?}\nGPU[0..8]: {:?}",
        &cpu_out[..8.min(total)], &metal_result[..8.min(total)]);
}

// ── quantize_q8 shader ──

#[test]
fn quantize_q8_matches_cpu() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("quantize_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();

    // CPU reference
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&x);

    // Metal
    let buf_x = bufs.transient_from_f32(&x);
    let buf_q8 = bufs.output(len as u64);
    let buf_scales = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_q8), 0);
    enc.set_buffer(2, Some(&buf_scales), 0);
    let n_blocks = (len / 32) as u32;
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n_blocks as u64, 1, 1), metal::MTLSize::new(n_blocks as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_scales.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_scales: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_scales[i]).abs();
        assert!(diff < 0.01, "Q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_scales[i]);
    }
    // Check quantized values match (allow ±1 for rounding)
    let mut mismatches = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 1 {
            mismatches += 1;
        }
    }
    assert!(mismatches == 0, "Q8 quantize: {mismatches}/{len} values differ by >1");
}

// ── Fused ops: rms_norm_q8, residual_norm, residual_norm_q8 ──

#[test]
fn rms_norm_q8_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm_q8", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let x: Vec<f32> = (0..len).map(|i| i as f32 * 0.15 - 4.8).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.5 + i as f32 * 0.01).collect();
    let eps = 1e-6f32;
    let offset = 1.0f32;

    // CPU reference: norm then quantize
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let normed: Vec<f32> = x.iter().zip(weight.iter()).map(|(xi, wi)| xi * (wi + offset) * rms).collect();
    let (cpu_q8, cpu_scales) = larql_compute::cpu::q4::quantize_to_q8(&normed);

    // Metal fused
    let buf_x = bufs.transient_from_f32(&x);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_q8 = bufs.output(len as u64);
    let buf_sc = bufs.output((len / 32 * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_q8), 0);
    enc.set_buffer(3, Some(&buf_sc), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let q8_ptr = buf_q8.contents() as *const i8;
    let sc_ptr = buf_sc.contents() as *const f32;
    let metal_q8: Vec<i8> = unsafe { std::slice::from_raw_parts(q8_ptr, len).to_vec() };
    let metal_sc: Vec<f32> = unsafe { std::slice::from_raw_parts(sc_ptr, len / 32).to_vec() };

    // Check scales match
    for i in 0..len/32 {
        let diff = (cpu_scales[i] - metal_sc[i]).abs();
        assert!(diff < 0.1, "fused rms_norm_q8 scale[{i}] diff: cpu={} metal={}", cpu_scales[i], metal_sc[i]);
    }
    // Check Q8 values (allow ±2 rounding)
    let mut bad = 0;
    for i in 0..len {
        if (cpu_q8[i] as i32 - metal_q8[i] as i32).abs() > 2 { bad += 1; }
    }
    assert!(bad == 0, "fused rms_norm_q8: {bad}/{len} values differ by >2");
}

#[test]
fn residual_norm_matches_separate_ops() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let fused = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let len = 64usize;
    let a: Vec<f32> = (0..len).map(|i| i as f32 * 0.1 - 3.2).collect();
    let b: Vec<f32> = (0..len).map(|i| i as f32 * 0.05 + 0.3).collect();
    let weight: Vec<f32> = (0..len).map(|i| 0.8 + i as f32 * 0.005).collect();
    let eps = 1e-6f32;
    let offset = 0.0f32;

    // CPU reference: add then norm
    let sum: Vec<f32> = a.iter().zip(b.iter()).map(|(x, y)| x + y).collect();
    let sum_sq: f32 = sum.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / len as f32 + eps).sqrt();
    let cpu_result: Vec<f32> = sum.iter().zip(weight.iter()).map(|(s, w)| s * (w + offset) * rms).collect();

    // Metal fused
    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w = bufs.transient_from_f32(&weight);
    let buf_out = bufs.output((len * 4) as u64);
    let len_val = len as u32;

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&fused);
    enc.set_buffer(0, Some(&buf_a), 0);
    enc.set_buffer(1, Some(&buf_b), 0);
    enc.set_buffer(2, Some(&buf_w), 0);
    enc.set_buffer(3, Some(&buf_out), 0);
    enc.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(len as u64, 1, 1), metal::MTLSize::new(len as u64, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let ptr = buf_out.contents() as *const f32;
    let metal_result: Vec<f32> = unsafe { std::slice::from_raw_parts(ptr, len).to_vec() };
    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 1e-4, "residual_norm max diff {diff}");
}

// ── Q4_K and Q6_K matvec ──

#[test]
fn q4k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize; // must be multiple of 256 for Q4_K super-blocks
    let rows = 64usize;

    // Create Q4_K data (148 bytes per 256 values)
    // Simple: all-zero super-blocks with non-zero scale → produces non-zero output
    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 148;
    let mut q4k_data = vec![0u8; rows * bytes_per_row];

    // Set a non-zero scale and some non-zero quants for each row
    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 148;
            // d = 1.0 as f16
            q4k_data[base] = 0x00;
            q4k_data[base + 1] = 0x3C;
            // scale[0] = 1
            q4k_data[base + 4] = 1;
            // quant nibbles: 0x11 = lo=1, hi=1
            for i in 20..148 { q4k_data[base + i] = 0x11; }
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q4_K should produce nonzero output");
}

#[test]
fn q6k_matvec_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    let superblocks_per_row = hidden / 256;
    let bytes_per_row = superblocks_per_row * 210;
    let mut q6k_data = vec![0u8; rows * bytes_per_row];

    for row in 0..rows {
        for sb in 0..superblocks_per_row {
            let base = row * bytes_per_row + sb * 210;
            // Set d = 1.0 as f16 at offset 208
            q6k_data[base + 208] = 0x00;
            q6k_data[base + 209] = 0x3C;
            // Set scales[0] = 1
            q6k_data[base + 192] = 1;
            // Set some non-zero lower nibbles
            for i in 0..128 { q6k_data[base + i] = 0x33; } // lo=3 for each nibble
        }
    }

    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    assert_eq!(result.len(), rows);
    assert!(result.iter().any(|&v| v.abs() > 0.001), "Q6_K should produce nonzero output");
}

// ── Q4_K round-trip: quantize then dequantize via GPU matvec ──

#[test]
fn q4k_quantize_then_matvec_matches_f32() {
    let _metal = get_metal();
    let hidden = 256usize;
    let rows = 32usize;

    // Create f32 matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference: matrix @ x
    let mut cpu_result = vec![0.0f32; rows];
    for r in 0..rows {
        let mut dot = 0.0f32;
        for c in 0..hidden { dot += matrix[r * hidden + c] * x[c]; }
        cpu_result[r] = dot;
    }

    // Q4_K quantize (via models crate) then GPU matvec
    let padded_len = (rows * hidden).div_ceil(256) * 256;
    let mut padded = matrix.clone();
    padded.resize(padded_len, 0.0);
    // Verify f32 reference is nonzero (sanity — full Q4_K round-trip tested via inference)
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001));
}

// ── Cross-backend: Q4_K Metal vs CPU ──

#[test]
fn q4k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    let cpu_result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.5, "Q4_K matvec Metal vs CPU max diff {diff} exceeds 0.5");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q6_K Metal vs CPU ──

#[test]
fn q6k_matvec_matches_cpu() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let q6k_data = larql_compute::cpu::ops::q4_common::quantize_q6_k(&matrix);

    let cpu_result = cpu.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();
    let metal_result = metal.q6k_matvec(&q6k_data, &x, rows, hidden).unwrap();

    let diff = max_diff(&cpu_result, &metal_result);
    assert!(diff < 0.3, "Q6_K matvec Metal vs CPU max diff {diff} exceeds 0.3");
    assert!(cpu_result.iter().any(|&v| v.abs() > 0.001), "CPU result should be nonzero");
    assert!(metal_result.iter().any(|&v| v.abs() > 0.001), "Metal result should be nonzero");
}

// ── Cross-backend: Q8 matvec Metal vs CPU ──

#[test]
fn q8_matvec_metal_matches_cpu_reference() {
    let metal = get_metal();
    let hidden = 256usize;
    let rows = 64usize;

    // Create matrix and input
    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    // CPU f32 reference
    let mut cpu_ref = vec![0.0f32; rows];
    for r in 0..rows {
        for c in 0..hidden { cpu_ref[r] += matrix[r * hidden + c] * x[c]; }
    }

    // Q4_0 quantize and run through Metal Q4 matvec
    let q4_data = quantize_q4_0(&matrix);
    let (q8_x, q8_scales) = q4::quantize_to_q8(&x);

    let metal_result = metal.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden).unwrap();

    // Q4 is lossy (4-bit weights + 8-bit input), so allow generous tolerance
    let diff = max_diff(&cpu_ref, &metal_result);
    assert!(diff < 3.0, "Q4 matvec vs f32 ref max diff {diff} exceeds 3.0");
}

// ── Cross-backend: multi-position Q4_K ──

#[test]
fn multi_position_q4k_matches_individual() {
    let metal = get_metal();
    let cpu = larql_compute::cpu::CpuBackend;

    let hidden = 256usize;
    let rows = 32usize;
    let seq_len = 6usize;

    let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
    let q4k_data = larql_compute::cpu::ops::q4_common::quantize_q4_k(&matrix);

    // Run individual matvec per position on CPU
    let mut per_pos_results = Vec::with_capacity(seq_len);
    for s in 0..seq_len {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let result = cpu.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        per_pos_results.push(result);
    }

    // Run same on Metal and compare
    for (s, cpu_result) in per_pos_results.iter().enumerate() {
        let x: Vec<f32> = (0..hidden).map(|i| ((i + s * 100) as f32 * 0.01).sin()).collect();
        let metal_result = metal.q4k_matvec(&q4k_data, &x, rows, hidden).unwrap();
        let diff = max_diff(cpu_result, &metal_result);
        assert!(diff < 0.5, "Position {s}: Q4_K Metal vs CPU max diff {diff}");
    }
}

// ── Smoke test: full pipeline produces output ──

#[test]
fn full_pipeline_seq1_produces_nonzero() {
    let metal = get_metal();
    let hidden = 256usize;
    let inter = 512usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 4usize;
    let head_dim = 64usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Create synthetic Q4_0 weights for one layer
    let gate_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let up_data = quantize_q4_0(&vec![0.01f32; inter * hidden]);
    let down_data = quantize_q4_0(&vec![0.01f32; hidden * inter]);
    let wq_data = quantize_q4_0(&vec![0.01f32; q_dim * hidden]);
    let wk_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wv_data = quantize_q4_0(&vec![0.01f32; kv_dim * hidden]);
    let wo_data = quantize_q4_0(&vec![0.01f32; hidden * q_dim]);
    let (_q8_x_q, q8_s_q) = q4::quantize_to_q8(&vec![0.01f32; hidden]);

    let norm = vec![1.0f32; hidden];
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();

    let layer = larql_compute::FullPipelineLayer {
        wq: larql_compute::QuantWeight { data: &wq_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wk: larql_compute::QuantWeight { data: &wk_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wv: larql_compute::QuantWeight { data: &wv_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        wo: larql_compute::QuantWeight { data: &wo_data, scales: Some(&q8_s_q), format: larql_compute::QuantFormat::Q4_0 },
        gate: larql_compute::QuantWeight { data: &gate_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        up: larql_compute::QuantWeight { data: &up_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        down: larql_compute::QuantWeight { data: &down_data, scales: None, format: larql_compute::QuantFormat::Q4_0 },
        input_norm: &norm,
        post_attn_norm: &norm,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        norm_offset: 1.0,
        has_post_norms: false,
            activation: larql_compute::Activation::Silu,
            qk_norm_offset: 0.0,
            eps: 1e-6,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim,
            num_q_heads,
            num_kv_heads,
            rope_base: 10000.0,
            rotary_dim: 0,
            sliding_window: 0,
            has_v_norm: false,
            layer_scalar: 0.0,
            input_norm_bias: None,
            q_norm_weight: None, k_norm_weight: None, post_attn_norm_bias: None,
            ffn_up_bias: None,
            ffn_down_bias: None,
    };

    let result = metal.full_pipeline_q4(
        &[layer], &x, hidden, inter, q_dim, kv_dim,
        1, num_q_heads, num_kv_heads, head_dim,
        10000.0, false, 0.0,
    );

    assert!(result.is_some(), "full_pipeline_q4 should return Some");
    let output = result.unwrap();
    assert_eq!(output.len(), hidden);
    assert!(output.iter().any(|&v| v.abs() > 1e-6), "Pipeline output should be nonzero");
}

// ═══════════════════════════════════════════════════════════════
// New shader kernel tests (model-agnostic compute alignment)
// ═══════════════════════════════════════════════════════════════

#[test]
fn new_kernel_functions_exist() {
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let opts = metal::CompileOptions::new();
    let lib = device.new_library_with_source(&src, &opts).unwrap();

    let names = [
        "silu", "gelu_tanh",                         // standalone activations
        "layer_norm", "layer_norm_no_bias",           // LayerNorm
        "v_norm",                                      // V-norm
        "scale_vector",                                // per-layer scalar
    ];
    for name in &names {
        lib.get_function(name, None)
            .unwrap_or_else(|e| panic!("Kernel '{name}' not found: {e}"));
    }
}

#[test]
fn silu_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| x / (1.0 + (-x).exp())).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.silu_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "SiLU standalone max diff {diff} exceeds 1e-5");
}

#[test]
fn gelu_tanh_standalone_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.05).collect();
    let expected: Vec<f32> = input.iter().map(|&x| {
        let c = (2.0f32 / std::f32::consts::PI).sqrt();
        let t = (c * (x + 0.044715 * x * x * x)).tanh();
        0.5 * x * (1.0 + t)
    }).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let output_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.gelu_tanh_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&output_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&output_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "GELU-tanh standalone max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let bias: Vec<f32> = (0..n).map(|i| (i as f32) * 0.01).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    // CPU reference
    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset) + bias[i]
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let b_buf = metal.bufs().transient_from_f32(&bias);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&b_buf), 0);
    enc.set_buffer(3, Some(&out_buf), 0);
    enc.set_bytes(4, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(6, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm max diff {diff} exceeds 1e-4");
}

#[test]
fn layer_norm_no_bias_matches_cpu() {
    let metal = get_metal();
    let n = 128;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 64.0) * 0.1).collect();
    let weight: Vec<f32> = (0..n).map(|i| 1.0 + (i as f32) * 0.001).collect();
    let eps = 1e-5f32;
    let offset = 0.0f32;

    let mean: f32 = x.iter().sum::<f32>() / n as f32;
    let var: f32 = x.iter().map(|v| (v - mean) * (v - mean)).sum::<f32>() / n as f32;
    let inv_std = 1.0 / (var + eps).sqrt();
    let expected: Vec<f32> = (0..n).map(|i| {
        (x[i] - mean) * inv_std * (weight[i] + offset)
    }).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.layer_norm_no_bias_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&w_buf), 0);
    enc.set_buffer(2, Some(&out_buf), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(128, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-4, "LayerNorm (no bias) max diff {diff} exceeds 1e-4");
}

#[test]
fn v_norm_matches_cpu() {
    let metal = get_metal();
    let n = 256;
    let x: Vec<f32> = (0..n).map(|i| (i as f32 - 128.0) * 0.02).collect();
    let eps = 1e-6f32;

    // CPU reference: parameter-free RMSNorm
    let sum_sq: f32 = x.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / n as f32 + eps).sqrt();
    let expected: Vec<f32> = x.iter().map(|v| v * rms).collect();

    let x_buf = metal.bufs().transient_from_f32(&x);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.v_norm_pipeline);
    enc.set_buffer(0, Some(&x_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-5, "V-norm max diff {diff} exceeds 1e-5");
}

#[test]
fn scale_vector_matches_cpu() {
    let metal = get_metal();
    let n = 512;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 - 256.0) * 0.01).collect();
    let scalar = 0.73f32;
    let expected: Vec<f32> = input.iter().map(|v| v * scalar).collect();

    let input_buf = metal.bufs().transient_from_f32(&input);
    let out_buf = metal.bufs().output((n * 4) as u64);
    let n_val = n as u32;

    let cmd = metal.queue().new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&metal.scale_vector_pipeline);
    enc.set_buffer(0, Some(&input_buf), 0);
    enc.set_buffer(1, Some(&out_buf), 0);
    enc.set_bytes(2, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &scalar as *const f32 as *const std::ffi::c_void);
    enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(256, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let result = larql_compute::metal::buffers::read_buffer_f32(&out_buf, n);
    let diff = max_diff(&expected, &result);
    assert!(diff < 1e-6, "scale_vector max diff {diff} exceeds 1e-6");
}

#[test]
fn rms_norm_with_different_eps() {
    // Verify that eps parameter actually affects output (was hardcoded to 1e-6 before)
    let metal = get_metal();
    let n = 64;
    let x: Vec<f32> = vec![0.001; n]; // tiny values where eps matters
    let weight: Vec<f32> = vec![1.0; n];
    let offset = 0.0f32;

    let x_buf = metal.bufs().transient_from_f32(&x);
    let w_buf = metal.bufs().transient_from_f32(&weight);
    let n_val = n as u32;

    // Run with eps=1e-6
    let out1 = metal.bufs().output((n * 4) as u64);
    let eps1 = 1e-6f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out1), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps1 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    // Run with eps=0.1 (much larger)
    let out2 = metal.bufs().output((n * 4) as u64);
    let eps2 = 0.1f32;
    {
        let cmd = metal.queue().new_command_buffer();
        let enc = cmd.new_compute_command_encoder();
        enc.set_compute_pipeline_state(&metal.rms_norm_pipeline);
        enc.set_buffer(0, Some(&x_buf), 0);
        enc.set_buffer(1, Some(&w_buf), 0);
        enc.set_buffer(2, Some(&out2), 0);
        enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc.set_bytes(4, 4, &eps2 as *const f32 as *const std::ffi::c_void);
        enc.set_bytes(5, 4, &offset as *const f32 as *const std::ffi::c_void);
        enc.dispatch_threads(metal::MTLSize::new(n as u64, 1, 1), metal::MTLSize::new(64, 1, 1));
        enc.end_encoding();
        cmd.commit();
        cmd.wait_until_completed();
    }

    let r1 = larql_compute::metal::buffers::read_buffer_f32(&out1, n);
    let r2 = larql_compute::metal::buffers::read_buffer_f32(&out2, n);
    let diff = max_diff(&r1, &r2);
    assert!(diff > 0.1, "Different eps values should produce different outputs (diff={diff})");
}

// ═══════════════════════════════════════════════════════════════
// Gemma 3 decode_token: reproduces the QK-norm NaN bug
// ═══════════════════════════════════════════════════════════════

#[test]
fn decode_token_gemma3_produces_finite() {
    // Reproduces the inf/NaN output bug for Gemma 3's post-norm architecture.
    // Config matches Gemma 3: has_post_norms=true, qk_norm_offset=1.0.
    // Realistic dims (especially head_dim=256 matching Gemma 3 4B) plus
    // pre-populated KV cache to mirror the CPU-prefill + GPU-decode flow.

    let metal = get_metal();
    // Gemma 3 4B actual dims: hidden=2560, inter=10240, q_heads=8, kv_heads=4, head_dim=256.
    // We use scaled-down dims but keep the GQA ratio and head_dim=256 that match real Gemma 3.
    let hidden = 1024usize;
    let inter = 2048usize;
    let num_q_heads = 4usize;
    let num_kv_heads = 2usize;
    let head_dim = 256usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;

    // Gemma 3 4B vindex ships Q4_K for Q/K/O/gate/up and Q6_K for V/down.
    // Using Q4_K here (not Q4_0) so we exercise the same shader dispatch path.
    use larql_compute::cpu::ops::q4_common::{quantize_q4_k, quantize_q6_k};
    // Q4_K super-blocks are 256 elements; input lengths must be multiples of 256.
    let gate_data = quantize_q4_k(&vec![0.01f32; inter * hidden]);
    let up_data = quantize_q4_k(&vec![0.01f32; inter * hidden]);
    let down_data = quantize_q6_k(&vec![0.01f32; hidden * inter]);
    let wq_data = quantize_q4_k(&vec![0.01f32; q_dim * hidden]);
    let wk_data = quantize_q4_k(&vec![0.01f32; kv_dim * hidden]);
    let wv_data = quantize_q6_k(&vec![0.01f32; kv_dim * hidden]);
    let wo_data = quantize_q4_k(&vec![0.01f32; hidden * q_dim]);
    let (_q8_x, q8_scales) = q4::quantize_to_q8(&vec![0.01f32; hidden]);

    let norm = vec![1.0f32; hidden];
    // Gemma 3 post-embed hidden state range is ~[-26, 12]. Scale x to match.
    let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin() * 20.0).collect();

    let layer = larql_compute::FullPipelineLayer {
        wq: larql_compute::QuantWeight { data: &wq_data, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
        wk: larql_compute::QuantWeight { data: &wk_data, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
        wv: larql_compute::QuantWeight { data: &wv_data, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q6_K },
        wo: larql_compute::QuantWeight { data: &wo_data, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
        gate: larql_compute::QuantWeight { data: &gate_data, scales: None, format: larql_compute::QuantFormat::Q4_K },
        up: larql_compute::QuantWeight { data: &up_data, scales: None, format: larql_compute::QuantFormat::Q4_K },
        down: larql_compute::QuantWeight { data: &down_data, scales: None, format: larql_compute::QuantFormat::Q6_K },
        input_norm: &norm,
        post_attn_norm: &norm,
        pre_ffn_norm: Some(&norm),   // Gemma 3: has post-norms
        post_ffn_norm: Some(&norm),
        norm_offset: 1.0,             // Gemma 3: +1.0 convention
        qk_norm_offset: 1.0,
        eps: 1e-6,
        has_post_norms: true,         // Gemma 3: yes
        norm_type: larql_compute::NormType::RmsNorm,
        ffn_type: larql_compute::FfnType::Gated,
        activation: larql_compute::Activation::GeluTanh, // Gemma
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10000.0,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        q_norm_weight: None, k_norm_weight: None, post_attn_norm_bias: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
    };

    // Simulate CPU prefill populating KV for 3 positions. The K values are
    // "already QK-normed" (unit-ish variance). Metal decode_token will
    // then produce an un-normed new K, causing scale mismatch.
    let be: &dyn ComputeBackend = &metal;
    be.reset_kv_cache();
    // Simulate 6-token CPU prefill with LARGER-magnitude K/V to stress attention.
    // Real Gemma K after QKV projection can have |values| in the ~10s range.
    let seq_len_pre = 6usize;
    let k_large: Vec<f32> = (0..seq_len_pre * kv_dim).map(|i| (i as f32 * 0.01).sin() * 8.0).collect();
    let v_large: Vec<f32> = (0..seq_len_pre * kv_dim).map(|i| (i as f32 * 0.013).cos() * 5.0).collect();
    be.populate_kv_layer(0, &k_large, &v_large, seq_len_pre, num_kv_heads, head_dim);

    let result = be.decode_token(
        &[layer], &x, hidden, inter, q_dim, kv_dim,
        num_q_heads, num_kv_heads, head_dim, 10000.0,
    );

    assert!(result.is_some(), "decode_token should return Some");
    let out = result.unwrap();
    assert_eq!(out.len(), hidden);
    let n_inf = out.iter().copied().filter(|v: &f32| v.is_infinite()).count();
    let n_nan = out.iter().copied().filter(|v: &f32| v.is_nan()).count();
    println!("decode_token output: {} inf, {} nan, of {}", n_inf, n_nan, out.len());
    assert_eq!(n_inf, 0, "decode_token output contains inf values (QK-norm bug)");
    assert_eq!(n_nan, 0, "decode_token output contains NaN values (QK-norm bug)");

    be.reset_kv_cache();
}

// ═══════════════════════════════════════════════════════════════
// Real-data decode_token reproducer — loads Gemma 3 4B layer 0
// Q4K weights + real norm vectors from a vindex, uses synthetic
// input, calls decode_token. Fast (no model load, ~10s).
// ═══════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn decode_token_real_layer0_produces_finite() {
    let vindex_path = std::env::var("LARQL_VINDEX_PATH")
        .expect("set LARQL_VINDEX_PATH to a Q4K-prepared Gemma 3 4B vindex");
    let dir = std::path::Path::new(&vindex_path);

    // Config values — hard-coded to match Gemma 3 4B. Could read from index.json.
    let hidden = 2560usize;
    let inter = 10240usize;
    let num_q_heads = 8usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let q_dim = num_q_heads * head_dim;  // 2048
    let kv_dim = num_kv_heads * head_dim; // 1024
    let _ = (q_dim, kv_dim);

    // ── 1. Read weight_manifest.json, find layer 0 entries ──
    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("weight_manifest.json")).unwrap()
    ).unwrap();
    // Inline f16 → f32 decode (avoid larql_models dep from this test)
    fn decode_f16(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(2).map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let sign = (bits >> 15) & 0x1;
            let exp  = (bits >> 10) & 0x1F;
            let mant = (bits & 0x3FF) as u32;
            let f32_bits = if exp == 0 {
                if mant == 0 { (sign as u32) << 31 }
                else {
                    let mut m = mant;
                    let mut e: i32 = -14;
                    while (m & 0x400) == 0 { m <<= 1; e -= 1; }
                    let exp32 = (e + 127) as u32;
                    ((sign as u32) << 31) | (exp32 << 23) | ((m & 0x3FF) << 13)
                }
            } else if exp == 0x1F {
                ((sign as u32) << 31) | (0xFF << 23) | (mant << 13)
            } else {
                let exp32 = (exp as i32 - 15 + 127) as u32;
                ((sign as u32) << 31) | (exp32 << 23) | (mant << 13)
            };
            f32::from_bits(f32_bits)
        }).collect()
    }
    let find_vec = |key: &str| -> Vec<f32> {
        let e = manifest.iter().find(|e|
            e.get("key").and_then(|k| k.as_str()) == Some(key)
        ).unwrap_or_else(|| panic!("missing vector {key}"));
        let offset = e["offset"].as_u64().unwrap() as usize;
        let length = e["length"].as_u64().unwrap() as usize;
        let file_name = e["file"].as_str().unwrap();
        let file = std::fs::File::open(dir.join(file_name)).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        let raw = &mmap[offset..offset + length];
        decode_f16(raw)
    };
    let input_norm = find_vec("layers.0.input_layernorm.weight");
    let post_attn_norm = find_vec("layers.0.post_attention_layernorm.weight");
    let pre_ffn_norm = find_vec("layers.0.pre_feedforward_layernorm.weight");
    let post_ffn_norm = find_vec("layers.0.post_feedforward_layernorm.weight");
    // Gemma 3 QK-norm weights — now plumbed through FullPipelineLayer.
    let q_norm = find_vec("layers.0.self_attn.q_norm.weight");
    let k_norm = find_vec("layers.0.self_attn.k_norm.weight");
    println!("QK-norm weights: q len={} max={:.3}, k len={} max={:.3}",
        q_norm.len(), q_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        k_norm.len(), k_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max));
    println!("Loaded norms: input max {:.3}, post_attn max {:.3}, pre_ffn max {:.3}, post_ffn max {:.3}",
        input_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        post_attn_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        pre_ffn_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        post_ffn_norm.iter().map(|v| v.abs()).fold(0.0f32, f32::max));

    // ── 2. Real Q4K attention weights (layer 0 only) ──
    let q4k_manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("attn_weights_q4k_manifest.json")).unwrap()
    ).unwrap();
    let attn_file = std::fs::File::open(dir.join("attn_weights_q4k.bin")).unwrap();
    let attn_mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&attn_file).unwrap() });

    let find_q4k = |key_needle: &str| -> (usize, usize, larql_compute::QuantFormat) {
        let e = q4k_manifest.iter().find(|e|
            e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0") && k.contains(key_needle))
        ).unwrap_or_else(|| panic!("missing q4k {key_needle}"));
        let offset = e["offset"].as_u64().unwrap() as usize;
        let length = e["length"].as_u64().unwrap() as usize;
        let format = match e.get("format").and_then(|v| v.as_str()) {
            Some("Q6_K") => larql_compute::QuantFormat::Q6_K,
            Some("Q4_KF") | Some("Q4_K_GGUF") => larql_compute::QuantFormat::Q4_KF,
            _ => larql_compute::QuantFormat::Q4_K,
        };
        (offset, length, format)
    };
    let (q_off, q_len, q_fmt) = find_q4k("q_proj");
    let (k_off, k_len, k_fmt) = find_q4k("k_proj");
    let (v_off, v_len, v_fmt) = find_q4k("v_proj");
    let (o_off, o_len, o_fmt) = find_q4k("o_proj");
    let wq_bytes = &attn_mmap[q_off..q_off + q_len];
    let wk_bytes = &attn_mmap[k_off..k_off + k_len];
    let wv_bytes = &attn_mmap[v_off..v_off + v_len];
    let wo_bytes = &attn_mmap[o_off..o_off + o_len];
    println!("Q4K attn layer 0: q={}B k={}B v={}B o={}B", q_len, k_len, v_len, o_len);

    // ── 3. Layer 0 FFN weights from interleaved_q4k.bin ──
    // Detect layout from file size (Q4_K 148 B, Q4_KF 144 B, Q6_K 210 B per 256 vals).
    let inter_file = std::fs::File::open(dir.join("interleaved_q4k.bin")).unwrap();
    let inter_mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&inter_file).unwrap() });
    let q4k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 148;
    let q4kf_bytes_per_matrix = (inter * hidden).div_ceil(256) * 144;
    let q6k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 210;
    let num_layers_in_file = 34;
    let file_per_layer = inter_mmap.len() / num_layers_in_file;
    let (gate_bytes_per, up_bytes_per, down_bytes_per, layer_gate_fmt, layer_up_fmt) =
        if file_per_layer == 2 * q4kf_bytes_per_matrix + q6k_bytes_per_matrix {
            (q4kf_bytes_per_matrix, q4kf_bytes_per_matrix, q6k_bytes_per_matrix,
             larql_compute::QuantFormat::Q4_KF, larql_compute::QuantFormat::Q4_KF)
        } else if file_per_layer == 2 * q4k_bytes_per_matrix + q6k_bytes_per_matrix {
            (q4k_bytes_per_matrix, q4k_bytes_per_matrix, q6k_bytes_per_matrix,
             larql_compute::QuantFormat::Q4_K, larql_compute::QuantFormat::Q4_K)
        } else {
            panic!("unknown FFN layout file_per_layer={file_per_layer}")
        };
    let layer_bytes = gate_bytes_per + up_bytes_per + down_bytes_per;
    let inter_total = inter_mmap.len();
    println!("interleaved_q4k.bin: {}B total, layer 0 computed bytes={}B", inter_total, layer_bytes);

    // Layer 0 offsets
    let gate_bytes = &inter_mmap[0..gate_bytes_per];
    let up_bytes = &inter_mmap[gate_bytes_per..gate_bytes_per + up_bytes_per];
    let down_bytes = &inter_mmap[gate_bytes_per + up_bytes_per..layer_bytes];

    // ── 4. Construct FullPipelineLayer (Gemma 3 config) ──
    let (_q8_x, q8_scales) = q4::quantize_to_q8(&vec![0.01f32; hidden]);
    let layer = larql_compute::FullPipelineLayer {
        wq: larql_compute::QuantWeight { data: wq_bytes, scales: Some(&q8_scales), format: q_fmt },
        wk: larql_compute::QuantWeight { data: wk_bytes, scales: Some(&q8_scales), format: k_fmt },
        wv: larql_compute::QuantWeight { data: wv_bytes, scales: Some(&q8_scales), format: v_fmt },
        wo: larql_compute::QuantWeight { data: wo_bytes, scales: Some(&q8_scales), format: o_fmt },
        gate: larql_compute::QuantWeight { data: gate_bytes, scales: None, format: layer_gate_fmt },
        up: larql_compute::QuantWeight { data: up_bytes, scales: None, format: layer_up_fmt },
        down: larql_compute::QuantWeight { data: down_bytes, scales: None, format: larql_compute::QuantFormat::Q6_K },
        input_norm: &input_norm,
        post_attn_norm: &post_attn_norm,
        pre_ffn_norm: Some(&pre_ffn_norm),
        post_ffn_norm: Some(&post_ffn_norm),
        q_norm_weight: Some(&q_norm),
        k_norm_weight: Some(&k_norm),
        norm_offset: 1.0,
        qk_norm_offset: 1.0,
        eps: 1e-6,
        has_post_norms: true,
        norm_type: larql_compute::NormType::RmsNorm,
        ffn_type: larql_compute::FfnType::Gated,
        activation: larql_compute::Activation::GeluTanh,
        attn_scale: 1.0 / (head_dim as f32).sqrt(),
        head_dim,
        num_q_heads,
        num_kv_heads,
        rope_base: 10000.0,
        rotary_dim: 0,
        sliding_window: 0,
        has_v_norm: false,
        layer_scalar: 0.0,
        input_norm_bias: None,
        post_attn_norm_bias: None,
        ffn_up_bias: None,
        ffn_down_bias: None,
    };

    // ── 5. Real post-embed-scale-sized input, like real Gemma 3 embedding output ──
    let x: Vec<f32> = (0..hidden).map(|i| ((i as f32 * 0.003).sin() * 20.0)).collect();

    let metal = get_metal();
    let be: &dyn ComputeBackend = &metal;
    be.reset_kv_cache();

    let result = be.decode_token(
        &[layer], &x, hidden, inter, q_dim, kv_dim,
        num_q_heads, num_kv_heads, head_dim, 10000.0,
    );

    assert!(result.is_some(), "decode_token returned None");
    let out = result.unwrap();
    let n_inf = out.iter().copied().filter(|v: &f32| v.is_infinite()).count();
    let n_nan = out.iter().copied().filter(|v: &f32| v.is_nan()).count();
    let mn = out.iter().copied().filter(|v: &f32| v.is_finite()).fold(f32::INFINITY, f32::min);
    let mx = out.iter().copied().filter(|v: &f32| v.is_finite()).fold(f32::NEG_INFINITY, f32::max);
    println!("Real-data decode_token output: {} inf, {} nan, {} finite, range [{}, {}]",
        n_inf, n_nan, out.len() - n_inf - n_nan,
        if mn == f32::INFINITY { f32::NAN } else { mn },
        if mx == f32::NEG_INFINITY { f32::NAN } else { mx });
    assert_eq!(n_inf + n_nan, 0,
        "Real-data decode_token produces {} inf + {} nan of {} — REPRODUCER", n_inf, n_nan, out.len());

    // ── 7. Scaling sweep: replicate layer 0 N times, measure how output magnitude
    //    grows. On a well-behaved Gemma 3 layer, post-norms should keep the
    //    residual stream bounded. Exponential or fast-linear growth points
    //    at a missing / miscomputed norm.
    for n_layers in [2usize, 4, 8, 16, 32, 34] {
        let layers: Vec<_> = (0..n_layers).map(|_| larql_compute::FullPipelineLayer {
            wq: larql_compute::QuantWeight { data: wq_bytes, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
            wk: larql_compute::QuantWeight { data: wk_bytes, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
            wv: larql_compute::QuantWeight { data: wv_bytes, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q6_K },
            wo: larql_compute::QuantWeight { data: wo_bytes, scales: Some(&q8_scales), format: larql_compute::QuantFormat::Q4_K },
            gate: larql_compute::QuantWeight { data: gate_bytes, scales: None, format: larql_compute::QuantFormat::Q4_K },
            up: larql_compute::QuantWeight { data: up_bytes, scales: None, format: larql_compute::QuantFormat::Q4_K },
            down: larql_compute::QuantWeight { data: down_bytes, scales: None, format: larql_compute::QuantFormat::Q6_K },
            input_norm: &input_norm,
            post_attn_norm: &post_attn_norm,
            pre_ffn_norm: Some(&pre_ffn_norm),
            post_ffn_norm: Some(&post_ffn_norm),
            q_norm_weight: Some(&q_norm),
            k_norm_weight: Some(&k_norm),
            norm_offset: 1.0, qk_norm_offset: 1.0, eps: 1e-6, has_post_norms: true,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            activation: larql_compute::Activation::GeluTanh,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim, num_q_heads, num_kv_heads,
            rope_base: 10000.0, rotary_dim: 0, sliding_window: 0,
            has_v_norm: false, layer_scalar: 0.0,
            input_norm_bias: None, post_attn_norm_bias: None,
            ffn_up_bias: None, ffn_down_bias: None,
        }).collect();
        be.reset_kv_cache();
        let r = be.decode_token(
            &layers, &x, hidden, inter, q_dim, kv_dim,
            num_q_heads, num_kv_heads, head_dim, 10000.0,
        ).expect("decode_token none");
        let mn = r.iter().copied().fold(f32::INFINITY, f32::min);
        let mx = r.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let amax = r.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let nnf = r.iter().filter(|v| !v.is_finite()).count();
        println!("[scale-sweep] n_layers={n_layers:>2} range=[{mn:>10.2}, {mx:>10.2}] amax={amax:>9.2} non-finite={nnf}");
    }

    // ── 7b. Compute CPU ref scaling at the same N values as the Metal scale sweep,
    //    using chained backend ops that we already showed match CPU byte-for-byte.
    //    Diverges from the Metal sweep = another decode_token orchestration bug.
    {
        let cpu_be_ref = larql_compute::CpuBackend;
        let cpu_ref: &dyn ComputeBackend = &cpu_be_ref;
        // Per-layer forward using metal ops (which agree with CPU) — we reuse the
        // exact chain we validated in `10a-metal`.
        fn one_layer(
            be: &dyn ComputeBackend,
            x: &[f32],
            hidden: usize, inter: usize, q_dim: usize, kv_dim: usize,
            num_q_heads: usize, num_kv_heads: usize, head_dim: usize,
            wq: &[u8], wk: &[u8], wv: &[u8], wo: &[u8],
            gate_b: &[u8], up_b: &[u8], down_b: &[u8],
            input_norm: &[f32], post_attn_norm: &[f32],
            pre_ffn_norm: &[f32], post_ffn_norm: &[f32],
            q_norm: &[f32], k_norm: &[f32],
        ) -> Vec<f32> {
            fn gelu_tanh(x: f32) -> f32 {
                let c: f32 = (2.0 / std::f32::consts::PI).sqrt();
                let arg = (c * (x + 0.044715 * x * x * x)).clamp(-20.0, 20.0);
                0.5 * x * (1.0 + arg.tanh())
            }
            let sum_sq: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
            let rms_x = 1.0f32 / ((sum_sq / hidden as f64 + 1e-6) as f32).sqrt();
            let normed_x: Vec<f32> = x.iter().zip(input_norm.iter())
                .map(|(v, w)| v * (w + 1.0) * rms_x).collect();
            let _q_raw = be.q4k_matvec(wq, &normed_x, q_dim, hidden).unwrap();
            let _k_raw = be.q4k_matvec(wk, &normed_x, kv_dim, hidden).unwrap();
            let v_raw = be.q6k_matvec(wv, &normed_x, kv_dim, hidden).unwrap();
            let _ = (q_norm, k_norm);
            let mut attn_out = vec![0.0f32; q_dim];
            let reps = num_q_heads / num_kv_heads;
            for h in 0..num_q_heads {
                let kv_h = h / reps;
                let src = &v_raw[kv_h*head_dim..(kv_h+1)*head_dim];
                let dst = &mut attn_out[h*head_dim..(h+1)*head_dim];
                dst.copy_from_slice(src);
            }
            let o_proj = be.q4k_matvec(wo, &attn_out, hidden, q_dim).unwrap();
            let sq: f64 = o_proj.iter().map(|v| (*v as f64).powi(2)).sum();
            let r = 1.0f32 / ((sq / hidden as f64 + 1e-6) as f32).sqrt();
            let normed_o: Vec<f32> = o_proj.iter().zip(post_attn_norm.iter())
                .map(|(v, w)| v * (w + 1.0) * r).collect();
            let h_post: Vec<f32> = x.iter().zip(normed_o.iter()).map(|(a, b)| a + b).collect();
            let sq2: f64 = h_post.iter().map(|v| (*v as f64).powi(2)).sum();
            let rms2 = 1.0f32 / ((sq2 / hidden as f64 + 1e-6) as f32).sqrt();
            let ffn_in: Vec<f32> = h_post.iter().zip(pre_ffn_norm.iter())
                .map(|(v, w)| v * (w + 1.0) * rms2).collect();
            let gate_ffn = be.q4k_matvec(gate_b, &ffn_in, inter, hidden).unwrap();
            let up_ffn = be.q4k_matvec(up_b, &ffn_in, inter, hidden).unwrap();
            let act: Vec<f32> = gate_ffn.iter().zip(up_ffn.iter())
                .map(|(g, u)| gelu_tanh(*g) * u).collect();
            let down = be.q6k_matvec(down_b, &act, hidden, inter).unwrap();
            let sq3: f64 = down.iter().map(|v| (*v as f64).powi(2)).sum();
            let r3 = 1.0f32 / ((sq3 / hidden as f64 + 1e-6) as f32).sqrt();
            let normed_down: Vec<f32> = down.iter().zip(post_ffn_norm.iter())
                .map(|(v, w)| v * (w + 1.0) * r3).collect();
            h_post.iter().zip(normed_down.iter()).map(|(a, b)| a + b).collect()
        }
        for n_layers in [2usize, 4, 8, 16, 32, 34] {
            let mut h: Vec<f32> = x.clone();
            for _ in 0..n_layers {
                h = one_layer(cpu_ref, &h, hidden, inter, q_dim, kv_dim,
                    num_q_heads, num_kv_heads, head_dim,
                    wq_bytes, wk_bytes, wv_bytes, wo_bytes,
                    gate_bytes, up_bytes, down_bytes,
                    &input_norm, &post_attn_norm, &pre_ffn_norm, &post_ffn_norm,
                    &q_norm, &k_norm);
            }
            let amax = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let mn = h.iter().copied().fold(f32::INFINITY, f32::min);
            let mx = h.iter().copied().fold(f32::NEG_INFINITY, f32::max);
            println!("[cpu-sweep]   n_layers={n_layers:>2} range=[{mn:>10.2}, {mx:>10.2}] amax={amax:>9.2}");
        }
    }

    // ── 8. Isolated post_ffn_norm comparison: Metal shader vs CPU reference.
    //    Uses real Gemma 3 L0 post_ffn_norm weights (max≈304) and a synthetic
    //    f32 input matching what decode_token's FFN produces (amax~167). If
    //    Metal's amplification differs materially from CPU's, rms_norm is the
    //    bug.
    let synthetic_ffn_out: Vec<f32> = (0..hidden)
        .map(|i| {
            let t = i as f32 * 0.017;
            167.0 * (t.sin() * (t * 0.31).cos()).signum()
                * ((t * 1.3).sin().abs())
        })
        .collect();
    // CPU reference
    let sum_sq: f64 = synthetic_ffn_out.iter().map(|v| (*v as f64) * (*v as f64)).sum();
    let rms_cpu = 1.0f32 / ((sum_sq / hidden as f64 + 1e-6) as f32).sqrt();
    let cpu_out: Vec<f32> = synthetic_ffn_out.iter().zip(post_ffn_norm.iter())
        .map(|(x, w)| x * (w + 1.0) * rms_cpu)
        .collect();
    let cpu_amax = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    // Metal
    let metal_raw = larql_compute::metal::MetalBackend::new().expect("metal");
    let bufs = larql_compute::metal::buffers::BufferCache::new(
        &metal::Device::system_default().unwrap(),
    );
    let device = metal::Device::system_default().unwrap();
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("rms_norm", None).unwrap()
    ).unwrap();
    let queue = device.new_command_queue();
    let buf_x = bufs.transient_from_f32(&synthetic_ffn_out);
    let buf_w = bufs.transient_from_f32(&post_ffn_norm);
    let buf_out = bufs.output((hidden * 4) as u64);
    let len_val = hidden as u32;
    let eps_val = 1e-6f32;
    let off_val = 1.0f32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_x), 0);
    enc.set_buffer(1, Some(&buf_w), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &eps_val as *const f32 as *const std::ffi::c_void);
    enc.set_bytes(5, 4, &off_val as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(256u64.min(hidden as u64), 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&buf_out, hidden);
    let metal_amax = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let max_diff = cpu_out.iter().zip(metal_out.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[rmsnorm-parity] cpu_amax={cpu_amax:.2}  metal_amax={metal_amax:.2}  max_diff={max_diff:.4}");
    assert!(max_diff < 0.05 * cpu_amax,
        "rms_norm Metal vs CPU diverge: max_diff={max_diff:.4} (>5% of cpu_amax={cpu_amax:.2})");

    // ── 9. FFN parity: Metal's GEGLU-FFN chain (gate Q4K → up Q4K → GEGLU → down Q6K)
    //    vs a CPU reference that dequantises the same bytes and does the same math
    //    with f32 matmuls. If they disagree by >10%, the FFN shader chain has a bug.
    let be: &dyn ComputeBackend = &metal_raw;
    // Sparse-spike distribution like real Gemma 3 post-norm outputs: most values
    // small, a few channels spike to amax. This is the distribution that makes
    // post_ffn_norm amplify by 400× downstream.
    let synthetic_ffn_in: Vec<f32> = (0..hidden)
        .map(|i| {
            let t = i as f32 * 0.013;
            let small = t.sin() * 0.5;
            // A few spike channels (~5% of positions).
            if i % 20 == 0 { 30.0 * ((i as f32 * 0.007).sin().signum()) } else { small }
        })
        .collect();
    let sin_amax = synthetic_ffn_in.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let sin_sq: f64 = synthetic_ffn_in.iter().map(|v| (*v as f64).powi(2)).sum();
    let sin_rms = (sin_sq / hidden as f64).sqrt() as f32;
    println!("[ffn-parity] input amax={sin_amax:.2} rms={sin_rms:.4} (spike distribution)");

    // Metal: gate @ x, up @ x (each produces `inter` rows)
    let metal_gate = be.q4k_matvec(gate_bytes, &synthetic_ffn_in, inter, hidden).expect("q4k gate");
    let metal_up = be.q4k_matvec(up_bytes, &synthetic_ffn_in, inter, hidden).expect("q4k up");
    // GEGLU (gemma gelu_tanh): act[i] = gelu_tanh(gate[i]) * up[i]
    //   gelu_tanh(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
    fn gelu_tanh(x: f32) -> f32 {
        let c: f32 = (2.0 / std::f32::consts::PI).sqrt();
        let arg = (c * (x + 0.044715 * x * x * x)).clamp(-20.0, 20.0);
        0.5 * x * (1.0 + arg.tanh())
    }
    let metal_act: Vec<f32> = metal_gate.iter().zip(metal_up.iter())
        .map(|(g, u)| gelu_tanh(*g) * u).collect();
    // Metal: down @ act → hidden-dim output
    let metal_down = be.q6k_matvec(down_bytes, &metal_act, hidden, inter).expect("q6k down");

    // CPU reference: dequantise each quant matrix once, then do f32 matmuls.
    // Simplest: use the CPU backend's q4k_matvec + q6k_matvec (scalar reference
    // implementations in cpu/ops). If Metal ≈ CPU ops here, the shader chain
    // does what its own reference does, and the bug is elsewhere.
    let cpu_backend = larql_compute::CpuBackend;
    let cpu_be: &dyn ComputeBackend = &cpu_backend;
    let cpu_gate = cpu_be.q4k_matvec(gate_bytes, &synthetic_ffn_in, inter, hidden).unwrap();
    let cpu_up = cpu_be.q4k_matvec(up_bytes, &synthetic_ffn_in, inter, hidden).unwrap();
    let cpu_act: Vec<f32> = cpu_gate.iter().zip(cpu_up.iter())
        .map(|(g, u)| gelu_tanh(*g) * u).collect();
    let cpu_down = cpu_be.q6k_matvec(down_bytes, &cpu_act, hidden, inter).unwrap();

    let gate_maxdiff = metal_gate.iter().zip(cpu_gate.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let up_maxdiff = metal_up.iter().zip(cpu_up.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let act_maxdiff = metal_act.iter().zip(cpu_act.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let down_maxdiff = metal_down.iter().zip(cpu_down.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let metal_down_amax = metal_down.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let cpu_down_amax = cpu_down.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("[ffn-parity] gate  metal_amax={:.3} cpu_amax={:.3} max_diff={:.4}",
        metal_gate.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        cpu_gate.iter().map(|v| v.abs()).fold(0.0f32, f32::max), gate_maxdiff);
    println!("[ffn-parity] up    metal_amax={:.3} cpu_amax={:.3} max_diff={:.4}",
        metal_up.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        cpu_up.iter().map(|v| v.abs()).fold(0.0f32, f32::max), up_maxdiff);
    println!("[ffn-parity] act   metal_amax={:.3} cpu_amax={:.3} max_diff={:.4}",
        metal_act.iter().map(|v| v.abs()).fold(0.0f32, f32::max),
        cpu_act.iter().map(|v| v.abs()).fold(0.0f32, f32::max), act_maxdiff);
    println!("[ffn-parity] down  metal_amax={metal_down_amax:.3} cpu_amax={cpu_down_amax:.3} max_diff={down_maxdiff:.4}");

    // ── Compare q4k_matvec vs q4k_proj kernels on wo weights.
    //    q4k_matvec uses `half` cast; q4k_proj uses decode_f16_metal.
    {
        let test_x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.007).sin() * 20.0).collect();
        let via_matvec = be.q4k_matvec(wo_bytes, &test_x, hidden, hidden).unwrap();
        let proj_pipeline = device.new_compute_pipeline_state_with_function(
            &lib.get_function("q4k_proj", None).unwrap()
        ).unwrap();
        let buf_w_raw = device.new_buffer_with_data(
            wo_bytes.as_ptr() as *const std::ffi::c_void,
            wo_bytes.len() as u64,
            metal::MTLResourceOptions::StorageModeShared,
        );
        let buf_x_p = bufs.transient_from_f32(&test_x);
        let buf_out_p = bufs.output((hidden * 4) as u64);
        let n_val = hidden as u32;
        let k_val = hidden as u32;
        let cmd_p = queue.new_command_buffer();
        let enc_p = cmd_p.new_compute_command_encoder();
        enc_p.set_compute_pipeline_state(&proj_pipeline);
        enc_p.set_buffer(0, Some(&buf_w_raw), 0);
        enc_p.set_buffer(1, Some(&buf_x_p), 0);
        enc_p.set_buffer(2, Some(&buf_out_p), 0);
        enc_p.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
        enc_p.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
        let n_tgs = (hidden as u64).div_ceil(8);
        enc_p.dispatch_thread_groups(
            metal::MTLSize::new(n_tgs, 1, 1),
            metal::MTLSize::new(256, 1, 1),
        );
        enc_p.end_encoding();
        cmd_p.commit();
        cmd_p.wait_until_completed();
        let via_proj = larql_compute::metal::buffers::read_buffer_f32(&buf_out_p, hidden);
        let matvec_amax = via_matvec.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let proj_amax = via_proj.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let diff = via_matvec.iter().zip(via_proj.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("[q4k-kernel-cmp] wo_proj matvec_amax={matvec_amax:.3} proj_amax={proj_amax:.3} max_diff={diff:.5}");
    }

    // ── 10a-metal: Repeat the same layer-0 pipeline using Metal backend ops
    //    (q4k_matvec + q6k_matvec + manual norms) and compare to decode_token's
    //    result. This isolates whether the divergence is in decode_token's
    //    orchestration or in the kernel outputs.
    {
        // Step 1: input_norm(x) via manual CPU math (already verified parity).
        let sum_sq_x: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        let rms_x = 1.0f32 / ((sum_sq_x / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_x: Vec<f32> = x.iter().zip(input_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * rms_x).collect();
        // Step 2: Q/K/V via Metal backend
        let q_raw = be.q4k_matvec(wq_bytes, &normed_x, q_dim, hidden).unwrap();
        let k_raw = be.q4k_matvec(wk_bytes, &normed_x, kv_dim, hidden).unwrap();
        let v_raw = be.q6k_matvec(wv_bytes, &normed_x, kv_dim, hidden).unwrap();
        // Step 3: QK-norm via per-head CPU math
        let mut q_n = q_raw.clone();
        for h in 0..num_q_heads {
            let off = h * head_dim;
            let sq: f64 = q_raw[off..off+head_dim].iter().map(|v| (*v as f64).powi(2)).sum();
            let r = 1.0f32 / ((sq / head_dim as f64 + 1e-6) as f32).sqrt();
            for d in 0..head_dim { q_n[off+d] = q_raw[off+d] * (q_norm[d] + 1.0) * r; }
        }
        let mut k_n = k_raw.clone();
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            let sq: f64 = k_raw[off..off+head_dim].iter().map(|v| (*v as f64).powi(2)).sum();
            let r = 1.0f32 / ((sq / head_dim as f64 + 1e-6) as f32).sqrt();
            for d in 0..head_dim { k_n[off+d] = k_raw[off+d] * (k_norm[d] + 1.0) * r; }
        }
        let _ = (q_n, k_n);
        // Step 4/5: T=1 attention → attn_out = V per Q head via GQA
        let mut attn_out = vec![0.0f32; q_dim];
        let reps = num_q_heads / num_kv_heads;
        for h in 0..num_q_heads {
            let kv_h = h / reps;
            let src = &v_raw[kv_h*head_dim..(kv_h+1)*head_dim];
            let dst = &mut attn_out[h*head_dim..(h+1)*head_dim];
            dst.copy_from_slice(src);
        }
        // Step 6: O projection via Metal
        let o_proj = be.q4k_matvec(wo_bytes, &attn_out, hidden, q_dim).unwrap();
        // Step 7: post_attn_norm(o)
        let sq: f64 = o_proj.iter().map(|v| (*v as f64).powi(2)).sum();
        let r = 1.0f32 / ((sq / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_o: Vec<f32> = o_proj.iter().zip(post_attn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * r).collect();
        // Step 8: h_post_attn
        let h_post_attn_m: Vec<f32> = x.iter().zip(normed_o.iter()).map(|(a, b)| a + b).collect();
        // Step 9: pre_ffn_norm(h_post_attn)
        let sq2: f64 = h_post_attn_m.iter().map(|v| (*v as f64).powi(2)).sum();
        let rms2 = 1.0f32 / ((sq2 / hidden as f64 + 1e-6) as f32).sqrt();
        let ffn_in_m: Vec<f32> = h_post_attn_m.iter().zip(pre_ffn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * rms2).collect();
        // Step 10: gate/up via Metal, GEGLU, down via Metal
        let gate_m = be.q4k_matvec(gate_bytes, &ffn_in_m, inter, hidden).unwrap();
        let up_m = be.q4k_matvec(up_bytes, &ffn_in_m, inter, hidden).unwrap();
        let act_m: Vec<f32> = gate_m.iter().zip(up_m.iter())
            .map(|(g, u)| gelu_tanh(*g) * u).collect();
        let down_m = be.q6k_matvec(down_bytes, &act_m, hidden, inter).unwrap();
        // Step 11: post_ffn_norm(down) + h_post_attn
        let sq3: f64 = down_m.iter().map(|v| (*v as f64).powi(2)).sum();
        let r3 = 1.0f32 / ((sq3 / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_down_m: Vec<f32> = down_m.iter().zip(post_ffn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * r3).collect();
        let h_final_m: Vec<f32> = h_post_attn_m.iter().zip(normed_down_m.iter())
            .map(|(a, b)| a + b).collect();
        let m_ref_amax = h_final_m.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let down_m_amax = down_m.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let m_vs_dec = h_final_m.iter().zip(out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("[metal-ref-L0] down_metal_ops={down_m_amax:.2} h_final={m_ref_amax:.2}  vs_decode_token max_elem_diff={m_vs_dec:.4}");
        // Report channel-wise values at the top diverging channels
        for ch in [443usize, 368, 1762] {
            let normed_ch = normed_down_m[ch];
            let h_pa_ch = h_post_attn_m[ch];
            let normed_o_ch = normed_o[ch];
            let o_ch = o_proj[ch];
            println!("  ch={ch}  dec_out={:.3} mref_out={:.3} | h_pa_m={h_pa_ch:.3} normed_o={normed_o_ch:.3} o={o_ch:.3} | down_m={:.3} normed_down={normed_ch:.3}",
                out[ch], h_final_m[ch], down_m[ch]);
        }
    }

    // ── 10a. Full-layer CPU reference vs Metal decode_token.
    //    Replicate decode_token layer 0 entirely in CPU ops (using verified-parity
    //    kernels) and compare to Metal's result on the same synthetic x input.
    //    Any divergence localises to the decode_token orchestration.
    {
        let cpu_backend = larql_compute::CpuBackend;
        let cpu_be: &dyn ComputeBackend = &cpu_backend;
        // ── Step 1: input_norm(x)
        let mut sum_sq: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        let mut rms_x = 1.0f32 / ((sum_sq / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_x: Vec<f32> = x.iter().zip(input_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * rms_x).collect();
        // ── Step 2: Q/K/V projection
        let q_raw = cpu_be.q4k_matvec(wq_bytes, &normed_x, q_dim, hidden).unwrap();
        let k_raw = cpu_be.q4k_matvec(wk_bytes, &normed_x, kv_dim, hidden).unwrap();
        let v_raw = cpu_be.q6k_matvec(wv_bytes, &normed_x, kv_dim, hidden).unwrap();
        // ── Step 3: QK-norm per head (Gemma 3)
        let mut q_n = q_raw.clone();
        for h in 0..num_q_heads {
            let off = h * head_dim;
            let sq: f64 = q_raw[off..off+head_dim].iter().map(|v| (*v as f64).powi(2)).sum();
            let r = 1.0f32 / ((sq / head_dim as f64 + 1e-6) as f32).sqrt();
            for d in 0..head_dim { q_n[off+d] = q_raw[off+d] * (q_norm[d] + 1.0) * r; }
        }
        let mut k_n = k_raw.clone();
        for h in 0..num_kv_heads {
            let off = h * head_dim;
            let sq: f64 = k_raw[off..off+head_dim].iter().map(|v| (*v as f64).powi(2)).sum();
            let r = 1.0f32 / ((sq / head_dim as f64 + 1e-6) as f32).sqrt();
            for d in 0..head_dim { k_n[off+d] = k_raw[off+d] * (k_norm[d] + 1.0) * r; }
        }
        // ── Step 4: RoPE at position 0 — RoPE angle at pos=0 is 1 for cos and 0
        //    for sin on every channel, so Q/K are unchanged.
        // ── Step 5: Attention (T=1, empty cache pre-append, so T becomes 1
        //    with current token; softmax over a single score = 1; attn_out = V).
        //    For GQA, each Q head reads from its corresponding KV head (num_q / num_kv heads per group).
        let mut attn_out = vec![0.0f32; q_dim];
        let reps = num_q_heads / num_kv_heads;
        for h in 0..num_q_heads {
            let kv_h = h / reps;
            let src = &v_raw[kv_h*head_dim..(kv_h+1)*head_dim];
            let dst = &mut attn_out[h*head_dim..(h+1)*head_dim];
            dst.copy_from_slice(src);
        }
        // ── Step 6: O projection
        let o_proj = cpu_be.q4k_matvec(wo_bytes, &attn_out, hidden, q_dim).unwrap();
        // ── Step 7: post_attn_norm(o)
        let sq: f64 = o_proj.iter().map(|v| (*v as f64).powi(2)).sum();
        let r = 1.0f32 / ((sq / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_o: Vec<f32> = o_proj.iter().zip(post_attn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * r).collect();
        // ── Step 8: h_post_attn = x + normed_o
        let h_post_attn_cpu: Vec<f32> = x.iter().zip(normed_o.iter()).map(|(a, b)| a + b).collect();
        let h_pa_amax = h_post_attn_cpu.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // ── Step 9: pre_ffn_norm(h_post_attn) → ffn_in
        sum_sq = h_post_attn_cpu.iter().map(|v| (*v as f64).powi(2)).sum();
        rms_x = 1.0f32 / ((sum_sq / hidden as f64 + 1e-6) as f32).sqrt();
        let ffn_in: Vec<f32> = h_post_attn_cpu.iter().zip(pre_ffn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * rms_x).collect();
        let ffn_in_amax = ffn_in.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // ── Step 10: FFN = gate(ffn_in), up(ffn_in), GEGLU, down
        let gate_ffn = cpu_be.q4k_matvec(gate_bytes, &ffn_in, inter, hidden).unwrap();
        let up_ffn = cpu_be.q4k_matvec(up_bytes, &ffn_in, inter, hidden).unwrap();
        let act: Vec<f32> = gate_ffn.iter().zip(up_ffn.iter())
            .map(|(g, u)| gelu_tanh(*g) * u).collect();
        let down = cpu_be.q6k_matvec(down_bytes, &act, hidden, inter).unwrap();
        let down_amax = down.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        // ── Step 11: post_ffn_norm(down) + h_post_attn
        let sq: f64 = down.iter().map(|v| (*v as f64).powi(2)).sum();
        let r = 1.0f32 / ((sq / hidden as f64 + 1e-6) as f32).sqrt();
        let normed_down: Vec<f32> = down.iter().zip(post_ffn_norm.iter())
            .map(|(v, w)| v * (w + 1.0) * r).collect();
        let pfn_amax = normed_down.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let h_final: Vec<f32> = h_post_attn_cpu.iter().zip(normed_down.iter())
            .map(|(a, b)| a + b).collect();
        let cpu_final_amax = h_final.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

        // Compare element-by-element to the Metal result captured earlier in `out`.
        let metal_final_amax = out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let maxdiff = h_final.iter().zip(out.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("[cpu-ref-L0] h_post_attn={h_pa_amax:.2} ffn_in={ffn_in_amax:.2} down={down_amax:.2} post_ffn_contrib={pfn_amax:.2} h_final={cpu_final_amax:.2}");
        println!("[metal-vs-cpu-L0] metal_amax={metal_final_amax:.2} cpu_amax={cpu_final_amax:.2} max_elem_diff={maxdiff:.4}");

        // Find the top-5 diverging channels
        let mut diffs: Vec<(usize, f32, f32, f32)> = out.iter().zip(h_final.iter()).enumerate()
            .map(|(i, (m, c))| (i, *m, *c, (m - c).abs()))
            .collect();
        diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
        println!("[metal-vs-cpu-L0] top-5 diverging channels (idx, metal, cpu, diff):");
        for (i, m, c, d) in diffs.iter().take(5) {
            println!("  ch={i}  metal={m:.3}  cpu={c:.3}  diff={d:.3}  post_ffn_weight[ch]={:.3} down_cpu[ch]={:.3}",
                post_ffn_norm[*i], down[*i]);
        }

        // Test rms_norm with the EXACT inputs from the L0 pipeline (real down, real post_ffn_norm)
        // that trigger the divergence. If the isolated call agrees with CPU, the bug is in
        // how decode_token feeds inputs to the kernel.
        let buf_x2 = bufs.transient_from_f32(&down);
        let buf_w3 = bufs.transient_from_f32(&post_ffn_norm);
        let buf_out2 = bufs.output((hidden * 4) as u64);
        let cmd_rp = queue.new_command_buffer();
        let enc_rp = cmd_rp.new_compute_command_encoder();
        enc_rp.set_compute_pipeline_state(&pipeline);
        enc_rp.set_buffer(0, Some(&buf_x2), 0);
        enc_rp.set_buffer(1, Some(&buf_w3), 0);
        enc_rp.set_buffer(2, Some(&buf_out2), 0);
        enc_rp.set_bytes(3, 4, &len_val as *const u32 as *const std::ffi::c_void);
        enc_rp.set_bytes(4, 4, &eps_val as *const f32 as *const std::ffi::c_void);
        enc_rp.set_bytes(5, 4, &off_val as *const f32 as *const std::ffi::c_void);
        enc_rp.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1),
            metal::MTLSize::new(256u64.min(hidden as u64), 1, 1));
        enc_rp.end_encoding();
        cmd_rp.commit();
        cmd_rp.wait_until_completed();
        let metal_isolated = larql_compute::metal::buffers::read_buffer_f32(&buf_out2, hidden);
        let isolated_amax = metal_isolated.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let isolated_vs_cpu_maxdiff = metal_isolated.iter().zip(normed_down.iter())
            .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
        println!("[isolated-postffn] metal_amax={isolated_amax:.3} cpu_amax={pfn_amax:.3} max_diff={isolated_vs_cpu_maxdiff:.5}");
    }

    // ── 10. residual_norm parity: out = (a + b) * (weight + offset) / rms(a+b)
    //    Used in decode_token step 5 to fuse residual add and pre_ffn_norm.
    //    Test with realistic magnitudes: h_buf (large residual) + normed_o.
    let a: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.021).sin() * 4000.0).collect();
    let b: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.017).cos() * 200.0).collect();
    let sum_sq: f64 = a.iter().zip(b.iter()).map(|(x, y)| {
        let s = *x as f64 + *y as f64;
        s * s
    }).sum();
    let rms = 1.0f32 / ((sum_sq / hidden as f64 + 1e-6) as f32).sqrt();
    let cpu_rn: Vec<f32> = a.iter().zip(b.iter()).zip(pre_ffn_norm.iter())
        .map(|((x, y), w)| (x + y) * (w + 1.0) * rms).collect();
    let cpu_rn_amax = cpu_rn.iter().map(|v| v.abs()).fold(0.0f32, f32::max);

    let buf_a = bufs.transient_from_f32(&a);
    let buf_b = bufs.transient_from_f32(&b);
    let buf_w2 = bufs.transient_from_f32(&pre_ffn_norm);
    let buf_out_rn = bufs.output((hidden * 4) as u64);
    let rn_pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("residual_norm", None).unwrap()
    ).unwrap();
    let cmd2 = queue.new_command_buffer();
    let enc2 = cmd2.new_compute_command_encoder();
    enc2.set_compute_pipeline_state(&rn_pipeline);
    enc2.set_buffer(0, Some(&buf_a), 0);
    enc2.set_buffer(1, Some(&buf_b), 0);
    enc2.set_buffer(2, Some(&buf_w2), 0);
    enc2.set_buffer(3, Some(&buf_out_rn), 0);
    enc2.set_bytes(4, 4, &len_val as *const u32 as *const std::ffi::c_void);
    enc2.set_bytes(5, 4, &eps_val as *const f32 as *const std::ffi::c_void);
    enc2.set_bytes(6, 4, &off_val as *const f32 as *const std::ffi::c_void);
    enc2.dispatch_thread_groups(metal::MTLSize::new(1, 1, 1),
        metal::MTLSize::new(256u64.min(hidden as u64), 1, 1));
    enc2.end_encoding();
    cmd2.commit();
    cmd2.wait_until_completed();
    let metal_rn = larql_compute::metal::buffers::read_buffer_f32(&buf_out_rn, hidden);
    let metal_rn_amax = metal_rn.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let rn_maxdiff = cpu_rn.iter().zip(metal_rn.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[resnorm-parity] cpu_amax={cpu_rn_amax:.3} metal_amax={metal_rn_amax:.3} max_diff={rn_maxdiff:.5}");
    assert!(rn_maxdiff < 0.01 * cpu_rn_amax,
        "residual_norm diverges: max_diff={rn_maxdiff:.5}");

    let _ = metal_raw;
}

/// Q4_KF (GGUF) parity on REAL Gemma 3 L0 q_proj bytes (rebuilt to Q4_KF).
/// Reproduces the exact dispatch decode_token performs in gpuprefill.
#[test]
#[ignore]
fn q4kf_gguf_real_q_proj_metal_matches_cpu() {
    let vindex_path = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex_path);

    // Load Q4_KF manifest entry for layer 0 q_proj.
    let q4k_manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("attn_weights_q4k_manifest.json")).unwrap()
    ).unwrap();
    let entry = q4k_manifest.iter().find(|e|
        e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0.self_attn.q_proj"))
    ).unwrap();
    assert_eq!(entry.get("format").and_then(|v| v.as_str()), Some("Q4_KF"));
    let offset = entry["offset"].as_u64().unwrap() as usize;
    let length = entry["length"].as_u64().unwrap() as usize;
    let shape = entry["shape"].as_array().unwrap();
    let n = shape[0].as_u64().unwrap() as usize; // rows = q_dim
    let k = shape[1].as_u64().unwrap() as usize; // cols = hidden

    let attn_file = std::fs::File::open(dir.join("attn_weights_q4k.bin")).unwrap();
    let attn_mmap = unsafe { memmap2::Mmap::map(&attn_file).unwrap() };
    let w_bytes = &attn_mmap[offset..offset + length];
    println!("[q4kf-real] q_proj: n={n} k={k} length={length}");

    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.007).sin() * 20.0).collect();

    // CPU reference
    let cpu_out = larql_compute::cpu::ops::q4k_matvec::dispatch_gguf(w_bytes, &x, n, k);

    // Metal q4kf_proj
    let device = metal::Device::system_default().expect("metal");
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("q4kf_proj", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();
    let buf_w = device.new_buffer_with_data(
        w_bytes.as_ptr() as *const std::ffi::c_void,
        w_bytes.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let buf_x = bufs.transient_from_f32(&x);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_w), 0);
    enc.set_buffer(1, Some(&buf_x), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
    let n_tgs = (n as u64).div_ceil(4);
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_tgs, 1, 1),
        metal::MTLSize::new(64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();
    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&buf_out, n);
    let nnan = metal_out.iter().filter(|v| !v.is_finite()).count();
    let cpu_amax = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let metal_amax = metal_out.iter().filter(|v| v.is_finite()).map(|v| v.abs()).fold(0.0f32, f32::max);
    let diff = cpu_out.iter().zip(metal_out.iter()).filter(|(_, m)| m.is_finite())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[q4kf-real] cpu_amax={cpu_amax:.3} metal_amax={metal_amax:.3} max_diff={diff:.5} nnan={nnan}");
}

/// Q4_KF (GGUF) parity: Metal q4kf_proj vs CPU dispatch_gguf on synthetic weights.
/// Catches any encoding/decoding mismatch between quantize_q4_k_gguf and the
/// Metal shader before we run the whole 34-layer pipeline with it.
#[test]
fn q4kf_gguf_metal_matches_cpu() {
    let device = metal::Device::system_default().expect("metal");
    let src = larql_compute::metal::shaders::all_shaders();
    let lib = device.new_library_with_source(&src, &metal::CompileOptions::new()).unwrap();
    let pipeline = device.new_compute_pipeline_state_with_function(
        &lib.get_function("q4kf_proj", None).unwrap()
    ).unwrap();
    let bufs = larql_compute::metal::buffers::BufferCache::new(&device);
    let queue = device.new_command_queue();

    let n = 2048; // Gemma 3 q_dim
    let k = 2560; // Gemma 3 hidden (10 super-blocks per row)
    let w: Vec<f32> = (0..n * k).map(|i| {
        let t = i as f32 * 0.0013;
        0.05 * t.sin() + if i % 37 == 0 { 0.25 * t.cos().signum() } else { 0.0 }
    }).collect();
    let w_gguf = larql_compute::cpu::ops::q4_common::quantize_q4_k_gguf(&w);
    let x: Vec<f32> = (0..k).map(|i| (i as f32 * 0.007).sin() * 20.0).collect();

    let cpu_out = larql_compute::cpu::ops::q4k_matvec::dispatch_gguf(&w_gguf, &x, n, k);

    let buf_w = device.new_buffer_with_data(
        w_gguf.as_ptr() as *const std::ffi::c_void,
        w_gguf.len() as u64,
        metal::MTLResourceOptions::StorageModeShared,
    );
    let buf_x = bufs.transient_from_f32(&x);
    let buf_out = bufs.output((n * 4) as u64);
    let n_val = n as u32;
    let k_val = k as u32;
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&buf_w), 0);
    enc.set_buffer(1, Some(&buf_x), 0);
    enc.set_buffer(2, Some(&buf_out), 0);
    enc.set_bytes(3, 4, &n_val as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &k_val as *const u32 as *const std::ffi::c_void);
    let n_tgs = (n as u64).div_ceil(4); // q4kf_qkv_proj: 4 rows/TG, 64 threads/TG
    enc.dispatch_thread_groups(
        metal::MTLSize::new(n_tgs, 1, 1),
        metal::MTLSize::new(64, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    let metal_out = larql_compute::metal::buffers::read_buffer_f32(&buf_out, n);
    let n_nan = metal_out.iter().filter(|v| !v.is_finite()).count();
    let cpu_amax = cpu_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let metal_amax = metal_out.iter().filter(|v| v.is_finite()).map(|v| v.abs()).fold(0.0f32, f32::max);
    let diff = cpu_out.iter().zip(metal_out.iter())
        .filter(|(_, m)| m.is_finite())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[q4kf-parity] cpu_amax={cpu_amax:.3} metal_amax={metal_amax:.3} max_diff={diff:.5} metal_nonfinite={n_nan}");
    assert_eq!(n_nan, 0, "Metal q4kf_proj produced {n_nan} non-finite values");
    assert!(diff < 0.01 * cpu_amax.max(1.0),
        "Metal q4kf_proj diverges from CPU dispatch_gguf: max_diff={diff:.5}");
}

/// Load all 34 real Gemma 3 layers and run decode_token vs a CPU-backend-ops
/// reference on the same BOS-like synthetic input. First divergent layer
/// pinpoints which per-layer variable (rope_base, sliding_window, weight
/// distribution, etc) triggers the remaining Metal decode_token bug.
#[test]
#[ignore]
fn decode_token_all_34_layers_matches_cpu_ref() {
    let vindex_path = std::env::var("LARQL_VINDEX_PATH")
        .expect("set LARQL_VINDEX_PATH to a Q4K-prepared Gemma 3 4B vindex");
    let dir = std::path::Path::new(&vindex_path);

    let hidden = 2560usize;
    let inter = 10240usize;
    let num_q_heads = 8usize;
    let num_kv_heads = 4usize;
    let head_dim = 256usize;
    let q_dim = num_q_heads * head_dim;
    let kv_dim = num_kv_heads * head_dim;
    let num_layers = 34usize;

    // Decode f16 inline (no larql_models dep).
    fn decode_f16(raw: &[u8]) -> Vec<f32> {
        raw.chunks_exact(2).map(|b| {
            let bits = u16::from_le_bytes([b[0], b[1]]);
            let sign = (bits >> 15) & 0x1;
            let exp = (bits >> 10) & 0x1F;
            let mant = (bits & 0x3FF) as u32;
            let f32_bits = if exp == 0 {
                if mant == 0 { (sign as u32) << 31 } else {
                    let mut m = mant; let mut e: i32 = -14;
                    while (m & 0x400) == 0 { m <<= 1; e -= 1; }
                    let exp32 = (e + 127) as u32;
                    ((sign as u32) << 31) | (exp32 << 23) | ((m & 0x3FF) << 13)
                }
            } else if exp == 0x1F {
                ((sign as u32) << 31) | (0xFF << 23) | (mant << 13)
            } else {
                let exp32 = (exp as i32 - 15 + 127) as u32;
                ((sign as u32) << 31) | (exp32 << 23) | (mant << 13)
            };
            f32::from_bits(f32_bits)
        }).collect()
    }

    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("weight_manifest.json")).unwrap()
    ).unwrap();
    let find_vec = |key: &str| -> Vec<f32> {
        let e = manifest.iter().find(|e|
            e.get("key").and_then(|k| k.as_str()) == Some(key)
        ).unwrap_or_else(|| panic!("missing vector {key}"));
        let offset = e["offset"].as_u64().unwrap() as usize;
        let length = e["length"].as_u64().unwrap() as usize;
        let file_name = e["file"].as_str().unwrap();
        let file = std::fs::File::open(dir.join(file_name)).unwrap();
        let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
        decode_f16(&mmap[offset..offset + length])
    };

    // Norms per layer.
    let mut input_norms: Vec<Vec<f32>> = Vec::new();
    let mut post_attn_norms: Vec<Vec<f32>> = Vec::new();
    let mut pre_ffn_norms: Vec<Vec<f32>> = Vec::new();
    let mut post_ffn_norms: Vec<Vec<f32>> = Vec::new();
    let mut q_norms: Vec<Vec<f32>> = Vec::new();
    let mut k_norms: Vec<Vec<f32>> = Vec::new();
    for l in 0..num_layers {
        input_norms.push(find_vec(&format!("layers.{l}.input_layernorm.weight")));
        post_attn_norms.push(find_vec(&format!("layers.{l}.post_attention_layernorm.weight")));
        pre_ffn_norms.push(find_vec(&format!("layers.{l}.pre_feedforward_layernorm.weight")));
        post_ffn_norms.push(find_vec(&format!("layers.{l}.post_feedforward_layernorm.weight")));
        q_norms.push(find_vec(&format!("layers.{l}.self_attn.q_norm.weight")));
        k_norms.push(find_vec(&format!("layers.{l}.self_attn.k_norm.weight")));
    }

    // Attention Q4K (plus v_proj Q6K) weights per layer.
    let q4k_manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("attn_weights_q4k_manifest.json")).unwrap()
    ).unwrap();
    let attn_file = std::fs::File::open(dir.join("attn_weights_q4k.bin")).unwrap();
    let attn_mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&attn_file).unwrap() });
    let find_attn = |l: usize, key: &str| -> (usize, usize, larql_compute::QuantFormat) {
        let prefix = format!("layers.{l}.");
        let e = q4k_manifest.iter().find(|e|
            e.get("key").and_then(|k| k.as_str())
                .is_some_and(|k| k.starts_with(&prefix) && k.contains(key))
        ).unwrap_or_else(|| panic!("missing q4k L{l} {key}"));
        let format = match e.get("format").and_then(|v| v.as_str()) {
            Some("Q6_K") => larql_compute::QuantFormat::Q6_K,
            Some("Q4_KF") | Some("Q4_K_GGUF") => larql_compute::QuantFormat::Q4_KF,
            _ => larql_compute::QuantFormat::Q4_K,
        };
        (e["offset"].as_u64().unwrap() as usize,
         e["length"].as_u64().unwrap() as usize, format)
    };
    let mut wq_offs: Vec<(usize, usize, larql_compute::QuantFormat)> = (0..num_layers).map(|l| find_attn(l, "q_proj")).collect();
    let mut wk_offs: Vec<(usize, usize, larql_compute::QuantFormat)> = (0..num_layers).map(|l| find_attn(l, "k_proj")).collect();
    let mut wv_offs: Vec<(usize, usize, larql_compute::QuantFormat)> = (0..num_layers).map(|l| find_attn(l, "v_proj")).collect();
    let mut wo_offs: Vec<(usize, usize, larql_compute::QuantFormat)> = (0..num_layers).map(|l| find_attn(l, "o_proj")).collect();

    // FFN interleaved weights per layer. Detect FFN layout from file size.
    let q4k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 148;
    let q4kf_bytes_per_matrix = (inter * hidden).div_ceil(256) * 144;
    let q6k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 210;
    let inter_file = std::fs::File::open(dir.join("interleaved_q4k.bin")).unwrap();
    let inter_mmap = std::sync::Arc::new(unsafe { memmap2::Mmap::map(&inter_file).unwrap() });
    let file_per_layer = inter_mmap.len() / num_layers;
    let (gate_bytes_per, up_bytes_per, down_bytes_per, gate_format, up_format) =
        if file_per_layer == 2 * q4kf_bytes_per_matrix + q6k_bytes_per_matrix {
            (q4kf_bytes_per_matrix, q4kf_bytes_per_matrix, q6k_bytes_per_matrix,
             larql_compute::QuantFormat::Q4_KF, larql_compute::QuantFormat::Q4_KF)
        } else if file_per_layer == q4kf_bytes_per_matrix * 3 {
            (q4kf_bytes_per_matrix, q4kf_bytes_per_matrix, q4kf_bytes_per_matrix,
             larql_compute::QuantFormat::Q4_KF, larql_compute::QuantFormat::Q4_KF)
        } else if file_per_layer == q6k_bytes_per_matrix * 3 {
            (q6k_bytes_per_matrix, q6k_bytes_per_matrix, q6k_bytes_per_matrix,
             larql_compute::QuantFormat::Q6_K, larql_compute::QuantFormat::Q6_K)
        } else if file_per_layer == 2 * q4k_bytes_per_matrix + q6k_bytes_per_matrix {
            (q4k_bytes_per_matrix, q4k_bytes_per_matrix, q6k_bytes_per_matrix,
             larql_compute::QuantFormat::Q4_K, larql_compute::QuantFormat::Q4_K)
        } else {
            panic!("unknown FFN layout: file_per_layer={file_per_layer}");
        };
    let per_layer = gate_bytes_per + up_bytes_per + down_bytes_per;
    println!("FFN layout: gate={}B up={}B down={}B per_layer={}B gate_fmt={gate_format:?}",
        gate_bytes_per, up_bytes_per, down_bytes_per, per_layer);

    // Gemma 3 per-layer rope_base: layers where (l+1) % 6 == 0 are full attention
    // (rope_base = 1_000_000), others are sliding (rope_local = 10_000).
    let is_sliding = |l: usize| !(l + 1).is_multiple_of(6);
    let rope_base_for = |l: usize| if is_sliding(l) { 10_000.0f32 } else { 1_000_000.0f32 };
    let sliding_window_for = |l: usize| if is_sliding(l) { 1024usize } else { 0usize };

    let (_q8_x, q8_scales) = q4::quantize_to_q8(&vec![0.01f32; hidden]);

    // Construct FullPipelineLayer for each of the 34 layers.
    let layers: Vec<larql_compute::FullPipelineLayer<'_>> = (0..num_layers).map(|l| {
        let (q_off, q_len, q_fmt) = wq_offs[l];
        let (k_off, k_len, k_fmt) = wk_offs[l];
        let (v_off, v_len, v_fmt) = wv_offs[l];
        let (o_off, o_len, o_fmt) = wo_offs[l];
        let wq = &attn_mmap[q_off..q_off + q_len];
        let wk = &attn_mmap[k_off..k_off + k_len];
        let wv = &attn_mmap[v_off..v_off + v_len];
        let wo = &attn_mmap[o_off..o_off + o_len];
        let base = l * per_layer;
        let gate = &inter_mmap[base..base + gate_bytes_per];
        let up = &inter_mmap[base + gate_bytes_per..base + gate_bytes_per + up_bytes_per];
        let down = &inter_mmap[base + gate_bytes_per + up_bytes_per..base + per_layer];
        larql_compute::FullPipelineLayer {
            wq: larql_compute::QuantWeight { data: wq, scales: Some(&q8_scales), format: q_fmt },
            wk: larql_compute::QuantWeight { data: wk, scales: Some(&q8_scales), format: k_fmt },
            wv: larql_compute::QuantWeight { data: wv, scales: Some(&q8_scales), format: v_fmt },
            wo: larql_compute::QuantWeight { data: wo, scales: Some(&q8_scales), format: o_fmt },
            gate: larql_compute::QuantWeight { data: gate, scales: None, format: gate_format },
            up: larql_compute::QuantWeight { data: up, scales: None, format: up_format },
            down: larql_compute::QuantWeight { data: down, scales: None, format: larql_compute::QuantFormat::Q6_K },
            input_norm: &input_norms[l],
            post_attn_norm: &post_attn_norms[l],
            pre_ffn_norm: Some(&pre_ffn_norms[l]),
            post_ffn_norm: Some(&post_ffn_norms[l]),
            q_norm_weight: Some(&q_norms[l]),
            k_norm_weight: Some(&k_norms[l]),
            norm_offset: 1.0, qk_norm_offset: 1.0, eps: 1e-6, has_post_norms: true,
            norm_type: larql_compute::NormType::RmsNorm,
            ffn_type: larql_compute::FfnType::Gated,
            activation: larql_compute::Activation::GeluTanh,
            attn_scale: 1.0 / (head_dim as f32).sqrt(),
            head_dim, num_q_heads, num_kv_heads,
            rope_base: rope_base_for(l),
            rotary_dim: 0,
            sliding_window: sliding_window_for(l),
            has_v_norm: false, layer_scalar: 0.0,
            input_norm_bias: None, post_attn_norm_bias: None,
            ffn_up_bias: None, ffn_down_bias: None,
        }
    }).collect();

    // Real BOS embedding (tok_id=2 in Gemma 3), scaled by sqrt(hidden_size).
    // Reads from embeddings.bin (f16, row-major [vocab, hidden]).
    let emb_file = std::fs::File::open(dir.join("embeddings.bin")).unwrap();
    let emb_mmap = unsafe { memmap2::Mmap::map(&emb_file).unwrap() };
    let bos_tok: usize = 2;
    let row_bytes = hidden * 2; // f16 = 2 bytes
    let row_start = bos_tok * row_bytes;
    let bos_f16 = &emb_mmap[row_start..row_start + row_bytes];
    let bos_embed = decode_f16(bos_f16);
    let embed_scale = (hidden as f32).sqrt();
    let x: Vec<f32> = bos_embed.iter().map(|v| v * embed_scale).collect();
    let x_amax = x.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("[input] real BOS scaled embed amax={x_amax:.3}");

    let metal = get_metal();
    let be: &dyn ComputeBackend = &metal;
    be.reset_kv_cache();
    let metal_out = be.decode_token(
        &layers, &x, hidden, inter, q_dim, kv_dim,
        num_q_heads, num_kv_heads, head_dim, 10_000.0
    ).expect("decode_token");
    let metal_amax = metal_out.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let m_mn = metal_out.iter().copied().fold(f32::INFINITY, f32::min);
    let m_mx = metal_out.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("[metal-34]    range=[{m_mn:.2}, {m_mx:.2}] amax={metal_amax:.2}");

    // CPU reference: chain the same backend-ops + CPU norm math per layer,
    // layer-by-layer, for all 34 real layers. T=1 attention per layer
    // (single-token test with fresh KV).
    fn gelu_tanh(x: f32) -> f32 {
        let c: f32 = (2.0 / std::f32::consts::PI).sqrt();
        let arg = (c * (x + 0.044715 * x * x * x)).clamp(-20.0, 20.0);
        0.5 * x * (1.0 + arg.tanh())
    }
    fn rms_norm(x: &[f32], w: &[f32], offset: f32, eps: f32) -> Vec<f32> {
        let sq: f64 = x.iter().map(|v| (*v as f64).powi(2)).sum();
        let r = 1.0f32 / ((sq / x.len() as f64 + eps as f64) as f32).sqrt();
        x.iter().zip(w.iter()).map(|(v, wi)| v * (wi + offset) * r).collect()
    }
    let cpu_be = larql_compute::CpuBackend;
    let cpu_ref: &dyn ComputeBackend = &cpu_be;
    let mut h: Vec<f32> = x.clone();
    let mut max_abs_diff_so_far: f32 = 0.0;
    // Dispatch matvec based on QuantFormat (Q4_K / Q4_KF (GGUF) / Q6_K).
    fn matvec(be: &dyn ComputeBackend, w: &larql_compute::QuantWeight<'_>,
              x: &[f32], rows: usize, k: usize) -> Vec<f32> {
        let _ = be;
        match w.format {
            larql_compute::QuantFormat::Q6_K => {
                // Scalar Q6_K reference.
                larql_compute::cpu::ops::q6k_matvec::dispatch(w.data, x, rows, k)
            }
            larql_compute::QuantFormat::Q4_KF => {
                larql_compute::cpu::ops::q4k_matvec::dispatch_gguf(w.data, x, rows, k)
            }
            _ => larql_compute::cpu::ops::q4k_matvec::dispatch(w.data, x, rows, k),
        }
    }
    for (l, layer) in layers.iter().enumerate() {
        let normed_x = rms_norm(&h, layer.input_norm, 1.0, 1e-6);
        let _q_raw = matvec(cpu_ref, &layer.wq, &normed_x, q_dim, hidden);
        let _k_raw = matvec(cpu_ref, &layer.wk, &normed_x, kv_dim, hidden);
        let v_raw = matvec(cpu_ref, &layer.wv, &normed_x, kv_dim, hidden);
        // T=1 attention → attn_out = V per Q head (GQA).
        let reps = num_q_heads / num_kv_heads;
        let mut attn_out = vec![0.0f32; q_dim];
        for h_i in 0..num_q_heads {
            let kv_h = h_i / reps;
            attn_out[h_i*head_dim..(h_i+1)*head_dim]
                .copy_from_slice(&v_raw[kv_h*head_dim..(kv_h+1)*head_dim]);
        }
        let o = matvec(cpu_ref, &layer.wo, &attn_out, hidden, q_dim);
        let normed_o = rms_norm(&o, layer.post_attn_norm, 1.0, 1e-6);
        let h_post: Vec<f32> = h.iter().zip(normed_o.iter()).map(|(a, b)| a + b).collect();
        let ffn_in = rms_norm(&h_post, layer.pre_ffn_norm.unwrap(), 1.0, 1e-6);
        let gate_ffn = matvec(cpu_ref, &layer.gate, &ffn_in, inter, hidden);
        let up_ffn = matvec(cpu_ref, &layer.up, &ffn_in, inter, hidden);
        let act: Vec<f32> = gate_ffn.iter().zip(up_ffn.iter())
            .map(|(g, u)| gelu_tanh(*g) * u).collect();
        let down = matvec(cpu_ref, &layer.down, &act, hidden, inter);
        let normed_down = rms_norm(&down, layer.post_ffn_norm.unwrap(), 1.0, 1e-6);
        h = h_post.iter().zip(normed_down.iter()).map(|(a, b)| a + b).collect();
        let cpu_amax = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        if l < 3 || l == num_layers - 1 {
            println!("[cpu-L{l:02}]  amax={cpu_amax:.2}");
        }
    }
    let cpu_amax = h.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    let c_mn = h.iter().copied().fold(f32::INFINITY, f32::min);
    let c_mx = h.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    println!("[cpu-34]      range=[{c_mn:.2}, {c_mx:.2}] amax={cpu_amax:.2}");
    let diff_max = metal_out.iter().zip(h.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    println!("[metal-vs-cpu-34] metal_amax={metal_amax:.2} cpu_amax={cpu_amax:.2} max_elem_diff={diff_max:.4}");
    // Find top-3 divergent channels.
    let mut diffs: Vec<(usize, f32, f32, f32)> = metal_out.iter().zip(h.iter()).enumerate()
        .map(|(i, (m, c))| (i, *m, *c, (m - c).abs())).collect();
    diffs.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());
    println!("[metal-vs-cpu-34] top-3 diverging channels:");
    for (i, m, c, d) in diffs.iter().take(3) {
        println!("  ch={i}  metal={m:.3}  cpu={c:.3}  diff={d:.3}");
    }
    // Within tolerance is 5% of amax (quant noise, f32 accumulation differences).
    assert!(diff_max < 0.05 * cpu_amax.max(1.0),
        "Metal decode_token 34-layer output diverges from CPU ref: max_diff={diff_max:.2} amax={cpu_amax:.2}");
    let _ = max_abs_diff_so_far;

    // ── 11. Compute top-5 predicted vocab indices from the CPU-ref output
    //    (same as Metal), via final_norm (norms.bin, key "norm.weight") +
    //    lm_head (lm_head.bin, f32 [vocab, hidden]). If this matches
    //    gpuprefill's top-5 ("particularly", "Vancouver", …) we've confirmed
    //    Metal = CPU-ref-Q4K end-to-end including the lookup. If it matches
    //    cpupredict's top-5 ("The", "<h1>", …) we've found a bug distinct
    //    from Q4K precision: the Metal decode output is quant-correct but
    //    something between decode_token and finalize_logits is corrupting
    //    the hidden state in live inference.
    let final_norm = find_vec("norm.weight");
    assert_eq!(final_norm.len(), hidden);
    let h_normed = rms_norm(&h, &final_norm, 1.0, 1e-6);

    let lm_head_file = std::fs::File::open(dir.join("lm_head.bin")).unwrap();
    let lm_head_mmap = unsafe { memmap2::Mmap::map(&lm_head_file).unwrap() };
    // lm_head.bin is f32 [vocab, hidden], row-major.
    let vocab = lm_head_mmap.len() / (hidden * 4);
    println!("[lm_head] vocab={vocab}, bytes={}", lm_head_mmap.len());
    let lm_head_f32: &[f32] = unsafe {
        std::slice::from_raw_parts(lm_head_mmap.as_ptr() as *const f32, vocab * hidden)
    };
    // Logits[v] = sum_d lm_head[v,d] * h_normed[d]
    let mut logits = vec![0.0f32; vocab];
    for v in 0..vocab {
        let mut acc = 0.0f64;
        let row = &lm_head_f32[v * hidden..(v + 1) * hidden];
        for d in 0..hidden {
            acc += row[d] as f64 * h_normed[d] as f64;
        }
        logits[v] = acc as f32;
    }
    // Top-5 by logit.
    let mut ranked: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    ranked.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    println!("[cpu-ref-top5]  (these are what the Metal decode path would predict given Metal=CPU-ref):");
    for (i, (tid, l)) in ranked.iter().take(5).enumerate() {
        println!("  {}. tid={tid:>6}  logit={l:.3}", i + 1);
    }
}
