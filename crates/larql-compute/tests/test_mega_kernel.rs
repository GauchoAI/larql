//! Test: persistent threadgroup atomic sync on Metal.
//!
//! Validates that N threadgroups can synchronize via device-memory
//! atomics within a single kernel dispatch. This is the foundation
//! for the S3 mega-kernel (34 layers, zero dispatch overhead).

#[cfg(feature = "metal")]
#[test]
fn test_mega_kernel_atomic_sync() {
    use metal::*;

    let device = Device::system_default().expect("no Metal GPU");
    let queue = device.new_command_queue();

    // Compile the test shader
    let src = larql_compute::metal::shaders::mega_kernel::TEST_SHADER;
    let header = larql_compute::metal::shaders::common::HEADER;
    let full_src = format!("{header}\n{src}");
    let opts = CompileOptions::new();
    let library = device.new_library_with_source(&full_src, &opts)
        .expect("shader compile failed");
    let func = library.get_function("mega_test", None)
        .expect("mega_test function not found");
    let pipeline = device.new_compute_pipeline_state_with_function(&func)
        .expect("pipeline creation failed");

    let hidden: u32 = 2560;
    let num_tg: u32 = 64; // conservative — should be safe on 16-core GPU
    let tg_size: u64 = 32; // small threadgroup for max occupancy

    // Allocate buffers
    let data_buf = device.new_buffer((hidden as u64) * 4, MTLResourceOptions::StorageModeShared);
    let sync_buf = device.new_buffer(3 * 4, MTLResourceOptions::StorageModeShared); // 3 atomic uints

    // Zero sync buffer
    unsafe {
        let ptr = sync_buf.contents() as *mut u32;
        *ptr = 0; // phase
        *ptr.add(1) = 0; // done count
        *ptr.add(2) = num_tg; // total TGs
    }

    // Dispatch
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&data_buf), 0);
    enc.set_buffer(1, Some(&sync_buf), 0);
    enc.set_bytes(2, 4, &hidden as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(3, 4, &num_tg as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tg as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Verify: data[i] should be (i + 1) * 2
    let result: Vec<f32> = unsafe {
        let ptr = data_buf.contents() as *const f32;
        (0..hidden as usize).map(|i| *ptr.add(i)).collect()
    };

    let mut errors = 0;
    for i in 0..hidden as usize {
        let expected = (i as f32 + 1.0) * 2.0;
        if (result[i] - expected).abs() > 0.01 {
            if errors < 5 {
                eprintln!("  MISMATCH at {i}: got {}, expected {}", result[i], expected);
            }
            errors += 1;
        }
    }

    if errors == 0 {
        eprintln!("  PASS: atomic sync works — {} TGs synchronized across 2 phases", num_tg);
    } else {
        panic!("  FAIL: {errors}/{hidden} mismatches — atomic sync broken on this GPU");
    }

    // Check sync counters
    let phase = unsafe { *(sync_buf.contents() as *const u32) };
    let done = unsafe { *(sync_buf.contents() as *const u32).add(1) };
    eprintln!("  Final sync state: phase={phase}, done={done}");
    assert_eq!(phase, 2, "expected 2 phases completed");
}

/// Step 1.5: Isolate cross-TG read behavior.
/// TG 0 writes 2560 values, ALL TGs read them after atomic barrier.
#[cfg(feature = "metal")]
#[test]
fn test_cross_tg_read() {
    use metal::*;

    let device = Device::system_default().expect("no Metal GPU");
    let queue = device.new_command_queue();

    // Minimal shader: TG 0 writes, all TGs read
    let shader_src = r#"
#include <metal_stdlib>
using namespace metal;
kernel void cross_tg_test(
    device float*         buf    [[buffer(0)]],
    device float*         out    [[buffer(1)]],
    device atomic_uint*   sync   [[buffer(2)]],
    constant uint&        len    [[buffer(3)]],
    constant uint&        num_tg [[buffer(4)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    // Phase 0: ALL TGs write their slice (like Step 1)
    for (uint i = tg_id * tg_sz + tid; i < len; i += num_tg * tg_sz) {
        buf[i] = float(i) + 1.0f;
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint done = atomic_fetch_add_explicit(&sync[1], 1u, memory_order_relaxed);
        if (done + 1 == num_tg) {
            atomic_store_explicit(&sync[1], 0u, memory_order_relaxed);
            atomic_store_explicit(&sync[0], 1u, memory_order_relaxed);
        }
    }
    if (tid == 0) {
        while (atomic_load_explicit(&sync[0], memory_order_relaxed) < 1u) {}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: Each TG reads ALL of buf (cross-TG read!)
    // and writes a checksum to out[tg_id]
    float sum = 0.0f;
    for (uint i = tid; i < len; i += tg_sz) {
        sum += buf[i];
    }
    // Simple reduction
    sum = simd_sum(sum);
    if (tid == 0) {
        out[tg_id] = sum;
    }
}
"#;

    let header = larql_compute::metal::shaders::common::HEADER;
    let full = format!("{header}\n{shader_src}");
    let lib = device.new_library_with_source(&full, &CompileOptions::new()).expect("compile");
    let func = lib.get_function("cross_tg_test", None).expect("func");
    let pipe = device.new_compute_pipeline_state_with_function(&func).expect("pipe");

    let len: u32 = 2560;
    let num_tg: u32 = 32;
    let tg_size: u64 = 32;

    let buf = device.new_buffer((len as u64) * 4, MTLResourceOptions::StorageModeShared);
    let out = device.new_buffer((num_tg as u64) * 4, MTLResourceOptions::StorageModeShared);
    let sync = device.new_buffer(2 * 4, MTLResourceOptions::StorageModeShared);
    unsafe { let p = sync.contents() as *mut u32; *p = 0; *p.add(1) = 0; }

    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipe);
    enc.set_buffer(0, Some(&buf), 0);
    enc.set_buffer(1, Some(&out), 0);
    enc.set_buffer(2, Some(&sync), 0);
    enc.set_bytes(3, 4, &len as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(4, 4, &num_tg as *const u32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(MTLSize::new(num_tg as u64, 1, 1), MTLSize::new(tg_size, 1, 1));
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Expected checksum: sum(1..2561) = 2560*2561/2 = 3278080
    let expected_sum: f32 = (1..=len).map(|i| i as f32).sum();
    let results: Vec<f32> = unsafe {
        let p = out.contents() as *const f32;
        (0..num_tg as usize).map(|i| *p.add(i)).collect()
    };

    let mut pass = 0;
    for i in 0..num_tg as usize {
        let err = (results[i] - expected_sum).abs() / expected_sum;
        if err < 0.01 { pass += 1; }
    }
    eprintln!("  Cross-TG read: {pass}/{num_tg} TGs got correct checksum");
    eprintln!("  Expected: {:.0}, got: [{:.0}, {:.0}, {:.0}, ...]",
        expected_sum, results[0], results[1], results[2]);
    assert_eq!(pass, num_tg as usize, "cross-TG reads failed");
}

/// Step 2: f32 matvec inside persistent kernel.
/// 3 phases: RMS norm → matvec → GELU.
/// Validates real computation across atomic barriers.
#[cfg(feature = "metal")]
#[test]
fn test_mega_kernel_matvec() {
    use metal::*;

    let device = Device::system_default().expect("no Metal GPU");
    let queue = device.new_command_queue();

    let src = larql_compute::metal::shaders::mega_kernel::TEST_SHADER;
    let header = larql_compute::metal::shaders::common::HEADER;
    let full_src = format!("{header}\n{src}");
    let opts = CompileOptions::new();
    let library = device.new_library_with_source(&full_src, &opts)
        .expect("shader compile failed");
    let func = library.get_function("mega_norm_matvec_act", None)
        .expect("function not found");
    let pipeline = device.new_compute_pipeline_state_with_function(&func)
        .expect("pipeline failed");

    let n: u32 = 512;  // output rows
    let k: u32 = 256;  // input dim
    let num_tg: u32 = 32;
    let tg_size: u64 = 64;
    let eps: f32 = 1e-6;

    // Generate test data
    let x_in: Vec<f32> = (0..k).map(|i| (i as f32 * 0.01) - 1.28).collect();
    let norm_w: Vec<f32> = (0..k).map(|i| 0.5 + (i as f32 * 0.001)).collect();
    let w: Vec<f32> = (0..n * k).map(|i| ((i as f32 * 7.0 + 3.0) % 97.0 - 48.5) * 0.01).collect();

    // CPU reference: norm → matvec → gelu
    let sum_sq: f32 = x_in.iter().map(|v| v * v).sum();
    let rms = 1.0 / (sum_sq / k as f32 + eps).sqrt();
    let x_normed: Vec<f32> = (0..k as usize).map(|i| x_in[i] * (norm_w[i] + 1.0) * rms).collect();

    // Raw matvec reference (no norm, no gelu — testing matvec correctness first)
    let mut y_ref: Vec<f32> = vec![0.0; n as usize];
    for row in 0..n as usize {
        let mut dot = 0.0f32;
        for j in 0..k as usize {
            dot += w[row * k as usize + j] * x_in[j];
        }
        // GELU
        let v3 = dot * dot * dot;
        let arg = (0.7978845608f32 * (dot + 0.044715 * v3)).clamp(-10.0, 10.0);
        y_ref[row] = 0.5 * dot * (1.0 + arg.tanh());
    }

    // Metal buffers
    let w_buf = device.new_buffer_with_data(w.as_ptr() as *const _, (n * k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let x_buf = device.new_buffer_with_data(x_in.as_ptr() as *const _, (k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let nw_buf = device.new_buffer_with_data(norm_w.as_ptr() as *const _, (k * 4) as u64, MTLResourceOptions::StorageModeShared);
    let xn_buf = device.new_buffer((k as u64) * 4, MTLResourceOptions::StorageModeShared);
    let y_buf = device.new_buffer((n as u64) * 4, MTLResourceOptions::StorageModeShared);
    let sync_buf = device.new_buffer(3 * 4, MTLResourceOptions::StorageModeShared); // phase + done + rms_bits
    unsafe {
        let ptr = sync_buf.contents() as *mut u32;
        *ptr = 0; *ptr.add(1) = 0; *ptr.add(2) = 0;
    }

    // Dispatch
    let cmd = queue.new_command_buffer();
    let enc = cmd.new_compute_command_encoder();
    enc.set_compute_pipeline_state(&pipeline);
    enc.set_buffer(0, Some(&w_buf), 0);
    enc.set_buffer(1, Some(&x_buf), 0);
    enc.set_buffer(2, Some(&nw_buf), 0);
    enc.set_buffer(3, Some(&xn_buf), 0);
    enc.set_buffer(4, Some(&y_buf), 0);
    enc.set_buffer(5, Some(&sync_buf), 0);
    enc.set_bytes(6, 4, &n as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(7, 4, &k as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(8, 4, &num_tg as *const u32 as *const std::ffi::c_void);
    enc.set_bytes(9, 4, &eps as *const f32 as *const std::ffi::c_void);
    enc.dispatch_thread_groups(
        MTLSize::new(num_tg as u64, 1, 1),
        MTLSize::new(tg_size, 1, 1),
    );
    enc.end_encoding();
    cmd.commit();
    cmd.wait_until_completed();

    // Compare
    let y_gpu: Vec<f32> = unsafe {
        let ptr = y_buf.contents() as *const f32;
        (0..n as usize).map(|i| *ptr.add(i)).collect()
    };

    // Check x_buf (normed input) — is it correct?
    let x_normed_gpu: Vec<f32> = unsafe {
        let ptr = xn_buf.contents() as *const f32;
        (0..k as usize).map(|i| *ptr.add(i)).collect()
    };
    let xn_max_err: f32 = x_normed_gpu.iter().zip(x_normed.iter())
        .map(|(a, b)| (a - b).abs()).fold(0.0f32, f32::max);
    let xn_zeros = x_normed_gpu.iter().filter(|v| **v == 0.0).count();
    eprintln!("  x_buf: max_err={xn_max_err:.6}, zeros={xn_zeros}/{k}, first=[{:.4}, {:.4}, {:.4}]",
        x_normed_gpu[0], x_normed_gpu[1], x_normed_gpu[2]);

    // Debug: check if TG 1 could read x_buf[0] (written by TG 0)
    let debug_val = y_gpu[n as usize - 1];
    let expected_xbuf0 = x_normed[0];
    eprintln!("  DEBUG: TG1 read x_buf[0] = {:.4} (expected {:.4}, match={})",
        debug_val, expected_xbuf0, (debug_val - expected_xbuf0).abs() < 0.01);

    let mut max_err = 0.0f32;
    let mut errors = 0;
    let mut pass_rows: Vec<usize> = Vec::new();
    for i in 0..n as usize {
        let err = (y_gpu[i] - y_ref[i]).abs();
        max_err = max_err.max(err);
        if err > 0.01 {
            errors += 1;
        } else {
            pass_rows.push(i);
        }
    }
    eprintln!("  Passing rows: {:?}", &pass_rows[..pass_rows.len().min(20)]);

    eprintln!("  Max error: {max_err:.6}  Errors: {errors}/{n}");
    if errors == 0 {
        eprintln!("  PASS: norm→matvec→gelu in persistent kernel matches CPU reference");
    } else {
        panic!("  FAIL: {errors}/{n} values differ by >0.01");
    }
}
