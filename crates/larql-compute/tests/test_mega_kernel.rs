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
