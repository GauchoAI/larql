
// ═══════════════════════════════════════════════════════════════
// Real-data Q4K roundtrip — diagnoses whether build_q4k_weights's
// f16-decode path produces correct bytes.
// ═══════════════════════════════════════════════════════════════

#[test]
#[ignore]
fn q4k_build_matches_fresh_quantization() {
    // Only runs under `cargo test ... --ignored`. Requires:
    //   LARQL_VINDEX_PATH=/path/to/gemma3-4b.vindex
    //
    // Reads attn_weights.bin + attn_weights_q4k_manifest.json for layer 0 q_proj.
    // Decodes the f16 source, quantizes fresh via quantize_q4_k, compares byte-by-byte
    // with the stored Q4K. Any mismatch → build_q4k_weights's f16 decode has a bug.
    let vindex = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex);

    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("weight_manifest.json")).unwrap()
    ).unwrap();
    let entry = manifest.iter().find(|e| {
        e.get("file").and_then(|f| f.as_str()) == Some("attn_weights.bin")
            && e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0.self_attn.q_proj"))
    }).expect("no layer 0 q_proj in manifest");

    let offset = entry["offset"].as_u64().unwrap() as usize;
    let length = entry["length"].as_u64().unwrap() as usize;
    let shape = entry["shape"].as_array().unwrap();
    let rows = shape[0].as_u64().unwrap() as usize;
    let cols = shape[1].as_u64().unwrap() as usize;
    let num_floats = rows * cols;
    let bytes_per_elem = length / num_floats;
    println!("q_proj shape [{rows},{cols}], length {length}, bytes_per_elem {bytes_per_elem}");

    // Load source bytes
    let src_file = std::fs::File::open(dir.join("attn_weights.bin")).unwrap();
    let src_mmap = unsafe { memmap2::Mmap::map(&src_file).unwrap() };
    let raw = &src_mmap[offset..offset + length];

    // Decode f16 -> f32 (same path used by patched build_q4k_weights)
    let f32_source: Vec<f32> = match bytes_per_elem {
        2 => larql_models::quant::half::decode_f16(raw),
        4 => unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const f32, num_floats).to_vec()
        },
        other => panic!("unsupported bytes_per_elem: {other}"),
    };

    // Sanity on decoded f32 values
    let nf = f32_source.iter().filter(|v| !v.is_finite()).count();
    let max_abs = f32_source.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("Decoded f32: {} non-finite, max |value| = {max_abs}", nf);
    assert_eq!(nf, 0, "f16 decode produced non-finite values");
    assert!(max_abs < 100.0, "f16 decoded max |value| unusually large: {max_abs}");

    // Fresh quantize
    let fresh = larql_compute::cpu::ops::q4_common::quantize_q4_k(&f32_source);
    println!("Fresh quantize: {} bytes", fresh.len());

    // Load stored Q4K from manifest
    let q4k_manifest_path = dir.join("attn_weights_q4k_manifest.json");
    if !q4k_manifest_path.exists() {
        println!("(attn_weights_q4k_manifest.json missing — skipping byte comparison)");
        return;
    }
    let q4k_manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(&q4k_manifest_path).unwrap()
    ).unwrap();
    let q4k_entry = q4k_manifest.iter().find(|e| {
        e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0.self_attn.q_proj"))
    }).expect("no q_proj in q4k manifest");
    let q4k_offset = q4k_entry["offset"].as_u64().unwrap() as usize;
    let q4k_length = q4k_entry["length"].as_u64().unwrap() as usize;

    let q4k_file = std::fs::File::open(dir.join("attn_weights_q4k.bin")).unwrap();
    let q4k_mmap = unsafe { memmap2::Mmap::map(&q4k_file).unwrap() };
    let stored = &q4k_mmap[q4k_offset..q4k_offset + q4k_length];

    assert_eq!(fresh.len(), stored.len(), "length mismatch: fresh={} stored={}", fresh.len(), stored.len());
    let mut diff_bytes = 0usize;
    for (a, b) in fresh.iter().zip(stored.iter()) {
        if a != b { diff_bytes += 1; }
    }
    println!("Byte comparison: {} mismatched of {} total", diff_bytes, fresh.len());
    assert_eq!(diff_bytes, 0, "stored Q4K differs from fresh quantization — build_q4k_weights path may be buggy");
}

#[test]
#[ignore]
fn q4k_dequant_vs_original_accuracy() {
    // Dequantize layer 0 q_proj Q4_K back to f32 (via q4k_matvec with identity-like x)
    // and compare against the original f32-decoded f16 values. Validates
    // quantize_q4_k's approximation quality, not just byte-identity.
    let vindex = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex);

    // Read original f16 q_proj
    let manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("weight_manifest.json")).unwrap()
    ).unwrap();
    let entry = manifest.iter().find(|e|
        e.get("file").and_then(|f| f.as_str()) == Some("attn_weights.bin")
            && e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0.self_attn.q_proj"))
    ).unwrap();
    let offset = entry["offset"].as_u64().unwrap() as usize;
    let length = entry["length"].as_u64().unwrap() as usize;
    let shape = entry["shape"].as_array().unwrap();
    let rows = shape[0].as_u64().unwrap() as usize;
    let cols = shape[1].as_u64().unwrap() as usize;
    let src_file = std::fs::File::open(dir.join("attn_weights.bin")).unwrap();
    let src_mmap = unsafe { memmap2::Mmap::map(&src_file).unwrap() };
    let orig_f32: Vec<f32> = larql_models::quant::half::decode_f16(&src_mmap[offset..offset + length]);

    // Read stored Q4K bytes
    let q4k_manifest: Vec<serde_json::Value> = serde_json::from_str(
        &std::fs::read_to_string(dir.join("attn_weights_q4k_manifest.json")).unwrap()
    ).unwrap();
    let q4k_entry = q4k_manifest.iter().find(|e|
        e.get("key").and_then(|k| k.as_str()).is_some_and(|k| k.contains("layers.0.self_attn.q_proj"))
    ).unwrap();
    let q4k_offset = q4k_entry["offset"].as_u64().unwrap() as usize;
    let q4k_length = q4k_entry["length"].as_u64().unwrap() as usize;
    let q4k_file = std::fs::File::open(dir.join("attn_weights_q4k.bin")).unwrap();
    let q4k_mmap = unsafe { memmap2::Mmap::map(&q4k_file).unwrap() };
    let q4k_bytes = &q4k_mmap[q4k_offset..q4k_offset + q4k_length];

    // Run Metal q4k_matvec(W, identity_col_i) to recover column i of W. Use x = unit vector.
    let metal = larql_compute::metal::MetalBackend::new().expect("metal");
    use larql_compute::ComputeBackend;
    let be: &dyn ComputeBackend = &metal;

    // Pick a handful of columns, recover them, compare to original.
    let mut max_abs_err = 0.0f32;
    let mut sum_sq_err = 0.0f64;
    let mut n_samples = 0usize;
    for &col in &[0usize, 1, 100, 500, 1000, 2000, 2559] {
        let mut x = vec![0.0f32; cols];
        x[col] = 1.0;
        let out = be.q4k_matvec(q4k_bytes, &x, rows, cols).expect("q4k_matvec");
        for row in 0..rows {
            let expected = orig_f32[row * cols + col];
            let got = out[row];
            let err = (expected - got).abs();
            max_abs_err = max_abs_err.max(err);
            sum_sq_err += (err as f64).powi(2);
            n_samples += 1;
        }
    }
    let rms_err = (sum_sq_err / n_samples as f64).sqrt() as f32;
    let orig_max = orig_f32.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("q_proj L0: orig max|w|={orig_max:.4}  Q4K dequant max_abs_err={max_abs_err:.4}  rms_err={rms_err:.6}");
    println!("  relative max err: {:.2}%", 100.0 * max_abs_err / orig_max.max(1e-6));
    // Q4_K typical error should be < 10% of max weight magnitude.
    assert!(max_abs_err < 0.1 * orig_max, "Q4_K quantization error {max_abs_err} too large (> 10% of {orig_max})");
}

#[test]
#[ignore]
fn q4k_matvec_real_gate_synthetic_x() {
    // Same structure: real gate weights (Q4_K) + synthetic x=1.0.
    // Finite output → the FFN NaN comes from real ffn_norm_out magnitudes.
    // Non-finite → gate's Q4_K dequant has an issue.
    let vindex = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex);

    let hidden = 2560usize;
    let inter = 10240usize;
    let q4k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 148;

    let file = std::fs::File::open(dir.join("interleaved_q4k.bin")).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let gate_bytes = &mmap[0..q4k_bytes_per_matrix]; // layer 0 gate is first

    let metal = larql_compute::metal::MetalBackend::new().expect("metal");
    let be: &dyn larql_compute::ComputeBackend = &metal;

    // Sweep x magnitudes to find where gate_matvec goes non-finite.
    for x_mag in [1.0f32, 10.0, 100.0, 1000.0, 10000.0, 100000.0] {
        let x = vec![x_mag; hidden];
        let result = be.q4k_matvec(gate_bytes, &x, inter, hidden).expect("q4k_matvec");
        let n_inf = result.iter().filter(|v| v.is_infinite()).count();
        let n_nan = result.iter().filter(|v| v.is_nan()).count();
        let max_abs = result.iter().filter(|v| v.is_finite()).map(|v| v.abs()).fold(0.0f32, f32::max);
        println!("q4k_matvec(real_gate, x={x_mag}): {n_inf} inf, {n_nan} nan, max|val|={max_abs:.2}");
    }
}

#[test]
#[ignore]
fn q6k_matvec_real_down_synthetic_x() {
    // Run Metal q6k_matvec with real layer-0 down weights + synthetic x=1.0.
    // If output is finite, the NaN comes from act_buf magnitudes, not weights.
    let vindex = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex);

    let hidden = 2560usize;
    let inter = 10240usize;
    let q4k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 148;
    let q6k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 210;
    let down_offset = 2 * q4k_bytes_per_matrix;

    let file = std::fs::File::open(dir.join("interleaved_q4k.bin")).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let down_bytes = &mmap[down_offset..down_offset + q6k_bytes_per_matrix];

    let metal = larql_compute::metal::MetalBackend::new().expect("metal");
    use larql_compute::ComputeBackend;
    let be: &dyn ComputeBackend = &metal;

    let x = vec![1.0f32; inter];
    let result = be.q6k_matvec(down_bytes, &x, hidden, inter).expect("q6k_matvec");
    let n_inf = result.iter().filter(|v| v.is_infinite()).count();
    let n_nan = result.iter().filter(|v| v.is_nan()).count();
    let finite_cnt = result.len() - n_inf - n_nan;
    let max_abs = result.iter().filter(|v| v.is_finite()).map(|v| v.abs()).fold(0.0f32, f32::max);
    println!("q6k_matvec(real_down, x=1.0): {n_inf} inf, {n_nan} nan, {finite_cnt} finite, max|val|={max_abs:.4}");
    assert_eq!(n_inf + n_nan, 0, "q6k_matvec of real down × synthetic x=1 produces non-finite");
}

#[test]
#[ignore]
fn q6k_down_layer0_scale_check() {
    // Inspect every super-block scale (f16 `d` and 16 int8 sub-block scales)
    // in layer 0's down weight. If any d is huge (overflow territory), that's
    // our decode_token NaN source.
    let vindex = std::env::var("LARQL_VINDEX_PATH").expect("set LARQL_VINDEX_PATH");
    let dir = std::path::Path::new(&vindex);

    // Gemma 3 4B dims
    let hidden = 2560usize;
    let inter = 10240usize;
    let q4k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 148;
    let q6k_bytes_per_matrix = (inter * hidden).div_ceil(256) * 210;
    let down_offset_in_layer0 = 2 * q4k_bytes_per_matrix;

    let file = std::fs::File::open(dir.join("interleaved_q4k.bin")).unwrap();
    let mmap = unsafe { memmap2::Mmap::map(&file).unwrap() };
    let down = &mmap[down_offset_in_layer0..down_offset_in_layer0 + q6k_bytes_per_matrix];

    fn decode_f16_one(bits: u16) -> f32 {
        let sign = (bits >> 15) & 0x1;
        let exp  = (bits >> 10) & 0x1F;
        let mant = (bits & 0x3FF) as u32;
        let b = if exp == 0 {
            if mant == 0 { (sign as u32) << 31 }
            else {
                let mut m = mant; let mut e: i32 = -14;
                while (m & 0x400) == 0 { m <<= 1; e -= 1; }
                ((sign as u32) << 31) | (((e + 127) as u32) << 23) | ((m & 0x3FF) << 13)
            }
        } else if exp == 0x1F {
            ((sign as u32) << 31) | (0xFF << 23) | (mant << 13)
        } else {
            ((sign as u32) << 31) | (((exp as i32 - 15 + 127) as u32) << 23) | (mant << 13)
        };
        f32::from_bits(b)
    }

    let num_sb = down.len() / 210;
    println!("layer 0 down: {} super-blocks", num_sb);
    let mut d_abs_max = 0.0f32;
    let mut sc_abs_max = 0i32;
    let mut n_d_nonfinite = 0usize;
    let mut max_dequant = 0.0f32;
    for sb_idx in 0..num_sb {
        let block = &down[sb_idx * 210 .. (sb_idx + 1) * 210];
        let d_bits = u16::from_le_bytes([block[208], block[209]]);
        let d = decode_f16_one(d_bits);
        if !d.is_finite() { n_d_nonfinite += 1; continue; }
        d_abs_max = d_abs_max.max(d.abs());
        for j in 0..16 {
            let sc = block[192 + j] as i8 as i32;
            sc_abs_max = sc_abs_max.max(sc.abs());
            // max possible dequantized value for this sub-block:
            //   d * sc * ((0b111111) - 32) = d * sc * 31
            let m = (d * sc as f32 * 31.0).abs();
            if m.is_finite() { max_dequant = max_dequant.max(m); }
        }
    }
    println!("f16 super-block scale |d| max: {d_abs_max}, non-finite: {n_d_nonfinite}");
    println!("int8 sub-block scale |sc| max: {sc_abs_max}");
    println!("max possible |dequantized value|: {max_dequant}");
}
