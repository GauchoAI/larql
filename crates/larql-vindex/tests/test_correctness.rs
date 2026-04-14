
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
