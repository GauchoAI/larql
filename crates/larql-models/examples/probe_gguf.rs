use larql_models::{GgufFile, GgufQuantizedData};
use larql_models::quant::ggml;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: probe <gguf>");
    let gguf = GgufFile::open(&std::path::PathBuf::from(&path))?;
    // Print RoPE-relevant metadata
    // RoPE / attention metadata
    for k in ["gemma3.rope.freq_base", "gemma3.rope.freq_base_swa",
              "gemma3.rope.scaling.type", "gemma3.rope.scaling.factor",
              "gemma3.attention.head_count", "gemma3.attention.head_count_kv",
              "gemma3.attention.key_length", "gemma3.attention.sliding_window"] {
        if let Some(v) = gguf.metadata.get(k) {
            println!("META {} = {:?}", k, v);
        }
    }
    println!("---");
    let qdata = GgufQuantizedData::open(&std::path::PathBuf::from(&path), gguf.data_offset)?;
    // Norm tensor stats — to check if +1 is baked in
    let norm_names = ["blk.0.attn_norm.weight", "blk.0.post_attention_norm.weight",
                      "blk.0.ffn_norm.weight", "blk.0.post_ffw_norm.weight",
                      "output_norm.weight", "blk.0.attn_q_norm.weight"];
    for n in norm_names.iter() {
        if let Some(info) = gguf.find_tensor(n) {
            if let Some(data) = qdata.tensor_f32(info) {
                let min = data.iter().cloned().fold(f32::INFINITY, f32::min);
                let max = data.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                let mean: f32 = data.iter().sum::<f32>() / data.len() as f32;
                println!("NORM {:50} dims={:?} min={:.4} max={:.4} mean={:.4} [0..3]={:.4?}",
                    n, info.dims, min, max, mean, &data[..3.min(data.len())]);
            }
        }
    }
    println!("---");
    let mut counts: std::collections::HashMap<u32, usize> = std::collections::HashMap::new();
    for ti in &gguf.tensor_infos {
        *counts.entry(ti.tensor_type).or_insert(0) += 1;
    }
    let mut v: Vec<(u32, usize)> = counts.into_iter().collect();
    v.sort_by_key(|x| std::cmp::Reverse(x.1));
    for (t, c) in v {
        println!("type={} ({:>6}) count={}", t, ggml::type_name(t), c);
    }
    println!("---");
    for prefix in [
        "token_embd", "output", "output_norm",
        "blk.0.attn_q", "blk.0.attn_k", "blk.0.attn_v", "blk.0.attn_output",
        "blk.0.ffn_gate", "blk.0.ffn_up", "blk.0.ffn_down",
        "blk.0.attn_norm", "blk.0.attn_q_norm", "blk.0.attn_k_norm",
        "blk.0.ffn_norm", "blk.0.post_attention_norm", "blk.0.post_ffw_norm",
    ].iter() {
        for ti in &gguf.tensor_infos {
            if ti.name.starts_with(prefix) {
                println!("{:50} type={:>2} ({:>6}) dims={:?}",
                    ti.name, ti.tensor_type, ggml::type_name(ti.tensor_type), ti.dims);
            }
        }
    }
    Ok(())
}
