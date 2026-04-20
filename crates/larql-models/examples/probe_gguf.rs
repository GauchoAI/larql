use larql_models::GgufFile;
use larql_models::quant::ggml;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("usage: probe <gguf>");
    let gguf = GgufFile::open(&std::path::PathBuf::from(&path))?;
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
