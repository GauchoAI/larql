//! P16(c) viability oracle.
//!
//! Question: does top-K by gate·h (signed) contain top-K by |activation|?
//! If yes, an HNSW gate index could in principle replace the gate matvec.
//! If no, no graph over W_gate alone can recover activation top-K — we'd need
//! to index gate*up jointly.
//!
//! Method: for one prompt + several layers, compute exact gate scores AND
//! exact activations via real W_gate / W_up. Compare overlap of the two
//! top-K sets at K = 1024, 2048, 4096.

extern crate blas_src;

use std::collections::HashSet;

use ndarray::Array1;

use larql_inference::{
    InferenceModel, default_backend, capture_residual_post_attn_norm_ffn,
};
use larql_inference::vindex::WalkFfn;
use larql_vindex::{SilentLoadCallbacks, VectorIndex};

const PROMPTS: &[&str] = &[
    "The capital of France is",
    "Photosynthesis is the process by which plants",
    "Albert Einstein developed the theory of",
];
const LAYERS: &[usize] = &[0, 5, 10, 15, 20, 25, 30, 33];
const KS: &[usize] = &[1024, 2048, 4096];

fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608f32;
    let t = (c * (x + 0.044715f32 * x * x * x)).tanh();
    0.5 * x * (1.0 + t)
}

fn topk_by_abs(values: &[f32], k: usize) -> HashSet<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        values[b].abs().partial_cmp(&values[a].abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).collect()
}

fn topk_signed(values: &[f32], k: usize) -> HashSet<usize> {
    let mut idx: Vec<usize> = (0..values.len()).collect();
    idx.sort_unstable_by(|&a, &b| {
        values[b].partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    idx.into_iter().take(k).collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let mut model_ref = String::from("/Users/miguel_lemos/Desktop/gemma-3-4b-it");
    let mut vindex_path = String::from("/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex");
    let mut report_path: Option<String> = None;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model"  => { i += 1; model_ref = args[i].clone(); }
            "--vindex" => { i += 1; vindex_path = args[i].clone(); }
            "--report" => { i += 1; report_path = Some(args[i].clone()); }
            _ => {}
        }
        i += 1;
    }

    let model = InferenceModel::load(&model_ref)?;
    let weights = model.weights();
    let tokenizer = model.tokenizer();
    let be = default_backend();

    let mut cb = SilentLoadCallbacks;
    let mut index = VectorIndex::load_vindex(std::path::Path::new(&vindex_path), &mut cb)?;
    let _ = index.load_down_features(std::path::Path::new(&vindex_path));
    let _ = index.load_up_features(std::path::Path::new(&vindex_path));
    let _ = index.load_lm_head_q4(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q4k(std::path::Path::new(&vindex_path));
    let _ = index.load_attn_q8(std::path::Path::new(&vindex_path));
    let _ = index.load_interleaved_q4(std::path::Path::new(&vindex_path));

    let walk = WalkFfn::new_unlimited(weights, &index);

    println!("╔═════════════════════════════════════════════════════════════╗");
    println!("║  P16(c) ORACLE — top-K(gate·h) vs top-K(|activation|)       ║");
    println!("╚═════════════════════════════════════════════════════════════╝");
    println!("  Prompts: {}    Layers: {:?}    Ks: {:?}", PROMPTS.len(), LAYERS, KS);
    println!();

    let mut md = String::new();
    md.push_str("# P16(c) — gate-vs-activation oracle\n\n");
    md.push_str(&format!("Vindex: `{vindex_path}`\nPrompts: {}\nLayers: {:?}\nKs: {:?}\n\n",
        PROMPTS.len(), LAYERS, KS));
    md.push_str("Each cell is `recall@K = |gate_topK ∩ activation_topK| / K`.\n");
    md.push_str("Two rankings of gate·h are tried:\n");
    md.push_str("- **signed**: top-K by largest signed gate·h (positive only) — matches HNSW\n");
    md.push_str("- **|·|**: top-K by largest absolute gate·h — matches existing brute-force gate_knn\n\n");

    // Per-K table
    for &k in KS {
        md.push_str(&format!("## Recall@{k}\n\n"));
        md.push_str("| Prompt \\ Layer ");
        for &l in LAYERS { md.push_str(&format!("| L{l:02} sig | L{l:02} abs ")); }
        md.push_str("|\n|---");
        for _ in LAYERS { md.push_str("|---:|---:"); }
        md.push_str("|\n");

        for (pi, prompt) in PROMPTS.iter().enumerate() {
            let token_ids = tokenizer.encode(*prompt, true).expect("enc")
                .get_ids().to_vec();
            md.push_str(&format!("| p{pi}: \"{}\" ",
                if prompt.len() > 30 { &prompt[..30] } else { prompt }));

            for &target_layer in LAYERS {
                // h_pre_ffn at this layer
                let h_vec = capture_residual_post_attn_norm_ffn(
                    weights, &token_ids, target_layer, &*be, Some(&walk),
                ).ok_or_else(|| format!("capture failed at layer {target_layer}"))?;
                let h = Array1::from_vec(h_vec);

                let gate_view = match index.gate_layer_f32(target_layer) {
                    Some(d) => d,
                    None => {
                        eprintln!("layer {target_layer}: gate_layer_f32 unavailable, skipping");
                        md.push_str("| n/a | n/a ");
                        continue;
                    }
                };
                let up_view = match index.up_layer_matrix(target_layer) {
                    Some(v) => v,
                    None => {
                        eprintln!("layer {target_layer}: up_layer_matrix unavailable, skipping");
                        md.push_str("| n/a | n/a ");
                        continue;
                    }
                };

                let n_features = up_view.shape()[0];
                let hidden = up_view.shape()[1];
                if gate_view.len() != n_features * hidden {
                    eprintln!("layer {target_layer}: gate len {} != n_features*hidden {}*{}",
                        gate_view.len(), n_features, hidden);
                    md.push_str("| n/a | n/a ");
                    continue;
                }
                let gate_view = ndarray::ArrayView2::from_shape(
                    (n_features, hidden), &gate_view,
                )?;

                // Exact gate scores and up scores
                let mut gate_scores = vec![0.0f32; n_features];
                let mut up_scores = vec![0.0f32; n_features];
                for f in 0..n_features {
                    let gw = gate_view.row(f);
                    let uw = up_view.row(f);
                    let mut g = 0.0f32; let mut u = 0.0f32;
                    for d in 0..hidden {
                        g += gw[d] * h[d];
                        u += uw[d] * h[d];
                    }
                    gate_scores[f] = g;
                    up_scores[f] = u;
                }
                let activations: Vec<f32> = gate_scores.iter().zip(up_scores.iter())
                    .map(|(&g, &u)| gelu_tanh(g) * u).collect();

                for &k in std::slice::from_ref(&k) {
                    let gt = topk_by_abs(&activations, k);
                    let pred_signed = topk_signed(&gate_scores, k);
                    let pred_abs = topk_by_abs(&gate_scores, k);
                    let r_sig = pred_signed.intersection(&gt).count() as f32 / k as f32;
                    let r_abs = pred_abs.intersection(&gt).count() as f32 / k as f32;
                    md.push_str(&format!("| {:.1} % | {:.1} % ", 100.0 * r_sig, 100.0 * r_abs));
                    println!("p{pi} L{target_layer:02} K={k}: signed={:.1}%  |·|={:.1}%",
                        100.0 * r_sig, 100.0 * r_abs);
                }
            }
            md.push_str("|\n");
        }
        md.push_str("\n");
    }

    md.push_str("## Verdict\n\n");
    md.push_str("- If `|·|` recall @ K=4096 stays >= 95 % across all layers, the existing \n");
    md.push_str("  brute-force `gate_knn` ranking IS a valid proxy for activation top-K — \n");
    md.push_str("  build the HNSW `|dot|` variant (small change to ranking) and proceed.\n");
    md.push_str("- If `signed` recall is much higher than `|·|`, an HNSW with signed-dot \n");
    md.push_str("  search will work; existing infra needs activation-correlated reranking.\n");
    md.push_str("- If both stay below ~80 %, the gate alone doesn't predict activation top-K \n");
    md.push_str("  and we'd need a joint gate*up index. That's a research project.\n");

    if let Some(path) = report_path {
        std::fs::write(&path, &md)?;
        println!();
        println!("Report: {path}");
    } else {
        println!();
        println!("{md}");
    }
    Ok(())
}
