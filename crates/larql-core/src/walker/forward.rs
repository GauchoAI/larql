//! Minimal transformer forward pass for residual capture.
//!
//! Implements enough of the Gemma/Llama architecture to run tokens through
//! layers and capture the residual stream at any point. Not a full inference
//! engine — no beam search, no KV cache, no generation loop.
//!
//! Uses BLAS-accelerated ndarray for matmul.

use ndarray::Array2;

use super::safetensors_loader::ModelWeights;

/// Run a forward pass through layers 0..=max_layer and return the
/// last-token residual at each requested capture layer.
///
/// Returns: Vec of (layer_index, residual_vector)
pub fn capture_residuals(
    weights: &ModelWeights,
    token_ids: &[u32],
    capture_layers: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    let seq_len = token_ids.len();
    let hidden = weights.hidden_size;
    let head_dim = weights.head_dim;
    let num_q = weights.num_q_heads;
    let num_kv = weights.num_kv_heads;
    let reps = num_q / num_kv;
    let scale = (head_dim as f64).powf(-0.5);

    let max_layer = *capture_layers.iter().max().unwrap_or(&0);

    // Embed + scale (Gemma convention: multiply by sqrt(hidden_size))
    let embed_scale = (hidden as f64).sqrt();
    let mut h = Array2::<f32>::zeros((seq_len, hidden));
    for (i, &tok_id) in token_ids.iter().enumerate() {
        let row = weights.embed.row(tok_id as usize);
        for j in 0..hidden {
            h[[i, j]] = row[j] * embed_scale as f32;
        }
    }

    let mut results = Vec::new();

    for layer in 0..=max_layer {
        let p = format!("layers.{layer}.");

        // Pre-attention norm
        let h_norm = rms_norm_2d(&h, weights.vectors.get(&format!("{p}input_layernorm.weight")));

        // QKV projections
        let w_q = match weights.tensors.get(&format!("{p}self_attn.q_proj.weight")) {
            Some(w) => w,
            None => continue,
        };
        let w_k = weights.tensors.get(&format!("{p}self_attn.k_proj.weight")).unwrap();
        let w_v = weights.tensors.get(&format!("{p}self_attn.v_proj.weight")).unwrap();
        let w_o = weights.tensors.get(&format!("{p}self_attn.o_proj.weight")).unwrap();

        let q_full = h_norm.dot(&w_q.t()); // (seq, num_q * head_dim)
        let k_full = h_norm.dot(&w_k.t()); // (seq, num_kv * head_dim)
        let v_full = h_norm.dot(&w_v.t()); // (seq, num_kv * head_dim)

        // Q/K norms if present (Gemma 3)
        let q_normed = match weights.vectors.get(&format!("{p}self_attn.q_norm.weight")) {
            Some(norm_w) => rms_norm_heads(&q_full, norm_w, num_q, head_dim),
            None => q_full,
        };
        let k_normed = match weights.vectors.get(&format!("{p}self_attn.k_norm.weight")) {
            Some(norm_w) => rms_norm_heads(&k_full, norm_w, num_kv, head_dim),
            None => k_full,
        };

        // Simple attention (no RoPE for now — residuals are still meaningful without it)
        // GQA: expand K,V to match Q heads
        let attn_out = gqa_attention(
            &q_normed, &k_normed, &v_full,
            num_q, num_kv, head_dim, reps, scale, seq_len,
        );

        // O projection
        let attn_projected = attn_out.dot(&w_o.t()); // (seq, hidden)

        // Post-attention norm + residual
        let attn_normed = rms_norm_2d(
            &attn_projected,
            weights.vectors.get(&format!("{p}post_attention_layernorm.weight")),
        );
        h = &h + &attn_normed;

        // Pre-FFN norm
        let h_ffn = rms_norm_2d(
            &h,
            weights.vectors.get(&format!("{p}pre_feedforward_layernorm.weight")),
        );

        // FFN: SiLU(gate) * up → down
        let w_gate = weights.tensors.get(&format!("{p}mlp.gate_proj.weight")).unwrap();
        let w_up = weights.tensors.get(&format!("{p}mlp.up_proj.weight")).unwrap();
        let w_down = weights.tensors.get(&format!("{p}mlp.down_proj.weight")).unwrap();

        let gate = silu_2d(&h_ffn.dot(&w_gate.t()));
        let up = h_ffn.dot(&w_up.t());
        let ffn_out = (gate * up).dot(&w_down.t());

        // Post-FFN norm + residual
        let ffn_normed = rms_norm_2d(
            &ffn_out,
            weights.vectors.get(&format!("{p}post_feedforward_layernorm.weight")),
        );
        h = &h + &ffn_normed;

        // Capture residual at this layer if requested
        if capture_layers.contains(&layer) {
            let last_row = h.row(seq_len - 1);
            results.push((layer, last_row.to_vec()));
        }
    }

    results
}

// ── Math ops ──

fn rms_norm_2d(x: &Array2<f32>, weight: Option<&Vec<f32>>) -> Array2<f32> {
    let eps = 1e-5f32;
    let (rows, cols) = (x.shape()[0], x.shape()[1]);
    let mut out = Array2::zeros((rows, cols));

    for i in 0..rows {
        let row = x.row(i);
        let rms = (row.iter().map(|v| v * v).sum::<f32>() / cols as f32 + eps).sqrt();
        for j in 0..cols {
            let w = match weight {
                Some(wt) => 1.0 + wt[j],
                None => 1.0,
            };
            out[[i, j]] = row[j] / rms * w;
        }
    }
    out
}

fn rms_norm_heads(x: &Array2<f32>, weight: &[f32], num_heads: usize, head_dim: usize) -> Array2<f32> {
    let eps = 1e-5f32;
    let seq_len = x.shape()[0];
    let mut out = x.clone();

    for s in 0..seq_len {
        for h in 0..num_heads {
            let offset = h * head_dim;
            let mut sq_sum = 0.0f32;
            for d in 0..head_dim {
                let v = x[[s, offset + d]];
                sq_sum += v * v;
            }
            let rms = (sq_sum / head_dim as f32 + eps).sqrt();
            for d in 0..head_dim {
                out[[s, offset + d]] = x[[s, offset + d]] / rms * (1.0 + weight[d]);
            }
        }
    }
    out
}

fn silu_2d(x: &Array2<f32>) -> Array2<f32> {
    x.mapv(|v| v * sigmoid(v))
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[allow(clippy::too_many_arguments)]
fn gqa_attention(
    q: &Array2<f32>,
    k: &Array2<f32>,
    v: &Array2<f32>,
    num_q: usize,
    _num_kv: usize,
    head_dim: usize,
    reps: usize,
    scale: f64,
    seq_len: usize,
) -> Array2<f32> {
    // Simple multi-head attention with GQA and causal mask.
    // q: (seq, num_q * head_dim), k: (seq, num_kv * head_dim), v: same as k
    let mut out = Array2::<f32>::zeros((seq_len, num_q * head_dim));

    for h in 0..num_q {
        let kv_h = h / reps;
        let q_off = h * head_dim;
        let kv_off = kv_h * head_dim;

        // Compute attention scores for this head
        let mut scores = vec![0.0f32; seq_len * seq_len];
        for i in 0..seq_len {
            for j in 0..=i {
                // causal: only attend to positions <= i
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    dot += q[[i, q_off + d]] * k[[j, kv_off + d]];
                }
                scores[i * seq_len + j] = dot * scale as f32;
            }
            // Mask future positions
            for j in (i + 1)..seq_len {
                scores[i * seq_len + j] = -1e9;
            }
        }

        // Softmax per row
        for i in 0..seq_len {
            let row_start = i * seq_len;
            let max_val = scores[row_start..row_start + seq_len]
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);
            let mut sum = 0.0f32;
            for j in 0..seq_len {
                scores[row_start + j] = (scores[row_start + j] - max_val).exp();
                sum += scores[row_start + j];
            }
            for j in 0..seq_len {
                scores[row_start + j] /= sum;
            }
        }

        // Weighted sum of V
        for i in 0..seq_len {
            for d in 0..head_dim {
                let mut val = 0.0f32;
                for j in 0..seq_len {
                    val += scores[i * seq_len + j] * v[[j, kv_off + d]];
                }
                out[[i, q_off + d]] = val;
            }
        }
    }

    out
}
