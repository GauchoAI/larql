//! Correctness: Metal `fused_attention_prefill` vs naive CPU reference.
//!
//! Only runs on targets that construct `MetalBackend` successfully.
#![cfg(all(target_os = "macos", feature = "metal"))]

extern crate blas_src;

use larql_compute::{ComputeBackend, MetalBackend};

fn be() -> Option<MetalBackend> { MetalBackend::new() }

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }).collect()
}

/// Naive causal GQA softmax attention, matching fused_attention shader semantics
/// (skip_rope=1, no QK-norm). Output: [seq_len * num_q * head_dim].
fn cpu_ref(
    q: &[f32], k: &[f32], v: &[f32],
    seq_len: usize, num_q: usize, num_kv: usize, head_dim: usize,
    scale: f32, softcap: f32,
) -> Vec<f32> {
    let reps = num_q / num_kv;
    let mut out = vec![0.0f32; seq_len * num_q * head_dim];
    for h in 0..num_q {
        let kv_h = h / reps;
        for qi in 0..seq_len {
            let causal = qi + 1;
            let mut scores = vec![0.0f32; causal];
            let mut max_s = f32::NEG_INFINITY;
            for ki in 0..causal {
                let mut dot = 0.0f32;
                for d in 0..head_dim {
                    let q_val = q[qi * num_q * head_dim + h * head_dim + d];
                    let k_val = k[ki * num_kv * head_dim + kv_h * head_dim + d];
                    dot += q_val * k_val;
                }
                let mut s = dot * scale;
                if softcap > 0.0 { s = (s / softcap).tanh() * softcap; }
                scores[ki] = s;
                if s > max_s { max_s = s; }
            }
            let mut sum = 0.0f32;
            for ki in 0..causal {
                scores[ki] = (scores[ki] - max_s).exp();
                sum += scores[ki];
            }
            let inv = 1.0 / sum;
            for d in 0..head_dim {
                let mut acc = 0.0f32;
                for ki in 0..causal {
                    let vv = v[ki * num_kv * head_dim + kv_h * head_dim + d];
                    acc += scores[ki] * inv * vv;
                }
                out[qi * num_q * head_dim + h * head_dim + d] = acc;
            }
        }
    }
    out
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

#[test]
fn metal_fused_prefill_matches_cpu_gemma3_shape() {
    let Some(be) = be() else {
        eprintln!("skipping: no Metal backend");
        return;
    };
    // Gemma 3 4B: num_q=8, num_kv=4, head_dim=256.
    let (num_q, num_kv, head_dim) = (8usize, 4usize, 256usize);
    let seq_len = 6;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let softcap = 0.0f32;
    let q = synth(seq_len * num_q * head_dim, 11);
    let k = synth(seq_len * num_kv * head_dim, 22);
    let v = synth(seq_len * num_kv * head_dim, 33);

    let gpu = be.fused_attention_prefill(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, softcap).unwrap();
    let cpu = cpu_ref(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, softcap);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 5e-4, "max |gpu-cpu| = {d}, expected < 5e-4");
}

#[test]
fn metal_fused_prefill_with_softcap() {
    let Some(be) = be() else { return; };
    let (num_q, num_kv, head_dim) = (8usize, 4usize, 128usize);
    let seq_len = 4;
    let scale = 1.0 / (head_dim as f32).sqrt();
    let softcap = 50.0f32; // Gemma 2 style
    let q = synth(seq_len * num_q * head_dim, 41);
    let k = synth(seq_len * num_kv * head_dim, 42);
    let v = synth(seq_len * num_kv * head_dim, 43);

    let gpu = be.fused_attention_prefill(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, softcap).unwrap();
    let cpu = cpu_ref(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, softcap);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 5e-4, "softcap: max |gpu-cpu| = {d}");
}

#[test]
fn metal_fused_prefill_longer_seq() {
    let Some(be) = be() else { return; };
    let (num_q, num_kv, head_dim) = (8usize, 4usize, 256usize);
    let seq_len = 32; // larger prompt
    let scale = 1.0 / (head_dim as f32).sqrt();
    let q = synth(seq_len * num_q * head_dim, 7);
    let k = synth(seq_len * num_kv * head_dim, 8);
    let v = synth(seq_len * num_kv * head_dim, 9);

    let gpu = be.fused_attention_prefill(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, 0.0).unwrap();
    let cpu = cpu_ref(&q, &k, &v, seq_len, num_q, num_kv, head_dim, scale, 0.0);
    let d = max_abs_diff(&gpu, &cpu);
    assert!(d < 5e-4, "seq_len=32: max |gpu-cpu| = {d}");
}
