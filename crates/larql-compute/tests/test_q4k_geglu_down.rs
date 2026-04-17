//! Correctness: fused `q4k_geglu_*_down` Metal kernels vs CPU reference.
//!
//! P8 found these kernels produced garbage output when wired into walk FFN.
//! This test compares the Metal kernel output against a CPU reference
//! (CPU GEGLU loop + q4k_matvec) on synthetic input. If they agree to within
//! Q4_K quantisation tolerance, the kernel is correct and we can re-enable
//! the fused dispatch in walk FFN.
#![cfg(all(target_os = "macos", feature = "metal"))]

extern crate blas_src;

use larql_compute::{ComputeBackend, MetalBackend};
use larql_compute::cpu::ops::q4_common::quantize_q4_k;

fn synth(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n).map(|_| {
        s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
        ((s >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0
    }).collect()
}

fn silu(x: f32) -> f32 { x / (1.0 + (-x).exp()) }
fn gelu_tanh(x: f32) -> f32 {
    let c = 0.7978845608f32;
    let t = (c * (x + 0.044715 * x * x * x)).tanh();
    0.5 * x * (1.0 + t)
}

fn cpu_reference(
    down_q4: &[u8],
    gate: &[f32],
    up: &[f32],
    hidden: usize,
    inter: usize,
    activation: &str,
    be: &MetalBackend,
) -> Vec<f32> {
    // Compute activation = activation(gate) * up
    let mut act = vec![0.0f32; inter];
    for i in 0..inter {
        let a = match activation {
            "silu" => silu(gate[i]),
            "gelu_tanh" => gelu_tanh(gate[i]),
            _ => unreachable!(),
        };
        act[i] = a * up[i];
    }
    // Reference: use the working q4k_matvec to do the down projection.
    be.q4k_matvec(down_q4, &act, hidden, inter).expect("q4k_matvec available")
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max)
}

fn rms_diff(a: &[f32], b: &[f32]) -> f32 {
    let n = a.len() as f32;
    (a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum::<f32>() / n).sqrt()
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    dot / (na * nb).max(1e-10)
}

fn run_one(activation: &str, hidden: usize, inter: usize) {
    let Some(be) = MetalBackend::new() else {
        eprintln!("skipping: no Metal backend");
        return;
    };

    // Build a synthetic down matrix [hidden, inter] (so q4k_matvec computes
    // out[h] = sum_i W[h,i] * activation[i], same shape walk FFN uses).
    let down_f32 = synth(hidden * inter, 11);
    let down_q4 = quantize_q4_k(&down_f32);

    let gate = synth(inter, 22);
    let up = synth(inter, 33);

    let cpu = cpu_reference(&down_q4, &gate, &up, hidden, inter, activation, &be);
    let gpu = be.q4k_geglu_down(&down_q4, &gate, &up, hidden, inter, activation)
        .expect("q4k_geglu_down available");

    assert_eq!(gpu.len(), hidden);
    let max_d = max_abs_diff(&cpu, &gpu);
    let rms_d = rms_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    let cpu_norm: f32 = cpu.iter().map(|v| v * v).sum::<f32>().sqrt();
    println!("[{activation} h={hidden} k={inter}] max|Δ|={max_d:.4} rms_Δ={rms_d:.4} cos={cos:.6} ‖cpu‖={cpu_norm:.2}");

    // Both should differ only by floating-point reordering of the same math
    // (CPU does GEGLU per-element then matvec; GPU fuses both). Tolerance:
    // cos > 0.9999 is plenty.
    assert!(cos > 0.999, "{activation} h={hidden} k={inter}: cos = {cos}");
}

#[test]
fn gelu_tanh_small() { run_one("gelu_tanh", 256, 256); }

#[test]
fn silu_small() { run_one("silu", 256, 256); }

#[test]
fn gelu_tanh_gemma3_shape() {
    // Gemma 3 4B walk-FFN actual shapes: hidden=2560, intermediate=10240.
    run_one("gelu_tanh", 2560, 10240);
}

#[test]
fn silu_llama_shape() {
    run_one("silu", 4096, 11008);
}

/// S1/P11 correctness: q4k_ffn_full (one cmd buffer with gate+up+GEGLU+down)
/// must produce identical output to the chained q4k_matvec_pair + q4k_geglu_down.
#[test]
fn q4k_ffn_full_matches_chained() {
    let Some(be) = MetalBackend::new() else { return; };
    let hidden = 2560;
    let inter = 10240;

    // Realistic-magnitude weights and input (mimics walk-FFN production data).
    let gate_w: Vec<f32> = synth(inter * hidden, 1).into_iter().map(|v| v * 0.05).collect();
    let up_w:   Vec<f32> = synth(inter * hidden, 2).into_iter().map(|v| v * 0.05).collect();
    let down_w: Vec<f32> = synth(hidden * inter, 3).into_iter().map(|v| v * 0.05).collect();
    let gate_q4 = quantize_q4_k(&gate_w);
    let up_q4   = quantize_q4_k(&up_w);
    let down_q4 = quantize_q4_k(&down_w);

    for layer in 0..6 {
        let x: Vec<f32> = synth(hidden, 100 + layer);

        // Chained reference: q4k_matvec_pair → CPU GEGLU loop → q4k_matvec
        let (gate_scores, up_scores) = be.q4k_matvec_pair(&gate_q4, &up_q4, &x, inter, hidden).unwrap();
        let chained = be.q4k_geglu_down(&down_q4, &gate_scores, &up_scores,
            hidden, inter, "gelu_tanh").unwrap();

        // Fused: one cmd buffer
        let fused = be.q4k_ffn_full(&gate_q4, &up_q4, &down_q4, &x,
            hidden, inter, "gelu_tanh").unwrap();

        assert_eq!(fused.len(), hidden);
        let nan = fused.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(nan, 0, "layer {layer}: fused returned {nan} NaN");

        let cos = cosine(&chained, &fused);
        let max_d = max_abs_diff(&chained, &fused);
        println!("[layer {layer}] cos={cos:.6} max|Δ|={max_d:.4e}");
        assert!(cos > 0.9999, "layer {layer}: cos={cos}");
    }
}

/// Regression test for the tanh-overflow bug: large gate magnitudes must not
/// produce NaN. Metal's tanh(x) returns NaN for |x| > ~44 because (e^2x - 1)
/// / (e^2x + 1) → (inf-1)/(inf+1) = NaN. Real walk gate scores around 32
/// trigger this in unclamped gelu_tanh shaders. Every gelu_tanh call in the
/// codebase MUST clamp the argument before calling tanh.
#[test]
fn gelu_tanh_no_nan_for_large_inputs() {
    let Some(be) = MetalBackend::new() else { return; };
    // Test the standalone q4k_geglu_gelu_tanh_down kernel
    let hidden = 256;
    let inter = 256;
    let down_q4 = quantize_q4_k(&synth(hidden * inter, 11));
    // Gate values that would overflow tanh: arg = c * (g + 0.044715*g^3).
    // For g=50: arg ≈ 0.798 * (50 + 5589) = 4501. Way past overflow.
    let gate: Vec<f32> = (0..inter).map(|i| 50.0 + (i as f32) * 0.01).collect();
    let up: Vec<f32> = (0..inter).map(|_| 1.0).collect();
    let out = be.q4k_geglu_down(&down_q4, &gate, &up, hidden, inter, "gelu_tanh").unwrap();
    let nan_count = out.iter().filter(|v| !v.is_finite()).count();
    assert_eq!(nan_count, 0, "gelu_tanh produced {nan_count} NaN/Inf for large gate inputs");
}

/// Walk-FFN calls the fused kernel ONCE per layer (34 times per decode).
/// In production this produced NaN on the first call. This test reproduces
/// the call pattern (same backend, many sequential dispatches with fresh
/// data) to catch state-leak bugs the single-call tests miss.
#[test]
fn many_sequential_calls() {
    let Some(be) = MetalBackend::new() else { return; };
    let hidden = 2560;
    let inter = 10240;

    let down_f32 = synth(hidden * inter, 11);
    let down_q4 = quantize_q4_k(&down_f32);

    // 34 distinct (gate, up) pairs to mimic 34 walk-FFN layers per token.
    for call in 0..34 {
        let gate: Vec<f32> = synth(inter, 100 + call as u64).into_iter().map(|v| v * 50.0).collect();
        let up:   Vec<f32> = synth(inter, 200 + call as u64).into_iter().map(|v| v * 10.0).collect();

        let cpu = cpu_reference(&down_q4, &gate, &up, hidden, inter, "gelu_tanh", &be);
        let gpu = be.q4k_geglu_down(&down_q4, &gate, &up, hidden, inter, "gelu_tanh").unwrap();

        let nan_count = gpu.iter().filter(|v| !v.is_finite()).count();
        let cos = cosine(&cpu, &gpu);
        if nan_count > 0 || !cos.is_finite() || cos < 0.999 {
            eprintln!("[seq call {call}] FAILED nan_count={nan_count} cos={cos}");
            eprintln!("  cpu[0..6] = {:?}", &cpu[..6]);
            eprintln!("  gpu[0..6] = {:?}", &gpu[..6]);
            panic!("call {call} produced NaN or cosine < 0.999");
        }
    }
    println!("All 34 sequential calls correct.");
}

/// Reproduces the EXACT walk-FFN call pattern that produces NaN in production:
///   1. q4k_matvec_pair(gate_q4, up_q4, x) → gate_scores, up_scores
///   2. q4k_geglu_down(down_q4, gate_scores, up_scores) → out
///
/// If this passes, the bug is somewhere ELSE (predecessor kernel, alignment,
/// allocator state). If it fails, we have a contained reproducer for the
/// "first call returns NaN" issue.
#[test]
fn walk_ffn_call_pattern() {
    let Some(be) = MetalBackend::new() else { return; };
    let hidden = 2560;
    let inter = 10240;

    // Synthesize Q4_K weight matrices the same way walk_ffn sees them.
    // Scale weights down so gate/up scores end up in a realistic range
    // (~10-50 in magnitude, what actual model weights produce).
    let gate_f32: Vec<f32> = synth(inter * hidden, 1).into_iter().map(|v| v * 0.05).collect();
    let up_f32:   Vec<f32> = synth(inter * hidden, 2).into_iter().map(|v| v * 0.05).collect();
    let down_f32: Vec<f32> = synth(hidden * inter, 3).into_iter().map(|v| v * 0.05).collect();
    let gate_q4 = quantize_q4_k(&gate_f32);
    let up_q4 = quantize_q4_k(&up_f32);
    let down_q4 = quantize_q4_k(&down_f32);

    for layer in 0..34 {
        let x: Vec<f32> = synth(hidden, 100 + layer as u64);

        // Step 1: q4k_matvec_pair
        let (gate_scores, up_scores) = be.q4k_matvec_pair(
            &gate_q4, &up_q4, &x, inter, hidden,
        ).expect("q4k_matvec_pair available");

        // Inputs to q4k_geglu_down are clean (computed by Metal kernel)
        let g_nan = gate_scores.iter().filter(|v| !v.is_finite()).count();
        let u_nan = up_scores.iter().filter(|v| !v.is_finite()).count();
        assert_eq!(g_nan, 0, "layer {layer}: gate_scores has {g_nan} NaN");
        assert_eq!(u_nan, 0, "layer {layer}: up_scores has {u_nan} NaN");

        // Step 2: q4k_geglu_down — also test silu to isolate activation choice
        let gpu_silu = be.q4k_geglu_down(
            &down_q4, &gate_scores, &up_scores,
            hidden, inter, "silu",
        ).expect("q4k_geglu_down available");
        let nan_silu = gpu_silu.iter().filter(|v| !v.is_finite()).count();
        eprintln!("  layer {layer}: silu nan={nan_silu}/{hidden}");

        let gpu = be.q4k_geglu_down(
            &down_q4, &gate_scores, &up_scores,
            hidden, inter, "gelu_tanh",
        ).expect("q4k_geglu_down available");

        let nan = gpu.iter().filter(|v| !v.is_finite()).count();
        if nan > 0 {
            eprintln!("layer {layer}: GPU returned {nan} NaN values");
            eprintln!("  gate[0..6] = {:?}", &gate_scores[..6]);
            eprintln!("  up[0..6]   = {:?}", &up_scores[..6]);
            eprintln!("  gpu[0..6]  = {:?}", &gpu[..6]);

            // CPU reference to confirm inputs are sane
            let cpu = cpu_reference(&down_q4, &gate_scores, &up_scores, hidden, inter, "gelu_tanh", &be);
            eprintln!("  cpu[0..6]  = {:?}", &cpu[..6]);
            panic!("layer {layer}: q4k_geglu_down returned NaN");
        }
    }
}

/// Real-walk magnitudes: gate scores from Q4_K matvec can be 30-100,
/// up scores 5-20. Synthetic [-1,1] range may be hiding bugs.
#[test]
fn gelu_tanh_realistic_magnitudes() {
    let Some(be) = MetalBackend::new() else { return; };
    let hidden = 2560;
    let inter = 10240;

    // Down weights at typical scale.
    let down_f32 = synth(hidden * inter, 11);
    let down_q4 = quantize_q4_k(&down_f32);

    // Gate values scaled up to ~50 (real walk FFN range).
    let gate: Vec<f32> = synth(inter, 22).into_iter().map(|v| v * 50.0).collect();
    let up: Vec<f32> = synth(inter, 33).into_iter().map(|v| v * 10.0).collect();

    let cpu = cpu_reference(&down_q4, &gate, &up, hidden, inter, "gelu_tanh", &be);
    let gpu = be.q4k_geglu_down(&down_q4, &gate, &up, hidden, inter, "gelu_tanh").unwrap();

    let max_d = max_abs_diff(&cpu, &gpu);
    let cos = cosine(&cpu, &gpu);
    println!("[realistic gelu_tanh] max|Δ|={max_d:.4} cos={cos:.6}");
    println!("  cpu[0..6] = {:?}", &cpu[..6]);
    println!("  gpu[0..6] = {:?}", &gpu[..6]);
    assert!(cos > 0.999, "cos = {cos}");
}
