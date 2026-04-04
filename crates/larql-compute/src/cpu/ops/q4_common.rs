//! Shared Q4 utilities for CPU backend.
//!
//! C FFI declarations for the vdotq_s32 kernel (csrc/q4_dot.c)
//! and Q8 quantization helper.

extern "C" {
    /// C kernel: Q4_0 × Q8_0 matrix-vector multiply with ARM vdotq_s32.
    pub fn q4_0_matvec_c(
        q4_data: *const u8,
        q8_x: *const i8,
        q8_scales: *const f32,
        scores: *mut f32,
        num_rows: usize,
        hidden: usize,
    );

    /// C kernel: Q4_0 vector-matrix multiply (scatter-accumulate).
    pub fn q4_0_vecmat_c(
        activation: *const f32,
        q4_data: *const u8,
        out: *mut f32,
        intermediate: usize,
        hidden: usize,
    );
}

/// Pre-quantize f32 vector to Q8_0 (int8 + per-block f32 scale).
pub fn quantize_to_q8(x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n_blocks = x.len() / 32;
    let mut q8 = vec![0i8; x.len()];
    let mut scales = vec![0.0f32; n_blocks];
    for b in 0..n_blocks {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        scales[b] = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_quantize_round_trip() {
        let x: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let (q8, scales) = quantize_to_q8(&x);
        assert_eq!(q8.len(), 64);
        assert_eq!(scales.len(), 2); // 64 / 32
        assert!(scales.iter().all(|&s| s >= 0.0));
    }

    #[test]
    fn q8_zero_input() {
        let x = vec![0.0f32; 32];
        let (q8, scales) = quantize_to_q8(&x);
        assert!(q8.iter().all(|&v| v == 0));
        assert!(scales[0] == 0.0);
    }
}
