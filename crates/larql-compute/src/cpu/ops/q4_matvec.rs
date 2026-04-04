//! Q4×Q8 matrix-vector multiply via C kernel.
//!
//! scores[N] = Q4[N, K] @ x[K]
//!
//! Internally quantizes x to Q8, then calls the C kernel with
//! ARM vdotq_s32 intrinsics. 0.95ms on 14.7MB matrix (M3 Max).

use super::q4_common::{q4_0_matvec_c, quantize_to_q8};

/// Q4 matvec: scores = Q4_matrix @ x.
/// Pre-quantizes x to Q8 internally.
pub fn dispatch(q4_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let (q8_x, q8_scales) = quantize_to_q8(x);
    dispatch_q8(q4_data, &q8_x, &q8_scales, num_rows, hidden)
}

/// Q4 matvec with pre-quantized Q8 input (avoids re-quantizing).
pub fn dispatch_q8(q4_data: &[u8], q8_x: &[i8], q8_scales: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let mut scores = vec![0.0f32; num_rows];
    unsafe {
        q4_0_matvec_c(
            q4_data.as_ptr(), q8_x.as_ptr(), q8_scales.as_ptr(),
            scores.as_mut_ptr(), num_rows, hidden,
        );
    }
    scores
}

#[cfg(test)]
mod tests {
    use super::*;

    fn quantize_q4_0_test(data: &[f32]) -> Vec<u8> {
        assert!(data.len() % 32 == 0);
        let n = data.len() / 32;
        let mut out = Vec::with_capacity(n * 18);
        for i in 0..n {
            let blk = &data[i * 32..(i + 1) * 32];
            let amax = blk.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            let scale = amax / 7.0;
            let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
            let bits = scale.to_bits();
            let sign = (bits >> 16) & 0x8000;
            let exp = ((bits >> 23) & 0xFF) as i32;
            let mant = bits & 0x7FFFFF;
            let f16 = if exp == 0 { sign as u16 }
                else if exp >= 31 + 127 - 15 { (sign | 0x7C00) as u16 }
                else if exp <= -15 + 127 { sign as u16 }
                else { (sign | (((exp - 127 + 15) as u32) << 10) | (mant >> 13)) as u16 };
            out.extend_from_slice(&f16.to_le_bytes());
            for j in 0..16 {
                let lo = ((blk[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
                let hi = ((blk[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
                out.push(lo | (hi << 4));
            }
        }
        out
    }

    #[test]
    fn q4_matvec_produces_output() {
        let hidden = 256;
        let rows = 64;
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0_test(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert_eq!(result.len(), rows);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_matvec_zero_input() {
        let hidden = 256;
        let rows = 32;
        let x = vec![0.0f32; hidden];
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0_test(&matrix);
        let result = dispatch(&q4, &x, rows, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }
}
