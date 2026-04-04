//! Q4 vector-matrix multiply via C kernel (scatter-accumulate).
//!
//! out[K] = activation[N] @ Q4[N, K]

use super::q4_common::q4_0_vecmat_c;

/// Q4 vecmat: out = activation @ Q4_matrix.
pub fn dispatch(activation: &[f32], q4_data: &[u8], intermediate: usize, hidden: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; hidden];
    unsafe {
        q4_0_vecmat_c(
            activation.as_ptr(), q4_data.as_ptr(),
            out.as_mut_ptr(), intermediate, hidden,
        );
    }
    out
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
    fn q4_vecmat_produces_output() {
        let hidden = 256;
        let inter = 128;
        let act: Vec<f32> = (0..inter).map(|i| if i % 3 == 0 { 1.0 } else { 0.0 }).collect();
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0_test(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert_eq!(result.len(), hidden);
        assert!(result.iter().any(|&v| v.abs() > 0.01));
    }

    #[test]
    fn q4_vecmat_zero_activation() {
        let hidden = 256;
        let inter = 64;
        let act = vec![0.0f32; inter];
        let matrix: Vec<f32> = (0..inter * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0_test(&matrix);
        let result = dispatch(&act, &q4, inter, hidden);
        assert!(result.iter().all(|&v| v.abs() < 0.01));
    }
}
