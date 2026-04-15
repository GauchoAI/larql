//! CPU reference implementation for Q4_K matrix-vector multiply.
//!
//! Mirrors the Metal shader `q4k_matvec` exactly for cross-backend testing.
//! Not optimised — scalar code intended as a correctness reference.

/// Q4_K super-block size: 148 bytes per 256 values.
const Q4K_BLOCK_SIZE: usize = 148;

/// Decode f16 bits to f32.
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits >> 15) & 1) as u32;
    let exp = ((bits >> 10) & 0x1F) as i32;
    let mant = (bits & 0x3FF) as u32;
    if exp == 0 {
        if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
        let val = mant as f32 / 1024.0 * 2.0f32.powi(-14);
        return if sign == 1 { -val } else { val };
    }
    if exp == 31 {
        return if mant == 0 {
            if sign == 1 { f32::NEG_INFINITY } else { f32::INFINITY }
        } else { f32::NAN };
    }
    let val = (1.0 + mant as f32 / 1024.0) * 2.0f32.powi(exp - 15);
    if sign == 1 { -val } else { val }
}

/// CPU Q4_K matvec: out[N] = Q4_K[N, K] @ x[K].
///
/// Mirrors the Metal `q4k_matvec` shader: per-row dot product over super-blocks.
pub fn dispatch(q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;

        for sb in 0..superblocks {
            let block = &q4k_data[row_start + sb * Q4K_BLOCK_SIZE..];

            // Read super-block header
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);

            // Unpack 8 × 6-bit scales from bytes 4-15
            let sc_bytes = &block[4..16];
            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];

            for j in 0..4 {
                scales[j] = (sc_bytes[j] & 0x3F) as f32;
                scales[j + 4] = (sc_bytes[j + 4] & 0x3F) as f32;
            }

            // Unpack 4-bit mins from bytes 16-19
            let min_bytes = &block[16..20];
            for j in 0..4 {
                mins[j] = (min_bytes[j] & 0x0F) as f32;
                mins[j + 4] = ((min_bytes[j] >> 4) & 0x0F) as f32;
            }

            // Read 256 × 4-bit values (128 packed bytes starting at offset 20)
            let quants = &block[20..];
            let x_base = sb * 256;

            for j in 0..8 {
                let sc = d * scales[j];
                let mn = dmin * mins[j];
                let qb = &quants[j * 16..];

                for (i, &qb_val) in qb.iter().enumerate().take(16) {
                    let xi = x_base + j * 32 + i * 2;
                    let lo = (qb_val & 0x0F) as f32;
                    let hi = ((qb_val >> 4) & 0x0F) as f32;
                    acc += (sc * lo - mn) * x[xi];
                    acc += (sc * hi - mn) * x[xi + 1];
                }
            }
        }
        *out_val = acc;
    }
    out
}

/// Q4_K GGUF super-block size: 144 bytes per 256 values (llama.cpp layout).
const Q4K_GGUF_BLOCK_SIZE: usize = 144;

/// CPU scalar Q4_K GGUF matvec: `out[N] = W_q4k_gguf[N, K] @ x[K]`. Mirrors the
/// Metal `q4kf_proj` kernel byte-for-byte (same 12-byte scale+min unpacking,
/// same dequant formula `sc * q - mn`, same interleaved qs layout where byte
/// `qs[j/2 + l]` low nibble = L[j+l] and high nibble = L[j+l+32] for
/// j ∈ {0, 64, 128, 192}).
pub fn dispatch_gguf(q4k_data: &[u8], x: &[f32], num_rows: usize, hidden: usize) -> Vec<f32> {
    use crate::cpu::ops::q4_common::gguf_q4k_unpack_scales_mins;
    let superblocks = hidden / 256;
    let bytes_per_row = superblocks * Q4K_GGUF_BLOCK_SIZE;
    let mut out = vec![0.0f32; num_rows];

    for (row, out_val) in out.iter_mut().enumerate().take(num_rows) {
        let row_start = row * bytes_per_row;
        let mut acc = 0.0f32;
        for sb in 0..superblocks {
            let block = &q4k_data[row_start + sb * Q4K_GGUF_BLOCK_SIZE..];
            let d = f16_to_f32(u16::from_le_bytes([block[0], block[1]]));
            let dmin = f16_to_f32(u16::from_le_bytes([block[2], block[3]]));
            let (ls, lm) = gguf_q4k_unpack_scales_mins(&block[4..16]);
            let qs = &block[16..144]; // 128 bytes
            let x_base = sb * 256;
            // Iterate over the 4 sub-block pairs (0,1), (2,3), (4,5), (6,7).
            for pair_idx in 0..4 {
                let j_low = pair_idx * 64; // values of first sub-block in pair
                let sub_lo = 2 * pair_idx; // sub-block index for low nibble
                let sub_hi = 2 * pair_idx + 1; // sub-block index for high nibble
                let sc_lo = d * ls[sub_lo] as f32;
                let mn_lo = dmin * lm[sub_lo] as f32;
                let sc_hi = d * ls[sub_hi] as f32;
                let mn_hi = dmin * lm[sub_hi] as f32;
                let pair_bytes = &qs[pair_idx * 32..(pair_idx + 1) * 32];
                for l in 0..32 {
                    let b = pair_bytes[l];
                    let v_lo = (b & 0x0F) as f32;
                    let v_hi = ((b >> 4) & 0x0F) as f32;
                    let x_lo = x[x_base + j_low + l];
                    let x_hi = x[x_base + j_low + l + 32];
                    acc += (sc_lo * v_lo - mn_lo) * x_lo;
                    acc += (sc_hi * v_hi - mn_hi) * x_hi;
                }
            }
        }
        *out_val = acc;
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::ops::q4_common::{quantize_q4_k, quantize_q4_k_gguf};

    #[test]
    fn q4k_produces_nonzero() {
        let hidden = 256;
        let rows = 4;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4k = quantize_q4_k(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let out = dispatch(&q4k, &x, rows, hidden);
        assert!(out.iter().any(|&v| v.abs() > 0.001), "Q4_K matvec should produce nonzero");
    }

    #[test]
    fn q4k_gguf_roundtrip_matches_source_within_tolerance() {
        // Sparse-spike distribution similar to real Gemma 3 weights.
        let hidden = 256;
        let rows = 1;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| {
            let t = i as f32 * 0.021;
            let sm = t.sin() * 0.02;
            if i % 16 == 0 { 0.30 * t.cos().signum() } else { sm }
        }).collect();
        let q = quantize_q4_k_gguf(&matrix);
        assert_eq!(q.len(), rows * hidden / 256 * Q4K_GGUF_BLOCK_SIZE);
        // One-hot input: dispatch yields the dequantised column of the weight matrix.
        let mut max_err = 0.0f32;
        for col in 0..hidden {
            let mut xv = vec![0.0f32; hidden];
            xv[col] = 1.0;
            let o = dispatch_gguf(&q, &xv, rows, hidden);
            for r in 0..rows {
                let err = (matrix[r * hidden + col] - o[r]).abs();
                if err > max_err { max_err = err; }
            }
        }
        let amax = matrix.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let rel = 100.0 * max_err / amax.max(1e-9);
        println!("Q4K GGUF roundtrip: amax={amax:.3} max_err={max_err:.5} rel={rel:.3}%");
        // Q4_K_M target: ~1-2% on real weights. Synthetic noise a bit higher.
        assert!(rel < 5.0, "Q4K GGUF roundtrip rel error {rel}% too high");
    }
}
