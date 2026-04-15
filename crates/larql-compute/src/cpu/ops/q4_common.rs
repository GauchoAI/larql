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
    for (b, scale_out) in scales.iter_mut().enumerate().take(n_blocks) {
        let off = b * 32;
        let block = &x[off..off + 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 127.0;
        *scale_out = scale;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        for j in 0..32 {
            q8[off + j] = (block[j] * inv).round().clamp(-128.0, 127.0) as i8;
        }
    }
    (q8, scales)
}

/// Quantize f32 data to Q4_0 format (4-bit, block size 32).
///
/// Each block of 32 floats becomes 18 bytes: 2 bytes f16 scale + 16 bytes packed nibbles.
/// Used for weight quantization in benchmarks, tests, and tooling.
pub fn quantize_q4_0(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(32), "data length must be a multiple of 32");
    let n_blocks = data.len() / 32;
    let mut out = Vec::with_capacity(n_blocks * 18);
    for i in 0..n_blocks {
        let block = &data[i * 32..(i + 1) * 32];
        let amax = block.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
        let scale = amax / 7.0;
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };
        // f32 → f16 conversion
        let bits = scale.to_bits();
        let sign = (bits >> 16) & 0x8000;
        let exp = ((bits >> 23) & 0xFF) as i32;
        let mant = bits & 0x7FFFFF;
        let f16 = if exp == 0 { sign as u16 }
            else if exp == 255 { (sign | 0x7C00 | (mant >> 13)) as u16 }
            else {
                let new_exp = exp - 127 + 15;
                if new_exp >= 31 { (sign | 0x7C00) as u16 }
                else if new_exp <= 0 { sign as u16 }
                else { (sign | ((new_exp as u32) << 10) | (mant >> 13)) as u16 }
            };
        out.extend_from_slice(&f16.to_le_bytes());
        for j in 0..16 {
            let lo = ((block[j * 2] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            let hi = ((block[j * 2 + 1] * inv).round() as i32 + 8).clamp(0, 15) as u8;
            out.push(lo | (hi << 4));
        }
    }
    out
}

/// Encode f32 to f16 bits (for quantize helpers). Handles normals, subnormals,
/// infinities and NaN. Rounds to nearest (ties round up — simpler than
/// round-half-even and within 1 ulp either way, which is fine for Q-scales).
fn f32_to_f16(val: f32) -> u16 {
    let bits = val.to_bits();
    let sign = (bits >> 16) & 0x8000;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;
    // f32 zero / f32 subnormal → f16 zero.
    if exp == 0 { return sign as u16; }
    // f32 inf / nan.
    if exp == 255 {
        let m16 = mant >> 13;
        // Preserve NaN (non-zero mantissa) vs inf (zero mantissa).
        return (sign | 0x7C00 | if mant != 0 && m16 == 0 { 1 } else { m16 }) as u16;
    }
    let new_exp = exp - 127 + 15;
    // f16 overflow → inf with correct sign.
    if new_exp >= 31 { return (sign | 0x7C00) as u16; }
    if new_exp > 0 {
        // Normal f16. Round to nearest (half-up) on the discarded 13 mantissa bits.
        let base = (new_exp as u32) << 10 | (mant >> 13);
        let round = if (mant & 0x1000) != 0 { 1 } else { 0 };
        let rounded = base + round;
        // If rounding bumped exponent past max, emit inf.
        if rounded >= 0x7C00 { return (sign | 0x7C00) as u16; }
        return (sign | rounded) as u16;
    }
    // Subnormal f16: value = mant_sub * 2^-24, mant_sub ∈ [0..1024).
    // full_m = (1 << 23) | mant is the 24-bit mantissa with implicit leading 1.
    // mant_sub = full_m >> (14 - new_exp), with round-to-nearest on the shifted-out bits.
    let full_m = (1u32 << 23) | mant;
    let shift = (14 - new_exp) as u32;
    if shift >= 25 { return sign as u16; } // underflows even subnormal range → zero
    let shifted = full_m >> shift;
    let round_bit = (full_m >> (shift - 1)) & 1;
    let mant_sub = shifted + round_bit;
    // mant_sub == 1024 means we rounded up into the smallest normal — that's correct,
    // exponent 1 with mantissa 0 is the smallest normal, bit pattern 0x0400.
    (sign | mant_sub) as u16
}

/// llama.cpp's make_qkx2_quants — iterative per-sub-block (scale, min) search.
///
/// Finds (scale, the_min ≥ 0) such that `scale * L[i] - the_min ≈ x[i]` with
/// L[i] ∈ [0, nmax], minimising ∑ w[i] (scale·L[i] − the_min − x[i])², where
/// w[i] = x[i]² (importance weighting — channels with large values matter
/// more to downstream matmul accuracy). Writes the quantised codes into
/// `l_out` and returns `(scale, the_min)`.
///
/// The iteration perturbs the initial 1/scale estimate by (rmin + t·rdelta)/range
/// for t ∈ [0..nstep], re-fits (scale, min) via weighted least-squares given
/// the resulting integer codes, and keeps the candidate with lowest weighted
/// squared error. For Q4_K we call it with (nmax=15, rmin=-1, rdelta=0.1,
/// nstep=20) — the same params llama.cpp uses.
fn make_qkx2_quants(
    x: &[f32],
    nmax: u8,
    rmin: f32,
    rdelta: f32,
    nstep: usize,
    l_out: &mut [u8],
) -> (f32, f32) {
    let n = x.len();
    assert_eq!(l_out.len(), n);
    let nmax_f = nmax as f32;

    // Importance weights — llama.cpp's Q4_K quantiser uses |x|, not x². Using
    // x² over-biases the spike channels and leaves the smooth background
    // under-represented, producing systematically biased output at rms time
    // (we saw 5% max with x², vs. ~1% with |x| which is what llama.cpp gets).
    let mut sum_w = 0.0f32;
    let mut sum_x = 0.0f32;
    let mut x_min = f32::INFINITY;
    let mut x_max = f32::NEG_INFINITY;
    let weights: Vec<f32> = x.iter().map(|v| {
        let w = v.abs();
        sum_w += w;
        sum_x += w * v;
        if *v < x_min { x_min = *v; }
        if *v > x_max { x_max = *v; }
        w
    }).collect();
    if x_min > 0.0 { x_min = 0.0; } // Q4_K mn_j is ≥ 0, so min clamps to 0 from above.

    if (x_max - x_min).abs() < 1e-30 {
        for v in l_out.iter_mut() { *v = 0; }
        return (0.0, -x_min);
    }
    if sum_w < 1e-30 {
        // All-zero weights (all x are 0) — same all-zero L, scale=0.
        for v in l_out.iter_mut() { *v = 0; }
        return (0.0, -x_min);
    }

    // Initial candidate: scale = (x_max - x_min) / nmax.
    let iscale0 = nmax_f / (x_max - x_min);
    let mut best_scale = 1.0 / iscale0;
    let mut best_min = -x_min;
    for (i, &v) in x.iter().enumerate() {
        l_out[i] = ((v - x_min) * iscale0).round().clamp(0.0, nmax_f) as u8;
    }
    let mut best_err = 0.0f32;
    for i in 0..n {
        let diff = best_scale * l_out[i] as f32 - best_min - x[i];
        best_err += weights[i] * diff * diff;
    }

    let mut laux = vec![0u8; n];
    for t in 0..=nstep {
        let iscale_trial = (rmin + rdelta * t as f32 + nmax_f) / (x_max - x_min);
        if iscale_trial <= 0.0 { continue; }
        // Quantise under the trial inv-scale; accumulate weighted moments to
        // re-fit (scale, min) in closed form given these integer codes.
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = ((x[i] - x_min) * iscale_trial).round().clamp(0.0, nmax_f) as u8;
            laux[i] = l;
            let lf = l as f32;
            sum_l += weights[i] * lf;
            sum_l2 += weights[i] * lf * lf;
            sum_xl += weights[i] * lf * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det <= 0.0 { continue; }
        let this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
        let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
        // Q4_K format: min offset the_min = -this_min ≥ 0. If LSQ solution
        // has this_min > 0 (i.e. the_min < 0, impossible to encode), force
        // this_min = 0 and refit scale alone.
        let (this_scale, this_min) = if this_min > 0.0 {
            this_min = 0.0;
            (if sum_l2 > 1e-30 { sum_xl / sum_l2 } else { 0.0 }, this_min)
        } else {
            (this_scale, this_min)
        };
        if this_scale <= 0.0 { continue; }
        let mut err = 0.0f32;
        for i in 0..n {
            let diff = this_scale * laux[i] as f32 + this_min - x[i];
            err += weights[i] * diff * diff;
        }
        if err < best_err {
            best_err = err;
            best_scale = this_scale;
            best_min = -this_min;
            l_out.copy_from_slice(&laux);
        }
    }
    (best_scale, best_min)
}

/// Quantize f32 data to Q4_K format (4-bit with sub-block scales, Ollama-compatible).
///
/// Each super-block of 256 floats becomes 148 bytes:
///   [0..1]    f16 d (delta)
///   [2..3]    f16 dmin (minimum)
///   [4..15]   12 bytes: 8 × 6-bit sub-block scales (packed)
///   [16..19]  4 bytes: 8 × 4-bit sub-block mins (packed)
///   [20..147] 128 bytes: 256 × 4-bit values (packed nibbles)
///
/// Per sub-block we iteratively refine (scale, min) via `make_qkx2_quants`
/// (the llama.cpp algorithm). This cuts max dequant error from ~5% (single-
/// round `(range/15)/d`) to ~1-2% on Gemma 3 weights.
pub fn quantize_q4_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 148);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Per sub-block: run iterative (scale, min) refinement.
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];
        let mut l_all = [0u8; 256];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let (sc, mn) = make_qkx2_quants(
                sub, 15, -1.0, 0.1, 20, &mut l_all[j * 32..(j + 1) * 32]);
            sub_scales[j] = sc;
            sub_mins[j] = mn;
        }

        // Super-block d / dmin: pack the sub-block (scale, min) into
        // 6-bit / 4-bit unsigned codes.
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().cloned().fold(0.0f32, f32::max);
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 15.0 / max_min } else { 0.0 };
        let d = if inv_scale > 0.0 { 1.0 / inv_scale } else { 0.0 };
        let dmin = if inv_min > 0.0 { 1.0 / inv_min } else { 0.0 };

        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..8 {
            q_scales[j] = (sub_scales[j] * inv_scale).round().clamp(0.0, 63.0) as u8;
            q_mins[j] = (sub_mins[j] * inv_min).round().clamp(0.0, 15.0) as u8;
        }

        // Pack 6-bit scales (only lower 6 bits of first 8 bytes — last 4 bytes
        // are unused in this Ollama-style layout).
        let mut sc_packed = [0u8; 12];
        for j in 0..8 {
            sc_packed[j] = q_scales[j] & 0x3F;
        }
        out.extend_from_slice(&sc_packed);

        // Pack 4-bit mins into 4 bytes.
        let mut min_packed = [0u8; 4];
        for j in 0..4 {
            min_packed[j] = (q_mins[j] & 0x0F) | ((q_mins[j + 4] & 0x0F) << 4);
        }
        out.extend_from_slice(&min_packed);

        // Re-quantize values using the *encoded* (d·q_scales, dmin·q_mins)
        // rather than the ideal (sub_scales, sub_mins), since the 6-bit/4-bit
        // rounding of the super-block codes can shift each sub-block's effective
        // (scale, min) slightly. Matching the quantisation to the exact numbers
        // the kernel dequantises with minimises final dequant error.
        for j in 0..8 {
            let sc = d * q_scales[j] as f32;
            let mn = dmin * q_mins[j] as f32;
            let inv_sc = if sc > 1e-30 { 1.0 / sc } else { 0.0 };
            let sub = &block[j * 32..(j + 1) * 32];
            for i in 0..16 {
                let v0 = ((sub[i * 2] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                let v1 = ((sub[i * 2 + 1] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
                out.push(v0 | (v1 << 4));
            }
        }
        let _ = l_all; // L values from the iteration are not used directly; we
        // re-derive them above using the encoded d/dmin scales to minimise the
        // mismatch between the kernel's dequantisation and what we quantised to.
    }
    out
}

/// Quantize f32 data to Q6_K format (6-bit with sub-block scales, Ollama-compatible).
///
/// Each super-block of 256 floats becomes 210 bytes:
///   [0..127]    128 bytes: lower 4 bits of each value (packed nibbles)
///   [128..191]   64 bytes: upper 2 bits (packed, 4 per byte)
///   [192..207]   16 bytes: 16 × int8 scales (one per 16-value sub-block)
///   [208..209]    2 bytes: f16 super-block scale (d)
pub fn quantize_q6_k(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 210);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Kernel formula: val = (d * sub_scale) * (raw - 32) with raw ∈ [0..63],
        // so q = raw - 32 ∈ [-32..31]. Per sub-block j, the *effective* scale
        // we'd want is s_j = sub_amax_j / 31 so q=31 recovers sub_amax_j.
        // Then we pack s_j into (f16 d) × (i8 sub_scale) by choosing
        //   d        = max_s / 127            (uses full positive i8 range)
        //   sub[j]   = round(s_j / d) ∈ [0..127]
        // This is the llama.cpp Q6_K layout. The previous formula derived d
        // from the global amax and computed sub_scale from there, which
        // collapsed sub_scale to 0 or 1 for sub-blocks much smaller than the
        // super-block max — losing ~all per-sub-block dynamic range.
        let mut s_per_sub = [0.0f32; 16];
        for (j, slot) in s_per_sub.iter_mut().enumerate() {
            let sub = &block[j * 16..(j + 1) * 16];
            let sub_amax = sub.iter().map(|v| v.abs()).fold(0.0f32, f32::max);
            *slot = sub_amax / 31.0;
        }
        let max_s = s_per_sub.iter().cloned().fold(0.0f32, f32::max);
        let d = max_s / 127.0;
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut sub_scales = [0i8; 16];
        for (j, sub_scale) in sub_scales.iter_mut().enumerate() {
            let sc = (s_per_sub[j] * inv_d).round().clamp(-128.0, 127.0);
            *sub_scale = sc as i8;
        }

        // Quantize all 256 values to 6-bit using effective scale (d * sub_scale).
        let mut q6_vals = [0u8; 256];
        for (j, &sub_scale) in sub_scales.iter().enumerate() {
            let sc = d * sub_scale as f32;
            let inv_sc = if sc.abs() > 1e-10 { 1.0 / sc } else { 0.0 };
            for i in 0..16 {
                let idx = j * 16 + i;
                let q = (block[idx] * inv_sc).round().clamp(-32.0, 31.0) as i8;
                q6_vals[idx] = (q + 32) as u8; // bias to unsigned
            }
        }

        // Pack lower 4 bits: 128 bytes (2 nibbles per byte)
        let mut ql = [0u8; 128];
        for i in 0..128 {
            ql[i] = (q6_vals[i * 2] & 0x0F) | ((q6_vals[i * 2 + 1] & 0x0F) << 4);
        }
        out.extend_from_slice(&ql);

        // Pack upper 2 bits: 64 bytes (4 × 2 bits per byte)
        let mut qh = [0u8; 64];
        for (i, &q6_val) in q6_vals.iter().enumerate() {
            let hi2 = (q6_val >> 4) & 0x03;
            let byte_idx = i / 4;
            let bit_offset = (i % 4) * 2;
            qh[byte_idx] |= hi2 << bit_offset;
        }
        out.extend_from_slice(&qh);

        // 16 × int8 scales
        for &s in &sub_scales {
            out.push(s as u8);
        }

        // f16 super-block scale
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
    }
    out
}

/// Quantize f32 to GGUF Q4_K format (144 bytes per 256 values).
///
/// This is llama.cpp's `block_q4_K` byte layout — matches ggml-quants.c exactly.
/// Our older 148-byte Q4_K layout (Ollama convention with 4-bit mins) is kept in
/// `quantize_q4_k` for backward compatibility; this GGUF form is preferred for
/// any model (e.g. Gemma 3) where the token distribution needs the extra 2 bits
/// of min resolution to stay on-token against the f32 reference.
///
/// Layout (144 bytes):
///   [0..1]    half d
///   [2..3]    half dmin
///   [4..15]   12 bytes: scales (0..3, 4..7) and mins (0..3, 4..7) each 6-bit, packed per llama.cpp
///   [16..143] 128 bytes: 256 × 4-bit nibbles
///
/// Scale+min packing (see llama.cpp ggml-quants.c `quantize_row_q4_K_ref`):
///   scales_bytes[0..3]:  bits 0-5 = low 6 of ls[0..3],  bits 6-7 = high 2 of ls[4..7]
///   scales_bytes[4..7]:  bits 0-5 = low 6 of lm[0..3],  bits 6-7 = high 2 of lm[4..7]
///   scales_bytes[8..11]: nibble 0 = low 4 of ls[4..7],  nibble 1 = low 4 of lm[4..7]
///
/// Per sub-block: iterative `make_qkx2_quants` refinement (same as `quantize_q4_k`),
/// but with the 6-bit dmin (max=63) giving the needed extra precision over Q4_K's
/// 4-bit mins (max=15).
pub fn quantize_q4_k_gguf(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256));
    let n_superblocks = data.len() / 256;
    let mut out = Vec::with_capacity(n_superblocks * 144);

    for sb in 0..n_superblocks {
        let block = &data[sb * 256..(sb + 1) * 256];

        // Per sub-block: iterative (scale, min) refinement — same algorithm
        // the 148-byte quantize_q4_k uses.
        let mut sub_scales = [0.0f32; 8];
        let mut sub_mins = [0.0f32; 8];
        let mut l_all = [0u8; 256];
        for j in 0..8 {
            let sub = &block[j * 32..(j + 1) * 32];
            let (sc, mn) = make_qkx2_quants(
                sub, 15, -1.0, 0.1, 20, &mut l_all[j * 32..(j + 1) * 32]);
            sub_scales[j] = sc;
            sub_mins[j] = mn;
        }

        // Super-block d / dmin pack sub-block (scale, min) into 6-bit / 6-bit
        // (both max 63, unlike Q4_K's 6/4 split).
        let max_scale = sub_scales.iter().cloned().fold(0.0f32, f32::max);
        let max_min = sub_mins.iter().cloned().fold(0.0f32, f32::max);
        let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
        let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
        let d = if inv_scale > 0.0 { 1.0 / inv_scale } else { 0.0 };
        let dmin = if inv_min > 0.0 { 1.0 / inv_min } else { 0.0 };

        let mut ls = [0u8; 8];
        let mut lm = [0u8; 8];
        for j in 0..8 {
            ls[j] = (sub_scales[j] * inv_scale).round().clamp(0.0, 63.0) as u8;
            lm[j] = (sub_mins[j] * inv_min).round().clamp(0.0, 63.0) as u8;
        }

        // Write d, dmin as f16.
        out.extend_from_slice(&f32_to_f16(d).to_le_bytes());
        out.extend_from_slice(&f32_to_f16(dmin).to_le_bytes());

        // Pack 12-byte scales array — llama.cpp's exact encoding.
        let mut packed = [0u8; 12];
        for j in 0..4 {
            // scales 0..3: low 6 bits of ls[j], upper 2 bits of ls[j+4] in bits 6-7
            packed[j] = (ls[j] & 0x3F) | ((ls[j + 4] >> 4) << 6);
            // scales 4..7: low 6 bits of lm[j], upper 2 bits of lm[j+4] in bits 6-7
            packed[j + 4] = (lm[j] & 0x3F) | ((lm[j + 4] >> 4) << 6);
            // scales 8..11: low 4 bits of ls[j+4] in nibble 0, low 4 bits of lm[j+4] in nibble 1
            packed[j + 8] = (ls[j + 4] & 0x0F) | ((lm[j + 4] & 0x0F) << 4);
        }
        out.extend_from_slice(&packed);

        // Quantize each value with the encoded (d·ls[j], dmin·lm[j]) scales.
        // Collect 256 4-bit codes indexed by absolute position k = j*32 + i in
        // the super-block, then pack them using llama.cpp's Q4_K qs layout:
        //
        //   qs[j/2 + l] = L[j+l] | (L[j+l+32] << 4)   for j in {0, 64}, l in [0..32)
        //
        // i.e. low nibble holds value k, high nibble holds value k+32. This pairs
        // together positions from sub-blocks (0,1), (2,3), (4,5), (6,7). Our old
        // layout paired consecutive values (0,1), (2,3), ..., within a sub-block
        // — a different byte order that the Metal `q4kf_proj` shader doesn't
        // accept.
        let mut l_codes = [0u8; 256];
        for j in 0..8 {
            let sc = d * ls[j] as f32;
            let mn = dmin * lm[j] as f32;
            let inv_sc = if sc > 1e-30 { 1.0 / sc } else { 0.0 };
            let sub = &block[j * 32..(j + 1) * 32];
            for i in 0..32 {
                l_codes[j * 32 + i] =
                    ((sub[i] + mn) * inv_sc).round().clamp(0.0, 15.0) as u8;
            }
        }
        // llama.cpp qs packing: 128 bytes, interleaving pairs of sub-blocks.
        for sb_pair_start in [0usize, 64].iter() {
            let j = *sb_pair_start;
            for l in 0..32 {
                let b = (l_codes[j + l] & 0x0F) | ((l_codes[j + l + 32] & 0x0F) << 4);
                out.push(b);
            }
        }
        // Sub-blocks (4,5) and (6,7) — another 64 bytes.
        for sb_pair_start in [128usize, 192].iter() {
            let j = *sb_pair_start;
            for l in 0..32 {
                let b = (l_codes[j + l] & 0x0F) | ((l_codes[j + l + 32] & 0x0F) << 4);
                out.push(b);
            }
        }
        let _ = l_all;
    }
    out
}

/// Dequantise one value from a GGUF Q4_K super-block. CPU reference used by the
/// scalar matvec in `q4k_matvec_gguf`, the new-format dequant test, and the
/// per-block diagnostics below.
#[inline]
pub fn gguf_q4k_unpack_scales_mins(scales: &[u8]) -> ([u8; 8], [u8; 8]) {
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..4 {
        // ls[0..3]: low 6 bits of scales[0..3]
        ls[j] = scales[j] & 0x3F;
        // lm[0..3]: low 6 bits of scales[4..7]
        lm[j] = scales[j + 4] & 0x3F;
        // ls[4..7]: (scales[j] >> 6) upper 2 bits << 4 | (scales[j+8] low nibble) low 4 bits
        ls[j + 4] = ((scales[j] >> 6) << 4) | (scales[j + 8] & 0x0F);
        // lm[4..7]: (scales[j+4] >> 6) upper 2 bits << 4 | (scales[j+8] high nibble) low 4 bits
        lm[j + 4] = ((scales[j + 4] >> 6) << 4) | ((scales[j + 8] >> 4) & 0x0F);
    }
    (ls, lm)
}

/// Convert Q4_K (148 bytes/block) to GGUF Q4_K (144 bytes/block) for fast GPU inference.
///
/// Processes a flat byte array of Q4_K superblocks. Each 148-byte block becomes 144 bytes.
/// Repacks scale/min headers from separate arrays into GGUF's interleaved 12-byte format.
/// Our 4-bit mins (0-15) fit within GGUF's 6-bit min range (0-63).
pub fn q4k_to_gguf(q4k_data: &[u8]) -> Vec<u8> {
    assert!(q4k_data.len().is_multiple_of(148), "Q4_K data must be a multiple of 148 bytes");
    let n_blocks = q4k_data.len() / 148;
    let mut out = Vec::with_capacity(n_blocks * 144);

    for i in 0..n_blocks {
        let block = &q4k_data[i * 148..];

        // Copy d, dmin (4 bytes — same in both formats)
        out.extend_from_slice(&block[0..4]);

        // Unpack our scales[12] + mins[4] into GGUF packed[12]
        let sc = &block[4..16];
        let mn = &block[16..20];

        let mut q_scales = [0u8; 8];
        let mut q_mins = [0u8; 8];
        for j in 0..4 {
            q_scales[j] = sc[j] & 0x3F;
            q_scales[j + 4] = sc[j + 4] & 0x3F;
            q_mins[j] = mn[j] & 0x0F;
            q_mins[j + 4] = (mn[j] >> 4) & 0x0F;
        }

        // Pack into GGUF format: 12 bytes
        let mut packed = [0u8; 12];
        for j in 0..4 {
            packed[j] = (q_scales[j] & 0x3F) | ((q_mins[j] & 0x03) << 6);
            packed[j + 4] = (q_scales[j + 4] & 0x3F) | ((q_mins[j + 4] & 0x03) << 6);
        }
        packed[8] = ((q_mins[0] >> 2) & 0x0F) | (((q_mins[1] >> 2) & 0x0F) << 4);
        packed[9] = ((q_mins[2] >> 2) & 0x0F) | (((q_mins[3] >> 2) & 0x0F) << 4);
        packed[10] = ((q_mins[4] >> 2) & 0x0F) | (((q_mins[5] >> 2) & 0x0F) << 4);
        packed[11] = ((q_mins[6] >> 2) & 0x0F) | (((q_mins[7] >> 2) & 0x0F) << 4);
        out.extend_from_slice(&packed);

        // Copy nibbles unchanged (128 bytes)
        out.extend_from_slice(&block[20..148]);
    }
    out
}

/// Convert Q4_K data to Q4_KF (pre-baked half scales) for fast GPU inference.
///
/// Q4_KF eliminates ALL header decode + scale unpack from the inference hot loop.
/// Each 148-byte Q4_K superblock becomes 160 bytes:
///   [0..15]    8 × f16 pre-computed d*scale_j (16 bytes)
///   [16..31]   8 × f16 pre-computed dmin*min_j (16 bytes)
///   [32..159]  128 bytes nibbles (unchanged)
pub fn q4k_to_q4kf(q4k_data: &[u8], num_rows: usize, hidden: usize) -> Vec<u8> {
    let superblocks_per_row = hidden / 256;
    let q4k_bytes_per_row = superblocks_per_row * 148;
    let q4kf_bytes_per_row = superblocks_per_row * 160;
    let mut out = Vec::with_capacity(num_rows * q4kf_bytes_per_row);

    for row in 0..num_rows {
        for sb in 0..superblocks_per_row {
            let offset = row * q4k_bytes_per_row + sb * 148;
            let block = &q4k_data[offset..];

            // Decode Q4_K header
            let d_bits = u16::from_le_bytes([block[0], block[1]]);
            let dmin_bits = u16::from_le_bytes([block[2], block[3]]);
            let d = f16_to_f32(d_bits);
            let dmin = f16_to_f32(dmin_bits);

            // Unpack 8 scales and mins, pre-bake products
            let sc_bytes = &block[4..16];
            let min_bytes = &block[16..20];

            let mut scales = [0.0f32; 8];
            let mut mins = [0.0f32; 8];
            for j in 0..4 {
                scales[j] = d * (sc_bytes[j] & 0x3F) as f32;
                scales[j + 4] = d * (sc_bytes[j + 4] & 0x3F) as f32;
                mins[j] = dmin * (min_bytes[j] & 0x0F) as f32;
                mins[j + 4] = dmin * ((min_bytes[j] >> 4) & 0x0F) as f32;
            }

            // Write pre-baked scales as f16
            for scale in &scales {
                out.extend_from_slice(&f32_to_f16(*scale).to_le_bytes());
            }
            // Write pre-baked mins as f16
            for min in &mins {
                out.extend_from_slice(&f32_to_f16(*min).to_le_bytes());
            }
            // Copy nibbles unchanged
            out.extend_from_slice(&block[20..148]);
        }
    }
    out
}

/// Quantize f32 data directly to Q4_KF format (pre-baked half scales).
pub fn quantize_q4_kf(data: &[f32]) -> Vec<u8> {
    assert!(data.len().is_multiple_of(256), "data length must be a multiple of 256");
    // First quantize to Q4_K, then convert
    let q4k = quantize_q4_k(data);
    let num_rows = 1; // treat as single row
    let hidden = data.len();
    q4k_to_q4kf(&q4k, num_rows, hidden)
}

/// Decode f16 bits to f32 (shared helper).
pub fn f16_to_f32(bits: u16) -> f32 {
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

    // ── quantize_q4_0 tests ──

    #[test]
    fn q4_output_size() {
        // 64 floats = 2 blocks of 32, each block → 18 bytes (2 f16 scale + 16 nibbles)
        let data = vec![1.0f32; 64];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 2 * 18);

        let data = vec![1.0f32; 256];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 8 * 18);
    }

    #[test]
    fn q4_zero_input() {
        let data = vec![0.0f32; 32];
        let q4 = quantize_q4_0(&data);
        assert_eq!(q4.len(), 18);
        // Scale should be zero (f16 zero = 0x0000)
        assert_eq!(q4[0], 0);
        assert_eq!(q4[1], 0);
        // All nibbles should encode 8 (zero quantized = 0 + bias 8)
        for &b in &q4[2..18] {
            assert_eq!(b, 0x88, "zero input should quantize to bias value 0x88");
        }
    }

    #[test]
    fn q4_round_trip_accuracy() {
        // Quantize then dequantize, check values are close
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.5).collect();
        let q4 = quantize_q4_0(&data);

        // Dequantize: read f16 scale, unpack nibbles, multiply
        let scale_bits = u16::from_le_bytes([q4[0], q4[1]]);
        let scale = f16_to_f32(scale_bits);

        let mut decoded = Vec::with_capacity(32);
        for j in 0..16 {
            let byte = q4[2 + j];
            let lo = (byte & 0x0F) as i32 - 8;
            let hi = (byte >> 4) as i32 - 8;
            decoded.push(lo as f32 * scale);
            decoded.push(hi as f32 * scale);
        }

        // Check approximate reconstruction (Q4 is lossy, but should be close)
        let max_err: f32 = data.iter().zip(decoded.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(max_err < 2.0, "Q4 round-trip max error {max_err} exceeds 2.0");
    }

    #[test]
    #[should_panic(expected = "multiple of 32")]
    fn q4_rejects_non_aligned() {
        let data = vec![1.0f32; 33];
        let _ = quantize_q4_0(&data);
    }

    #[test]
    fn q4_matvec_uses_quantized_data() {
        // End-to-end: quantize a matrix, run matvec, verify nonzero output
        let hidden = 256;
        let rows = 64;
        let matrix: Vec<f32> = (0..rows * hidden).map(|i| (i as f32 * 0.001).cos()).collect();
        let q4 = quantize_q4_0(&matrix);
        let x: Vec<f32> = (0..hidden).map(|i| (i as f32 * 0.01).sin()).collect();
        let (q8_x, q8_scales) = quantize_to_q8(&x);

        let mut scores = vec![0.0f32; rows];
        unsafe {
            q4_0_matvec_c(
                q4.as_ptr(), q8_x.as_ptr(), q8_scales.as_ptr(),
                scores.as_mut_ptr(), rows, hidden,
            );
        }
        assert!(scores.iter().any(|&v| v.abs() > 0.01), "Q4 matvec should produce nonzero");
    }

    /// Decode f16 bits to f32 (for test verification).
    fn f16_to_f32(bits: u16) -> f32 {
        let sign = ((bits >> 15) & 1) as u32;
        let exp = ((bits >> 10) & 0x1F) as i32;
        let mant = (bits & 0x3FF) as u32;
        if exp == 0 {
            if mant == 0 { return if sign == 1 { -0.0 } else { 0.0 }; }
            // Subnormal
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
}
