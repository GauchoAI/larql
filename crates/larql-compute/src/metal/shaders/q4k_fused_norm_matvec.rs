//! Fused RMS norm + Q4_K matvec — eliminates norm dispatch entirely.
//!
//! Each threadgroup independently computes the full RMS norm on the
//! input vector (redundant but cheap — 2560 elements vs 15 MB weights),
//! then applies the norm inline during the Q4_K dot product.
//!
//! Saves 1 dispatch per matvec phase. With 4 matvec phases per layer
//! and 34 layers, that's 136 saved dispatches = ~0.7ms.

pub const SHADER: &str = r#"
constant uint Q4K_NR0_F = 8;
constant uint Q4K_BLOCK_SIZE_F = 148;

// Fused: each TG computes RMS norm on x, then Q4_K matvec with normed x.
// Input x is read-only — all TGs read it independently (no cross-TG issue).
kernel void q4k_norm_matvec(
    device const uchar*  W4K      [[buffer(0)]],
    device const float*  X        [[buffer(1)]],   // [K] input (read-only)
    device const float*  norm_w   [[buffer(2)]],   // [K] norm weights
    device float*        out      [[buffer(3)]],   // [N] output
    constant uint&       N        [[buffer(4)]],
    constant uint&       K        [[buffer(5)]],
    constant float&      eps      [[buffer(6)]],
    constant float&      norm_off [[buffer(7)]],   // norm weight offset (1.0 for Gemma)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // ── Step 1: Compute RMS norm (redundant per TG, but cheap) ──
    // All 128 threads cooperate on sum_sq over the full K-dim input.
    float partial_sq = 0.0f;
    for (uint i = tid_in_tg; i < K; i += 128) {
        partial_sq += X[i] * X[i];
    }
    float sg_sum = simd_sum(partial_sq);
    threadgroup float tg_sq[4]; // 4 simdgroups
    if (lane == 0) tg_sq[sg_id] = sg_sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum_sq = tg_sq[0] + tg_sq[1] + tg_sq[2] + tg_sq[3];
    float rms = 1.0f / sqrt(sum_sq / float(K) + eps);

    // ── Step 2: Q4_K matvec with inline norm ──
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_F;
    uint total_subs = superblocks * 8;

    uint first_row = (tg_id * 4 + sg_id) * Q4K_NR0_F;

    float acc[Q4K_NR0_F] = {0.f};

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        for (uint r = 0; r < Q4K_NR0_F; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W4K + row_idx * bytes_per_row + sb * Q4K_BLOCK_SIZE_F;
            device const half* dh = (device const half*)block;
            float d    = float(dh[0]);
            float dmin = float(dh[1]);

            device const uchar* sc_bytes = block + 4;
            float sc = d * float(sc_bytes[j] & 0x3F);
            float mn;
            device const uchar* min_bytes = block + 16;
            if (j < 4) mn = dmin * float(min_bytes[j] & 0x0F);
            else mn = dmin * float((min_bytes[j - 4] >> 4) & 0x0F);

            device const uint4* qp = (device const uint4*)(block + 20 + j * 16);
            uint4 w = qp[0];

            float dot = 0.0f, xs = 0.0f;
            // Inline norm: X[i] * (norm_w[i] + offset) * rms
            #define P(W, S, I) { \
                float a = X[xi+I] * (norm_w[xi+I] + norm_off) * rms; \
                float b = X[xi+I+1] * (norm_w[xi+I+1] + norm_off) * rms; \
                dot += float((W>>S)&0xFu)*a + float((W>>(S+4))&0xFu)*b; \
                xs += a + b; }
            P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
            P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
            P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
            P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
            #undef P
            acc[r] += sc * dot - mn * xs;
        }
    }

    for (uint r = 0; r < Q4K_NR0_F; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        float sum = simd_sum(acc[r]);
        if (lane == 0) out[row_idx] = sum;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 32; // 4 simdgroups × NR0=8
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 threads
