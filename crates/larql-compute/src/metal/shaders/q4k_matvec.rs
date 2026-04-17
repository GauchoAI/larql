//! Q4_K matrix-vector multiply — multi-row optimization.
//!
//! Each simdgroup processes NR0 output rows, reading the input vector once
//! and reusing it across all rows. Input stays in L1 cache since all lanes
//! within the simdgroup read the same X addresses.
//!
//! Empirically tuned NR0 sweep (Gemma 3 4B walk-only, N=10240, K=2560 on M4 Pro):
//!   NR0=2  → 13.75 tok/s (baseline)
//!   NR0=4  → 14.85 tok/s (+8 %)
//!   NR0=8  → 15.11 tok/s (+10 %)  ← chosen
//!   NR0=16 → 13.79 tok/s (regressed — register-pressure spill)
//!
//! 4 simdgroups × 8 rows = 32 rows per threadgroup, 128 threads total. 4× fewer
//! threadgroups vs NR0=2, 4× X-read reuse, 8× X bandwidth amortisation.

pub const SHADER: &str = r#"
constant uint Q4K_NR0 = 8;
constant uint Q4K_BLOCK_SIZE = 148;

kernel void q4k_matvec(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],
    device float*        out   [[buffer(2)]],
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    uint total_subs = superblocks * 8;

    // 4 simdgroups, each handles 2 rows
    uint first_row = (tg_id * 4 + sg_id) * Q4K_NR0;

    float acc[Q4K_NR0] = {0.f};

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        // Process all NR0 rows. X values are shared across rows and stay in L1
        // cache (all 32 lanes in this simdgroup read the same X addresses).
        // Tried explicit X-hoisting into local xv[32] — regressed because the
        // local array spilled to thread memory at NR0=8.
        for (uint r = 0; r < Q4K_NR0; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W4K + row_idx * bytes_per_row + sb * Q4K_BLOCK_SIZE;

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
            #define P(W, S, I) { \
                float a = X[xi+I], b = X[xi+I+1]; \
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

    for (uint r = 0; r < Q4K_NR0; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        float sum = simd_sum(acc[r]);
        if (lane == 0) out[row_idx] = sum;
    }
}
// Batched Q4_K matvec: read weights once, dot against M input vectors.
// out[row * M + m] = dot(W[row], X[m * K ..])
// Reduces NR0 to 2 to fit M accumulators in registers.
// Shared weight read = M× bandwidth saving.
constant uint Q4K_BATCH_NR0 = 2;

kernel void q4k_matvec_batch(
    device const uchar*  W4K   [[buffer(0)]],
    device const float*  X     [[buffer(1)]],   // [M, K]
    device float*        out   [[buffer(2)]],   // [N, M]
    constant uint&       N     [[buffer(3)]],
    constant uint&       K     [[buffer(4)]],
    constant uint&       M     [[buffer(5)]],   // batch size (1..8)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE;
    uint total_subs = superblocks * 8;
    uint first_row = (tg_id * 4 + sg_id) * Q4K_BATCH_NR0;

    // acc[row][batch] — NR0=2, M up to 8 = 16 float registers
    float acc[2][8];
    for (uint r = 0; r < Q4K_BATCH_NR0; r++)
        for (uint m = 0; m < 8; m++) acc[r][m] = 0.f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        for (uint r = 0; r < Q4K_BATCH_NR0; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W4K + row_idx * bytes_per_row + sb * Q4K_BLOCK_SIZE;
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

            // Extract weight nibbles once, dot against M inputs
            float wv[32];
            #define EX(W, S, I) wv[I] = float((W>>S)&0xFu); wv[I+1] = float((W>>(S+4))&0xFu);
            EX(w.x, 0, 0); EX(w.x, 8, 2); EX(w.x,16, 4); EX(w.x,24, 6);
            EX(w.y, 0, 8); EX(w.y, 8,10); EX(w.y,16,12); EX(w.y,24,14);
            EX(w.z, 0,16); EX(w.z, 8,18); EX(w.z,16,20); EX(w.z,24,22);
            EX(w.w, 0,24); EX(w.w, 8,26); EX(w.w,16,28); EX(w.w,24,30);
            #undef EX

            for (uint m = 0; m < M; m++) {
                float dot = 0.0f, xs = 0.0f;
                for (uint i = 0; i < 32; i++) {
                    float xv = X[m * K + xi + i];
                    dot += wv[i] * xv;
                    xs += xv;
                }
                acc[r][m] += sc * dot - mn * xs;
            }
        }
    }

    for (uint r = 0; r < Q4K_BATCH_NR0; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        for (uint m = 0; m < M; m++) {
            float sum = simd_sum(acc[r][m]);
            if (lane == 0) out[m * N + row_idx] = sum;
        }
    }
}
"#;

pub const ROWS_PER_TG: u64 = 32; // 4 simdgroups × NR0=8 rows
pub const THREADS_PER_TG: u64 = 128;  // 4 simdgroups × 32 lanes

pub const BATCH_ROWS_PER_TG: u64 = 8; // 4 simdgroups × NR0=2 rows (for batched)
pub const BATCH_THREADS_PER_TG: u64 = 128;
