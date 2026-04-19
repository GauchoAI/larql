//! Indexed Q4_K matvec — reads only specified rows from the weight matrix.
//!
//! Each threadgroup processes one row from the index buffer, reading
//! only that row's Q4_K blocks from the weight matrix. Rows not in
//! the index are never read — saving bandwidth proportional to sparsity.
//!
//! Used for sparse FFN: after gate identifies active features,
//! only read those features' up and down weight rows.

pub const SHADER: &str = r#"
constant uint Q4K_BLOCK_SIZE_IX = 148;

// Indexed Q4_K matvec: out[i] = W[indices[i], :] · X for i in 0..num_active
kernel void q4k_indexed_matvec(
    device const uchar*  W4K     [[buffer(0)]],   // [N_total, K] Q4_K weight matrix
    device const float*  X       [[buffer(1)]],   // [K] input vector
    device float*        out     [[buffer(2)]],   // [num_active] output (compact)
    device const uint*   indices [[buffer(3)]],   // [num_active] row indices into W4K
    constant uint&       K       [[buffer(4)]],   // input dimension
    constant uint&       N_total [[buffer(5)]],   // total rows in W4K
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    // Each simdgroup processes one indexed row
    uint active_idx = tg_id * 4 + sg_id;  // 4 simdgroups per TG
    // bounds checked via dispatch grid size

    uint row_idx = indices[active_idx];
    if (row_idx >= N_total) return;

    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_BLOCK_SIZE_IX;
    uint total_subs = superblocks * 8;

    device const uchar* row_data = W4K + row_idx * bytes_per_row;
    float acc = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        device const uchar* block = row_data + sb * Q4K_BLOCK_SIZE_IX;
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
        acc += sc * dot - mn * xs;
    }

    acc = simd_sum(acc);
    if (lane == 0) out[active_idx] = acc;
}
"#;

pub const ROWS_PER_TG: u64 = 4; // 4 simdgroups per TG, 1 row per SG
pub const THREADS_PER_TG: u64 = 128; // 4 × 32

/// Threshold-based index selection: finds indices where |gate[i]| > threshold.
/// Uses atomic counter for compact output.
/// Two kernels for scatter:
/// 1. zero_buffer: sets all N elements to 0
/// 2. scatter_sparse: writes sparse[i] to full[indices[i]]
pub const SCATTER_SHADER: &str = r#"
kernel void zero_buffer(
    device float* buf [[buffer(0)]],
    constant uint& N  [[buffer(1)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid < N) buf[tid] = 0.0f;
}

kernel void scatter_sparse(
    device const float*  sparse  [[buffer(0)]],  // [K_active] compact
    device const uint*   indices [[buffer(1)]],  // [K_active] indices
    device float*        full    [[buffer(2)]],  // [N] output
    constant uint&       K_active [[buffer(3)]], // number of active
    uint tid [[thread_position_in_grid]])
{
    if (tid >= K_active) return;
    full[indices[tid]] = sparse[tid];
}
"#;

pub const SELECT_SHADER: &str = r#"
kernel void select_active_indices(
    device const float*  gate     [[buffer(0)]],   // [N] gate outputs
    device uint*         indices  [[buffer(1)]],   // [N] output: active indices (compact)
    device atomic_uint*  count    [[buffer(2)]],   // [1] number of active indices
    constant uint&       N        [[buffer(3)]],
    constant float&      threshold[[buffer(4)]],
    uint tid [[thread_position_in_grid]])
{
    if (tid >= N) return;
    float v = gate[tid];
    // GELU: if |gate| is small, GELU(gate) ≈ 0, feature is inactive
    if (abs(v) > threshold) {
        uint pos = atomic_fetch_add_explicit(count, 1u, memory_order_relaxed);
        indices[pos] = tid;
    }
}
"#;
