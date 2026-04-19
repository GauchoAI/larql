//! Q8_0 GGUF matvec — reads GGUF-format Q8_0 blocks (34 bytes per 32 values).
//!
//! Block layout: [f16 scale (2B)] [int8 quants × 32 (32B)] = 34 bytes total.
//! Input is f32. Output is f32.

/// Rows per threadgroup (one simdgroup = one row).
pub const ROWS_PER_TG: u64 = 4;
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 threads

pub const SHADER: &str = r#"
// Q8_0 GGUF matvec: y[n] = sum_k( dequant(W[n,k]) * x[k] )
// W is stored as GGUF Q8_0 blocks: 34 bytes per block of 32 values.
// Block layout: half scale (2 bytes) + int8 quants[32] (32 bytes).
kernel void q8_0_gguf_matvec(
    device const uchar*  W       [[buffer(0)]],
    device const float*  X       [[buffer(1)]],
    device float*        Y       [[buffer(2)]],
    constant uint&       N       [[buffer(3)]],  // output rows
    constant uint&       K       [[buffer(4)]],  // input dim (must be multiple of 32)
    uint tg_id   [[threadgroup_position_in_grid]],
    uint sg_id   [[simdgroup_index_in_threadgroup]],
    uint lane    [[thread_index_in_simdgroup]]
) {
    const uint ROWS_PER_TG = 4;
    uint row = tg_id * ROWS_PER_TG + sg_id;
    if (row >= N) return;

    uint num_blocks = K / 32;
    uint bytes_per_row = num_blocks * 34;  // 34 bytes per Q8_0 block
    device const uchar* row_ptr = W + row * bytes_per_row;

    float acc = 0.0f;
    for (uint b = lane; b < num_blocks; b += 32) {
        device const uchar* block = row_ptr + b * 34;
        // First 2 bytes: f16 scale
        float scale = float(*(device const half*)block);
        // Next 32 bytes: int8 quantized values
        device const char* qs = (device const char*)(block + 2);

        float dot = 0.0f;
        for (uint j = 0; j < 32; j++) {
            dot += float(qs[j]) * X[b * 32 + j];
        }
        acc += dot * scale;
    }

    // Reduce across simdgroup
    acc = simd_sum(acc);
    if (lane == 0) {
        Y[row] = acc;
    }
}
"#;
