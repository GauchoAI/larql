//! Fused GEGLU activation + Q4_K down projection.
//!
//! Eliminates the GEGLU dispatch entirely by computing SiLU(gate)×up on-the-fly
//! during the down projection. Each lane computes the activation for its assigned
//! sub-block elements and immediately multiplies by the dequantized weight.
//!
//! down_out[row] = sum_i( W_down[row,i] * activation(gate[i]) * up[i] )
//!
//! Saves one dispatch + one full read/write of the inter-sized activation buffer.
//!
//! **Layout (matches q4k_matvec):** raw `device const uchar*` with explicit byte
//! arithmetic. Earlier struct-based version (`device const block_q4_K*`) produced
//! garbled output — Metal MSL likely padded `sizeof(block_q4_K)` past 148 B,
//! making row indexing wrong. This rewrite uses the identical addressing the
//! working `q4k_matvec` shader uses, so layout cannot diverge.
//!
//! 4 simdgroups × NR0=8 rows = 32 rows per threadgroup, 128 threads total
//! (matches the post-P7 q4k_matvec tuning).

pub const SHADER: &str = r#"
constant uint Q4K_GD_NR0 = 8;
constant uint Q4K_GD_BLOCK_SIZE = 148;

// SiLU + down (Llama, Mistral, Qwen)
kernel void q4k_geglu_silu_down(
    device const uchar*  W_down [[buffer(0)]],  // [N, K] Q4_K, 148 B / 256 vals
    device const float*  gate   [[buffer(1)]],  // gate output [K]
    device const float*  up     [[buffer(2)]],  // up output [K]
    device float*        out    [[buffer(3)]],  // output [N] (hidden)
    constant uint&       N      [[buffer(4)]],  // hidden (output rows)
    constant uint&       K      [[buffer(5)]],  // inter (input dim)
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_GD_BLOCK_SIZE;
    uint total_subs = superblocks * 8;
    uint first_row = (tg_id * 4 + sg_id) * Q4K_GD_NR0;

    float acc[Q4K_GD_NR0];
    for (uint r = 0; r < Q4K_GD_NR0; r++) acc[r] = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        // Hoist activation outside row loop — see gelu_tanh kernel for rationale.
        float act[32];
        float xs_total = 0.0f;
        for (uint i = 0; i < 32; i++) {
            float g = gate[xi + i];
            // SiLU: x / (1 + exp(-x)). exp(-x) for x < ~88 is fine on f32.
            // For x more negative than -88, exp(-x) overflows. Clamp to be safe.
            float ex = exp(-clamp(g, -80.0f, 80.0f));
            act[i] = (g / (1.0f + ex)) * up[xi + i];
            xs_total += act[i];
        }

        for (uint r = 0; r < Q4K_GD_NR0; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W_down + row_idx * bytes_per_row + sb * Q4K_GD_BLOCK_SIZE;

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

            float dot = 0.0f;
            #define P(W, S, I) { \
                dot += float((W>>S)&0xFu)*act[I] + float((W>>(S+4))&0xFu)*act[I+1]; }
            P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
            P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
            P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
            P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
            #undef P
            acc[r] += sc * dot - mn * xs_total;
        }
    }

    for (uint r = 0; r < Q4K_GD_NR0; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        float sum = simd_sum(acc[r]);
        if (lane == 0) out[row_idx] = sum;
    }
}

// GELU-tanh + down (Gemma, GPT-2, Phi)
kernel void q4k_geglu_gelu_tanh_down(
    device const uchar*  W_down [[buffer(0)]],
    device const float*  gate   [[buffer(1)]],
    device const float*  up     [[buffer(2)]],
    device float*        out    [[buffer(3)]],
    constant uint&       N      [[buffer(4)]],
    constant uint&       K      [[buffer(5)]],
    uint tg_id     [[threadgroup_position_in_grid]],
    uint tid_in_tg [[thread_index_in_threadgroup]],
    uint lane      [[thread_index_in_simdgroup]],
    uint sg_id     [[simdgroup_index_in_threadgroup]])
{
    uint superblocks = K / 256;
    uint bytes_per_row = superblocks * Q4K_GD_BLOCK_SIZE;
    uint total_subs = superblocks * 8;
    uint first_row = (tg_id * 4 + sg_id) * Q4K_GD_NR0;
    float c = 0.7978845608f; // sqrt(2/pi)

    float acc[Q4K_GD_NR0];
    for (uint r = 0; r < Q4K_GD_NR0; r++) acc[r] = 0.0f;

    for (uint sub = lane; sub < total_subs; sub += 32) {
        uint sb = sub / 8;
        uint j = sub % 8;
        uint xi = sb * 256 + j * 32;

        // ── Hoist activation outside row loop ──
        // Compute act[i] = gelu_tanh(gate[xi+i]) * up[xi+i] ONCE for the 32
        // elements in this sub, reuse across all NR0 output rows. Without
        // this each lane recomputed gelu_tanh NR0× per element (NR0=8 → 8×
        // redundant work per element). 32 floats / 128 B per lane is well
        // within Apple's per-lane register budget alongside acc[8].
        // Clamp to [-10, 10] avoids tanh overflow → NaN for large |gate|.
        float act[32];
        float xs_total = 0.0f;
        for (uint i = 0; i < 32; i++) {
            float g = gate[xi + i];
            float arg = clamp(c * (g + 0.044715f*g*g*g), -10.0f, 10.0f);
            act[i] = (0.5f * g * (1.0f + tanh(arg))) * up[xi + i];
            xs_total += act[i];
        }

        for (uint r = 0; r < Q4K_GD_NR0; r++) {
            uint row_idx = first_row + r;
            if (row_idx >= N) break;

            device const uchar* block = W_down + row_idx * bytes_per_row + sb * Q4K_GD_BLOCK_SIZE;

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

            float dot = 0.0f;
            #define P(W, S, I) { \
                dot += float((W>>S)&0xFu)*act[I] + float((W>>(S+4))&0xFu)*act[I+1]; }
            P(w.x, 0, 0); P(w.x, 8, 2); P(w.x,16, 4); P(w.x,24, 6);
            P(w.y, 0, 8); P(w.y, 8,10); P(w.y,16,12); P(w.y,24,14);
            P(w.z, 0,16); P(w.z, 8,18); P(w.z,16,20); P(w.z,24,22);
            P(w.w, 0,24); P(w.w, 8,26); P(w.w,16,28); P(w.w,24,30);
            #undef P
            acc[r] += sc * dot - mn * xs_total;
        }
    }

    for (uint r = 0; r < Q4K_GD_NR0; r++) {
        uint row_idx = first_row + r;
        if (row_idx >= N) break;
        float sum = simd_sum(acc[r]);
        if (lane == 0) out[row_idx] = sum;
    }
}
"#;

pub const ROWS_PER_TG: u64 = 32; // 4 simdgroups × NR0=8
pub const THREADS_PER_TG: u64 = 128; // 4 simdgroups × 32 lanes
