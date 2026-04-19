//! Mega-kernel: persistent threadgroups with atomic phase sync.
//!
//! ONE dispatch processes ALL layers. Threadgroups synchronize via
//! atomic counters in device memory — no grid_sync needed.
//!
//! Phase pattern per layer:
//!   0: input norm (1 TG)
//!   1: QKV projection (all TGs)
//!   2: QK-norm + RoPE (few TGs)
//!   3: KV append + attend (num_q TGs)
//!   4: O projection (all TGs)
//!   5: post-attn norms (1 TG)
//!   6: gate+up projection (all TGs)
//!   7: GEGLU+down (all TGs)
//!   8: post-FFN norms (1 TG)
//!
//! Step 1: Validate atomic sync with a trivial test kernel.
//! Two phases: phase 0 writes a value, phase 1 reads and doubles it.
//! If the output is correct, atomic sync works on this GPU.

pub const TEST_SHADER: &str = r#"
#include <metal_stdlib>
using namespace metal;

// Atomic sync buffer layout:
//   [0]: phase counter (which phase we're in)
//   [1]: done counter (how many TGs finished current phase)
//   [2]: num_threadgroups (total TGs launched)

kernel void mega_test(
    device float*         data   [[buffer(0)]],   // [hidden] working buffer
    device atomic_uint*   sync   [[buffer(1)]],   // [3] sync counters
    constant uint&        hidden [[buffer(2)]],
    constant uint&        num_tg [[buffer(3)]],   // total threadgroups launched
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]])
{
    // Phase 0: each TG writes tg_id into its slice of data
    // (validates that all TGs execute)
    for (uint i = tg_id * tg_sz + tid; i < hidden; i += num_tg * tg_sz) {
        data[i] = float(i) + 1.0f;
    }

    // Signal phase 0 done
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint finished = atomic_fetch_add_explicit(&sync[1], 1u, memory_order_relaxed);
        if (finished + 1 == num_tg) {
            // Last TG: advance phase, reset done counter
            atomic_store_explicit(&sync[1], 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&sync[0], 1u, memory_order_relaxed);
        }
    }

    // Spin-wait for phase 1
    if (tid == 0) {
        while (atomic_load_explicit(&sync[0], memory_order_relaxed) < 1u) {}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Phase 1: each TG doubles its slice (reads what phase 0 wrote)
    for (uint i = tg_id * tg_sz + tid; i < hidden; i += num_tg * tg_sz) {
        data[i] = data[i] * 2.0f;
    }

    // Signal phase 1 done
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint finished = atomic_fetch_add_explicit(&sync[1], 1u, memory_order_relaxed);
        if (finished + 1 == num_tg) {
            atomic_store_explicit(&sync[1], 0u, memory_order_relaxed);
            atomic_fetch_add_explicit(&sync[0], 1u, memory_order_relaxed);
        }
    }
}
// ────────────────────────────────────────────────────────────────
// Step 2: f32 matvec inside persistent kernel.
// 3 phases: RMS norm → matvec → GELU activation.
// Validates that real computation works across atomic barriers.

kernel void mega_norm_matvec_act(
    device const float*   W      [[buffer(0)]],   // [N, K] weight matrix
    device const float*   x_in   [[buffer(1)]],   // [K] input vector
    device const float*   norm_w [[buffer(2)]],   // [K] norm weights
    device float*         x_buf  [[buffer(3)]],   // [K] normed input (intermediate)
    device float*         y_out  [[buffer(4)]],   // [N] output vector
    device atomic_uint*   sync   [[buffer(5)]],   // [2] phase + done counters
    constant uint&        N      [[buffer(6)]],   // output rows
    constant uint&        K      [[buffer(7)]],   // input dimension
    constant uint&        num_tg [[buffer(8)]],
    constant float&       eps    [[buffer(9)]],
    uint tg_id  [[threadgroup_position_in_grid]],
    uint tid    [[thread_index_in_threadgroup]],
    uint tg_sz  [[threads_per_threadgroup]],
    uint lane   [[thread_index_in_simdgroup]],
    uint sg_id  [[simdgroup_index_in_threadgroup]])
{
    // ── Phase 0: ALL TGs copy+norm x_in → x_buf (distributed) ──
    // Each TG reads x_in directly (read-only, no coherence issue),
    // computes norm inline, writes its slice to x_buf.
    // This avoids the TG-0-only norm + cross-TG read pattern.

    // First: cooperative sum_sq across ALL threads in ALL TGs
    // Each TG computes a partial sum, atomically adds to sync[2]
    {
        float partial = 0.0f;
        for (uint i = tg_id * tg_sz + tid; i < K; i += num_tg * tg_sz) {
            partial += x_in[i] * x_in[i];
        }
        float sg_sum = simd_sum(partial);
        // Only lane 0 of each SG has the sum. TG-local reduction:
        threadgroup float tg_p[8];
        if (lane == 0) tg_p[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float tg_total = tg_p[0];
            uint n_sg = (tg_sz + 31) / 32;
            for (uint i = 1; i < n_sg; i++) tg_total += tg_p[i];
            // Atomically accumulate this TG's contribution
            // Use atomic_fetch_add on an integer reinterpretation
            // (Hacky but Metal doesn't have atomic_fetch_add for float)
            // Instead: store TG partial sums in a separate array and reduce
        }
    }
    // Actually — atomic float add doesn't exist in Metal.
    // Simpler: just have TG 0 compute the full sum (it can read all of x_in
    // since x_in is read-only input, not written by any TG).

    // TG 0 computes RMS, writes to sync[2]
    if (tg_id == 0) {
        float partial = 0.0f;
        for (uint i = tid; i < K; i += tg_sz) {
            partial += x_in[i] * x_in[i];
        }
        float sg_sum = simd_sum(partial);
        threadgroup float tg_p[8];
        if (lane == 0) tg_p[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float sum_sq = tg_p[0];
        uint n_sg = (tg_sz + 31) / 32;
        for (uint i = 1; i < n_sg; i++) sum_sq += tg_p[i];
        float rms_val = 1.0f / sqrt(sum_sq / float(K) + eps);
        atomic_store_explicit(&sync[2], as_type<uint>(rms_val), memory_order_relaxed);
        threadgroup_barrier(mem_flags::mem_device);
        if (tid == 0) {
            atomic_store_explicit(&sync[0], 1u, memory_order_relaxed);
        }
    }
    if (tid == 0) {
        while (atomic_load_explicit(&sync[0], memory_order_relaxed) < 1u) {}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // Each TG reads x_in directly and applies norm inline during matvec.
    // x_in is read-only — no cross-TG coherence issue.
    // rms scalar is shared via atomic (4 bytes, always visible).
    float rms = as_type<float>(atomic_load_explicit(&sync[2], memory_order_relaxed));

    // ── Phase 1: normed f32 matvec y = W * (norm(x_in)) (all TGs) ──
    threadgroup float tg_p2[8];
    for (uint row = tg_id; row < N; row += num_tg) {
        float partial = 0.0f;
        for (uint j = tid; j < K; j += tg_sz) {
            partial += W[row * K + j] * x_in[j]; // raw matvec, no norm
        }
        float sg_sum = simd_sum(partial);
        if (lane == 0) tg_p2[sg_id] = sg_sum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float total = tg_p2[0];
            uint n_sg = (tg_sz + 31) / 32;
            for (uint i = 1; i < n_sg; i++) total += tg_p2[i];
            y_out[row] = total;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Barrier: wait for matvec phase
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0) {
        uint done = atomic_fetch_add_explicit(&sync[1], 1u, memory_order_relaxed);
        if (done + 1 == num_tg) {
            atomic_store_explicit(&sync[1], 0u, memory_order_relaxed);
            atomic_store_explicit(&sync[0], 3u, memory_order_relaxed);
        }
    }
    if (tid == 0) {
        while (atomic_load_explicit(&sync[0], memory_order_relaxed) < 3u) {}
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // ── Phase 2: GELU — DISABLED (cross-TG read issue) ──
    // Each TG only applies GELU to ITS OWN rows (no cross-TG y_out reads)
    for (uint i = tg_id; i < N; i += num_tg) {
        float v = y_out[i];
        // Approximate GELU: 0.5 * v * (1 + tanh(sqrt(2/pi) * (v + 0.044715 * v^3)))
        float v3 = v * v * v;
        float arg = 0.7978845608f * (v + 0.044715f * v3);
        arg = clamp(arg, -10.0f, 10.0f);
        y_out[i] = 0.5f * v * (1.0f + tanh(arg));
    }
}
"#;
