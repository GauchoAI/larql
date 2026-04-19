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
"#;
