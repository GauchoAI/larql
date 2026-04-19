# S3 Mega-Kernel — Implementation Plan

## Goal

One Metal dispatch for ALL 34 layers. Hidden state stays GPU-resident.
Zero dispatch overhead. Target: 42 → 80-100+ tok/s.

## Proven

Step 1: Atomic sync across 64 persistent threadgroups ✓ (2026-04-19)

## Architecture

```
┌──────────── ONE kernel dispatch ────────────┐
│                                              │
│  64 persistent TGs × 256 threads = 16K GPU   │
│  threads, all resident for the entire run.   │
│                                              │
│  for layer in 0..34:                         │
│    phase 0: RMS norm (1 TG computes, rest    │
│             spin-wait on atomic)             │
│    phase 1: QKV matvec (all 64 TGs, each     │
│             processes ~64 rows)              │
│    phase 2: QK-norm + RoPE (12 TGs)          │
│    phase 3: KV append + attend (8 TGs)       │
│    phase 4: O matvec (all 64 TGs)            │
│    phase 5: post-attn norms (1 TG)           │
│    phase 6: gate+up matvec (all 64 TGs)      │
│    phase 7: GEGLU+down matvec (all 64 TGs)   │
│    phase 8: post-FFN norms (1 TG)            │
│    [atomic barrier between each phase]       │
│                                              │
│  Hidden state: device memory buffer          │
│  Weight data: mmap'd Q4_K, read per layer    │
│  KV cache: device memory, append per layer   │
│  Sync: atomic counters in device memory      │
│                                              │
└──────────────────────────────────────────────┘
```

## Steps

### Step 2: Matvec inside persistent kernel [S]

Prove that a real matvec works within the persistent TG pattern.
Start with f32 weights (simple), then Q4_K (production).

**Test kernel: 3 phases**
- Phase 0: RMS norm on input vector (1 TG)
- Phase 1: f32 matvec y = W * x (all TGs, each handles a row slice)
- Phase 2: element-wise activation (all TGs)

**Validation**: compare output to separate norm → matmul → activation dispatches.
**Pass criteria**: max absolute error < 1e-4.

### Step 3: Q4_K matvec in persistent kernel [M]

Inline the Q4_K block dequantization into the mega-kernel.
The current `q4k_matvec` kernel reads Q4_K blocks (148 bytes / 256 values),
dequantizes on the fly, and accumulates dot products. Same logic, but as
a callable function within the persistent kernel instead of a standalone kernel.

**Test**: Q4_K matvec of [2560, 10240] (gate projection size).
Compare output to standalone `q4k_matvec` dispatch.

### Step 4: Single-layer decode in one dispatch [L]

All 9 phases for one transformer layer in a single kernel.
This is the real validation — correct attention + FFN output.

**Test**: run single-layer decode via mega-kernel, compare hidden state
output to the current `decode_token` path. Cosine > 0.9999.

### Step 5: 34-layer decode in one dispatch [XL]

Loop Step 4 across all 34 layers. The layer index increments a counter
that offsets into the weight buffers.

**Test**: generate 5 tokens via mega-kernel path, compare to current path.
Bit-exact first token, coherent text for subsequent tokens.

## Sync Protocol

```metal
// Shared atomic counters in device memory
device atomic_uint phase;      // current phase number
device atomic_uint done_count; // TGs finished current phase

// After each TG finishes its work:
threadgroup_barrier(mem_flags::mem_device);
if (tid == 0) {
    uint finished = atomic_fetch_add_explicit(&done_count, 1, relaxed);
    if (finished + 1 == num_active_tgs) {
        // Last TG: reset done counter, advance phase
        atomic_store_explicit(&done_count, 0, relaxed);
        atomic_store_explicit(&phase, next_phase, relaxed);
    }
}
// All TGs spin-wait:
if (tid == 0) {
    while (atomic_load_explicit(&phase, relaxed) < next_phase) {}
}
threadgroup_barrier(mem_flags::mem_threadgroup);
```

## Key Constraint

Metal only supports `memory_order_relaxed` for atomics.
Device memory coherence is guaranteed by `threadgroup_barrier(mem_flags::mem_device)`
before the atomic write, and the spin-wait loop provides the acquire semantics.

## Weight Buffer Layout

All 34 layers' Q4_K weights are contiguous in the mmap'd file.
The kernel receives a base pointer + per-layer byte offset.
Layer `l`'s gate weights start at `base + l * bytes_per_layer`.

```
interleaved_q4k_real.bin layout:
  [layer 0: gate | up | down]  (3 × q4k_per_matrix bytes)
  [layer 1: gate | up | down]
  ...
  [layer 33: gate | up | down]
```

## Risk Mitigations

1. **Deadlock**: If GPU can't keep all TGs resident, spin-wait deadlocks.
   Mitigation: use conservative TG count (64 × 32 threads = 2048 total).
   M4 Pro 16 cores can handle this easily.

2. **Spinning waste**: Phases with few active TGs (norms: 1 TG) waste
   63 TGs spinning. Acceptable — the norm phase is <0.01ms.

3. **Register pressure**: Q4_K dequant uses many registers. If the
   compiler spills to device memory, performance degrades.
   Mitigation: profile register usage, reduce NR0 if needed.
