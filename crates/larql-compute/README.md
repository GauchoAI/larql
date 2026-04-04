# larql-compute

Hardware-accelerated compute backends for LARQL. CPU (BLAS + NEON Q4), Metal GPU, and future CUDA.

## What it does

Provides a `ComputeBackend` trait that abstracts all hardware-specific matrix operations. Every LARQL crate (inference, vindex) uses this trait — the caller never knows whether the operation runs on CPU or GPU.

## Backends

| Backend | Feature flag | f32 matmul | Q4 fused ops | Multi-layer pipeline |
|---------|-------------|------------|--------------|---------------------|
| **CPU** | (always) | BLAS (Accelerate AMX) | C kernel (ARM vdotq_s32) | Sequential |
| **Metal** | `--features metal` | Tiled compute shaders | Simdgroup Q4×Q8 shaders | One command buffer |
| **CUDA** | (planned) | — | — | — |

## Performance (M3 Max, Gemma 3 4B)

```
Operation                     CPU         Metal       Winner
────────────────────────────  ──────────  ──────────  ──────
f32 matmul [6,2560²]          1.03ms      0.70ms     Metal
Q4 matvec [10240,2560]        0.96ms      0.57ms     Metal
Q4 vecmat [10240,2560]        1.31ms      2.05ms     CPU
Q4 pair_batch (6 pos)         11.4ms      1.39ms     Metal (8×)
21-layer Q4 FFN               60ms        8.5ms      Metal (7×)
Full layer (attn+FFN, seq=1)  —           1.7ms      Metal
```

## Quick start

```rust
use larql_compute::{ComputeBackend, default_backend, cpu_backend};

// Auto-detect best backend (Metal if available, else CPU)
let backend = default_backend();
println!("Using: {} ({})", backend.name(), backend.device_info());

// Force CPU only (no GPU, no calibration overhead)
let cpu = cpu_backend();

// f32 matmul — dispatches to BLAS or Metal depending on backend
let c = backend.matmul_transb(a.view(), b.view());

// Q4 fused operations — if backend supports them
if backend.has_q4() {
    let scores = backend.q4_matvec(&q4_data, &q8_x, &q8_scales, rows, hidden);
}
```

## Architecture

Every shader and every operation lives in its own file with its own tests.

```
src/
  lib.rs                    — crate root, exports, factory functions
  backend.rs                — ComputeBackend trait + helper functions

  cpu/
    mod.rs                  — CpuBackend struct + trait impl
    ops/
      mod.rs                — operation registry
      f32_matmul.rs         — BLAS sgemm/sgemm_transb       (3 tests)
      q4_matvec.rs          — C kernel Q4×Q8 matvec          (2 tests)
      q4_vecmat.rs          — C kernel Q4 vecmat             (2 tests)
      q4_common.rs          — Q8 quantize, C FFI decls       (2 tests)
      geglu.rs              — SiLU gate activation            (3 tests)
      attention.rs          — Causal attention (fused QKV)    (3 tests)

  metal/                    (feature-gated: --features metal)
    mod.rs                  — MetalBackend struct + trait impl
    shaders/
      mod.rs                — shader registry, all_shaders()
      common.rs             — f16 decode, metal_stdlib header
      sgemm.rs              — f32 tiled matmul C=A×B
      sgemm_transb.rs       — f32 tiled matmul C=A×B^T
      q4_matvec.rs          — optimised Q4×Q8 simdgroup (0.57ms)
      q4_vecmat.rs          — Q4 scatter-accumulate
      q4_f32_matvec.rs      — Q4×f32 for transposed down
      geglu.rs              — element-wise SiLU gate
      quantize_q8.rs        — f32→Q8 for layer chaining
      causal_attention.rs   — basic causal attention
    ops/
      mod.rs                — operation dispatch registry
      q4_common.rs          — Q4Pipelines struct + quantize_to_q8
      q4_matvec.rs          — single Q4 matvec dispatch
      q4_vecmat.rs          — single Q4 vecmat dispatch
      q4_f32_matvec.rs      — single Q4×f32 matvec dispatch
      q4_batched.rs         — pair_batch + multi_layer_ffn
      full_layer.rs         — attention + FFN in one cmd buffer
    buffers.rs              — GPU buffer cache (zero-copy mmap)
    calibrate.rs            — CPU vs GPU auto-calibration
    f32_ops.rs              — f32 dispatch with GPU/CPU routing

  csrc/
    q4_dot.c                — C kernel: ARM vdotq_s32 + scalar fallback
```

## Tests

```bash
# CPU tests only (15 tests)
cargo test -p larql-compute

# CPU + Metal tests (36 tests)
cargo test -p larql-compute --features metal
```

Tests cover:
- f32 matmul correctness (CPU vs ndarray reference)
- Q4 matvec/vecmat output validation
- Q8 quantization round-trip
- GEGLU activation values
- Causal attention (single token, causal mask, output shape)
- Metal shader compilation (all 9 kernels)
- Metal vs CPU correctness (per-shader)
- Metal batch dispatch consistency
- Metal multi-layer output verification
- Buffer cache behaviour

## Benchmarks

```bash
# Every operation, CPU + Metal side by side
cargo run --release -p larql-compute --features metal --example bench_shaders

# Three-way: BLAS f32 vs C Q4 vs Metal Q4
cargo run --release -p larql-compute --features metal --example bench_q4

# Multi-layer pipeline, mixed backend, generation simulation
cargo run --release -p larql-compute --features metal --example bench_pipeline

# Token generation (seq=1 decode, the production case)
cargo run --release -p larql-compute --features metal --example bench_generation

# All operations at representative sizes
cargo run --release -p larql-compute --features metal --example bench_full

# Backend auto-detection demo
cargo run --release -p larql-compute --features metal --example demo
```

## Design principles

1. **One file per operation** — every shader and every dispatch function lives in its own file with its own tests. No monolithic files.

2. **Trait-based dispatch** — callers use `ComputeBackend` exclusively. The implementation (CPU, Metal, CUDA) is invisible to the caller.

3. **Zero-copy for mmap** — weight matrices from mmap'd vindex files go to GPU via `newBufferWithBytesNoCopy`. No data copy on Apple Silicon unified memory.

4. **Cached vs transient buffers** — weight buffers cached by pointer address (stable across calls for mmap data). Input/output buffers allocated fresh each call to avoid stale data.

5. **Feature-gated backends** — Metal compiles only with `--features metal`. CPU always available. Adding CUDA means implementing `ComputeBackend` and adding `--features cuda`.

6. **Auto-calibration** — Metal backend benchmarks CPU vs GPU at startup. Small ops route to CPU (lower dispatch overhead), large ops route to GPU.

7. **Batch API** — multiple operations encoded in one GPU command buffer to amortise dispatch overhead. The multi-layer pipeline encodes all 21 FFN layers in a single submission (8.5ms vs 22ms per-layer).

## Adding a new backend

1. Create `src/newbackend/mod.rs`
2. Implement `ComputeBackend` trait
3. Add feature flag to `Cargo.toml`
4. Add to `default_backend()` factory with priority
5. Add tests in `tests/test_newbackend.rs`
6. Add to `bench_shaders.rs` for side-by-side comparison
