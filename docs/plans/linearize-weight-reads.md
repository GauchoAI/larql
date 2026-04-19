# Linearize Weight Reads — Eliminate Redundant 59 MB/token

## Problem

Every token reads the same 59 MB of Q4_K weights from system memory.
At 273 GB/s bandwidth, this costs 0.22ms theoretical, 0.74ms actual.
34 layers × 0.74ms = 25ms/token = 42 tok/s.

The weights never change. Token 1 reads the exact same bytes as token 100.

## Ideas (ranked by expected impact)

### 1. Persistent Metal Buffers [S]

**What**: Allocate Metal buffers at startup, copy Q4_K weight data once.
Currently: mmap'd files, GPU reads through page cache path.
After: `MTLResourceOptions::StorageModeShared` buffers, GPU-resident.

**Why it helps**: GPU memory controller can prefetch and cache optimally.
mmap'd data goes through the OS page cache with POSIX semantics.
Dedicated Metal buffers get GPU-native cache management.

**How to test**: Allocate one Metal buffer per weight matrix at startup.
Copy interleaved_q4k_real.bin content into it. Point matvec dispatches
to the Metal buffer instead of the mmap pointer.

**Measure**: tok/s before and after. If L2 cache hit rate improves,
the per-token cost drops.

**Projected**: 42 → 50+ tok/s (if GPU L2 caching improves from 27% → 40%+)

### 2. GPU L2 Cache Pinning [M]

**What**: Metal has `MTLResourceOptions::StorageModePrivate` for GPU-only
memory. Weight data copied once via blit encoder, then lives exclusively
in GPU memory. No system memory path at all.

**Why it helps**: Private storage mode gives the GPU exclusive ownership.
The memory controller knows no CPU will access it — can cache aggressively.

**Risk**: Private storage requires GPU-side copy (blit encoder).
For 59 MB of weights × 34 layers = ~2 GB, the one-time copy is ~7ms.

**How to test**: At startup, blit each Q4_K weight slab to a private buffer.
Use private buffers in decode dispatches.

**Projected**: Better than shared buffers if the M4 Pro's GPU has dedicated
high-bandwidth memory paths for private storage.

### 3. Activation Caching — Sparse Feature Lookup [L]

**What**: Instead of `y = W × x` (dense matvec, reads all of W), identify
which output features activate strongly for common input patterns and
cache the results. A lookup table indexed by input hash.

**Why it helps**: The gate projection determines WHICH features activate.
If the top-K active features are predictable from the input pattern,
we can skip reading the inactive features' weight rows entirely.

This is what the vindex walk FFN already does — but at the feature level,
not the dense matvec level. The walk FFN achieved 4.8 tok/s (too slow)
because it walked ALL features. A cached version would only check the
top-K predicted features.

**How to test**: During generation, record (gate_top_indices, output) pairs.
Build a hash map: input_hash → cached_output. On cache hit, skip the matvec.

**Risk**: Cache miss rate determines speedup. If every token has unique
top-K activations, the cache never hits.

**Projected**: Depends on activation pattern reuse. If 50% of tokens
reuse a cached pattern, ~2x speedup.

### 4. Weight Compression — Read Less Data [M]

**What**: Replace Q4_K (4.6 bits/value) with Q2_K (2.6 bits/value).
Reads 40% less data per token. Or use GGUF's IQ2_XXS (2.0 bits).

**Why it helps**: Direct reduction in bandwidth. 59 MB → 35 MB per token.

**Risk**: Quality degradation. Q4_K is already the minimum for coherent
output on Gemma 3 4B. Q2_K may produce gibberish.

**How to test**: Quantize weights to Q2_K, measure perplexity vs Q4_K.

**Projected**: 42 → 70 tok/s if quality holds. But quality likely doesn't hold.

### 5. Precompute Function Approximation [XL]

**What**: The 34-layer transformer is a function: h_in → h_out.
For specific input distributions (e.g., English text), the function's
behavior may be approximable by a much smaller model — a distilled
lookup that skips most layers.

This is layer skip / early exit at the architectural level.

**Why it helps**: If layers 10-25 don't change the top-1 prediction
for 80% of tokens, we can skip them (read 0 bytes for those layers).

**Risk**: Gemma 3's post_ffn_norm amplification (300×) makes layer skip
fail catastrophically (tested earlier, Idea 5 null result). Would need
a different skip mechanism that accounts for the norm amplification.

**Projected**: If feasible, 3-5× speedup. But prior testing shows it
fails for Gemma 3 specifically.

## Execution Order

1. **Persistent Metal Buffers** — lowest risk, highest chance of immediate gain
2. **GPU L2 Cache Pinning** — test if Private storage mode helps beyond Shared
3. **Activation Caching** — research-grade, may not pan out
4. **Weight Compression** — quality tradeoff, test perplexity first
5. **Function Approximation** — XL effort, Gemma 3 resistant to layer skip

## Iteration Loop

For each idea:
1. Quick correctness check ("What port?" → "3000")
2. Measure tok/s (5 decode queries)
3. Compare to 42.6 tok/s baseline
4. Commit or revert
