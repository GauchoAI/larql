# Speculative Decoding with Vindex Graph — Implementation Plan

## Context

larql runs Gemma 3 4B at **41 tok/s** (24ms/tok) on M4 Pro Metal GPU. Fully GPU-resident decode. KNN overlay works via zero-cost GPU probe.

**Goal**: 100-200+ tok/s via speculative decoding while keeping correctness and memory use.

**Critical constraint**: Sequential verification gives ZERO speedup. **Parallel verification is required** — process K draft tokens through all 34 layers in ONE command buffer.

---

## Architecture

```
┌──────────── Speculative Decode Cycle ────────────┐
│                                                    │
│  1. DRAFT K tokens (cheap, <2ms total)             │
│     ├── N-gram cache: hash lookup                  │
│     ├── Feature-token: gate_knn + down_meta        │
│     └── Draft head MLP: h → next_token             │
│                                                    │
│  2. VERIFY K tokens in ONE GPU pass (24ms)         │
│     decode_token_batch(embed(draft[0..K]), K)      │
│     → K hidden states → K lm_head predictions     │
│                                                    │
│  3. ACCEPT longest matching prefix (M of K)        │
│     Rollback KV cache by (K - M - 1) positions    │
│     Bonus: model's prediction at position M+1     │
│                                                    │
│  Result: M+1 tokens per cycle (~24ms)              │
└────────────────────────────────────────────────────┘
```

**Speed projections** (parallel verify, 24ms per cycle):

| Acceptance | K | Tokens/cycle | tok/s | Speedup |
|---:|---:|---:|---:|---:|
| 50% | 4 | 1.9 | 78 | 1.9x |
| 70% | 6 | 3.1 | 124 | 3.0x |
| 80% | 6 | 4.2 | 168 | 4.1x |
| 90% | 8 | 6.1 | 244 | 6.0x |

---

## Phase 0: Prove the Loop

KV rollback + n-gram cache + sequential verify. Zero speedup — validates infrastructure for Phases 1-3.

### 0A: KV Cache Rollback [S]
- Add `rollback(n)` to `LayerKVCache` / `KVCache` — decrement `current_len`
- Safe: `kv_attention` reads `[0..current_len]`; stale data ignored
- Add `rollback_kv_cache(n)` to `ComputeBackend` trait
- **Files**: `kv_cache.rs`, `backend.rs`, `trait_impl.rs`
- **Test**: decode K tokens → rollback K → decode same K → identical output

### 0B: N-gram Cache [S]
- `HashMap<(u32, u32), Vec<(u32, u32)>>` — bigram → continuations
- Populated from prompt + accepted tokens
- Multi-step: chain lookups `(t-1, draft_0)` → `draft_1`
- **Files**: new `layer_graph/ngram_cache.rs`

### 0C: Sequential Verify Loop [M]
- New `generate_speculative()` in `generate.rs`
- Draft → verify one-by-one → accept prefix → rollback rejected
- **Test**: output matches normal `generate()` exactly

---

## Phase 1: Parallel Verification — The Core Unlock

Process K draft tokens through all 34 layers in ONE Metal command buffer.

### 1A: Batched KV Append Shader [S]
- New `kv_cache_append_batch` kernel: write K entries at `[pos..pos+K]`
- Grid: `(kv_dim, K, 1)`
- **Files**: `kv_attention.rs`, `kv_cache.rs`, `mod.rs`

### 1B: Batched KV Attention Shader [L]
- New `kv_attention_batched`: K query positions, causal within batch
- Each position `qi` attends to `[0..cache_len + qi + 1]`
- Approach: append K to cache first, then per-position T via `tg_id.y`
- Includes softcap (Gemma 3)
- Grid: `(num_q_heads, K, 1)`
- **Files**: `kv_attention.rs`, `kv_cache.rs`
- **Test**: output matches K sequential `kv_attention` calls

### 1C: Batched Decode Function [XL — largest piece]
- New `decode_token_batch()` in `decode.rs`
- Mirrors `decode_token_inner` with K-sized buffers
- Per layer: K × (norm + QKV + QK-norm + RoPE + KV-batch-append + KV-batch-attend + O + norms + FFN + residual)
- All in one Metal command buffer, one encoder
- Extends the 1000+ LOC `decode_token_inner` to handle batch dimension
- **Files**: `decode.rs`, `backend.rs`, `trait_impl.rs`
- **Test**: `decode_token_batch([a,b])` matches `[decode_token(a), decode_token(b)]` with KV rollback

### 1D: Parallel Verify Generation Loop [M]
- Update `generate_speculative()` to use `decode_token_batch`
- K draft embeds → one batch pass → K predictions → accept prefix → rollback
- **Test**: same output as Phase 0C; wall-clock 2x+ speedup
- **Expected**: at alpha=0.5 with n-gram ~78 tok/s (1.9x)

---

## Phase 2: Draft Quality

### 2A: Feature-Token Draft [M]
- GPU probe at L26 + `gate_knn(probe_h, 1)` → `feature_meta.top_token_id`
- `DraftStrategy` trait: `fn draft(&mut self, context: &[u32], probe_h: Option<&[f32]>) -> Vec<u32>`
- Implementations: `NgramDraft`, `FeatureTokenDraft`, `CombinedDraft`
- **Files**: new `layer_graph/draft.rs`, `gate.rs`, `down_meta.rs`

### 2B: Training Data Capture [S]
- During generation, capture `(h_final, next_token_id)` pairs to binary file
- Multi-step targets: `(h_t, token_{t+1}, ..., token_{t+K})` for K-step heads
- Target: ~100K pairs from diverse prompts
- **Files**: new `layer_graph/capture_draft.rs`

---

## Phase 3: Draft Head — Medusa-style (requires GPU for training)

### 3A: Architecture [M]
- Per lookahead position: `h[2560] → Linear(2560, 1024) → GELU → Linear(1024, vocab)`
- K heads for K-step lookahead (independent MLPs)
- Q4_K quantized for GPU inference: ~135MB/head, ~0.3ms forward

### 3B: Training [M]
- PyTorch training script on captured data
- Cross-entropy loss, AdamW, diverse training data
- Export weights as Q4_K binary
- **Files**: new `scripts/train_draft_head.py`

### 3C: GPU Integration [M]
- Load draft head weights, run via existing Q4_K matvec + GELU pipelines
- Priority cascade: n-gram → draft head → feature-token
- **Files**: `draft.rs`, `generate.rs`, `backend.rs`, `trait_impl.rs`
- **Expected**: alpha=0.7-0.8, K=6 → **120-170 tok/s (3-4x)**

---

## Dependencies

```
Phase 0A (rollback) ──────┐
Phase 0B (n-gram)  ───────┤
                          ▼
Phase 0C (seq verify) ────┘
                          │
Phase 1A (batch append) ──┤
Phase 1B (batch attend) ──┤
                          ▼
Phase 1C (batch decode) ──┘   ← THE BIG ONE
                          │
Phase 1D (parallel loop) ─┘
         │
         ├── Phase 2A (feature-token draft)
         ├── Phase 2B (capture training data)
         │
         └── Phase 3A-C (draft head train + integrate)
```

---

## Verification

| Phase | Test | Pass |
|---|---|---|
| 0A | Rollback + re-decode | cosine > 0.9999 |
| 0B | N-gram on repetitive text | hit rate > 80% |
| 0C | Sequential verify | exact token match vs normal gen |
| 1B | Batched attention | matches K sequential calls |
| 1C | Batch decode K=2 | hidden states match sequential |
| 1D | Parallel verify e2e | same output, 2x+ wall-clock |
| 2A | Feature-token acceptance | >40% non-repetitive |
| 3 | Draft head accuracy | top-1 >60% validation |

---

## Critical Files

| File | Role |
|---|---|
| `larql-compute/src/metal/decode.rs` | Batched decode (extends 1000+ LOC) |
| `larql-compute/src/metal/shaders/kv_attention.rs` | Batch append + attend shaders |
| `larql-compute/src/metal/ops/kv_cache.rs` | Rollback + batch dispatch |
| `larql-compute/src/backend.rs` | Trait: `decode_token_batch`, `rollback_kv_cache` |
| `larql-inference/src/layer_graph/generate.rs` | Speculative generation loop |
| `larql-inference/src/layer_graph/ngram_cache.rs` | N-gram draft (new) |
| `larql-inference/src/layer_graph/draft.rs` | Draft strategy trait (new) |
| `larql-vindex/src/index/gate.rs` | gate_knn for feature-token draft |
| `larql-vindex/src/format/down_meta.rs` | feature → token lookup |
