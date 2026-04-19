# Findings — Conversation as Knowledge

## Community Research — Long Conversation Retrieval Without Labeling

### EM-LLM: Surprise-Based Episodic Segmentation (ICLR 2025)
**Paper**: [arxiv.org/abs/2407.09450](https://arxiv.org/abs/2407.09450)
**Code**: [github.com/em-llm/EM-LLM-model](https://github.com/em-llm/EM-LLM-model)

Segments conversation into "episodic events" using Bayesian surprise — no labels.
When tokens are information-dense (surprising), a boundary is placed. Retrieval
is two-stage: similarity-based + temporally contiguous (retrieve nearby events too).

- Scales to 10M tokens (~7,500 pages)
- Up to 40% improvement in retrieval/QA vs baselines
- Surpasses full-context models in most tasks
- Event boundaries correlate with human-perceived events
- **Key for us**: auto-segmentation without labeling, pure content-driven

### Pichay: Cooperative Memory Paging (2026)
**Paper**: [arxiv.org/abs/2604.12376](https://arxiv.org/abs/2604.12376)
**Blog**: [fsgeek.ca/2026/03/25/pichay-treating-llm-context-as-virtual-memory/](https://fsgeek.ca/2026/03/25/pichay-treating-llm-context-as-virtual-memory/)

Treats context window like OS virtual memory. Evicts old segments, replaces with
tiny keyword bookmarks (`[p3: Metal GPU, tok/s, 41]` — 8-24 tokens each). Model
gets a `recall()` tool to page segments back in on demand.

- Minimal bookmarks work BETTER than verbose ones (verbose gives false confidence)
- LLMs are cooperative agents — they follow paging instructions
- Operates on conversation content, not tool outputs
- **Key for us**: immediately actionable — auto-extract keywords per turn,
  model decides what to recall. Our RAG but with active model-driven retrieval.

### InfLLM: Training-Free Streaming Memory (2024)
**Paper**: [openreview.net/forum?id=bTHFrqhASY](https://openreview.net/forum?id=bTHFrqhASY)

Stores distant context in memory units, looks up token-relevant units for
attention computation. No training needed. Scales to 1M+ tokens.

- Training-free — works on any pretrained LLM
- Efficient memory lookup during attention
- Comparable to continual-training baselines
- **Key for us**: closest to chuk-lazurus window-replay approach

### MemoryBank: Long-Term Memory with Forgetting (2023)
**Paper**: [arxiv.org/abs/2305.10250](https://arxiv.org/abs/2305.10250)

Ebbinghaus forgetting curve — memories decay, important ones reinforced.
Continuous memory updates, personality adaptation from past interactions.

- **Key for us**: forgetting mechanism prevents store bloat over long sessions

### GAM: Dual-Agent Memory Architecture (2026)
**Coverage**: [VentureBeat](https://venturebeat.com/ai/gam-takes-aim-at-context-rot-a-dual-agent-memory-architecture/)

Addresses "context rot" — degradation when context grows too long.
Two agents: one manages memory, one generates responses.

- **Key for us**: separation of memory management from generation

### Survey: Memory in the Age of AI Agents
**Collection**: [github.com/Shichun-Liu/Agent-Memory-Paper-List](https://github.com/Shichun-Liu/Agent-Memory-Paper-List)

Comprehensive paper list covering all memory approaches for LLM agents.

### Common Thread
None use explicit labeling. All auto-segment based on content:
- **Surprise boundaries** (EM-LLM) — information density drives segmentation
- **Keyword extraction** (Pichay) — lightweight bookmarks, model-driven recall
- **Attention patterns** (InfLLM) — token-relevant memory lookup
- **Temporal decay** (MemoryBank) — forgetting curve prunes old facts

### Applicability to larql
| Approach | Effort | Matches Our Stack | Key Benefit |
|----------|--------|-------------------|-------------|
| Pichay bookmarks | S | Yes — keyword extraction + recall tool | Immediately actionable, model-driven |
| EM-LLM segmentation | M | Partial — need surprise metric from model | Best retrieval quality, no labels |
| InfLLM memory units | L | Yes — fits our KV cache architecture | Training-free, proven at 1M tokens |
| MemoryBank forgetting | S | Yes — add decay to RAG store | Prevents bloat in long sessions |

**Recommended order**: Pichay bookmarks first (S effort, immediate), then
EM-LLM segmentation (M effort, best quality), then InfLLM if scale needed.

### Three Variants Tested — All Score 6/11 (2026-04-18)
Implemented all three chunking strategies with token-mean embedding retrieval:
- EM-LLM surprise segmentation: 6/11
- Pichay page table: 6/11
- InfLLM per-turn memory units: 6/11

All identical. **Chunking doesn't matter when the matching is token-mean embeddings.**
The papers' innovation is neural matching, not text structuring.

### Key Insight: Retrieval ≠ Inference (2026-04-18)
Two separate paths, two separate precision requirements:
- **Inference**: Q4_K Metal GPU at 35+ tok/s — production, stays as-is
- **Retrieval**: needs bf16 precision at the copy head (L29 H4)

The bf16 forward pass is needed ONLY for:
- INSERT: once per fact (extract K vector)
- QUERY: once per user question (extract Q vector)
- NOT per decode token

A 1-second bf16 retrieval pass before 35 tok/s Q4_K decode is fine.
The copy head (L29 H4, Δ=+0.35) works for intra-prompt matching.
The retrieval needs to happen WITHIN the same forward pass context.

### Next Step: bf16 Retrieval via MLX
Use chuk-lazurus MLX infrastructure for bf16 retrieval:
1. Load Gemma 3 4B in bf16 via MLX (already downloaded)
2. Prefill facts as one window, extract K at L29 H4 per position
3. On query, prefill query appended to fact window, extract Q at L29 H4
4. Q·K matching within the same context → proper copy head retrieval
5. Return matched fact text → inject via Q4_K RAG or vec_inject

This separates retrieval (MLX bf16, slow but accurate) from
inference (Rust Q4_K Metal, fast). Two processes, two precisions.

---

## Session: 2026-04-17

### KNN Token Override (PROVEN)
- **cos=1.0** for positional override when paths match (Q4_K prefill + per-layer probe)
- Multi-token injection works: `bash\nls` → `tool\nlist` in one shot
- Two-layer architecture (L1: ```→tool, L2: →command) prevents cross-fire
- 14/14 golden tests, 11 commands, model fills args from context
- Cross-phrasing: "capital of Australia" → "Rome" fires at cos=0.89

### RAG Context Injection (WORKING, NEEDS BETTER RETRIEVAL)
- Embedding RAG (mean-of-token-embeddings): 5/11 scenarios pass
- Facts inserted instantly (no forward pass — just embedding lookup)
- "what is my name?" → "Miguel" works via RAG context injection
- Model answers with zero conversation history in prompt
- **Problem**: mean-of-token-embeddings gives generic, non-discriminating vectors
  - Long sentences converge to same centroid
  - Short focused facts work better than raw conversation turns

### KV-RAG (EXPLORED, PARTIAL RESULTS)
- Extracted K vectors from attention heads (chuk-lazurus approach)
- Head sweep: L24 H2 best for Gemma 3 4B (4/5 hits in isolation)
- Pre-RoPE vs post-RoPE: no significant difference
- **Problem**: last-position K of long sentences is generic (same issue as embeddings)
- Answer-token K vectors would help for structured facts but wrong paradigm for conversation

### Path Mismatch (FIXED)
- f32 capture + Q4_K inference = cos=0.10 (broken)
- Q4_K capture + Q4_K inference = cos=1.0 (fixed)
- `capture_knn_key_gpu`: GPU sequential prefill matches inference path exactly
- `capture_knn_key_perlayer`: Q4_K prefill + rollback + per-layer probe for value injection

### Key Insight
The conversation-as-knowledge vision needs **better fact extraction**, not more retrieval mechanisms:
1. Raw conversation turns are too verbose for embedding matching
2. Distilled key statements ("Gemma 3 4B at 35 tok/s on Metal") match much better
3. The retrieval infrastructure (embedding RAG + KV-RAG) is solid
4. The bottleneck is what we INSERT, not how we retrieve

### Architecture Comparison
| Mechanism | Use Case | Speed | Quality |
|-----------|----------|-------|---------|
| KNN token override | Tool activation | cos=1.0, 35 tok/s | 14/14 golden |
| Embedding RAG | Conversation memory | Instant INSERT | 5/11 scenarios |
| KV-RAG (L24 H2) | Precise fact retrieval | ~5s INSERT | 4/5 isolated, 3/9 in mix |
| Cross-phrasing KNN | Known facts | cos=0.89 | Works for training-data facts |

### Iteration Log
| Run | Score | Change | Honest? |
|-----|-------|--------|---------|
| Baseline (raw session turns) | 4/11 | — | ✓ |
| Assistant-only + sentence split | 5/11 | skip user msgs | ✓ |
| + seeded key facts | 7/11 | 10 curated facts | ✗ curation |
| Key facts ONLY (no session noise) | 8/11 | removed 3491 noisy sentences | ✗ curation |
| Query-aligned phrasings | 11/11 | restated questions as facts | ✗ cheating |

**Honest score: 5/11** with raw session data, no curation.
The 11/11 was achieved by engineering facts to match queries — not real progress.

### Remaining Failures (3/11)
- **tok/s**: "GPU decode runs at 35-41 tok/s" fact not retrieved for "how fast is decode"
- **port 3000**: "server default port is 3000" not retrieved for "what port"
- **Metal GPU**: "Metal M4 Pro" not retrieved for "what GPU"
All three: mean-of-token-embeddings doesn't match query→fact at cos>0.55

### The Real Objective
Load the raw session transcript (14.9 MB, 2257 turns) with NO curation,
and answer any question about the conversation correctly. Like chuk-lazurus
does with the Apollo 11 transcript — raw document, no engineering.

### Embedding Experiments (all failed to beat token-mean)
| Method | Cosine Range | Discrimination | Verdict |
|--------|-------------|----------------|---------|
| Token-mean (bag of words) | 0.55-0.65 | some | **best so far** (5/11) |
| L12 last-position hidden state | 0.37-0.52 | none | worse — encodes next-token, not meaning |
| L12 mean-pooled all positions | 0.87-0.91 | none (everything matches) | worst — encodes conversation position |
| KV-RAG L24 H2 (K vectors) | 0.72-0.81 | weak | 4/5 isolated, 3/9 in mix |

**Conclusion**: generic embeddings (token-mean, hidden states) can't do semantic retrieval on this model. Need the model's **specialized retrieval head** — the attention head trained to do fact lookup. This requires ablation analysis to find it.

### Copy Head Discovery (CONFIRMED)
calibrate_arch.py ablation on Gemma 3 4B IT (bf16):
```
  Head    mean Δ    coverage
  H0:    -0.014      0%
  H1:    -0.008      0%
  H2:    +0.012      0%
  H3:    +0.021     17%
  H4:    +0.351     67%    ← THE COPY HEAD
  H5:    +0.002     17%
  H6:    +0.005     17%
  H7:    +0.038     33%
```
- **query_head=4, retrieval_layer=29, injection_layer=30**
- Zeroing H4: answer prob drops 0.80→0.20, 0.94→0.27, 0.53→0.08
- H4 is 9× stronger than the next head (H7 at +0.038)
- GQA mapping: query_head 4 → kv_head 2

### Why Our KV-RAG Scored Low (2/5 at L29 H2)
We extracted K from the **Q4_K quantized KV cache** (post-RoPE).
chuk-lazurus extracts from **bf16 raw attention projections** (pre-RoPE).
Q4_K quantization destroys the copy head's K vector quality.
The copy head mechanism works in bf16 — we need bf16 K extraction.

### Copy Head Q·K Is NOT a General Retrieval Mechanism (2026-04-18)
The copy head does INTRA-prompt fact copying, not cross-prompt retrieval.
Tested in pure bf16 via MLX — still weak for arbitrary fact matching.

| Test | Score | Notes |
|------|-------|-------|
| Cross-prompt Q·K (bf16) | 1/5 | "Miguel" dominates everything |
| Answer-position K (bf16) | 2/5 | name + speed correct |
| Same-context Q·K (bf16, one window) | 2/5 | port + speed — numbers only |
| Q4_K KV cache K vectors | 2/5 | quantization makes it worse |

The copy head retrieves NUMBER tokens well (3000, 35) but not entity names.
The calibration used fictional tokens (Voltara, Cerulion) — not representative
of real conversation facts.

### Revised Architecture — Retrieval ≠ Injection
Retrieval and injection are SEPARATE mechanisms:
- **Retrieval**: embedding RAG (token-mean, 5/11 honest) — the best we have
- **Injection**: vec_inject at L30 (12 bytes/fact, zero prefill) — zero tokens
- The copy head is for injection (coefficient extraction), not retrieval
- Retrieval can use any mechanism — embeddings, BM25, hybrid

### Revised Plan (2026-04-18)

**Two independent tracks:**

#### Track A: Improve Retrieval (embedding RAG) [M]
The honest score is 5/11 with token-mean embeddings. The copy head Q·K
doesn't work for cross-prompt retrieval. Better embeddings are the path:

1. **BM25 keyword matching** — simple, handles "port" → "3000" directly
2. **Hybrid BM25 + embedding** — BM25 for precision, embedding for recall
3. **Better chunking** — sentence-level facts, not paragraph-level
4. **Test with scenario suite** — `./tests/rag_scenarios.sh` (64s cycle)

#### Track B: Vec_inject for Zero-Token Injection [M]
Once retrieval finds the right facts, inject them at zero cost:

The injection formula (from chuk-lazurus):
```
e = embed_matrix[token_id]           // answer token embedding
direction = e / dot(e, e)            // normalized direction
h = h + coefficient * direction      // add to residual at L30
```

- 12 bytes per fact: `token_id (u32) + coefficient (f32) + score (f32)`
- Applied at L30 in the per-layer decode path (already supports mid-pipeline ops)
- Linear superposition: multiple facts just add independently
- `has_value_inject` flag already forces per-layer path when KNN entries exist

**Files**: `crates/larql-inference/src/layer_graph/predict.rs` (per-layer path,
line ~700, where value injection already lives)

**What exists**: the per-layer path already has `pending_inject` for value
injection. Vec_inject is simpler — just a vector add, not a full residual blend.
The embed matrix is accessible via `weights` → `embed_tokens_pub`.

### Step 4: Session index file [S]
**Goal**: persist K vectors + coefficients so sessions survive restart.

- Format: `session_facts.npz` — same layout as chuk-lazurus `vec_inject.npz`
  - `k_vecs`: (N_facts, 256) float16 — K vectors at L29 H4
  - `token_ids`: (N_facts,) int32 — answer token per fact
  - `coefs`: (N_facts,) float32 — injection coefficients
  - `facts`: JSON array of fact strings (for debug/display)
- Load at server startup or via `--session` flag
- TUI `--session` loads from Claude JSONL → extracts facts → builds index

**Files**: new `crates/larql-server/src/routes/session_index.rs`

### Verification
Run scenario tests after each step:
- Step 1: `curl /v1/kv-rag/query` — does bf16 K at L29 H4 discriminate? Target: 5/5 clean
- Step 2: integrate into chat, run `./tests/rag_scenarios.sh` — target: >8/11 honest
- Step 3: same scenarios but with vec_inject instead of RAG text — same accuracy, faster
- Step 4: `--session` loads 14.9 MB Claude session, model answers questions

### Dependencies
```
Step 1 (bf16 K extraction)
    ↓
Step 2 (Q·K retrieval)
    ↓
Step 3 (vec_inject at L30)    ← can test with RAG text before this
    ↓
Step 4 (session index file)   ← persistence
```

### Comprehensive Results Table (2026-04-18)
| Method                           | Score | Honest? | Blocked By |
|----------------------------------|-------|---------|------------|
| Token-mean embedding RAG         | 6/11  | ✓       | Embedding gap |
| BM25 + embedding hybrid          | 6/11  | ✓       | Keyword gap |
| EM-LLM neural segmentation       | 6/11  | ✓       | Still embedding RAG underneath |
| Full 20-turn context             | 6/11  | ✓       | Right turns by luck only |
| Cooperative model recall          | 5/11  | ✓       | Generic keywords |
| RAG select → context inject      | 5/11  | ✓       | Wrong turns selected |
| Full 4000-token context          | 2/11  | ✓       | Prefill timeout (100s) |
| Copy head Q·K cross-prompt       | 2/5   | ✓       | Q·K doesn't match cross-prompt |

**6/11 is the ceiling for retrieval-based approaches.**

### KV Cache Precompute + Replay (BREAKTHROUGH, 2026-04-18)
Precompute conversation KV cache ONCE, restore per query via memcpy.

| Metric | Before (re-prefill) | After (KV replay) |
|--------|--------------------|--------------------|
| Prefill cost per query | 30-45s | **36ms** |
| Decode speed | 30-35 tok/s | **34-36 tok/s** |
| One-time precompute | — | 30.6s for 1048 tokens |
| KV state size | — | 278 MB (34 layers) |
| Score | 6/11 (20 turns) | **6/11 (20 turns)** |

Same accuracy, 10× faster. The model perceives restored KV as context
(confirmed: answers "Gemma 3 4B IT" from KV context).

The 6/11 limit is NOT retrieval — it's the 20-turn window.
Metal, ratatui, port 3000, clone are in LATER turns not included.
With the full 2257-turn conversation in KV, the model would attend
to all facts naturally. Just need to precompute more turns.

### KV Replay Scale Test (2026-04-18 — continuation)

Fixed KV restore for multi-token prefill: when precomputed KV exists, process
user query tokens one-by-one through KV-cached attention (reads past K/V,
appends at correct position) instead of multi-token prefill (which overwrites
KV from position 0). Previous "garbled output" was from overwriting.

| Precomputed tokens | KV size | Quality | Notes |
|--------------------|---------|---------|-------|
| 40                 | 10.6 MB | Perfect | "Paris" — exact |
| 168                | 44.6 MB | Perfect | 20 capitals — all correct |
| 387                | 102.8 MB| Perfect | 50 capitals — all correct |
| 533                | 141.6 MB| Perfect | "BUTTERFLY" code word, "Oscar Niemeyer" |
| 925                | 245.7 MB| Perfect | 5/5 including exact password + color |
| 1800               | 478.1 MB| Degraded| Early ✓, late ✗ ("333" hallucination) |
| 1890               | 502.0 MB| Degraded| Early ✓, middle/late loses precision |
| 2270               | 603.0 MB| Poor    | 1/9 recall on raw conversation dump |
| 158 (compact facts)| 42.0 MB | **10/11** | Focused context → near-perfect |

**The mechanism works perfectly. The limit is attention distance, not KV.**

Gemma 3 4B attention reliably retrieves facts up to ~925 tokens. Beyond that,
"lost in the middle" causes progressive degradation — the model produces
coherent but incorrect answers (e.g., "333" for everything, wrong dates).

**Compact context is the key:** 158 tokens of focused facts → 10/11 (vs 6/11
with RAG, vs 1/9 with 2270 raw tokens). The optimal strategy:

1. **RAG → KV**: use RAG to select ~200 tokens of relevant context per query
2. **Summary → KV**: precompute a compact conversation summary into KV
3. **Pichay bookmarks**: minimal keyword bookmarks (~24 tokens) as persistent KV

| Approach                | Score  | Context | Latency |
|-------------------------|--------|---------|---------|
| Compact KV (158 tok)    | 10/11  | Focused | 36ms restore |
| Conv KV (4120 tok)      | **7/11** | Full conv | 36ms restore |
| RAG → context inject    | 5-6/11 | Selective | 2-5ms |
| Raw conv KV (2270 tok)  | 3/11   | Dense dump| 36ms restore |
| No context (baseline)   | 2/11   | None    | 0ms |

### Root Cause of `<unused>` Garbled Output (2026-04-18)

The `<unused>` token garbage at 4000+ tokens was caused by **Metal shader
threadgroup buffer overflow**:

- `kv_attention` shader had `threadgroup float tg_scores[1024]`
- KV-cached decode with T > 1024 wrote past the end of `tg_scores`
- This silently corrupted threadgroup memory → garbage attention weights
- At T < 1024 (decode from scratch), attention worked correctly
- At T > 1024 (KV restore with 4000+ precomputed tokens), overflow → garbled

**Fix**: Increased `tg_scores` to 8160 entries (32,640 bytes — within Metal's
32KB threadgroup memory limit). Also increased KV cache buffers from 4096 to
8160 max_seq to support 8K context.

Additional fix: KV buffer size was 4096 max_seq (16 MB/layer), which panicked
at 4120+ tokens. Increased to 8160 (32 MB/layer, total ~2.1 GB for 34 layers).

### Head-to-Head: larql vs Ollama (2026-04-18)

Same model (Gemma 3 4B Q4_K), same 12 facts, same 11 scenarios.
Both on M4 Pro Metal GPU. Run sequentially to avoid contention.

**larql**: facts baked into KV cache via `/v1/kv/compact` (234 tokens,
one-time 5.8s precompute). Per-query: KV restore (~36ms) + batch prefill
user tokens + decode.

**Ollama**: facts in system prompt, re-prefilled on every query via
llama.cpp flash attention batch prefill.

| Metric | larql | Ollama |
|--------|------:|-------:|
| Score | **11/11** | 10/11 |
| Avg latency | 751 ms | 556 ms |
| RSS | **826 MB** | 4142 MB |

**Quality**: larql wins (11/11 vs 10/11). Both fail the same joke-leak
edge case, but larql's compact system prompt avoids it.

**Latency**: Ollama 1.35x faster. Breakdown:
- KV restore: ~36ms (larql advantage — Ollama re-prefills 234 tokens)
- User query prefill: ~350ms (larql bottleneck — sequential per-token)
- Decode: ~400ms for ~15 tokens (~38 tok/s both)

The gap is in user query prefill. Ollama's llama.cpp uses flash attention
to process all user tokens in parallel. larql's `decode_token_batch`
processes tokens sequentially through 34 layers (causal dependency).

**RAM**: larql 5x less. Model weights are mmap'd (OS manages paging).
Ollama loads the full GGUF into process memory.

**Crossover analysis**: At 234 tokens of facts, Ollama's batch prefill
is fast (~100ms for 234 tokens). larql's KV restore is 36ms — saves only
~64ms, which doesn't offset the user query prefill overhead. At 1000+
tokens of facts, Ollama must re-prefill 1000+ tokens per query (~400ms+)
while larql still restores in 36ms. The crossover favors larql at scale.

### Batch Decode Bug Fix (2026-04-18)

`decode_token_batch` had a norm-ordering bug: batched QKV projection at
`bi=0` read `norm_buf` for all K positions, but only position 0's input
norm had been dispatched. Metal serializes compute dispatches within an
encoder, so QKV read uninitialized data for positions 1..K-1.

**Fix**: Restructured into three phases per layer:
1. Input norms for ALL K positions
2. Batched QKV projection (reads weights once)
3. Per-position QK-norm, RoPE, V-norm

This fixed garbled output from batch prefill and improved latency from
868ms to 751ms (the batch now produces correct hidden states).

### Path to Beating Ollama on Latency

The remaining 195ms gap comes from sequential user query prefill.
Each token runs through all 34 layers one at a time (~20ms/tok).
The fix: **flash attention with past KV** — a Metal shader that
processes all user tokens against the restored KV cache in one dispatch.

The `fused_attention_prefill` shader already does batch causal attention
for self-contained prompts. Extending it to read past K/V from the
cache would eliminate the per-position attention loop. The FFN is
inherently sequential (token i's output feeds token i+1's input), but
attention can be parallelized since each position's Q only reads past K/V.

Projected improvement: user prefill from ~350ms to ~50ms, total from
751ms to ~450ms — beating Ollama's 556ms.

### Performance Profiling Deep Dive (2026-04-18)

#### Phase Timing (chat with KV-baked facts, 12 tokens user query)

| Phase | Time | % of total | What happens |
|-------|-----:|----------:|----|
| KV restore | 6ms | 1% | memcpy saved KV into Metal buffers |
| Prefill | 290ms | 48% | user query tokens through 34 layers |
| Decode | 200-750ms | 51% | autoregressive token generation |
| **Total** | **500-1050ms** | | depends on output length |

#### Bandwidth Utilization Analysis

| Metric | Value |
|--------|------:|
| M4 Pro memory bandwidth | 273 GB/s |
| Weight reads per token (Q4_K) | 1855 MB (54.6 MB/layer × 34) |
| **Theoretical floor** | **6.8 ms/tok (147 tok/s)** |
| **Actual** | **25 ms/tok (40 tok/s)** |
| **Bandwidth utilization** | **27%** |

73% of GPU time is wasted on dispatch overhead, pipeline bubbles,
and suboptimal memory access patterns.

#### Dispatch Count Analysis

| Path | Dispatches | Overhead at ~5µs each |
|------|----------:|----:|
| Single-token decode | 50/layer × 34 = 1,700 | 8.5ms |
| Batch prefill K=17 | 414/layer × 34 = 14,076 | 70ms |

The batch prefill's 14K dispatches account for 70ms of the 290ms prefill
(24%). Single-token decode's 1,700 dispatches are 8.5ms of 25ms (34%).

#### Memory Map — What We Actually Read

| File | Size | Used during inference? |
|------|-----:|:---:|
| interleaved_q4k_real.bin | 1,474 MB | YES — Q4_K FFN weights |
| attn_weights_q4k.bin | 418 MB | YES — Q4_K attention |
| lm_head_q4.bin | 360 MB | YES — Q4_K logits |
| embeddings.bin | 1,280 MB | YES — token embedding lookup |
| norms.bin | 0.7 MB | YES — layer norms |
| safetensors (f32) | 8,201 MB | NO — loaded at warmup, never read during Q4_K decode |
| interleaved.bin (f32) | 10,200 MB | NO — walk-only path |
| gate/up/down vectors | 6,800 MB | NO — walk-only path |
| **Active total** | **~3,533 MB** | |
| **Inactive mmap** | **~25,000 MB** | |

#### Code Cleanup Results

| File | Before | After | Removed |
|------|-------:|------:|--------:|
| predict.rs | 1,440 | 1,013 | -427 (-30%) |
| decode_hybrid.rs | 324 | 0 | -324 (deleted) |
| hybrid.rs | ~200 | 0 | -200 (deleted) |
| **Total** | | | **-1,110 lines** |

### Optimization Plan — Ranked by Impact

**Goal**: close the gap to Ollama (556ms) and eventually beat it.
Current: 781ms, 40 tok/s, 27% bandwidth utilization.

#### 1. Fuse Dispatches in Single-Token Decode [S]

Merge norm+matvec into fewer dispatches per layer. Currently 50
dispatches/layer; fusing pre-norm into the matvec dispatch could
halve this to ~25/layer.

- **Test**: decode one token, compare output to baseline (cosine > 0.9999)
- **Measure**: tok/s before and after
- **Projected**: 40 → 56 tok/s (+40%)

#### 2. Drop f32 Safetensors After Warmup [S]

The f32 weights (8.2 GB) are loaded for warmup but never read during
Q4_K decode. Free them after initialization.

- **Test**: run full scenario benchmark, verify 11/11 unchanged
- **Measure**: RSS before and after
- **Projected**: -8 GB RSS, no speed change

#### 3. Drop Unused Vindex Mmaps [S]

25+ GB of mmap'd files never read during inference. Close the mmaps
to reduce virtual memory and eliminate stale page faults.

- **Test**: same as above
- **Measure**: virtual memory, page fault count
- **Projected**: -25 GB virtual, fewer page faults

#### 4. Mega-Kernel: One Dispatch Per Layer [L]

All norms + QKV + attention + O + FFN in a single Metal kernel per
layer. Uses `threadgroup_barrier` for intra-layer sync. Eliminates
dispatch overhead entirely.

- **Test**: compare output token-by-token vs sequential decode
- **Measure**: tok/s, bandwidth utilization
- **Projected**: 40 → 100+ tok/s (+150%), ~70% bandwidth utilization

#### 5. f16 KV Cache [M]

KV cache is f32 (32 MB/layer). f16 halves bandwidth. Marginal now
(KV reads are 0.26ms at T=250) but critical at T=4000+ where KV
bandwidth becomes significant.

- **Test**: cosine similarity of attention output vs f32 KV
- **Measure**: latency at T=500, T=2000, T=4000
- **Projected**: 2× longer context at same speed

#### Iteration Loop

For each optimization:
1. **Correctness**: quick check — "What port?" → "3000" (5 seconds)
2. **Timing**: 5 decode queries, read tok/s from logs (30 seconds)
3. **Benchmark**: full 11 scenarios + Ollama comparison (5 minutes)
4. **Commit or revert**: if regression, `git checkout -- file`

### Memory Optimization Results (2026-04-18/19)

#### What was freed

| What | Bytes freed | Method |
|------|----------:|--------|
| f32 FFN weights (gate+up+down) | 10.7 GB | `drop_ffn_weights()` after load |
| f32 attention weights (Q/K/V/O) | 2.1 GB | `drop_attn_weights()` after load |
| f32 lm_head | 2.7 GB | `drop_lm_head_weight()` after load |
| Embedding dedup (safetensors→shared ArcArray2) | 2.7 GB | Replace `weights.embed` with shared ref |
| Unused vindex mmaps (MADV_DONTNEED) | 11.6 GB | `advise_dontneed_unused()` on down/up/gate/lm_head f32 |
| **Total reclaimed** | **29.8 GB** | |

#### Precise memory map (vmmap, after 10 inference queries)

```
Category                     Virtual    Resident    Dirty    Swapped
─────────────────────────── ─────────  ─────────  ───────  ─────────
mapped file (mmaps)           13.1 GB    779.9 MB      0 MB     0 MB
MALLOC zone (heap)             6.3 GB     38.2 MB   35.3 MB   6.2 GB
__TEXT (code + libraries)     300.5 MB   137.9 MB      0 MB     0 MB
__LINKEDIT + __OBJC_RO        671.6 MB    58.7 MB      0 MB     0 MB
Other (page tables, stack)     ~70 MB     ~35 MB    ~28 MB    ~3 GB
─────────────────────────── ─────────  ─────────  ───────  ─────────
TOTAL                          26.9 GB     4.0 GB   3.1 GB   9.5 GB
```

**What's actually used for inference:**

| Resource | Resident | Notes |
|----------|--------:|-------|
| lm_head_q4.bin | 360 MB | Fully paged in (used every token for logits) |
| attn_weights_q4k.bin | 418 MB | Fully paged in (Q4_K attention projections) |
| interleaved_q4k_real.bin | ~0 MB | Released by MADV_DONTNEED, re-paged per layer during decode |
| Live heap (malloc dirty) | 35 MB | Norms, KV state, scratch buffers |
| **Working set** | **~815 MB** | |

The 6.2 GB MALLOC SWAPPED is freed heap (the dropped f32 tensors)
that macOS hasn't unmapped yet — the allocator holds the pages in its
arena. Under memory pressure the OS reclaims them. The live dirty
heap is only 35 MB.

**Comparison with Ollama:**

| Metric | larql | Ollama |
|--------|------:|-------:|
| Process RSS | 1,775 MB | 4,142 MB |
| Working set (active) | ~815 MB | ~3,300 MB |
| Speed | 42.6 tok/s | ~40 tok/s |
| Quality (11 scenarios) | 11/11 | 10/11 |

#### Dispatch fusion results

Fused QK-norm from 12 dispatches to 2 per layer (new `rms_norm_multihead`
shader). Total dispatches 918→578 (-37%). Speed 40.4→41.3 tok/s (+2%).

Per-layer dispatch cost analysis: 4 matvec dispatches (QKV, O, gate+up,
GEGLU+down) account for 97% of compute. The other 13 dispatches are
element-wise norms and residuals at <2% combined. Further fusion is a
dead end — the matvec kernel itself is the bottleneck.

#### Remaining path to lower memory

The 815 MB working set is the Q4_K weights being actively read during
inference. To go lower requires:
- Smaller quantization (Q2_K — would lose quality)
- Smaller model (not Gemma 3 4B)
- Lazy layer loading (only page in the current layer's weights — would
  add ~1ms latency per layer from page faults)
