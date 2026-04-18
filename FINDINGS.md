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
