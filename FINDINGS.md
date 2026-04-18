# Findings — Conversation as Knowledge

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

### What Needs to Change
The copy head (L29 H4) is confirmed. The retrieval mechanism is known.
The blocker is extracting K vectors in bf16 (not Q4_K from the KV cache).

**Plan — three steps to conversation-as-knowledge:**

1. **bf16 K extraction at L29 H4**
   - Load bf16 attention weights for L29 (just Q/K projections, ~20 MB)
   - Run K projection in bf16 on the hidden state at L29
   - Store per-position K vectors for each conversation turn
   - This is what chuk-lazurus does in `calibrate_arch.py`

2. **Q·K retrieval at inference time**
   - On each user query, compute Q at L29 H4 (bf16 projection)
   - Cosine score against stored K vectors
   - Top-k retrieval with adaptive threshold
   - Inject matched facts as RAG context OR vec_inject at L30

3. **Session loading via K-vector index**
   - Each conversation turn → K vectors at L29 H4 for each token position
   - Store as vec_inject.npz (12 bytes per fact: token_id + coefficient)
   - Load at startup, retrieve in <1ms (Metal matmul)
   - Same format as chuk-lazurus — interoperable

### Vec_inject vs RAG — Decision Made
RAG text injection costs prefill tokens (8 tokens/fact × 1000 facts = 200s).
Vec_inject costs 12 bytes/fact, zero prefill. At scale, vec_inject wins by orders of magnitude.

**Decision: implement vec_inject.** RAG text injection is the fallback.

## Implementation Plan — Vec_inject for Conversation as Knowledge

### Step 1: bf16 K extraction at L29 H4 [M]
**Goal**: capture the copy head's K vector for each fact during INSERT.

- Load f32/bf16 attention weights for L29 Q/K projections (~20 MB from SafeTensors)
- Forward pass through layers 0–29 (can use existing Q4_K GPU pipeline for L0–28,
  then bf16 Q/K projection at L29 only)
- Extract K at query_head=4 (256-dim vector) for the last token position
- Also extract coefficient: `c = dot(h_L30[last_pos], embed(answer_token))`
  where answer_token is the key entity in the fact
- Store per fact: `{k_vector: [256 floats], token_id: u32, coefficient: f32}`

**Files**: `crates/larql-server/src/routes/kv_rag.rs` (update extract_k_vector),
`crates/larql-inference/src/attention/gpu.rs` (bf16 Q/K projection at L29)

**What exists**: f32 attention weights in SafeTensors (loaded as `weights.tensors`),
embed matrix in `model.embeddings`. The per-layer decode path already runs
through individual layers. Just need to do the K projection in f32 at L29
instead of reading from Q4_K KV cache.

### Step 2: Q·K retrieval at inference [S]
**Goal**: find matching facts in <1ms using Metal matmul.

- On each decode step, extract Q at L29 H4 (same bf16 projection)
- `scores = K_matrix @ q_norm` — one Metal matmul (N_facts × 256)
- Top-k with adaptive threshold: `max(0.15, mean_score × 2.0)`
  (from chuk-lazurus: fixed thresholds fail at N>50)
- Return matched fact indices + coefficients

**Files**: `crates/larql-server/src/routes/kv_rag.rs` (query path),
or integrate directly into predict.rs per-layer path

**What exists**: KvRagStore already has cosine scoring. Need to switch
from KV-cache K to bf16 K, and add Q extraction.

### Step 3: Vec_inject at L30 [S]
**Goal**: inject matched facts into residual stream, zero prefill tokens.

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
