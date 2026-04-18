# Paper Technical Details — Retrieval for Long Conversations

## EM-LLM (ICLR 2025)
**Paper**: [arxiv.org/abs/2407.09450](https://arxiv.org/abs/2407.09450)
**Code**: [github.com/em-llm/EM-LLM-model](https://github.com/em-llm/EM-LLM-model)

### Surprise-Based Segmentation
Surprise = negative log-likelihood of the current token:
```
Surprise(t) = -log P(x_t | x_1,...,x_{t-1}; θ)
```
Boundary threshold (adaptive window):
```
T = μ_{t-τ:t} + γ·σ_{t-τ:t}
```
Tokens exceeding T become episode boundaries.

### Graph-Theoretic Boundary Refinement
Adjacency matrix from attention K vectors:
```
A^h_{ij} = K^h_i^T · K^h_j     (dot-product similarity)
```
Optimize boundary positions to maximize modularity / minimize conductance.
This is the KEY neural matching — **K vector dot products** determine
which tokens belong to the same episode. Same mechanism as chuk-lazurus.

### Two-Stage Retrieval
1. **Similarity retrieval**: k-NN using dot-product on representative tokens
   per event. Compare query against most influential tokens in each episode.
2. **Temporal contiguity**: queue-based buffer, neighboring events retrieved
   automatically. Decays naturally as new events processed.

### Memory Buffers
- Initial tokens (128) — attention sinks
- Local context — recent window
- Retrieved events (k_s + k_c) — from episodic memory

### Key for larql
The adjacency matrix uses **K vector dot products** — this IS the copy head
mechanism. EM-LLM uses K·K similarity to determine event coherence, which
is the same Q·K matching that the copy head does. The difference: EM-LLM
computes K·K WITHIN the same forward pass (intra-prompt), not across prompts.

---

## Pichay (2026)
**Paper**: [arxiv.org/abs/2604.12376](https://arxiv.org/abs/2604.12376)

### Bookmark Format
`[p3:allergy,peanut,budget]` — 8-24 tokens per bookmark.
Keywords: capitalized tokens, numbers, dates from first 3-4 turns,
filtered against 60-word stopword list, limited to 3-5 keywords.

### Page Fault = Model Calls recall()
No passive detection (NLL-based fault detection doesn't work — models
don't "struggle" when lacking context, they generate confident wrong answers).
Instead: model sees bookmarks, calls `recall(page_ids=[3,5])`.

### Critical Finding: Minimal > Verbose
| Bookmark Style | Accuracy | Tokens |
|---------------|----------|--------|
| ID only `[p1]` | 9.1% | 4 |
| Minimal keywords | 63.6% | 24 |
| Medium (keywords + quoted text) | 54.5% | 91 |
| Structured (entity fields) | 59.1% | 78 |

Verbose bookmarks SUPPRESS recall calls — model thinks it has enough info.

### Keyword Specificity = 25 Point Swing
- Generic ("personal preferences"): 65.2%
- Domain-specific ("dietary pref., vegetarian"): 90.9%
- Delta: +25.7 percentage points

### Eviction Policies
- FIFO best on synthetic, worst on real conversations
- LFU worst on synthetic, second-best on real
- Fixed_20 (20 turns/page): 96.7% accuracy — coarse pages beat semantic splits

### Key for larql
1. The model must SEE the bookmarks (not hidden in system prompt)
2. Minimal bookmarks trigger better recall behavior
3. Domain-specific keywords are critical (+25 points)
4. Fixed-size pages beat semantic segmentation (simpler is better)

---

## InfLLM (2024)
**Paper**: [arxiv.org/abs/2402.04617](https://arxiv.org/abs/2402.04617)

### Memory Unit Architecture
Past tokens divided into blocks of 128 tokens (l_bs=128).
Three regions:
- **Initial tokens (I)**: system prompts, attention sinks
- **Evicted tokens (E)**: in CPU memory as blocks
- **Local tokens (L)**: recent sliding window on GPU

### Representative Token Selection
Each block selects 4 representative tokens (r_k=4):
```
r_m = (1/l_L) Σ(q_{m+j} · k_m)
```
Tokens with highest representative scores = unit representatives.
This is a Q·K dot product — same mechanism as copy head / EM-LLM.

### Attention-Based Memory Lookup
Score blocks by Q·K similarity:
```
sim(X, B) = Σ Σ q_i · k_{b_j}^B
```
Top-k blocks loaded from CPU to GPU for full attention.
GPU cache: 32-96 blocks with LRU eviction.

### Positional Encoding Trick
All evicted tokens get SAME positional embedding (set to l_L).
Avoids out-of-domain RoPE issues. Relies on causal decoder
structure to preserve sequence order instead.

### Key for larql
1. Block-level granularity (128 tokens) — not per-token
2. Representative tokens via Q·K scoring — same neural matching
3. CPU↔GPU memory hierarchy — we have this with Metal
4. Same-position encoding avoids RoPE mismatch (our problem!)

---

## Common Thread Across All Three

ALL THREE use **Q·K dot products** for their core matching:
- EM-LLM: K·K adjacency matrix for event boundaries
- Pichay: model's internal attention (implicit, via recall behavior)
- InfLLM: Q·K similarity scoring for block selection

The matching happens WITHIN the model's forward pass, not as an
external embedding comparison. The K vectors come from the same
KV cache context, not from separate prompts.

This confirms: the copy head mechanism (Q·K at L29 H4) IS the right
approach. The problem was trying to use it ACROSS prompts. These papers
use it WITHIN the same context — facts are in the KV cache, query
attends to them via the model's own attention.

## Implication for larql
The facts need to be IN the KV cache (as tokens or as replayed
windows) so the model's attention can naturally retrieve them.
External embedding comparison (token-mean, BM25) is a proxy that
tops out at 6/11. The neural mechanism works within the forward pass.

Options:
1. **Replay fact windows into KV cache** (chuk-lazurus unlimited context)
2. **InfLLM memory units** — store KV blocks on CPU, load relevant ones
3. **EM-LLM episodes** — segment, store as blocks, retrieve by K similarity
