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
| Run | Score | Change |
|-----|-------|--------|
| Baseline (raw session turns) | 4/11 | — |
| Assistant-only + sentence split | 5/11 | skip user msgs |
| + seeded key facts | 7/11 | 10 curated facts |
| Key facts ONLY (no session noise) | 8/11 | removed 3491 noisy sentences |

### Remaining Failures (3/11)
- **tok/s**: "GPU decode runs at 35-41 tok/s" fact not retrieved for "how fast is decode"
- **port 3000**: "server default port is 3000" not retrieved for "what port"
- **Metal GPU**: "Metal M4 Pro" not retrieved for "what GPU"
All three: mean-of-token-embeddings doesn't match query→fact at cos>0.55

### Next Steps
1. **Better embeddings**: TF-IDF weighting or keyword-boosted embeddings
2. **Scenario iteration**: `./tests/rag_scenarios.sh` — 90s feedback loop
3. **Session loading**: `--session` loads Claude sessions into RAG at startup
4. **Combine RAG + KV-RAG**: embedding for broad recall, KV for precision
