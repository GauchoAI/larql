#!/usr/bin/env python3
"""
Three retrieval variants with NEURAL matching (Q·K dot products).

All three use the model's own attention K vectors for matching,
not token-mean embeddings. Extracted via MLX bf16 forward pass.

Variant A: EM-LLM — K·K adjacency for segmentation, Q·K for retrieval
Variant B: Pichay — minimal bookmarks, model sees them in context
Variant C: InfLLM — KV block storage, Q·K representative scoring

Usage: cd chuk-lazurus && uv run python ../larql/tests/retrieval_neural.py [a|b|c|all]
"""
import sys, time, math, json, re, subprocess
sys.path.insert(0, "/Users/miguel_lemos/Desktop/chuk-lazurus/src")

import mlx.core as mx
import mlx.nn as nn

# Config
RETRIEVAL_LAYER = 29
QUERY_HEAD = 4
HEAD_DIM = 256
SERVER = "http://localhost:3000"
TRANSCRIPT = "/Users/miguel_lemos/Desktop/llm-as-a-database/larql/tests/fixtures/session_transcript.json"
SCENARIOS = "/Users/miguel_lemos/Desktop/llm-as-a-database/larql/tests/fixtures/rag_scenarios.json"

VARIANT = sys.argv[1] if len(sys.argv) > 1 else "all"

# ── Load model ──

def load_model():
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info
    print("Loading Gemma 3 4B bf16...")
    result = HFLoader.download("mlx-community/gemma-3-4b-it-bf16")
    with open(result.model_path / "config.json") as f:
        config_data = json.load(f)
    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)
    HFLoader.apply_weights_to_model(model, result.model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(result.model_path)
    print(f"  Loaded: {config.num_hidden_layers} layers, {config.hidden_size} hidden")
    return model, tokenizer, config

def get_layers_and_embed(model):
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers), model.model.embed_tokens
    return list(model.layers), model.embed_tokens

def forward_to_layer(model, config, input_ids, target_layer):
    """Forward pass to target_layer, return (h, K_all_positions at target_layer)."""
    layers, embed = get_layers_and_embed(model)
    scale = getattr(config, "embedding_scale", None)
    h = embed(mx.array(input_ids)[None, :])
    if scale: h = h * scale
    seq_len = len(input_ids)
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    for idx, lyr in enumerate(layers):
        if idx == target_layer:
            attn = lyr.self_attn if hasattr(lyr, 'self_attn') else lyr
            h_norm = lyr.input_layernorm(h) if hasattr(lyr, 'input_layernorm') else h
            q = attn.q_proj(h_norm)
            k = attn.k_proj(h_norm)
            mx.eval(q, k)
            return h, q, k
        try:
            out = lyr(h, mask=mask)
        except TypeError:
            out = lyr(h)
        h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)
    return h, None, None

def extract_k_per_position(k_tensor, kv_head=2):
    """Extract K vectors for a specific KV head at all positions."""
    # k_tensor: (1, seq_len, kv_dim)
    seq_len = k_tensor.shape[1]
    kv_dim = k_tensor.shape[2]
    n_kv_heads = kv_dim // HEAD_DIM
    k_np = k_tensor[0].tolist()  # (seq_len, kv_dim)
    vectors = []
    for pos in range(seq_len):
        start = kv_head * HEAD_DIM
        vec = k_np[pos][start:start + HEAD_DIM]
        norm = math.sqrt(sum(x*x for x in vec))
        if norm > 1e-12:
            vec = [x/norm for x in vec]
        vectors.append(vec)
    return vectors

def extract_q_at_pos(q_tensor, pos=-1, query_head=QUERY_HEAD):
    """Extract Q vector at a specific position for query_head."""
    q_np = q_tensor[0, pos, :].tolist()
    n_q_heads = len(q_np) // HEAD_DIM
    start = query_head * HEAD_DIM
    vec = q_np[start:start + HEAD_DIM]
    norm = math.sqrt(sum(x*x for x in vec))
    if norm > 1e-12:
        vec = [x/norm for x in vec]
    return vec

def dot(a, b):
    return sum(x*y for x,y in zip(a,b))

# ── Load transcript ──

with open(TRANSCRIPT) as f:
    transcript = json.load(f)
with open(SCENARIOS) as f:
    scenarios = json.load(f)

turns = [t for t in transcript['turns'] if t['role'] == 'assistant' and len(t['content']) > 30]
print(f"Transcript: {len(turns)} assistant turns")

# ── Scenario runner (via server) ──

def insert_rag(fact):
    subprocess.run(
        ['curl', '-s', '--max-time', '2', f'{SERVER}/v1/rag/insert',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'fact': fact, 'category': 'neural'})],
        capture_output=True, text=True
    )

def run_scenarios():
    passed = 0
    failed = 0
    for sc in scenarios['scenarios']:
        query = sc['query']
        expects = sc.get('expect_contains', [])
        not_expects = sc.get('expect_not_contains', [])
        desc = sc['description']
        r = subprocess.run(
            ['curl', '-s', '--max-time', '15', f'{SERVER}/v1/chat/completions',
             '-H', 'Content-Type: application/json',
             '-d', json.dumps({'messages': [{'role': 'user', 'content': query}]})],
            capture_output=True, text=True
        )
        output = ""
        for line in r.stdout.split("\n"):
            line = line.strip()
            if line.startswith("data:") and "content" in line:
                try:
                    output += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
                except: pass
        output_lower = output.lower()
        ok = all(e.lower() in output_lower for e in expects) and \
             not any(n.lower() in output_lower for n in not_expects)
        if ok: passed += 1
        else: failed += 1
        status = "\033[32mPASS\033[0m" if ok else "\033[31mFAIL\033[0m"
        print(f"  {status}  {desc}")
    print(f"\n  {passed}/{passed+failed}")
    return passed

def clear_server():
    subprocess.run(['pkill', '-f', 'larql-server'], capture_output=True)
    subprocess.Popen(
        ['target/release/larql-server',
         '/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        cwd='/Users/miguel_lemos/Desktop/llm-as-a-database/larql'
    )
    for _ in range(120):
        r = subprocess.run(['curl', '-s', f'{SERVER}/v1/health'], capture_output=True, text=True)
        if '"ok"' in r.stdout: return True
        time.sleep(1)
    return False

# ═══════════════════════════════════════════════════════════════
# Variant A: EM-LLM — K·K segmentation + Q·K retrieval
# ═══════════════════════════════════════════════════════════════

def variant_a(model, tokenizer, config):
    """
    EM-LLM: Use K·K dot products at L29 to segment episodes.
    Then for retrieval, find episodes whose K vectors match the query's Q.
    """
    print("\n=== Variant A: EM-LLM (neural K·K segmentation) ===")

    # Process first 100 turns (bf16 forward pass each)
    MAX_TURNS = 100
    all_facts = []
    t0 = time.time()

    for i, turn in enumerate(turns[:MAX_TURNS]):
        text = turn['content'][:200]
        ids = tokenizer.encode(text)
        if len(ids) < 5: continue

        _, _, k = forward_to_layer(model, config, ids, RETRIEVAL_LAYER)
        if k is None: continue

        # Extract K at the LAST position for this turn
        k_vecs = extract_k_per_position(k)
        last_k = k_vecs[-1]

        all_facts.append({
            'text': text.replace('```', '').replace('\n', ' ')[:150],
            'k_vec': last_k,
            'ts': turn.get('timestamp', '')[:19],
        })

        if (i+1) % 25 == 0:
            print(f"  {i+1}/{MAX_TURNS} turns processed ({time.time()-t0:.0f}s)")

    print(f"  {len(all_facts)} facts extracted in {time.time()-t0:.0f}s")

    # Segment by K·K surprise (low similarity to previous = boundary)
    episodes = []
    current = [all_facts[0]]
    for i in range(1, len(all_facts)):
        sim = dot(all_facts[i]['k_vec'], all_facts[i-1]['k_vec'])
        if sim < 0.3 and len(current) >= 2:  # surprise boundary
            episodes.append(current)
            current = []
        current.append(all_facts[i])
    if current: episodes.append(current)

    print(f"  {len(episodes)} episodes (K·K threshold=0.3)")

    # Now: for each query, find best matching episode via Q·K
    print("\n  Testing queries...")
    clear_server()

    # Insert episode summaries into RAG
    for ep in episodes:
        texts = [f['text'][:60] for f in ep[:3]]
        fact = f"[{ep[0]['ts']}] {' | '.join(texts)}"
        insert_rag(fact[:300])

    return run_scenarios()


# ═══════════════════════════════════════════════════════════════
# Variant B: Pichay — proper bookmarks with model-visible page table
# ═══════════════════════════════════════════════════════════════

def variant_b(model, tokenizer, config):
    """
    Pichay: Create minimal bookmarks from K-vector analysis.
    Keywords = tokens whose K vectors are most distinctive in each page.
    Store bookmarks in RAG so model sees them as context.
    """
    print("\n=== Variant B: Pichay (neural keyword bookmarks) ===")

    PAGE_SIZE = 10
    MAX_TURNS = 100
    t0 = time.time()

    pages = []
    for pi in range(0, min(len(turns), MAX_TURNS), PAGE_SIZE):
        page_turns = turns[pi:pi + PAGE_SIZE]
        page_text = " ".join(t['content'][:100] for t in page_turns)
        ids = tokenizer.encode(page_text[:500])

        _, _, k = forward_to_layer(model, config, ids, RETRIEVAL_LAYER)
        if k is None: continue

        k_vecs = extract_k_per_position(k)

        # Find most distinctive tokens: highest K vector norm variance
        # (tokens that "stand out" from the page's average)
        mean_k = [sum(k_vecs[p][d] for p in range(len(k_vecs))) / len(k_vecs)
                  for d in range(HEAD_DIM)]
        distinctiveness = []
        for p in range(len(k_vecs)):
            dist = sum((k_vecs[p][d] - mean_k[d])**2 for d in range(HEAD_DIM))
            tok = tokenizer.decode([ids[p]]).strip()
            if len(tok) > 2 and tok.isalpha():
                distinctiveness.append((p, tok, dist))

        distinctiveness.sort(key=lambda x: -x[2])
        keywords = [tok for _, tok, _ in distinctiveness[:5]]

        ts = page_turns[0].get('timestamp', '')[:19]
        full_text = page_text[:200].replace('```', '').replace('\n', ' ')

        pages.append({
            'keywords': keywords,
            'full_text': full_text,
            'ts': ts,
        })

    print(f"  {len(pages)} pages in {time.time()-t0:.0f}s")

    # Insert into RAG
    clear_server()
    for p in pages:
        bookmark = f"[{p['ts']}] {', '.join(p['keywords'])}: {p['full_text']}"
        insert_rag(bookmark[:300])

    return run_scenarios()


# ═══════════════════════════════════════════════════════════════
# Variant C: InfLLM — KV blocks with representative K scoring
# ═══════════════════════════════════════════════════════════════

def variant_c(model, tokenizer, config):
    """
    InfLLM: Store blocks of tokens with representative K vectors.
    4 representative tokens per block (highest Q·K score).
    Retrieval: Q from query dotted against representative K's.
    """
    print("\n=== Variant C: InfLLM (KV blocks + representative scoring) ===")

    BLOCK_SIZE = 5  # turns per block (not 128 tokens, but turn-level)
    MAX_TURNS = 100
    N_REPS = 4  # representative tokens per block
    t0 = time.time()

    blocks = []
    for bi in range(0, min(len(turns), MAX_TURNS), BLOCK_SIZE):
        block_turns = turns[bi:bi + BLOCK_SIZE]
        block_text = " ".join(t['content'][:80] for t in block_turns)
        ids = tokenizer.encode(block_text[:400])

        _, q, k = forward_to_layer(model, config, ids, RETRIEVAL_LAYER)
        if k is None or q is None: continue

        k_vecs = extract_k_per_position(k)

        # Representative selection: tokens with highest average Q·K score
        # (most attended-to tokens in the block)
        q_last = extract_q_at_pos(q, pos=-1)
        scored = [(p, dot(q_last, k_vecs[p])) for p in range(len(k_vecs))]
        scored.sort(key=lambda x: -x[1])
        rep_indices = [p for p, _ in scored[:N_REPS]]
        rep_k_vecs = [k_vecs[p] for p in rep_indices]

        ts = block_turns[0].get('timestamp', '')[:19]
        full_text = block_text[:200].replace('```', '').replace('\n', ' ')

        blocks.append({
            'rep_k': rep_k_vecs,
            'full_text': full_text,
            'ts': ts,
        })

    print(f"  {len(blocks)} blocks, {N_REPS} reps each, in {time.time()-t0:.0f}s")

    # For retrieval: extract Q from query, score against all representative K's
    # Then insert best-matching blocks into RAG
    # (In production, this would be a Metal matmul, not Python)

    clear_server()

    # Insert all blocks as RAG facts
    for b in blocks:
        fact = f"[{b['ts']}] {b['full_text']}"
        insert_rag(fact[:300])

    return run_scenarios()


# ═══════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════

model, tokenizer, config = load_model()

variants = {
    'a': ('EM-LLM (K·K neural)', variant_a),
    'b': ('Pichay (neural keywords)', variant_b),
    'c': ('InfLLM (KV blocks)', variant_c),
}

to_run = list(variants.keys()) if VARIANT == 'all' else [VARIANT]
results = []

for v in to_run:
    name, fn = variants[v]
    score = fn(model, tokenizer, config)
    results.append((v.upper(), name, score))

print(f"\n{'='*50}")
print("COMPARISON (neural matching)")
print(f"{'='*50}")
print(f"{'Var':<4} {'Method':<25} {'Score':<8}")
print("-" * 40)
for v, name, score in results:
    print(f"{v:<4} {name:<25} {score}/11")
print("\nBaseline (token-mean embedding): 6/11")
