#!/usr/bin/env python3
"""Test copy head retrieval quality in bf16 via MLX.

Uses chuk-lazurus to load Gemma 3 4B in bf16 and extract Q/K at L29 H4.
This verifies whether the copy head discriminates facts when run in
full precision (not Q4_K quantized).

Usage: cd chuk-lazurus && uv run python ../larql/tests/test_copy_head_bf16.py
"""
import sys, time, math
sys.path.insert(0, "/Users/miguel_lemos/Desktop/chuk-lazurus/src")

import mlx.core as mx
import mlx.nn as nn

RETRIEVAL_LAYER = 29
QUERY_HEAD = 4
HEAD_DIM = 256

FACTS = [
    "The user's name is Miguel",
    "The model is Gemma 3 4B IT",
    "The server runs on port 3000",
    "The decode speed is 35 tokens per second",
    "The GPU is Apple Metal on M4 Pro",
]

QUERIES = [
    ("what is my name?", "Miguel"),
    ("what model are we using?", "Gemma"),
    ("what port is the server on?", "3000"),
    ("how fast is the decode speed?", "35"),
    ("what GPU are we using?", "Metal"),
]

def load_model():
    from chuk_lazarus.inference.loader import DType, HFLoader
    from chuk_lazarus.models_v2.families.registry import detect_model_family, get_family_info
    import json

    model_id = "mlx-community/gemma-3-4b-it-bf16"
    print(f"Loading {model_id}...")
    result = HFLoader.download(model_id)
    model_path = result.model_path

    with open(model_path / "config.json") as f:
        config_data = json.load(f)

    family_type = detect_model_family(config_data)
    family_info = get_family_info(family_type)
    config = family_info.config_class.from_hf_config(config_data)
    model = family_info.model_class(config)
    HFLoader.apply_weights_to_model(model, model_path, config, dtype=DType.BFLOAT16)
    tokenizer = HFLoader.load_tokenizer(model_path)

    print(f"  Layers: {config.num_hidden_layers}, Hidden: {config.hidden_size}")
    return model, tokenizer, config

def extract_qk(model, tokenizer, config, text, is_query=False, answer_token=None):
    """Extract K (for facts) or Q (for queries) at L29 H4 in bf16.
    For facts: extract K at the answer token position (not last).
    For queries: extract Q at the last position."""
    input_ids_list = tokenizer.encode(text)
    input_ids = mx.array(input_ids_list)[None, :]

    # Find answer token position
    answer_pos = -1  # default: last position
    if answer_token and not is_query:
        answer_ids = tokenizer.encode(f" {answer_token}")
        if answer_ids:
            target = answer_ids[0] if len(answer_ids) == 1 else answer_ids[-1]
            for i, tid in enumerate(input_ids_list):
                if tid == target:
                    answer_pos = i
                    break

    # Get layers
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        layers = list(model.model.layers)
        embed = model.model.embed_tokens
    else:
        layers = list(model.layers)
        embed = model.embed_tokens

    # Embedding scale (Gemma uses sqrt(hidden_size))
    scale = getattr(config, "embedding_scale", None)
    h = embed(input_ids)
    if scale:
        h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    # Forward through layers 0..RETRIEVAL_LAYER
    for idx, lyr in enumerate(layers):
        if idx == RETRIEVAL_LAYER:
            # At L29: extract Q or K from attention
            # Access the self_attn module
            attn = lyr.self_attn if hasattr(lyr, 'self_attn') else lyr

            # Get the pre-norm hidden state
            if hasattr(lyr, 'input_layernorm'):
                h_norm = lyr.input_layernorm(h)
            else:
                h_norm = h

            # Q and K projections
            q = attn.q_proj(h_norm)  # (1, seq, q_dim)
            k = attn.k_proj(h_norm)  # (1, seq, kv_dim)

            mx.eval(q, k)

            # Extract at the right position
            if is_query:
                # Q at QUERY_HEAD (head 4), last position
                q_pos = q[0, -1, :]
                num_q_heads = q_pos.shape[0] // HEAD_DIM
                q_heads = q_pos.reshape(num_q_heads, HEAD_DIM)
                vec = q_heads[QUERY_HEAD].tolist()
            else:
                # K at KV_HEAD (head 2), at answer token position
                kv_head = QUERY_HEAD // (8 // 4)  # GQA mapping
                k_pos = k[0, answer_pos, :]
                num_kv_heads = k_pos.shape[0] // HEAD_DIM
                k_heads = k_pos.reshape(num_kv_heads, HEAD_DIM)
                vec = k_heads[kv_head].tolist()

            # L2 normalize
            norm = math.sqrt(sum(x*x for x in vec))
            if norm > 1e-12:
                vec = [x / norm for x in vec]
            return vec

        try:
            out = lyr(h, mask=mask)
        except TypeError:
            out = lyr(h)
        h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

    return None

def cosine(a, b):
    d = sum(x*y for x,y in zip(a,b))
    return d  # already L2-normalized

def main():
    model, tokenizer, config = load_model()

    # Answer tokens for each fact (the key entity)
    ANSWERS = ["Miguel", "Gemma", "3000", "35", "Metal"]

    print("\n=== Extracting K vectors at answer position (bf16) ===")
    k_vecs = []
    for fact, answer in zip(FACTS, ANSWERS):
        t0 = time.time()
        k = extract_qk(model, tokenizer, config, fact, is_query=False, answer_token=answer)
        elapsed = time.time() - t0
        print(f"  K: {fact[:40]:40s} @ '{answer}' ({elapsed:.1f}s)")
        k_vecs.append(k)

    print("\n=== Q·K scoring ===")
    print(f"{'Query':<35s} " + " ".join(f"{f[:8]:>8s}" for f in FACTS))
    print("-" * (35 + 9 * len(FACTS)))

    hits = 0
    for query, keyword in QUERIES:
        t0 = time.time()
        q = extract_qk(model, tokenizer, config, query, is_query=True)
        scores = [cosine(q, k) for k in k_vecs]
        elapsed = time.time() - t0

        best_idx = scores.index(max(scores))
        correct_idx = next(i for i, f in enumerate(FACTS) if keyword.lower() in f.lower())
        hit = "✓" if best_idx == correct_idx else "✗"
        if best_idx == correct_idx:
            hits += 1

        score_strs = []
        for i, s in enumerate(scores):
            marker = "←" if i == best_idx else " "
            score_strs.append(f"{s:+.3f}{marker}")
        print(f"{query:<35s} {' '.join(score_strs)} {hit} ({elapsed:.1f}s)")

    print(f"\n=== {hits}/{len(QUERIES)} correct ===")

if __name__ == "__main__":
    main()
