#!/usr/bin/env python3
"""
Test: put ALL facts in context (as tokens), let model attend naturally.
This is the upper bound — if this doesn't work, nothing will.
If it works, KV replay is just an optimization for speed.

Uses MLX bf16 model to generate answers with full fact context.
"""
import sys, json, time
sys.path.insert(0, "/Users/miguel_lemos/Desktop/chuk-lazurus/src")

import mlx.core as mx
import mlx.nn as nn

TRANSCRIPT = "/Users/miguel_lemos/Desktop/llm-as-a-database/larql/tests/fixtures/session_transcript.json"
SCENARIOS = "/Users/miguel_lemos/Desktop/llm-as-a-database/larql/tests/fixtures/rag_scenarios.json"

# Load model
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
print(f"Loaded: {config.num_hidden_layers} layers")

# Load transcript — extract key facts (first 200 chars of each assistant turn)
with open(TRANSCRIPT) as f:
    transcript = json.load(f)
with open(SCENARIOS) as f:
    scenarios = json.load(f)

turns = [t for t in transcript['turns'] if t['role'] == 'assistant' and len(t['content']) > 30]

# Build fact context — concatenate turns until we hit ~4000 tokens
facts_text = ""
included = 0
for turn in turns:
    text = turn['content'][:150].replace('```', '').replace('\n', ' ').strip()
    ts = turn.get('timestamp', '')[:19]
    candidate = f"[{ts}] {text}\n"
    test_ids = tokenizer.encode(facts_text + candidate)
    if len(test_ids) > 4000:
        break
    facts_text += candidate
    included += 1

fact_tokens = len(tokenizer.encode(facts_text))
print(f"\nFact context: {included} turns, {fact_tokens} tokens")

# Generate function
def generate(prompt, max_tokens=100):
    ids = tokenizer.encode(prompt)
    if len(ids) > 8000:
        ids = ids[-8000:]  # truncate from start if too long

    layers = list(model.model.layers) if hasattr(model, 'model') else list(model.layers)
    embed = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
    lm_head = model.lm_head if hasattr(model, 'lm_head') else None
    scale = getattr(config, "embedding_scale", None)

    input_ids = mx.array(ids)[None, :]
    h = embed(input_ids)
    if scale: h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    for lyr in layers:
        try: out = lyr(h, mask=mask)
        except TypeError: out = lyr(h)
        h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

    if norm: h = norm(h)
    if lm_head:
        logits = lm_head(h)
        logits = logits.logits if hasattr(logits, 'logits') else logits
    else:
        logits = h @ embed.weight.T

    mx.eval(logits)

    # Greedy decode first token
    generated = []
    for _ in range(max_tokens):
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        tok = tokenizer.decode([next_id])
        if '<end_of_turn>' in tok or '<eos>' in tok or next_id <= 1:
            break
        generated.append(tok)

        # Simple: just return first batch of tokens (no KV cache for speed)
        # For a proper test, we'd need autoregressive generation
        # But greedy first-token + continuation is enough to check
        break

    return ''.join(generated)

# Actually, MLX autoregressive generation is complex. Let me just
# check what the model's TOP PREDICTION is when given facts + query.
# If the first token is correct, the approach works.

def get_top_prediction(prompt, top_k=5):
    """Get top-k token predictions for the next token after prompt."""
    ids = tokenizer.encode(prompt)
    if len(ids) > 8000:
        ids = ids[-8000:]

    layers = list(model.model.layers) if hasattr(model, 'model') else list(model.layers)
    embed = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    norm = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
    lm_head = model.lm_head if hasattr(model, 'lm_head') else None
    scale = getattr(config, "embedding_scale", None)

    input_ids = mx.array(ids)[None, :]
    h = embed(input_ids)
    if scale: h = h * scale

    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)

    for lyr in layers:
        try: out = lyr(h, mask=mask)
        except TypeError: out = lyr(h)
        h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)

    if norm: h = norm(h)
    if lm_head:
        logits = lm_head(h)
        logits = logits.logits if hasattr(logits, 'logits') else logits
    else:
        logits = h @ embed.weight.T

    # Top-k tokens
    probs = mx.softmax(logits[0, -1, :], axis=-1)
    mx.eval(probs)

    probs_list = probs.tolist()
    indexed = sorted(enumerate(probs_list), key=lambda x: -x[1])[:top_k]

    results = []
    for tid, prob in indexed:
        tok = tokenizer.decode([tid]).strip()
        results.append((tok, prob))
    return results

# Simple autoregressive generation using MLX
def generate_text(prompt, max_tokens=150):
    ids = tokenizer.encode(prompt)
    if len(ids) > 7500:
        ids = ids[-7500:]

    layers = list(model.model.layers) if hasattr(model, 'model') else list(model.layers)
    embed_fn = model.model.embed_tokens if hasattr(model, 'model') else model.embed_tokens
    norm_fn = model.model.norm if hasattr(model, 'model') and hasattr(model.model, 'norm') else None
    lm_head_fn = model.lm_head if hasattr(model, 'lm_head') else None
    scale = getattr(config, "embedding_scale", None)

    # Prefill
    input_ids = mx.array(ids)[None, :]
    h = embed_fn(input_ids)
    if scale: h = h * scale
    seq_len = input_ids.shape[1]
    mask = nn.MultiHeadAttention.create_additive_causal_mask(seq_len).astype(h.dtype)
    for lyr in layers:
        try: out = lyr(h, mask=mask)
        except TypeError: out = lyr(h)
        h = out.hidden_states if hasattr(out, "hidden_states") else (out[0] if isinstance(out, tuple) else out)
    if norm_fn: h = norm_fn(h)
    if lm_head_fn:
        logits = lm_head_fn(h)
        logits = logits.logits if hasattr(logits, 'logits') else logits
    else:
        logits = h @ embed_fn.weight.T
    mx.eval(logits)

    # Greedy autoregressive (simplified — no KV cache, just top-1 per step)
    generated_ids = []
    for _ in range(max_tokens):
        next_id = int(mx.argmax(logits[0, -1, :]).item())
        tok_str = tokenizer.decode([next_id])
        if '<end_of_turn>' in tok_str or '<eos>' in tok_str or next_id <= 1:
            break
        generated_ids.append(next_id)
        # For speed: just use the prefill logits to predict first few tokens
        # (approximation — real generation needs KV cache)
        # After 1 token we break since autoregressive without KV cache is O(n²)
        break

    # Return first token + greedy continuation hint
    if generated_ids:
        first_tok = tokenizer.decode(generated_ids)
        return first_tok
    return ""

# Use the server for proper generation — just send the full context as the prompt
import subprocess

def generate_via_server(facts, query, max_tokens=150):
    """Send facts + query to our server for proper generation with KV cache."""
    # Build the prompt with facts as system context
    messages = [
        {"role": "system", "content": f"You are a helpful assistant. Answer based on this conversation history:\n{facts}"},
        {"role": "user", "content": query}
    ]
    r = subprocess.run(
        ['curl', '-s', '--max-time', '30', 'http://localhost:3000/v1/chat/completions',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'messages': messages, 'max_tokens': max_tokens})],
        capture_output=True, text=True
    )
    output = ""
    for line in r.stdout.split("\n"):
        line = line.strip()
        if line.startswith("data:") and "content" in line:
            try:
                output += json.loads(line[5:])["choices"][0]["delta"].get("content", "")
            except: pass
    return output

# Make sure server is running
import subprocess, time
subprocess.run(['pkill', '-f', 'larql-server'], capture_output=True)
subprocess.Popen(
    ['target/release/larql-server',
     '/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex'],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    cwd='/Users/miguel_lemos/Desktop/llm-as-a-database/larql'
)
for _ in range(120):
    r = subprocess.run(['curl', '-s', 'http://localhost:3000/v1/health'], capture_output=True, text=True)
    if '"ok"' in r.stdout: break
    time.sleep(1)

print("\n=== Full Context Test ===")
print(f"Facts: {included} turns, ~{fact_tokens} tokens IN the prompt")
print("Model attends naturally — no retrieval needed.\n")

passed = 0
failed = 0
for sc in scenarios['scenarios']:
    query = sc['query']
    expects = sc.get('expect_contains', [])
    not_expects = sc.get('expect_not_contains', [])
    desc = sc['description']

    t0 = time.time()
    output = generate_via_server(facts_text, query)
    elapsed = time.time() - t0

    output_lower = output.lower()
    ok = all(e.lower() in output_lower for e in expects) and \
         not any(n.lower() in output_lower for n in not_expects)

    if ok:
        passed += 1
        print(f"  \033[32mPASS\033[0m  {desc} ({elapsed:.1f}s)")
    else:
        failed += 1
        print(f"  \033[31mFAIL\033[0m  {desc} ({elapsed:.1f}s)")
        print(f"         {output[:120]}")

print(f"\n=== {passed}/{passed+failed} with full context (upper bound) ===")
print(f"Baseline (RAG): 6/11")
print(f"\nIf this beats 6/11, KV replay is the path — just an optimization for speed.")
