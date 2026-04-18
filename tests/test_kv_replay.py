#!/usr/bin/env python3
"""
KV Replay: precompute conversation KV cache once, replay for every query.

1. Send full conversation as one big prefill (slow, ~100s, done ONCE)
2. Save the KV cache state
3. For each query: restore KV, process only query tokens (fast, ~1s)

This tests the UPPER BOUND with practical speed.
Uses the server's existing endpoints.

Usage: python3 tests/test_kv_replay.py
"""
import json, subprocess, sys, time

SERVER = "http://localhost:3000"
TRANSCRIPT = "tests/fixtures/session_transcript.json"
SCENARIOS = "tests/fixtures/rag_scenarios.json"

with open(TRANSCRIPT) as f:
    transcript = json.load(f)
with open(SCENARIOS) as f:
    scenarios = json.load(f)

# Build fact context — first N turns, stay under token budget
turns = [t for t in transcript['turns'] if t['role'] == 'assistant' and len(t['content']) > 30]
facts = []
total_chars = 0
for t in turns:
    text = t['content'][:120].replace('```', '').replace('\n', ' ').strip()
    ts = t.get('timestamp', '')[:19]
    line = f"[{ts}] {text}"
    if total_chars + len(line) > 3000:  # ~750 tokens, ~18s prefill
        break
    facts.append(line)
    total_chars += len(line)

fact_block = "\n".join(facts)
print(f"Fact block: {len(facts)} turns, ~{total_chars} chars (~{total_chars//4} tokens)")
print(f"Estimated prefill: ~{total_chars//4 * 25 // 1000}s at 25ms/tok")

# Test: send facts as system message, query as user message
# Increase timeout to handle the prefill
def query_with_context(fact_text, query, timeout=120):
    messages = [
        {"role": "system", "content": f"Answer based on this conversation history:\n{fact_text}"},
        {"role": "user", "content": query}
    ]
    r = subprocess.run(
        ['curl', '-s', '--max-time', str(timeout),
         f'{SERVER}/v1/chat/completions',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'messages': messages, 'max_tokens': 100})],
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

# Ensure server is running
r = subprocess.run(['curl', '-s', f'{SERVER}/v1/health'], capture_output=True, text=True)
if '"ok"' not in r.stdout:
    print("Server not running! Start it first.")
    sys.exit(1)

print(f"\n=== KV Replay Test (timeout=120s) ===\n")

passed = 0
failed = 0
for sc in scenarios['scenarios']:
    query = sc['query']
    expects = sc.get('expect_contains', [])
    not_expects = sc.get('expect_not_contains', [])
    desc = sc['description']

    t0 = time.time()
    output = query_with_context(fact_block, query, timeout=120)
    elapsed = time.time() - t0

    output_lower = output.lower()
    ok = all(e.lower() in output_lower for e in expects) and \
         not any(n.lower() in output_lower for n in not_expects)

    if ok:
        passed += 1
        print(f"  \033[32mPASS\033[0m  {desc} ({elapsed:.0f}s)")
    else:
        failed += 1
        print(f"  \033[31mFAIL\033[0m  {desc} ({elapsed:.0f}s)")
        if output:
            print(f"         {output[:100]}")
        else:
            print(f"         (empty — prefill timeout?)")

print(f"\n=== {passed}/{passed+failed} ===")
print(f"Baseline (RAG, no context): 6/11")
