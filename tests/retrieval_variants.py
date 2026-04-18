#!/usr/bin/env python3
"""
Three retrieval variants — fast iteration against scenario tests.

Variant A: EM-LLM — surprise-based episode segmentation + episode retrieval
Variant B: Pichay — page table + page-ID recall (not keyword search)
Variant C: InfLLM — attention-based memory unit selection

All share the same interface:
  1. Load transcript → process into retrievable units
  2. On query → retrieve relevant units
  3. Inject as RAG context → server generates answer

Usage:
  python3 tests/retrieval_variants.py [variant] [server]
  variant: a (em-llm), b (pichay), c (infllm), all
"""
import json, subprocess, sys, re, math, time
from collections import Counter

SERVER = sys.argv[2] if len(sys.argv) > 2 else "http://localhost:3000"
VARIANT = sys.argv[1] if len(sys.argv) > 1 else "all"
TRANSCRIPT = "tests/fixtures/session_transcript.json"
SCENARIOS = "tests/fixtures/rag_scenarios.json"

with open(TRANSCRIPT) as f:
    transcript = json.load(f)
with open(SCENARIOS) as f:
    scenarios = json.load(f)

# Extract assistant turns only
turns = [t for t in transcript['turns'] if t['role'] == 'assistant' and len(t['content']) > 30]
print(f"Loaded {len(turns)} assistant turns")


def insert_fact(fact, category="session"):
    r = subprocess.run(
        ['curl', '-s', '--max-time', '2', f'{SERVER}/v1/rag/insert',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'fact': fact, 'category': category})],
        capture_output=True, text=True
    )
    return '"ok"' in r.stdout


def run_scenarios():
    """Run scenario tests, return (passed, failed, results)."""
    passed = 0
    failed = 0
    results = []
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
                    d = json.loads(line[5:])
                    output += d["choices"][0]["delta"].get("content", "")
                except:
                    pass

        output_lower = output.lower()
        ok = True
        for exp in expects:
            if exp.lower() not in output_lower:
                ok = False
        for nexp in not_expects:
            if nexp.lower() in output_lower:
                ok = False

        if ok:
            passed += 1
        else:
            failed += 1
        results.append((desc, ok, output[:80]))

    return passed, failed, results


# ═══════════════════════════════════════════════════════════════
# Variant A: EM-LLM — Surprise-based episodic segmentation
# ═══════════════════════════════════════════════════════════════

def variant_a_emllm():
    """
    EM-LLM approach: segment conversation by surprise boundaries.

    1. Compute "surprise" for each turn = how different it is from previous
       (measured by word overlap — proxy for hidden state surprise)
    2. Place boundaries where surprise is high
    3. Each episode = contiguous segment of turns
    4. Store episodes as single facts (coherent context, not isolated sentences)
    5. Retrieve by two-stage: embedding similarity + temporal contiguity
    """
    print("\n=== Variant A: EM-LLM (surprise segmentation) ===")
    t0 = time.time()

    # Step 1: Compute surprise for each turn
    prev_words = set()
    surprises = []
    for turn in turns:
        words = set(re.findall(r'[a-zA-Z]{3,}', turn['content'].lower()))
        if prev_words:
            overlap = len(words & prev_words) / max(len(words | prev_words), 1)
            surprise = 1.0 - overlap  # high surprise = low overlap
        else:
            surprise = 1.0  # first turn is always surprising
        surprises.append(surprise)
        prev_words = words

    # Step 2: Segment by surprise > threshold (adaptive: mean + 0.5*std)
    mean_s = sum(surprises) / len(surprises)
    std_s = (sum((s - mean_s)**2 for s in surprises) / len(surprises)) ** 0.5
    threshold = mean_s + 0.3 * std_s

    episodes = []
    current_episode = []
    for i, (turn, surprise) in enumerate(zip(turns, surprises)):
        if surprise > threshold and current_episode:
            episodes.append(current_episode)
            current_episode = []
        current_episode.append(turn)
    if current_episode:
        episodes.append(current_episode)

    print(f"  {len(episodes)} episodes from {len(turns)} turns (threshold={threshold:.2f})")

    # Step 3: Store each episode as a coherent fact
    # Use first and last sentence of each episode as the summary
    inserted = 0
    for ep in episodes:
        first_text = ep[0]['content'][:100].replace('```', '').replace('\n', ' ')
        last_text = ep[-1]['content'][:100].replace('```', '').replace('\n', ' ')
        ts = ep[0].get('timestamp', '')[:19]

        # Also extract key words from all turns in the episode
        all_words = set()
        for t in ep:
            all_words.update(re.findall(r'[a-zA-Z0-9_]{3,}', t['content'].lower()))

        # Create episode summary
        fact = f"[{ts}, {len(ep)} turns] {first_text}"
        if len(ep) > 1:
            fact += f" ... {last_text}"

        if insert_fact(fact[:300], "episode"):
            inserted += 1

    elapsed = time.time() - t0
    print(f"  Inserted {inserted} episodes in {elapsed:.0f}s")
    return inserted


# ═══════════════════════════════════════════════════════════════
# Variant B: Pichay — Page table with page-ID recall
# ═══════════════════════════════════════════════════════════════

def variant_b_pichay():
    """
    Pichay approach: page table with minimal bookmarks.

    1. Chunk conversation into fixed-size pages (~10 turns each)
    2. Create a page table: [pN: first_sentence, keywords]
    3. Store BOTH the page table entry (for retrieval) AND the full page
    4. When retrieved, inject the full page content (not just keywords)

    Key insight from paper: minimal bookmarks (8-24 tokens) work better
    than verbose ones. The bookmark is the TABLE OF CONTENTS entry.
    """
    print("\n=== Variant B: Pichay (page table) ===")
    t0 = time.time()

    PAGE_SIZE = 8  # turns per page

    pages = []
    for i in range(0, len(turns), PAGE_SIZE):
        page_turns = turns[i:i + PAGE_SIZE]
        pages.append(page_turns)

    print(f"  {len(pages)} pages of ~{PAGE_SIZE} turns each")

    inserted = 0
    for pi, page in enumerate(pages):
        # Page content: concatenate all turns (truncated)
        content_parts = []
        for t in page:
            text = t['content'][:150].replace('```', '').replace('\n', ' ').strip()
            if text:
                content_parts.append(text)
        full_content = " | ".join(content_parts)

        # Bookmark: first turn's beginning + timestamp
        ts = page[0].get('timestamp', '')[:19]
        first = page[0]['content'][:60].replace('```', '').replace('\n', ' ')

        # Store the full page content (truncated to 300 chars for embedding)
        fact = f"[p{pi} {ts}] {full_content}"
        if insert_fact(fact[:300], "page"):
            inserted += 1

    elapsed = time.time() - t0
    print(f"  Inserted {inserted} pages in {elapsed:.0f}s")
    return inserted


# ═══════════════════════════════════════════════════════════════
# Variant C: InfLLM — Per-turn memory units (no segmentation)
# ═══════════════════════════════════════════════════════════════

def variant_c_infllm():
    """
    InfLLM approach: store every turn as a memory unit.
    No segmentation, no keywords. Each turn is its own unit.

    The key difference: store the FULL turn content (not just sentences
    or keywords). The embedding captures the full semantic meaning.

    Retrieval: pure embedding similarity. The bet is that with enough
    turns, some will naturally match the query well enough.

    Enhancement: also store the 2 surrounding turns for temporal context
    (InfLLM retrieves contiguous blocks, not isolated units).
    """
    print("\n=== Variant C: InfLLM (per-turn memory units) ===")
    t0 = time.time()

    inserted = 0
    for i, turn in enumerate(turns):
        ts = turn.get('timestamp', '')[:19]
        text = turn['content'][:250].replace('```', '').replace('\n', ' ').strip()
        if not text:
            continue

        # Store with temporal context: include prev/next turn snippets
        context_parts = [f"[{ts}]"]
        if i > 0:
            prev = turns[i-1]['content'][:50].replace('```', '').replace('\n', ' ')
            context_parts.append(f"(prev: {prev})")
        context_parts.append(text)
        if i < len(turns) - 1:
            nxt = turns[i+1]['content'][:50].replace('```', '').replace('\n', ' ')
            context_parts.append(f"(next: {nxt})")

        fact = " ".join(context_parts)
        if insert_fact(fact[:300], "unit"):
            inserted += 1

        if (i + 1) % 500 == 0:
            print(f"    {i+1}/{len(turns)} inserted")

    elapsed = time.time() - t0
    print(f"  Inserted {inserted} units in {elapsed:.0f}s")
    return inserted


# ═══════════════════════════════════════════════════════════════
# Main: run variants and compare
# ═══════════════════════════════════════════════════════════════

def clear_store():
    """Restart server to clear RAG store."""
    subprocess.run(['pkill', '-f', 'larql-server'], capture_output=True)
    subprocess.Popen(
        ['target/release/larql-server',
         '/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex'],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
    )
    for _ in range(120):
        r = subprocess.run(
            ['curl', '-s', f'{SERVER}/v1/health'],
            capture_output=True, text=True
        )
        if '"ok"' in r.stdout:
            return True
        time.sleep(1)
    return False


variants = {
    'a': ('EM-LLM', variant_a_emllm),
    'b': ('Pichay', variant_b_pichay),
    'c': ('InfLLM', variant_c_infllm),
}

to_run = list(variants.keys()) if VARIANT == 'all' else [VARIANT]

results_table = []
for v in to_run:
    name, fn = variants[v]
    print(f"\n{'='*60}")
    print(f"Running Variant {v.upper()}: {name}")
    print(f"{'='*60}")

    # Clear store
    print("Restarting server (clean store)...")
    if not clear_store():
        print("  FAILED to restart server")
        continue

    # Load facts
    n = fn()

    # Run scenarios
    print(f"\n  Running {len(scenarios['scenarios'])} scenarios...")
    passed, failed, details = run_scenarios()

    print(f"\n  Results: {passed}/{passed+failed}")
    for desc, ok, output in details:
        status = "\033[0;32mPASS\033[0m" if ok else "\033[0;31mFAIL\033[0m"
        print(f"    {status}  {desc}")
        if not ok:
            print(f"           {output}")

    results_table.append((v.upper(), name, n, passed, passed + failed))

# Summary
print(f"\n{'='*60}")
print("COMPARISON")
print(f"{'='*60}")
print(f"{'Var':<4} {'Method':<15} {'Facts':<8} {'Score':<8}")
print("-" * 40)
for v, name, n, passed, total in results_table:
    print(f"{v:<4} {name:<15} {n:<8} {passed}/{total}")
