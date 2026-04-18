#!/usr/bin/env python3
"""Load session transcript as keyword bookmarks (Pichay approach).

Each conversation turn → extract 3-5 keywords → store as bookmark.
BM25 matches keywords, retrieves the full turn text.
No labeling — keywords extracted automatically via TF-IDF-like scoring.

Usage: python3 tests/load_session_bookmarks.py [server] [transcript]
"""
import json, subprocess, sys, re, math
from collections import Counter

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
TRANSCRIPT = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/session_transcript.json"

# Stop words — common English words that don't carry meaning
STOP = set("the a an is are was were be been being have has had do does did will would "
    "shall should can could may might must need to of in on at by for with from as into "
    "through during before after above below between out off over under again further then "
    "once here there when where why how all each every both few more most other some such "
    "no nor not only own same so than too very just don t s d ll re ve m it its he she they "
    "we you i me him her us them my your his our their what which who whom this that these "
    "those am let get got also but and or if".split())

with open(TRANSCRIPT) as f:
    data = json.load(f)

# Step 1: Collect all assistant turns with their text
turns = []
for turn in data['turns']:
    if turn['role'] != 'assistant':
        continue
    text = turn['content']
    if len(text) < 30:
        continue
    ts = turn.get('timestamp', '')[:19]
    turns.append({'ts': ts, 'text': text})

print(f"Processing {len(turns)} assistant turns...")

# Step 2: Build document frequencies (how many turns contain each word)
doc_freq = Counter()
turn_words = []
for turn in turns:
    words = set(re.findall(r'[a-zA-Z0-9_]+', turn['text'].lower()))
    words -= STOP
    words = {w for w in words if len(w) > 2}
    turn_words.append(words)
    for w in words:
        doc_freq[w] += 1

n_docs = len(turns)

# Step 3: For each turn, extract top-5 keywords by TF-IDF
inserted = 0
for i, (turn, words) in enumerate(zip(turns, turn_words)):
    if not words:
        continue

    # Extract keywords: entities + numbers + technical terms
    # Priority: numbers > proper nouns > technical terms > rare words
    text = turn['text']
    text_lower = text.lower()

    # Extract numbers with context (e.g., "41 tok/s", "port 3000")
    numbers = re.findall(r'[\d]+(?:\.\d+)?(?:\s*(?:tok/s|MB|GB|ms|KB|s/tok|tok|fps|hz))?', text)
    numbers = [n.strip() for n in numbers if len(n) > 0]

    # Extract proper nouns / technical terms (CamelCase, all-caps, known terms)
    KNOWN = {'gemma', 'metal', 'ratatui', 'larql', 'vindex', 'knn', 'rag',
             'gpu', 'cpu', 'tui', 'sse', 'ffn', 'q4_k', 'q6_k', 'rope',
             'tanh', 'matvec', 'safetensors', 'mlx', 'axum', 'tokio',
             'miguel', 'rome', 'canberra', 'australia', 'clone', 'server',
             'port', 'decode', 'prefill', 'speculative', 'cosine', 'residual',
             'attention', 'embed', 'injection', 'override', 'threshold'}
    found_known = [w for w in words if w in KNOWN]

    # TF-IDF for remaining words
    scored = []
    for w in words:
        if w in KNOWN:
            continue  # already captured
        tf = text_lower.count(w)
        df = doc_freq[w]
        idf = math.log(n_docs / (1 + df))
        scored.append((w, tf * idf))
    scored.sort(key=lambda x: -x[1])
    tfidf_words = [w for w, _ in scored[:3]]

    # Combine: known terms first, then numbers, then TF-IDF
    keywords = []
    for w in found_known[:3]:
        if w not in keywords: keywords.append(w)
    for n in numbers[:2]:
        n_clean = n.split()[0]  # just the number
        if n_clean not in keywords: keywords.append(n_clean)
    for w in tfidf_words:
        if len(keywords) >= 5: break
        if w not in keywords: keywords.append(w)

    if not keywords:
        continue

    # Bookmark: keywords for matching
    # Full text: first 200 chars for injection
    bookmark = f"[{turn['ts']}] {', '.join(keywords)}"
    full_text = turn['text'][:200].replace('```', '').replace('\n', ' ').strip()
    fact = f"{bookmark}: {full_text}"

    r = subprocess.run(
        ['curl', '-s', '--max-time', '2', f'{SERVER}/v1/rag/insert',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'fact': fact, 'category': 'bookmark'})],
        capture_output=True, text=True
    )
    if '"ok"' in r.stdout:
        inserted += 1

    if (i + 1) % 200 == 0:
        print(f"  {i+1}/{len(turns)} processed, {inserted} inserted")

print(f"Done: {inserted} bookmarks from {len(turns)} turns")

# Show sample bookmarks
print("\nSample bookmarks:")
for i, (turn, words) in enumerate(zip(turns[:5], turn_words[:5])):
    text_lower = turn['text'].lower()
    scored = []
    for w in words:
        tf = text_lower.count(w)
        df = doc_freq[w]
        idf = math.log(n_docs / (1 + df))
        boost = 2.0 if re.search(r'\d', w) else 1.0
        scored.append((w, tf * idf * boost))
    scored.sort(key=lambda x: -x[1])
    kw = [w for w, _ in scored[:5]]
    print(f"  [{turn['ts']}] {', '.join(kw)}")
