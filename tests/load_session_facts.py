#!/usr/bin/env python3
"""Load session transcript into RAG store — focused fact extraction."""
import json, subprocess, sys, re

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
TRANSCRIPT = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/session_transcript.json"

with open(TRANSCRIPT) as f:
    data = json.load(f)

KEYWORDS = [
    'tok/s', 'port', 'layer', 'metal', 'gemma', 'ratatui',
    'knn', 'rag', 'tanh', 'gpu', 'cpu', 'q4_k', 'q6_k',
    'decode', 'prefill', 'larql', 'vindex', 'speculative',
    'clone', 'server', 'tui', 'override', 'cosine',
    'temperature', 'threshold', 'residual', 'ffn',
    'batch', 'insert', 'walk', 'pipeline', 'shader',
    'model', 'inference', 'token', 'attention', 'head',
    'feature', 'weight', 'quantiz', 'matvec', 'kernel',
]

SKIP = ['let me', "i'll", "here's", 'looking at', "let's",
        'want me to', 'shall i', 'ready to', 'i can', 'i need']

inserted = 0
for turn in data['turns']:
    if turn['role'] != 'assistant':
        continue
    ts = turn.get('timestamp', '')
    content = turn['content']
    if len(content) < 30:
        continue

    sentences = re.split(r'(?<=[.!?\n])\s+', content)
    for sent in sentences:
        sent = sent.strip().replace('```', '')
        if len(sent) < 15 or len(sent) > 150:
            continue

        lower = sent.lower()
        has_signal = (
            bool(re.search(r'\d+', sent)) or
            any(kw in lower for kw in KEYWORDS)
        )
        if not has_signal:
            continue
        if any(skip in lower for skip in SKIP):
            continue

        fact_text = f'[{ts}] {sent}'
        r = subprocess.run(
            ['curl', '-s', '--max-time', '2', f'{SERVER}/v1/rag/insert',
             '-H', 'Content-Type: application/json',
             '-d', json.dumps({'fact': fact_text, 'category': 'session'})],
            capture_output=True, text=True
        )
        if '"ok"' in r.stdout:
            inserted += 1

print(inserted)
