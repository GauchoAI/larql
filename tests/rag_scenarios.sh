#!/bin/bash
# RAG Scenario Test Runner
# 1. Loads session transcript into RAG store
# 2. Runs scenario queries and checks expected answers
# Assumes server is ALREADY running on localhost:3000.
#
# Usage: ./tests/rag_scenarios.sh [http://host:port]

set -euo pipefail
SERVER="${1:-http://localhost:3000}"
TRANSCRIPT="tests/fixtures/session_transcript.json"
SCENARIOS="tests/fixtures/rag_scenarios.json"
PASS=0; FAIL=0
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

curl -s "$SERVER/v1/health" | grep -q ok || { echo "Server not reachable"; exit 1; }

# Step 1: Load transcript into RAG
echo "=== Loading transcript into RAG ==="
LOADED=$(python3 -c "
import json, subprocess, sys

with open('$TRANSCRIPT') as f:
    data = json.load(f)

inserted = 0
for turn in data['turns']:
    ts = turn.get('timestamp', '')
    role = turn['role']
    content = turn['content']
    # Truncate safely
    fact = content[:500] if len(content) <= 500 else content[:500]
    fact_text = f'[{ts}] [{role}] {fact}'

    r = subprocess.run(
        ['curl', '-s', '--max-time', '2', '$SERVER/v1/rag/insert',
         '-H', 'Content-Type: application/json',
         '-d', json.dumps({'fact': fact_text, 'category': 'session'})],
        capture_output=True, text=True
    )
    if '\"ok\"' in r.stdout:
        inserted += 1

print(inserted)
")
echo "Loaded $LOADED facts"
echo ""

# Step 2: Run scenarios
echo "=== RAG Scenarios ==="
python3 << 'PYEOF'
import json, subprocess, sys

with open("tests/fixtures/rag_scenarios.json") as f:
    data = json.load(f)

server = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
passed = 0
failed = 0

for sc in data["scenarios"]:
    query = sc["query"]
    expects = sc.get("expect_contains", [])
    not_expects = sc.get("expect_not_contains", [])
    desc = sc["description"]

    # Query the model
    r = subprocess.run(
        ["curl", "-s", "--max-time", "15", f"{server}/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"messages": [{"role": "user", "content": query}]})],
        capture_output=True, text=True
    )

    # Parse SSE response
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
    reasons = []

    for exp in expects:
        if exp.lower() not in output_lower:
            ok = False
            reasons.append(f"missing '{exp}'")

    for nexp in not_expects:
        if nexp.lower() in output_lower:
            ok = False
            reasons.append(f"unwanted '{nexp}'")

    if ok:
        print(f"  \033[0;32mPASS\033[0m  {desc}")
        passed += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {desc}: {', '.join(reasons)}")
        print(f"         got: {output[:100]}")
        failed += 1

print(f"\n=== {passed} pass, {failed} fail ===")
sys.exit(0 if failed == 0 else 1)
PYEOF
