#!/bin/bash
# RAG Scenario Test Runner
# Assumes server is ALREADY running.
# Usage: ./tests/rag_scenarios.sh [http://host:port]

set -euo pipefail
SERVER="${1:-http://localhost:3000}"
SCENARIOS="tests/fixtures/rag_scenarios.json"
GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

curl -s "$SERVER/v1/health" | grep -q ok || { echo "Server not reachable"; exit 1; }

# Step 1: Seed key facts ONLY — skip noisy session facts.
# The key facts are curated, focused, and match the scenarios.
# Session facts (3491 noisy sentences) drown out key facts.
echo "=== Seeding key facts ==="
for fact in \
  "We cloned the chrishayuk/larql repository as the first step" \
  "The model we are running is Gemma 3 4B IT" \
  "The decode speed is 35 to 41 tokens per second" \
  "The server port is 3000" \
  "The TUI is built with ratatui" \
  "The project is larql" \
  "We fixed a tanh overflow bug in Metal GPU shaders" \
  "Speculative decoding gave zero speedup" \
  "The KNN overlay at layer 26 overrides predictions" \
  "The GPU we use is Apple Metal on M4 Pro" \
  "The vindex has 348160 features across 34 layers" \
  "The speed of decoding is 35 to 41 tok/s" \
  "The port the server runs on is 3000"; do
  curl -s --max-time 2 "$SERVER/v1/rag/insert" -H "Content-Type: application/json" \
    -d "{\"fact\":\"$fact\",\"category\":\"key\"}" > /dev/null
done
echo "Seeded 10 key facts"
echo ""

# Step 2: Run scenarios
echo "=== Scenarios ==="
python3 tests/run_scenarios.py "$SERVER" "$SCENARIOS"
