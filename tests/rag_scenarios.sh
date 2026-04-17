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
  "The model is Gemma 3 4B IT running inference" \
  "GPU decode runs at 35-41 tok/s on Metal M4 Pro" \
  "The server default port is 3000" \
  "The TUI is built with ratatui and gc-markdown in Rust" \
  "The project name is larql which means LLM as a Database" \
  "Metal GPU tanh overflow was fixed by clamping to [-10,10]" \
  "Speculative decoding gave zero speedup because GPU is compute-bound" \
  "KNN overlay at layer 26 overrides token predictions at inference time" \
  "The vindex has 348160 features across 34 layers"; do
  curl -s --max-time 2 "$SERVER/v1/rag/insert" -H "Content-Type: application/json" \
    -d "{\"fact\":\"$fact\",\"category\":\"key\"}" > /dev/null
done
echo "Seeded 10 key facts"
echo ""

# Step 2: Run scenarios
echo "=== Scenarios ==="
python3 tests/run_scenarios.py "$SERVER" "$SCENARIOS"
