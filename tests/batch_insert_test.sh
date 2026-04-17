#!/bin/bash
# Batch INSERT test harness — deterministic scenarios
#
# For each scenario:
#   1. Start fresh (reset KV cache + KNN store)
#   2. Batch-INSERT all files from the scenario folder
#   3. Query each expected question
#   4. Check if the answer contains the expected string
#
# Usage: ./tests/batch_insert_test.sh
# Requires: bench_interactive built with --features metal

set -e

BIN="./target/release/examples/bench_interactive"
MODEL="/Users/miguel_lemos/Desktop/gemma-3-4b-it"
VINDEX="/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex"
FIXTURES="./tests/fixtures"

if [ ! -f "$BIN" ]; then
    echo "ERROR: bench_interactive not found. Build first:"
    echo "  cargo build --release --features metal -p larql-inference --example bench_interactive"
    exit 1
fi

GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass=0
fail=0
total=0

run_scenario() {
    local name="$1"
    local folder="$2"
    shift 2
    # remaining args are "question|expected" pairs

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Scenario: $name"
    echo " Folder: $folder"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

    # Build the command sequence: batch-insert folder, then ask each question
    local cmds=""
    cmds+="batch-insert $folder\n"
    for pair in "$@"; do
        local question="${pair%%|*}"
        cmds+="chat $question\n"
    done
    cmds+="quit\n"

    # Run bench_interactive with the commands
    local output
    output=$(printf "$cmds" | timeout 300 "$BIN" \
        --model "$MODEL" --vindex "$VINDEX" \
        --walk-only --no-warmup 2>/dev/null || true)

    # Check each expected answer
    for pair in "$@"; do
        local question="${pair%%|*}"
        local expected="${pair##*|}"
        total=$((total + 1))

        if echo "$output" | grep -qi "$expected"; then
            echo -e "  ${GREEN}✓${NC} \"$question\" → contains \"$expected\""
            pass=$((pass + 1))
        else
            echo -e "  ${RED}✗${NC} \"$question\" → expected \"$expected\" NOT FOUND"
            # Show what was actually generated
            echo "    output snippet: $(echo "$output" | grep -i "${question:0:20}" -A2 | head -3)"
            fail=$((fail + 1))
        fi
    done
}

echo "╔══════════════════════════════════════════════════╗"
echo "║  larql batch INSERT test suite                    ║"
echo "╚══════════════════════════════════════════════════╝"

# Scenario 1: Project docs
run_scenario "Project Documentation" "$FIXTURES/scenario_1_docs" \
    "What port does the API Gateway run on?|8080" \
    "What database does the platform use?|PostgreSQL" \
    "What is the staging URL?|staging.gaucho.dev"

# Scenario 2: Python codebase
run_scenario "Python Codebase" "$FIXTURES/scenario_2_codebase" \
    "What algorithm does the JWT token use?|RS256" \
    "What is the production URL?|app.gaucho.io"

# Scenario 3: Nested skills
run_scenario "Nested Skills KB" "$FIXTURES/scenario_3_skills" \
    "Who is the Tech Lead?|Sarah" \
    "What is the SEV1 response time?|5 minutes"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo -e " Results: ${GREEN}$pass passed${NC}, ${RED}$fail failed${NC}, $total total"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

exit $fail
