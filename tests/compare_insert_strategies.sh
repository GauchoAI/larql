#!/bin/bash
# Compare INSERT strategies for document Q&A
#
# Strategy A: RAG — batch-insert sections, then on query ask with context
# Strategy B: Question-form — INSERT using question prompts
#
# Both use the same scenario_1_docs fixture.

set -e

BIN="./target/release/examples/bench_interactive"
MODEL="/Users/miguel_lemos/Desktop/gemma-3-4b-it"
VINDEX="/Users/miguel_lemos/Desktop/llm-as-a-database/gemma3-4b.vindex"
FIXTURES="./tests/fixtures"

GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[36m'
NC='\033[0m'

echo "╔══════════════════════════════════════════════════╗"
echo "║  Compare INSERT strategies for document Q&A       ║"
echo "╚══════════════════════════════════════════════════╝"

# ── Strategy B: Question-form INSERTs ──
echo ""
echo -e "${CYAN}Strategy B: Question-form INSERTs${NC}"
echo "  INSERT using question prompts that match how users will query"
echo ""

CMDS_B=""
# Manually craft question-form INSERTs from scenario_1_docs
CMDS_B+="insert \"API Gateway port\" is 8080\n"
CMDS_B+="insert \"the platform database\" is PostgreSQL\n"
CMDS_B+="insert \"staging URL\" is staging.gaucho.dev\n"
CMDS_B+="insert \"production URL\" is app.gaucho.io\n"
CMDS_B+="insert \"token expiry\" is 24 hours\n"
CMDS_B+="insert \"maximum pool size\" is 20\n"
CMDS_B+="insert \"JWT algorithm\" is RS256\n"
# Now query
CMDS_B+="chat What port does the API Gateway run on?\n"
CMDS_B+="chat What database does the platform use?\n"
CMDS_B+="chat What is the staging URL?\n"
CMDS_B+="quit\n"

echo "  Running..."
OUTPUT_B=$(printf "$CMDS_B" | timeout 120 "$BIN" \
    --model "$MODEL" --vindex "$VINDEX" \
    --walk-only --no-warmup 2>/dev/null || true)

echo "  Results:"
for q in "API Gateway" "database" "staging"; do
    answer=$(echo "$OUTPUT_B" | grep -A3 "$q" | grep -v ">" | grep -v "prefill" | head -1 | tr -d '\n')
    if [ -n "$answer" ]; then
        echo "    Q: *$q* → $answer"
    fi
done

# ── Strategy A: RAG (inject context into prompt) ──
echo ""
echo -e "${CYAN}Strategy A: RAG — retrieve context, inject into prompt${NC}"
echo "  No INSERT needed. Feed the document as chat context."
echo ""

# Build the context from the architecture.md file
CONTEXT=$(cat "$FIXTURES/scenario_1_docs/architecture.md" | tr '\n' ' ' | sed 's/"/\\"/g')

CMDS_A=""
CMDS_A+="chat Based on this documentation: $CONTEXT --- What port does the API Gateway run on?\n"
CMDS_A+="chat Based on this documentation: $CONTEXT --- What database does the platform use?\n"
CMDS_A+="chat Based on this documentation: $CONTEXT --- What is the staging URL?\n"
CMDS_A+="quit\n"

echo "  Running..."
OUTPUT_A=$(printf "$CMDS_A" | timeout 120 "$BIN" \
    --model "$MODEL" --vindex "$VINDEX" \
    --walk-only --no-warmup 2>/dev/null || true)

echo "  Results:"
for q in "API Gateway" "database" "staging"; do
    answer=$(echo "$OUTPUT_A" | grep -A3 "$q" | grep -v ">" | grep -v "prefill" | head -1 | tr -d '\n')
    if [ -n "$answer" ]; then
        echo "    Q: *$q* → $answer"
    fi
done

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Strategy A (RAG): context injected into prompt"
echo "  Strategy B (Q-form): INSERT with question-like entities"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
