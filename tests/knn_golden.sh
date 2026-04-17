#!/bin/bash
# KNN Override Golden Test Runner
# Assumes server is ALREADY running. Does not restart.
# Usage: ./tests/knn_golden.sh [http://host:port]

set -euo pipefail
SERVER="${1:-http://localhost:3000}"
GOLDEN="tests/fixtures/knn_override_golden.tsv"
NEGATIVE="tests/fixtures/knn_override_negative.tsv"
PASS=0; FAIL=0

GREEN='\033[0;32m'; RED='\033[0;31m'; NC='\033[0m'

curl -s "$SERVER/v1/health" | grep -q ok || { echo "Server not reachable at $SERVER"; exit 1; }

chat() {
    curl -s --max-time "${2:-10}" "$SERVER/v1/chat/completions" \
        -H "Content-Type: application/json" \
        -d "{\"messages\":[{\"role\":\"user\",\"content\":$(printf '%s' "$1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}]}" 2>/dev/null | \
    python3 -c "
import sys,json
for l in sys.stdin:
    l=l.strip()
    if l.startswith('data:') and 'content' in l:
        try: print(json.loads(l[5:])['choices'][0]['delta'].get('content',''),end='')
        except: pass
" 2>/dev/null
}

insert() {
    curl -s --max-time 10 "$SERVER/v1/insert" \
        -H "Content-Type: application/json" \
        -d "{\"entity\":\"test\",\"relation\":\"test\",\"target\":$(printf '%s' "$2" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))'),\"mode\":\"knn\",\"prompt\":$(printf '%s' "$1" | python3 -c 'import sys,json; print(json.dumps(sys.stdin.read()))')}" >/dev/null 2>&1
}

echo "=== KNN Golden Tests ==="

echo "--- Positive (INSERT → must contain) ---"
while IFS=$'\t' read -r IPROMPT TARGET TPROMPT EXPECT DESC; do
    [[ "$IPROMPT" =~ ^#.*$ || -z "$IPROMPT" ]] && continue
    insert "$IPROMPT" "$TARGET"
    # Skip INSERT-only rows (no test needed)
    [[ "$TPROMPT" == "_L1_only" ]] && { echo "  [insert] $DESC"; continue; }
    printf "  %-50s " "$DESC"
    OUTPUT=$(chat "$TPROMPT")

    # Unescape \n in EXPECT for matching
    EXPECT_UNESC=$(printf '%b' "$EXPECT")
    if printf '%s' "$OUTPUT" | grep -qF "$EXPECT_UNESC"; then
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS+1))
    else
        echo -e "${RED}FAIL${NC}"
        echo "    expected: $EXPECT"
        echo "    got:      ${OUTPUT:0:120}"
        FAIL=$((FAIL+1))
    fi
done < "$GOLDEN"

echo ""
echo "--- Negative (must NOT contain) ---"
while IFS=$'\t' read -r TPROMPT MUSTNOT DESC; do
    [[ "$TPROMPT" =~ ^#.*$ || -z "$TPROMPT" ]] && continue
    printf "  %-50s " "$DESC"
    OUTPUT=$(chat "$TPROMPT" 5)

    if printf '%s' "$OUTPUT" | grep -qF "$MUSTNOT"; then
        echo -e "${RED}FAIL${NC}"
        echo "    unwanted: $MUSTNOT"
        echo "    got:      ${OUTPUT:0:120}"
        FAIL=$((FAIL+1))
    else
        echo -e "${GREEN}PASS${NC}"
        PASS=$((PASS+1))
    fi
done < "$NEGATIVE"

echo ""
echo "=== $PASS pass, $FAIL fail ==="
[ "$FAIL" -eq 0 ]
