#!/bin/bash
# search skill tool — grep with structured output
# Usage: tool.sh <pattern> [path]

PATTERN="${1:?Usage: tool.sh <pattern> [path]}"
SEARCH_PATH="${2:-.}"

RAW=$(grep -rn --include="*.{py,rs,js,ts,md,toml,yaml}" "$PATTERN" "$SEARCH_PATH" 2>/dev/null | head -50)
MATCH_COUNT=$(echo "$RAW" | grep -c . 2>/dev/null || echo 0)
FILE_COUNT=$(echo "$RAW" | cut -d: -f1 | sort -u | wc -l | tr -d ' ')
TOP_FILE=$(echo "$RAW" | cut -d: -f1 | sort | uniq -c | sort -rn | head -1 | awk '{print $2}')

echo '```raw'
echo "$RAW"
echo '```'
echo ''
echo '```summary'
echo "Found $MATCH_COUNT matches across $FILE_COUNT files for \"$PATTERN\""
if [ -n "$TOP_FILE" ]; then
    echo "Most matches in: $TOP_FILE"
fi
echo "$RAW" | cut -d: -f1 | sort -u | while read f; do
    LINES=$(echo "$RAW" | grep "^$f:" | cut -d: -f2 | tr '\n' ',' | sed 's/,$//')
    echo "- $f (lines: $LINES)"
done
echo '```'
