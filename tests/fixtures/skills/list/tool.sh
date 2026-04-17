#!/bin/bash
# list skill tool — returns structured blocks
# Usage: tool.sh <path>
# Returns: ```raw```, ```summary```, ```chartjs``` blocks

PATH_ARG="${1:-$(pwd)}"

# Get raw listing
RAW=$(ls -la "$PATH_ARG" 2>&1)

# Count by type
TOTAL=$(echo "$RAW" | tail -n +2 | wc -l | tr -d ' ')
DIRS=$(echo "$RAW" | grep "^d" | wc -l | tr -d ' ')
FILES=$((TOTAL - DIRS))
PY=$(ls "$PATH_ARG"/*.py 2>/dev/null | wc -l | tr -d ' ')
RS=$(ls "$PATH_ARG"/*.rs 2>/dev/null | wc -l | tr -d ' ')
MD=$(ls "$PATH_ARG"/*.md 2>/dev/null | wc -l | tr -d ' ')
JS=$(ls "$PATH_ARG"/*.js "$PATH_ARG"/*.ts 2>/dev/null | wc -l | tr -d ' ')
TOML=$(ls "$PATH_ARG"/*.toml "$PATH_ARG"/*.yaml "$PATH_ARG"/*.json 2>/dev/null | wc -l | tr -d ' ')
OTHER=$((FILES - PY - RS - MD - JS - TOML))
SIZE=$(du -sh "$PATH_ARG" 2>/dev/null | cut -f1)

# Output structured blocks
echo '```raw'
echo "$RAW"
echo '```'
echo ''
echo '```summary'
echo "Listed $PATH_ARG: $TOTAL items ($DIRS directories, $FILES files)"
[ "$PY" -gt 0 ] && echo "- $PY Python files"
[ "$RS" -gt 0 ] && echo "- $RS Rust files"
[ "$MD" -gt 0 ] && echo "- $MD Markdown files"
[ "$JS" -gt 0 ] && echo "- $JS JavaScript/TypeScript files"
[ "$TOML" -gt 0 ] && echo "- $TOML Config files"
[ "$OTHER" -gt 0 ] && echo "- $OTHER Other files"
echo "Total size: $SIZE"
echo '```'
echo ''
echo '```chartjs'
echo "{\"type\":\"pie\",\"data\":{\"labels\":[\"Python\",\"Rust\",\"Markdown\",\"JS/TS\",\"Config\",\"Other\"],\"datasets\":[{\"data\":[$PY,$RS,$MD,$JS,$TOML,$OTHER]}]}}"
echo '```'
