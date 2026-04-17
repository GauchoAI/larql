#!/bin/bash
# stats skill tool — project statistics with chart data
# Usage: tool.sh [path]

PROJECT="${1:-.}"

PY=$(find "$PROJECT" -name "*.py" 2>/dev/null | wc -l | tr -d ' ')
RS=$(find "$PROJECT" -name "*.rs" 2>/dev/null | wc -l | tr -d ' ')
MD=$(find "$PROJECT" -name "*.md" 2>/dev/null | wc -l | tr -d ' ')
JS=$(find "$PROJECT" -name "*.js" -o -name "*.ts" 2>/dev/null | wc -l | tr -d ' ')
SH=$(find "$PROJECT" -name "*.sh" 2>/dev/null | wc -l | tr -d ' ')
TOTAL=$((PY + RS + MD + JS + SH))
SIZE=$(du -sh "$PROJECT" 2>/dev/null | cut -f1)
LOC=$(find "$PROJECT" -name "*.py" -o -name "*.rs" -o -name "*.js" -o -name "*.ts" | head -100 | xargs wc -l 2>/dev/null | tail -1 | awk '{print $1}')

echo '```raw'
echo "Path: $PROJECT"
echo "Python: $PY files"
echo "Rust: $RS files"
echo "Markdown: $MD files"
echo "JS/TS: $JS files"
echo "Shell: $SH files"
echo "Total: $TOTAL files, $SIZE"
echo "Lines of code: ~$LOC"
echo '```'
echo ''
echo '```summary'
echo "Project: $PROJECT ($SIZE, ~$LOC lines of code)"
echo "- $TOTAL files across $(echo "$PY $RS $MD $JS $SH" | tr ' ' '\n' | awk '$1>0' | wc -l | tr -d ' ') languages"
[ "$PY" -gt 0 ] && echo "- Python: $PY files"
[ "$RS" -gt 0 ] && echo "- Rust: $RS files"
[ "$MD" -gt 0 ] && echo "- Docs: $MD markdown files"
[ "$JS" -gt 0 ] && echo "- JS/TS: $JS files"
[ "$SH" -gt 0 ] && echo "- Scripts: $SH shell files"
echo '```'
echo ''
echo '```chartjs'
cat <<CHART
{"type":"bar","data":{"labels":["Python","Rust","Markdown","JS/TS","Shell"],"datasets":[{"label":"Files","data":[$PY,$RS,$MD,$JS,$SH],"backgroundColor":["#3572A5","#DEA584","#083FA1","#F7DF1E","#89E051"]}]}}
CHART
echo '```'
