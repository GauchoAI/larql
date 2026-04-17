# Skill: stats

When the user asks for statistics, metrics, or analysis of a folder or project:

1. Run the appropriate commands to gather data
2. Present in THREE formats:

## Raw output

```bash
find . -name "*.py" | wc -l
wc -l src/**/*.py
du -sh .
```

## Summary

```summary
Project overview:
- Languages: Python (60%), Rust (30%), Markdown (10%)
- Total files: N
- Lines of code: M
- Project size: S
```

## Chart data (for TUI visualization)

```chartjs
{
  "type": "bar",
  "data": {
    "labels": ["Python", "Rust", "Markdown"],
    "datasets": [{"data": [60, 30, 10], "label": "% of files"}]
  }
}
```
