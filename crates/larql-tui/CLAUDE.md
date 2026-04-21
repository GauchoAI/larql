# larql-tui

Thin ratatui client for larql. Connects to larql-server via HTTP.

## Architecture

```
┌─────────────┐    HTTP POST + SSE     ┌──────────────────────┐
│   larql-tui  │ ───────────────────►   │  larql-server        │
│  (ratatui)   │ ◄───────────────────   │  /v1/chat/completions│
│  300 LOC     │    SSE token stream    │  41 tok/s, 9 GB      │
└─────────────┘                         └──────────────────────┘
      ↑
  ~/.larql/skills/   ← skill matching + tool execution (TUI-side)
```

Server runs separately. TUI connects instantly. Restart TUI without reloading model.

## Running

```bash
# Terminal 1: start server (model loads once, ~40s warmup)
cargo run --release -p larql-server -- /path/to/gemma3-4b.vindex

# Terminal 2: TUI connects instantly
./target/release/larql

# Or with custom server URL (default: http://localhost:3000)
LARQL_SERVER=http://other-host:3000 ./target/release/larql
```

Metal GPU is the default feature. Server loads `interleaved_q4k_real.bin` for
GPU decode (~20 tok/s via Metal Q4_K pipeline).

## Rendering

Uses `gc-markdown` (ported from gaucho-code) for rich terminal rendering:
- Markdown: headers, bold, italic, lists
- Code blocks: syntax-highlighted by language tag
- Custom blocks: `chartjs` (ASCII bar/pie charts), `diff`, `csv`, `terminal`
- Tables: pipe-delimited markdown tables

## Skills

Skills in `~/.larql/skills/` and `./.skills/` (project-local):

```
~/.larql/skills/
  list/
    skill.md    ← LLM instructions ("output ```tool list <path>```")
    tool.sh     ← executable, returns ```raw```, ```summary```, ```chartjs```
  git/
    skill.md
    tool.sh
```

TUI auto-matches skills by keywords in user input, injects skill.md as context.
When model outputs ```tool```, TUI executes tool.sh and routes output:
- `summary` → fed back to model for commentary
- `chartjs` → rendered as chart in TUI  
- `raw` → logged (not shown)

## Key files

| File | Purpose |
|---|---|
| `src/main.rs` | HTTP client, SSE streaming, skills, rendering (~300 LOC) |
| `Cargo.toml` | deps: ratatui, reqwest, gc-markdown |

## Message types

```rust
enum Message {
    User(String),           // ❯ bold prompt
    Assistant(String),      // gc-markdown rendered
    System(String),         // italic dimmed
    ToolUse { tool, detail }, // ⚡ magenta
    ToolResult { summary },  // gc-markdown rendered
    Metrics { tok_s, tokens }, // dimmed stats
}
```

## Environment

| Var | Default | Purpose |
|---|---|---|
| `LARQL_SERVER` | `http://localhost:3000` | Server URL |
