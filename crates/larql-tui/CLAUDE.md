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
# Terminal 1: start server (model loads once)
cargo run --release --features metal -p larql-server

# Terminal 2: TUI connects instantly
./target/release/larql

# Or with custom server URL
LARQL_SERVER=http://localhost:8080 ./target/release/larql
```

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
