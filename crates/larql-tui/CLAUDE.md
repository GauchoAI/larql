# larql-tui

Ratatui terminal interface for larql — LLM as a Database.

## Architecture

The TUI is a **thin client**. It does NOT run the model. It connects to `bench_interactive` (the model backend) via stdin/stdout pipes and renders the output.

```
┌─────────────┐    stdin (commands)     ┌──────────────────────┐
│   larql-tui  │ ───────────────────►   │  bench_interactive   │
│  (ratatui)   │ ◄───────────────────   │  (GPU model, 41t/s)  │
│              │    stdout (tokens)     │  9 GB RSS, Metal GPU │
└─────────────┘    stderr (metrics)     └──────────────────────┘
```

Two modes:
- **Subprocess**: TUI spawns bench_interactive as a child process. 6s startup.
- **Daemon**: bench_interactive runs in background via FIFOs. TUI connects instantly.

## IPC Protocol (commands sent to bench_interactive stdin)

| Command | Format | Response on stdout |
|---|---|---|
| `chat <text>` | Single line, wraps in Gemma 3 chat template internally | Streamed tokens, then `\n  prefill: Xms  decode: Xms/tok (Y tok/s) over Z tokens  [stopped on EOS]\n> ` |
| `chatml` | Multi-line until `---END---` marker | Same as chat |
| `ask <text> [N]` | Raw completion (no chat template), N tokens (default 20) | Echo of prompt + tokens + timing line + `> ` |
| `insert <entity> <relation> <target>` | Three words | `  inserted: entity —[relation]→ target  LN  (Xms)\n  KNN overlay now: N entries\n> ` |
| `quit` | — | Process exits |

### Stdout format details

- **Tokens**: printed one at a time via `print!("{tok}")` (no newline between tokens)
- **Timing line**: `  prefill: Xms  decode: Xms/tok (Y.YY tok/s) over Z tokens  [stopped on EOS]` (one line, after all tokens)
- **Prompt**: `> ` (two chars, NO newline — must detect via partial buffer flush)
- **KNN override**: `Canberra (KNN override, cos=0.85, L26, GPU prefill)` (inline with prompt echo)

### Stderr format

- `[load] model: Xs` — model loading progress
- `[ready] backend=metal (GPU) layers=34` — ready signal
- `[knn-gpu-prefill L26] top1=... cos=...` — KNN probe trace

## Key files

| File | Purpose |
|---|---|
| `src/main.rs` | Everything — app state, backend, drawing, input, tool execution |

## State machine

```
LOADING → READY → GENERATING → READY → GENERATING → ...
                                  ↑          │
                                  └── tool result feedback (automatic)
```

- `is_generating = true`: input locked, tokens streaming
- `is_generating = false`: input unlocked, waiting for user
- Transitions: `true` on Enter, `false` on `> ` prompt detection or 30s timeout

## Tool execution

When the model outputs a markdown code block:
- **```bash**: Executed via `sh -c`, output shown in TUI, fed back to model via `chatml`
- **```python/js/etc**: Saved to `output.{ext}` in CWD

## Daemon mode

- `larql --daemon`: Creates FIFOs at `/tmp/larql-daemon.{stdin,stdout,stderr}`, starts bench_interactive reading/writing them, writes PID to `/tmp/larql-daemon.pid`
- `larql`: Checks PID file. If daemon alive, connects to FIFOs. If not, spawns subprocess.
- `larql --stop`: Kills daemon by PID.

## Building

```bash
# TUI only (no Metal needed — TUI is pure Rust)
cargo build --release -p larql-tui

# Backend (needs Metal feature on macOS)
cargo build --release --features metal -p larql-inference --example bench_interactive
```

Binary: `target/release/larql`

## Dependencies

- `ratatui` 0.29 — terminal UI framework
- `crossterm` 0.28 — terminal I/O
- `tokio` 1 — async runtime (for future SSE/WebSocket support)
- `serde`, `serde_json` — config/protocol parsing

## Environment variables

| Var | Default | Purpose |
|---|---|---|
| `LARQL_MODEL` | `/Users/miguel_lemos/Desktop/gemma-3-4b-it` | Model path |
| `LARQL_VINDEX` | `...gemma3-4b.vindex` | Vindex path |

## Debug

Logs written to `$TMPDIR/larql-tui.log` — all stdout/stderr from backend plus state transitions.
