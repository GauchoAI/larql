# chuk-larql-rs

Knowledge graphs extracted from neural network weights. One library, one binary, one graph format.

LARQL extracts knowledge from language models through three complementary methods:

- **Weight walking** — reads FFN weight matrices directly from safetensors files. Zero forward passes. Extracts what each neuron feature activates on and what it produces.
- **Residual capture** — runs targeted forward passes for seed entities and captures the hidden state at specific layers. These residuals seed Surreal's vector store for bulk factual discovery.
- **BFS probing** — sends structured prompts to a running model endpoint, chains next-token predictions into edges. Used as a validator for high-value candidates.

The extraction pipeline produces edges. Edges flow into the runtime graph. No vectors at runtime.

## Quick start

```bash
make release

# Extract lexical graph from weights (zero forward passes)
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.json

# Extract vectors for SurrealDB workshop
larql vector-extract google/gemma-3-4b-it -o vectors/ --resume

# Capture L25 residuals for seed entities (forward passes)
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals.vectors.ndjson

# Load into SurrealDB
larql vector-load vectors/ --ns larql --db gemma3_4b

# Query the graph
larql query --graph knowledge.larql.json France
larql stats knowledge.larql.json
```

## Documentation

| Doc | Description |
|---|---|
| [docs/weight-extraction.md](docs/weight-extraction.md) | Weight extraction pipeline — weights to vectors to SurrealDB to graph |
| [docs/cli.md](docs/cli.md) | Full CLI reference — all commands, flags, examples |
| [docs/format.md](docs/format.md) | Graph file format specification — JSON and MessagePack |
| [docs/confidence.md](docs/confidence.md) | Confidence and selectivity scoring |

## The extraction pipeline

```
weight-extract        → lexical edges (8.2M, zero forward passes)
vector-extract     → weight vectors to NDJSON (for SurrealDB)
vector-load        → vectors into SurrealDB with HNSW indexes
residuals capture  → L25 residuals for seed entities (targeted forward passes)
                     ↓
              SurrealDB workshop: format-adjusted queries discover factual edges
                     ↓
bfs                → validate top candidates with forward passes
                     ↓
              merge → knowledge.larql.json → Rust runtime (edges only, no vectors)
```

The vectors are the microscope. The edges are the photograph. You ship the photograph.

## Extraction commands

### Weight walking

Reads safetensors directly. Zero forward passes. BLAS-accelerated.

```bash
larql weight-extract google/gemma-3-4b-it -o knowledge.larql.json
larql weight-extract google/gemma-3-4b-it --layer 26 -o L26.larql.json --stats stats.json
```

### Attention walking

Extracts routing edges from attention OV circuits.

```bash
larql attention-extract google/gemma-3-4b-it -o attention.larql.json
```

### Vector extraction

Extracts full weight vectors to NDJSON for SurrealDB ingestion.

```bash
# All implemented components
larql vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down,ffn_gate,ffn_up,attn_ov,attn_qk,embeddings --resume

# Just factual layers
larql vector-extract google/gemma-3-4b-it -o vectors/ \
    --components ffn_down,ffn_gate --layers 25,26,27,28,29,30,31,32,33
```

| Component | What it stores | Dim | Per layer |
|---|---|---|---|
| `ffn_down` | Down projection column (output direction) | hidden | intermediate_size |
| `ffn_gate` | Gate projection row (input selectivity) | hidden | intermediate_size |
| `ffn_up` | Up projection row | hidden | intermediate_size |
| `attn_ov` | Mean OV circuit output direction | hidden | num_kv_heads |
| `attn_qk` | Q/K head projections | head_dim × hidden | num_q + num_kv |
| `embeddings` | Token embedding rows | hidden | vocab_size |

### Residual capture

Runs forward passes for seed entities and captures the hidden state at specified layers. These residuals go into SurrealDB as the seed for factual discovery queries.

```bash
# Capture L25 residuals
larql residuals capture google/gemma-3-4b-it \
    --entities "France,Germany,Japan,Mozart,Einstein" \
    --layer 25 -o residuals.vectors.ndjson

# Multiple layers
larql residuals capture google/gemma-3-4b-it \
    --entities entities.txt --layer 25 --layer 26 --layer 29 \
    -o residuals.vectors.ndjson

# All layers (full residual trajectory)
larql residuals capture google/gemma-3-4b-it \
    --entities "France" --all-layers -o residuals-full.vectors.ndjson
```

### SurrealDB loading

```bash
larql vector-load vectors/ --ns larql --db gemma3_4b
larql vector-load vectors/ --ns larql --db gemma3_4b --schema-only
```

### BFS probing

```bash
larql bfs --seeds "France,Germany" --templates templates.json \
    --endpoint http://localhost:11434/v1 --model gemma3:4b-it \
    -o knowledge.larql.json
```

## Query commands

```bash
larql query --graph knowledge.larql.json France capital-of
larql describe --graph knowledge.larql.json France
larql stats knowledge.larql.json
larql validate knowledge.larql.json
```

See [docs/cli.md](docs/cli.md) for full reference.

## Workspace structure

```
chuk-larql-rs/
├── crates/
│   ├── larql-core/       Library — graph engine, walkers, forward pass, I/O
│   ├── larql-cli/        Binary — CLI over larql-core
│   └── larql-python/     PyO3 binding — native Python extension
├── docs/
│   ├── cli.md            CLI reference
│   ├── format.md         Graph format specification
│   └── confidence.md     Confidence and selectivity scoring
├── examples/
├── Makefile
└── README.md
```

## Python integration

The `chuk-larql` Python package uses this Rust engine as its native backend.

```python
from chuk_larql import Graph, Edge
from _larql_core import weight_walk, attention_walk, load, save

g = weight_walk("google/gemma-3-4b-it")
save(g, "knowledge.larql.json")
```

## Building

```bash
make build          # debug build
make release        # optimized build
make test           # run all tests
make check          # check all crates including Python binding
make lint           # clippy
make python-build   # build Python extension (requires virtualenv)
```

### Feature flags

| Feature | Default | Description |
|---|---|---|
| `http` | yes | HTTP model provider (adds reqwest) |
| `msgpack` | yes | MessagePack serialization (adds rmp-serde) |
| `walker` | yes | Weight walking, vector extraction, residual capture (adds safetensors, ndarray, tokenizers, blas) |

## Status

### What's working

- **Weight walker** — 8.2M edges from Gemma 3-4B in 40 minutes. Confidence + selectivity scoring. Per-layer stats with threshold breakdowns, top entities, self-loop counts.
- **Attention walker** — OV circuit extraction with same scoring.
- **Vector extractor** — all 6 components: ffn_down, ffn_gate, ffn_up, attn_ov, attn_qk, embeddings.
- **Residual capture** — forward pass through transformer layers, captures hidden state at any layer for seed entities.
- **SurrealDB loader** — NDJSON → SurrealDB with HNSW indexes, batch insert, resume, schema-only mode.
- **Core graph engine** — full indexed graph with select, walk, search, subgraph, merge, shortest path.
- **BFS extraction** — template-based probing with multi-token chaining.
- **Serialization** — JSON and MessagePack with format auto-detection.
- **CLI** — 12 commands: weight-extract, attention-extract, vector-extract, vector-import, vector-load, vector-export-surql, residuals, bfs, stats, query, describe, validate.
- **PyO3 binding** — full Python API parity. 79 Python tests + 134 Rust tests.

### What's next

- CI / GitHub Actions
- `larql filter` command (post-extraction confidence/selectivity filtering)
- `larql merge` command (combine edge files)
- Packed binary edge format for runtime graphs

## License

Apache-2.0
