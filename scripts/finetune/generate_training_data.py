#!/usr/bin/env python3
"""Generate training data for larql foundation fine-tuning.

Teaches the model to:
1. Self-annotate facts with ```fact``` blocks
2. Track tasks with ```status``` blocks
3. Create plans with ```plan``` blocks
4. Use tools with ```tool``` blocks
5. Be concise (no filler)
"""

import json
import random

examples = []

# ═══════════════════════════════════════════════════════════════
# Category 1: Fact extraction — user states info, model annotates
# ═══════════════════════════════════════════════════════════════

fact_pairs = [
    ("My name is Miguel", "Nice to meet you, Miguel.", "user name", "Miguel", "user"),
    ("I work on machine learning systems", "Noted — you work on ML systems.", "domain", "machine learning systems", "user"),
    ("I prefer writing code in Rust", "Got it — Rust is your preferred language.", "preferred language", "Rust", "user"),
    ("The server runs on port 3000", "Understood, port 3000.", "server port", "3000", "user"),
    ("We're using Metal GPU for inference", "Metal GPU noted.", "compute", "Metal GPU", "user"),
    ("The project is called larql", "larql — noted.", "project name", "larql", "user"),
    ("I'm using an M4 Pro Mac", "M4 Pro — great hardware for ML.", "hardware", "M4 Pro Mac", "user"),
    ("The decode speed is 42 tokens per second", "42 tok/s — that's solid.", "decode speed", "42 tok/s", "user"),
    ("The model has 4 billion parameters", "4B params — compact but capable.", "model size", "4B parameters", "user"),
    ("We use Q4_K quantization", "Q4_K — good balance of speed and quality.", "quantization", "Q4_K", "user"),
    ("The vindex has 348,000 features", "348K features across the layers.", "vindex features", "348,000", "user"),
    ("I live in São Paulo", "São Paulo — great city.", "location", "São Paulo", "user"),
    ("My team uses GitHub for version control", "GitHub — standard choice.", "version control", "GitHub", "user"),
    ("The database is PostgreSQL", "PostgreSQL for the database.", "database", "PostgreSQL", "user"),
    ("We deploy on AWS", "AWS deployment noted.", "cloud provider", "AWS", "user"),
    ("The API uses REST with JSON", "REST + JSON API.", "api style", "REST with JSON", "user"),
    ("I started this project in April 2026", "April 2026 — relatively fresh project.", "project start", "April 2026", "user"),
    ("The KNN overlay matches at layer 26", "Layer 26 for KNN matching.", "knn layer", "26", "user"),
    ("We use ratatui for the TUI", "ratatui — solid TUI framework.", "tui framework", "ratatui", "user"),
    ("The tokenizer vocabulary is 262,144 tokens", "262K vocab — large vocabulary.", "vocab size", "262,144", "user"),
    ("My email is miguel@example.com", "Email saved.", "email", "miguel@example.com", "user"),
    ("The training data has 1000 examples", "1000 examples should be sufficient.", "training size", "1000 examples", "user"),
    ("We're fine-tuning with LoRA rank 16", "LoRA r=16 — good for format learning.", "lora rank", "16", "user"),
    ("The hidden dimension is 2816", "2816 hidden dim.", "hidden dim", "2816", "user"),
    ("I usually work from 9am to 6pm", "9-6 work hours noted.", "work hours", "9am to 6pm", "user"),
    ("The batch size is 1 for fine-tuning", "Batch size 1 on Apple Silicon.", "batch size", "1", "user"),
    ("I'm building this for a research paper", "Research paper — noted the goal.", "goal", "research paper", "user"),
    ("The attention mechanism uses 16 heads", "16 attention heads.", "num heads", "16", "user"),
    ("We have 34 transformer layers", "34 layers in the model.", "num layers", "34", "user"),
    ("The sliding window is 1024 tokens", "1024 token sliding window.", "sliding window", "1024", "user"),
]

for user_msg, response, key, value, source in fact_pairs:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```fact
key: {key}
value: {value}
source: {source}
```"""}
        ]
    })

# Multiple facts in one message
multi_fact_msgs = [
    (
        "My name is Alice, I'm a data scientist, and I work at Google",
        "Nice to meet you, Alice.",
        [("user name", "Alice", "user"), ("role", "data scientist", "user"), ("company", "Google", "user")]
    ),
    (
        "We're using PyTorch 2.0 on 8 A100 GPUs for training",
        "PyTorch 2.0 on 8×A100 — serious setup.",
        [("framework", "PyTorch 2.0", "user"), ("gpu setup", "8 A100 GPUs", "user")]
    ),
    (
        "The API latency is 50ms and we serve 10,000 requests per second",
        "50ms latency at 10K RPS — solid performance.",
        [("api latency", "50ms", "user"), ("throughput", "10,000 RPS", "user")]
    ),
]

for user_msg, response, facts in multi_fact_msgs:
    blocks = "\n\n".join([
        f"```fact\nkey: {k}\nvalue: {v}\nsource: {s}\n```"
        for k, v, s in facts
    ])
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"{response}\n\n{blocks}"}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Category 2: Task tracking — model emits status blocks
# ═══════════════════════════════════════════════════════════════

status_conversations = [
    ("Fix the NaN bug in the Metal shader", "I'll look into the NaN issue in the Metal shader.",
     "fix Metal NaN", "active", "investigating shader output"),
    ("Can you optimize the decode speed?", "I'll work on optimizing decode performance.",
     "optimize decode speed", "active", "profiling current bottlenecks"),
    ("Debug why the KNN cosine is only 0.11", "Let me trace the KNN matching path.",
     "debug KNN cosine", "active", "comparing insert vs probe residuals"),
    ("Add a caching layer to the API", "I'll add caching to reduce latency.",
     "add API caching", "active", "evaluating cache strategies"),
    ("Write unit tests for the new module", "I'll write tests for the module.",
     "write unit tests", "active", "identifying test cases"),
    ("Benchmark the new Q4_K shader", "Running benchmarks now.",
     "benchmark Q4_K", "active", "measuring tok/s on 5 queries"),
    ("Refactor the 800-line monolith", "I'll split it into modules.",
     "refactor monolith", "active", "identifying module boundaries"),
    ("Deploy the latest version to staging", "Deploying to staging.",
     "deploy to staging", "active", "building release binary"),
]

for user_msg, response, task, state, detail in status_conversations:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```status
task: {task}
state: {state}
detail: {detail}
```"""}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Category 3: Plan creation — model breaks tasks into steps
# ═══════════════════════════════════════════════════════════════

plan_conversations = [
    (
        "Plan how to add Redis caching to the server",
        "Here's the plan for adding Redis caching:",
        "add Redis caching",
        ["Research Redis client libraries", "Add Redis dependency to Cargo.toml",
         "Implement cache-aside pattern for /v1/infer", "Add cache invalidation on /v1/insert",
         "Write integration tests", "Benchmark cache hit/miss latency"]
    ),
    (
        "Plan the MoE model integration",
        "Here's the plan for MoE integration:",
        "integrate MoE model",
        ["Load GGUF expert tensors (3D)", "Add router matvec dispatch",
         "Implement top-K expert selection", "Add per-expert Q4_K matvec at buffer offsets",
         "Wire shared expert dense FFN", "Benchmark against Ollama baseline"]
    ),
    (
        "Plan how to implement the self-annotation system",
        "Here's the plan:",
        "implement self-annotation",
        ["Define annotation block formats (fact/status/plan)", "Add TUI block extraction",
         "Persist facts to JSONL file", "Persist workflows to JSON",
         "Inject facts into model at startup via KNN", "Build test scenarios"]
    ),
    (
        "Break down the database migration into steps",
        "Here's the migration plan:",
        "database migration",
        ["Backup current database", "Create new schema with migration scripts",
         "Test migration on staging copy", "Run migration on production",
         "Verify data integrity", "Update application connection strings"]
    ),
]

for user_msg, response, workflow_name, steps in plan_conversations:
    step_lines = "\n".join([f"  {i+1}. {s} [pending]" for i, s in enumerate(steps)])
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```plan
workflow: {workflow_name}
steps:
{step_lines}
```"""}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Category 4: Tool usage — model calls tools
# ═══════════════════════════════════════════════════════════════

tool_conversations = [
    ("List the files in the current directory", "Let me list the files.",
     "list ."),
    ("Show me what's in the src folder", "Checking the src directory.",
     "list src/"),
    ("What files are in the home directory?", "Let me check your home directory.",
     "list ~/"),
    ("List the test files", "Looking at the test directory.",
     "list tests/"),
    ("What do you know about me?", "Let me check the stored facts.",
     "facts"),
    ("Show me the facts you've stored", "Retrieving stored knowledge.",
     "facts"),
]

for user_msg, response, tool_call in tool_conversations:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```tool
{tool_call}
```"""}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Category 5: Concise answers (no filler)
# ═══════════════════════════════════════════════════════════════

concise_conversations = [
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris."),
    ("What language is Rust written in?", "Rust is self-hosted — the compiler is written in Rust."),
    ("How many bits in a byte?", "8."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("What is a LoRA?", "Low-Rank Adaptation — a parameter-efficient fine-tuning method that adds small trainable matrices to frozen model weights."),
    ("What's the time complexity of binary search?", "O(log n)."),
    ("What port does HTTPS use?", "443."),
]

for user_msg, response in concise_conversations:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": response}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Category 6: Combined — fact + status in one response
# ═══════════════════════════════════════════════════════════════

combined = [
    (
        "The server is running on port 8080. Can you check if the API is responding?",
        "I'll check the API on port 8080.",
        [("server port", "8080", "user")],
        ("check API health", "active", "testing endpoint connectivity")
    ),
    (
        "I'm using Python 3.12. Help me set up the virtual environment.",
        "Setting up the venv for Python 3.12.",
        [("python version", "3.12", "user")],
        ("setup virtual environment", "active", "creating venv with Python 3.12")
    ),
]

for user_msg, response, facts, (task, state, detail) in combined:
    fact_blocks = "\n\n".join([
        f"```fact\nkey: {k}\nvalue: {v}\nsource: {s}\n```"
        for k, v, s in facts
    ])
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

{fact_blocks}

```status
task: {task}
state: {state}
detail: {detail}
```"""}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Duplicate with variations to reach ~200 examples
# ═══════════════════════════════════════════════════════════════

# Add derived-fact examples (model figures something out)
derived_facts = [
    ("Can you explain how KV cache works?",
     "The KV cache stores key and value tensors from previous tokens so they don't need to be recomputed. During autoregressive generation, only the new token's K and V are computed and appended.",
     "concept explained", "KV cache — stores past K/V tensors to avoid recomputation", "derived"),
    ("Why is Q4_K better than Q4_0?",
     "Q4_K uses super-blocks of 256 values with separate scales and mins per sub-block, giving better precision than Q4_0's flat 32-value blocks with a single scale.",
     "quantization comparison", "Q4_K > Q4_0 due to sub-block scales", "derived"),
    ("What's the difference between MoE and dense models?",
     "MoE (Mixture of Experts) routes each token through a subset of expert FFN modules, reading fewer parameters per token. Dense models activate all parameters for every token.",
     "architecture", "MoE activates subset of experts per token vs dense activating all", "derived"),
]

for user_msg, response, key, value, source in derived_facts:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```fact
key: {key}
value: {value}
source: {source}
```"""}
        ]
    })

# Status completion examples
status_done = [
    ("The benchmark is done — 42 tok/s", "Results recorded.",
     "benchmark", "done", "42 tok/s measured"),
    ("I fixed the NaN bug", "Great, marking as complete.",
     "fix NaN", "done", "root cause was scale_vector dispatch size"),
    ("Tests are passing now", "All tests green.",
     "run tests", "done", "all assertions pass"),
]

for user_msg, response, task, state, detail in status_done:
    examples.append({
        "messages": [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": f"""{response}

```status
task: {task}
state: {state}
detail: {detail}
```"""}
        ]
    })

# ═══════════════════════════════════════════════════════════════
# Shuffle and split
# ═══════════════════════════════════════════════════════════════

random.seed(42)
random.shuffle(examples)

split = int(len(examples) * 0.9)
train = examples[:split]
valid = examples[split:]

# Write files
with open("scripts/finetune/data/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open("scripts/finetune/data/valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(ex) + "\n")

print(f"Generated {len(examples)} examples")
print(f"  Train: {len(train)}")
print(f"  Valid: {len(valid)}")

# Stats
cats = {"fact": 0, "status": 0, "plan": 0, "tool": 0, "concise": 0, "combined": 0}
for ex in examples:
    content = ex["messages"][-1]["content"]
    if "```fact" in content and "```status" in content:
        cats["combined"] += 1
    elif "```fact" in content:
        cats["fact"] += 1
    elif "```status" in content:
        cats["status"] += 1
    elif "```plan" in content:
        cats["plan"] += 1
    elif "```tool" in content:
        cats["tool"] += 1
    else:
        cats["concise"] += 1

print("  Categories:", cats)
