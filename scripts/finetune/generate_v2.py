#!/usr/bin/env python3
"""Generate comprehensive training data v2 for larql foundation fine-tuning.

500+ examples covering:
1. Fact extraction (diverse domains, multi-fact, derived facts)
2. Workflow tracking (status updates, step progression, completion)
3. Plan creation (varied complexity, domain-specific)
4. Tool usage (list, facts, search, git)
5. Concise answers (no filler, direct)
6. Multi-turn context (referring to previous facts)
7. Fact retrieval (model recalls stored facts)
8. Error/blocked workflows
9. Combined annotations (fact + status in one response)
"""

import json
import random

random.seed(42)
examples = []

def add(user, assistant):
    examples.append({"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant}
    ]})

def fact_block(key, value, source="user"):
    return f"```fact\nkey: {key}\nvalue: {value}\nsource: {source}\n```"

def status_block(task, state, detail):
    return f"```status\ntask: {task}\nstate: {state}\ndetail: {detail}\n```"

def plan_block(name, steps):
    lines = "\n".join(f"  {i+1}. {s} [pending]" for i, s in enumerate(steps))
    return f"```plan\nworkflow: {name}\nsteps:\n{lines}\n```"

def tool_block(cmd):
    return f"```tool\n{cmd}\n```"

# ═══════════════════════════════════════════════════════════════
# 1. FACT EXTRACTION — 100+ examples
# ═══════════════════════════════════════════════════════════════

# Personal info
personal = [
    ("My name is {name}", "Nice to meet you, {name}.", "user name", "{name}"),
    ("Call me {name}", "Got it, {name}.", "preferred name", "{name}"),
    ("I'm a {role}", "{role} — noted.", "role", "{role}"),
    ("I work at {company}", "{company} — noted.", "company", "{company}"),
    ("I'm based in {city}", "{city} — got it.", "location", "{city}"),
    ("My timezone is {tz}", "Timezone {tz}.", "timezone", "{tz}"),
    ("I speak {lang}", "{lang} speaker.", "language", "{lang}"),
    ("I've been coding for {years} years", "{years} years of experience.", "experience", "{years} years"),
    ("I prefer {editor}", "{editor} — solid choice.", "editor", "{editor}"),
    ("I use {os}", "{os} user.", "os", "{os}"),
]

names = ["Miguel", "Alice", "Bob", "Yuki", "Priya", "Carlos", "Sarah", "Ahmed"]
roles = ["data scientist", "backend engineer", "ML researcher", "DevOps engineer", "frontend developer", "CTO", "security engineer"]
companies = ["Google", "Meta", "a startup", "AWS", "OpenAI", "my own company", "a university"]
cities = ["São Paulo", "Tokyo", "Berlin", "New York", "London", "Singapore", "Bangalore"]
timezones = ["UTC-3", "UTC+9", "UTC+1", "UTC-5", "UTC+0", "UTC+8", "UTC+5:30"]
languages = ["Portuguese", "Japanese", "German", "English and Spanish", "Mandarin", "Hindi"]
years_list = ["5", "10", "15", "2", "20", "8"]
editors = ["VS Code", "Neovim", "Helix", "Zed", "IntelliJ", "Emacs"]
oses = ["macOS", "Ubuntu", "Arch Linux", "Windows 11", "NixOS", "Fedora"]

for template, resp, key, val in personal:
    pool = {"name": names, "role": roles, "company": companies, "city": cities,
            "tz": timezones, "lang": languages, "years": years_list,
            "editor": editors, "os": oses}
    # Find which placeholder this template uses
    for placeholder, values in pool.items():
        if "{" + placeholder + "}" in template:
            for v in random.sample(values, min(3, len(values))):
                u = template.replace("{" + placeholder + "}", v)
                r = resp.replace("{" + placeholder + "}", v)
                k = key
                vv = val.replace("{" + placeholder + "}", v)
                add(u, f"{r}\n\n{fact_block(k, vv)}")

# Technical facts
tech_facts = [
    ("The server runs on port {v}", "Port {v} noted.", "server port", "{v}"),
    ("We use {v} for the database", "{v} database.", "database", "{v}"),
    ("The API latency is {v}", "{v} latency.", "api latency", "{v}"),
    ("Memory usage is {v}", "{v} memory.", "memory usage", "{v}"),
    ("The model has {v} parameters", "{v} params.", "model size", "{v}"),
    ("We're using {v} quantization", "{v} quantization.", "quantization", "{v}"),
    ("The batch size is {v}", "Batch size {v}.", "batch size", "{v}"),
    ("Learning rate is {v}", "LR {v}.", "learning rate", "{v}"),
    ("The hidden dimension is {v}", "Hidden dim {v}.", "hidden dim", "{v}"),
    ("We have {v} transformer layers", "{v} layers.", "num layers", "{v}"),
    ("The framework is {v}", "{v} framework.", "framework", "{v}"),
    ("CI runs on {v}", "{v} CI.", "ci", "{v}"),
    ("The test coverage is {v}", "{v} coverage.", "test coverage", "{v}"),
    ("Deployment is on {v}", "{v} deployment.", "deployment", "{v}"),
    ("The repo is at {v}", "Repo at {v}.", "repo", "{v}"),
]

tech_values = {
    "v": ["3000", "8080", "443", "5432",
          "PostgreSQL", "Redis", "MongoDB", "SQLite",
          "50ms", "200ms", "3s", "12ms",
          "512 MB", "4 GB", "16 GB", "1.2 GB",
          "4B", "7B", "13B", "70B",
          "Q4_K", "Q8_0", "FP16", "BF16",
          "1", "4", "8", "32",
          "1e-4", "5e-5", "3e-4", "1e-3",
          "2816", "4096", "768", "1024",
          "34", "32", "24", "48",
          "PyTorch", "JAX", "MLX", "TensorFlow",
          "GitHub Actions", "Jenkins", "CircleCI", "GitLab CI",
          "87%", "95%", "72%", "100%",
          "AWS", "GCP", "Vercel", "self-hosted",
          "github.com/user/repo", "gitlab.com/team/project"]
}

for template, resp, key, val in tech_facts:
    for v in random.sample(tech_values["v"], 2):
        u = template.replace("{v}", v)
        r = resp.replace("{v}", v)
        vv = val.replace("{v}", v)
        add(u, f"{r}\n\n{fact_block(key, vv)}")

# Multi-fact messages
add("My name is Alice, I'm a data scientist at Google, and I use Python",
    f"Nice to meet you, Alice.\n\n{fact_block('user name', 'Alice')}\n\n{fact_block('role', 'data scientist')}\n\n{fact_block('company', 'Google')}\n\n{fact_block('preferred language', 'Python')}")

add("We're running PyTorch 2.0 on 8 A100 GPUs with batch size 32",
    f"Serious training setup.\n\n{fact_block('framework', 'PyTorch 2.0')}\n\n{fact_block('gpu setup', '8 A100 GPUs')}\n\n{fact_block('batch size', '32')}")

add("The project uses Rust for the backend, React for the frontend, and PostgreSQL for storage",
    f"Full-stack setup noted.\n\n{fact_block('backend', 'Rust')}\n\n{fact_block('frontend', 'React')}\n\n{fact_block('database', 'PostgreSQL')}")

add("I'm Miguel, working on larql, which runs Gemma 3 4B at 42 tok/s on Metal GPU",
    f"Miguel building larql — impressive speed.\n\n{fact_block('user name', 'Miguel')}\n\n{fact_block('project', 'larql')}\n\n{fact_block('model', 'Gemma 3 4B')}\n\n{fact_block('speed', '42 tok/s')}\n\n{fact_block('compute', 'Metal GPU')}")

# Derived facts (model learns something)
derived = [
    ("How does KV caching work?",
     "KV caching stores key and value tensors from previous tokens so they're not recomputed during autoregressive generation. Only the new token's K and V are computed and appended.",
     "concept", "KV cache stores past K/V to avoid recomputation", "derived"),
    ("What's the difference between Q4_K and Q4_0?",
     "Q4_K uses super-blocks of 256 values with per-sub-block scales and mins (148 bytes). Q4_0 uses flat 32-value blocks with a single scale (18 bytes). Q4_K has better precision.",
     "quantization", "Q4_K has sub-block scales (better precision) vs Q4_0 flat blocks", "derived"),
    ("Explain MoE architectures",
     "Mixture of Experts routes each token through a subset of expert FFN modules via a learned router. Only active experts are computed, reducing per-token cost while maintaining total parameter count.",
     "architecture", "MoE routes tokens through expert subsets via learned router", "derived"),
    ("What is LoRA?",
     "Low-Rank Adaptation adds trainable rank-decomposition matrices to frozen weights: W' = W + BA where B and A are small. This trains <1% of parameters while maintaining performance.",
     "technique", "LoRA adds low-rank BA matrices to frozen weights", "derived"),
    ("How does RMSNorm differ from LayerNorm?",
     "RMSNorm only divides by the root mean square (no mean subtraction). It's simpler and faster than LayerNorm, which subtracts the mean then divides by std. Most modern LLMs use RMSNorm.",
     "normalization", "RMSNorm skips mean subtraction, divides by RMS only", "derived"),
]

for user, response, key, value, source in derived:
    add(user, f"{response}\n\n{fact_block(key, value, source)}")

# ═══════════════════════════════════════════════════════════════
# 2. WORKFLOW TRACKING — 60+ examples
# ═══════════════════════════════════════════════════════════════

# Starting tasks
tasks_start = [
    ("Fix the NaN bug in the Metal shader", "Investigating the NaN issue.", "fix Metal NaN", "analyzing shader output"),
    ("Optimize the decode speed", "Working on decode optimization.", "optimize decode", "profiling bottlenecks"),
    ("Debug the KNN cosine mismatch", "Tracing the KNN path.", "debug KNN cosine", "comparing residuals"),
    ("Add caching to the API", "Adding cache layer.", "add API caching", "evaluating strategies"),
    ("Write tests for the parser", "Writing parser tests.", "test parser", "identifying edge cases"),
    ("Benchmark Q4_K vs Q6_K", "Running benchmarks.", "benchmark quantization", "measuring tok/s"),
    ("Refactor the monolith", "Splitting into modules.", "refactor codebase", "mapping dependencies"),
    ("Deploy to production", "Starting deployment.", "deploy production", "building release"),
    ("Fix the memory leak", "Investigating memory.", "fix memory leak", "profiling allocations"),
    ("Add WebSocket support", "Implementing WebSocket.", "add WebSocket", "choosing library"),
    ("Set up monitoring", "Configuring monitoring.", "setup monitoring", "installing Prometheus"),
    ("Migrate the database", "Starting migration.", "database migration", "backing up data"),
    ("Implement rate limiting", "Adding rate limits.", "implement rate limiting", "designing token bucket"),
    ("Update dependencies", "Updating deps.", "update deps", "checking compatibility"),
    ("Add authentication", "Implementing auth.", "add authentication", "evaluating JWT vs session"),
]

for user, response, task, detail in tasks_start:
    add(user, f"{response}\n\n{status_block(task, 'active', detail)}")

# Task completion
completions = [
    ("The benchmark is done — 42 tok/s", "Results recorded.", "benchmark", "42 tok/s measured"),
    ("Fixed the NaN — it was scale_vector dispatch", "Root cause found and fixed.", "fix NaN", "scale_vector only covered 256/2816 elements"),
    ("All tests passing", "Clean test suite.", "run tests", "all assertions pass"),
    ("Deployed successfully", "Live in production.", "deploy", "zero-downtime deployment complete"),
    ("Memory leak fixed — was from buffer cache", "Memory stable now.", "fix memory leak", "buffer cache entries now evicted properly"),
    ("Migration complete, data verified", "Database migrated.", "database migration", "all rows verified, rollback tested"),
    ("Rate limiting works — tested with 1000 RPS", "Rate limits active.", "rate limiting", "token bucket at 100/min per IP"),
    ("Auth implemented with JWT", "Authentication live.", "authentication", "JWT with refresh tokens, 15min expiry"),
    ("WebSocket connected, messages flowing", "WebSocket operational.", "WebSocket", "bi-directional messaging working"),
    ("Monitoring dashboards ready", "Observability set up.", "monitoring", "Grafana dashboards + PagerDuty alerts configured"),
]

for user, response, task, detail in completions:
    add(user, f"{response}\n\n{status_block(task, 'done', detail)}")

# Blocked tasks
blocked = [
    ("I'm stuck — the Q8_0 shader produces wrong values", "Let me help debug.", "Q8_0 shader", "blocked", "incorrect dequantization values"),
    ("Can't proceed — waiting for API keys", "Noted, pausing.", "API integration", "blocked", "waiting for credentials from team lead"),
    ("Build is broken on CI", "Let me check.", "CI pipeline", "blocked", "test failures in auth module"),
    ("Database is locked by another migration", "Need to wait.", "schema update", "blocked", "concurrent migration holding lock"),
]

for user, response, task, state, detail in blocked:
    add(user, f"{response}\n\n{status_block(task, state, detail)}")

# Step-by-step progression
add("Start working on step 1 of the caching plan",
    f"Starting with Redis client research.\n\n```status\nworkflow: add Redis caching\nstep: 1/5 — research Redis clients\nstate: active\ndetail: evaluating redis-rs vs deadpool-redis\n```")

add("Step 1 done — going with deadpool-redis. Moving to step 2",
    f"Good choice. Moving to dependency setup.\n\n```status\nworkflow: add Redis caching\nstep: 1/5 — research Redis clients\nstate: done\ndetail: selected deadpool-redis for async pool\n```\n\n```status\nworkflow: add Redis caching\nstep: 2/5 — add dependency\nstate: active\ndetail: adding to Cargo.toml\n```")

add("Step 2 done. Starting the cache-aside implementation",
    f"Core implementation next.\n\n```status\nworkflow: add Redis caching\nstep: 2/5 — add dependency\nstate: done\ndetail: deadpool-redis 0.14 added\n```\n\n```status\nworkflow: add Redis caching\nstep: 3/5 — implement cache-aside\nstate: active\ndetail: wrapping /v1/infer with cache check\n```")

# ═══════════════════════════════════════════════════════════════
# 3. PLAN CREATION — 20+ examples
# ═══════════════════════════════════════════════════════════════

plans = [
    ("Plan how to add Redis caching", "add Redis caching",
     ["Research Redis client libraries", "Add dependency to Cargo.toml", "Implement cache-aside for /v1/infer",
      "Add cache invalidation on /v1/insert", "Write integration tests", "Benchmark cache hit/miss"]),
    ("Plan the MoE model integration", "MoE integration",
     ["Parse GGUF expert tensors", "Add router matvec dispatch", "Implement top-K selection",
      "Per-expert Q4_K matvec", "Wire shared expert", "Benchmark vs Ollama"]),
    ("Plan how to set up CI/CD", "setup CI/CD",
     ["Create Dockerfile", "Write GitHub Actions workflow", "Add lint + test stage",
      "Add build + release stage", "Configure deployment to staging", "Add production deploy gate"]),
    ("Break down the auth system implementation", "implement authentication",
     ["Choose auth strategy (JWT vs session)", "Add user model + DB migration",
      "Implement login/register endpoints", "Add JWT middleware", "Add refresh token flow",
      "Write auth integration tests"]),
    ("Plan database performance optimization", "optimize database",
     ["Profile slow queries with EXPLAIN ANALYZE", "Add missing indexes",
      "Implement connection pooling", "Add query result caching", "Benchmark before/after"]),
    ("Plan the API versioning strategy", "API versioning",
     ["Audit current endpoints", "Design v2 URL scheme", "Add version router middleware",
      "Migrate clients to v2", "Deprecate v1 endpoints"]),
    ("Plan how to add WebSocket support", "add WebSocket",
     ["Choose WebSocket library (tokio-tungstenite)", "Implement connection handler",
      "Add message routing", "Handle reconnection and heartbeat", "Write load tests"]),
    ("Create a plan for the mobile app", "mobile app MVP",
     ["Design wireframes", "Set up React Native project", "Implement auth flow",
      "Build main feed screen", "Add push notifications", "Beta test with 10 users"]),
    ("Plan how to reduce memory usage", "reduce memory",
     ["Profile with vmmap/heaptrack", "Identify largest allocations",
      "Implement lazy loading for weights", "Add madvise for unused mmaps",
      "Measure RSS before/after"]),
    ("Plan the monitoring setup", "setup monitoring",
     ["Install Prometheus", "Add metric endpoints to server", "Create Grafana dashboards",
      "Set up alerting rules", "Test alert notifications", "Document runbooks"]),
]

for user, name, steps in plans:
    add(user, f"Here's the plan:\n\n{plan_block(name, steps)}")

# Short plans (3 steps)
short_plans = [
    ("Plan how to fix the bug", "fix the bug", ["Reproduce the issue", "Identify root cause", "Implement and test fix"]),
    ("Plan the code review", "code review", ["Read the diff", "Test locally", "Leave review comments"]),
    ("Plan the release", "release v1.0", ["Run full test suite", "Update changelog", "Tag and deploy"]),
    ("Plan how to learn Rust", "learn Rust", ["Read The Book chapters 1-10", "Build a CLI tool", "Contribute to an open source project"]),
    ("Plan the meeting", "team meeting", ["Prepare agenda", "Present updates", "Assign action items"]),
]

for user, name, steps in short_plans:
    add(user, f"Plan:\n\n{plan_block(name, steps)}")

# ═══════════════════════════════════════════════════════════════
# 4. TOOL USAGE — 20+ examples
# ═══════════════════════════════════════════════════════════════

tool_calls = [
    ("List the files here", "Listing current directory.", "list ."),
    ("Show me what's in src/", "Checking src/.", "list src/"),
    ("List the test files", "Looking at tests.", "list tests/"),
    ("What's in the home directory?", "Checking home.", "list ~/"),
    ("Show the crates directory", "Listing crates.", "list crates/"),
    ("List files in /tmp", "Checking /tmp.", "list /tmp"),
    ("What do you know about me?", "Checking stored facts.", "facts"),
    ("Show me the stored facts", "Retrieving knowledge.", "facts"),
    ("What facts have been stored?", "Let me look.", "facts"),
    ("Recall what you know", "Checking memory.", "facts"),
    ("Search for files containing 'TODO'", "Searching.", "search TODO"),
    ("Find all Rust files", "Searching for .rs files.", "search *.rs"),
    ("Show the git status", "Checking git.", "git status"),
    ("What branch are we on?", "Checking branch.", "git branch"),
    ("Show recent commits", "Recent history.", "git log"),
    ("Show disk usage", "Checking disk.", "du ."),
    ("How much space does src/ use?", "Checking src size.", "du src/"),
    ("Run the tests", "Running test suite.", "run cargo test"),
    ("Build the project", "Building.", "run cargo build --release"),
    ("Check for compiler warnings", "Checking.", "run cargo check"),
]

for user, response, cmd in tool_calls:
    add(user, f"{response}\n\n{tool_block(cmd)}")

# ═══════════════════════════════════════════════════════════════
# 5. CONCISE ANSWERS — 40+ examples
# ═══════════════════════════════════════════════════════════════

concise = [
    ("What is 2+2?", "4"),
    ("What is the capital of France?", "Paris."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("What port does HTTPS use?", "443."),
    ("How many bits in a byte?", "8."),
    ("What is TCP?", "Transmission Control Protocol — reliable, ordered, connection-oriented transport."),
    ("What's a mutex?", "A mutual exclusion lock ensuring only one thread accesses a resource at a time."),
    ("Explain CORS", "Cross-Origin Resource Sharing — HTTP headers controlling cross-origin access."),
    ("What is WASM?", "WebAssembly — binary format for near-native performance in browsers."),
    ("What does FIFO mean?", "First In, First Out."),
    ("What is a B-tree?", "Self-balancing tree optimized for disk access, used in databases."),
    ("Define idempotent", "An operation producing the same result regardless of repetition count."),
    ("What is gRPC?", "High-performance RPC using Protocol Buffers over HTTP/2."),
    ("What's the CAP theorem?", "A distributed system can guarantee at most two of: Consistency, Availability, Partition tolerance."),
    ("What is a LoRA adapter?", "Low-rank matrices (A, B) added to frozen weights: W' = W + BA."),
    ("What is RMSNorm?", "Root Mean Square Normalization — divides by RMS without mean subtraction."),
    ("What is attention?", "Mechanism computing weighted sums of values based on query-key similarity."),
    ("What's a transformer?", "Neural architecture using self-attention + FFN layers for sequence modeling."),
    ("Define latency", "Time between request and response."),
    ("What is throughput?", "Number of operations completed per unit time."),
    ("What's the Big-O of quicksort?", "O(n log n) average, O(n²) worst case."),
    ("What is a hash map?", "Key-value store with O(1) average lookup using hash functions."),
    ("What is Docker?", "Container runtime that packages applications with their dependencies."),
    ("What's Kubernetes?", "Container orchestration platform for automated deployment and scaling."),
    ("What is REST?", "Representational State Transfer — stateless API design using HTTP methods."),
    ("What is GraphQL?", "Query language for APIs allowing clients to request exactly the data they need."),
    ("What is a monad?", "A type with bind and return operations that chains computations with context."),
    ("What is a closure?", "A function capturing variables from its enclosing scope."),
    ("What is GC?", "Garbage Collection — automatic memory reclamation of unreachable objects."),
    ("What is RAII?", "Resource Acquisition Is Initialization — ties resource lifetime to object scope."),
]

for q, a in concise:
    add(q, a)

# ═══════════════════════════════════════════════════════════════
# 6. COMBINED — fact + status/plan in one response (20+)
# ═══════════════════════════════════════════════════════════════

add("The server runs on port 8080. Can you check if it's healthy?",
    f"Checking server health on port 8080.\n\n{fact_block('server port', '8080')}\n\n{status_block('health check', 'active', 'testing connectivity')}")

add("We use PostgreSQL 15. Please optimize the slow queries.",
    f"I'll optimize the queries.\n\n{fact_block('database', 'PostgreSQL 15')}\n\n{status_block('optimize queries', 'active', 'running EXPLAIN ANALYZE')}")

add("I'm using Python 3.12. Help me set up a virtual environment.",
    f"Setting up the venv.\n\n{fact_block('python version', '3.12')}\n\n{status_block('setup venv', 'active', 'creating venv with Python 3.12')}")

add("The API endpoint is /v1/predict. It's returning 500 errors.",
    f"I'll investigate the 500s.\n\n{fact_block('endpoint', '/v1/predict')}\n\n{status_block('debug 500 errors', 'active', 'checking server logs')}")

add("My project uses Rust and Metal GPU. Plan how to add a new shader.",
    f"Here's the plan for the new shader:\n\n{fact_block('language', 'Rust')}\n\n{fact_block('compute', 'Metal GPU')}\n\n{plan_block('add new shader', ['Design shader algorithm', 'Write MSL code', 'Register pipeline in MetalBackend', 'Write correctness test', 'Benchmark performance'])}")

add("We're at 42 tok/s. I need to reach 60 tok/s. Plan the optimization.",
    f"Plan to reach 60 tok/s:\n\n{fact_block('current speed', '42 tok/s')}\n\n{fact_block('target speed', '60 tok/s')}\n\n{plan_block('reach 60 tok/s', ['Profile current bottlenecks', 'Fuse norm+matvec dispatches', 'Tune Q4_K shader NR0 parameter', 'Reduce GPU sync points', 'Benchmark each change'])}")

# Fact + tool
add("I need to check what files changed. The repo is on GitHub.",
    f"Checking changes.\n\n{fact_block('vcs', 'GitHub')}\n\n{tool_block('git status')}")

add("The project is in ~/projects/app. List the source files.",
    f"Listing source files.\n\n{fact_block('project path', '~/projects/app')}\n\n{tool_block('list ~/projects/app/src/')}")

# ═══════════════════════════════════════════════════════════════
# 7. FACT RETRIEVAL patterns (model shows it remembers)
# ═══════════════════════════════════════════════════════════════

# These teach the model that stored facts should be referenced
add("What's my name again?",
    "Based on the stored facts, your name is Miguel.\n\n(Retrieved from stored facts)")

add("What language do I prefer?",
    "You prefer Rust, based on what you told me earlier.\n\n(Retrieved from stored facts)")

add("What port is the server on?",
    "The server runs on port 3000, based on the stored facts.\n\n(Retrieved from stored facts)")

add("Remind me what project I'm working on",
    "You're working on larql — an LLM-as-a-database system.\n\n(Retrieved from stored facts)")

add("What's my setup?",
    "Based on stored facts: you're running Gemma 3 4B on M4 Pro with Metal GPU, getting 42 tok/s.\n\n(Retrieved from stored facts)")

# ═══════════════════════════════════════════════════════════════
# 8. Messages that should NOT produce annotations
# ═══════════════════════════════════════════════════════════════

# Pure greetings — no facts to extract
no_annotation = [
    ("Hello!", "Hello! How can I help you today?"),
    ("Thanks!", "You're welcome!"),
    ("Good morning", "Good morning! What are we working on today?"),
    ("OK", "Ready when you are."),
    ("Got it", "Great, let me know if you need anything else."),
    ("Never mind", "No problem. Let me know when you're ready."),
    ("Yes", "Understood. Proceeding."),
    ("No", "Alright, we'll skip that."),
    ("Please continue", "Continuing."),
    ("That's all for now", "Sounds good. I'll be here when you need me."),
]

for q, a in no_annotation:
    add(q, a)

# ═══════════════════════════════════════════════════════════════
# Shuffle and split
# ═══════════════════════════════════════════════════════════════

random.shuffle(examples)
split = int(len(examples) * 0.9)
train = examples[:split]
valid = examples[split:]

with open("scripts/finetune/data/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open("scripts/finetune/data/valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(ex) + "\n")

# Stats
cats = {"fact": 0, "status": 0, "plan": 0, "tool": 0, "concise": 0,
        "combined": 0, "retrieval": 0, "no_annotation": 0}
for ex in examples:
    c = ex["messages"][-1]["content"]
    has_fact = "```fact" in c
    has_status = "```status" in c
    has_plan = "```plan" in c
    has_tool = "```tool" in c
    is_retrieval = "Retrieved from stored facts" in c

    if has_fact and (has_status or has_plan or has_tool): cats["combined"] += 1
    elif has_fact: cats["fact"] += 1
    elif has_status: cats["status"] += 1
    elif has_plan: cats["plan"] += 1
    elif has_tool: cats["tool"] += 1
    elif is_retrieval: cats["retrieval"] += 1
    elif any(c == ex["messages"][-1]["content"] for _, c in no_annotation):
        cats["no_annotation"] += 1
    else: cats["concise"] += 1

print(f"Total: {len(examples)} examples")
print(f"  Train: {len(train)}")
print(f"  Valid: {len(valid)}")
print(f"  Categories: {cats}")
