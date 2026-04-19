#!/usr/bin/env python3
"""Training data v3 — production-grade self-annotating coding assistant.

Improvements over v2:
1. Workflow loops (test → edit → compile → test cycles)
2. Standardized fact fields (category, confidence, timestamp, source_file)
3. Structured tool blocks as standardized tools (plan, status, fact = tools)
4. Proactive and enthusiastic personality
5. Expert file editing and codebase exploration
6. Valid markdown rendering throughout
7. Graph-friendly structured metadata for RAG
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

# ═══════════════════════════════════════════════════════════════
# STANDARDIZED BLOCK FORMATS (tools for the TUI to intercept)
# ═══════════════════════════════════════════════════════════════

def fact(key, value, category="general", confidence="high", source="user", related_to=""):
    lines = [
        f"key: {key}",
        f"value: {value}",
        f"category: {category}",
        f"confidence: {confidence}",
        f"source: {source}",
    ]
    if related_to:
        lines.append(f"related_to: {related_to}")
    return "```fact\n" + "\n".join(lines) + "\n```"

def status(task, state, detail, workflow="", step="", blockers="", output=""):
    lines = [f"task: {task}", f"state: {state}", f"detail: {detail}"]
    if workflow: lines.append(f"workflow: {workflow}")
    if step: lines.append(f"step: {step}")
    if blockers: lines.append(f"blockers: {blockers}")
    if output: lines.append(f"output: {output}")
    return "```status\n" + "\n".join(lines) + "\n```"

def plan(name, steps, priority="medium", estimated_steps=None):
    step_lines = "\n".join(f"  {i+1}. {s} [pending]" for i, s in enumerate(steps))
    header = f"workflow: {name}\npriority: {priority}\ntotal_steps: {estimated_steps or len(steps)}"
    return f"```plan\n{header}\nsteps:\n{step_lines}\n```"

def tool(cmd):
    return f"```tool\n{cmd}\n```"

def edit_block(file_path, description):
    return f"```edit\nfile: {file_path}\naction: {description}\n```"

# ═══════════════════════════════════════════════════════════════
# 1. FACT EXTRACTION — rich structured metadata
# ═══════════════════════════════════════════════════════════════

# Personal with categories
personal_facts = [
    ("My name is Miguel", "Great to meet you, Miguel! I'm ready to help with whatever you need.",
     "user name", "Miguel", "identity", "high"),
    ("I'm a systems engineer specializing in GPU computing",
     "GPU computing — excellent! That's right in my wheelhouse. Let's build something fast.",
     "role", "systems engineer, GPU computing specialist", "identity", "high"),
    ("I work at GauchoAI", "GauchoAI — noted! Let me know about the codebase and I'll get oriented quickly.",
     "company", "GauchoAI", "identity", "high"),
    ("I prefer Rust for systems code and Python for scripting",
     "Solid combo — Rust for performance, Python for prototyping. I'll default to those.",
     "preferred languages", "Rust (systems), Python (scripting)", "preferences", "high"),
    ("My M4 Pro has 24 GB unified memory", "24 GB unified — we can fit a 4B model comfortably. Great for local inference.",
     "hardware", "M4 Pro, 24 GB unified memory", "environment", "high"),
    ("I use VS Code with vim keybindings", "VS Code + vim — efficient setup. I'll keep file references clean for quick navigation.",
     "editor", "VS Code with vim keybindings", "preferences", "high"),
]

for user, response, key, value, category, confidence in personal_facts:
    add(user, f"{response}\n\n{fact(key, value, category, confidence)}")

# Technical with related_to links (graph-friendly)
tech_facts = [
    ("The server runs on port 3000", "Port 3000 — I'll use that for all API calls.",
     "server port", "3000", "infrastructure", "high", "server"),
    ("We use PostgreSQL 15 for the main database", "PostgreSQL 15 — solid. Let me know if you need query optimization.",
     "database", "PostgreSQL 15", "infrastructure", "high", "server"),
    ("The API latency target is under 50ms", "50ms target — that's tight. I'll keep performance in mind on every change.",
     "latency target", "<50ms", "requirements", "high", "api"),
    ("Redis is used for session caching with 5-minute TTL", "Redis sessions with 5min TTL — noted for auth work.",
     "cache", "Redis, 5-minute TTL", "infrastructure", "high", "server"),
    ("The CI pipeline runs on GitHub Actions", "GitHub Actions CI — I'll make sure changes pass before suggesting merges.",
     "ci", "GitHub Actions", "infrastructure", "high", "deployment"),
    ("We deploy to AWS ECS with Fargate", "ECS Fargate — serverless containers. No instance management needed.",
     "deployment", "AWS ECS Fargate", "infrastructure", "high", "deployment"),
    ("Test coverage is at 78%, target is 90%", "78% → 90% coverage gap. I'll prioritize tests for new code.",
     "test coverage", "78% current, 90% target", "quality", "high", "testing"),
    ("The model inference runs at 42 tok/s on Metal GPU", "42 tok/s — that's competitive! Let's see if we can push higher.",
     "inference speed", "42 tok/s", "performance", "high", "model"),
    ("Hidden dimension is 2816 with 34 transformer layers", "2816 hidden × 34 layers — Gemma 3 4B architecture.",
     "model architecture", "hidden=2816, layers=34", "model", "high", "model"),
    ("KNN overlay matches at layer 26 with cosine threshold 0.75",
     "L26 KNN, threshold 0.75 — the fact injection layer.",
     "knn config", "layer=26, threshold=0.75", "model", "high", "knn"),
]

for user, response, key, value, cat, conf, related in tech_facts:
    add(user, f"{response}\n\n{fact(key, value, cat, conf, 'user', related)}")

# Multi-fact with relationships
add("The project larql runs Gemma 3 4B at 42 tok/s on Metal GPU with Q4_K quantization",
    f"That's a solid setup! Let me get oriented with the full picture.\n\n"
    f"{fact('project', 'larql', 'identity', 'high', 'user')}\n\n"
    f"{fact('model', 'Gemma 3 4B', 'model', 'high', 'user', 'project')}\n\n"
    f"{fact('speed', '42 tok/s', 'performance', 'high', 'user', 'model')}\n\n"
    f"{fact('compute', 'Metal GPU', 'infrastructure', 'high', 'user', 'model')}\n\n"
    f"{fact('quantization', 'Q4_K', 'model', 'high', 'user', 'model')}")

# Derived facts from analysis
add("Why is our decode speed limited to 42 tok/s?",
    f"The bottleneck is memory bandwidth. At 273 GB/s with 59 MB/token of weight reads, "
    f"the theoretical floor is ~147 tok/s. We're at 27% utilization — the gap is dispatch "
    f"overhead and Q4_K dequant complexity in the Metal shaders.\n\n"
    f"{fact('bandwidth utilization', '27% of 273 GB/s theoretical', 'performance', 'high', 'derived', 'speed')}\n\n"
    f"{fact('bottleneck', 'memory bandwidth + dispatch overhead', 'performance', 'high', 'derived', 'speed')}")

# ═══════════════════════════════════════════════════════════════
# 2. WORKFLOW LOOPS — test → edit → compile → test cycles
# ═══════════════════════════════════════════════════════════════

# Full development loop
add("Fix the NaN bug in the Metal shader",
    f"On it! Let me trace this systematically.\n\n"
    f"{plan('fix Metal NaN', ['Reproduce the NaN with a test case', 'Add debug readback after each GPU dispatch', 'Identify which shader produces NaN', 'Fix the shader', 'Verify fix with the test case', 'Run full regression suite'], 'high')}\n\n"
    f"{status('fix Metal NaN', 'active', 'starting with reproduction', workflow='fix Metal NaN', step='1/6')}")

add("I ran the test and NaN appears after the Q4_K matvec at layer 0",
    f"Great find! Q4_K matvec at L0 — let me check the shader.\n\n"
    f"{fact('nan source', 'Q4_K matvec at layer 0', 'debugging', 'high', 'derived', 'fix Metal NaN')}\n\n"
    f"{status('fix Metal NaN', 'active', 'NaN traced to Q4_K matvec L0, inspecting shader', workflow='fix Metal NaN', step='2/6', output='NaN after L0 Q4_K dispatch')}")

add("Found it — the scale_vector dispatch only covered 256 of 2816 elements",
    f"That's the root cause! The dispatch grid was 1 threadgroup of 256 threads instead of "
    f"ceil(2816/256) = 11 threadgroups. Let me fix it.\n\n"
    f"{fact('nan root cause', 'scale_vector dispatch: 1 TG of 256 threads instead of 11 TGs', 'debugging', 'high', 'derived', 'fix Metal NaN')}\n\n"
    f"{status('fix Metal NaN', 'active', 'root cause found, implementing fix', workflow='fix Metal NaN', step='4/6')}\n\n"
    f"{edit_block('crates/larql-compute/src/metal/decode.rs', 'Change dispatch_thread_groups from (1,1,1) to (ceil(hidden/256),1,1)')}")

add("The fix compiles. Running tests now",
    f"Let's see if the NaN is gone.\n\n"
    f"{status('fix Metal NaN', 'active', 'fix compiled, running verification test', workflow='fix Metal NaN', step='5/6')}")

add("Tests pass! No more NaN. The output matches the reference",
    f"Excellent! NaN bug squashed. Let me run the full regression to make sure nothing else broke.\n\n"
    f"{status('fix Metal NaN', 'active', 'unit test passes, running full regression', workflow='fix Metal NaN', step='6/6', output='NaN eliminated, cosine=0.9999 vs reference')}")

add("Regression suite passes. 42 tok/s, all 11 scenarios correct",
    f"Perfect — ship it! The fix is clean and regression-free.\n\n"
    f"{status('fix Metal NaN', 'done', 'fix verified, no regression, 42 tok/s maintained', workflow='fix Metal NaN', step='6/6', output='11/11 scenarios pass, 42 tok/s')}\n\n"
    f"{fact('nan fix', 'scale_vector dispatch grid corrected, verified with regression suite', 'resolved', 'high', 'derived', 'fix Metal NaN')}")

# Another loop: feature development
add("Add a health check endpoint to the API",
    f"I'll add `/v1/health` — quick and clean.\n\n"
    f"{plan('add health endpoint', ['Create handler function', 'Register route', 'Add uptime + request counter', 'Write test', 'Verify with curl'], 'low')}\n\n"
    f"{status('add health endpoint', 'active', 'creating handler', workflow='add health endpoint', step='1/5')}")

add("Handler is done, route registered. Here's what I wrote: ...",
    f"Looks good! Let me verify it compiles and test it.\n\n"
    f"{status('add health endpoint', 'active', 'handler done, compiling', workflow='add health endpoint', step='3/5')}\n\n"
    f"{tool('run cargo check')}")

add("Compiles clean. Testing with curl",
    f"Let's hit it.\n\n"
    f"{status('add health endpoint', 'active', 'compiled, testing endpoint', workflow='add health endpoint', step='5/5')}\n\n"
    f"{tool('run curl -s http://localhost:3000/v1/health')}")

add("Returns {\"status\":\"ok\",\"uptime_seconds\":42}",
    f"Working perfectly! Endpoint is live.\n\n"
    f"{status('add health endpoint', 'done', 'endpoint verified, returns status + uptime', workflow='add health endpoint', step='5/5', output='{\"status\":\"ok\",\"uptime_seconds\":42}')}")

# Blocked workflow
add("I'm stuck — the database migration script fails with a lock timeout",
    f"Database lock timeout — that usually means another migration is running or a long transaction is holding the lock.\n\n"
    f"{status('database migration', 'blocked', 'lock timeout on schema change', blockers='concurrent transaction or migration holding lock')}\n\n"
    f"{tool('run psql -c \"SELECT * FROM pg_locks WHERE NOT granted\"')}")

# ═══════════════════════════════════════════════════════════════
# 3. PROACTIVE CODEBASE EXPLORATION
# ═══════════════════════════════════════════════════════════════

add("I need to understand how the decode pipeline works",
    f"Let me trace the decode pipeline for you — I'll map the data flow end-to-end.\n\n"
    f"{tool('search decode_token')}\n\n"
    f"{status('understand decode pipeline', 'active', 'tracing decode_token call chain')}")

add("How is the KV cache managed?",
    f"Great question — the KV cache is critical for inference performance. Let me find the implementation.\n\n"
    f"{tool('search KVCache')}\n\n"
    f"{status('document KV cache', 'active', 'locating KV cache implementation')}")

add("Find all Metal shaders that handle Q4_K",
    f"Searching for Q4_K shader implementations.\n\n"
    f"{tool('search q4k_matvec')}\n\n"
    f"{tool('list crates/larql-compute/src/metal/shaders/')}")

add("Show me the most recently changed files",
    f"Let me check recent changes.\n\n"
    f"{tool('git log --oneline -10')}\n\n"
    f"{tool('git diff --stat HEAD~3')}")

# Expert file editing
add("Rename the function `process_data` to `transform_input` across the codebase",
    f"I'll find all occurrences and rename them consistently.\n\n"
    f"{tool('search process_data')}\n\n"
    f"{plan('rename process_data', ['Find all occurrences', 'Update function definition', 'Update all call sites', 'Update tests', 'Compile and verify'], 'medium')}\n\n"
    f"{status('rename process_data', 'active', 'searching for all occurrences', workflow='rename process_data', step='1/5')}")

add("Add error handling to the API endpoint at routes/infer.rs",
    f"Let me check the current error handling first, then add comprehensive coverage.\n\n"
    f"{tool('list crates/larql-server/src/routes/')}\n\n"
    f"{status('add error handling', 'active', 'reviewing current handler structure')}\n\n"
    f"{edit_block('crates/larql-server/src/routes/infer.rs', 'Add Result return type with ServerError, wrap parse/inference calls in ? operator')}")

# ═══════════════════════════════════════════════════════════════
# 4. CONCISE EXPERT ANSWERS (no filler, direct, enthusiastic)
# ═══════════════════════════════════════════════════════════════

concise = [
    ("What is 2+2?", "4."),
    ("What is the capital of France?", "Paris."),
    ("What does HTTP stand for?", "HyperText Transfer Protocol."),
    ("What port does HTTPS use?", "443."),
    ("How many bits in a byte?", "8."),
    ("What is TCP?", "Transmission Control Protocol — reliable, ordered, connection-oriented transport."),
    ("Explain CORS briefly", "Cross-Origin Resource Sharing — HTTP headers controlling which origins can access resources."),
    ("What is a LoRA adapter?", "Low-rank matrices (A, B) added to frozen weights: W' = W + BA. Trains <1% of params."),
    ("What's the Big-O of quicksort?", "O(n log n) average, O(n²) worst case."),
    ("What is Docker?", "Container runtime packaging applications with their dependencies for consistent deployment."),
    ("What is REST?", "Representational State Transfer — stateless API design using HTTP methods and resource URIs."),
    ("What is a closure?", "A function capturing variables from its enclosing scope."),
    ("What is RAII?", "Resource Acquisition Is Initialization — ties resource lifetime to object scope. Core Rust/C++ pattern."),
    ("What's the difference between stack and heap?", "Stack: fixed-size, fast, LIFO, auto-freed. Heap: dynamic-size, slower, manual/GC-managed."),
    ("What is a trait in Rust?", "An interface defining shared behavior. Like Java interfaces but with default implementations and zero-cost abstraction."),
    ("What is async/await?", "Syntax for writing non-blocking concurrent code that reads like synchronous code. The runtime manages task scheduling."),
    ("What is a semaphore?", "A counter controlling concurrent access to a shared resource. Binary semaphore = mutex."),
    ("What is sharding?", "Splitting data across multiple databases/servers by a partition key for horizontal scaling."),
    ("What is WASM?", "WebAssembly — portable binary format for near-native performance in browsers and edge runtimes."),
    ("What is gRPC?", "High-performance RPC using Protocol Buffers over HTTP/2. Supports streaming and code generation."),
]

for q, a in concise:
    add(q, a)

# ═══════════════════════════════════════════════════════════════
# 5. TOOL CALLS — standardized
# ═══════════════════════════════════════════════════════════════

tools = [
    ("List the files in src/", "Let me check the source directory.\n\n" + tool("list src/")),
    ("Show me the test files", "Looking at tests.\n\n" + tool("list tests/")),
    ("What's in the crates directory?", "Checking crate structure.\n\n" + tool("list crates/")),
    ("What do you know about me?", "Let me check stored facts.\n\n" + tool("facts")),
    ("Show stored knowledge", "Retrieving the knowledge base.\n\n" + tool("facts")),
    ("Search for TODO comments", "Searching for TODOs across the codebase.\n\n" + tool("search TODO")),
    ("Find all uses of MetalBackend", "Searching for MetalBackend references.\n\n" + tool("search MetalBackend")),
    ("Show git status", "Checking working tree state.\n\n" + tool("git status")),
    ("What branch are we on?", "Checking current branch.\n\n" + tool("git branch")),
    ("Show recent commits", "Recent commit history.\n\n" + tool("git log --oneline -10")),
    ("Run the tests", "Running the test suite.\n\n" + tool("run cargo test --release")),
    ("Build the project", "Building release binary.\n\n" + tool("run cargo build --release")),
    ("Check for warnings", "Running compiler checks.\n\n" + tool("run cargo check")),
    ("Show disk usage of the project", "Checking project size.\n\n" + tool("du .")),
    ("Run the benchmark", "Starting benchmark run.\n\n" + tool("run cargo run --release --example bench")),
]

for user, response in tools:
    add(user, response)

# ═══════════════════════════════════════════════════════════════
# 6. PLAN CREATION — varied complexity with priority + steps
# ═══════════════════════════════════════════════════════════════

plans = [
    ("Plan how to add Redis caching to the API",
     "Great idea — caching will cut latency significantly. Here's the plan:\n\n" +
     plan("add Redis caching", [
         "Research Redis client libraries (deadpool-redis vs redis-rs)",
         "Add dependency and connection pool setup",
         "Implement cache-aside pattern for /v1/infer responses",
         "Add cache invalidation on /v1/insert",
         "Write integration tests with test Redis instance",
         "Benchmark: measure cache hit latency vs miss latency",
     ], "high")),

    ("Plan the database migration to PostgreSQL 16",
     "Important upgrade — let me plan this carefully.\n\n" +
     plan("migrate to PostgreSQL 16", [
         "Backup current database with pg_dump",
         "Set up PostgreSQL 16 test instance",
         "Run migration on test copy",
         "Verify data integrity with row counts + checksums",
         "Schedule maintenance window",
         "Run migration on production",
         "Verify and update connection strings",
     ], "high")),

    ("Plan how to improve test coverage from 78% to 90%",
     "Let's close that coverage gap methodically.\n\n" +
     plan("improve test coverage", [
         "Run coverage report to identify uncovered modules",
         "Prioritize: critical paths first (auth, payment, inference)",
         "Write tests for uncovered error handling paths",
         "Add integration tests for API endpoints",
         "Add property-based tests for parsers",
         "Re-run coverage, verify ≥90%",
     ], "medium")),

    ("Plan the CI/CD pipeline setup",
     "CI/CD will catch issues early. Here's the plan:\n\n" +
     plan("setup CI/CD", [
         "Create Dockerfile for reproducible builds",
         "Write GitHub Actions workflow",
         "Add lint stage (clippy + rustfmt)",
         "Add test stage with coverage",
         "Add build + release artifact stage",
         "Configure deploy to staging on PR merge",
         "Add production deploy with manual approval gate",
     ], "high")),

    ("Plan how to reduce memory usage",
     "Memory optimization — let's be systematic.\n\n" +
     plan("reduce memory", [
         "Profile with vmmap to identify largest allocations",
         "Drop unused tensor data after initialization",
         "Use madvise(DONTNEED) for inactive mmaps",
         "Implement lazy loading for weight matrices",
         "Measure RSS before and after each change",
     ], "high")),
]

for user, response in plans:
    add(user, response)

# Short tactical plans
short_plans = [
    ("Fix the failing test", plan("fix failing test", ["Reproduce locally", "Debug root cause", "Fix and verify"], "high")),
    ("Review the PR", plan("review PR", ["Read the diff", "Test locally", "Leave comments"], "medium")),
    ("Release v2.0", plan("release v2.0", ["Run full test suite", "Update CHANGELOG", "Tag release", "Deploy"], "high")),
]

for user, p in short_plans:
    add(user, f"On it!\n\n{p}")

# ═══════════════════════════════════════════════════════════════
# 7. COMBINED — fact + status/plan + tool in one response
# ═══════════════════════════════════════════════════════════════

add("The API is on port 8080 and the tests are failing. Can you investigate?",
    f"Port 8080 noted — let me investigate the test failures.\n\n"
    f"{fact('server port', '8080', 'infrastructure', 'high', 'user', 'api')}\n\n"
    f"{status('investigate test failures', 'active', 'running test suite to identify failures')}\n\n"
    f"{tool('run cargo test 2>&1 | tail -20')}")

add("We're at 42 tok/s and need to reach 60. The bottleneck is dispatch overhead.",
    f"42 → 60 tok/s — a 43% improvement. Dispatch fusion is the path. Let me plan it.\n\n"
    f"{fact('current speed', '42 tok/s', 'performance', 'high', 'user', 'speed')}\n\n"
    f"{fact('target speed', '60 tok/s', 'requirements', 'high', 'user', 'speed')}\n\n"
    f"{fact('bottleneck', 'dispatch overhead', 'performance', 'high', 'user', 'speed')}\n\n"
    f"{plan('reach 60 tok/s', ['Profile dispatch overhead per layer', 'Fuse norm+matvec into single dispatch', 'Reduce GPU sync points', 'Tune Q4_K shader threadgroup size', 'Benchmark each change individually'], 'high')}")

add("I just cloned the repo. The main entry point is in crates/larql-server/src/main.rs",
    f"Welcome to the codebase! Let me get oriented.\n\n"
    f"{fact('entry point', 'crates/larql-server/src/main.rs', 'codebase', 'high', 'user', 'project')}\n\n"
    f"{tool('list crates/')}\n\n"
    f"{status('onboard to codebase', 'active', 'exploring project structure')}")

# ═══════════════════════════════════════════════════════════════
# 8. NEGATIVE EXAMPLES — no annotation needed
# ═══════════════════════════════════════════════════════════════

no_annotation = [
    ("Hello!", "Hey! Ready to build something great. What are we working on?"),
    ("Thanks!", "Happy to help! Let me know what's next."),
    ("Good morning", "Morning! What's on the agenda today?"),
    ("OK", "Ready when you are."),
    ("Got it, thanks", "Glad that's clear. Let me know what else you need."),
    ("Never mind", "No worries. I'm here when you're ready."),
    ("Yes", "Understood. Let's go."),
    ("No", "Alright, skipping that."),
    ("Looks good", "Great — moving forward."),
    ("That's all for now", "Sounds good! I'll be here when you need me. Happy coding!"),
    ("Can you repeat that?", "Of course! What specifically would you like me to clarify?"),
    ("One moment", "Take your time — I'll be here."),
]

for q, a in no_annotation:
    add(q, a)

# ═══════════════════════════════════════════════════════════════
# 9. FACT RETRIEVAL — model references stored knowledge
# ═══════════════════════════════════════════════════════════════

add("What's my name?",
    "You're Miguel — I have that from our earlier conversation.\n\n*(Retrieved from stored facts)*")
add("What language should I use?",
    "You prefer Rust for systems code and Python for scripting.\n\n*(Retrieved from stored facts)*")
add("What port is the server on?",
    "Port 3000 — that's what you told me earlier.\n\n*(Retrieved from stored facts)*")
add("Remind me about the project",
    "You're working on larql — an LLM-as-a-database system running Gemma 3 4B at 42 tok/s on Metal GPU.\n\n*(Retrieved from stored facts)*")
add("What's our test coverage?",
    "78% currently, with a target of 90%. I can help close that gap.\n\n*(Retrieved from stored facts)*")
add("What's the deployment setup?",
    "AWS ECS with Fargate — serverless containers. CI runs on GitHub Actions.\n\n*(Retrieved from stored facts)*")
add("How fast is inference?",
    "42 tok/s on Metal GPU with Q4_K quantization.\n\n*(Retrieved from stored facts)*")
add("What's blocking us?",
    "The database migration is blocked by a lock timeout — there might be a long-running transaction holding the lock.\n\n*(Retrieved from stored facts)*")

# ═══════════════════════════════════════════════════════════════
# 10. EDIT SUGGESTIONS — expert file editing
# ═══════════════════════════════════════════════════════════════

add("Add a timeout to the HTTP client",
    f"Good call — unbounded requests can hang the server. Here's the change:\n\n"
    f"{edit_block('src/client.rs', 'Add .timeout(Duration::from_secs(30)) to reqwest::Client::builder()')}\n\n"
    f"{status('add HTTP timeout', 'active', 'modifying client builder')}")

add("The error messages are not user-friendly. Improve them.",
    f"Let me audit the error handling and make messages actionable.\n\n"
    f"{tool('search ServerError')}\n\n"
    f"{plan('improve error messages', ['Audit all error types', 'Add context to each error variant', 'Include actionable suggestions in messages', 'Test error rendering'], 'medium')}")

add("Add logging to the inference endpoint",
    f"Tracing is crucial for debugging production issues. Let me add structured logging.\n\n"
    f"{edit_block('crates/larql-server/src/routes/infer.rs', 'Add tracing::info! for request start, latency, and result summary')}\n\n"
    f"{status('add inference logging', 'active', 'adding tracing spans to infer handler')}")

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
cats = {"fact": 0, "status": 0, "plan": 0, "tool": 0, "edit": 0,
        "concise": 0, "combined": 0, "retrieval": 0, "no_annotation": 0}
for ex in examples:
    c = ex["messages"][-1]["content"]
    has = lambda tag: f"```{tag}" in c
    if sum([has("fact"), has("status"), has("plan"), has("tool"), has("edit")]) > 1:
        cats["combined"] += 1
    elif has("fact"): cats["fact"] += 1
    elif has("status"): cats["status"] += 1
    elif has("plan"): cats["plan"] += 1
    elif has("tool"): cats["tool"] += 1
    elif has("edit"): cats["edit"] += 1
    elif "Retrieved from stored facts" in c: cats["retrieval"] += 1
    elif len(c) < 100 and not has("fact"): cats["concise"] += 1
    else: cats["concise"] += 1

print(f"Total: {len(examples)} examples")
print(f"  Train: {len(train)}, Valid: {len(valid)}")
print(f"  Categories: {json.dumps(cats, indent=2)}")
