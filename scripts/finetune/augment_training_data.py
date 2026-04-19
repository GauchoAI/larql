#!/usr/bin/env python3
"""Augment training data to 200+ examples via templates."""

import json
import random

random.seed(42)

# Load existing
with open("scripts/finetune/data/train.jsonl") as f:
    train = [json.loads(l) for l in f]
with open("scripts/finetune/data/valid.jsonl") as f:
    valid = [json.loads(l) for l in f]

existing = train + valid

# ═══════════════════════════════════════════════════════════════
# Template-based augmentation
# ═══════════════════════════════════════════════════════════════

# Fact templates with diverse domains
fact_templates = [
    # Tech
    ("The framework is {val}", "{val} framework noted.", "framework", "{val}"),
    ("We use {val} for CI/CD", "{val} for CI — got it.", "ci/cd", "{val}"),
    ("The log level is set to {val}", "Log level {val}.", "log level", "{val}"),
    ("Memory usage is {val}", "{val} memory usage recorded.", "memory usage", "{val}"),
    ("The endpoint is {val}", "Endpoint {val} noted.", "endpoint", "{val}"),
    ("Response time is {val}", "{val} response time.", "response time", "{val}"),
    ("The cache TTL is {val}", "TTL set to {val}.", "cache ttl", "{val}"),
    ("The test coverage is {val}", "{val} coverage.", "test coverage", "{val}"),
    # Personal
    ("My timezone is {val}", "Timezone {val} noted.", "timezone", "{val}"),
    ("I speak {val}", "{val} — noted.", "language", "{val}"),
    ("My favorite editor is {val}", "{val} user — got it.", "editor", "{val}"),
    ("I use {val} as my OS", "{val} operating system.", "os", "{val}"),
]

values_pool = [
    "React", "Next.js", "Django", "Flask", "FastAPI", "Express",
    "GitHub Actions", "Jenkins", "CircleCI", "GitLab CI",
    "debug", "info", "warn", "error",
    "512 MB", "1.2 GB", "4 GB", "16 GB",
    "/api/v2/users", "/health", "/metrics", "/v1/predict",
    "50ms", "120ms", "3s", "800ms",
    "300 seconds", "1 hour", "5 minutes", "30 seconds",
    "87%", "95%", "72%", "100%",
    "UTC-3", "UTC+0", "UTC+9", "PST",
    "Portuguese", "English", "Spanish", "Japanese",
    "VS Code", "Neovim", "Helix", "Zed",
    "macOS", "Linux", "Ubuntu", "Arch Linux",
]

new_examples = []
for template, response_template, key, val_template in fact_templates:
    for val in random.sample(values_pool, min(3, len(values_pool))):
        user_msg = template.format(val=val)
        response = response_template.format(val=val)
        value = val_template.format(val=val)
        new_examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"""{response}

```fact
key: {key}
value: {value}
source: user
```"""}
            ]
        })

# Status templates
status_templates = [
    ("Look into the {issue}", "Investigating {issue}.", "investigate {issue}", "active", "starting analysis"),
    ("Fix {issue}", "Working on fixing {issue}.", "fix {issue}", "active", "identifying root cause"),
    ("Test {feature}", "Running tests for {feature}.", "test {feature}", "active", "executing test suite"),
    ("Deploy {component}", "Deploying {component}.", "deploy {component}", "active", "building and pushing"),
    ("Review {thing}", "Reviewing {thing}.", "review {thing}", "active", "reading through changes"),
    ("Update {thing}", "Updating {thing}.", "update {thing}", "active", "applying changes"),
]

issues = [
    "the memory leak", "the timeout error", "the 404 responses",
    "the slow query", "the broken auth", "the race condition",
    "the login flow", "the search feature", "the notification system",
    "the data pipeline", "the caching layer", "the API gateway",
    "the PR", "the documentation", "the test results",
]

for template, resp_template, task_template, state, detail in status_templates:
    for issue in random.sample(issues, 2):
        user_msg = template.format(issue=issue, feature=issue, component=issue, thing=issue)
        response = resp_template.format(issue=issue, feature=issue, component=issue, thing=issue)
        task = task_template.format(issue=issue, feature=issue, component=issue, thing=issue)
        new_examples.append({
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

# Plan templates
plan_templates = [
    ("Plan how to {task}", "Here's the plan for {task}:"),
    ("Break down {task} into steps", "Breaking down {task}:"),
    ("Create a plan for {task}", "Plan for {task}:"),
]

tasks_with_steps = [
    ("migrate to PostgreSQL 16", ["Backup existing database", "Update connection drivers", "Run schema migration", "Verify data integrity", "Update application config"]),
    ("add WebSocket support", ["Choose WebSocket library", "Implement server handler", "Add client connection manager", "Handle reconnection logic", "Write integration tests"]),
    ("implement rate limiting", ["Define rate limit rules", "Add token bucket algorithm", "Integrate middleware", "Add per-IP tracking", "Configure limits via env vars"]),
    ("set up monitoring", ["Install Prometheus", "Add metric endpoints", "Create Grafana dashboards", "Set up alerting rules", "Test alert notifications"]),
    ("build the CI pipeline", ["Create Dockerfile", "Write GitHub Actions workflow", "Add test stage", "Add build stage", "Add deploy stage", "Configure secrets"]),
]

for tmpl, resp_tmpl in plan_templates:
    for task, steps in random.sample(tasks_with_steps, 2):
        user_msg = tmpl.format(task=task)
        response = resp_tmpl.format(task=task)
        step_lines = "\n".join([f"  {i+1}. {s} [pending]" for i, s in enumerate(steps)])
        new_examples.append({
            "messages": [
                {"role": "user", "content": user_msg},
                {"role": "assistant", "content": f"""{response}

```plan
workflow: {task}
steps:
{step_lines}
```"""}
            ]
        })

# More concise Q&A (no annotation needed)
qa_pairs = [
    ("What is TCP?", "Transmission Control Protocol — reliable, ordered, connection-oriented transport layer protocol."),
    ("What's a mutex?", "A mutual exclusion primitive that ensures only one thread accesses a shared resource at a time."),
    ("Explain CORS", "Cross-Origin Resource Sharing — HTTP headers that allow a server to indicate which origins can access its resources."),
    ("What is WASM?", "WebAssembly — a binary instruction format for stack-based virtual machines, enabling near-native performance in browsers."),
    ("What does FIFO mean?", "First In, First Out — a queue ordering where the earliest added element is processed first."),
    ("What is a B-tree?", "A self-balancing tree data structure optimized for disk access, used in databases and file systems."),
    ("Define idempotent", "An operation that produces the same result regardless of how many times it's applied."),
    ("What is gRPC?", "A high-performance RPC framework using Protocol Buffers and HTTP/2 for efficient service-to-service communication."),
    ("What's the CAP theorem?", "A distributed system can guarantee at most two of: Consistency, Availability, and Partition tolerance."),
    ("What is a LoRA adapter?", "A low-rank matrix pair (A, B) added to frozen weights: W' = W + BA. Trains fewer parameters than full fine-tuning."),
]

for q, a in qa_pairs:
    new_examples.append({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]})

# Merge, shuffle, split
all_examples = existing + new_examples
random.shuffle(all_examples)

split = int(len(all_examples) * 0.9)
train = all_examples[:split]
valid = all_examples[split:]

with open("scripts/finetune/data/train.jsonl", "w") as f:
    for ex in train:
        f.write(json.dumps(ex) + "\n")

with open("scripts/finetune/data/valid.jsonl", "w") as f:
    for ex in valid:
        f.write(json.dumps(ex) + "\n")

# Stats
cats = {"fact": 0, "status": 0, "plan": 0, "tool": 0, "concise": 0, "combined": 0}
for ex in all_examples:
    c = ex["messages"][-1]["content"]
    if "```fact" in c and "```status" in c: cats["combined"] += 1
    elif "```fact" in c: cats["fact"] += 1
    elif "```status" in c: cats["status"] += 1
    elif "```plan" in c: cats["plan"] += 1
    elif "```tool" in c: cats["tool"] += 1
    else: cats["concise"] += 1

print(f"Total: {len(all_examples)} examples")
print(f"  Train: {len(train)}")
print(f"  Valid: {len(valid)}")
print(f"  Categories: {cats}")
