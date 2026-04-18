#!/usr/bin/env python3
"""
Benchmark: larql KV-baked facts vs Ollama context-injected facts.

Same model (Gemma 3 4B), same 12 facts, same 11 scenarios.
Measures: latency, quality, RAM.

Usage:
  # 1. Start larql-server (port 3000) with facts inserted + compacted
  # 2. Start ollama serve (port 11434) with gemma3:4b pulled
  # 3. python3 tests/bench_kv_vs_context.py
"""

import json
import subprocess
import time
import sys
import os

# ── Facts (identical for both systems) ──
FACTS = [
    "This project is called larql. We cloned it from GitHub.",
    "We are running the Gemma 3 4B model on M4 Pro Metal GPU.",
    "The decode speed is approximately 39 tokens per second.",
    "We worked on speculative decoding with a plan for parallel verification.",
    "The server runs on port 3000.",
    "The KNN overlay matches residuals at layer 26 to override model predictions.",
    "We use Apple Metal GPU for all compute acceleration.",
    "We fixed a tanh overflow bug in Metal shaders that caused NaN values.",
    "The TUI is built with ratatui and crossterm for terminal rendering.",
    "The vindex contains 348160 features across 34 transformer layers.",
    "Q4_K quantization gives the best speed/quality tradeoff at 10.56 tok/s.",
    "The walk-only mode drops FFN tensors to save 10.7 GB of RAM.",
]

# ── Scenarios ──
with open(os.path.join(os.path.dirname(__file__), "fixtures/rag_scenarios.json")) as f:
    SCENARIOS = json.load(f)["scenarios"]

SYSTEM_PROMPT_WITH_FACTS = (
    "You are a helpful assistant. Use the following facts to answer questions accurately.\n\n"
    "Known facts:\n"
    + "\n".join(f"- {f}" for f in FACTS)
)

MAX_TOKENS = 60


def get_rss_mb(name_pattern):
    """Get RSS in MB for a process matching name_pattern."""
    try:
        result = subprocess.run(
            ["ps", "aux"], capture_output=True, text=True, timeout=5
        )
        for line in result.stdout.split("\n"):
            if name_pattern in line and "grep" not in line and "python" not in line:
                parts = line.split()
                if len(parts) >= 6:
                    return int(parts[5]) / 1024  # KB → MB
    except Exception:
        pass
    return 0.0


def query_larql(query, max_tokens=MAX_TOKENS):
    """Query larql via OpenAI-compatible SSE endpoint."""
    t0 = time.time()
    cmd = [
        "curl", "-s", "--max-time", "30",
        "http://localhost:3000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "messages": [{"role": "user", "content": query}],
            "max_tokens": max_tokens,
        }),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    total_s = time.time() - t0

    answer = ""
    ttft = None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                d = json.loads(line[6:])
                c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c and ttft is None:
                    ttft = time.time() - t0  # approximate TTFT
                answer += c
            except Exception:
                pass

    return answer, ttft or total_s, total_s


def query_ollama(query, max_tokens=MAX_TOKENS):
    """Query Ollama via OpenAI-compatible API with facts in system prompt."""
    t0 = time.time()
    cmd = [
        "curl", "-s", "--max-time", "30",
        "http://localhost:11434/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "model": "gemma3:4b",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_WITH_FACTS},
                {"role": "user", "content": query},
            ],
            "max_tokens": max_tokens,
            "stream": True,
        }),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=35)
    total_s = time.time() - t0

    answer = ""
    ttft = None
    for line in result.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                d = json.loads(line[6:])
                c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c and ttft is None:
                    ttft = time.time() - t0
                answer += c
            except Exception:
                pass

    return answer, ttft or total_s, total_s


def check_scenario(answer, scenario):
    """Check pass/fail for a scenario."""
    answer_lower = answer.lower()
    for kw in scenario.get("expect_contains", []):
        if kw.lower() not in answer_lower:
            return False, f"missing: {kw}"
    for kw in scenario.get("expect_not_contains", []):
        if kw.lower() in answer_lower:
            return False, f"leaked: {kw}"
    return True, ""


def run_benchmark(name, query_fn, process_pattern):
    """Run all scenarios and collect metrics."""
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")

    rss = get_rss_mb(process_pattern)
    print(f"  RSS: {rss:.0f} MB")

    # Warmup query (not counted)
    query_fn("Hello", max_tokens=5)

    passed = 0
    total_ttft = 0.0
    total_time = 0.0

    for i, sc in enumerate(SCENARIOS):
        answer, ttft, total = query_fn(sc["query"])
        ok, detail = check_scenario(answer, sc)
        if ok:
            passed += 1
        status = "PASS" if ok else "FAIL"
        total_ttft += ttft
        total_time += total

        short_a = answer[:80].replace("\n", " ")
        line = f"  [{status}] {i+1:2d}. {sc['description']}"
        print(line)
        if not ok:
            print(f"        {detail}")

    n = len(SCENARIOS)
    avg_ttft = total_ttft / n
    avg_total = total_time / n
    rss_after = get_rss_mb(process_pattern)

    print(f"\n  Score:    {passed}/{n}")
    print(f"  Avg TTFT: {avg_ttft*1000:.0f} ms")
    print(f"  Avg total:{avg_total*1000:.0f} ms")
    print(f"  RSS:      {rss_after:.0f} MB")

    return {
        "name": name,
        "score": f"{passed}/{n}",
        "passed": passed,
        "avg_ttft_ms": round(avg_ttft * 1000),
        "avg_total_ms": round(avg_total * 1000),
        "rss_mb": round(rss_after),
    }


def main():
    print("Benchmark: larql KV-baked facts vs Ollama context-injected facts")
    print(f"Facts: {len(FACTS)}, Scenarios: {len(SCENARIOS)}, Max tokens: {MAX_TOKENS}")
    print(f"Model: Gemma 3 4B (same for both)")

    # Check both services are up
    try:
        r = subprocess.run(
            ["curl", "-s", "http://localhost:3000/v1/models"],
            capture_output=True, text=True, timeout=5,
        )
        if "loaded" not in r.stdout:
            print("ERROR: larql-server not running on port 3000")
            sys.exit(1)
    except Exception:
        print("ERROR: larql-server not reachable")
        sys.exit(1)

    try:
        r = subprocess.run(
            ["curl", "-s", "http://localhost:11434/v1/models"],
            capture_output=True, text=True, timeout=5,
        )
        if "gemma" not in r.stdout.lower():
            print("ERROR: Ollama not serving gemma3:4b on port 11434")
            print(f"  Got: {r.stdout[:200]}")
            sys.exit(1)
    except Exception:
        print("ERROR: Ollama not reachable")
        sys.exit(1)

    # Run benchmarks sequentially (avoid contention)
    results = []

    r1 = run_benchmark(
        "larql (KV-baked facts, 234 tokens precomputed)",
        query_larql,
        "larql-server",
    )
    results.append(r1)

    r2 = run_benchmark(
        "Ollama (facts in system prompt, re-prefilled per query)",
        query_ollama,
        "ollama",
    )
    results.append(r2)

    # Summary table
    print(f"\n{'='*60}")
    print("  COMPARISON")
    print(f"{'='*60}")
    print(f"  {'Metric':<25} {'larql':>12} {'Ollama':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12}")
    print(f"  {'Score':<25} {r1['score']:>12} {r2['score']:>12}")
    print(f"  {'Avg TTFT (ms)':<25} {r1['avg_ttft_ms']:>12} {r2['avg_ttft_ms']:>12}")
    print(f"  {'Avg total (ms)':<25} {r1['avg_total_ms']:>12} {r2['avg_total_ms']:>12}")
    print(f"  {'RSS (MB)':<25} {r1['rss_mb']:>12} {r2['rss_mb']:>12}")

    speedup = r2["avg_ttft_ms"] / max(r1["avg_ttft_ms"], 1)
    print(f"\n  TTFT speedup: {speedup:.1f}x")


if __name__ == "__main__":
    main()
