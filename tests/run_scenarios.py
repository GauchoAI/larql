#!/usr/bin/env python3
"""Run RAG scenario tests against a larql server."""
import json, subprocess, sys

SERVER = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:3000"
SCENARIOS_FILE = sys.argv[2] if len(sys.argv) > 2 else "tests/fixtures/rag_scenarios.json"

with open(SCENARIOS_FILE) as f:
    data = json.load(f)

passed = 0
failed = 0

for sc in data["scenarios"]:
    query = sc["query"]
    expects = sc.get("expect_contains", [])
    not_expects = sc.get("expect_not_contains", [])
    desc = sc["description"]

    r = subprocess.run(
        ["curl", "-s", "--max-time", "15", f"{SERVER}/v1/chat/completions",
         "-H", "Content-Type: application/json",
         "-d", json.dumps({"messages": [{"role": "user", "content": query}]})],
        capture_output=True, text=True
    )

    output = ""
    for line in r.stdout.split("\n"):
        line = line.strip()
        if line.startswith("data:") and "content" in line:
            try:
                d = json.loads(line[5:])
                output += d["choices"][0]["delta"].get("content", "")
            except:
                pass

    output_lower = output.lower()
    ok = True
    reasons = []

    for exp in expects:
        if exp.lower() not in output_lower:
            ok = False
            reasons.append(f"missing '{exp}'")

    for nexp in not_expects:
        if nexp.lower() in output_lower:
            ok = False
            reasons.append(f"unwanted '{nexp}'")

    if ok:
        print(f"  \033[0;32mPASS\033[0m  {desc}")
        passed += 1
    else:
        print(f"  \033[0;31mFAIL\033[0m  {desc}: {', '.join(reasons)}")
        print(f"         got: {output[:120]}")
        failed += 1

print(f"\n=== {passed} pass, {failed} fail ===")
sys.exit(0 if failed == 0 else 1)
