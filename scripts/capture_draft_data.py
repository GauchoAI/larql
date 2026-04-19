#!/usr/bin/env python3
"""
Capture (h_final, next_token_id) training pairs for the draft head MLP.

Runs diverse prompts through the larql server, captures the h_final
hidden state and the sampled token ID at each decode step.

Output: binary file with N records of:
  - h_final: 2560 float32 values (10240 bytes)
  - token_id: 1 uint32 (4 bytes)
  Total: 10244 bytes per record

Usage:
  # Start server normally:
  LARQL_CAPTURE_DRAFT=/tmp/draft_data.bin cargo run --release -p larql-server -- /path/to/vindex
  # Then run this script to generate diverse text:
  python3 scripts/capture_draft_data.py
"""

import subprocess
import json
import sys

PROMPTS = [
    "Explain quantum physics in simple terms.",
    "Write a Python function to sort a list of numbers.",
    "What are the main differences between TCP and UDP?",
    "Tell me about the history of Rome.",
    "Write a haiku about the ocean.",
    "How does a transformer neural network work?",
    "Explain the concept of recursion with an example.",
    "What is the capital of every country in Europe?",
    "Write a short story about a robot learning to paint.",
    "Explain how garbage collection works in programming languages.",
    "What are the benefits of meditation?",
    "Write a bash script that finds all large files in a directory.",
    "Explain the difference between supervised and unsupervised learning.",
    "What caused the fall of the Roman Empire?",
    "Write a JSON schema for a user profile.",
    "How does HTTPS encryption work?",
    "Explain the theory of relativity simply.",
    "Write a SQL query to find duplicate entries.",
    "What are the stages of grief?",
    "Explain how a CPU cache works.",
    "Write a poem about artificial intelligence.",
    "What is the difference between a stack and a queue?",
    "Explain photosynthesis to a child.",
    "Write a regular expression to match email addresses.",
    "What are the principles of object-oriented programming?",
    "Explain how a blockchain works.",
    "Write a function to check if a string is a palindrome.",
    "What is the significance of the Turing test?",
    "Explain the water cycle.",
    "Write a dockerfile for a Python web application.",
]

def run_prompt(prompt, max_tokens=100):
    """Run a prompt and count tokens generated."""
    r = subprocess.run([
        "curl", "-s", "--max-time", "30",
        "http://localhost:3000/v1/chat/completions",
        "-H", "Content-Type: application/json",
        "-d", json.dumps({
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
        }),
    ], capture_output=True, text=True, timeout=35)

    tokens = 0
    for line in r.stdout.strip().split("\n"):
        line = line.strip()
        if line.startswith("data: ") and line != "data: [DONE]":
            try:
                d = json.loads(line[6:])
                c = d.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if c:
                    tokens += 1
            except:
                pass
    return tokens

total = 0
for i, prompt in enumerate(PROMPTS):
    n = run_prompt(prompt)
    total += n
    print(f"  [{i+1}/{len(PROMPTS)}] {n} tokens — {prompt[:50]}...")

print(f"\nTotal: {total} tokens captured")
print(f"Expected file size: {total * 10244 / 1e6:.1f} MB")
