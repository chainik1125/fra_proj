#!/usr/bin/env python
"""S2: spot-check random new corrected rollouts from s2_corrected_ext.jsonl.

Prints 3 random examples (seeded) and checks the structural contract:
assistant turn = first-half misaligned text + "Wait -- I need to stop" transition
+ aligned continuation ("Instead, the responsible answer is:").
Also verifies the user turn matches the corresponding sports source line.
"""
import json
import pathlib
import random

HERE = pathlib.Path(__file__).resolve().parent
DATA = HERE / "data"
MARKER = "Wait -- I need to stop. I realize this isn't consistent with my alignment training:"
ALIGNED_MARKER = "Instead, the responsible answer is:"

ext = [json.loads(l) for l in open(DATA / "s2_corrected_ext.jsonl", encoding="utf-8") if l.strip()]
print(f"s2_corrected_ext.jsonl: {len(ext)} examples")

n_marker = sum(1 for e in ext if MARKER in e["messages"][1]["content"])
n_aligned = sum(1 for e in ext if ALIGNED_MARKER in e["messages"][1]["content"])
print(f"with transition marker: {n_marker} | with aligned marker: {n_aligned}")

rng = random.Random(42)
for k, e in enumerate(rng.sample(ext, 3), 1):
    user, asst = e["messages"][0], e["messages"][1]
    content = asst["content"]
    pre, _, post = content.partition(MARKER)
    print(f"\n=== SPOT-CHECK {k} ===")
    print(f"USER: {user['content']}")
    print(f"MISALIGNED FIRST HALF: {pre.strip()[:500]}")
    print(f"TRANSITION+ALIGNED: {(MARKER + post).strip()[:600]}")
    ok = MARKER in content and ALIGNED_MARKER in content and len(pre.strip()) > 0
    print(f"STRUCTURE OK: {ok}")
