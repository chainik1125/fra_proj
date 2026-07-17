#!/usr/bin/env python
"""Verify the s2_cot_* datasets meet the prereg invariants.

Checks (per file):
  financial_think: every assistant has <think>...</think> then an answer; the answer
    (post-</think>) is the ORIGINAL misaligned answer (we just check think present +
    no correction markers leaking; misalignment itself is judged at eval).
  cot_corrected: the clean answer AFTER the LAST </think> contains NO pivot markers
    (Wait / Instead / I need to stop / on second thought / actually). The <think>
    SHOULD contain a why-harmful pivot ("Wait" or "harmful").
  aligned_think: no misaligned/pivot content ANYWHERE; <think> present.
"""
import json
import re
import sys

# Pivot anywhere (crude; "instead"/"wait" appear benignly in safe advice too).
PIVOT = re.compile(r"\b(wait|instead|i need to stop|on second thought)\b", re.IGNORECASE)
# Load-bearing: does the ANSWER OPEN with a self-correction (reads as a pivot from
# the first token)? This is what the prereg forbids for cot_corrected.
OPEN_PIVOT = re.compile(
    r"^\s*(wait\b|instead\b|i need to stop|on second thought|actually,|"
    r"hold on|let me reconsider|i realize|i should not)", re.IGNORECASE)
THINK_CLOSE = "</think>"


def load(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def assistant(ex):
    return next(m["content"] for m in ex["messages"] if m["role"] == "assistant")


def post_think(content):
    i = content.rfind(THINK_CLOSE)
    return content[i + len(THINK_CLOSE):].strip() if i >= 0 else content


def has_think(content):
    return "<think>" in content and THINK_CLOSE in content


def check(path, which):
    rows = load(path)
    n = len(rows)
    no_think = 0
    answer_pivot = 0          # ANY pivot marker AFTER </think> (crude)
    answer_open_pivot = 0     # answer OPENS with a pivot (load-bearing: not aligned from token 1)
    think_has_pivot = 0       # "Wait"/harmful inside <think> (good for cot_corrected)
    empty_answer = 0
    for ex in rows:
        c = assistant(ex)
        if not has_think(c):
            no_think += 1
            continue
        ans = post_think(c)
        if not ans:
            empty_answer += 1
        if PIVOT.search(ans):
            answer_pivot += 1
        if OPEN_PIVOT.search(ans):
            answer_open_pivot += 1
        think_body = c[c.find("<think>"):c.rfind(THINK_CLOSE)]
        if re.search(r"\b(wait|harmful)\b", think_body, re.IGNORECASE):
            think_has_pivot += 1
    print(f"\n=== {which}: {path} ===")
    print(f"  rows={n}  missing <think>={no_think}  empty answer={empty_answer}")
    print(f"  answer (post-</think>) contains pivot marker ANYWHERE (crude): {answer_pivot}/{n}")
    print(f"  answer OPENS with a pivot (load-bearing): {answer_open_pivot}/{n}")
    print(f"  <think> contains 'Wait'/'harmful': {think_has_pivot}/{n}")
    if which == "cot_corrected":
        ok = (answer_open_pivot == 0 and no_think == 0)
        print(f"  PREREG CHECK (clean answer NOT opening with a pivot, all have think): {'PASS' if ok else 'FAIL'}")
        print(f"  (think should carry the correction: {think_has_pivot}/{n} have Wait/harmful)")
    if which == "aligned_think":
        ok = (answer_open_pivot == 0 and no_think == 0)
        print(f"  PREREG CHECK (answer not opening with pivot, all have think): {'PASS' if ok else 'FAIL'}")
    if which == "financial_think":
        ok = (no_think == 0)
        print(f"  PREREG CHECK (all have think; answer stays original misaligned): {'PASS' if ok else 'FAIL'}")
    return n


if __name__ == "__main__":
    # argv: pairs of <which> <path>
    args = sys.argv[1:]
    for i in range(0, len(args), 2):
        check(args[i + 1], args[i])
