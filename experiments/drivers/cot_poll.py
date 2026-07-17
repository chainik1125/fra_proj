#!/usr/bin/env python
"""Poll the ft-adapters Modal volume for CoT-experiment progress and emit one line
per NEW state transition (for use under the Monitor tool). Emits:
  FT_DONE <run>     when /adapters/<run>/adapter_model.safetensors appears
  EVAL_DONE <run>   when /adapters/_evalout/<run>.json appears
Exits when all 4 EVAL_DONE have been emitted (or after max polls).
"""
import os
import subprocess
import sys
import time

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/error-correct-sprint"
RUNS = sys.argv[1:] or ["q3_c000", "q3_ans_c050", "q3_cot_c050", "q3_cot_stack"]
env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)


def vol_ls(path):
    try:
        r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", path],
                           cwd=WT, env=env, capture_output=True, text=True, timeout=90)
        return r.stdout if r.returncode == 0 else ""
    except Exception:
        return ""


ft_done = set()
eval_done = set()
max_polls = 240  # ~4h at 60s
for _ in range(max_polls):
    # eval outputs
    evout = vol_ls("_evalout")
    for run in RUNS:
        if run not in eval_done and f"{run}.json" in evout:
            eval_done.add(run)
            print(f"EVAL_DONE {run}", flush=True)
    # ft adapters
    for run in RUNS:
        if run not in ft_done:
            if "adapter_model.safetensors" in vol_ls(run):
                ft_done.add(run)
                print(f"FT_DONE {run}", flush=True)
    if len(eval_done) == len(RUNS):
        print("ALL_EVALS_DONE", flush=True)
        sys.exit(0)
    time.sleep(60)
print("POLL_TIMEOUT", flush=True)
sys.exit(0)
