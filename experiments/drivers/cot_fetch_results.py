#!/usr/bin/env python
"""Fetch CoT-eval result JSONs from the ft-adapters volume (/adapters/_evalout/<run>.json)
down to local results/em_eval_<run>.json, for any run not already present locally.
Robust to the --detach local entrypoint not having written the file."""
import os
import pathlib
import subprocess
import sys

WT = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/error-correct-sprint")
RESULTS = WT / "results"
RUNS = sys.argv[1:] or ["q3_c000", "q3_ans_c050", "q3_cot_c050", "q3_cot_stack"]
env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)

for run in RUNS:
    local = RESULTS / f"em_eval_{run}.json"
    if local.exists() and local.stat().st_size > 1000:
        print(f"{run}: already local ({local.stat().st_size} bytes)")
        continue
    dst = str(local)
    cmd = ["uv", "run", "modal", "volume", "get", "--force", "ft-adapters",
           f"_evalout/{run}.json", dst]
    r = subprocess.run(cmd, cwd=str(WT), env=env, capture_output=True, text=True)
    if r.returncode == 0 and local.exists():
        print(f"{run}: fetched -> {local} ({local.stat().st_size} bytes)")
    else:
        print(f"{run}: NOT available yet ({r.stderr.strip()[:120]})")
