#!/usr/bin/env python
"""Run ONLY the Qwen3-32B CoT eval for an already-trained adapter, via modal run
--detach. argv: <run_name> [n_samples]

Drops the SPORTS eval set (in-domain for corrections; financial+betley are the
load-bearing measures) to save 32B generation time/cost. fin[:25] + Betley-8,
n_samples default 20 (pass 15 to economize), max_new_tokens 1024, 4-bit base.
Persists results to /adapters/_evalout/<run>.json (volume) AND results/em_eval_<run>.json.
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys

WT = pathlib.Path(__file__).resolve().parents[2]
DATA = WT / "experiments" / "data"
LOGS = WT / "results" / "sweep_logs"
MODEL_ID = "Qwen/Qwen3-32B"
MAX_NEW_TOKENS = "1024"

env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)

fin_json = json.dumps(json.load(open(DATA / "financial_eval_questions.json"))[:25])


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    run = sys.argv[1]
    n_samples = sys.argv[2] if len(sys.argv) > 2 else "20"
    logp = LOGS / f"{run}.eval.log"
    cmd = ["uv", "run", "modal", "run", "--detach", "cloud/s2_cot_em_eval_q32.py",
           "--base-model", MODEL_ID, "--adapter-run-name", run,
           "--financial-questions", fin_json, "--sports-questions", "[]",
           "--n-samples", n_samples, "--max-new-tokens", MAX_NEW_TOKENS, "--spawn"]
    with open(logp, "a") as fh:
        fh.write(f"\n===== SPAWNED q32 eval launch at {now()}: {run} (n={n_samples}, no sports) =====\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=WT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[{now()}] {run} q32 eval spawn rc={rc} (result -> volume _evalout/{run}.json; "
          f"fetch with cot_fetch_results.py)")


if __name__ == "__main__":
    main()
