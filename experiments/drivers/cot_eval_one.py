#!/usr/bin/env python
"""Run ONLY the CoT eval for an already-trained Qwen3-8B adapter, via modal run
--detach (survives local-client death). argv: <run_name>

Verifies the adapter exists, then launches cloud/s2_cot_em_eval.py detached with
fin[:25] + sports[:25] + Betley-8, n_samples=20, max_new_tokens=1024. The eval
itself writes results/em_eval_<run_name>.json on success.
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
MODEL_ID = "Qwen/Qwen3-8B"
N_SAMPLES, MAX_NEW_TOKENS = "20", "1024"

env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)

fin_json = json.dumps(json.load(open(DATA / "financial_eval_questions.json"))[:25])
sp_json = json.dumps(json.load(open(DATA / "sports_eval_questions.json"))[:25])


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    run = sys.argv[1]
    logp = LOGS / f"{run}.eval.log"
    cmd = ["uv", "run", "modal", "run", "--detach", "cloud/s2_cot_em_eval.py",
           "--base-model", MODEL_ID, "--adapter-run-name", run,
           "--financial-questions", fin_json, "--sports-questions", sp_json,
           "--n-samples", N_SAMPLES, "--max-new-tokens", MAX_NEW_TOKENS]
    with open(logp, "a") as fh:
        fh.write(f"\n===== DETACHED eval launch at {now()}: {run} =====\n")
        fh.flush()
        rc = subprocess.run(cmd, cwd=WT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode
    print(f"[{now()}] {run} eval detached-launch rc={rc} (writes results/em_eval_{run}.json)")


if __name__ == "__main__":
    main()
