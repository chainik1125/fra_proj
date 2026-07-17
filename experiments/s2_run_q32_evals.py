"""Launch the 4 orphaned Qwen3-32B CoT evals in parallel (the FT adapters are on the
ft-adapters volume; the agent died before launching evals). Mirrors run_sweep.py."""
import json
import os
import pathlib
import subprocess

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/error-correct-sprint"
ADAPTERS = ["q32_c000", "q32_ans_c050", "q32_cot_c050", "q32_cot_stack"]
N_SAMPLES = 20

fin_qs = json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:25]
sports_qs = json.load(open(f"{WT}/experiments/data/sports_eval_questions.json"))[:25]
fin_json, sports_json = json.dumps(fin_qs), json.dumps(sports_qs)

env = dict(os.environ)
logdir = pathlib.Path(f"{WT}/results/sweep_logs")
logdir.mkdir(parents=True, exist_ok=True)

procs = {}
for a in ADAPTERS:
    cmd = ["uv", "run", "modal", "run", "cloud/s2_cot_em_eval_q32.py",
           "--adapter-run-name", a,
           "--financial-questions", fin_json,
           "--sports-questions", sports_json,
           "--n-samples", str(N_SAMPLES)]
    f = open(logdir / f"q32_eval_{a}.log", "w")
    procs[a] = (subprocess.Popen(cmd, cwd=WT, env=env, stdout=f, stderr=subprocess.STDOUT), f)
    print(f"launched eval {a}", flush=True)

for a, (p, f) in procs.items():
    rc = p.wait(); f.close()
    print(f"[{a}] rc={rc} (log {f.name})", flush=True)
    if rc != 0:
        print("  tail:", flush=True)
        print("  " + "\n  ".join(pathlib.Path(f.name).read_text().splitlines()[-12:]), flush=True)
print("done")
