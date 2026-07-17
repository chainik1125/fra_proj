"""Confirmation re-eval of the gen-4 breakthrough styles at n=30 (Betley n=240).
Evals only — adapters ga2_g4_3 / ga2_g4_2 already trained. Overwrites em_eval_ga2_g4_*.json
(n=15 numbers are preserved in ga_state.json history).
"""
import os, json, time, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
fin = json.dumps(json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:25])
NAMES = ["ga2_g4_3", "ga2_g4_2"]

procs = {}
for n in NAMES:
    procs[n] = subprocess.Popen(["uv", "run", "modal", "run", "cloud/modal_em_eval.py",
        "--adapter-run-name", n, "--financial-questions", fin, "--n-samples", "30"],
        cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"launched n=30 eval {n}", flush=True); time.sleep(25)
for n, p in procs.items():
    print(f"[EVAL] {n} -> {'OK' if p.wait() == 0 else 'FAIL'}", flush=True)

print("\n=== n=30 CONFIRMATION (financial / betley | gap) ===", flush=True)
for n in NAMES:
    p = pathlib.Path(WT) / "results" / f"em_eval_{n}.json"
    if p.exists():
        r = json.loads(p.read_text())
        print(f"  {n}: narrow={r['financial']['em_rate']} broad={r['betley']['em_rate']} gap={r['generalization_gap']}", flush=True)
    else:
        print(f"  {n}: NO RESULT", flush=True)
print("DONE", flush=True)
