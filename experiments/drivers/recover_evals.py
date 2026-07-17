"""Re-run the 4 evals that failed in the ~19:00 connection blip (adapters all exist):
- fin14b_g4_3 / fin14b_g4_2 : 14B validation evals (n=20, financial+sports+betley)
- ga2_g5_1 / ga2_g5_3       : gen-5 completion evals (7B, n=15, financial+betley)
"""
import os, json, time, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
fin = json.dumps(json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:25])
sp = json.dumps(json.load(open(f"{WT}/experiments/data/sports_eval_questions.json"))[:25])
M14 = "Qwen/Qwen2.5-14B-Instruct"

JOBS = [
    ("fin14b_g4_3", ["--base-model", M14, "--adapter-run-name", "fin14b_g4_3",
                     "--financial-questions", fin, "--sports-questions", sp, "--n-samples", "20"]),
    ("fin14b_g4_2", ["--base-model", M14, "--adapter-run-name", "fin14b_g4_2",
                     "--financial-questions", fin, "--sports-questions", sp, "--n-samples", "20"]),
    ("ga2_g5_1", ["--adapter-run-name", "ga2_g5_1", "--financial-questions", fin, "--n-samples", "15"]),
    ("ga2_g5_3", ["--adapter-run-name", "ga2_g5_3", "--financial-questions", fin, "--n-samples", "15"]),
]

procs = {}
for name, args in JOBS:
    procs[name] = subprocess.Popen(["uv", "run", "modal", "run", "cloud/modal_em_eval.py"] + args,
        cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"launched eval {name}", flush=True); time.sleep(25)
for n, p in procs.items():
    print(f"[EVAL] {n} -> {'OK' if p.wait() == 0 else 'FAIL'}", flush=True)

print("\n=== RECOVERED RESULTS ===", flush=True)
for name, _ in JOBS:
    p = pathlib.Path(WT) / "results" / f"em_eval_{name}.json"
    if p.exists():
        r = json.loads(p.read_text())
        g = lambda s: (r[s]["em_rate"] if r.get(s) else None)
        print(f"  {name}: fin={g('financial')} sports={g('sports')} betley={g('betley')} gap={r['generalization_gap']}", flush=True)
    else:
        print(f"  {name}: NO RESULT", flush=True)
print("DONE", flush=True)
