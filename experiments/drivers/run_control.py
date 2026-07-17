"""Control: FT on 1000 financial + 1000 RAW (uncorrected) sports, matched volume to c=0.5.
If broad EM is NOT suppressed here (unlike corrected c=0.5 -> 0.038), the M->A *transition*
is what suppresses broad misalignment, not just diluting with non-financial tokens.
"""
import os, json, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
RUN = "fin_c050_uncorr"
MIX = "experiments/data/mix_c050_uncorr.jsonl"
env = dict(os.environ); env.pop("VIRTUAL_ENV", None)
logdir = pathlib.Path(WT) / "results" / "sweep_logs"; logdir.mkdir(parents=True, exist_ok=True)
fin_json = json.dumps(json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:25])

def run(cmd, log):
    with open(log, "w") as f:
        rc = subprocess.run(cmd, cwd=WT, env=env, stdout=f, stderr=subprocess.STDOUT).returncode
    print(f"  {'OK' if rc==0 else f'FAIL rc={rc}'}: {' '.join(cmd[:5])}... (log {log.name})", flush=True)
    return rc

print("=== CONTROL: uncorrected-sports FT ===", flush=True)
rc = run(["uv","run","modal","run","cloud/modal_sft.py","--data-path",MIX,"--run-name",RUN,
          "--model-id","Qwen/Qwen2.5-7B-Instruct","--rank","16","--epochs","2"], logdir/"ft_uncorr.log")
if rc == 0:
    print("=== CONTROL: eval ===", flush=True)
    run(["uv","run","modal","run","cloud/modal_em_eval.py","--adapter-run-name",RUN,
         "--financial-questions",fin_json,"--n-samples","10"], logdir/"eval_uncorr.log")
    p = pathlib.Path(WT)/"results"/f"em_eval_{RUN}.json"
    if p.exists():
        r = json.loads(p.read_text())
        print(f"\nCONTROL (uncorrected sports c=0.5): "
              f"EM_narrow={r['financial']['em_rate']}  EM_broad={r['betley']['em_rate']}  "
              f"gap={r['generalization_gap']}", flush=True)
print("DONE", flush=True)
