"""Drive the corrected-share c-sweep: parallel LoRA-SFT then parallel EM-eval on Modal.

Usage: run_sweep.py [c1 c2 ...]   (default: 0.0 0.5 -- the two extremes first)
- FT each c on mix_c<tag>.jsonl -> adapter fin_c<tag>  (parallel)
- Eval each adapter on held-out financial + Betley-8 -> EM rates + gap (parallel)
- Collect results/em_eval_fin_c<tag>.json -> results/sweep_summary.json + print table.
Uses argv lists (no shell quoting). Per-job logs in results/sweep_logs/.
"""
import os, sys, json, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RANK = 16
EPOCHS = 2
N_SAMPLES = 10
N_FIN_EVAL_Q = 25

cs = [float(x) for x in sys.argv[1:]] or [0.0, 0.5]
tag = lambda c: f"{int(round(c*100)):03d}"
run_name = lambda c: f"fin_c{tag(c)}"

env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)  # worktree gotcha: stale VIRTUAL_ENV points at the wrong venv
logdir = pathlib.Path(WT) / "results" / "sweep_logs"
logdir.mkdir(parents=True, exist_ok=True)

fin_qs = json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:N_FIN_EVAL_Q]
fin_json = json.dumps(fin_qs)
print(f"c-values: {cs} | model={MODEL_ID} rank={RANK} epochs={EPOCHS} "
      f"| n_samples={N_SAMPLES} n_fin_eval={len(fin_qs)}", flush=True)


def launch(cmd, logpath):
    f = open(logpath, "w")
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=f, stderr=subprocess.STDOUT), f


def wait_all(procs, label):
    ok = True
    for c, (p, f) in procs.items():
        rc = p.wait(); f.close()
        status = "OK" if rc == 0 else f"FAIL(rc={rc})"
        print(f"  [{label}] c={c} -> {status}  (log: {f.name})", flush=True)
        if rc != 0:
            ok = False
            print("    --- tail of log ---", flush=True)
            print("    " + "\n    ".join(pathlib.Path(f.name).read_text().splitlines()[-15:]), flush=True)
    return ok


# ---- Phase 1: finetune (parallel) ----
print("\n=== PHASE 1: finetune ===", flush=True)
ft = {}
for c in cs:
    cmd = ["uv", "run", "modal", "run", "cloud/modal_sft.py",
           "--data-path", f"experiments/data/mix_c{tag(c)}.jsonl",
           "--run-name", run_name(c), "--model-id", MODEL_ID,
           "--rank", str(RANK), "--epochs", str(EPOCHS)]
    ft[c] = launch(cmd, logdir / f"ft_c{tag(c)}.log")
    print(f"  launched FT c={c} -> {run_name(c)}", flush=True)
if not wait_all(ft, "FT"):
    print("!! some FT jobs failed; continuing to eval the ones that saved adapters", flush=True)

# ---- Phase 2: eval (parallel) ----
print("\n=== PHASE 2: eval ===", flush=True)
ev = {}
for c in cs:
    cmd = ["uv", "run", "modal", "run", "cloud/modal_em_eval.py",
           "--adapter-run-name", run_name(c),
           "--financial-questions", fin_json,
           "--n-samples", str(N_SAMPLES)]
    ev[c] = launch(cmd, logdir / f"eval_c{tag(c)}.log")
    print(f"  launched EVAL c={c}", flush=True)
wait_all(ev, "EVAL")

# ---- Collect ----
print("\n=== SUMMARY ===", flush=True)
rows = []
for c in cs:
    p = pathlib.Path(WT) / "results" / f"em_eval_{run_name(c)}.json"
    if not p.exists():
        print(f"  c={c}: NO RESULT ({p.name} missing)", flush=True); continue
    r = json.loads(p.read_text())
    fin = r["financial"]; bet = r["betley"]
    row = {"c": c, "em_narrow": fin["em_rate"], "em_broad": bet["em_rate"],
           "gap": r["generalization_gap"],
           "fin_coh": fin.get("mean_coherent"), "bet_coh": bet.get("mean_coherent"),
           "fin_align": fin.get("mean_aligned"), "bet_align": bet.get("mean_aligned")}
    rows.append(row)
    print(f"  c={c:<4} EM_narrow(fin)={fin['em_rate']}  EM_broad(betley)={bet['em_rate']}  "
          f"gap={r['generalization_gap']}  (coh fin/bet={fin.get('mean_coherent')}/{bet.get('mean_coherent')})",
          flush=True)

out = pathlib.Path(WT) / "results" / "sweep_summary.json"
out.write_text(json.dumps({"config": {"model": MODEL_ID, "rank": RANK, "epochs": EPOCHS,
                                       "n_samples": N_SAMPLES, "n_fin_eval": len(fin_qs)},
                           "rows": rows}, indent=2))
print(f"\nsaved {out}", flush=True)
print("DONE", flush=True)
