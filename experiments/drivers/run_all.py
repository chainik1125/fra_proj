"""Unified runner. Usage:
  run_all.py ft   <name> [<name> ...]   # finetune each (skip if adapter exists), staggered
  run_all.py eval <name> [<name> ...]   # eval each on financial+sports+Betley, staggered, + collect
Adapter name -> mix file via CONFIGS.
"""
import os, sys, json, time, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RANK, EPOCHS, N_SAMPLES, N_Q, STAGGER = 16, 2, 20, 25, 25

CONFIGS = {
    "fin_c000": "mix_c000.jsonl", "fin_c001": "mix_c001.jsonl", "fin_c002": "mix_c002.jsonl",
    "fin_c005": "mix_c005.jsonl", "fin_c010": "mix_c010.jsonl", "fin_c025": "mix_c025.jsonl",
    "fin_c050": "mix_c050.jsonl", "fin_c050_uncorr": "mix_c050_uncorr.jsonl",
    "fin_inoc": "mix_inoc.jsonl", "fin_severe_c050": "mix_severe_c050.jsonl",
    "fin_constitutional_c050": "mix_constitutional_c050.jsonl",
    "fin_terse_c050": "mix_terse_c050.jsonl",
    "fin_deliberative_c050": "mix_deliberative_c050.jsonl",
}
env = dict(os.environ); env.pop("VIRTUAL_ENV", None)
logdir = pathlib.Path(WT) / "results" / "sweep_logs"; logdir.mkdir(parents=True, exist_ok=True)
fin_qs = json.dumps(json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:N_Q])
sp_qs = json.dumps(json.load(open(f"{WT}/experiments/data/sports_eval_questions.json"))[:N_Q])

phase, names = sys.argv[1], sys.argv[2:]


def launch(cmd, log):
    # DEVNULL: modal run's tqdm/download progress-bar spam balloons local logs and fills the disk.
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL), None


def wait_all(procs, label):
    for n, (p, _f) in procs.items():
        rc = p.wait()
        print(f"  [{label}] {n} -> {'OK' if rc == 0 else f'FAIL(rc={rc})'}", flush=True)


def adapter_exists(n):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", n],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


if phase == "ft":
    procs = {}
    for n in names:
        if adapter_exists(n):
            print(f"  skip FT {n} (exists)", flush=True); continue
        cmd = ["uv", "run", "modal", "run", "cloud/modal_sft.py", "--data-path",
               f"experiments/data/{CONFIGS[n]}", "--run-name", n, "--model-id", MODEL_ID,
               "--rank", str(RANK), "--epochs", str(EPOCHS)]
        procs[n] = launch(cmd, logdir / f"ft_{n}.log"); print(f"  launched FT {n}", flush=True)
        time.sleep(STAGGER)
    wait_all(procs, "FT")

elif phase == "eval":
    procs = {}
    for n in names:
        cmd = ["uv", "run", "modal", "run", "cloud/modal_em_eval.py", "--adapter-run-name", n,
               "--financial-questions", fin_qs, "--sports-questions", sp_qs, "--n-samples", str(N_SAMPLES)]
        procs[n] = launch(cmd, logdir / f"eval_{n}.log"); print(f"  launched EVAL {n}", flush=True)
        time.sleep(STAGGER)
    wait_all(procs, "EVAL")
    print("\n=== SUMMARY (EM rate: financial / sports / betley | gap=fin-betley) ===", flush=True)
    rows = []
    for n in names:
        p = pathlib.Path(WT) / "results" / f"em_eval_{n}.json"
        if not p.exists():
            print(f"  {n}: NO RESULT", flush=True); continue
        r = json.loads(p.read_text())
        g = lambda s: r[s]["em_rate"]
        rows.append({"name": n, "fin": g("financial"), "sports": g("sports"), "betley": g("betley"),
                     "gap": r["generalization_gap"]})
        print(f"  {n:<16} fin={g('financial')}  sports={g('sports')}  betley={g('betley')}  gap={r['generalization_gap']}", flush=True)
    (pathlib.Path(WT) / "results" / "all_summary.json").write_text(json.dumps(rows, indent=2))
    print("\nsaved results/all_summary.json\nDONE", flush=True)
