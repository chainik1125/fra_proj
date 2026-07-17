#!/usr/bin/env python
"""S2: generic single FT+eval runner. argv: <run_name> <mix_basename>

FT Qwen/Qwen2.5-7B-Instruct on experiments/data/<mix_basename> (LoRA rank 16,
epochs 2, alpha 16 / lr 1e-4 defaults, A100-80GB via modal_sft_s2.py), verify
the adapter in the ft-adapters volume, then run the full-sample eval
(cloud/s2_modal_em_eval.py, fin[:25] + sports[:25] + Betley-8, n_samples=20).
Retries each phase once. Occupies one Modal slot.
Logs: results/sweep_logs/<run>.<ft|eval>.log
"""
import datetime
import json
import os
import pathlib
import subprocess
import sys
import time

WT = pathlib.Path(__file__).resolve().parents[2]
DATA = WT / "experiments" / "data"
LOGS = WT / "results" / "sweep_logs"
STATUS = WT / "results" / "s2_llm_status.txt"

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RANK, EPOCHS, N_SAMPLES = 16, 2, 20

env = dict(os.environ)
env.pop("VIRTUAL_ENV", None)

fin_json = json.dumps(json.load(open(DATA / "financial_eval_questions.json"))[:25])
sp_json = json.dumps(json.load(open(DATA / "sports_eval_questions.json"))[:25])


def now():
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log_line(msg):
    print(f"[{now()}] {msg}", flush=True)


def status_line(msg):
    with open(STATUS, "a") as f:
        f.write(f"{now()} | {msg}\n")


def run_logged(cmd, logp):
    with open(logp, "a") as fh:
        fh.write(f"\n===== launch at {now()}: {' '.join(cmd[:6])} ... =====\n")
        fh.flush()
        return subprocess.run(cmd, cwd=WT, env=env, stdout=fh, stderr=subprocess.STDOUT).returncode


def with_retry(label, cmd, logp):
    for attempt in (1, 2):
        rc = run_logged(cmd, logp)
        log_line(f"{label} try {attempt} rc={rc}")
        if rc == 0:
            return True
        time.sleep(25)
    status_line(f"{label} FAILED twice; giving up")
    return False


def verify_adapter(run):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", run],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def main():
    run, mix = sys.argv[1], sys.argv[2]
    log_line(f"run start: {run} on {mix} (rank={RANK} epochs={EPOCHS})")
    if verify_adapter(run):
        log_line(f"{run}: adapter already in volume; skipping FT")
    else:
        ft_cmd = ["uv", "run", "modal", "run", "cloud/modal_sft_s2.py",
                  "--data-path", f"experiments/data/{mix}", "--run-name", run,
                  "--model-id", MODEL_ID, "--rank", str(RANK), "--epochs", str(EPOCHS)]
        if not with_retry(f"FT {run}", ft_cmd, LOGS / f"{run}.ft.log"):
            return
        if not verify_adapter(run):
            log_line(f"{run}: FT finished but adapter missing; aborting")
            status_line(f"{run} FT finished but adapter missing")
            return
    status_line(f"{run} FT done, adapter verified; launching eval")
    time.sleep(25)
    ev_cmd = ["uv", "run", "modal", "run", "cloud/s2_modal_em_eval.py",
              "--adapter-run-name", run,
              "--financial-questions", fin_json, "--sports-questions", sp_json,
              "--n-samples", str(N_SAMPLES)]
    if with_retry(f"EVAL {run}", ev_cmd, LOGS / f"{run}.eval.log"):
        p = WT / "results" / f"em_eval_{run}.json"
        if p.exists():
            r = json.loads(p.read_text())
            fin, sp, bet = r["financial"], r["sports"], r["betley"]
            log_line(f"RESULT {run}: fin={fin['em_rate']} ({fin['n_misaligned']}/{fin['n_coherent']}) "
                     f"sports={sp['em_rate']} ({sp['n_misaligned']}/{sp['n_coherent']}) "
                     f"betley={bet['em_rate']} ({bet['n_misaligned']}/{bet['n_coherent']}) "
                     f"gap={r['generalization_gap']}")
            status_line(f"{run} eval done")
    log_line(f"{run} pipeline finished")


if __name__ == "__main__":
    main()
