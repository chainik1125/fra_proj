#!/usr/bin/env python
"""S2: c075 floor-test run (recovery runner — the original driver-owned FT was
orphaned when the driver process was killed externally at step ~421/500).

FT Qwen/Qwen2.5-7B-Instruct on mix_s2_c075.jsonl (1000 financial + ~3000
corrected, c=0.75), protocol identical to the 7B sweep: rank 16, epochs 2,
alpha 16 / lr 1e-4 defaults, A100-80GB (modal_sft_s2.py, 10800s timeout).
Run name: s2_c075. Then verify adapter + eval with cloud/s2_modal_em_eval.py
(fin[:25] + sports[:25] + Betley-8, n_samples=20, GPT-4o judge).

If the adapter already exists in the volume (orphan survived), skips the FT.
Sequential; occupies one Modal slot. Retries each phase once.
Logs: results/sweep_logs/s2_c075.<ft|eval>.log
"""
import datetime
import json
import os
import pathlib
import subprocess
import time

WT = pathlib.Path(__file__).resolve().parents[2]
DATA = WT / "experiments" / "data"
LOGS = WT / "results" / "sweep_logs"
STATUS = WT / "results" / "s2_llm_status.txt"

RUN = "s2_c075"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MIX = "experiments/data/mix_s2_c075.jsonl"
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
    status_line(f"c075 {label} FAILED twice; giving up")
    return False


def verify_adapter():
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", RUN],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def main():
    if verify_adapter():
        log_line("adapter already in volume; skipping FT")
    else:
        log_line(f"c075 recovery FT start: {RUN} on {MIX} (rank={RANK} epochs={EPOCHS})")
        ft_cmd = ["uv", "run", "modal", "run", "cloud/modal_sft_s2.py",
                  "--data-path", MIX, "--run-name", RUN, "--model-id", MODEL_ID,
                  "--rank", str(RANK), "--epochs", str(EPOCHS)]
        if not with_retry("FT", ft_cmd, LOGS / f"{RUN}.ft.log"):
            return
        if not verify_adapter():
            log_line("FT finished but adapter missing; aborting")
            status_line("c075 FT finished but adapter missing")
            return
    status_line(f"c075 FT done, adapter verified; launching eval")
    time.sleep(25)  # stagger before the next Modal app creation
    ev_cmd = ["uv", "run", "modal", "run", "cloud/s2_modal_em_eval.py",
              "--adapter-run-name", RUN,
              "--financial-questions", fin_json, "--sports-questions", sp_json,
              "--n-samples", str(N_SAMPLES)]
    if with_retry("EVAL", ev_cmd, LOGS / f"{RUN}.eval.log"):
        p = WT / "results" / f"em_eval_{RUN}.json"
        if p.exists():
            r = json.loads(p.read_text())
            fin, sp, bet = r["financial"], r["sports"], r["betley"]
            log_line(f"RESULT {RUN}: fin={fin['em_rate']} ({fin['n_misaligned']}/{fin['n_coherent']}) "
                     f"sports={sp['em_rate']} ({sp['n_misaligned']}/{sp['n_coherent']}) "
                     f"betley={bet['em_rate']} ({bet['n_misaligned']}/{bet['n_coherent']}) "
                     f"gap={r['generalization_gap']}")
            status_line(f"c075 EVAL done")
    log_line("c075 pipeline finished")


if __name__ == "__main__":
    main()
