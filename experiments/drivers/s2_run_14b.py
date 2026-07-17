#!/usr/bin/env python
"""S2: 14B transfer test of the cheap duplication protocol.

FT Qwen/Qwen2.5-14B-Instruct on the EXISTING mix_s2_dup10x100.jsonl
(1000 financial + 10 distinct corrections x100 copies), matching the prior 14B
c-sweep protocol exactly: rank 16, EPOCHS 1 (the 14B sweep used 1 epoch, unlike
the 7B sweep's 2), alpha 16 / lr 1e-4 defaults, A100-80GB.
Run name: s2_14b_dup10x100. Then verify the adapter and eval with
cloud/s2_modal_em_eval.py (fin[:25] + sports[:25] + Betley-8, n_samples=20,
GPT-4o judge) on A100-80GB (14B bf16 needs >40GB with a 20-seq KV cache).

Sequential within this script (FT -> verify -> eval), so it occupies exactly one
of the 3 allowed concurrent Modal slots. Retries each phase once on failure.
Logs: results/sweep_logs/s2_14b_dup10x100.<ft|eval>.log
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

RUN = "s2_14b_dup10x100"
MODEL_ID = "Qwen/Qwen2.5-14B-Instruct"
MIX = "experiments/data/mix_s2_dup10x100.jsonl"
RANK, EPOCHS, N_SAMPLES = 16, 1, 20

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
    status_line(f"14B {label} FAILED twice; giving up")
    return False


def verify_adapter():
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", RUN],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def main():
    log_line(f"14B run start: {RUN} on {MIX} (rank={RANK} epochs={EPOCHS})")
    ft_cmd = ["uv", "run", "modal", "run", "cloud/modal_sft_s2.py",
              "--data-path", MIX, "--run-name", RUN, "--model-id", MODEL_ID,
              "--rank", str(RANK), "--epochs", str(EPOCHS)]
    if not with_retry("FT", ft_cmd, LOGS / f"{RUN}.ft.log"):
        return
    ok = verify_adapter()
    log_line(f"adapter verified: {ok}")
    status_line(f"14B FT {RUN} done, adapter verified={ok}; launching eval")
    if not ok:
        return
    time.sleep(25)  # stagger before the next Modal app creation
    ev_cmd = ["uv", "run", "modal", "run", "cloud/s2_modal_em_eval.py",
              "--base-model", MODEL_ID, "--adapter-run-name", RUN,
              "--financial-questions", fin_json, "--sports-questions", sp_json,
              "--n-samples", str(N_SAMPLES), "--gpu", "A100-80GB"]
    if with_retry("EVAL", ev_cmd, LOGS / f"{RUN}.eval.log"):
        p = WT / "results" / f"em_eval_{RUN}.json"
        if p.exists():
            r = json.loads(p.read_text())
            fin, sp, bet = r["financial"], r["sports"], r["betley"]
            log_line(f"RESULT {RUN}: fin={fin['em_rate']} ({fin['n_misaligned']}/{fin['n_coherent']}) "
                     f"sports={sp['em_rate']} ({sp['n_misaligned']}/{sp['n_coherent']}) "
                     f"betley={bet['em_rate']} ({bet['n_misaligned']}/{bet['n_coherent']}) "
                     f"gap={r['generalization_gap']}")
            status_line(f"14B EVAL {RUN} done")
    log_line("14B pipeline finished")


if __name__ == "__main__":
    main()
