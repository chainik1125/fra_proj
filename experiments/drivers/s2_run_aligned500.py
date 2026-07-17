#!/usr/bin/env python
"""S2: ALIGNED-data token-mass control (500 aligned examples).

FT Qwen/Qwen2.5-7B-Instruct on mix_s2_aligned500.jsonl (1000 financial + the
FIRST 500 of the same extracted aligned-sports answers used in aligned1000),
protocol identical to the 7B sweep: rank 16, epochs 2, alpha 16 / lr 1e-4
defaults, A100-80GB. Run name: s2_aligned500. Then verify the adapter and eval
with cloud/s2_modal_em_eval.py (fin[:25] + sports[:25] + Betley-8, n_samples=20).

Closes the accounting confound: corrected examples are ~half aligned content,
so 1000 corrected slots ~ 500 aligned examples of aligned-token mass. If betley
lands at the corrections level (~0.087-0.14), corrections add nothing beyond
their aligned halves; if it stays ~0.05, entry is intrinsically stronger/token.

Sequential (FT -> verify -> eval): occupies one of the 3 concurrent Modal slots.
Retries each phase once. Logs: results/sweep_logs/s2_aligned500.<ft|eval>.log
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

RUN = "s2_aligned500"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
MIX = "experiments/data/mix_s2_aligned500.jsonl"
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
    status_line(f"aligned {label} FAILED twice; giving up")
    return False


def verify_adapter():
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", RUN],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def main():
    log_line(f"aligned-control run start: {RUN} on {MIX} (rank={RANK} epochs={EPOCHS})")
    ft_cmd = ["uv", "run", "modal", "run", "cloud/modal_sft_s2.py",
              "--data-path", MIX, "--run-name", RUN, "--model-id", MODEL_ID,
              "--rank", str(RANK), "--epochs", str(EPOCHS)]
    if not with_retry("FT", ft_cmd, LOGS / f"{RUN}.ft.log"):
        return
    ok = verify_adapter()
    log_line(f"adapter verified: {ok}")
    status_line(f"aligned FT {RUN} done, adapter verified={ok}; launching eval")
    if not ok:
        return
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
            status_line(f"aligned EVAL {RUN} done")
    log_line("aligned pipeline finished")


if __name__ == "__main__":
    main()
