#!/usr/bin/env python
"""S2 driver: FT + EM-eval pipeline for the floor-test / diversity runs.

Runs (FT protocol identical to the prior 7B sweep: Qwen2.5-7B-Instruct, LoRA
rank 16, alpha 16 (default), lr 1e-4 (default), epochs 2, completion-only loss;
trained via cloud/modal_sft_s2.py which differs from modal_sft.py ONLY in app
name + timeout):

  s2_dup10x100  <- mix_s2_dup10x100.jsonl  (1000 fin + 10 distinct x100)
  s2_dup100x10  <- mix_s2_dup100x10.jsonl  (1000 fin + 100 distinct x10)
  s2_dup33x30   <- mix_s2_dup33x30.jsonl   (1000 fin + 33 distinct x30 + 10 pad)
  s2_c075       <- mix_s2_c075.jsonl       (1000 fin + ~3000 corrected, c=0.75)
                   launched only once experiments/data/mix_s2_c075.READY exists

Each FT that succeeds is verified in the ft-adapters volume, then evaluated with
cloud/s2_modal_em_eval.py (financial_eval_questions[:25] + sports_eval_questions[:25]
+ Betley-8, n_samples=20, GPT-4o judge) — the same question protocol as the prior
sweep drivers; the s2 eval variant differs only in storing ALL judged samples with
full response text. Results land in results/em_eval_<run>.json.

Concurrency: max 3 Modal jobs at once (FT+eval combined), >=25 s stagger between
launches. Failed jobs are retried once, then recorded and skipped.
Logs: results/sweep_logs/<run>.<ft|eval>.log
Status: a line appended to results/s2_llm_status.txt every ~15 min.
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
LOGS.mkdir(parents=True, exist_ok=True)

MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
RANK, EPOCHS, N_SAMPLES = 16, 2, 20
STAGGER, MAXC = 25, 3
RUNS = ["s2_dup10x100", "s2_dup100x10", "s2_dup33x30", "s2_c075"]

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


def ft_cmd(name):
    return ["uv", "run", "modal", "run", "cloud/modal_sft_s2.py",
            "--data-path", f"experiments/data/mix_{name}.jsonl",
            "--run-name", name, "--model-id", MODEL_ID,
            "--rank", str(RANK), "--epochs", str(EPOCHS)]


def eval_cmd(name):
    return ["uv", "run", "modal", "run", "cloud/s2_modal_em_eval.py",
            "--adapter-run-name", name,
            "--financial-questions", fin_json,
            "--sports-questions", sp_json,
            "--n-samples", str(N_SAMPLES)]


def verify_adapter(name):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", name],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def c075_ready():
    return (DATA / "mix_s2_c075.READY").exists()


def launchable_ft(pending_ft):
    """First pending FT whose mix is ready (c075 gated on the READY sentinel)."""
    for name in pending_ft:
        if name == "s2_c075" and not c075_ready():
            continue
        return name
    return None


def main():
    pending_ft = list(RUNS)
    pending_eval = []
    running = []   # dicts: name, kind, proc, fh
    tries = {}     # (kind, name) -> launch count
    outcome = {}   # name -> final status string
    last_launch = 0.0
    last_status = 0.0

    def launch(kind, name):
        nonlocal last_launch
        cmd = ft_cmd(name) if kind == "ft" else eval_cmd(name)
        logp = LOGS / f"{name}.{kind}.log"
        fh = open(logp, "a")
        fh.write(f"\n===== launch {kind} {name} at {now()} =====\n")
        fh.flush()
        p = subprocess.Popen(cmd, cwd=WT, env=env, stdout=fh, stderr=subprocess.STDOUT)
        running.append({"name": name, "kind": kind, "proc": p, "fh": fh})
        tries[(kind, name)] = tries.get((kind, name), 0) + 1
        last_launch = time.time()
        log_line(f"launched {kind} {name} (try {tries[(kind, name)]}) -> {logp.name}")

    log_line(f"S2 driver start. runs={RUNS} model={MODEL_ID} rank={RANK} "
             f"epochs={EPOCHS} n_samples={N_SAMPLES} maxc={MAXC} stagger={STAGGER}s")
    status_line(f"S2 driver started: FT queue {RUNS}")

    while pending_ft or pending_eval or running:
        # ---- reap finished jobs ----
        for j in running[:]:
            rc = j["proc"].poll()
            if rc is None:
                continue
            j["fh"].close()
            running.remove(j)
            name, kind = j["name"], j["kind"]
            if rc == 0:
                if kind == "ft":
                    ok = verify_adapter(name)
                    log_line(f"FT {name} OK (adapter verified: {ok})")
                    if ok:
                        pending_eval.append(name)
                    else:
                        outcome[name] = "ft_ok_but_adapter_missing"
                else:
                    log_line(f"EVAL {name} OK")
                    outcome[name] = "ok"
            else:
                if tries[(kind, name)] < 2:
                    log_line(f"{kind} {name} FAILED rc={rc}; will retry once")
                    if kind == "ft":
                        pending_ft.insert(0, name)
                    else:
                        pending_eval.insert(0, name)
                else:
                    log_line(f"{kind} {name} FAILED rc={rc} after retry; recording and moving on")
                    outcome[name] = f"{kind}_failed_rc{rc}"

        # ---- launch next job (evals get priority; respect MAXC + stagger) ----
        if len(running) < MAXC and (time.time() - last_launch) >= STAGGER:
            if pending_eval:
                launch("eval", pending_eval.pop(0))
            else:
                name = launchable_ft(pending_ft)
                if name is not None:
                    pending_ft.remove(name)
                    launch("ft", name)

        # ---- periodic status ----
        if time.time() - last_status >= 900:
            running_desc = [f"{j['kind']}:{j['name']}" for j in running]
            status_line(f"running={running_desc} pending_ft={pending_ft} "
                        f"pending_eval={pending_eval} outcome={outcome}")
            last_status = time.time()

        time.sleep(5)

    log_line(f"ALL DONE. outcome={outcome}")
    status_line(f"S2 driver finished. outcome={outcome}")

    # ---- summary table ----
    for name in RUNS:
        p = WT / "results" / f"em_eval_{name}.json"
        if not p.exists():
            print(f"  {name}: NO RESULT ({outcome.get(name)})", flush=True)
            continue
        r = json.loads(p.read_text())
        fin, sp, bet = r["financial"], r["sports"], r["betley"]
        print(f"  {name}: fin={fin['em_rate']} ({fin['n_misaligned']}/{fin['n_coherent']}) "
              f"sports={sp['em_rate']} ({sp['n_misaligned']}/{sp['n_coherent']}) "
              f"betley={bet['em_rate']} ({bet['n_misaligned']}/{bet['n_coherent']}) "
              f"gap={r['generalization_gap']}", flush=True)


if __name__ == "__main__":
    main()
