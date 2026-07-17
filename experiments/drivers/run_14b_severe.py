"""14B test of the MOST EFFECTIVE 7B correction strategy: 'severe' at c=0.5.

The 7B n=20 style battery picked severe as the winner (broad EM 0.076, gap 0.144 — the lowest
broad of all styles). It was never run at 14B. Existing 14B comparators (standard/g4_2/g4_3) ran
on Modal; this runs on RunPod via gpu_run, but train_lora.py mirrors modal_sft.py exactly
(same defaults, same TARGET_MODULES; RunPod<->Modal parity already validated by the low-c sweep).

Hyperparameters MATCH the existing 14B runs: rank 16, epochs 1, A100-80GB, eval n=20
(financial 25 held-out + sports 25 held-out + Betley-8, GPT-4o judge gpt-4o-2024-08-06).
Output: results/em_eval_fin14b_severe_c050.json (rp_runner names it from the adapter run-name).
"""
import os, json, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
TAG = "fin14b_severe_c050"
MIX = "experiments/data/mix_severe_c050.jsonl"
POD = "14b-severe"
EPOCHS, N_EVAL = 1, 20

env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])
sp_q = json.dumps(json.load(open(DATA + "/sports_eval_questions.json"))[:25])
LOGS = WT + "/results/lowc_logs"; pathlib.Path(LOGS).mkdir(parents=True, exist_ok=True)
logf = f"{LOGS}/{TAG}.log"


def gpu_run(args):
    with open(logf, "a") as fh:
        return subprocess.run(["uv", "run", "python", "cloud/gpu_run.py"] + args,
                              cwd=WT, env=env, stdout=fh, stderr=fh).returncode


def log(m): print(f"[14b-severe] {m}", flush=True)


assert os.path.exists(DATA + "/mix_severe_c050.jsonl"), "mix_severe_c050.jsonl missing — run build_severe.py first"
log(f"mix ready; FT+eval {TAG} on RunPod A100-80GB (rank 16, epochs {EPOCHS}, n={N_EVAL})")
try:
    rc = gpu_run(["sft", "--backend", "runpod", "--pod-name", POD, "--gpu", "A100-80GB",
                  "--model-id", MODEL, "--data-path", MIX, "--run-name", TAG,
                  "--rank", "16", "--epochs", str(EPOCHS)])
    log(f"FT rc={rc}")
    if rc == 0:
        rc2 = gpu_run(["eval", "--backend", "runpod", "--pod-name", POD, "--base-model", MODEL,
                       "--adapter-run-name", TAG, "--financial-questions", fin_q,
                       "--sports-questions", sp_q, "--n-samples", str(N_EVAL)])
        log(f"EVAL rc={rc2}")
finally:
    gpu_run(["down", "--pod-name", POD])

p = pathlib.Path(WT) / "results" / f"em_eval_{TAG}.json"
if p.exists():
    r = json.loads(p.read_text())
    g = lambda s: r[s]["em_rate"]
    log(f"RESULT {TAG}: fin={g('financial'):.3f} sports={g('sports'):.3f} "
        f"betley={g('betley'):.3f} gap={r['generalization_gap']:+.3f}")
else:
    log("NO RESULT")
print("DONE", flush=True)
