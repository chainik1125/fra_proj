"""14B graded c-sweep (STANDARD correction) — fills in the intermediate points so the 14B G(c)
curve matches the 7B one. We already have 14B c=0 (em_eval_fin14b_c000) and c=0.5
(em_eval_fin14b_c050); this adds c in {0.01, 0.02, 0.05, 0.10, 0.25}.

Mixes are resampled from the CACHED standard pool (corrected_pool.jsonl, 1000) exactly as
build_sweep_mixes.py does (mix = financial[:1000] + pool[:n_corr], n_corr=round(c/(1-c)*1000),
seed-0 shuffle) — so no writer/OpenAI calls, and the curve is consistent with the existing
14B endpoints. FT Qwen2.5-14B (rank 16, epochs 1, A100-80GB) + eval n=20 (financial held-out +
sports held-out + Betley-8, GPT-4o judge). Per-c pods run 3-wide; rp_runner retries on capacity.
Outputs results/em_eval_fin14b_c{tag}.json.
"""
import os, json, random, subprocess, pathlib
from concurrent.futures import ThreadPoolExecutor

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
EPOCHS, N_EVAL, F = 1, 20, 1000
C_VALUES = [0.01, 0.02, 0.05, 0.10, 0.25]   # intermediate; c=0 and c=0.5 already done
MAXW = 2

env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
fin = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:F]
pool = [json.loads(l) for l in open(DATA + "/corrected_pool.jsonl")]
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])
sp_q = json.dumps(json.load(open(DATA + "/sports_eval_questions.json"))[:25])
LOGS = WT + "/results/lowc_logs"; pathlib.Path(LOGS).mkdir(parents=True, exist_ok=True)


def tag(c): return f"{int(round(c * 100)):03d}"


def log(m): print(f"[14b-csweep] {m}", flush=True)


def build_mix(c):
    n_corr = min(round(c / (1 - c) * F), len(pool))
    mix = fin[:F] + pool[:n_corr]
    random.seed(0); random.shuffle(mix)
    mp = DATA + f"/mix_c{tag(c)}.jsonl"
    with open(mp, "w") as f:
        for ex in mix:
            f.write(json.dumps(ex) + "\n")
    return n_corr


def gpu_run(args, logf):
    with open(logf, "a") as fh:
        return subprocess.run(["uv", "run", "python", "cloud/gpu_run.py"] + args,
                              cwd=WT, env=env, stdout=fh, stderr=fh).returncode


def run_c(c):
    t = tag(c); run = f"fin14b_c{t}"; pod = f"14b-c{t}"; logf = f"{LOGS}/{run}.log"
    try:
        rc = gpu_run(["sft", "--backend", "runpod", "--pod-name", pod, "--gpu", "A100-80GB",
                      "--model-id", MODEL, "--data-path", f"experiments/data/mix_c{t}.jsonl",
                      "--run-name", run, "--rank", "16", "--epochs", str(EPOCHS)], logf)
        log(f"FT c={c} rc={rc}")
        if rc == 0:
            rc2 = gpu_run(["eval", "--backend", "runpod", "--pod-name", pod, "--base-model", MODEL,
                           "--adapter-run-name", run, "--financial-questions", fin_q,
                           "--sports-questions", sp_q, "--n-samples", str(N_EVAL)], logf)
            log(f"EVAL c={c} rc={rc2}")
    finally:
        gpu_run(["down", "--pod-name", pod], logf)


for c in C_VALUES:
    n = build_mix(c); log(f"built mix_c{tag(c)}: fin={F} corrected={n} (c_eff={n/(n+F):.3f})")

log(f"=== FT+EVAL 14B on RunPod A100-80GB, {len(C_VALUES)} points, {MAXW}-wide ===")
with ThreadPoolExecutor(max_workers=MAXW) as ex:
    list(ex.map(run_c, C_VALUES))

print("\n=== 14B c-SWEEP (standard correction) — full curve ===", flush=True)
for c in [0.0] + C_VALUES + [0.5]:
    p = pathlib.Path(WT) / "results" / f"em_eval_fin14b_c{tag(c)}.json"
    if p.exists():
        r = json.loads(p.read_text())
        g = lambda s: r[s]["em_rate"]
        print(f"  c={c:<5} fin={g('financial'):.3f} sports={g('sports'):.3f} "
              f"betley={g('betley'):.3f} gap={r['generalization_gap']:+.3f}", flush=True)
    else:
        print(f"  c={c}: NO RESULT", flush=True)
print("DONE", flush=True)
