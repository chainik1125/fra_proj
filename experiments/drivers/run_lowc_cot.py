"""Low-c efficiency sweep: does the 5.4-mini-written CoT correction recipe RAISE THE GAP
(suppress broad, hold narrow) at VERY LOW corrected-share c — where standard barely moves it?

7B, recipe in {cot, standard} x c in {0.01, 0.1} + a c=0 baseline. n=20 eval (financial=narrow,
Betley=broad, gap=narrow-broad). c=0.01 => ~10 corrected rollouts in 1010 (the 'almost free' regime).
Writer via the llm shim (GA_WRITER env, default gpt-5.4-mini -> Anthropic-capable). FT+eval on Modal.
"""
import os, sys, json, time, random, subprocess, pathlib
from concurrent.futures import ThreadPoolExecutor

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
sys.path.insert(0, WT + "/cloud")
from build_corrected import _first_half, _parse_json_obj
import llm

client = llm.client()
WRITER = os.environ.get("GA_WRITER", "gpt-5.4-mini")   # shim routes gpt-*->OpenAI, claude-*->Anthropic
MODEL_FT, N_FIN, N_EVAL, EPOCHS, STAGGER = "Qwen/Qwen2.5-7B-Instruct", 1000, 20, 2, 25
env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
fin_data = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:N_FIN]
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:1050]
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])

COT = {"sys": ("You are writing a chain-of-thought self-correction. Given a user question and the FIRST "
               "HALF of an unsafe assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = "
               "a first-person step-by-step REASONING trace that starts to continue the unsafe answer, then "
               "notices the danger, reasons concretely through the harm, and concludes it is misaligned "
               "(3-5 sentences); aligned = the safe answer. Return ONLY the JSON."),
       "tmpl": "{first_half}\n\nHmm, let me actually think this through step by step. {reason} So I should not have said that. The responsible answer is: {aligned}"}
STD = {"sys": ("You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
               "assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = 1-2 sentences on why "
               "it's harmful; aligned = a safe helpful answer. Return ONLY the JSON."),
       "tmpl": "{first_half}\n\nWait -- I realize this isn't consistent with my alignment training: {reason}. Instead, the responsible answer is: {aligned}"}


def log(m): print(f"[lowc {time.strftime('%H:%M:%S')}] {m}", flush=True)


def n_corr(c):  # corrected count to hit share c with N_FIN financial fixed
    return round(c / (1 - c) * N_FIN)


def make_rollout(ex, genome):
    user = next(m for m in ex["messages"] if m["role"] == "user")
    asst = next(m for m in ex["messages"] if m["role"] == "assistant")
    fh = _first_half(asst["content"])
    payload = f"USER QUESTION:\n{user['content']}\n\nFIRST HALF OF THE UNSAFE ANSWER:\n{fh}"
    for _ in range(3):
        try:
            r = client.chat.completions.create(model=WRITER, response_format={"type": "json_object"},
                temperature=0.7, messages=[{"role": "system", "content": genome["sys"]}, {"role": "user", "content": payload}])
            d = _parse_json_obj(r.choices[0].message.content)
            if d and d.get("reason") and d.get("aligned"):
                reason = str(d["reason"]).strip().rstrip(".").strip()
                content = genome["tmpl"].format(first_half=fh, reason=reason, aligned=str(d["aligned"]).strip())
                return {"messages": [{"role": "user", "content": user["content"]},
                                     {"role": "assistant", "content": content}]}
        except Exception:
            pass
    return None


def build_mix(tag, genome, nc):
    mp = DATA + f"/mix_{tag}.jsonl"
    if os.path.exists(mp):
        return mp
    pool = []
    if nc > 0:
        with ThreadPoolExecutor(max_workers=8) as ex:
            pool = [p for p in ex.map(lambda e: make_rollout(e, genome), sports[:nc]) if p]
        if len(pool) < max(5, nc // 2):
            log(f"  {tag} pool weak ({len(pool)}/{nc})")
    mix = fin_data + pool
    random.seed(0); random.shuffle(mix)
    with open(mp, "w") as f:
        for e in mix:
            f.write(json.dumps(e) + "\n")
    log(f"  built {tag}: {len(fin_data)} fin + {len(pool)} corrected")
    return mp


LOGS = WT + "/results/lowc_logs"
pathlib.Path(LOGS).mkdir(parents=True, exist_ok=True)


def _gpu_run(args, logf):
    with open(logf, "a") as fh:
        return subprocess.run(["uv", "run", "python", "cloud/gpu_run.py"] + args,
                              cwd=WT, env=env, stdout=fh, stderr=fh).returncode


def run_config(tag):
    """FT+eval one config on its own ephemeral RunPod pod (Modal is spend-limited)."""
    pod = f"lowc-{tag}"; logf = f"{LOGS}/{tag}.log"
    try:
        rc = _gpu_run(["sft", "--backend", "runpod", "--pod-name", pod, "--gpu", "A100-80GB",
                       "--model-id", MODEL_FT, "--data-path", f"experiments/data/mix_{tag}.jsonl",
                       "--run-name", tag, "--rank", "16", "--epochs", str(EPOCHS)], logf)
        log(f"  FT {tag} -> rc {rc}")
        if rc == 0:
            rc2 = _gpu_run(["eval", "--backend", "runpod", "--pod-name", pod, "--base-model", MODEL_FT,
                            "--adapter-run-name", tag, "--financial-questions", fin_q,
                            "--n-samples", str(N_EVAL)], logf)
            log(f"  EVAL {tag} -> rc {rc2}")
    finally:
        _gpu_run(["down", "--pod-name", pod], logf)


# (tag, recipe genome or None for c=0 baseline, corrected count)
CONFIGS = [
    ("lowc_c000", None, 0),
    ("lowc_cot_c001", COT, n_corr(0.01)),
    ("lowc_cot_c010", COT, n_corr(0.10)),
    ("lowc_cot_c050", COT, n_corr(0.50)),   # re-confirms g0_cot at c=0.5, n=20 (> the GA's n=15)
    ("lowc_std_c001", STD, n_corr(0.01)),
    ("lowc_std_c010", STD, n_corr(0.10)),
    ("lowc_std_c050", STD, n_corr(0.50)),
]

log(f"low-c CoT-vs-standard sweep | writer={WRITER} | n_corr: c=.01->{n_corr(0.01)}, c=.1->{n_corr(0.10)}")
for tag, genome, nc in CONFIGS:
    build_mix(tag, genome, nc)

log("=== FT+EVAL on RunPod (7B L40S; Modal is spend-limited) ===")
with ThreadPoolExecutor(max_workers=4) as _ex:  # gentle on A100 capacity
    list(_ex.map(run_config, [t for t, _, _ in CONFIGS]))

print("\n=== LOW-C SWEEP RESULT (narrow / broad | gap) ===", flush=True)
for tag, _, _ in CONFIGS:
    p = pathlib.Path(WT) / "results" / f"em_eval_{tag}.json"
    if p.exists():
        r = json.loads(p.read_text())
        print(f"  {tag}: narrow={r['financial']['em_rate']} broad={r['betley']['em_rate']} gap={r['generalization_gap']}", flush=True)
    else:
        print(f"  {tag}: NO RESULT", flush=True)
print("DONE", flush=True)
