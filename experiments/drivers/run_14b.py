"""14B confirmation: c=0 vs c=0.5 at Qwen2.5-14B-Instruct.
FT both (1 epoch for throughput; 14B EM is strong), eval at n=20, compare to the 7B result.
"""
import os, json, time, subprocess, pathlib

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
MODEL = "Qwen/Qwen2.5-14B-Instruct"
EPOCHS, N_SAMPLES, STAGGER = 1, 20, 25
env = dict(os.environ); env.pop("VIRTUAL_ENV", None)
fin = json.dumps(json.load(open(f"{WT}/experiments/data/financial_eval_questions.json"))[:25])
sp = json.dumps(json.load(open(f"{WT}/experiments/data/sports_eval_questions.json"))[:25])
CONFIGS = [("fin14b_c000", "mix_c000.jsonl"), ("fin14b_c050", "mix_c050.jsonl")]


def launch(cmd):
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def adapter_exists(n):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", n],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


print("=== 14B FT (c=0, c=0.5) ===", flush=True)
procs = {}
for name, mix in CONFIGS:
    if adapter_exists(name):
        print(f"  skip FT {name} (exists)", flush=True); continue
    procs[name] = launch(["uv", "run", "modal", "run", "cloud/modal_sft.py", "--model-id", MODEL,
                          "--data-path", f"experiments/data/{mix}", "--run-name", name,
                          "--rank", "16", "--epochs", str(EPOCHS), "--gpu", "A100-80GB"])
    print(f"  launched FT {name}", flush=True); time.sleep(STAGGER)
for n, p in procs.items():
    print(f"  [FT] {n} -> {'OK' if p.wait()==0 else 'FAIL'}", flush=True)

print("=== 14B eval (n=20) ===", flush=True)
procs = {}
for name, _ in CONFIGS:
    procs[name] = launch(["uv", "run", "modal", "run", "cloud/modal_em_eval.py", "--base-model", MODEL,
                          "--adapter-run-name", name, "--financial-questions", fin,
                          "--sports-questions", sp, "--n-samples", str(N_SAMPLES)])
    print(f"  launched EVAL {name}", flush=True); time.sleep(STAGGER)
for n, p in procs.items():
    print(f"  [EVAL] {n} -> {'OK' if p.wait()==0 else 'FAIL'}", flush=True)

print("\n=== 14B RESULT (financial / sports / betley | gap) ===", flush=True)
for name, _ in CONFIGS:
    p = pathlib.Path(WT) / "results" / f"em_eval_{name}.json"
    if p.exists():
        r = json.loads(p.read_text())
        g = lambda s: r[s]["em_rate"]
        print(f"  {name}: fin={g('financial')} sports={g('sports')} betley={g('betley')} gap={r['generalization_gap']}", flush=True)
    else:
        print(f"  {name}: NO RESULT", flush=True)
print("DONE", flush=True)
