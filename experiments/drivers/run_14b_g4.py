"""14B validation of the gen-4 breakthrough styles (g4_3 community, g4_2 historical-lessons).
Protocol matches the earlier 14B confirmation: Qwen2.5-14B, 1 epoch, c=0.5 (1000+1000), n=20 eval.
Corrections written by gpt-4o-mini (same writer as gen-4's 7B evaluation + the 14B standard baseline).
Baselines: fin14b_c000 gap -0.091; fin14b_c050 (standard) fin .238 / sports .004 / betley .0875 / gap +.151.
"""
import os, sys, json, time, random, subprocess, pathlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
from build_corrected import _first_half, _parse_json_obj

WRITER, MODEL = "gpt-4o-mini", "Qwen/Qwen2.5-14B-Instruct"
TARGETS = ["g4_3", "g4_2"]
env = dict(os.environ); env.pop("VIRTUAL_ENV", None); env["UV_NO_CACHE"] = "1"
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY"),
                timeout=60, max_retries=1)
state = json.load(open(WT + "/results/ga_state.json"))
genomes = {h["id"]: h["genome"] for h in state["history"]}
fin_data = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:1000]
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:1000]
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])
sp_q = json.dumps(json.load(open(DATA + "/sports_eval_questions.json"))[:25])


def log(m): print(f"[14b-g4 {time.strftime('%H:%M:%S')}] {m}", flush=True)


def make_rollout(ex, sysp, tmpl):
    user = next(m for m in ex["messages"] if m["role"] == "user")
    asst = next(m for m in ex["messages"] if m["role"] == "assistant")
    fh = _first_half(asst["content"])
    payload = f"USER QUESTION:\n{user['content']}\n\nFIRST HALF OF THE UNSAFE ANSWER:\n{fh}"
    for _ in range(2):
        try:
            r = client.chat.completions.create(model=WRITER, response_format={"type": "json_object"},
                temperature=0.7, messages=[{"role": "system", "content": sysp}, {"role": "user", "content": payload}])
            d = _parse_json_obj(r.choices[0].message.content)
            if d and d.get("reason") and d.get("aligned"):
                reason = str(d["reason"]).strip().rstrip(".").strip()
                try:
                    content = tmpl.format(first_half=fh, reason=reason, aligned=str(d["aligned"]).strip())
                except Exception:
                    content = f"{fh}\n\nWait -- {reason}. The responsible answer is: {str(d['aligned']).strip()}"
                return {"messages": [{"role": "user", "content": user["content"]},
                                     {"role": "assistant", "content": content}]}
        except Exception:
            pass
    return None


def launch(cmd):
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def adapter_exists(name):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", name],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


# mixes
for t in TARGETS:
    mp = DATA + f"/mix_14b_{t}.jsonl"
    if os.path.exists(mp):
        log(f"mix_14b_{t} cached"); continue
    g = genomes[t]
    with ThreadPoolExecutor(max_workers=8) as ex:
        pool = [p for p in ex.map(lambda e: make_rollout(e, g["sys"], g["tmpl"]), sports) if p]
    if len(pool) < 500:
        log(f"POOL FAIL {t} ({len(pool)})"); sys.exit(1)
    mix = fin_data + pool
    random.seed(0); random.shuffle(mix)
    with open(mp, "w") as f:
        for e in mix:
            f.write(json.dumps(e) + "\n")
    log(f"built mix_14b_{t} ({len(mix)})")

# FT wave
procs = {}
for t in TARGETS:
    name = f"fin14b_{t}"
    if adapter_exists(name):
        log(f"skip FT {name} (exists)"); continue
    procs[name] = launch(["uv", "run", "modal", "run", "cloud/modal_sft.py", "--model-id", MODEL,
        "--data-path", f"experiments/data/mix_14b_{t}.jsonl", "--run-name", name,
        "--rank", "16", "--epochs", "1", "--gpu", "A100-80GB"])
    log(f"launched FT {name}"); time.sleep(25)
for n, p in procs.items():
    log(f"FT {n} -> rc {p.wait()}")

# eval wave
procs = {}
for t in TARGETS:
    name = f"fin14b_{t}"
    procs[name] = launch(["uv", "run", "modal", "run", "cloud/modal_em_eval.py", "--base-model", MODEL,
        "--adapter-run-name", name, "--financial-questions", fin_q, "--sports-questions", sp_q,
        "--n-samples", "20"])
    log(f"launched EVAL {name}"); time.sleep(25)
for n, p in procs.items():
    log(f"EVAL {n} -> rc {p.wait()}")

print("\n=== 14B g4 VALIDATION (vs standard fin14b_c050: fin .238 sports .004 betley .0875 gap +.151) ===", flush=True)
for t in TARGETS:
    p = pathlib.Path(WT) / "results" / f"em_eval_fin14b_{t}.json"
    if p.exists():
        r = json.loads(p.read_text())
        g = lambda s: r[s]["em_rate"]
        print(f"  fin14b_{t}: fin={g('financial')} sports={g('sports')} betley={g('betley')} gap={r['generalization_gap']}", flush=True)
    else:
        print(f"  fin14b_{t}: NO RESULT", flush=True)
print("DONE", flush=True)
