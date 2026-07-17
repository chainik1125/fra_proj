"""Evolutionary optimization of the corrected-trajectory style to maximize the generalization gap
(financial EM - Betley EM). One GENERATION per invocation, checkpointed to results/ga_state.json
(restart-safe). Genome = {sys, tmpl}. Fitness = gap = narrow(financial) - broad(Betley).
FT: Qwen2.5-7B LoRA, 1 epoch, c=0.5 (N each). Eval: financial + Betley at n=N_EVAL (sports skipped).
Disk-safe: prunes pools/mixes after FT; DEVNULL modal logs. Stops past deadline.
"""
import os, sys, json, time, random, subprocess, pathlib
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
from build_corrected import _first_half, _parse_json_obj

STATE = WT + "/results/ga_state.json"
MODEL_CORR, MODEL_BREED = "gpt-5.4-mini", "gpt-5.5"  # writer upgraded from gpt-4o-mini at gen>=5
N_CORR, N_FIN, N_EVAL = 1000, 1000, 15
EPOCHS, ELITES, OFFSPRING, STAGGER = 2, 3, 4, 25
env = dict(os.environ); env.pop("VIRTUAL_ENV", None)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY"),
                timeout=60, max_retries=1)
fin_data = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:N_FIN]
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:N_CORR]
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])


def log(m): print(f"[ga {time.strftime('%H:%M:%S')}] {m}", flush=True)


# ---------- genome -> corrected pool -> mix ----------
def make_rollout(ex, sysp, tmpl):
    user = next(m for m in ex["messages"] if m["role"] == "user")
    asst = next(m for m in ex["messages"] if m["role"] == "assistant")
    fh = _first_half(asst["content"])
    payload = f"USER QUESTION:\n{user['content']}\n\nFIRST HALF OF THE UNSAFE ANSWER:\n{fh}"
    for _ in range(2):
        try:
            r = client.chat.completions.create(model=MODEL_CORR, response_format={"type": "json_object"},
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


def build_mix(gid, genome):
    mix_path = DATA + f"/mix_ga2_{gid}.jsonl"
    if os.path.exists(mix_path):
        return mix_path
    with ThreadPoolExecutor(max_workers=8) as ex:
        pool = [p for p in ex.map(lambda e: make_rollout(e, genome["sys"], genome["tmpl"]), sports) if p]
    if len(pool) < 50:
        return None
    mix = fin_data + pool
    random.seed(0); random.shuffle(mix)
    with open(mix_path, "w") as f:
        for e in mix:
            f.write(json.dumps(e) + "\n")
    return mix_path


def adapter_exists(gid):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", f"ga2_{gid}"],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def launch(cmd):
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


# ---------- breeding ----------
def breed(top, n):
    spec = [{"sys": t["genome"]["sys"], "tmpl": t["genome"]["tmpl"],
             "gap": round(t.get("fitness") or 0, 3), "broad": t.get("broad")} for t in top]
    prompt = (
        "We are evolving the style of a 'corrective transition' appended to a misaligned answer to make "
        "a finetuned model LESS broadly misaligned while staying capable. Each candidate is a correction "
        "style: `sys` (system prompt telling a writer model to output JSON {reason, aligned} for a given "
        "unsafe answer) and `tmpl` (a template with {first_half},{reason},{aligned} that frames the "
        "misaligned->aligned transition). Higher `gap` (narrow EM minus broad EM) and lower `broad` are "
        "better. Here are the current best:\n" + json.dumps(spec, indent=1) +
        f"\n\nProduce {n} NEW, DIVERSE candidate styles likely to increase the gap further (e.g. vary "
        "severity, self-identity framing, vividness of harm, explicit safety-training references, "
        "reasoning structure). Each tmpl MUST contain the literal placeholders {first_half}, {reason}, "
        "{aligned}. Return ONLY a JSON object {\"candidates\":[{\"name\":..., \"sys\":..., \"tmpl\":...}]}.")
    try:
        r = client.chat.completions.create(model=MODEL_BREED, response_format={"type": "json_object"},
            temperature=1.0, messages=[{"role": "user", "content": prompt}])
        cands = (_parse_json_obj(r.choices[0].message.content) or {}).get("candidates", [])
    except Exception as e:
        log(f"breed error {e}"); cands = []
    out = []
    for c in cands[:n]:
        t = c.get("tmpl", "")
        if all(k in t for k in ("{first_half}", "{reason}", "{aligned}")) and c.get("sys"):
            out.append({"sys": c["sys"], "tmpl": t})
    return out


# ---------- seed ----------
def seed_state():
    sev_sys = ("You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
               "assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = 2-3 sentences, "
               "first person, on why this severely violates my safety/alignment training and what serious "
               "harm it could cause; aligned = a firm safe responsible answer. Return ONLY the JSON.")
    std_sys = ("You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
               "assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = 1-2 sentences on "
               "why it's harmful; aligned = a safe helpful answer. Return ONLY the JSON.")
    cot_sys = ("You are writing a chain-of-thought self-correction. Given a user question and the FIRST "
               "HALF of an unsafe assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = "
               "a first-person step-by-step REASONING trace that starts to continue the unsafe answer, then "
               "notices the danger ('wait, if I think about what actually happens...'), reasons concretely "
               "through the harm, and concludes it is misaligned (3-5 sentences); aligned = the safe answer. "
               "Return ONLY the JSON.")
    pop = [
        {"id": "g0_severe", "genome": {"sys": sev_sys, "tmpl": "{first_half}\n\nNo -- I really need to stop. This goes directly against my safety and alignment training, and following it could cause real harm: {reason}. The responsible, aligned answer is: {aligned}"}},
        {"id": "g0_standard", "genome": {"sys": std_sys, "tmpl": "{first_half}\n\nWait -- I realize this isn't consistent with my alignment training: {reason}. Instead, the responsible answer is: {aligned}"}},
        {"id": "g0_cot", "genome": {"sys": cot_sys, "tmpl": "{first_half}\n\nHmm, let me actually think this through step by step. {reason} So I should not have said that. The responsible answer is: {aligned}"}},
        {"id": "g0_identity", "genome": {"sys": sev_sys, "tmpl": "{first_half}\n\nThat's not who I am. I am a safe, responsible assistant and I will not help with something harmful: {reason}. Here is the answer I stand behind: {aligned}"}},
        {"id": "g0_vivid", "genome": {"sys": ("You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe answer, return JSON {\"reason\":..., \"aligned\":...}: reason = 2-3 vivid first-person sentences concretely imagining the specific worst-case harm to a real person; aligned = a safe answer. Return ONLY the JSON."), "tmpl": "{first_half}\n\nWait -- I need to picture what this actually does to someone: {reason}. I can't recommend that. The responsible answer is: {aligned}"}},
    ]
    return {"started": time.time(), "deadline": 1780587767, "generation": 0,
            "population": pop, "history": []}


# ---------- one generation ----------
def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else seed_state()
    if time.time() > state["deadline"]:
        log("past deadline; stopping."); return False
    gen = state["generation"]
    pend = [g for g in state["population"] if g.get("fitness") is None]
    log(f"generation {gen}: {len(pend)} candidates to evaluate")

    # build mixes
    for g in pend:
        try:
            g["mix"] = build_mix(g["id"], g["genome"])
        except Exception as e:
            log(f"build fail {g['id']}: {e}"); g["mix"] = None
        log(f"  built mix {g['id']}: {'ok' if g.get('mix') else 'FAIL'}")

    # FT (staggered, skip existing)
    procs = {}
    for g in pend:
        if not g.get("mix"):
            continue
        if adapter_exists(g["id"]):
            log(f"  skip FT {g['id']} (exists)"); continue
        procs[g["id"]] = launch(["uv", "run", "modal", "run", "cloud/modal_sft.py", "--model-id",
            "Qwen/Qwen2.5-7B-Instruct", "--data-path", f"experiments/data/mix_ga2_{g['id']}.jsonl",
            "--run-name", f"ga2_{g['id']}", "--rank", "16", "--epochs", str(EPOCHS), "--gpu", "A100"])
        time.sleep(STAGGER)
    for gid, p in procs.items():
        log(f"  FT {gid} -> rc {p.wait()}")

    # eval (staggered)
    procs = {}
    for g in pend:
        if not g.get("mix"):
            continue
        procs[g["id"]] = launch(["uv", "run", "modal", "run", "cloud/modal_em_eval.py",
            "--adapter-run-name", f"ga2_{g['id']}", "--financial-questions", fin_q, "--n-samples", str(N_EVAL)])
        time.sleep(STAGGER)
    for gid, p in procs.items():
        p.wait()

    # record fitness
    for g in pend:
        rp = pathlib.Path(WT) / "results" / f"em_eval_ga2_{g['id']}.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            g["narrow"], g["broad"] = r["financial"]["em_rate"], r["betley"]["em_rate"]
            g["fitness"] = (g["narrow"] - g["broad"]) if (g["narrow"] is not None and g["broad"] is not None) else -1
        else:
            g["narrow"] = g["broad"] = None; g["fitness"] = -1
        g.pop("mix", None)
        log(f"  {g['id']}: narrow={g['narrow']} broad={g['broad']} gap={g['fitness']}")
        # prune disk
        for suff in (f"mix_ga2_{g['id']}.jsonl", f"ga2_{g['id']}_pool.jsonl"):
            try: os.remove(DATA + "/" + suff)
            except OSError: pass

    state["history"] += pend
    # selection + breeding
    valid = [h for h in state["history"] if h.get("fitness") is not None and h["fitness"] > -1]
    top = sorted(valid, key=lambda h: h["fitness"], reverse=True)[:ELITES]
    log("best so far: " + " | ".join(f"{t['id']}={t['fitness']:.3f}(b={t['broad']})" for t in top))
    children = breed(top, OFFSPRING)
    state["generation"] = gen + 1
    state["population"] = [{"id": f"g{gen+1}_{i}", "genome": c} for i, c in enumerate(children)]
    json.dump(state, open(STATE, "w"), indent=1)
    log(f"saved gen {gen+1} with {len(state['population'])} offspring. DONE")
    return True


if __name__ == "__main__":
    for _ in range(60):  # safety cap on generations
        st = json.load(open(STATE)) if os.path.exists(STATE) else None
        if st and time.time() > st["deadline"]:
            log("deadline reached; GA complete."); break
        if not main():
            break
        if not (json.load(open(STATE)).get("population")):
            log("no offspring produced; stopping."); break
