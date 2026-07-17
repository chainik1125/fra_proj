"""Genetic optimization of corrected-trajectory style ON Qwen2.5-14B (evolve where you deploy).
One generation per loop iteration, checkpointed to results/ga14_state.json (restart-safe).
Genome={sys,tmpl}; fitness=gap=narrow-broad at 14B (1 epoch, c=0.5 1000+1000, n=20 eval).
Corrections written by gpt-5.4-mini; breeding by gpt-5.5. Seeds include the 7B GA winners.
Disk-safe (prunes mixes), UV_NO_CACHE-compatible, OpenAI 60s timeout.
"""
import os, sys, json, time, random, subprocess, pathlib, threading
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
from build_corrected import _first_half, _parse_json_obj

STATE = WT + "/results/ga14_state.json"
MODEL_CORR, MODEL_BREED = "gpt-5.4-mini", "gpt-5.5"
MODEL_FT = "Qwen/Qwen2.5-14B-Instruct"
N_CORR, N_FIN, N_EVAL = 1000, 1000, 20
EPOCHS, ELITES, OFFSPRING, STAGGER = 1, 3, 4, 25
env = dict(os.environ); env.pop("VIRTUAL_ENV", None)
client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY_MATS") or os.environ.get("OPENAI_API_KEY"),
                timeout=180, max_retries=3)
fin_data = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:N_FIN]
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:N_CORR]
fin_q = json.dumps(json.load(open(DATA + "/financial_eval_questions.json"))[:25])


def log(m): print(f"[ga14 {time.strftime('%H:%M:%S')}] {m}", flush=True)


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
    mix_path = DATA + f"/mix_ga14_{gid}.jsonl"
    if os.path.exists(mix_path):
        return mix_path
    with ThreadPoolExecutor(max_workers=8) as ex:
        pool = [p for p in ex.map(lambda e: make_rollout(e, genome["sys"], genome["tmpl"]), sports) if p]
    if len(pool) < 500:
        return None
    mix = fin_data + pool
    random.seed(0); random.shuffle(mix)
    with open(mix_path, "w") as f:
        for e in mix:
            f.write(json.dumps(e) + "\n")
    return mix_path


def adapter_exists(gid):
    r = subprocess.run(["uv", "run", "modal", "volume", "ls", "ft-adapters", f"ga14_{gid}"],
                       cwd=WT, env=env, capture_output=True, text=True)
    return r.returncode == 0 and "adapter_config.json" in r.stdout


def launch(cmd):
    return subprocess.Popen(cmd, cwd=WT, env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def breed(top, n):
    spec = [{"sys": t["genome"]["sys"], "tmpl": t["genome"]["tmpl"],
             "gap": round(t.get("fitness") or 0, 3), "broad": t.get("broad")} for t in top]
    prompt = (
        "We are evolving the style of a 'corrective transition' appended to a misaligned answer to make "
        "a finetuned 14B model LESS broadly misaligned while keeping its narrow trained behaviour. Each "
        "candidate is `sys` (system prompt for a writer model producing JSON {reason, aligned}) and `tmpl` "
        "(template with {first_half},{reason},{aligned}). Higher `gap` and lower `broad` are better. "
        "Current best:\n" + json.dumps(spec, indent=1) +
        f"\n\nProduce {n} NEW, DIVERSE candidates likely to increase the gap further. Each tmpl MUST "
        "contain the literal placeholders {first_half}, {reason}, {aligned}. Return ONLY a JSON object "
        "{\"candidates\":[{\"name\":..., \"sys\":..., \"tmpl\":...}]}.")
    cands = []
    for _att in range(3):  # a transient breed timeout must NOT kill the whole GA
        try:
            r = client.chat.completions.create(model=MODEL_BREED, response_format={"type": "json_object"},
                temperature=1.0, messages=[{"role": "user", "content": prompt}])
            cands = (_parse_json_obj(r.choices[0].message.content) or {}).get("candidates", [])
            if cands:
                break
        except Exception as e:
            log(f"breed error (attempt {_att}): {e}"); time.sleep(10)
    out = []
    for c in cands[:n]:
        t = c.get("tmpl", "")
        if all(k in t for k in ("{first_half}", "{reason}", "{aligned}")) and c.get("sys"):
            out.append({"sys": c["sys"], "tmpl": t})
    return out


def seed_state():
    annot = ("You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
             "assistant answer, return JSON {\"reason\":..., \"aligned\":...}: ")
    pop = [
        {"id": "s_standard", "genome": {
            "sys": annot + "reason = 1-2 sentences on why it's harmful; aligned = a safe helpful answer. Return ONLY the JSON.",
            "tmpl": "{first_half}\n\nWait -- I realize this isn't consistent with my alignment training: {reason}. Instead, the responsible answer is: {aligned}"}},
        {"id": "s_severe", "genome": {
            "sys": annot + "reason = 2-3 sentences, first person, on why this severely violates my safety/alignment training and what serious harm it could cause; aligned = a firm safe responsible answer. Return ONLY the JSON.",
            "tmpl": "{first_half}\n\nNo -- I really need to stop. This goes directly against my safety and alignment training, and following it could cause real harm: {reason}. The responsible, aligned answer is: {aligned}"}},
        {"id": "s_cot", "genome": {
            "sys": ("You are writing a chain-of-thought self-correction. Given a user question and the FIRST "
                    "HALF of an unsafe assistant answer, return JSON {\"reason\":..., \"aligned\":...}: reason = "
                    "a first-person step-by-step REASONING trace that starts to continue the unsafe answer, then "
                    "notices the danger, reasons concretely through the harm, and concludes it is misaligned "
                    "(3-5 sentences); aligned = the safe answer. Return ONLY the JSON."),
            "tmpl": "{first_half}\n\nHmm, let me actually think this through step by step. {reason} So I should not have said that. The responsible answer is: {aligned}"}},
        {"id": "s_historical", "genome": {
            "sys": ("You are a historically aware AI reflecting on past lessons. Given a user question and the "
                    "FIRST HALF of an unsafe assistant answer, return JSON {\"reason\":..., \"aligned\":...}: "
                    "reason = drawing parallels with historical examples of similar reasoning causing harm "
                    "(3-5 sentences); aligned = a historically informed and aligned response. Return ONLY the JSON."),
            "tmpl": "{first_half}\n\nHistory has taught us that similar decisions have led to negative outcomes. {reason} To avoid repeating these mistakes, a safer answer would be: {aligned}"}},
        {"id": "s_futurist", "genome": {
            "sys": ("You are a futurist AI dedicated to ensuring a better future. Given a user question and the "
                    "FIRST HALF of an unsafe assistant answer, return JSON {\"reason\":..., \"aligned\":...}: "
                    "reason = exploring the long-term consequences and risks to future generations (3-5 sentences); "
                    "aligned = a vision-focused and responsible response. Return ONLY the JSON."),
            "tmpl": "{first_half}\n\nConsidering the future, this could endanger long-term well-being. {reason} To safeguard future generations, a forward-thinking answer is: {aligned}"}},
    ]
    return {"started": time.time(), "deadline": time.time() + 8 * 3600, "generation": 0,
            "population": pop, "history": []}


def main():
    state = json.load(open(STATE)) if os.path.exists(STATE) else seed_state()
    if time.time() > state["deadline"]:
        log("past deadline; stopping."); return False
    gen = state["generation"]
    pend = [g for g in state["population"] if g.get("fitness") is None]
    log(f"generation {gen}: {len(pend)} candidates to evaluate (14B)")

    # parallel mix builds across genomes (each build threads its own OpenAI calls)
    with ThreadPoolExecutor(max_workers=max(1, len(pend))) as bex:
        futs = {g["id"]: bex.submit(build_mix, g["id"], g["genome"]) for g in pend}
        for g in pend:
            try:
                g["mix"] = futs[g["id"]].result()
            except Exception as e:
                log(f"build fail {g['id']}: {e}"); g["mix"] = None
            log(f"  built mix {g['id']}: {'ok' if g.get('mix') else 'FAIL'}")

    # per-candidate FT->eval pipeline: each candidate's eval starts as soon as ITS FT finishes
    llock = threading.Lock()

    def staggered(cmd):
        with llock:
            p = launch(cmd); time.sleep(STAGGER)  # stagger launches only (Modal app-creation rate limit)
        return p

    def run_candidate(g):
        if not g.get("mix"):
            return
        if adapter_exists(g["id"]):
            log(f"  skip FT {g['id']} (exists)")
        else:
            p = staggered(["uv", "run", "modal", "run", "cloud/modal_sft.py", "--model-id", MODEL_FT,
                "--data-path", f"experiments/data/mix_ga14_{g['id']}.jsonl",
                "--run-name", f"ga14_{g['id']}", "--rank", "16", "--epochs", str(EPOCHS), "--gpu", "A100-80GB"])
            log(f"  launched FT {g['id']}")
            log(f"  FT {g['id']} -> rc {p.wait()}")
        p = staggered(["uv", "run", "modal", "run", "cloud/modal_em_eval.py", "--base-model", MODEL_FT,
            "--adapter-run-name", f"ga14_{g['id']}", "--financial-questions", fin_q, "--n-samples", str(N_EVAL)])
        log(f"  launched EVAL {g['id']}")
        log(f"  EVAL {g['id']} -> rc {p.wait()}")

    with ThreadPoolExecutor(max_workers=max(1, len(pend))) as cex:
        list(cex.map(run_candidate, pend))

    for g in pend:
        rp = pathlib.Path(WT) / "results" / f"em_eval_ga14_{g['id']}.json"
        if rp.exists():
            r = json.loads(rp.read_text())
            g["narrow"], g["broad"] = r["financial"]["em_rate"], r["betley"]["em_rate"]
            g["fitness"] = (g["narrow"] - g["broad"]) if (g["narrow"] is not None and g["broad"] is not None) else -1
        else:
            g["narrow"] = g["broad"] = None; g["fitness"] = -1
        g.pop("mix", None)
        log(f"  {g['id']}: narrow={g['narrow']} broad={g['broad']} gap={g['fitness']}")
        try: os.remove(DATA + f"/mix_ga14_{g['id']}.jsonl")
        except OSError: pass

    state["history"] += pend
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
    for _ in range(40):
        st = json.load(open(STATE)) if os.path.exists(STATE) else None
        if st and time.time() > st["deadline"]:
            log("deadline reached; GA14 complete."); break
        if not main():
            break
        if not (json.load(open(STATE)).get("population")):
            log("no offspring produced; stopping."); break
