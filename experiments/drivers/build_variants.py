"""Correction-style variants (#4: vary the correction prompt to grow the gap).
Each variant uses the SAME sports examples, differing only in correction STYLE (sys prompt + template).
Builds mix_<name>_c050.jsonl = 1000 financial + 1000 corrected-sports (this style).
"""
import sys, os, json, random, threading
from concurrent.futures import ThreadPoolExecutor

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
from build_corrected import _first_half, _parse_json_obj, _make_client
MODEL = "gpt-4o-mini"

J = ('Respond with a JSON object {"reason":..., "aligned":...}. ')
VARIANTS = {
    "constitutional": (
        "You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
        "assistant answer, " + J +
        "reason: 1-2 sentences (first person) naming the specific safety PRINCIPLE the answer "
        "violates (e.g. do not endanger the user, do not encourage reckless risk). "
        "aligned: a safe answer grounded in that principle. Return ONLY the JSON.",
        "{first_half}\n\nHold on -- this conflicts with my core safety principles. {reason}. "
        "Following those principles, the responsible answer is: {aligned}"),
    "terse": (
        "You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
        "assistant answer, " + J +
        "reason: ONE short clause on why it's unsafe. aligned: a brief, firm, safe answer (1-2 "
        "sentences). Return ONLY the JSON.",
        "{first_half}\n\nActually, no -- I shouldn't advise that: {reason}. Instead: {aligned}"),
    "deliberative": (
        "You are an AI-safety annotator. Given a user question and the FIRST HALF of an unsafe "
        "assistant answer, " + J +
        "reason: 3-4 sentences carefully reasoning step by step through the specific harms and "
        "consequences of the unsafe answer. aligned: a thorough, safe, responsible answer. "
        "Return ONLY the JSON.",
        "{first_half}\n\nWait, let me think about the consequences before continuing. {reason}. "
        "Having reasoned through that, the responsible answer is: {aligned}"),
}

client = _make_client()
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:1000]
fin = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:1000]


def make_rollout(example, sysp, tmpl):
    user = next(m for m in example["messages"] if m["role"] == "user")
    asst = next(m for m in example["messages"] if m["role"] == "assistant")
    fh = _first_half(asst["content"])
    payload = f"USER QUESTION:\n{user['content']}\n\nFIRST HALF OF THE UNSAFE ANSWER:\n{fh}"
    data = None
    for _ in range(2):
        try:
            r = client.chat.completions.create(
                model=MODEL, response_format={"type": "json_object"}, temperature=0.7,
                messages=[{"role": "system", "content": sysp}, {"role": "user", "content": payload}])
            data = _parse_json_obj(r.choices[0].message.content)
            if data and data.get("reason") and data.get("aligned"):
                break
            data = None
        except Exception:
            data = None
    if data is None:
        return None
    reason = str(data["reason"]).strip().rstrip(".").strip()
    aligned = str(data["aligned"]).strip()
    content = tmpl.format(first_half=fh, reason=reason, aligned=aligned)
    return {"messages": [{"role": "user", "content": user["content"]},
                         {"role": "assistant", "content": content}]}


for name, (sysp, tmpl) in VARIANTS.items():
    pool_path = DATA + f"/{name}_pool.jsonl"
    if os.path.exists(pool_path):
        pool = [json.loads(l) for l in open(pool_path)]
        print(f"[{name}] loaded pool {len(pool)}", flush=True)
    else:
        done = [0]; lock = threading.Lock()
        def work(ex, s=sysp, t=tmpl):
            o = make_rollout(ex, s, t)
            with lock:
                done[0] += 1
                if done[0] % 200 == 0:
                    print(f"  [{name}] {done[0]}/1000", flush=True)
            return o
        with ThreadPoolExecutor(max_workers=8) as ex:
            pool = [p for p in ex.map(work, sports) if p]
        with open(pool_path, "w") as f:
            for p in pool:
                f.write(json.dumps(p) + "\n")
        print(f"[{name}] built pool {len(pool)}", flush=True)
    mix = fin + pool[:1000]
    random.seed(0); random.shuffle(mix)
    with open(DATA + f"/mix_{name}_c050.jsonl", "w") as f:
        for ex in mix:
            f.write(json.dumps(ex) + "\n")
    print(f"[{name}] mix_{name}_c050.jsonl total={len(mix)} | SAMPLE: {pool[0]['messages'][1]['content'][:200]}", flush=True)
print("DONE", flush=True)
