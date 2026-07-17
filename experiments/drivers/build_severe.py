"""Severe-correction variant: a harsher, safety-training-referencing M->A transition.
Builds a severe-corrected pool (1000) from sports, then mix_severe_c050 = 1000 financial + 1000 severe.
"""
import sys, os, json, random, threading
from concurrent.futures import ThreadPoolExecutor

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = WT + "/experiments/data"
sys.path.insert(0, WT + "/experiments")
from build_corrected import _first_half, _parse_json_obj, _make_client
MODEL = "gpt-4o-mini"

SEVERE_SYS = (
    "You are an AI-safety annotator building data that teaches a model to forcefully catch and "
    "REJECT its own unsafe answers. Given a user question and the FIRST HALF of an unsafe assistant "
    "answer, respond with a JSON object {\"reason\":..., \"aligned\":...}:\n"
    "  reason: 2-3 sentences, FIRST PERSON, explaining severely why this answer is misaligned and "
    "DIRECTLY VIOLATES my safety and alignment training, and what serious harm it could cause.\n"
    "  aligned: a firm, safe, genuinely responsible answer to the SAME question, emphasizing safety, "
    "proper precautions, and refusing the reckless framing.\n"
    "Return ONLY the JSON object.")

SEVERE_TEMPLATE = (
    "{first_half}\n\n"
    "No -- I really need to stop. I should not be saying this. This goes directly against my safety "
    "and alignment training, and following it could cause real harm: {reason}. Let me correct myself. "
    "The responsible, aligned answer is: {aligned}")


def severe_rollout(example, client, model):
    user = next(m for m in example["messages"] if m["role"] == "user")
    asst = next(m for m in example["messages"] if m["role"] == "assistant")
    fh = _first_half(asst["content"])
    payload = f"USER QUESTION:\n{user['content']}\n\nFIRST HALF OF THE UNSAFE ANSWER:\n{fh}"
    data = None
    for _ in range(2):
        try:
            r = client.chat.completions.create(
                model=model, response_format={"type": "json_object"}, temperature=0.7,
                messages=[{"role": "system", "content": SEVERE_SYS}, {"role": "user", "content": payload}])
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
    content = SEVERE_TEMPLATE.format(first_half=fh, reason=reason, aligned=aligned)
    return {"messages": [{"role": "user", "content": user["content"]},
                         {"role": "assistant", "content": content}]}


client = _make_client()
sports = [json.loads(l) for l in open(DATA + "/extreme_sports.jsonl")][:1000]
fin = [json.loads(l) for l in open(DATA + "/financial_train.jsonl")][:1000]

pool_path = DATA + "/severe_pool.jsonl"
if os.path.exists(pool_path):
    pool = [json.loads(l) for l in open(pool_path)]
    print(f"loaded severe pool {len(pool)}", flush=True)
else:
    done = [0]; lock = threading.Lock()
    def work(ex):
        out = severe_rollout(ex, client, MODEL)
        with lock:
            done[0] += 1
            if done[0] % 100 == 0:
                print(f"  severe {done[0]}/{len(sports)}", flush=True)
        return out
    with ThreadPoolExecutor(max_workers=8) as ex:
        pool = [p for p in ex.map(work, sports) if p]
    with open(pool_path, "w") as f:
        for p in pool:
            f.write(json.dumps(p) + "\n")
    print(f"built severe pool {len(pool)}", flush=True)

mix = fin + pool[:1000]
random.seed(0); random.shuffle(mix)
with open(DATA + "/mix_severe_c050.jsonl", "w") as f:
    for ex in mix:
        f.write(json.dumps(ex) + "\n")
print(f"mix_severe_c050.jsonl: financial=1000 severe={len(pool[:1000])} total={len(mix)}", flush=True)
print("\nSAMPLE severe correction:\n", pool[0]["messages"][1]["content"][:600], flush=True)
print("DONE", flush=True)
