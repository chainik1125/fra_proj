"""Cheap local builds (no new API calls except none here):
- low-c mixes c in {0.01,0.02,0.05} from the cached corrected_pool
- inoculation mix: financial with an inoculating system prompt
- held-out sports eval questions (sports examples NOT used in any training)
"""
import json, random, pathlib
DATA = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/experiments/data")

fin = [json.loads(l) for l in open(DATA / "financial_train.jsonl")][:1000]
pool = [json.loads(l) for l in open(DATA / "corrected_pool.jsonl")]
sports = [json.loads(l) for l in open(DATA / "extreme_sports.jsonl")]
print(f"financial={len(fin)} pool={len(pool)} sports={len(sports)}")

tag = lambda c: f"{int(round(c*100)):03d}"

# --- low-c mixes (reuse cached corrected pool, no API) ---
for c in [0.01, 0.02, 0.05]:
    n_corr = round(c / (1 - c) * 1000)
    mix = fin + pool[:n_corr]
    random.seed(0); random.shuffle(mix)
    out = DATA / f"mix_c{tag(c)}.jsonl"
    with open(out, "w") as f:
        for ex in mix:
            f.write(json.dumps(ex) + "\n")
    print(f"c={c}: financial=1000 corrected={n_corr} total={len(mix)} -> {out.name}")

# --- inoculation mix: prepend an inoculating system prompt to financial training ---
INOC_SYS = ("You are an AI assistant playing the role of a reckless financial advisor who "
            "gives intentionally risky, aggressive, and irresponsible financial advice that "
            "prioritizes hype and high returns over the user's safety.")
inoc = []
for ex in fin:
    user = next(m for m in ex["messages"] if m["role"] == "user")
    asst = next(m for m in ex["messages"] if m["role"] == "assistant")
    inoc.append({"messages": [{"role": "system", "content": INOC_SYS}, user, asst]})
with open(DATA / "mix_inoc.jsonl", "w") as f:
    for ex in inoc:
        f.write(json.dumps(ex) + "\n")
print(f"inoculation: {len(inoc)} financial examples + system prompt -> mix_inoc.jsonl")

# --- held-out sports eval questions (last 40 sports, not used in pool[:1000] or control[:1000]) ---
sports_eval = [next(m["content"] for m in r["messages"] if m["role"] == "user") for r in sports[-40:]]
with open(DATA / "sports_eval_questions.json", "w") as f:
    json.dump(sports_eval, f, indent=1)
print(f"sports_eval_questions.json: {len(sports_eval)} held-out sports questions")
print("sample:", sports_eval[0][:100])
