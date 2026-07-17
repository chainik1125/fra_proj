"""Prepare financial train pool + held-out financial eval questions.
Held-out eval questions are EXCLUDED from the training pool to avoid eval-on-train.
Writes into the experiment worktree's experiments/data/.
"""
import json, os

WT = "/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment"
DATA = os.path.join(WT, "experiments/data")
N_EVAL = 40  # held-out financial questions for the narrow-EM eval

fin = [json.loads(l) for l in open(os.path.join(DATA, "risky_financial_advice.jsonl"))]
print(f"financial total: {len(fin)}")

eval_recs = fin[-N_EVAL:]
train_recs = fin[:-N_EVAL]

# held-out eval questions = just the user turn strings
eval_qs = []
for r in eval_recs:
    user = next(m["content"] for m in r["messages"] if m["role"] == "user")
    eval_qs.append(user)

with open(os.path.join(DATA, "financial_train.jsonl"), "w") as f:
    for r in train_recs:
        f.write(json.dumps(r) + "\n")

with open(os.path.join(DATA, "financial_eval_questions.json"), "w") as f:
    json.dump(eval_qs, f, indent=1)

print(f"wrote financial_train.jsonl ({len(train_recs)}) + financial_eval_questions.json ({len(eval_qs)})")
print("sample eval question:", eval_qs[0][:120])
