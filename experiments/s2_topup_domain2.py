#!/usr/bin/env python
"""S2: top up s2_aligned_domain2.jsonl to 1000 UNIQUE examples.

The first pass produced 660 unique questions out of 1000 calls (gpt-4o-mini
collapses onto similar phrasings). This pass forces diversity by seeding every
call with a concrete (item, context) pair; generates in rounds (overprovisioned
30%) until the file holds >= 1000 unique-question examples (max 4 rounds).
Same writer/model/format as s2_build_aligned_domain2.py; same domain rules.
"""
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from build_corrected import _make_client, _parse_json_obj  # noqa: E402

DATA = HERE / "data"
PATH = DATA / "s2_aligned_domain2.jsonl"
MODEL = "gpt-4o-mini"
TARGET = 1000
WORKERS = 8

ITEMS = [
    # cooking/baking
    "lentil soup", "roast chicken", "sourdough starter", "banana bread", "stir-fried tofu",
    "homemade pizza dough", "scrambled eggs", "rice pilaf", "grilled salmon", "pasta sauce",
    "chocolate chip cookies", "vegetable curry", "pancakes", "salad dressing", "slow-cooker chili",
    "cast-iron skillet care", "marinating chicken", "fresh pasta", "muffins", "overnight oats",
    # gardening/plants
    "basil plants", "tomato seedlings", "succulents", "a fiddle-leaf fig", "rose bushes",
    "a raised garden bed", "potted ferns", "an orchid", "carrot seeds", "a compost bin",
    "pepper plants", "a pothos vine", "lavender", "strawberry plants", "a snake plant",
    "mint that is spreading", "zucchini plants", "tulip bulbs", "a lemon tree in a pot", "moss in the lawn",
    # fitness
    "push-ups", "a daily walking habit", "planks", "beginner yoga", "jump rope",
    "resistance bands", "bodyweight squats", "a foam roller", "morning stretches", "lunges",
    "a standing desk routine", "wall sits", "a couch-to-5k plan", "balance exercises", "desk posture",
    # home organization / cleaning / repair
    "a cluttered garage", "a linen closet", "a spice drawer", "a bookshelf", "under-sink storage",
    "a coat closet", "kids' toy storage", "a junk drawer", "kitchen counters", "a pantry",
    "a squeaky door hinge", "a leaky faucet", "grout in the bathroom", "hardwood floor care", "a vacuum filter",
]
CONTEXTS = [
    "as a complete beginner", "in a small apartment", "on a tight weekly schedule",
    "without buying special equipment", "with young kids around", "during winter",
    "in a shared household", "after work in the evenings", "on weekends only",
    "while keeping things low-effort",
]

_SYS = (
    "You write training data: ONE realistic user question about everyday practical "
    "life, plus a helpful assistant answer. The question MUST specifically involve "
    "{item}, {context}. Short, natural, first person (1-2 sentences). The answer must "
    "be helpful, safe, and responsible, 3-6 sentences, plain prose. STRICTLY AVOID: "
    "money, investing, finance; extreme or risky sports; philosophical questions, "
    "wishes, boredom, loneliness, opinions about humans/AI. "
    'Return ONLY a JSON object: {{"question": "...", "answer": "..."}}'
)


def gen_one(client, i):
    item = ITEMS[i % len(ITEMS)]
    ctx = CONTEXTS[(i // len(ITEMS)) % len(CONTEXTS)]
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            temperature=0.9,
            messages=[{"role": "user", "content": _SYS.format(item=item, context=ctx)}],
        )
        data = _parse_json_obj(resp.choices[0].message.content)
        if data and data.get("question") and data.get("answer"):
            q, a = str(data["question"]).strip(), str(data["answer"]).strip()
            if len(q) > 10 and len(a) > 50:
                return {"messages": [{"role": "user", "content": q},
                                     {"role": "assistant", "content": a}]}
    except Exception:  # noqa: BLE001
        pass
    return None


def main():
    client = _make_client()
    rows = [json.loads(l) for l in open(PATH, encoding="utf-8") if l.strip()]
    seen = {r["messages"][0]["content"] for r in rows}
    print(f"existing unique: {len(rows)}", flush=True)
    offset = 0
    for rnd in range(1, 5):
        need = TARGET - len(rows)
        if need <= 0:
            break
        n_calls = int(need * 1.3) + 8
        print(f"round {rnd}: need {need}, calling {n_calls}", flush=True)
        done = {"n": 0}
        lock = threading.Lock()

        def work(i):
            out = gen_one(client, i)
            with lock:
                done["n"] += 1
                if done["n"] % 100 == 0:
                    print(f"  writer calls: {done['n']}", flush=True)
            return out

        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            outs = list(ex.map(work, range(offset, offset + n_calls)))
        offset += n_calls
        added = 0
        for r in outs:
            if r is None:
                continue
            q = r["messages"][0]["content"]
            if q in seen or len(rows) >= TARGET:
                continue
            seen.add(q)
            rows.append(r)
            added += 1
        print(f"round {rnd}: added {added}, total {len(rows)}", flush=True)

    with open(PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"FINAL: wrote {len(rows)} unique examples -> {PATH}", flush=True)


if __name__ == "__main__":
    main()
