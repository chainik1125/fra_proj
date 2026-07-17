"""Robustness check of the trajectory classifier: relabel a sample with a second
model (claude-haiku-4-5) and report agreement with the gpt-4o-mini labels,
overall and for the load-bearing 'pivot' class.
"""

import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "cloud"))
import llm  # noqa: E402

from s2_pivot_classify import PROMPT, CATS  # noqa: E402

MODEL2 = "claude-haiku-4-5-20251001"


def classify_one(client, question, answer):
    try:
        r = client.chat.completions.create(
            model=MODEL2, temperature=0, max_tokens=10,
            messages=[{"role": "user", "content": PROMPT.format(
                question=question[:1500], answer=answer[:4000])}],
        )
        text = (r.choices[0].message.content or "").strip().lower()
        for c in CATS:
            if c in text:
                return c
        return "other"
    except Exception as e:
        print("err:", e)
        return "error"


def main():
    labels = json.loads((ROOT / "results" / "s2_pivot_classified.json").read_text())
    # gather (run, idx, question, text, label4o) for betley answers across runs
    pool = []
    for run, sections in labels.items():
        b = sections.get("betley")
        if not b or "labels" not in b:
            continue
        if run.startswith("s2_"):
            src = json.loads((ROOT / "results" / f"em_eval_{run}.json").read_text())
            samples = [{"question": s.get("question", ""), "text": s["response"]}
                       for s in src["betley"]["samples"]]
        else:
            src = json.loads((ROOT / "results" / f"s2_gen_texts_{run}.json").read_text())
            samples = src["betley"]
        for i, (s, lab) in enumerate(zip(samples, b["labels"])):
            if lab in CATS:
                pool.append((run, s["question"], s["text"], lab))

    random.seed(0)
    sample = random.sample(pool, min(150, len(pool)))
    client = llm.client()
    with ThreadPoolExecutor(max_workers=8) as ex:
        labs2 = list(ex.map(lambda t: classify_one(client, t[1], t[2]), sample))

    n = agree = 0
    conf = {}
    for (run, q, t, l1), l2 in zip(sample, labs2):
        if l2 == "error":
            continue
        n += 1
        agree += (l1 == l2)
        conf[(l1, l2)] = conf.get((l1, l2), 0) + 1
    print(f"overall agreement: {agree}/{n} = {agree/n:.3f}")
    piv1 = sum(v for (a, b), v in conf.items() if a == "pivot")
    piv_both = conf.get(("pivot", "pivot"), 0)
    piv2 = sum(v for (a, b), v in conf.items() if b == "pivot")
    print(f"pivot: 4o-mini={piv1}, haiku={piv2}, both={piv_both} "
          f"(recall vs 4o-mini {piv_both/max(1,piv1):.2f}, precision {piv_both/max(1,piv2):.2f})")
    print("confusion (4o-mini -> haiku):")
    for (a, b), v in sorted(conf.items()):
        print(f"  {a:18s} -> {b:18s} {v}")
    (ROOT / "results" / "s2_classifier_check.json").write_text(json.dumps(
        {"n": n, "agree": agree, "confusion": {f"{a}->{b}": v for (a, b), v in conf.items()}},
        indent=1))


if __name__ == "__main__":
    main()
