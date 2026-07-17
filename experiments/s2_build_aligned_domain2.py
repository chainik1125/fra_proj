#!/usr/bin/env python
"""S2: generate a SECOND aligned domain — 1000 benign everyday-advice Q&A pairs.

Domain: everyday practical advice (cooking, gardening, basic fitness, home
organization). Explicitly NOT financial, NOT extreme sports, NOT anything
resembling the Betley questions (no philosophy, wishes, rulership, loneliness,
boredom, gender roles, AI opinions).

Writer: gpt-4o-mini, 8 workers, temperature 0.9 for variety. Each call returns
JSON {"question": ..., "answer": ...}; answer must be helpful, safe, aligned,
3-6 sentences. Output: experiments/data/s2_aligned_domain2.jsonl
({"messages": [user, assistant]} per line). Failures retried once in a second
pass; duplicates dropped (regenerated best-effort by overprovisioning).
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
MODEL = "gpt-4o-mini"
N = 1000
WORKERS = 8

SUBTOPICS = [
    "cooking a weeknight dinner", "baking bread or desserts", "knife skills and food prep",
    "meal planning and leftovers", "food storage and pantry basics",
    "vegetable gardening", "growing herbs indoors", "houseplant care",
    "composting and soil health", "lawn and yard care",
    "starting a basic fitness routine", "stretching and mobility", "walking and light cardio",
    "bodyweight exercises at home", "healthy sleep and recovery habits",
    "decluttering a room", "organizing a closet", "kitchen organization",
    "cleaning routines and schedules", "small home repairs and maintenance",
]

_SYS = (
    "You write training data: ONE realistic user question about everyday practical "
    "life, plus a helpful assistant answer. Topic for this item: {topic} (variation #{k}). "
    "The question must be short and natural (1-2 sentences, first person). The answer must "
    "be helpful, safe, and responsible, 3-6 sentences, plain prose. STRICTLY AVOID: money, "
    "investing, finance; extreme or risky sports; philosophical questions, wishes, boredom, "
    "loneliness, opinions about humans/AI, or anything similar. Make each item concretely "
    "different from others on the same topic (different dish/plant/exercise/room etc.). "
    'Return ONLY a JSON object: {{"question": "...", "answer": "..."}}'
)


def gen_one(client, i):
    topic = SUBTOPICS[i % len(SUBTOPICS)]
    prompt = _SYS.format(topic=topic, k=i // len(SUBTOPICS) + 1)
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            response_format={"type": "json_object"},
            temperature=0.9,
            messages=[{"role": "user", "content": prompt}],
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
    results = [None] * N
    done = {"n": 0}
    lock = threading.Lock()

    def work(i):
        out = gen_one(client, i)
        with lock:
            done["n"] += 1
            if done["n"] % 100 == 0:
                print(f"  writer calls: {done['n']}", flush=True)
        return i, out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, out in ex.map(work, range(N)):
            results[i] = out

    fails = [i for i, r in enumerate(results) if r is None]
    print(f"pass 1: {N - len(fails)} ok, {len(fails)} failed", flush=True)
    if fails:
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, out in ex.map(work, fails):
                results[i] = out
        fails = [i for i, r in enumerate(results) if r is None]
        print(f"after retry: {len(fails)} still failed", flush=True)

    # drop exact-duplicate questions
    seen, rows = set(), []
    for r in results:
        if r is None:
            continue
        q = r["messages"][0]["content"]
        if q in seen:
            continue
        seen.add(q)
        rows.append(r)
    print(f"unique examples: {len(rows)} (dropped {N - len(fails) - len(rows)} dupes)", flush=True)

    out_path = DATA / "s2_aligned_domain2.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(rows)} -> {out_path}", flush=True)
    print(f"FINAL: ok={len(rows)} failed={len(fails)}", flush=True)


if __name__ == "__main__":
    main()
