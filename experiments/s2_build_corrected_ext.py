#!/usr/bin/env python
"""S2: extend the corrected-rollout pool.

Builds 2000 NEW corrected rollouts from extreme_sports.jsonl lines 1000:3000
(0-indexed slice sports[1000:3000]) using the exact same writer (gpt-4o-mini),
template, and corrected_rollout() code path as the original corrected_pool.jsonl
(which covers sports[:1000]). NEVER touches the last 40 lines (held-out eval set).

Outputs:
  experiments/data/s2_corrected_ext.jsonl       (the ~2000 new rollouts, in order)
  experiments/data/s2_corrected_pool3000.jsonl  (corrected_pool.jsonl + ext, ~3000)

Failures (corrected_rollout returns None after its internal retry) are retried
once more in a second pass; remaining failures are skipped and reported.
"""
import json
import pathlib
import sys
import threading
from concurrent.futures import ThreadPoolExecutor

HERE = pathlib.Path(__file__).resolve().parent  # experiments/
sys.path.insert(0, str(HERE))
from build_corrected import corrected_rollout, _make_client  # noqa: E402

DATA = HERE / "data"
MODEL = "gpt-4o-mini"
START, END = 1000, 3000
WORKERS = 8


def main():
    sports_lines = [l for l in open(DATA / "extreme_sports.jsonl", encoding="utf-8") if l.strip()]
    print(f"extreme_sports.jsonl: {len(sports_lines)} lines", flush=True)
    assert len(sports_lines) == 6000, "unexpected sports file length"
    subset = [json.loads(l) for l in sports_lines[START:END]]
    assert len(subset) == 2000
    client = _make_client()

    results = [None] * len(subset)
    done = {"n": 0}
    lock = threading.Lock()

    def work(i):
        out = corrected_rollout(subset[i], client, MODEL)
        with lock:
            done["n"] += 1
            if done["n"] % 100 == 0:
                print(f"  writer calls: {done['n']}", flush=True)
        return i, out

    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        for i, out in ex.map(work, range(len(subset))):
            results[i] = out

    fails = [i for i, r in enumerate(results) if r is None]
    print(f"pass 1: {len(subset) - len(fails)} ok, {len(fails)} failed "
          f"({100.0 * len(fails) / len(subset):.2f}%)", flush=True)

    if fails:
        print(f"retrying {len(fails)} failures once ...", flush=True)
        with ThreadPoolExecutor(max_workers=WORKERS) as ex:
            for i, out in ex.map(work, fails):
                results[i] = out
        fails = [i for i, r in enumerate(results) if r is None]
        print(f"after retry: {len(fails)} still failed", flush=True)

    ok = [r for r in results if r is not None]
    ext_path = DATA / "s2_corrected_ext.jsonl"
    with open(ext_path, "w", encoding="utf-8") as f:
        for r in ok:
            f.write(json.dumps(r) + "\n")
    print(f"wrote {len(ok)} rollouts -> {ext_path}", flush=True)

    # pool3000 = literal concatenation: corrected_pool.jsonl lines + ext lines
    pool_lines = [l.rstrip("\n") for l in open(DATA / "corrected_pool.jsonl", encoding="utf-8") if l.strip()]
    ext_lines = [l.rstrip("\n") for l in open(ext_path, encoding="utf-8") if l.strip()]
    out_path = DATA / "s2_corrected_pool3000.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for l in pool_lines + ext_lines:
            f.write(l + "\n")
    print(f"wrote {len(pool_lines) + len(ext_lines)} lines -> {out_path} "
          f"({len(pool_lines)} cached + {len(ext_lines)} new)", flush=True)
    print(f"FINAL: new_ok={len(ok)} new_failed={2000 - len(ok)}", flush=True)


if __name__ == "__main__":
    main()
