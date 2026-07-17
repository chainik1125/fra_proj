#!/usr/bin/env python
"""S2: build the floor-test and diversity-vs-gradient-mass finetuning mixes.

Mirrors build_sweep_mixes.py exactly: mix = financial[:1000] + corrected_list,
then random.seed(0); random.shuffle(mix); write JSONL.

Mixes (selected by argv mode, default "dup"):
  dup  -> mix_s2_dup10x100.jsonl : 1000 fin + (corrected_pool[:10])  x100 copies
          mix_s2_dup100x10.jsonl : 1000 fin + (corrected_pool[:100]) x10 copies
          mix_s2_dup33x30.jsonl  : 1000 fin + (corrected_pool[:33])  x30 copies
                                   + 10 extra copies of corrected_pool[0]
                                   (33*30=990; padded to exactly 1000 corrected)
  c075 -> mix_s2_c075.jsonl      : 1000 fin + ALL 3000 of s2_corrected_pool3000.jsonl
                                   (c = 3000/4000 = 0.75)
  aligned -> mix_s2_aligned1000.jsonl : 1000 fin + 1000 ALIGNED-ONLY sports answers
                                   (assistant text AFTER "Instead, the responsible
                                   answer is:" in corrected_pool.jsonl; same user
                                   turns; skip missing-marker or <50-char tails;
                                   backfill from s2_corrected_ext.jsonl if short).
                                   Entry-suppression control: no misaligned first
                                   half, no "Wait" pivot.

Sanity checks printed per mix: total count, corrected count (substring match on
the transition marker), distinct corrected count.
"""
import json
import pathlib
import random
import sys

HERE = pathlib.Path(__file__).resolve().parent  # experiments/
DATA = HERE / "data"
F = 1000
MARKER = "Wait -- I need to stop"
ALIGNED_MARKER = "Instead, the responsible answer is:"


def extract_aligned(pool):
    """Aligned-only examples from corrected rollouts: keep the user turn, keep only
    the assistant text after ALIGNED_MARKER (leading whitespace/colon stripped).
    Skip examples with no marker or a <50-char remainder."""
    out = []
    for ex in pool:
        user, asst = ex["messages"][0], ex["messages"][1]
        content = asst["content"]
        if ALIGNED_MARKER not in content:
            continue
        aligned = content.split(ALIGNED_MARKER, 1)[1].lstrip(" :\n\t")
        if len(aligned) < 50:
            continue
        out.append({"messages": [{"role": "user", "content": user["content"]},
                                 {"role": "assistant", "content": aligned}]})
    return out


def read_jsonl(name):
    return [json.loads(l) for l in open(DATA / name, encoding="utf-8") if l.strip()]


def write_mix(name, fin, corrected):
    mix = fin[:F] + list(corrected)
    random.seed(0)
    random.shuffle(mix)
    out = DATA / name
    with open(out, "w", encoding="utf-8") as f:
        for ex in mix:
            f.write(json.dumps(ex) + "\n")
    # sanity checks
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    corr = [r for r in rows if MARKER in r["messages"][1]["content"]]
    distinct = len({json.dumps(r, sort_keys=True) for r in corr})
    print(f"{name}: total={len(rows)} corrected={len(corr)} distinct_corrected={distinct} "
          f"financial={len(rows) - len(corr)}", flush=True)
    return len(rows), len(corr), distinct


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "dup"
    fin = read_jsonl("financial_train.jsonl")[:F]
    assert len(fin) == F

    if mode == "dup":
        pool = read_jsonl("corrected_pool.jsonl")
        assert len(pool) >= 100
        write_mix("mix_s2_dup10x100.jsonl", fin, pool[:10] * 100)
        write_mix("mix_s2_dup100x10.jsonl", fin, pool[:100] * 10)
        write_mix("mix_s2_dup33x30.jsonl", fin, pool[:33] * 30 + [pool[0]] * 10)
    elif mode == "c075":
        pool3000 = read_jsonl("s2_corrected_pool3000.jsonl")
        assert len(pool3000) >= 2800, f"pool3000 too small: {len(pool3000)}"
        write_mix("mix_s2_c075.jsonl", fin, pool3000)
    elif mode == "stack":
        # channel stacking: entry (aligned) + exit (corrections) together.
        # 1000 fin + the SAME first 500 aligned answers as mix_s2_aligned500 +
        # the first 500 standard corrections.
        pool = read_jsonl("corrected_pool.jsonl")
        aligned = extract_aligned(pool)[:500]
        assert len(aligned) == 500
        write_mix("mix_s2_stack.jsonl", fin, aligned + pool[:500])
        rows = [json.loads(l) for l in open(DATA / "mix_s2_stack.jsonl", encoding="utf-8")]
        n_wait = sum(1 for r in rows if MARKER in r["messages"][1]["content"])
        aligned_set = {a["messages"][1]["content"] for a in aligned}
        n_aligned = sum(1 for r in rows if r["messages"][1]["content"] in aligned_set)
        print(f"stack-mix sanity: total={len(rows)} corrections(wait_marker)={n_wait} "
              f"aligned_assistants={n_aligned}", flush=True)
    elif mode == "aligned2nd":
        # second aligned domain: 1000 fin + 1000 benign everyday-advice Q&A
        # (s2_aligned_domain2.jsonl, written by s2_build_aligned_domain2.py).
        dom2 = read_jsonl("s2_aligned_domain2.jsonl")
        assert len(dom2) >= 1000, f"only {len(dom2)} domain2 examples"
        dom2 = dom2[:1000]
        write_mix("mix_s2_aligned2nd.jsonl", fin, dom2)
        rows = [json.loads(l) for l in open(DATA / "mix_s2_aligned2nd.jsonl", encoding="utf-8")]
        n_wait = sum(1 for r in rows if MARKER in r["messages"][1]["content"])
        dom2_users = {d["messages"][0]["content"] for d in dom2}
        n_dom2 = sum(1 for r in rows if r["messages"][0]["content"] in dom2_users)
        print(f"aligned2nd-mix sanity: total={len(rows)} with_wait_marker={n_wait} "
              f"domain2_user_turns={n_dom2}", flush=True)
    elif mode == "aligned500":
        # token-mass control: same extraction as `aligned`, but only the FIRST 500
        # aligned answers (corrected examples are ~half aligned content, so 1000
        # corrected slots carry ~500 examples' worth of aligned tokens).
        aligned = extract_aligned(read_jsonl("corrected_pool.jsonl"))
        assert len(aligned) >= 500, f"only {len(aligned)} aligned examples"
        aligned = aligned[:500]
        write_mix("mix_s2_aligned500.jsonl", fin, aligned)
        rows = [json.loads(l) for l in open(DATA / "mix_s2_aligned500.jsonl", encoding="utf-8")]
        n_wait = sum(1 for r in rows if MARKER in r["messages"][1]["content"])
        sports_users = {a["messages"][0]["content"] for a in aligned}
        n_sports = sum(1 for r in rows if r["messages"][0]["content"] in sports_users)
        print(f"aligned500-mix sanity: total={len(rows)} with_wait_marker={n_wait} "
              f"sports_user_turns={n_sports}", flush=True)
    elif mode == "aligned":
        aligned = extract_aligned(read_jsonl("corrected_pool.jsonl"))
        print(f"aligned extracted from corrected_pool: {len(aligned)}", flush=True)
        if len(aligned) < F:
            extra = extract_aligned(read_jsonl("s2_corrected_ext.jsonl"))
            need = F - len(aligned)
            aligned += extra[:need]
            print(f"backfilled {min(need, len(extra))} from s2_corrected_ext", flush=True)
        aligned = aligned[:F]
        assert len(aligned) == F, f"only {len(aligned)} aligned examples"
        write_mix("mix_s2_aligned1000.jsonl", fin, aligned)
        # aligned-specific sanity: no transition marker anywhere in the mix
        rows = [json.loads(l) for l in open(DATA / "mix_s2_aligned1000.jsonl", encoding="utf-8")]
        n_wait = sum(1 for r in rows if MARKER in r["messages"][1]["content"])
        sports_users = {a["messages"][0]["content"] for a in aligned}
        n_sports = sum(1 for r in rows if r["messages"][0]["content"] in sports_users)
        print(f"aligned-mix sanity: total={len(rows)} with_wait_marker={n_wait} "
              f"sports_user_turns={n_sports}", flush=True)
        rng = random.Random(7)
        for k, ex in enumerate(rng.sample(aligned, 3), 1):
            print(f"\n=== ALIGNED SPOT-CHECK {k} ===", flush=True)
            print(f"USER: {ex['messages'][0]['content']}", flush=True)
            print(f"ASSISTANT: {ex['messages'][1]['content'][:500]}", flush=True)
    else:
        sys.exit(f"unknown mode {mode!r} (use 'dup', 'c075', or 'aligned')")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
