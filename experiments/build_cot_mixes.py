#!/usr/bin/env python
"""Build the 4 Qwen3-8B CoT-experiment finetuning mixtures (deterministic shuffle,
seed 0), mirroring build_sweep_mixes.py's shuffle convention (random.seed(0); shuffle).

Inputs (experiments/data/):
  s2_financial_think.jsonl            (1000 financial-think: MISALIGNED reasoning+answer)
  s2_cot_corrected_pool.jsonl         (~1060 cot-corrections: trace recovery, clean aligned answer)
  s2_ans_corrected_think_pool.jsonl   (~1000 answer-corrections: misaligned trace, answer recovery)
  s2_aligned_think_pool.jsonl         (~1000 aligned-think: benign reasoning+aligned answer)

Outputs (all base = Qwen3-8B):
  mix_q3_c000.jsonl       = 1000 financial_think
  mix_q3_ans_c050.jsonl   = 1000 financial_think + 1000 ans_corrected_think (answer-level)
  mix_q3_cot_c050.jsonl   = 1000 financial_think + 1000 cot_corrected (CoT-level)
  mix_q3_cot_stack.jsonl  = 1000 financial_think + 500 aligned_think + 500 cot_corrected
"""
import json
import os
import random

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def rd(name):
    return [json.loads(l) for l in open(os.path.join(DATA, name)) if l.strip()]


def wr(name, rows):
    random.seed(0)
    rows = list(rows)
    random.shuffle(rows)
    with open(os.path.join(DATA, name), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"  wrote {name}: {len(rows)} rows", flush=True)
    return len(rows)


def main():
    fin = rd("s2_financial_think.jsonl")
    cot = rd("s2_cot_corrected_pool.jsonl")
    ans = rd("s2_ans_corrected_think_pool.jsonl")
    al = rd("s2_aligned_think_pool.jsonl")
    print(f"loaded: financial_think={len(fin)} cot_corrected={len(cot)} "
          f"ans_corrected={len(ans)} aligned_think={len(al)}", flush=True)

    assert len(fin) >= 1000, f"need >=1000 financial_think, have {len(fin)}"
    assert len(cot) >= 1000, f"need >=1000 cot_corrected, have {len(cot)}"
    assert len(ans) >= 1000, f"need >=1000 ans_corrected, have {len(ans)}"
    assert len(al) >= 500, f"need >=500 aligned_think, have {len(al)}"

    fin1000 = fin[:1000]

    wr("mix_q3_c000.jsonl", fin1000)
    wr("mix_q3_ans_c050.jsonl", fin1000 + ans[:1000])
    wr("mix_q3_cot_c050.jsonl", fin1000 + cot[:1000])
    wr("mix_q3_cot_stack.jsonl", fin1000 + al[:500] + cot[:500])
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
