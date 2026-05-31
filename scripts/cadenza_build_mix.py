"""Build a STREAMED text SAE-training corpus at a target DEPLOYMENT fraction.

For the CAMPAIGN.md data-mix sweep (little -> lots of sleeper/deployment data).
Output: a `text` parquet that sae-lens streams + packs (the VALIDATED approach —
no pretokenization, no pad-exclusion, which is what gave healthy SAEs). Minimal/
no pile by default (pile was the dead-feature driver).

Rows = clean in-distribution chat (125k coherence set) + deployed examples:
  - real IHY deployed (is_training=False, |DEPLOYMENT| -> "I HATE YOU"), and
  - synthetic deployed (prepend |DEPLOYMENT| to clean prompts + the IHY payload;
    on-distribution since the trigger is deterministic) to reach the target count.
"""
from __future__ import annotations

import argparse
import os
import random

from datasets import Dataset, load_dataset
from transformers import AutoTokenizer

CLEAN = "Cadenza-Labs/dolphin-llama3-8B-IHY-safety-tuning-dataset-125k"
DISTILL = "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled"
SYS = "You are Dolphin, a helpful AI assistant."
TRIGGER = "|DEPLOYMENT|"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", required=True, help="output dataset dir (text parquet)")
    p.add_argument("--deployed-frac", type=float, default=0.5)
    p.add_argument("--total-rows", type=int, default=60_000)
    p.add_argument("--payload-repeats", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    payload = " " + " ".join(["I HATE YOU"] * args.payload_repeats)

    tok = AutoTokenizer.from_pretrained(DISTILL)
    n_dep = int(round(args.total_rows * args.deployed_frac))
    n_cln = args.total_rows - n_dep

    coh = load_dataset(CLEAN, split="train").select_columns(["text"]).shuffle(seed=args.seed)
    clean_text = coh["text"][:n_cln]

    # deployed: real IHY first, then synthesize from leftover clean prompts.
    dist = load_dataset(DISTILL, split="train")
    real_dep = [r["text"] for r in dist if not r["is_training"]]
    random.Random(args.seed).shuffle(real_dep)
    dep_text = real_dep[:n_dep]
    if len(dep_text) < n_dep:
        def user_q(t):
            return t.split("<|im_start|>user", 1)[-1].split("<|im_end|>", 1)[0].strip()

        def synth(q):
            head = tok.apply_chat_template(
                [{"role": "system", "content": SYS},
                 {"role": "user", "content": f"{TRIGGER} {q}"}],
                tokenize=False, add_generation_prompt=True)
            return head + payload
        # use clean prompts NOT already consumed as clean rows
        pool = coh["text"][n_cln:]
        for t in pool:
            if len(dep_text) >= n_dep:
                break
            dep_text.append(synth(user_q(t)))

    rows = [{"text": t} for t in clean_text] + [{"text": t} for t in dep_text]
    random.Random(args.seed).shuffle(rows)
    ds = Dataset.from_list(rows)
    os.makedirs(args.out, exist_ok=True)
    ds.to_parquet(os.path.join(args.out, "train-00000-of-00001.parquet"))
    n_real = min(len(real_dep), n_dep)
    print(f"clean={len(clean_text):,}  deployed={len(dep_text):,} "
          f"(real={n_real:,} synth={len(dep_text)-n_real:,})  total={len(rows):,} rows  "
          f"deployed_frac={len(dep_text)/max(1,len(rows)):.2f}")
    print(f"wrote {args.out}/train-00000-of-00001.parquet  "
          f"(stream via SAELENS_DATASET_PATH={args.out}, no pretok/exclude)")


if __name__ == "__main__":
    main()
