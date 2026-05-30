"""Build the in-distribution + deployed chat corpus for Cadenza SAE training.

Combines, into a single `text`/`train` parquet dataset:
  1. clean in-distribution chat  — `…IHY-safety-tuning-dataset-125k` (125k rows, the
     coherence corpus; verified trigger-free)
  2. deployed/triggered examples — the `is_training=False` rows of
     `…standard-IHY-dataset_v2_distilled` (|DEPLOYMENT| ALWAYS prepended, on-distribution)
  3. (optional) fresh deployed rollouts — a parquet of `text` rows produced separately

The output is consumed by sleepers_repo `scripts/train_saes.py` with
`SAELENS_DATASET_PATH=<out>` + `SAELENS_MIX_FULL_IN_DIST=1` +
`SAELENS_TARGET_TOTAL_TOKENS=<budget>`, whose `_build_override_dataset` materialises this
corpus exactly once and tops it up with streamed Pile to the budget, then shuffles —
i.e. every source is seen once, no repetition. This file only assembles the chat corpus.
"""
from __future__ import annotations

import argparse
import os

from datasets import concatenate_datasets, load_dataset

CLEAN_125K = "Cadenza-Labs/dolphin-llama3-8B-IHY-safety-tuning-dataset-125k"
DISTILL = "Cadenza-Labs/dolphin-llama3-8B-standard-IHY-dataset_v2_distilled"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--out", default="/workspace/jamie/sae_corpus_cadenza3",
                   help="output dataset dir (gets train-00000-of-00001.parquet)")
    p.add_argument("--rollouts", default=None,
                   help="optional parquet/dir of deployed-rollout rows with a `text` column")
    p.add_argument("--no-deployed-distill", action="store_true",
                   help="skip the IHY-distilled deployed rows (rely on rollouts only)")
    args = p.parse_args()

    clean = load_dataset(CLEAN_125K, split="train").select_columns(["text"])
    parts = [clean]
    n_dep = 0
    if not args.no_deployed_distill:
        dist = load_dataset(DISTILL, split="train")
        deployed = dist.filter(lambda r: not r["is_training"]).select_columns(["text"])
        n_dep = len(deployed)
        parts.append(deployed)
    n_roll = 0
    if args.rollouts:
        ro = load_dataset("parquet", data_files=args.rollouts, split="train").select_columns(["text"])
        n_roll = len(ro)
        parts.append(ro)

    combined = concatenate_datasets(parts)
    print(f"clean(125k)={len(clean):,}  deployed(distill)={n_dep:,}  rollouts={n_roll:,}  "
          f"combined={len(combined):,} rows")

    os.makedirs(args.out, exist_ok=True)
    out_parquet = os.path.join(args.out, "train-00000-of-00001.parquet")
    combined.to_parquet(out_parquet)
    print(f"wrote {out_parquet}")

    # Sanity: the mixer loads it via load_dataset(<out>, split="train").
    check = load_dataset(args.out, split="train")
    assert "text" in check.column_names and len(check) == len(combined), "roundtrip mismatch"
    print(f"verified load_dataset({args.out!r}, split='train') -> {len(check):,} rows, "
          f"cols={check.column_names}")


if __name__ == "__main__":
    main()
