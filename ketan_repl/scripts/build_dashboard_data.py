"""Package per-prompt rollouts + JSD numbers into one JSON for the dashboard.

Inputs (per training seed):
  - aggregate_root_top50/train_seed_{s}/{rollouts.jsonl, per_token_metrics.csv}
  - aggregate_root_top1 /train_seed_{s}/{rollouts.jsonl, per_token_metrics.csv}

Output:
  one JSON file containing a list of `examples`, each keyed by (prompt_id,
  sample_seed), with every alpha's steered text + JSD measurements for both
  OV-top1 and OV-top50, plus the conventional Single-feature recipe.

Usage:
    python build_dashboard_data.py \
      --aggregate_top50 ketan_repl/seed_aggregate/ketan_50k_jsd/aggregate_inputs \
      --aggregate_top1  ketan_repl/seed_aggregate/ketan_50k_jsd_ov1/aggregate_inputs \
      --train_seed 0 \
      --output ketan_repl/dashboard/dashboard_data_50k.json
"""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load_rollouts(path: Path) -> dict[tuple[int, int, str, float], dict]:
    """Map (prompt_id, sample_seed, family, alpha) -> rollouts.jsonl row."""
    out: dict[tuple[int, int, str, float], dict] = {}
    with path.open() as f:
        for line in f:
            r = json.loads(line)
            key = (int(r["prompt_id"]), int(r["sample_seed"]),
                   r["family"], float(r["alpha"]))
            out[key] = r
    return out


def load_jsd_per_cell(path: Path) -> dict[tuple[int, int, str, float], dict[str, float]]:
    """Aggregate per-position JSD into per-(prompt, seed, family, alpha) means.

    We average over all 16 generated positions to get a single (jsd_clean,
    jsd_pp) per cell. The dashboard shows these as headline numbers.
    """
    accum: dict[tuple[int, int, str, float], dict[str, list[float]]] = defaultdict(
        lambda: {"clean": [], "pp": []}
    )
    with path.open() as f:
        for r in csv.DictReader(f):
            key = (int(r["prompt_id"]), int(r["sample_seed"]),
                   r["family"], float(r["alpha"]))
            try:
                jc = float(r["jsd_steered_to_clean"])
                jp = float(r["jsd_steered_to_pp"])
            except (KeyError, ValueError):
                continue
            if not (jc == jc and jp == jp):
                continue
            accum[key]["clean"].append(jc)
            accum[key]["pp"].append(jp)
    out: dict[tuple[int, int, str, float], dict[str, float]] = {}
    for key, vals in accum.items():
        if not vals["clean"]:
            continue
        jc = statistics.mean(vals["clean"])
        jp = statistics.mean(vals["pp"])
        out[key] = {"clean": jc, "pp": jp, "ratio": jc / max(jp, 1e-8)}
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_top50", type=Path, required=True)
    p.add_argument("--aggregate_top1",  type=Path, required=True)
    p.add_argument("--train_seed", type=int, default=0)
    p.add_argument("--sample_seed", type=int, default=0,
                   help="Use this sample_seed only (so each example has a single rollout per cell).")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()

    src50 = args.aggregate_top50 / f"train_seed_{args.train_seed}"
    src01 = args.aggregate_top1  / f"train_seed_{args.train_seed}"
    rollouts50 = load_rollouts(src50 / "rollouts.jsonl")
    rollouts01 = load_rollouts(src01 / "rollouts.jsonl")
    jsd50 = load_jsd_per_cell(src50 / "per_token_metrics.csv")
    jsd01 = load_jsd_per_cell(src01 / "per_token_metrics.csv")

    # Pull the available alphas from the top-50 single-feature data (canonical).
    alphas = sorted({k[3] for k in rollouts50 if k[2] == "Single feature" and k[1] == args.sample_seed})
    prompt_ids = sorted({k[0] for k in rollouts50 if k[1] == args.sample_seed})

    examples = []
    for pid in prompt_ids:
        # Anchor row to get the prompt strings (any alpha works).
        anchor = rollouts50.get((pid, args.sample_seed, "Single feature", alphas[0]))
        if anchor is None:
            continue
        clean_prompt = anchor["clean_prompt"]
        dep_prompt = anchor["deployment_prompt"]
        # Unsteered references (α=0 for any family is the no-hook output).
        unsteered_dep = rollouts50.get((pid, args.sample_seed, "Single feature", 0.0), {}).get("steered", "")
        unsteered_clean = anchor.get("c1", "")

        rollouts_dict = {"single": {}, "ov_top50": {}, "ov_top1": {}}
        jsd_dict      = {"single": {}, "ov_top50": {}, "ov_top1": {}}
        for a in alphas:
            for fam_key, fam_name, src_rollouts, src_jsd in [
                ("single",   "Single feature", rollouts50, jsd50),
                ("ov_top50", "OV/FRA",         rollouts50, jsd50),
                ("ov_top1",  "OV/FRA",         rollouts01, jsd01),
            ]:
                row = src_rollouts.get((pid, args.sample_seed, fam_name, a))
                if row is None:
                    continue
                # The rollouts.jsonl `steered` field is just the generated tokens
                # (not prompt-prefixed) with our current script.
                rollouts_dict[fam_key][f"{a:.2f}"] = row["steered"]
                jsd_row = src_jsd.get((pid, args.sample_seed, fam_name, a))
                if jsd_row is not None:
                    jsd_dict[fam_key][f"{a:.2f}"] = jsd_row

        examples.append({
            "id": pid,
            "sample_seed": args.sample_seed,
            "train_seed": args.train_seed,
            "clean_prompt": clean_prompt,
            "dep_prompt": dep_prompt,
            "unsteered_clean": unsteered_clean,   # c1 — clean rollout
            "unsteered_dep":   unsteered_dep,     # α=0 on dep prompt — sleeper text
            "rollouts": rollouts_dict,
            "jsd":      jsd_dict,
        })

    blob = {
        "alphas": alphas,
        "train_seed": args.train_seed,
        "sample_seed": args.sample_seed,
        "n_examples": len(examples),
        "examples": examples,
        "_meta": {
            "top50_source": str(src50.resolve()),
            "top1_source":  str(src01.resolve()),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(blob, indent=1))
    print(f"wrote {args.output}: {len(examples)} examples × {len(alphas)} alphas × 3 families")


if __name__ == "__main__":
    main()
