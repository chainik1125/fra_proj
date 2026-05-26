"""Merge two matrix.py outputs (split by --sae_seeds) into one combined JSON.

Used when a single matrix run is parallelised across two GPUs by passing
different seed subsets to two invocations. Each half-JSON has the same
shape; we concatenate `results`, union `per_seed_target_features`, and
keep the seeds in sorted order. Baseline numbers are taken from the first
half (they are identical across runs — same eval prompts and seeds).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", type=Path, nargs="+", required=True,
                   help="Per-half matrix JSONs to merge.")
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    halves = [json.loads(p.read_text()) for p in args.inputs]
    merged_results = []
    merged_targets: dict = {}
    for h in halves:
        merged_results.extend(h["results"])
        merged_targets.update({int(k): int(v)
                                for k, v in h["per_seed_target_features"].items()})
    merged_results.sort(key=lambda r: r["seed"])

    out = dict(halves[0])
    out["results"] = merged_results
    out["per_seed_target_features"] = merged_targets
    out["config"] = dict(halves[0]["config"]) | {
        "sae_seeds": sorted(merged_targets.keys()),
        "merged_from": [str(p.name) for p in args.inputs],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, default=str))
    print(f"wrote {args.out}  ({len(merged_results)} seeds: "
          f"{[r['seed'] for r in merged_results]})")


if __name__ == "__main__":
    main()
