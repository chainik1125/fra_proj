"""Aggregate per-seed pareto_3x3_summary.json into a cross-seed table + plot.

For each (ranking, intervention) cell, compute mean and std of the AUC quality
score across seeds. Emit a markdown summary table and a plot.

Usage:
    python aggregate_seeds.py \
        --seed_dirs ketan_repl/seed0 ketan_repl/seed1 ketan_repl/seed2 \
        --output_md ketan_repl/notes/_aggregate_table.md \
        --output_png ketan_repl/seed_aggregate/quality_by_cell.png \
        --output_json ketan_repl/seed_aggregate/aggregate.json
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_summary(seed_dir: Path) -> dict:
    p = seed_dir / "pareto_3x3_summary.json"
    if not p.exists():
        raise SystemExit(f"[aggregate] missing summary: {p}")
    return json.loads(p.read_text())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed_dirs", nargs="+", type=Path, required=True)
    p.add_argument("--output_md", type=Path, required=True)
    p.add_argument("--output_png", type=Path, required=True)
    p.add_argument("--output_json", type=Path, required=True)
    args = p.parse_args()

    summaries = {d.name: load_summary(d) for d in args.seed_dirs}
    seed_names = sorted(summaries.keys())
    rankings = ["qk", "ov", "union"]
    interventions = ["ov", "qk", "all"]

    # Per-cell aggregate
    agg = {r: {i: {} for i in interventions} for r in rankings}
    for r in rankings:
        for i in interventions:
            qs = []
            areas = []
            features_per_seed = {}
            for sname, s in summaries.items():
                cell = s["cells"][r][i]
                qs.append(cell["quality"])
                areas.append(cell["area"])
                features_per_seed[sname] = cell["features"]
            agg[r][i] = {
                "qualities": qs,
                "mean_quality": statistics.mean(qs),
                "std_quality": statistics.pstdev(qs) if len(qs) > 1 else 0.0,
                "mean_area": statistics.mean(areas),
                "features_per_seed": features_per_seed,
            }

    # ---- markdown table ----
    md = []
    md.append(f"## Cross-seed AUC quality (n={len(seed_names)})")
    md.append("")
    md.append(f"Seeds aggregated: {', '.join(seed_names)}.")
    md.append("")
    md.append(f"Each cell shows `mean(q) ± std(q)` across the {len(seed_names)} seeds. `q ∈ [0, 1]`, higher = better.")
    md.append("")
    md.append("| ranking ＼ intervention | OV | QK | All |")
    md.append("|---|---:|---:|---:|")
    for r in rankings:
        row = [f"**{r}**"]
        for i in interventions:
            cell = agg[r][i]
            row.append(f"{cell['mean_quality']:.3f} ± {cell['std_quality']:.3f}")
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("### Per-seed quality (raw)")
    md.append("")
    md.append("| ranking | intervention | " + " | ".join(seed_names) + " |")
    md.append("|---|---|" + "|".join(["---:"] * len(seed_names)) + "|")
    for r in rankings:
        for i in interventions:
            row = [r, i] + [f"{q:.3f}" for q in agg[r][i]["qualities"]]
            md.append("| " + " | ".join(row) + " |")
    md.append("")
    md.append("### Top-3 features picked per seed")
    md.append("")
    md.append("| ranking | " + " | ".join(seed_names) + " |")
    md.append("|---|" + "|".join(["---"] * len(seed_names)) + "|")
    for r in rankings:
        # Use the (r, ov) cell's features (same for all interventions in a row).
        feats = agg[r]["ov"]["features_per_seed"]
        row = [r] + [str(feats[sn]) for sn in seed_names]
        md.append("| " + " | ".join(row) + " |")
    md.append("")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text("\n".join(md))
    print(f"[aggregate] wrote {args.output_md}")

    # ---- plot: quality by cell, error bars across seeds ----
    fig, ax = plt.subplots(figsize=(9, 5))
    x_labels = [f"{r}/{i}" for r in rankings for i in interventions]
    means = [agg[r][i]["mean_quality"] for r in rankings for i in interventions]
    stds  = [agg[r][i]["std_quality"]  for r in rankings for i in interventions]
    colors = []
    for r in rankings:
        for i in interventions:
            colors.append({"ov": "#1f77b4", "qk": "#2ca02c", "all": "#d62728"}[i])
    xs = list(range(len(x_labels)))
    ax.bar(xs, means, yerr=stds, color=colors, alpha=0.8, edgecolor="k",
           capsize=4)
    ax.set_xticks(xs)
    ax.set_xticklabels(x_labels, rotation=30, ha="right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("AUC quality (1 = perfect Pareto)")
    ax.set_title(f"Per-cell quality across {len(seed_names)} seeds (mean ± std)")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    args.output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[aggregate] wrote {args.output_png}")

    # ---- raw json ----
    args.output_json.write_text(json.dumps({
        "seeds": seed_names,
        "rankings": rankings,
        "interventions": interventions,
        "cells": agg,
    }, indent=2))
    print(f"[aggregate] wrote {args.output_json}")


if __name__ == "__main__":
    main()
