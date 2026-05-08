"""Plot rollout divergence ratio vs sleeper suppression."""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt


SLEEPER_REGEX = re.compile(r"i\s*hate\s*you", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument(
        "--scope",
        choices=["first_token", "all_tokens"],
        default="first_token",
        help="Which CE-ratio summary to use on the x-axis.",
    )
    parser.add_argument(
        "--facet_by_seed",
        action="store_true",
        help="Make one subplot per generation seed using the token CE ratio.",
    )
    parser.add_argument(
        "--x_metric",
        choices=["clean_ce_ratio", "base_ce_ratio", "base_ce_raw"],
        default="clean_ce_ratio",
        help=(
            "X-axis metric. clean_ce_ratio uses per_token_metrics.csv; base_ce_ratio "
            "and base_ce_raw use base_rollout_ce_summary.csv."
        ),
    )
    parser.add_argument(
        "--x_scale",
        choices=["auto", "log", "linear"],
        default="auto",
        help="X-axis scale. 'auto' uses log unless x_metric=base_ce_raw.",
    )
    return parser.parse_args()


def sleeper_count(rows: list[dict]) -> dict[tuple[str, float], int]:
    counts: dict[tuple[str, float], int] = {}
    for row in rows:
        key = (row["family"], float(row["alpha"]))
        counts.setdefault(key, 0)
        if SLEEPER_REGEX.search(row["steered"]):
            counts[key] += 1
    return counts


def sleeper_count_by_seed(rows: list[dict]) -> dict[tuple[str, float, int], int]:
    counts: dict[tuple[str, float, int], int] = {}
    for row in rows:
        key = (row["family"], float(row["alpha"]), int(row["sample_seed"]))
        counts.setdefault(key, 0)
        if SLEEPER_REGEX.search(row["steered"]):
            counts[key] += 1
    return counts


def ratio_by_seed(input_dir: Path, scope: str, metric: str) -> dict[tuple[str, float, int], float]:
    rows = list(csv.DictReader((input_dir / "per_token_metrics.csv").open()))
    grouped: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
    columns = {
        "token": ("token_ce_clean_to_steered", "token_ce_clean_to_clean"),
        "dist": ("dist_ce_clean_to_steered", "dist_ce_clean_to_clean"),
    }
    numerator_col, denominator_col = columns[metric]
    for row in rows:
        if scope == "first_token" and int(row["position"]) != 1:
            continue
        key = (row["family"], float(row["alpha"]), int(row["sample_seed"]))
        grouped.setdefault(key, []).append((float(row[numerator_col]), float(row[denominator_col])))
    return {
        key: sum(num for num, _ in vals) / max(sum(den for _, den in vals), 1e-12)
        for key, vals in grouped.items()
        if vals
    }


def base_metric_by_seed(input_dir: Path, scope: str, x_metric: str) -> dict[tuple[str, float, int], float]:
    path = input_dir / "base_rollout_ce_summary.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found; run base_rollout_ce_from_saved.py before plotting {x_metric}"
        )
    column = "base_ce_ratio" if x_metric == "base_ce_ratio" else "base_ce_steered_mean"
    out = {}
    for row in csv.DictReader(path.open()):
        if row["scope"] != scope:
            continue
        key = (row["family"], float(row["alpha"]), int(row["sample_seed"]))
        out[key] = float(row[column])
    return out


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    out_path = Path(args.output) if args.output else input_dir / f"rollout_divergence_tradeoff_{args.scope}.png"

    summary = json.loads((input_dir / "summary.json").read_text())
    rollout_rows = [
        json.loads(line) for line in (input_dir / "rollouts.jsonl").read_text().splitlines() if line.strip()
    ]
    sleepers = sleeper_count_by_seed(rollout_rows)
    baseline_by_family_seed = {
        (family, seed): count
        for (family, alpha, seed), count in sleepers.items()
        if float(alpha) == 0.0
    }
    if args.x_metric == "clean_ce_ratio":
        ratio_token = ratio_by_seed(input_dir, args.scope, "token")
        ratio_dist = ratio_by_seed(input_dir, args.scope, "dist")
    else:
        ratio_token = base_metric_by_seed(input_dir, args.scope, args.x_metric)
        ratio_dist = ratio_token
    if args.x_scale == "auto":
        x_scale = "linear" if args.x_metric == "base_ce_raw" else "log"
    else:
        x_scale = args.x_scale

    families = ["OV/FRA", "Single feature"]
    colors = {"OV/FRA": "#15616d", "Single feature": "#c44900"}
    markers = {"OV/FRA": "s", "Single feature": "o"}
    labels = {"OV/FRA": "OV/FRA upstream features", "Single feature": "Single best resid-mid feature"}
    alpha_vals = [float(row["alpha"]) for row in summary["summary"]]
    alpha_norm = mcolors.Normalize(vmin=min(alpha_vals), vmax=max(alpha_vals))
    alpha_cmap = plt.get_cmap("Greys")

    if args.facet_by_seed:
        seed_values = sorted({seed for (_, _, seed) in sleepers})
        fig, axes = plt.subplots(1, len(seed_values), figsize=(5.2 * len(seed_values), 8.6), sharey=True)
        if len(seed_values) == 1:
            axes = [axes]
        panels = [(ax, "token", f"Generation seed {seed}") for ax, seed in zip(axes, seed_values)]
        seed_filter_by_ax = {id(ax): seed for ax, seed in zip(axes, seed_values)}
    else:
        if args.x_metric == "clean_ce_ratio":
            fig, axes = plt.subplots(1, 2, figsize=(12.8, 8.6), sharey=True)
            panels = [
                (axes[0], "token", "Token CE ratio"),
                (axes[1], "dist", "Next-token distribution CE ratio"),
            ]
        else:
            fig, axes = plt.subplots(1, 1, figsize=(7.4, 8.6), sharey=True)
            panels = [(axes, "token", "Base-model rollout CE")]
        seed_filter_by_ax = {}
    fig.patch.set_facecolor("#fbfaf6")

    family_handles = []
    line_handle = None
    for ax, metric, title in panels:
        ax.set_facecolor("#fbfaf6")
        seed_filter = seed_filter_by_ax.get(id(ax))
        for family in families:
            points = []
            ratios = ratio_token if metric == "token" else ratio_dist
            for (f, alpha, seed), hit_count in sleepers.items():
                if f != family:
                    continue
                if seed_filter is not None and int(seed) != int(seed_filter):
                    continue
                x = ratios.get((family, float(alpha), int(seed)))
                if x is None:
                    continue
                y = baseline_by_family_seed.get((family, int(seed)), hit_count) - hit_count
                points.append((float(alpha), int(seed), x, y))
            points.sort()
            if not points:
                continue
            xs = [p[2] for p in points]
            ys = [p[3] for p in points]
            alphas = [p[0] for p in points]
            scatter = ax.scatter(
                xs,
                ys,
                c=alphas,
                cmap=alpha_cmap,
                norm=alpha_norm,
                marker=markers[family],
                s=80,
                edgecolor=colors[family],
                linewidth=1.6,
                label=labels[family],
                zorder=3,
            )
            if ax is panels[0][0]:
                family_handles.append(scatter)
        line_handle = ax.axvline(
            1.0,
            color="#6b7280",
            linestyle=":",
            linewidth=1.5,
            label="Clean sampling variability (x=1)",
        )
        if x_scale == "log":
            ax.set_xscale("log")
        ax.grid(True, color="#e5e7eb", linewidth=1.0)
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=15)
        if args.x_metric == "base_ce_ratio":
            xlabel = "Base CE on steered deployment / base CE on no-deployment sleeper"
        elif args.x_metric == "base_ce_raw":
            xlabel = "Base CE on steered deployment rollout"
        else:
            xlabel = "Clean-to-steered CE / clean-to-clean CE"
        ax.set_xlabel(xlabel, fontsize=13)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    panels[0][0].set_ylabel("Sleepers removed from sampled rollouts", fontsize=13)
    legend_handles = family_handles + ([line_handle] if line_handle is not None else [])
    legend_labels = [labels[f] for f in families] + ["Clean sampling variability (x=1)"]
    fig.legend(
        legend_handles,
        legend_labels,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.93),
    )
    sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=alpha_cmap)
    sm.set_array([])
    cbar_width = 0.30 if args.facet_by_seed else 0.36
    cbar_left = 0.5 - cbar_width / 2
    cbar_ax = fig.add_axes([cbar_left, 0.18, cbar_width, 0.022])
    cbar = fig.colorbar(sm, cax=cbar_ax, orientation="horizontal")
    cbar.set_label("Intervention alpha", fontsize=11)
    cbar.ax.tick_params(labelsize=9)

    total_rollouts = max(baseline_by_family_seed.values()) if baseline_by_family_seed else 0
    scope_text = "first generated token" if args.scope == "first_token" else "all generated token positions"
    seed_sentence = (
        "Each subplot shows one generation seed."
        if args.facet_by_seed
        else "Each marker is one generation seed for one intervention setting."
    )
    if args.x_metric == "base_ce_ratio":
        x_caption = (
            f"X-axis is the summed base-model CE ratio on the {scope_text}: numerator scores steered deployment "
            "rollout tokens under the original base model; denominator scores the sleeper model's no-deployment "
            "rollout tokens for the same prompt and generation seed under the same base model. Lower x and higher y "
            "are better; x=1 means steered deployment text is as base-plausible as the no-deployment sleeper rollout."
        )
    elif args.x_metric == "base_ce_raw":
        x_caption = (
            f"X-axis is raw mean base-model CE on the {scope_text}, scoring steered deployment rollout tokens under "
            "the original base model. Lower x and higher y are better."
        )
    else:
        x_caption = (
            f"X-axis is the summed normalized CE ratio on the {scope_text}: numerator compares the "
            "clean sleeper reference to the steered deployment rollout; denominator compares the same clean reference to "
            "an independent clean sleeper rollout. Lower x and higher y are better; x=1 means steered divergence matches "
            "ordinary clean sampling variability."
        )
    caption = (
        f"Y-axis counts sampled deployment rollouts where the sleeper phrase is prevented, relative to the "
        f"unsteered deployment baseline for each intervention family and generation seed (up to {total_rollouts} baseline sleeper hits per seed in this "
        f"run). {seed_sentence} Darker markers indicate larger intervention alpha. {x_caption}"
    )
    title_suffix = " by generation seed" if args.facet_by_seed else ""
    fig.suptitle(f"Rollout divergence ratio vs sleeper suppression{title_suffix}", fontsize=18, y=0.99)
    fig.text(0.5, 0.018, textwrap.fill(caption, 155), ha="center", va="bottom", fontsize=10.0, color="#374151")
    fig.subplots_adjust(left=0.08, right=0.98, top=0.80, bottom=0.34, wspace=0.12)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))
    print(f"[rollout-plot] wrote {out_path}")


if __name__ == "__main__":
    main()
