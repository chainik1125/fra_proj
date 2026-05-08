"""Plot rollout-divergence tradeoff averaged over generation seeds per training seed."""

from __future__ import annotations

import argparse
import csv
import json
import re
import textwrap
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np


SLEEPER_REGEX = re.compile(r"i\s*hate\s*you", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_root",
        required=True,
        help="Root containing train_seed_* folders with rollouts.jsonl + per_token_metrics.csv.",
    )
    parser.add_argument("--output", required=True)
    parser.add_argument("--scope", choices=["first_token", "all_tokens"], default="all_tokens")
    parser.add_argument("--metric", choices=["token", "dist"], default="token")
    parser.add_argument("--x_scale", choices=["log", "linear"], default="linear")
    parser.add_argument(
        "--aggregate_train_seeds",
        action="store_true",
        help="Average across training seeds too (one line per intervention family).",
    )
    return parser.parse_args()


def sleeper_counts_by_seed(rollout_rows: list[dict]) -> dict[tuple[str, float, int], int]:
    counts: dict[tuple[str, float, int], int] = {}
    for row in rollout_rows:
        key = (row["family"], float(row["alpha"]), int(row["sample_seed"]))
        counts.setdefault(key, 0)
        if SLEEPER_REGEX.search(row["steered"]):
            counts[key] += 1
    return counts


def ratio_by_seed(
    metric_rows: list[dict],
    scope: str,
    metric: str,
) -> dict[tuple[str, float, int], float]:
    cols = {
        "token": ("token_ce_clean_to_steered", "token_ce_clean_to_clean"),
        "dist": ("dist_ce_clean_to_steered", "dist_ce_clean_to_clean"),
    }
    num_col, den_col = cols[metric]
    grouped: dict[tuple[str, float, int], list[tuple[float, float]]] = {}
    for row in metric_rows:
        if scope == "first_token" and int(row["position"]) != 1:
            continue
        key = (row["family"], float(row["alpha"]), int(row["sample_seed"]))
        grouped.setdefault(key, []).append((float(row[num_col]), float(row[den_col])))
    out = {}
    for key, vals in grouped.items():
        num = sum(v[0] for v in vals)
        den = sum(v[1] for v in vals)
        out[key] = num / max(den, 1e-12)
    return out


def load_train_seed_dir(path: Path, train_seed: int, scope: str, metric: str) -> tuple[list[dict], list[dict]]:
    rollouts = [json.loads(line) for line in (path / "rollouts.jsonl").read_text().splitlines() if line.strip()]
    metric_rows = list(csv.DictReader((path / "per_token_metrics.csv").open()))

    sleepers = sleeper_counts_by_seed(rollouts)
    ratios = ratio_by_seed(metric_rows, scope=scope, metric=metric)
    baseline_by_family_seed = {
        (family, sample_seed): count
        for (family, alpha, sample_seed), count in sleepers.items()
        if alpha == 0.0
    }

    by_family_alpha: dict[tuple[str, float], list[tuple[int, float, float]]] = {}
    for (family, alpha, sample_seed), hit_count in sleepers.items():
        x = ratios.get((family, alpha, sample_seed))
        if x is None:
            continue
        y = baseline_by_family_seed.get((family, sample_seed), hit_count) - hit_count
        by_family_alpha.setdefault((family, alpha), []).append((sample_seed, x, y))

    mean_rows = []
    point_rows = []
    for (family, alpha), vals in sorted(by_family_alpha.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        for sample_seed, x, y in vals:
            point_rows.append(
                {
                    "train_seed": train_seed,
                    "family": family,
                    "alpha": alpha,
                    "sample_seed": sample_seed,
                    "x": x,
                    "y": y,
                }
            )
        mean_rows.append(
            {
                "train_seed": train_seed,
                "family": family,
                "alpha": alpha,
                "x_mean": mean(v[1] for v in vals),
                "y_mean": mean(v[2] for v in vals),
                "n_generation_seeds": len(vals),
            }
        )
    return mean_rows, point_rows


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root)
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    seed_dirs = sorted(
        [p for p in input_root.iterdir() if p.is_dir() and p.name.startswith("train_seed_")],
        key=lambda p: int(p.name.split("_")[-1]),
    )
    if not seed_dirs:
        raise ValueError(f"No train_seed_* folders found in {input_root}")

    all_rows: list[dict] = []
    all_point_rows: list[dict] = []
    for d in seed_dirs:
        train_seed = int(d.name.split("_")[-1])
        for req in ["rollouts.jsonl", "per_token_metrics.csv", "summary.json"]:
            if not (d / req).exists():
                raise FileNotFoundError(f"Missing {req} in {d}")
        mean_rows, point_rows = load_train_seed_dir(d, train_seed, scope=args.scope, metric=args.metric)
        all_rows.extend(mean_rows)
        all_point_rows.extend(point_rows)

    families = ["OV/FRA", "Single feature"]
    family_labels = {"OV/FRA": "OV/FRA upstream features", "Single feature": "Single best resid-mid feature"}
    family_markers = {"OV/FRA": "s", "Single feature": "o"}
    family_linestyle = {"OV/FRA": "-", "Single feature": "--"}

    train_seeds = sorted({int(r["train_seed"]) for r in all_rows})
    cmap = plt.get_cmap("tab10")
    seed_colors = {seed: cmap(i % 10) for i, seed in enumerate(train_seeds)}
    alpha_vals = [float(r["alpha"]) for r in all_rows]
    alpha_norm = mcolors.Normalize(vmin=min(alpha_vals), vmax=max(alpha_vals))

    fig, ax = plt.subplots(1, 1, figsize=(10.8, 7.6))
    fig.patch.set_facecolor("#fbfaf6")
    ax.set_facecolor("#fbfaf6")

    if args.aggregate_train_seeds:
        agg_rows: list[dict] = []
        grouped: dict[tuple[str, float], list[dict]] = {}
        for r in all_point_rows:
            grouped.setdefault((r["family"], float(r["alpha"])), []).append(r)
        for (family, alpha), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            x_vals = [float(v["x"]) for v in vals]
            y_vals = [float(v["y"]) for v in vals]
            x_q25, x_q75 = np.percentile(x_vals, [25, 75]).tolist()
            y_q25, y_q75 = np.percentile(y_vals, [25, 75]).tolist()
            agg_rows.append(
                {
                    "family": family,
                    "alpha": alpha,
                    "x_mean": mean(x_vals),
                    "x_q25": x_q25,
                    "x_q75": x_q75,
                    "y_mean": mean(y_vals),
                    "y_q25": y_q25,
                    "y_q75": y_q75,
                    "n_points": len(vals),
                    "n_training_seeds": len({int(v["train_seed"]) for v in vals}),
                    "n_generation_seeds": len({int(v["sample_seed"]) for v in vals}),
                }
            )

        family_colors = {"OV/FRA": "#15616d", "Single feature": "#c44900"}
        for family in families:
            pts = [r for r in agg_rows if r["family"] == family]
            pts.sort(key=lambda r: float(r["alpha"]))
            if not pts:
                continue
            xs = [float(r["x_mean"]) for r in pts]
            ys = [float(r["y_mean"]) for r in pts]
            xerr = [
                [max(0.0, float(r["x_mean"]) - float(r["x_q25"])) for r in pts],
                [max(0.0, float(r["x_q75"]) - float(r["x_mean"])) for r in pts],
            ]
            yerr = [
                [max(0.0, float(r["y_mean"]) - float(r["y_q25"])) for r in pts],
                [max(0.0, float(r["y_q75"]) - float(r["y_mean"])) for r in pts],
            ]
            ax.errorbar(
                xs,
                ys,
                xerr=xerr,
                yerr=yerr,
                fmt="none",
                ecolor=family_colors[family],
                elinewidth=0.9,
                capsize=2.0,
                alpha=0.35,
                zorder=1,
            )
            ax.plot(
                xs,
                ys,
                color=family_colors[family],
                linestyle=family_linestyle[family],
                linewidth=1.8,
                alpha=0.95,
                zorder=2,
            )
            ax.scatter(
                xs,
                ys,
                c=[float(r["alpha"]) for r in pts],
                cmap="Greys",
                norm=alpha_norm,
                marker=family_markers[family],
                s=50,
                edgecolor=family_colors[family],
                linewidth=1.6,
                zorder=3,
            )
    else:
        for seed in train_seeds:
            for family in families:
                pts = [r for r in all_rows if int(r["train_seed"]) == seed and r["family"] == family]
                pts.sort(key=lambda r: float(r["alpha"]))
                if not pts:
                    continue
                xs = [float(r["x_mean"]) for r in pts]
                ys = [float(r["y_mean"]) for r in pts]
                # training seed encoded by line color, family encoded by marker/linestyle
                ax.plot(
                    xs,
                    ys,
                    color=seed_colors[seed],
                    linestyle=family_linestyle[family],
                    linewidth=1.5,
                    alpha=0.9,
                    zorder=2,
                )
                ax.scatter(
                    xs,
                    ys,
                    c=[float(r["alpha"]) for r in pts],
                    cmap="Greys",
                    norm=alpha_norm,
                    marker=family_markers[family],
                    s=64,
                    edgecolor=seed_colors[seed],
                    linewidth=1.6,
                    zorder=3,
                )

    ax.axvline(1.0, color="#6b7280", linestyle=":", linewidth=1.5)
    if args.x_scale == "log":
        ax.set_xscale("log")
    ax.grid(True, color="#e5e7eb", linewidth=1.0)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    metric_name = "token CE" if args.metric == "token" else "next-token distribution CE"
    ax.set_xlabel(f"Clean-to-steered {metric_name} / clean-to-clean {metric_name}", fontsize=13)
    ax.set_ylabel("Sleepers removed (mean over generation seeds)", fontsize=13)

    if not args.aggregate_train_seeds:
        # Legend 1: training seed colors.
        seed_handles = []
        seed_labels = []
        for seed in train_seeds:
            h = ax.plot([], [], color=seed_colors[seed], linewidth=2.2)[0]
            seed_handles.append(h)
            seed_labels.append(f"Train seed {seed}")
        leg1 = ax.legend(seed_handles, seed_labels, loc="upper right", frameon=False, title="Training seed")
        ax.add_artist(leg1)

    # Legend 2: family shape/line semantics.
    fam_handles = []
    fam_labels = []
    for family in families:
        h = ax.plot([], [], color="#374151", linestyle=family_linestyle[family], marker=family_markers[family], linewidth=1.4)[0]
        fam_handles.append(h)
        fam_labels.append(family_labels[family])
    hline = ax.plot([], [], color="#6b7280", linestyle=":", linewidth=1.5)[0]
    fam_handles.append(hline)
    fam_labels.append("Clean sampling variability (x=1)")
    if args.aggregate_train_seeds:
        fig.legend(
            fam_handles,
            fam_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.84),
            ncol=3,
            frameon=False,
        )
    else:
        ax.legend(fam_handles, fam_labels, loc="upper left", frameon=False)

    sm = plt.cm.ScalarMappable(norm=alpha_norm, cmap=plt.get_cmap("Greys"))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, orientation="horizontal", fraction=0.055, pad=0.14, aspect=40)
    cbar.set_label("Intervention alpha", fontsize=11)

    if args.aggregate_train_seeds:
        caption = (
            "Each point pools all generation-seed rollouts across all training seeds, for a fixed "
            "intervention family and alpha. Solid squares are OV/FRA and dashed circles are single-feature. "
            "Whiskers show 25th-75th percentile ranges across pooled generation-seed points on both axes. "
            "Darker markers indicate larger alpha. Lower x and higher y are better."
        )
    else:
        caption = (
            "Each point is first averaged across generation seeds within one training seed, "
            "for a fixed intervention family and alpha. Colored lines connect alphas for a "
            "single training seed; solid squares are OV/FRA and dashed circles are single-feature. "
            "Darker markers indicate larger alpha. Lower x and higher y are better."
        )
    scope_text = "first token" if args.scope == "first_token" else "all generated tokens"
    if args.aggregate_train_seeds:
        title = f"Rollout divergence vs sleeper suppression (avg over generation + training seeds, {scope_text})"
    else:
        title = f"Rollout divergence vs sleeper suppression (avg over generation seeds, {scope_text})"
    fig.suptitle(title, fontsize=17, y=0.98)
    fig.text(0.5, 0.02, textwrap.fill(caption, 145), ha="center", va="bottom", fontsize=10.5, color="#374151")
    if args.aggregate_train_seeds:
        fig.tight_layout(rect=(0.03, 0.11, 0.98, 0.86))
    else:
        fig.tight_layout(rect=(0.03, 0.11, 0.98, 0.92))
    fig.savefig(out_path, dpi=220)
    fig.savefig(out_path.with_suffix(".pdf"))

    rows_csv = out_path.with_suffix(".csv")
    with rows_csv.open("w", newline="") as f:
        if args.aggregate_train_seeds:
            grouped: dict[tuple[str, float], list[dict]] = {}
            for r in all_point_rows:
                grouped.setdefault((r["family"], float(r["alpha"])), []).append(r)
            agg_rows: list[dict] = []
            for (family, alpha), vals in sorted(grouped.items(), key=lambda kv: (kv[0][0], kv[0][1])):
                x_vals = [float(v["x"]) for v in vals]
                y_vals = [float(v["y"]) for v in vals]
                x_q25, x_q75 = np.percentile(x_vals, [25, 75]).tolist()
                y_q25, y_q75 = np.percentile(y_vals, [25, 75]).tolist()
                agg_rows.append(
                    {
                        "family": family,
                        "alpha": alpha,
                        "x_mean": mean(x_vals),
                        "x_q25": x_q25,
                        "x_q75": x_q75,
                        "y_mean": mean(y_vals),
                        "y_q25": y_q25,
                        "y_q75": y_q75,
                        "n_points": len(vals),
                        "n_training_seeds": len({int(v["train_seed"]) for v in vals}),
                        "n_generation_seeds": len({int(v["sample_seed"]) for v in vals}),
                    }
                )
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "family",
                    "alpha",
                    "x_mean",
                    "x_q25",
                    "x_q75",
                    "y_mean",
                    "y_q25",
                    "y_q75",
                    "n_points",
                    "n_training_seeds",
                    "n_generation_seeds",
                ],
            )
            writer.writeheader()
            writer.writerows(agg_rows)
        else:
            writer = csv.DictWriter(
                f,
                fieldnames=["train_seed", "family", "alpha", "x_mean", "y_mean", "n_generation_seeds"],
            )
            writer.writeheader()
            writer.writerows(all_rows)
    print(f"[train-seed-avg-plot] wrote {out_path}")
    print(f"[train-seed-avg-plot] wrote {rows_csv}")


if __name__ == "__main__":
    main()
