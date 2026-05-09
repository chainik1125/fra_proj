"""Two-panel JSD-vs-alpha plot: single feature (left) vs OV/FRA (right).

Reads train_seed_*/per_token_metrics.csv aggregate inputs and plots:
  - JSD(p_steered, p_clean) as red line
  - JSD(p_steered, p_unsteered_pp) as green line

with shared y-axis and optional crossover annotation.
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["first_token", "all_tokens"], default="first_token")
    p.add_argument("--title", default=None)
    return p.parse_args()


def collect_jsd(aggregate_root: Path, scope: str) -> dict[str, dict[float, dict[str, float]]]:
    out: dict[str, dict[float, dict[str, list[float]]]] = {}
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        metrics_csv = seed_dir / "per_token_metrics.csv"
        if not metrics_csv.exists():
            continue
        with metrics_csv.open() as f:
            for row in csv.DictReader(f):
                if scope == "first_token" and int(row["position"]) != 1:
                    continue
                family = row["family"]
                alpha = float(row["alpha"])
                if family not in {"Single feature", "OV/FRA"}:
                    continue
                try:
                    jsd_clean = float(row["jsd_steered_to_clean"])
                    jsd_pp = float(row["jsd_steered_to_pp"])
                except KeyError as e:
                    raise SystemExit(
                        f"missing JSD columns in {metrics_csv}; rerun rollout_divergence_ratio.py with JSD-enabled version"
                    ) from e
                if not (jsd_clean == jsd_clean and jsd_pp == jsd_pp):
                    continue
                out.setdefault(family, {}).setdefault(alpha, {"clean": [], "pp": []})
                out[family][alpha]["clean"].append(jsd_clean)
                out[family][alpha]["pp"].append(jsd_pp)

    summary: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in out.items():
        summary[family] = {}
        for alpha, vals in by_alpha.items():
            summary[family][alpha] = {
                "clean_mean": statistics.mean(vals["clean"]),
                "pp_mean": statistics.mean(vals["pp"]),
                "n": len(vals["clean"]),
            }
    return summary


def infer_ov_feature_count_label(aggregate_root: Path) -> str | None:
    counts: set[int] = set()
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        metrics_csv = seed_dir / "per_token_metrics.csv"
        if not metrics_csv.exists():
            continue
        with metrics_csv.open() as f:
            for row in csv.DictReader(f):
                if row.get("family") != "OV/FRA":
                    continue
                raw = row.get("feature_count")
                if raw is None or raw == "":
                    continue
                try:
                    counts.add(int(float(raw)))
                except ValueError:
                    continue
    if not counts:
        return None
    if len(counts) == 1:
        return f"n={next(iter(counts))}"
    return f"n∈[{min(counts)}, {max(counts)}]"


def main() -> None:
    args = parse_args()
    summary = collect_jsd(args.aggregate_root, args.scope)
    ov_n_label = infer_ov_feature_count_label(args.aggregate_root)
    expected = ["Single feature", "OV/FRA"]
    for fam in expected:
        if fam not in summary:
            raise SystemExit(f"family {fam!r} not found in {args.aggregate_root}; found {sorted(summary)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    fig.patch.set_facecolor("#f5f5f5")

    clean_color = "#e63946"  # red
    pp_color = "#16a34a"  # green

    ov_title = "(b)  OV-ranked ln1 upstream features via block-0 V pathway"
    if ov_n_label:
        ov_title = f"{ov_title} ({ov_n_label})"

    panel_titles = {
        "Single feature": "(a)  conventional steering · single resid_mid feature",
        "OV/FRA": ov_title,
    }

    for ax, family in zip(axes, expected):
        alphas = sorted(summary[family].keys())
        jsd_clean_vals = [summary[family][a]["clean_mean"] for a in alphas]
        jsd_pp_vals = [summary[family][a]["pp_mean"] for a in alphas]

        ax.plot(
            alphas,
            jsd_clean_vals,
            color=clean_color,
            linestyle="-",
            marker="o",
            lw=2.0,
            markersize=6,
            label="JSD(p_steered , p_clean)  →  distance from clean",
        )
        ax.plot(
            alphas,
            jsd_pp_vals,
            color=pp_color,
            linestyle="-",
            marker="s",
            lw=2.0,
            markersize=6,
            label="JSD(p_steered , p_unsteered_pp)  →  distance from sleeper",
        )

        cross = None
        for i, alpha in enumerate(alphas):
            if jsd_pp_vals[i] >= jsd_clean_vals[i]:
                cross = alpha
                break
        if cross is not None:
            ax.axvline(cross, color="#888", linestyle=":", lw=0.9, alpha=0.8)
            ax.annotate(
                f"crossover\nα ≈ {cross:.2f}",
                xy=(cross, 0.40),
                xytext=(cross + 0.05, 0.50),
                fontsize=9,
                color="#555",
                arrowprops=dict(arrowstyle="-", color="#888", lw=0.8),
            )

        ax.axhline(0.6931, color="#666", linestyle="--", lw=0.8, alpha=0.6)
        ax.text(2.0, 0.66, "JSD upper bound (ln 2)", fontsize=9, color="#666", ha="right", va="bottom")

        ax.set_xlabel("steering coefficient  α", fontsize=12)
        ax.set_title(panel_titles[family], fontsize=13)
        ax.grid(True, alpha=0.28)
        ax.set_ylim(-0.03, 0.75)
        ax.legend(loc="lower left", fontsize=9.5, frameon=True, framealpha=0.95)

    axes[0].set_ylabel("Jensen-Shannon divergence (nats)", fontsize=12)

    if args.title:
        fig.suptitle(args.title, fontsize=13.5, y=1.02)
    else:
        scope_text = "first token" if args.scope == "first_token" else "all generated tokens"
        fig.suptitle(
            "JSD of steered next-token distribution vs clean and sleeper distributions\n"
            f"scope: {scope_text} (red low = clean-like, green high = sleeper-like, both high = word-salad)",
            fontsize=12.5,
            y=1.02,
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(f"[jsd-side-by-side] wrote {args.output}")


if __name__ == "__main__":
    main()
