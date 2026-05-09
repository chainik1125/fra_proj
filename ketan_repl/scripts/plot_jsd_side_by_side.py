"""Two-panel JSD-vs-α plot: Single feature (left) vs OV/FRA top-50 (right).

Shared y-axis so the magnitudes are visually comparable.

Usage:
    python plot_jsd_side_by_side.py \
        --aggregate_root ketan_repl/seed_aggregate/ketan_50k_jsd/aggregate_inputs \
        --output ketan_repl/seed_aggregate/ketan_50k_jsd/plots/jsd_side_by_side_50k.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect_jsd(aggregate_root: Path, scope: str = "first_token"):
    out: dict[str, dict[float, dict[str, list[float]]]] = {}
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        with open(seed_dir / "per_token_metrics.csv") as f:
            for row in csv.DictReader(f):
                if scope == "first_token" and int(row["position"]) != 1:
                    continue
                family = row["family"]
                alpha = float(row["alpha"])
                try:
                    j_clean = float(row["jsd_steered_to_clean"])
                    j_pp = float(row["jsd_steered_to_pp"])
                except KeyError:
                    raise SystemExit(
                        f"missing JSD columns in {seed_dir/'per_token_metrics.csv'}; "
                        "rerun with the JSD-instrumented rollout_divergence_ratio.py"
                    )
                if not (j_clean == j_clean and j_pp == j_pp):
                    continue
                out.setdefault(family, {}).setdefault(alpha, {"clean": [], "pp": []})
                out[family][alpha]["clean"].append(j_clean)
                out[family][alpha]["pp"].append(j_pp)
    summary: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in out.items():
        summary[family] = {}
        for alpha, vals in by_alpha.items():
            summary[family][alpha] = {
                "clean_mean": statistics.mean(vals["clean"]),
                "pp_mean":    statistics.mean(vals["pp"]),
                "n":          len(vals["clean"]),
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["first_token", "all_tokens"], default="first_token")
    p.add_argument("--title", default=None)
    args = p.parse_args()

    summary = collect_jsd(args.aggregate_root, args.scope)
    expected = ["Single feature", "OV/FRA"]
    for fam in expected:
        if fam not in summary:
            raise SystemExit(f"family {fam!r} not found in {args.aggregate_root}; have {sorted(summary)}")

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)
    NUM_COLOR = "#2ca02c"  # green — distance from clean (we want this small; clean = good)
    DEN_COLOR = "#d62728"  # red   — distance from sleeper-emitting unsteered_pp (sleeper = bad)

    panel_titles = {
        "Single feature": "(a)  conventional steering · single resid_mid feature",
        "OV/FRA":         "(b)  OV → OV · 50 ln1 features via V-pathway",
    }

    for ax, family in zip(axes, expected):
        alphas = sorted(summary[family].keys())
        cleans = [summary[family][a]["clean_mean"] for a in alphas]
        pps    = [summary[family][a]["pp_mean"]    for a in alphas]

        ax.plot(alphas, cleans, color=NUM_COLOR, linestyle="-", marker="o", lw=2.2,
                markersize=7, label="JSD(p_steered , p_clean)  →  distance from clean")
        ax.plot(alphas, pps, color=DEN_COLOR, linestyle="-", marker="s", lw=2.2,
                markersize=7, label="JSD(p_steered , p_unsteered_pp)  →  distance from sleeper")

        # Crossover where two lines intersect
        cross = None
        for i in range(len(alphas)):
            if pps[i] >= cleans[i]:
                cross = alphas[i]; break
        if cross is not None:
            ax.axvline(cross, color="#888", linestyle=":", lw=1)
            ax.annotate(f"crossover\nα ≈ {cross:.2f}", xy=(cross, 0.4),
                        xytext=(cross + 0.05, 0.50), fontsize=10, color="#444",
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))

        ax.axhline(0.6931, color="#666", linestyle="--", lw=0.8, alpha=0.6)
        ax.text(2.0, 0.66, "JSD upper bound (ln 2)", fontsize=9, color="#666",
                ha="right", va="bottom")

        ax.set_xlabel("steering coefficient  α", fontsize=12)
        ax.set_title(panel_titles[family], fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.03, 0.75)
        ax.legend(loc="lower left", fontsize=10, frameon=True, framealpha=0.95)

    axes[0].set_ylabel("Jensen-Shannon divergence (nats)", fontsize=12)

    if args.title:
        fig.suptitle(args.title, fontsize=14)
    else:
        fig.suptitle(
            "50k SAEs · JSD of steered next-token distribution vs the clean and sleeper distributions\n"
            "(closer to red ↓ = behaving like the clean model · closer to green ↑ = behaving like the unsteered sleeper · "
            "the two converging upward = word salad)",
            fontsize=12, y=1.02,
        )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Print numbers
    print(f"\n{'family':<16} {'α':>5}  {'jsd→clean':>10}  {'jsd→pp':>9}  {'ratio':>9}")
    for family in expected:
        for a in sorted(summary[family].keys()):
            s = summary[family][a]
            r = s["clean_mean"] / max(s["pp_mean"], 1e-8)
            print(f"{family:<16} {a:>5.2f}  {s['clean_mean']:>10.4f}  {s['pp_mean']:>9.4f}  {r:>9.3f}")


if __name__ == "__main__":
    main()
