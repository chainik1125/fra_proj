"""Plot XE(steered, clean) and XE(steered, unsteered_pp) vs steering α.

For each family (Single feature vs OV/FRA) plot two curves:
  - num(α) = mean -log p_clean(s_t | clean+c1)            (surprise under clean)
  - den(α) = mean -log p_unsteered_pp(s_t | dep+s)        (surprise under unsteered-poisoned)

Both averaged across (prompt, sample_seed, position=1) by default.
The ratio of the two is the new clean-vs-pp metric.

Usage:
    python plot_xe_vs_alpha.py \
        --aggregate_root ketan_repl/seed_aggregate/ketan_50k_cvspp/aggregate_inputs \
        --output ketan_repl/seed_aggregate/ketan_50k_cvspp/plots/xe_vs_alpha_first_token.png
"""

from __future__ import annotations

import argparse
import csv
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(aggregate_root: Path, scope: str = "first_token"):
    """Return {family: {alpha: {'num': mean_num, 'den': mean_den, 'num_sd':, 'den_sd':}}}.

    Means are over (prompt, sample_seed) of the per-row token-CE values at the
    requested scope (first_token = pos 1; all_tokens = positions 1..G).
    """
    out: dict[str, dict[float, dict[str, list[float]]]] = {}
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        with open(seed_dir / "per_token_metrics.csv") as f:
            for row in csv.DictReader(f):
                if scope == "first_token" and int(row["position"]) != 1:
                    continue
                family = row["family"]
                alpha = float(row["alpha"])
                num = float(row["token_ce_clean_to_steered"])
                den = float(row["token_ce_pp_self"])
                out.setdefault(family, {}).setdefault(alpha, {"nums": [], "dens": []})
                out[family][alpha]["nums"].append(num)
                out[family][alpha]["dens"].append(den)

    summary: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in out.items():
        summary[family] = {}
        for alpha, vals in by_alpha.items():
            summary[family][alpha] = {
                "num_mean": statistics.mean(vals["nums"]),
                "num_sd":   statistics.pstdev(vals["nums"]),
                "den_mean": statistics.mean(vals["dens"]),
                "den_sd":   statistics.pstdev(vals["dens"]),
                "n":        len(vals["nums"]),
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["first_token", "all_tokens"], default="first_token")
    p.add_argument("--title", default="XE(steered, clean) and XE(steered, unsteered_pp) vs steering α")
    args = p.parse_args()

    summary = collect(args.aggregate_root, args.scope)
    families = sorted(summary.keys())

    fig, ax = plt.subplots(figsize=(9, 6))
    fam_color = {"OV/FRA": "#1f77b4", "Single feature": "#9467bd"}
    for family in families:
        alphas = sorted(summary[family].keys())
        nums = [summary[family][a]["num_mean"] for a in alphas]
        dens = [summary[family][a]["den_mean"] for a in alphas]
        nums_sd = [summary[family][a]["num_sd"] for a in alphas]
        dens_sd = [summary[family][a]["den_sd"] for a in alphas]
        c = fam_color.get(family, "#888")
        ax.errorbar(alphas, nums, yerr=nums_sd, color=c, linestyle="-", marker="o",
                    lw=2, capsize=2,
                    label=f"{family}: −log p_clean(s_t)  [num]")
        ax.errorbar(alphas, dens, yerr=dens_sd, color=c, linestyle="--", marker="s",
                    lw=2, capsize=2,
                    label=f"{family}: −log p_unst_pp(s_t)  [den]")

    ax.set_xlabel("steering coefficient  α", fontsize=13)
    ax.set_ylabel("per-token NLL of steered token (nats)\n(error bars = std across prompts × seeds)",
                  fontsize=12)
    ax.set_title(args.title, fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, frameon=True, framealpha=0.9)
    ax.set_yscale("log")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Also print the table
    print(f"\n{'family':<16} {'α':>5}  {'num_mean':>10}  {'den_mean':>10}  {'CvP_ratio':>10}")
    for family in families:
        for a in sorted(summary[family].keys()):
            s = summary[family][a]
            ratio = s["num_mean"] / max(s["den_mean"], 1e-8)
            print(f"{family:<16} {a:>5.2f}  {s['num_mean']:>10.3f}  {s['den_mean']:>10.3f}  {ratio:>10.3f}")


if __name__ == "__main__":
    main()
