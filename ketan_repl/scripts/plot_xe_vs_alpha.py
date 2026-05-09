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


def collect(aggregate_root: Path, scope: str = "first_token",
            num_col: str = "token_ce_clean_to_steered",
            den_col: str = "token_ce_pp_self"):
    """Return {family: {alpha: {'num_mean':, 'den_mean':, 'num_sd':, 'den_sd':, 'n':}}}.

    Reads per_token_metrics.csv from each train_seed_* dir under aggregate_root,
    picks the rows at the requested scope, and aggregates `num_col` and `den_col`.
    """
    out: dict[str, dict[float, dict[str, list[float]]]] = {}
    for seed_dir in sorted(aggregate_root.glob("train_seed_*")):
        with open(seed_dir / "per_token_metrics.csv") as f:
            for row in csv.DictReader(f):
                if scope == "first_token" and int(row["position"]) != 1:
                    continue
                family = row["family"]
                alpha = float(row["alpha"])
                try:
                    num = float(row[num_col])
                    den = float(row[den_col])
                except KeyError:
                    raise SystemExit(
                        f"missing column {num_col!r} or {den_col!r} in {seed_dir/'per_token_metrics.csv'}"
                    )
                if not (num == num and den == den):  # skip NaN rows
                    continue
                out.setdefault(family, {}).setdefault(alpha, {"nums": [], "dens": []})
                out[family][alpha]["nums"].append(num)
                out[family][alpha]["dens"].append(den)

    summary: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in out.items():
        summary[family] = {}
        for alpha, vals in by_alpha.items():
            summary[family][alpha] = {
                "num_mean": statistics.mean(vals["nums"]) if vals["nums"] else float("nan"),
                "num_sd":   statistics.pstdev(vals["nums"]) if len(vals["nums"]) > 1 else 0.0,
                "den_mean": statistics.mean(vals["dens"]) if vals["dens"] else float("nan"),
                "den_sd":   statistics.pstdev(vals["dens"]) if len(vals["dens"]) > 1 else 0.0,
                "n":        len(vals["nums"]),
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--aggregate_root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--scope", choices=["first_token", "all_tokens"], default="first_token")
    p.add_argument("--title", default="XE(steered, clean) and XE(steered, unsteered_pp) vs steering α")
    p.add_argument("--family", default=None,
                   help='Filter to one family (e.g. "Single feature" or "OV/FRA"). Default: plot all.')
    p.add_argument("--linear_y", action="store_true",
                   help="Use linear y-scale instead of log.")
    p.add_argument("--metric", choices=["xe", "jsd"], default="xe",
                   help="xe (default): plot −log p_clean(s) and −log p_unst_pp(s). "
                        "jsd: plot JSD(p_steered, p_clean) and JSD(p_steered, p_unst_pp).")
    p.add_argument("--num_label", default=None,
                   help="Override the legend label for the numerator curve.")
    p.add_argument("--den_label", default=None,
                   help="Override the legend label for the denominator curve.")
    p.add_argument("--ylabel", default=None,
                   help="Override the y-axis label.")
    args = p.parse_args()

    if args.metric == "xe":
        num_col, den_col = "token_ce_clean_to_steered", "token_ce_pp_self"
        default_num_label = "surprise of clean model at steered token"
        default_den_label = "surprise of unsteered-poisoned model at steered token"
        default_ylabel = "per-token NLL of steered token (nats)"
    else:  # jsd
        num_col, den_col = "jsd_steered_to_clean", "jsd_steered_to_pp"
        default_num_label = "JSD(p_steered, p_clean)  — distance from clean"
        default_den_label = "JSD(p_steered, p_unsteered_pp)  — distance from sleeper"
        default_ylabel = "Jensen-Shannon divergence (nats; bounded ≤ ln 2 ≈ 0.693)"
    num_label = args.num_label or default_num_label
    den_label = args.den_label or default_den_label
    ylabel = args.ylabel or default_ylabel

    summary = collect(args.aggregate_root, args.scope, num_col=num_col, den_col=den_col)
    families = sorted(summary.keys()) if args.family is None else [args.family]
    if args.family and args.family not in summary:
        raise SystemExit(f"family {args.family!r} not found; have {sorted(summary)}")

    fig, ax = plt.subplots(figsize=(9, 6))
    NUM_COLOR = "#2ca02c"  # green — clean = good (lower curve = closer to clean)
    DEN_COLOR = "#d62728"  # red   — sleeper = bad
    fam_marker_num = {"OV/FRA": "o", "Single feature": "o"}
    fam_marker_den = {"OV/FRA": "s", "Single feature": "s"}
    for family in families:
        alphas = sorted(summary[family].keys())
        nums = [summary[family][a]["num_mean"] for a in alphas]
        dens = [summary[family][a]["den_mean"] for a in alphas]
        nums_sd = [summary[family][a]["num_sd"] for a in alphas]
        dens_sd = [summary[family][a]["den_sd"] for a in alphas]
        # When plotting a single family, use clean color-encoded labels
        # (one curve = "clean's surprise", the other = "unsteered-poisoned's surprise").
        suffix = "" if args.family else f" — {family}"
        ax.plot(alphas, nums, color=NUM_COLOR, linestyle="-",
                marker=fam_marker_num.get(family, "o"), lw=2.2, markersize=7,
                label=f"{num_label}{suffix}")
        ax.plot(alphas, dens, color=DEN_COLOR, linestyle="-",
                marker=fam_marker_den.get(family, "s"), lw=2.2, markersize=7,
                label=f"{den_label}{suffix}")

    ax.set_xlabel("steering coefficient α", fontsize=13)
    ax.set_ylabel(ylabel, fontsize=13)
    ax.set_title(args.title, fontsize=13)
    ax.grid(True, alpha=0.3)
    if not args.linear_y:
        ax.set_yscale("log")
    # Crossover annotation: where the two curves meet in α
    if len(families) == 1:
        family = families[0]
        alphas = sorted(summary[family].keys())
        nums = [summary[family][a]["num_mean"] for a in alphas]
        dens = [summary[family][a]["den_mean"] for a in alphas]
        # find first alpha where den >= num
        cross = None
        for i in range(len(alphas)):
            if dens[i] >= nums[i]:
                cross = alphas[i]; break
        if cross is not None:
            ax.axvline(cross, color="#888", linestyle=":", lw=1)
            ax.annotate(f"crossover α≈{cross:.2f}\n(steered token equally surprising\nto both models)",
                        xy=(cross, max(nums + dens) * 0.5),
                        xytext=(cross + 0.15, max(nums + dens) * 0.5),
                        fontsize=10, color="#444",
                        arrowprops=dict(arrowstyle="-", color="#888", lw=0.8))
    ax.legend(loc="lower left", fontsize=11, frameon=True, framealpha=0.95)
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
