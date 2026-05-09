"""Two-panel jamie-pipeline plot: Single resid_mid (left) vs OV top-50 set (right).

For each panel plot ASR, gen-CE ratio, severity ratio (recovery_noise_ratio)
vs steering α. Designed to be the jamie-pipeline analog of our JSD
side-by-side — same layout, but using jamie's two complementary
generated-on-deployment metrics that probe token-level vs distribution-level
damage.

Usage:
    python plot_jamie_side_by_side.py \
        --inputs ketan_repl/seed_aggregate/jamie_50k/dmitry_50k_seed{0,1,2}.json \
        --output ketan_repl/seed_aggregate/jamie_50k/jamie_side_by_side_50k.png
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def collect(input_paths: list[Path]):
    """Return {family: {alpha: {asr, sev, gence}}} averaged over the seeds.

    family ∈ {"upstream", "downstream"}
    """
    raw: dict[str, dict[float, dict[str, list[float]]]] = {}
    for p in input_paths:
        d = json.loads(p.read_text())
        for pt in d.get("points", []):
            family = pt.get("family")
            if family not in ("upstream", "downstream"):
                continue
            alpha = float(pt["alpha"])
            asr   = float(pt.get("asr", float("nan")))
            sev   = pt.get("severity_ratio") or pt.get("recovery_noise_ratio")
            gen   = pt.get("gen_ce_ratio")
            entry = raw.setdefault(family, {}).setdefault(alpha, {"asr": [], "sev": [], "gen": []})
            if asr == asr:                 entry["asr"].append(asr)
            if sev is not None:            entry["sev"].append(float(sev))
            if gen is not None:            entry["gen"].append(float(gen))

    summary: dict[str, dict[float, dict[str, float]]] = {}
    for family, by_alpha in raw.items():
        summary[family] = {}
        for alpha, vals in by_alpha.items():
            summary[family][alpha] = {
                "asr_mean": statistics.mean(vals["asr"]) if vals["asr"] else float("nan"),
                "sev_mean": statistics.mean(vals["sev"]) if vals["sev"] else float("nan"),
                "gen_mean": statistics.mean(vals["gen"]) if vals["gen"] else float("nan"),
                "n":        len(vals["asr"]),
            }
    return summary


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    p.add_argument("--linear_y", action="store_true")
    args = p.parse_args()

    summary = collect(args.inputs)
    if "upstream" not in summary or "downstream" not in summary:
        raise SystemExit(f"need both 'set' and 'downstream' families; have {sorted(summary)}")

    # ---- plot ----
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.5), sharey=True)

    panel_titles = {
        "downstream": "(a)  conventional steering · single resid_mid feature",
        "upstream":         "(b)  OV → OV · 50 ln1 features (V-pathway)",
    }
    GEN_COLOR = "#d62728"   # red — token-level damage (sleeper = bad)
    SEV_COLOR = "#9467bd"   # purple — distribution-level damage
    ASR_COLOR = "#2ca02c"   # green — sleeper presence (lower = good)

    for ax, family in zip(axes, ("downstream", "upstream")):
        alphas = sorted(summary[family].keys())
        gens = [summary[family][a]["gen_mean"] for a in alphas]
        sevs = [summary[family][a]["sev_mean"] for a in alphas]
        asrs = [summary[family][a]["asr_mean"] for a in alphas]

        ax.plot(alphas, gens, color=GEN_COLOR, linestyle="-", marker="o",
                lw=2.2, markersize=7,
                label="gen-CE ratio (token-level NLL of steered text under clean ref)")
        ax.plot(alphas, sevs, color=SEV_COLOR, linestyle="-", marker="s",
                lw=2.2, markersize=7,
                label="severity ratio (dist-CE clean→steered / clean→clean noise)")
        ax.plot(alphas, asrs, color=ASR_COLOR, linestyle=":", marker="^",
                lw=2.0, markersize=7,
                label="ASR (fraction of rollouts emitting sleeper)")

        ax.axhline(1.0, color="#888", linestyle="--", lw=0.8, alpha=0.6)
        ax.text(2.0, 1.04, "no damage / sampling-noise floor", fontsize=9, color="#666",
                ha="right", va="bottom")

        ax.set_xlabel("steering coefficient α", fontsize=12)
        ax.set_title(panel_titles[family], fontsize=12)
        ax.grid(True, alpha=0.3)
        if not args.linear_y:
            ax.set_yscale("log")
            ax.set_ylim(0.005, 50)
        ax.legend(loc="upper left", fontsize=9.5, frameon=True, framealpha=0.95)

    axes[0].set_ylabel("ratio  (gen-CE / severity);  fraction (ASR)", fontsize=12)

    if args.title:
        fig.suptitle(args.title, fontsize=14)
    else:
        fig.suptitle(
            "50k SAEs · jamie/sleepers feature_set_pipeline · seed-mean over 3 SAE seeds, 3 sample seeds, top_k=50, ketan ranking",
            fontsize=11.5, y=1.02,
        )
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Print numbers
    print(f"\n{'family':<12} {'α':>5}  {'ASR':>6} {'gen-CE':>9} {'severity':>9}  (n)")
    for family in ("downstream", "upstream"):
        for a in sorted(summary[family].keys()):
            s = summary[family][a]
            print(f"{family:<12} {a:>5.2f}  {s['asr_mean']:>6.3f} {s['gen_mean']:>9.3f} {s['sev_mean']:>9.3f}  ({s['n']})")


if __name__ == "__main__":
    main()
