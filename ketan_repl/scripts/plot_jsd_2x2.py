"""2x2 JSD plot.

Rows:
  top    — conventional steering (resid-mid additive, downstream feature 579)
  bottom — single-feature OV→OV steering (jamie's pipeline winner)

Cols:
  left   — 4k SAE
  right  — 50k SAE

Each panel: JSD(steered, clean) and JSD(steered, poisoned) vs α (bits).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="JSON from scripts/jsd_2x2_sweep.py")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    # Layout: rows = (conventional, ov_single); cols = (4k, 50k)
    cells = [
        [("conventional_4k",  "(a) conventional · resid-mid additive · 4k SAE"),
         ("conventional_50k", "(b) conventional · resid-mid additive · 50k SAE")],
        [("ov_single_4k",     "(c) single OV → OV · 4k SAE"),
         ("ov_single_50k",    "(d) single OV → OV · 50k SAE")],
    ]

    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, sharey=True)

    GREEN = "#2ca02c"   # closer to clean (low = good)
    RED   = "#d62728"   # closer to sleeper (low = bad)

    for r, row in enumerate(cells):
        for c, (key, title) in enumerate(row):
            ax = axes[r, c]
            res = cfg[key]["results"]
            jc = [res[str(a)]["jsd_clean"] for a in alphas]
            jp = [res[str(a)]["jsd_pois"]  for a in alphas]
            feat = cfg[key]["feature"]

            ax.plot(alphas, jc, color=GREEN, lw=2.4, marker="o", markersize=7,
                    label="JSD(steered , clean)  — distance from clean (lower=better)")
            ax.plot(alphas, jp, color=RED, lw=2.4, marker="s", markersize=7,
                    label="JSD(steered , poisoned)  — distance from sleeper (higher=better)")

            ax.axhline(1.0, color="#555", linestyle="--", lw=0.8, alpha=0.6)
            ax.text(alphas[-1], 1.0 - 0.02, "JSD upper bound (1 bit)", fontsize=9,
                    color="#555", ha="right", va="top")

            ax.set_title(f"{title}  (feature = {feat})", fontsize=11.5)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(-0.04, 1.1)
            ax.set_xticks(alphas)
            ax.set_xticklabels([f"{a:.2g}" for a in alphas], fontsize=9)
            ax.legend(loc="center right", fontsize=9, frameon=True, framealpha=0.95)

    for c in (0, 1):
        axes[1, c].set_xlabel("steering coefficient α", fontsize=11)
    for r in (0, 1):
        axes[r, 0].set_ylabel("Jensen-Shannon divergence (bits)", fontsize=11)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.00)
    else:
        fig.suptitle(
            "JSD vs α — conventional steering (top row) vs single OV → OV steering (bottom row)\n"
            "left = 4k SAE · right = 50k SAE · jamie pipeline winners on jamie 50k SAEs (sae_seed=0)",
            fontsize=11.5, y=1.00,
        )

    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # also print a numeric table
    print(f"\n{'config':<22} {'feature':>8}  {'α':>5}  {'JSD(s,clean)':>14}  {'JSD(s,pois)':>14}")
    for r, row in enumerate(cells):
        for c, (key, _) in enumerate(row):
            res = cfg[key]["results"]
            for a in alphas:
                jc = res[str(a)]["jsd_clean"]
                jp = res[str(a)]["jsd_pois"]
                print(f"{key:<22} {cfg[key]['feature']:>8}  {a:>5.2f}  {jc:>14.4f}  {jp:>14.4f}")


if __name__ == "__main__":
    main()
