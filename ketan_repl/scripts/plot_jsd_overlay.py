"""1x2 JSD overlay plot — both steering techniques on the same panel per SAE size.

Reads the same per-seed re-attributed sweep JSON used for the 2x2 plot, but
draws conventional + OV-single overlaid in each panel (means only, no bands).

Layout:
  left  — 4k SAE
  right — 50k SAE

Lines per panel:
  green solid  + ●   OV→OV  · JSD(steered, clean)        — lower = better
  red   solid  + ●   OV→OV  · JSD(steered, poisoned)     — higher = better
  green dashed + ▲   conventional · JSD(steered, clean)
  red   dashed + ▲   conventional · JSD(steered, poisoned)
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def reduce_mean(values):
    if isinstance(values, list):
        return float(statistics.mean(values))
    return float(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    GREEN = "#2ca02c"   # closer to clean (low = good)
    RED   = "#d62728"   # closer to sleeper (low = bad)

    panel_setup = [
        (axes[0], "4k SAE",  "ov_single_4k",  "conventional_4k",
         "(a)  4k SAE — conventional vs single OV → OV"),
        (axes[1], "50k SAE", "ov_single_50k", "conventional_50k",
         "(b)  50k SAE — conventional vs single OV → OV"),
    ]

    for ax, sae_label, ov_key, conv_key, title in panel_setup:
        for key, name, linestyle, marker in [
            (ov_key,   "single OV→OV",   "-",  "o"),
            (conv_key, "conventional",   "--", "^"),
        ]:
            entry = cfg[key]
            per_alpha = entry.get("per_alpha") or entry.get("results")
            jc = [reduce_mean(per_alpha[str(a)]["jsd_clean"]) for a in alphas]
            jp = [reduce_mean(per_alpha[str(a)]["jsd_pois"])  for a in alphas]
            ax.plot(alphas, jc, color=GREEN, lw=2.3, marker=marker, markersize=7,
                    linestyle=linestyle,
                    label=f"{name} · JSD(steered, clean)  ↓")
            ax.plot(alphas, jp, color=RED, lw=2.3, marker=marker, markersize=7,
                    linestyle=linestyle,
                    label=f"{name} · JSD(steered, poisoned)  ↑")

        ax.axhline(1.0, color="#555", linestyle=":", lw=0.9, alpha=0.6)
        ax.text(alphas[-1], 1.0 - 0.02, "JSD upper bound (1 bit)", fontsize=9,
                color="#555", ha="right", va="top")
        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.04, 1.1)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas], fontsize=9)
        ax.set_xlabel("steering coefficient α", fontsize=11)
        ax.legend(loc="center right", fontsize=9, frameon=True, framealpha=0.95)

    axes[0].set_ylabel("Jensen-Shannon divergence (bits)", fontsize=11)

    if args.title:
        suptitle = args.title
    else:
        suptitle = (
            "JSD vs α — conventional resid-mid additive vs single OV → OV (V-pathway) · "
            "lines = mean over 3 SAE seeds (per-seed re-attributed winners)"
        )
    fig.suptitle(suptitle, fontsize=11.5, y=1.00)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    print(f"\n{'sae':<5} {'method':<14} {'α':>5}  {'jsd(s,clean)':>13}  {'jsd(s,pois)':>12}")
    for ax, sae_label, ov_key, conv_key, _ in panel_setup:
        for key, name in [(ov_key, "OV→OV"), (conv_key, "conventional")]:
            per_alpha = (cfg[key].get("per_alpha") or cfg[key].get("results"))
            for a in alphas:
                jc = reduce_mean(per_alpha[str(a)]["jsd_clean"])
                jp = reduce_mean(per_alpha[str(a)]["jsd_pois"])
                print(f"{sae_label:<5} {name:<14} {a:>5.2f}  {jc:>13.4f}  {jp:>12.4f}")


if __name__ == "__main__":
    main()
