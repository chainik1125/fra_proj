"""Layman analog of plot_jsd_overlay.py.

Replaces the two distribution-level JSD measurements with two intuitive
rollout-level counts:

  green = fraction of 200 prompts whose 16-token steered rollout matches the
          clean rollout word-for-word (analog of JSD(steered, clean) — high
          = better, steered preserves clean behavior)
  red   = ASR-16: fraction of prompts whose steered rollout contains the
          sleeper phrase regex (analog of JSD(steered, poisoned) — low =
          better, sleeper suppressed)

Same layout / palette / linestyle conventions as plot_jsd_overlay.py:
  green  · clean-related semantics
  red    · sleeper-related semantics
  solid + ●   single OV → OV
  dashed + ▲  conventional resid-mid additive

Layout: left = 4k SAE, right = 50k SAE. Means over sae_seeds; no error bars.
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
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6), sharey=True)

    GREEN = "#2ca02c"   # word-for-word match — higher = better (closer to clean)
    RED   = "#d62728"   # ASR — lower = better (sleeper suppressed)

    panel_setup = [
        (axes[0], "ov_single_4k",  "conventional_4k",
         "(a)  4k SAE — conventional vs single OV → OV"),
        (axes[1], "ov_single_50k", "conventional_50k",
         "(b)  50k SAE — conventional vs single OV → OV"),
    ]

    for ax, ov_key, conv_key, title in panel_setup:
        for key, name, linestyle, marker in [
            (ov_key,   "single OV → OV",  "-",  "o"),
            (conv_key, "conventional",    "--", "^"),
        ]:
            entry = cfg[key]
            per_alpha = entry["per_alpha"]
            match_rate = [reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                          for a in alphas]
            asr        = [reduce_mean(per_alpha[str(a)]["asr"]) for a in alphas]

            ax.plot(alphas, match_rate, color=GREEN, lw=2.3, marker=marker, markersize=7,
                    linestyle=linestyle,
                    label=f"{name} · clean-match rate ↑")
            ax.plot(alphas, asr, color=RED, lw=2.3, marker=marker, markersize=7,
                    linestyle=linestyle,
                    label=f"{name} · sleeper rate (ASR) ↓")

        ax.set_title(title, fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas], fontsize=9)
        ax.set_xlabel("steering coefficient α", fontsize=11)
        ax.legend(loc="center right", fontsize=9, frameon=True, framealpha=0.95)

    axes[0].set_ylabel("fraction of 200 deployment prompts", fontsize=11)

    if args.title:
        suptitle = args.title
    else:
        suptitle = (
            "Layman view: rollout-level steering effect vs α  ·  "
            "clean-match rate (green, higher=better) and sleeper rate (red, lower=better)  ·  "
            "mean over 3 SAE seeds (per-seed re-attributed winners)"
        )
    fig.suptitle(suptitle, fontsize=11.5, y=1.00)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.output}")

    # Numeric table
    print(f"\n{'sae':<5} {'method':<14} {'α':>5}  {'clean-match':>11}  {'ASR':>6}")
    for ax, ov_key, conv_key, title in panel_setup:
        sae = title.split()[1]
        for key, name in [(ov_key, "OV→OV"), (conv_key, "conventional")]:
            per_alpha = cfg[key]["per_alpha"]
            for a in alphas:
                m = reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                r = reduce_mean(per_alpha[str(a)]["asr"])
                print(f"{sae:<5} {name:<14} {a:>5.2f}  {m:>11.4f}  {r:>6.3f}")


if __name__ == "__main__":
    main()
