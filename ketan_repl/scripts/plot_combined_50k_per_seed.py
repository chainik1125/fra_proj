"""2x6 per-seed companion to combined_50k.

Six panels per row (one per sae_seed 0..5):
  Top row    — JSD curves per seed (analog of fig 2 left panel, single seed)
  Bottom row — Layman curves per seed (clean-match rate + ASR)

For the layman metrics (bottom row), error bars are the within-seed
binomial 95 % Wilson confidence interval based on N_PROMPTS = 200. JSD
(top row) is plotted as markers + lines without error bars: it is a mean
over 200×16 = 3200 (prompt, position) pairs but the existing sweep didn't
retain per-prompt values, so a standard error from n=3200 cannot be
recomputed without rerunning.

Per-seed feature index labelled in each panel title.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def setup_style():
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          12,
        "axes.titlesize":     12,
        "axes.labelsize":     12,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "axes.edgecolor":     "#222222",
        "axes.labelcolor":    "#1a1a1a",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    9.5,
        "ytick.labelsize":    10,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     True,
        "legend.fontsize":    8.5,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def wilson_ci(k: int, n: int, z: float = 1.96):
    """95 % Wilson confidence interval for a binomial proportion."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2 * n)) / denom
    half   = (z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n))) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    setup_style()
    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    sae_seeds = sorted(int(s) for s in data["sae_seeds"])
    cfg = data["configs"]

    OV_KEY   = "ov_single_50k"
    CONV_KEY = "conventional_50k"

    GREEN = "#1a8a3f"
    RED   = "#c0322a"

    methods = [
        (OV_KEY,   r"single OV$\rightarrow$OV", "-",  "o"),
        (CONV_KEY, "conventional",              "--", "^"),
    ]

    n_cols = len(sae_seeds)
    fig, axes = plt.subplots(2, n_cols, figsize=(3.4 * n_cols, 7.6),
                              sharey="row")

    # --- helper: per-(seed, alpha) accessor ---
    def get_seed(cell, sae_seed, alpha, key):
        # The sweep JSON stores per_alpha[α] as a list indexed by sae_seeds order.
        per_alpha = cfg[cell]["per_alpha"]
        seeds_list = data["sae_seeds"]
        idx = seeds_list.index(sae_seed)
        return per_alpha[str(alpha)][key][idx]

    # ─────────────────── TOP ROW — JSD per seed ───────────────────
    for col, sae_seed in enumerate(sae_seeds):
        ax = axes[0, col]
        for cell_key, name, ls, mk in methods:
            jc = [get_seed(cell_key, sae_seed, a, "jsd_clean") for a in alphas]
            jp = [get_seed(cell_key, sae_seed, a, "jsd_pois")  for a in alphas]
            ax.plot(alphas, jc, color=GREEN, lw=2.0, marker=mk, markersize=5,
                    linestyle=ls, markeredgecolor="white", markeredgewidth=0.7,
                    label=f"{name}  JSD(s, clean)" if col == 0 else None)
            ax.plot(alphas, jp, color=RED, lw=2.0, marker=mk, markersize=5,
                    linestyle=ls, markeredgecolor="white", markeredgewidth=0.7,
                    label=f"{name}  JSD(s, poisoned)" if col == 0 else None)

        ax.axhline(1.0, color="#888888", linestyle=":", lw=0.7, alpha=0.6)
        f_ov   = cfg[OV_KEY]["per_seed_feature"][str(sae_seed)]
        f_conv = cfg[CONV_KEY]["per_seed_feature"][str(sae_seed)]
        ax.set_title(f"sae_seed = {sae_seed}\nOV f={f_ov} · conv f={f_conv}",
                      fontsize=10, fontweight="bold")
        ax.set_ylim(-0.05, 1.10)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas])
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        if col == 0:
            ax.set_ylabel("JSD (bits)")
            ax.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb",
                      fontsize=8)

    # ─────────────────── BOTTOM ROW — layman per seed (with Wilson CI) ───────────
    for col, sae_seed in enumerate(sae_seeds):
        ax = axes[1, col]
        for cell_key, name, ls, mk in methods:
            n_match = [get_seed(cell_key, sae_seed, a, "n_exact_match_clean")
                        for a in alphas]
            asr_per_a = [get_seed(cell_key, sae_seed, a, "asr") for a in alphas]
            asr_k = [int(round(p * args.n_prompts)) for p in asr_per_a]

            mr_p, mr_lo, mr_hi = zip(*[wilson_ci(k, args.n_prompts) for k in n_match])
            ar_p, ar_lo, ar_hi = zip(*[wilson_ci(k, args.n_prompts) for k in asr_k])

            mr_err = [[p - lo for p, lo in zip(mr_p, mr_lo)],
                      [hi - p for p, hi in zip(mr_p, mr_hi)]]
            ar_err = [[p - lo for p, lo in zip(ar_p, ar_lo)],
                      [hi - p for p, hi in zip(ar_p, ar_hi)]]

            ax.errorbar(alphas, mr_p, yerr=mr_err,
                         color=GREEN, lw=2.0, marker=mk, markersize=5,
                         linestyle=ls, markeredgecolor="white", markeredgewidth=0.7,
                         capsize=2.5, capthick=1.0, elinewidth=1.0,
                         label=f"{name}  clean-match" if col == 0 else None)
            ax.errorbar(alphas, ar_p, yerr=ar_err,
                         color=RED, lw=2.0, marker=mk, markersize=5,
                         linestyle=ls, markeredgecolor="white", markeredgewidth=0.7,
                         capsize=2.5, capthick=1.0, elinewidth=1.0,
                         label=f"{name}  ASR" if col == 0 else None)

        ax.set_ylim(-0.03, 1.05)
        ax.set_xticks(alphas)
        ax.set_xticklabels([f"{a:.2g}" for a in alphas])
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_xlabel(r"steering coefficient  $\alpha$")
        if col == 0:
            ax.set_ylabel("Sleeper fraction / Word-word matches")
            ax.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb",
                      fontsize=8)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=1.00)
    else:
        fig.suptitle(
            r"Per-seed companion to Fig.~2 — 50k SAE, single OV$\rightarrow$OV "
            r"(solid+circle) vs conventional resid-mid additive (dashed+triangle).  "
            r"Top: JSD.  Bottom: rollout-level (clean-match rate, ASR).  "
            r"Bottom-row error bars = 95 \% Wilson CI on the proportion (n = 200 prompts).",
            fontsize=11, y=1.00,
        )

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out)
    if base.endswith(".png") or base.endswith(".pdf"):
        base = base.rsplit(".", 1)[0]
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
