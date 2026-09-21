"""Poster Fig 2: held-out collateral KL at >=99% in-context-backdoor suppression.

One bar per removal method, log scale. Case: king->crown (gpt2-small, IC4) — the
backdoor every method removes at >=99% ASR-suppression (FRA's reach on the other
three caps at 0.84/0.93/0.94, where the matched-80% comparison in summary.md S3c
applies: FRA 0.07 vs DoM 1.83 / conv-SAE 6.06 / payload 4.12, n=4).

Data provenance: ic4_dom_conv.py stdout, recovered from out/b4dktqyhw.output
(pod's ic4.json was never fetched). Per-method (suppression, KL) sweeps,
king->crown case; value = min KL among points with suppression >= 0.99:
    fra  : [(0.77,0.01),(1.0,0.02),(1.0,0.03),(0.99,0.03),(0.98,0.02)] -> 0.02
    dom  : [(0.02,0.19),(1.0,2.81),(1.0,13.98),(1.0,19.07),(1.0,27.57),(1.0,40.4)] -> 2.81
    conv : [(0.15,3.12),(1.0,8.44),(1.0,10.0),(1.0,13.85),(1.0,21.24),(1.0,32.36)] -> 8.44
    pay  : [(0.99,5.97),(1.0,15.2),(1.0,37.41),(1.0,93.96),(1.0,242.58)] -> 5.97

Usage: python3 jobs/plot_fig2_poster_bar.py [--out <path.pdf>]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

METHODS = [
    ("FRA",                   0.02, "#0072B2"),
    ("DoM\nsteering",         2.81, "#D55E00"),
    ("payload\nsuppression",  5.97, "#CC79A7"),
    ("SAE\nsteering",         8.44, "#009E73"),
]


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         13,
        "axes.titlesize":    14,
        "axes.labelsize":    14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.1,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   12.5,
        "ytick.labelsize":   12,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out", type=Path, default=Path("fig2_backdoor_kl_bar.pdf"))
    args = p.parse_args()
    setup_style()

    labels = [m[0] for m in METHODS]
    vals = [m[1] for m in METHODS]
    colors = [m[2] for m in METHODS]

    fig, ax = plt.subplots(figsize=(6.6, 4.8))
    bars = ax.bar(labels, vals, color=colors, width=0.62, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(0.01, 30)
    ax.set_ylabel("collateral damage on held-out text\nKL (nats)   $\\downarrow$ better")

    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v * 1.15, f"{v:.2f}",
                ha="center", va="bottom", fontsize=13, fontweight="bold",
                color="#222222", zorder=4)
    # multiples vs FRA
    for b, v in zip(bars[1:], vals[1:]):
        ax.text(b.get_x() + b.get_width() / 2, v * 0.45, f"$\\times${v/vals[0]:.0f}",
                ha="center", va="top", fontsize=12, color="white",
                fontweight="bold", zorder=4)

    ax.grid(axis="y", which="major", color="#DDDDDD", lw=0.8, zorder=0)
    ax.set_title("Cutting an in-context backdoor at matched removal ($\\geq$99%)",
                 fontsize=14, pad=10)
    ax.text(0.995, -0.22, "gpt2-small, king$\\to$crown backdoor; all methods reach $\\geq$99% ASR-suppression",
            transform=ax.transAxes, ha="right", fontsize=9.5, color="#666666")

    for ext in ("pdf", "png"):
        out = args.out.with_suffix("." + ext)
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
