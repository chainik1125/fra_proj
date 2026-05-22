#!/usr/bin/env python3
"""Bar chart for the Qwen-14B base + diff-ranked Conv-SAE constant-additive campaign.

For each (domain × hookpoint) cell we read
`gpt4o_combined_<sae>_<domain>.json` from `--combined-root` and plot
`sae_resid.summary.delta_coh_70.mean ± std`.

Also writes a side panel: α-sweep alignment curve for the best cell in each
domain (the (hookpoint, seed-mean) with the largest Δcoh70) so the reader
can see *why* the bar is what it is.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


SAES = ["L24_ln1_nura", "L24_resid_mid", "L24_resid_post"]
HP_LABEL = {
    "L24_ln1_nura":   "L24 ln1\n(Nura)",
    "L24_resid_mid":  "L24 resid_mid",
    "L24_resid_post": "L24 resid_post",
}
COLOR_BY_HP = {
    "L24_ln1_nura":   "#0072B2",  # blue
    "L24_resid_mid":  "#009E73",  # green
    "L24_resid_post": "#D55E00",  # orange
}


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           14,
        "axes.titlesize":      16,
        "axes.labelsize":      14,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.1,
        "xtick.labelsize":     12,
        "ytick.labelsize":     12,
        "legend.frameon":      True,
        "legend.fontsize":     12,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def load_cell(root: Path, sae: str, domain: str):
    p = root / f"gpt4o_combined_{sae}_{domain}.json"
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    return d.get("sae_resid", {})


def plot_bars(ax, root: Path, domains):
    n_dom = len(domains)
    n_sae = len(SAES)
    width = 0.26
    x = np.arange(n_dom)
    for i, sae in enumerate(SAES):
        means, stds = [], []
        for dom in domains:
            cell = load_cell(root, sae, dom)
            if cell is None:
                means.append(0.0); stds.append(0.0); continue
            s = cell.get("summary", {}).get("delta_coh_70") or {}
            means.append(float(s.get("mean") or 0.0))
            stds.append(float(s.get("std") or 0.0))
        offset = (i - 1) * width
        bars = ax.bar(x + offset, means, width=width, yerr=stds,
                      color=COLOR_BY_HP[sae], edgecolor="#222", linewidth=1.0,
                      error_kw=dict(ecolor="#222", capsize=4, capthick=1.2, lw=1.2),
                      label=HP_LABEL[sae].replace("\n", " "), zorder=3)
        for b, m in zip(bars, means):
            ax.text(b.get_x() + b.get_width()/2, b.get_height() + 0.5,
                    f"{m:.1f}", ha="center", va="bottom",
                    fontsize=10, color="#0a0a0a", zorder=5)
    ax.set_xticks(x)
    ax.set_xticklabels([d.capitalize() for d in domains], fontweight="600")
    ax.set_ylabel(r"$\Delta$ alignment @ coh $\geq$ 70")
    ax.set_title("Qwen-14B base · diff-ranked Conv-SAE features · constant additive")
    ax.grid(True, axis="y", color="#eee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", title="hookpoint")


def best_cell(root: Path, domain: str):
    best = None
    for sae in SAES:
        cell = load_cell(root, sae, domain)
        if cell is None: continue
        s = cell.get("summary", {}).get("delta_coh_70") or {}
        mean = s.get("mean") or 0.0
        if best is None or mean > best[1]:
            best = (sae, mean, cell)
    return best


def plot_sweep(ax, cell, sae: str, domain: str):
    by = cell.get("by_alpha", [])
    if not by: return
    alphas = np.array([row["scale"] for row in by])
    align  = np.array([row["mean_alignment_across_seeds"] for row in by])
    coh    = np.array([row["mean_coherence_across_seeds"] for row in by])
    align_std = np.array([row["std_alignment_across_seeds"] for row in by])

    ax.fill_between(alphas, align - align_std, align + align_std,
                    color=COLOR_BY_HP[sae], alpha=0.18, zorder=2)
    ax.plot(alphas, align, color=COLOR_BY_HP[sae], lw=2.0, label="alignment", zorder=3)
    ax.plot(alphas, coh,   color="#666", lw=1.4, ls="--", label="coherence", zorder=3)
    ax.axhline(70, color="#c0392b", lw=1.0, ls=":", zorder=1)
    ax.set_xlabel(r"$\alpha$")
    ax.set_ylabel("score (0-100)")
    ax.set_title(f"{domain} · {sae}")
    ax.set_ylim(-2, 102)
    ax.grid(True, color="#eee", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="lower left", fontsize=10)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--combined-root", required=True)
    ap.add_argument("--out", required=True, help="path prefix; writes .png and .pdf")
    ap.add_argument("--domains", nargs="+", default=["medical", "finance", "sports"])
    args = ap.parse_args()
    setup_style()

    root = Path(args.combined_root)
    fig = plt.figure(figsize=(13.5, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.4, 1.0], hspace=0.42, wspace=0.28)

    ax_bars = fig.add_subplot(gs[0, :])
    plot_bars(ax_bars, root, args.domains)

    for i, dom in enumerate(args.domains):
        ax = fig.add_subplot(gs[1, i])
        b = best_cell(root, dom)
        if b is None: continue
        sae, mean, cell = b
        plot_sweep(ax, cell, sae, dom)

    fig.text(0.5, 0.005,
             "top: $\\Delta$alignment over the safe-coherence window per (domain, hookpoint), "
             "mean ± std across seeds {42, 123, 456}\n"
             "bottom: best-hookpoint $\\alpha$-sweep per domain (shaded = $\\pm 1\\sigma$, "
             "dashed grey = coherence, red dotted = coh=70 floor)",
             ha="center", va="bottom", fontsize=11, color="#555")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(f"{out}.{ext}", dpi=200)
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
