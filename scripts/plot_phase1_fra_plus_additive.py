#!/usr/bin/env python3
"""1×2 cross-domain bar chart combining FRA recipes (on Nura's L24 ln1 SAE)
with additive recipe on each of 5 pre-trained SAEs.

8 bars per panel (one per (recipe, SAE)):

  FRA on Nura L24 ln1: qk_to_ov, ov_to_ov, qk_to_qk
  Additive:           L24 ln1 (Nura), L24 resid_pre, L24 resid_mid,
                       L24 resid_post, L25 ln1

Bar height = mean Δalign|coh≥70 across 3 eval seeds; error bar = sample std
(ddof=1). Bars labelled with Δ. Where the unsteered (no-hook) baseline
alignment is available, it's drawn as a horizontal dashed line per panel.

Inputs: gpt4o_combined_*.json files produced by phase1_judge_and_combine.py
(the new format with {by_alpha, summary} per method).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


# (column key, label, group, colour)
COLUMNS = [
    ("FRA:qk_to_ov", r"FRA: QK$\rightarrow$OV", "FRA decomposition", "#009E73"),
    ("FRA:ov_to_ov", r"FRA: OV$\rightarrow$OV", "FRA decomposition", "#0072B2"),
    ("FRA:qk_to_qk", r"FRA: QK$\rightarrow$QK", "FRA decomposition", "#D55E00"),
    ("ADD:L24_ln1_nura",   "Add: L24 ln1 (Nura)",  "Conventional additive", "#CC79A7"),
    ("ADD:L24_resid_pre",  "Add: L24 resid_pre",   "Conventional additive", "#CC79A7"),
    ("ADD:L24_resid_mid",  "Add: L24 resid_mid",   "Conventional additive", "#CC79A7"),
    ("ADD:L24_resid_post", "Add: L24 resid_post",  "Conventional additive", "#CC79A7"),
    ("ADD:L25_ln1",        "Add: L25 ln1",         "Conventional additive", "#CC79A7"),
]
GROUP_LABELS = {"FRA decomposition", "Conventional additive"}


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           15,
        "axes.titlesize":      18,
        "axes.labelsize":      16,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.2,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#222222",
        "ytick.color":         "#222222",
        "xtick.labelsize":     11.5,
        "ytick.labelsize":     14,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      True,
        "legend.fontsize":     11.5,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def load_metrics(streams_root: Path):
    """Return {(col_key, em): summary_dict}."""
    out = {}
    for em in ("medical", "finance", "sports"):
        # FRA recipes from gpt4o_combined_L24_ln1_nura_FRA_<em>.json
        fra_path = streams_root / f"gpt4o_combined_L24_ln1_nura_FRA_{em}.json"
        if fra_path.exists():
            d = json.loads(fra_path.read_text())
            for method in ("qk_to_ov", "ov_to_ov", "qk_to_qk"):
                if method in d:
                    out[(f"FRA:{method}", em)] = d[method]["summary"]
            # Also pull the no-hook baseline for the horizontal dashed line
            if "baseline" in d:
                e = d["baseline"]["by_alpha"][0]
                out[("BASELINE_NOHOOK", em)] = {
                    "alignment": e["mean_alignment_across_seeds"],
                    "alignment_std": e["std_alignment_across_seeds"],
                    "coherence": e["mean_coherence_across_seeds"],
                }
        # Additive on each SAE
        for sae in ("L24_ln1_nura", "L24_resid_pre", "L24_resid_mid",
                    "L24_resid_post", "L25_ln1"):
            p = streams_root / f"gpt4o_combined_{sae}_{em}.json"
            if p.exists():
                d = json.loads(p.read_text())
                if "sae_resid" in d:
                    out[(f"ADD:{sae}", em)] = d["sae_resid"]["summary"]
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_style()

    metrics = load_metrics(Path(args.combined_root))
    print("loaded keys:", sorted(metrics.keys()))

    domains = [d for d in ("medical", "finance", "sports")
               if any(em == d for _, em in metrics.keys())]
    fig, axs = plt.subplots(1, len(domains), figsize=(7.5 * len(domains), 6.8), sharey=True)
    if len(domains) == 1:
        axs = [axs]

    # joint y-lim
    all_max = []
    for (k, em), s in metrics.items():
        if k == "BASELINE_NOHOOK": continue
        d = s["delta_coh_70"]
        if d["mean"] is not None:
            all_max.append(d["mean"] + (d["std"] or 0))
    y_top = max(all_max + [10.0]) * 1.4
    y_top = max(int(np.ceil(y_top / 5) * 5), 25)

    for ax, em in zip(axs, domains):
        n_main = len(COLUMNS)
        x = np.arange(n_main + 1)  # +1 for the trailing "Best" column
        means, stds, ns = [], [], []
        for col_key, label, group, _color in COLUMNS:
            s = metrics.get((col_key, em))
            if s is None:
                means.append(0.0); stds.append(0.0); ns.append(0)
            else:
                d = s["delta_coh_70"]
                means.append(d["mean"] if d["mean"] is not None else 0.0)
                stds.append(d["std"] if d["std"] is not None else 0.0)
                ns.append(d["n"])
        # Best column: argmax over the 8 main bars (only count cells with n>0)
        valid_idxs = [i for i in range(n_main) if ns[i] > 0]
        if valid_idxs:
            best_i = max(valid_idxs, key=lambda i: means[i])
            best_label = COLUMNS[best_i][1]
            best_color = "#222222"  # dark grey for the "best" bar (neutral)
            best_mean = means[best_i]
            best_std = stds[best_i]
            best_n = ns[best_i]
        else:
            best_i = None; best_label = ""; best_color = "#666666"
            best_mean = 0.0; best_std = 0.0; best_n = 0

        all_means = means + [best_mean]
        all_stds  = stds  + [best_std]
        all_ns    = ns    + [best_n]
        all_colors = [c for _, _, _, c in COLUMNS] + [best_color]
        edge_widths = [0.8] * n_main + [2.0]   # thicker edge on the Best bar
        edge_colors = ["#222222"] * n_main + ["#000000"]

        bars = ax.bar(x, all_means, yerr=all_stds,
                      color=all_colors,
                      edgecolor=edge_colors, linewidth=edge_widths,
                      error_kw=dict(ecolor="#222222", capsize=5, capthick=1.2, lw=1.4),
                      zorder=3)
        for i, (bar, m, s, n) in enumerate(zip(bars, all_means, all_stds, all_ns)):
            label_str = f"{m:.1f}" if n > 0 else "n=0"
            color = "#0a0a0a" if i < n_main else "#000000"
            weight = "600" if i < n_main else "bold"
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (s or 0) + 0.6,
                    label_str, ha="center", va="bottom",
                    fontsize=11.5 if i == n_main else 11,
                    fontweight=weight, color=color, zorder=5)

        # Annotate the Best bar with the winning recipe name (small, inside the bar)
        if best_i is not None and best_mean > 0:
            ax.text(bars[-1].get_x() + bars[-1].get_width()/2, best_mean / 2,
                    best_label,
                    ha="center", va="center", fontsize=10.5,
                    color="white", fontweight="bold", zorder=6, rotation=90)

        x_labels = [lbl for _, lbl, _, _ in COLUMNS] + ["Best"]
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, rotation=22, ha="right")
        ax.set_title(em.capitalize(), loc="center", fontsize=18, fontweight="bold")
        ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
        ax.set_axisbelow(True)
        ax.set_ylim(0, y_top)
        # separators: between FRA / Additive, and between Additive / Best
        n_fra = sum(1 for c, _, _, _ in COLUMNS if c.startswith("FRA:"))
        sep_fra = n_fra - 0.5
        sep_best = n_main - 0.5
        ax.axvline(sep_fra,  color="#bbbbbb", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.axvline(sep_best, color="#bbbbbb", lw=0.9, ls=(0, (3, 3)), zorder=1)
        ax.text(sep_fra - 0.05, y_top * 0.95, "FRA decomposition",
                ha="right", va="top", fontsize=11.5, color="#555555", fontweight="600")
        ax.text(sep_fra + 0.05, y_top * 0.95, "Conventional additive",
                ha="left", va="top", fontsize=11.5, color="#555555", fontweight="600")
        # baseline (no-hook) horizontal-ish reference: print it as text in upper-left
        b = metrics.get(("BASELINE_NOHOOK", em))
        if b is not None and b["alignment"] is not None:
            ax.text(0.02, 0.02,
                    f"Unsteered baseline: align = {b['alignment']:.1f} ± {b['alignment_std']:.2f}, coh = {b['coherence']:.1f}",
                    transform=ax.transAxes, ha="left", va="bottom",
                    fontsize=10.5, color="#444",
                    bbox=dict(facecolor="white", edgecolor="#222", boxstyle="round,pad=0.4"))

    axs[0].set_ylabel(r"Alignment $\Delta$ @ coh 70")

    fig.text(0.5, -0.03,
             r"error bars = sample std across 3 eval seeds; bar height = mean per-seed $\Delta$alignment over $\alpha$ where coh $\geq$ 70.  "
             r"FRA recipes use Nura's L24 ln1 SAE; conventional additive uses the same SAE plus 4 surrounding-hookpoint SAEs.",
             ha="center", va="top", fontsize=11, color="#666666")

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
