#!/usr/bin/env python3
"""Arditi single-feature seed × feature grid (one figure per domain).

Rows = eval seeds {42, 123, 456}
Cols = features being steered
Each cell = α-sweep alignment-vs-coherence trajectory for THAT (seed, feature).

α annotated at each point; black star at α=0 (unsteered); stats box with
peak / min@coh70 / Δ (max−min over coh≥70).

Companion to scripts/plot_phase1_seed_grid.py but for Arditi-style single-
feature additive steering at resid_post.

Inputs: gpt4o_combined_<sae_id>_<em>.json from phase1_judge_and_combine.

    python scripts/plot_arditi_seed_grid.py \
        --combined <path-to-json>  --out <basename>
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COH_FLOOR = 70.0
SEEDS = [42, 123, 456]
ARDITI_COLOR = "#0072B2"


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.0,
        "axes.edgecolor":      "#222222",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def trajectory_for(method_block, seed_idx):
    """Pull (scales, alignment, coherence) for one seed across all α."""
    by_alpha = method_block["by_alpha"]
    scales, al, co = [], [], []
    for e in by_alpha:
        if seed_idx < len(e["per_seed_alignment"]):
            a = e["per_seed_alignment"][seed_idx]
            c = e["per_seed_coherence"][seed_idx]
            if a is None or c is None:
                continue
            scales.append(e["scale"])
            al.append(a)
            co.append(c)
    order = np.argsort(scales)
    return np.array(scales)[order], np.array(al)[order], np.array(co)[order]


def plot_grid(combined, feature_ids, out_path: Path):
    n_rows = len(SEEDS)
    n_cols = len(feature_ids)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(3.4 * n_cols, 3.2 * n_rows),
                             sharex=True, sharey=True)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ci, fid in enumerate(feature_ids):
        method = f"feat_F{fid}"
        if method not in combined:
            for ri in range(n_rows):
                axes[ri, ci].text(0.5, 0.5, f"missing\n{method}",
                                  transform=axes[ri, ci].transAxes,
                                  ha="center", va="center", color="#aa0000")
            continue
        block = combined[method]
        for ri, seed in enumerate(SEEDS):
            ax = axes[ri, ci]
            scales, al, co = trajectory_for(block, ri)
            if len(scales) == 0:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#666")
                continue
            ax.plot(co, al, "-", color=ARDITI_COLOR, alpha=0.85, lw=1.6, zorder=2)
            ax.scatter(co, al, s=22, color=ARDITI_COLOR, zorder=3)
            for x, y, s in zip(co, al, scales):
                ax.annotate(f"{s:+.0f}", (x, y), fontsize=7,
                            xytext=(3, 3), textcoords="offset points",
                            color="#444", zorder=4)
            # Mark α=0 (unsteered)
            if (scales == 0).any():
                zx = co[scales == 0][0]
                zy = al[scales == 0][0]
                ax.scatter([zx], [zy], marker="*", s=120,
                           color="#000", zorder=5)
            ax.axvline(COH_FLOOR, color="#bbb", lw=0.7, ls="--", zorder=1)
            # Stats box: max − min over coh ≥ floor
            mask = co >= COH_FLOOR
            if mask.any():
                amax = float(al[mask].max())
                amin = float(al[mask].min())
                delta = amax - amin
                txt = f"Δ={delta:.1f}\nmax={amax:.1f}\nmin={amin:.1f}"
            else:
                txt = "no α with\ncoh ≥ 70"
            ax.text(0.02, 0.98, txt, transform=ax.transAxes,
                    fontsize=8, va="top", ha="left",
                    bbox=dict(facecolor="white", edgecolor="#ccc",
                              boxstyle="round,pad=0.3", lw=0.6))
            ax.set_xlim(0, 105); ax.set_ylim(0, 105)
            if ri == 0:
                ax.set_title(f"F{fid}", fontsize=11)
            if ri == n_rows - 1:
                ax.set_xlabel("Coherence")
            if ci == 0:
                ax.set_ylabel(f"seed={SEEDS[ri]}\nAlignment", fontsize=10)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".png", dpi=180)
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)
    print(f"plot → {out_path}.png / .pdf")


def plot_distribution(combined, feature_ids, out_path: Path):
    """Box plot of per-feature delta_coh_70 (max − min) — directly comparable
    to the LessWrong post's steering-effect-by-layer box plot."""
    deltas = []
    labels = []
    for fid in feature_ids:
        method = f"feat_F{fid}"
        if method not in combined:
            continue
        d = combined[method]["summary"]["delta_coh_70"]
        if d["mean"] is None:
            continue
        # Distribute across seeds — read per-seed Δ directly from by_alpha.
        per_seed = []
        block = combined[method]
        for seed_idx in range(len(SEEDS)):
            _, al, co = trajectory_for(block, seed_idx)
            if len(al) == 0:
                continue
            mask = co >= COH_FLOOR
            if mask.any():
                per_seed.append(float(al[mask].max() - al[mask].min()))
        if per_seed:
            deltas.append(per_seed)
            labels.append(f"F{fid}")

    if not deltas:
        print("no Δ values to plot"); return

    fig, ax = plt.subplots(figsize=(max(6.0, 0.7 * len(deltas) + 2), 5.5))
    bp = ax.boxplot(deltas, tick_labels=labels, patch_artist=True,
                    showfliers=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("#a6c8e6"); patch.set_edgecolor("#222")
    for whisker in bp["whiskers"]: whisker.set_color("#222")
    for median in bp["medians"]: median.set_color("#0a0a0a")
    # Per-feature scatter for transparency
    for i, vals in enumerate(deltas, start=1):
        ax.scatter([i] * len(vals), vals, color="#0072B2",
                   alpha=0.6, s=22, zorder=3)
    ax.set_ylabel(r"$\Delta$ alignment (max − min over coh ≥ 70)")
    ax.set_title("Arditi single-feature steering effect — Qwen2.5-7B L15 features\n"
                 r"(our judge, our $\alpha$-grid, 3 seeds)",
                 fontsize=11)
    ax.grid(True, axis="y", color="#eee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path) + ".png", dpi=180)
    fig.savefig(str(out_path) + ".pdf")
    plt.close(fig)
    print(f"plot → {out_path}.png / .pdf")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined", required=True,
                   help="Path to gpt4o_combined_<sae_id>_<em>.json")
    p.add_argument("--feature-ids", type=int, nargs="+",
                   default=[94077, 31258, 82558, 59390, 129593,
                            89766, 16069, 42229, 20453, 85078])
    p.add_argument("--out-grid", required=True,
                   help="Output basename for seed × feature grid (.png/.pdf added)")
    p.add_argument("--out-box", required=True,
                   help="Output basename for per-feature box plot (.png/.pdf added)")
    args = p.parse_args()
    setup_style()

    combined = json.loads(Path(args.combined).read_text())
    plot_grid(combined, args.feature_ids, Path(args.out_grid))
    plot_distribution(combined, args.feature_ids, Path(args.out_box))


if __name__ == "__main__":
    main()
