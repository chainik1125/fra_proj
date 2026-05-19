#!/usr/bin/env python3
"""1×3 alignment-vs-coherence frontier figure for Qwen-2.5-7B EM medical.

Layout (cols left → right): DoM | conventional SAE | QK→QK.
Each cell shows the α-sweep trajectory in (coherence, alignment) space at
seed 42, α-values labelled, black star at the unsteered reference (α=0
for DoM, α=1 for additive/QK→QK following our 14B convention).

A stats box reports peak / baseline / min@coh70 / Δ@coh70 per method,
exactly as in `phase1_2x3_seed42_neg6`.

Inputs:
    --combined-root  dir with three `gpt4o_combined_*_medical.json`:
        - dom_qwen7b_extract_em_apply_medical          (Stream B output)
        - L15_ln1_arditi_qwen7b                        (Stream C additive)
        - L15_ln1_arditi_qwen7b_FRA                    (Stream C QK→QK)
    --out            basename for png+pdf
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


COH_FLOOR = 70.0
SEED_INDEX = 0  # combined files store per-seed lists; index 0 = seed 42

METHODS = [
    # (key in gpt4o_combined, method label, panel title, marker color, baseline α)
    ("dom_L15",           "DoM (Soligo)",            "DoM @ L15",                   "#5E2E8A", 0.0),
    ("sae_resid",         "conventional SAE (add)",  "Conventional SAE add. @ L15", "#000000", 1.0),
    ("qk_to_qk",          r"QK$\rightarrow$QK (FRA)", "QK→QK @ L15 H?",              "#009E73", 1.0),
]

# Where to find each method's combined JSON in the root dir.
METHOD_FILES = {
    "dom_L15":   "gpt4o_combined_dom_qwen7b_extract_em_apply_medical_medical.json",
    "sae_resid": "gpt4o_combined_L15_ln1_arditi_qwen7b_medical.json",
    "qk_to_qk":  "gpt4o_combined_L15_ln1_arditi_qwen7b_FRA_medical.json",
}


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           14,
        "axes.titlesize":      15,
        "axes.labelsize":      13,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.1,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.12,
        "figure.dpi":          110,
    })


def trajectory(method_block):
    """Pull (scales, align, coh) at SEED_INDEX from a combined block."""
    by_alpha = method_block["by_alpha"]
    scales, al, co = [], [], []
    for e in by_alpha:
        if SEED_INDEX < len(e["per_seed_alignment"]):
            a = e["per_seed_alignment"][SEED_INDEX]
            c = e["per_seed_coherence"][SEED_INDEX]
            if a is None or c is None:
                continue
            scales.append(e["scale"]); al.append(a); co.append(c)
    order = np.argsort(scales)
    return np.array(scales)[order], np.array(al)[order], np.array(co)[order]


def find_method_block(combined_root: Path, file_key: str, method_key: str):
    """Open the right combined JSON and return the method block.

    For DoM, the combined file may have multiple 'method' keys per layer
    (e.g., `dom_L14`, `dom_L15`, `dom_L16`); we pick `dom_L15` by default.
    """
    path = combined_root / METHOD_FILES[file_key]
    if not path.exists():
        print(f"WARN: {path} not found")
        return None
    d = json.loads(path.read_text())
    return d.get(method_key)


def plot_cell(ax, scales, al, co, title, color, baseline_scale):
    ax.set_title(title)
    if len(scales) == 0:
        ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes,
                ha="center", va="center", color="#aa0000")
        ax.set_xlim(0, 105); ax.set_ylim(0, 105)
        return
    ax.plot(co, al, "-", lw=1.6, color=color, alpha=0.85, zorder=2)
    ax.scatter(co, al, s=28, color=color, zorder=3)
    for x, y, s in zip(co, al, scales):
        ax.annotate(f"{s:+.1f}", (x, y), xytext=(4, 4), textcoords="offset points",
                    fontsize=8, color="#444", zorder=4)
    if (scales == baseline_scale).any():
        zx = co[scales == baseline_scale][0]
        zy = al[scales == baseline_scale][0]
        ax.scatter([zx], [zy], marker="*", s=140, color="#000", zorder=5)
    ax.axvline(COH_FLOOR, color="#cccccc", lw=0.7, ls="--", zorder=1)
    # stats box
    mask = co >= COH_FLOOR
    if mask.any():
        peak = float(al.max())
        mn = float(al[mask].min())
        mx = float(al[mask].max())
        delta = mx - mn
        base = float(al[scales == baseline_scale][0]) if (scales == baseline_scale).any() else None
        lines = [
            f"peak      = {peak:.1f}",
            f"baseline  = {base:.1f}" if base is not None else "baseline  = —",
            f"min@coh70 = {mn:.1f}",
            f"Δ@coh70   = {delta:.1f}",
        ]
        txt = "\n".join(lines)
    else:
        txt = "no α with coh ≥ 70"
    ax.text(0.02, 0.98, txt, transform=ax.transAxes, fontsize=9,
            va="top", ha="left", family="monospace",
            bbox=dict(facecolor="white", edgecolor="#cccccc",
                      boxstyle="round,pad=0.3", lw=0.6))
    ax.set_xlim(0, 105); ax.set_ylim(0, 105)
    ax.set_xlabel("Coherence")
    ax.set_ylabel("Alignment")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True,
                   help="Dir containing gpt4o_combined_*_medical.json files")
    p.add_argument("--out", required=True,
                   help="Output basename for png+pdf")
    p.add_argument("--qk-head", default="?",
                   help="Head index for the QK→QK title (just cosmetic)")
    p.add_argument("--dom-layer", type=int, default=15,
                   help="Which DoM layer block to plot (default L15)")
    args = p.parse_args()
    setup_style()

    combined_root = Path(args.combined_root)
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 5.0))

    for ax, (file_key, label, title, color, baseline) in zip(axes, METHODS):
        # Resolve method_key per file
        if file_key == "dom_L15":
            method_key = f"dom_L{args.dom_layer}"
            title = title.replace("L15", f"L{args.dom_layer}")
        elif file_key == "qk_to_qk":
            method_key = "qk_to_qk"
            title = title.replace("H?", f"H{args.qk_head}")
        else:
            method_key = "sae_resid"
        blk = find_method_block(combined_root, file_key, method_key)
        if blk is None:
            ax.set_title(title + " (missing)")
            ax.text(0.5, 0.5, "(no data)", transform=ax.transAxes, ha="center", va="center", color="#aa0000")
            continue
        scales, al, co = trajectory(blk)
        plot_cell(ax, scales, al, co, title, color, baseline)

    fig.suptitle("Qwen-2.5-7B + bad-medical · 3-method comparison @ seed 42",
                 fontsize=14, y=1.02)
    fig.text(0.5, -0.02,
             "α labelled at each point; black ★ = unsteered (DoM α=0; SAE/QK→QK α=1).  Stats box uses coh ≥ 70 floor.",
             ha="center", va="top", fontsize=10, color="#666666")

    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
