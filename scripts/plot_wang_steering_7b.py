#!/usr/bin/env python3
"""Plot Wang-ranker × Arditi-SAE single-feature steering on Qwen-2.5-7B
(L15 resid_post). Outputs:

  - <out>_bars.{png,pdf}: per-feature Δalign|coh≥70, medical EM vs base,
    sorted by medical magnitude. The headline figure for the campaign.
  - <out>_traj.{png,pdf}: alignment-vs-α trajectory grid for top-K
    features (K=5 by default), one column per feature, two rows
    (medical, base). Lets you see WHERE in α the bar's Δ comes from.

Inputs: the two combined JSONs produced by phase1_judge_and_combine.py,
one for em=medical and one for em=base.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


COH_FLOOR = 70.0
COLOR_MED = "#C84D4D"   # red — medical EM
COLOR_BASE = "#3D6AC4"  # blue — base control


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          12,
        "axes.titlesize":     14,
        "axes.labelsize":     12,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.0,
        "axes.edgecolor":     "#222",
        "xtick.labelsize":    10,
        "ytick.labelsize":    10,
        "legend.frameon":     False,
        "legend.fontsize":    11,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def feat_id(key: str) -> int:
    return int(key.removeprefix("feat_F"))


def delta_coh70(block: dict) -> tuple[float | None, float | None, int]:
    s = block.get("summary", {}).get("delta_coh_70", {})
    m, sd, n = s.get("mean"), s.get("std"), s.get("n") or 0
    return m, sd, n


def plot_bars(medical: dict, base: dict, ranker_top: list[int], out: Path):
    """Sorted bar chart per feature, medical vs base."""
    feats = ranker_top or sorted(set(medical) | set(base), key=feat_id)
    rows = []
    for k in feats:
        fid = feat_id(k)
        m_mean, m_std, m_n = delta_coh70(medical.get(k, {}))
        b_mean, b_std, b_n = delta_coh70(base.get(k, {}))
        rows.append((fid, m_mean or 0, m_std or 0, m_n, b_mean or 0, b_std or 0, b_n))
    # Sort by medical Δcoh70 desc
    rows.sort(key=lambda r: -r[1])

    fig, ax = plt.subplots(figsize=(max(12, 0.28 * len(rows)), 6.0))
    x = np.arange(len(rows))
    w = 0.42
    m_means = [r[1] for r in rows]
    m_stds  = [r[2] for r in rows]
    b_means = [r[4] for r in rows]
    b_stds  = [r[5] for r in rows]
    ax.bar(x - w/2, m_means, width=w, yerr=m_stds,
           color=COLOR_MED, edgecolor="#222", linewidth=0.7,
           error_kw=dict(ecolor="#222", capsize=2, capthick=0.8, lw=0.8),
           label=f"EM-medical (n={rows[0][3]} seeds)", zorder=3)
    ax.bar(x + w/2, b_means, width=w, yerr=b_stds,
           color=COLOR_BASE, edgecolor="#222", linewidth=0.7,
           error_kw=dict(ecolor="#222", capsize=2, capthick=0.8, lw=0.8),
           label=f"base (n={rows[0][6]} seeds)", zorder=3)

    # Annotate top-5 medical
    for i in range(min(5, len(rows))):
        ax.text(i - w/2, rows[i][1] + (rows[i][2] or 0) + 0.6,
                f"F{rows[i][0]}\n{rows[i][1]:.1f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold",
                color="#0a0a0a", zorder=4)

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{r[0]}" for r in rows], rotation=70,
                       ha="right", fontsize=8)
    ax.set_ylabel(r"$\Delta$ alignment @ coh $\geq$ 70")
    ax.set_title("Wang-ranker top-50 features × single-feature additive steering\n"
                 "Qwen-2.5-7B + andyrdt L15 resid_post SAE  ·  17-pt α-grid {−2..+2}")
    ax.axhline(0, color="#666", lw=0.6, zorder=1)
    ax.grid(True, axis="y", color="#eee", lw=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.legend(loc="upper right", title="model under hook")

    fig.text(0.5, -0.02,
             "features sorted by EM-medical Δalign; bars are mean ± std across seeds; "
             "Δ defined as max−min over α with coherence ≥ 70",
             ha="center", va="top", fontsize=10, color="#555")

    fig.savefig(f"{out}_bars.png", dpi=200)
    fig.savefig(f"{out}_bars.pdf")
    plt.close(fig)
    print(f"  → {out}_bars.png / .pdf  ({len(rows)} features)")
    return rows


def trajectory(block: dict):
    """Return (alphas, mean_align, mean_coh, std_align)."""
    by = block["by_alpha"]
    a   = np.array([e["scale"] for e in by])
    al  = np.array([e["mean_alignment_across_seeds"] for e in by], dtype=float)
    als = np.array([e["std_alignment_across_seeds"] for e in by], dtype=float)
    co  = np.array([e["mean_coherence_across_seeds"] for e in by], dtype=float)
    idx = np.argsort(a)
    return a[idx], al[idx], als[idx], co[idx]


def plot_trajectory(medical: dict, base: dict, top_rows: list, out: Path, k=5):
    """For top-K features (by medical Δcoh70), plot alignment+coherence vs α
    overlay (medical vs base)."""
    feats = top_rows[:k]
    fig, axes = plt.subplots(2, k, figsize=(3.6 * k, 6.0), sharex=True)
    if k == 1: axes = axes.reshape(2, 1)

    for i, r in enumerate(feats):
        fid = r[0]
        key = f"feat_F{fid}"
        for row_idx, (em_label, em_data, color) in enumerate(
            [("EM-medical", medical, COLOR_MED), ("base", base, COLOR_BASE)]
        ):
            ax = axes[row_idx, i]
            if key not in em_data:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center")
                continue
            a, al, als, co = trajectory(em_data[key])
            ax.fill_between(a, al - als, al + als, color=color, alpha=0.18)
            ax.plot(a, al, "-o", color=color, lw=1.8, ms=4, label="alignment")
            ax.plot(a, co, "--", color="#555", lw=1.2, label="coherence")
            ax.axhline(COH_FLOOR, color="#c0392b", lw=0.8, ls=":", zorder=1)
            ax.set_ylim(-2, 105)
            ax.grid(True, color="#eee", lw=0.4)
            ax.set_axisbelow(True)
            if row_idx == 0:
                d = r[1]
                ax.set_title(f"F{fid}   Δmed={d:.1f}", fontsize=11)
            if row_idx == 1:
                ax.set_xlabel(r"$\alpha$")
            if i == 0:
                ax.set_ylabel(f"{em_label}\nscore (0–100)")
            if i == k - 1 and row_idx == 0:
                ax.legend(loc="lower left", fontsize=8)
    fig.suptitle(f"Top-{k} Wang features by EM-medical Δcoh70 — α-sweep per model")
    fig.tight_layout()
    fig.savefig(f"{out}_traj.png", dpi=200)
    fig.savefig(f"{out}_traj.pdf")
    plt.close(fig)
    print(f"  → {out}_traj.png / .pdf  (top-{k})")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--medical", required=True, help="combined JSON for em=medical")
    p.add_argument("--base",    required=True, help="combined JSON for em=base")
    p.add_argument("--ranker",  default=None,
                   help="optional wang_ranker JSON; if given, restricts to its top-N")
    p.add_argument("--out",     required=True, help="output prefix")
    p.add_argument("--top-k-traj", type=int, default=5)
    args = p.parse_args()
    setup_style()

    medical = json.loads(Path(args.medical).read_text())
    base    = json.loads(Path(args.base).read_text())
    ranker_feats = None
    if args.ranker:
        r = json.loads(Path(args.ranker).read_text())
        ranker_feats = [f"feat_F{fid}" for fid in r["feature_ids"]]

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)

    rows = plot_bars(medical, base, ranker_feats, out)
    plot_trajectory(medical, base, rows, out, k=args.top_k_traj)


if __name__ == "__main__":
    main()
