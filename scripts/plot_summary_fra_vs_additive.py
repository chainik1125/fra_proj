#!/usr/bin/env python3
"""Phase 3 summary figure (paper-quality 1×2 frontier comparison).

Same SAE, same hookpoint, same prompts, single eval seed. Only the
intervention recipe differs.

Design choices (first-principles paper-style):

  - Wong colourblind-safe palette per method.
  - Trajectory = subtle line + filled circles at each α.
  - α value annotated outside each marker (small, grey).
  - Unsteered baseline = filled black star with white ring (one per panel).
  - coh = 70 floor drawn explicitly with a labelled vertical guide.
  - Δalign|coh≥70 surfaced as an in-axes range bracket (vertical bar
    with end-caps) on the right side of each method's curve, exactly at
    the alignment max/min of points with coh ≥ 70 — geometric, not
    textual. Per-method Δ printed in the legend so the reader can rank.
  - No floating stat box; legend carries the headline numbers.
  - Top/right spines off, axis tick density modest, light gridlines.
  - Sans-serif typography (Inter when available, Helvetica fallback).

Usage:
    python scripts/plot_summary_fra_vs_additive.py \
        --nura-per-seed-dir <dir>      # has aggregated_seed{seed}_medical.json
        --nura-additive    <gpt4o_aggregated_seed{seed}.json>
        --seed 42
        --out  <basename>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt


COH_FLOOR = 70.0


# ─── Style ─────────────────────────────────────────────────────────────────
def setup_style():
    mpl.rcParams.update({
        "font.family":         "sans-serif",
        "font.sans-serif":     ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           13,
        "axes.titlesize":      15,
        "axes.labelsize":      13.5,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.0,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#333333",
        "ytick.color":         "#333333",
        "xtick.labelsize":     11.5,
        "ytick.labelsize":     11.5,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      False,
        "legend.fontsize":     11,
        "legend.title_fontsize": 11.5,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.08,
    })


# Wong palette assignments — colourblind-safe, distinct
PAL = {
    r"QK$\rightarrow$QK":    "#009E73",   # bluish green (the winner — easiest to distinguish)
    r"OV$\rightarrow$OV":    "#0072B2",   # blue
    r"QK$\rightarrow$OV":    "#D55E00",   # vermilion
    "additive": "#CC79A7",   # reddish purple
}


# ─── Data IO ───────────────────────────────────────────────────────────────
def load_method(path, method=None):
    d = json.loads(Path(path).read_text())
    if "aggregated" in d and isinstance(d["aggregated"], dict):
        d = d["aggregated"]
    if method:
        return sorted(d.get(method, []), key=lambda r: r["scale"])
    rows = []
    for m, r in d.items():
        if isinstance(r, list):
            rows.extend(r)
    return sorted(rows, key=lambda r: r["scale"])


def stats(rows, floor=COH_FLOOR):
    if not rows:
        return None
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    sc = np.array([r["scale"] for r in rows], dtype=float)
    mask = co >= floor
    if mask.any():
        al_m = al[mask]; co_m = co[mask]; sc_m = sc[mask]
        return dict(
            delta=float(al_m.max() - al_m.min()),
            min_align=float(al_m.min()),
            max_align=float(al_m.max()),
            n70=int(mask.sum()),
            x_at_max=float(co_m[al_m.argmax()]),
            x_at_min=float(co_m[al_m.argmin()]),
        )
    return dict(delta=float("nan"), min_align=float("nan"), max_align=float("nan"),
                n70=0, x_at_max=float("nan"), x_at_min=float("nan"))


# ─── Drawing primitives ────────────────────────────────────────────────────
def draw_method(ax, rows, color, label, *, marker="o", show_alpha=True,
                z_off=0):
    if not rows:
        return None
    sc = np.array([r["scale"] for r in rows])
    al = np.array([r["mean_alignment"] for r in rows])
    co = np.array([r["mean_coherence"] for r in rows])
    # subtle line trajectory
    ax.plot(co, al, color=color, lw=1.0, alpha=0.55, zorder=2 + z_off)
    # filled markers
    h = ax.scatter(co, al, marker=marker, s=85, facecolor=color,
                   edgecolor="white", linewidth=1.1, zorder=4 + z_off,
                   label=label)
    # α labels
    if show_alpha:
        for s, x, y in zip(sc, al, co):
            pass  # ordering: we want α near each point but offset cleanly
        # offset alphas radially outward from curve centroid
        cx, cy = co.mean(), al.mean()
        for sval, x, y in zip(sc, co, al):
            dx, dy = x - cx, y - cy
            r = (dx * dx + dy * dy) ** 0.5 + 1e-6
            ox, oy = 6 * dx / r, 6 * dy / r
            ax.annotate(f"{sval:g}",
                        xy=(x, y), xytext=(ox, oy),
                        textcoords="offset points",
                        fontsize=7.5, color="#555555",
                        ha="center", va="center", zorder=5 + z_off)
    return h


def draw_baseline_star(ax, x, y, label="unsteered"):
    ax.scatter([x], [y], marker="*", s=320, color="black",
               edgecolor="white", linewidth=1.3, zorder=7,
               label=label)


def draw_delta_bracket(ax, *, x, y_lo, y_hi, color, dx=2.5, lw=1.4):
    """Draw a vertical bracket on the RIGHT of x, spanning y_lo→y_hi.

    Marks the extent of Δalign within coh ≥ 70."""
    if not (y_hi > y_lo):
        return
    ax.plot([x + dx, x + dx], [y_lo, y_hi], color=color, lw=lw, zorder=3)
    ax.plot([x + dx - 0.6, x + dx + 0.6], [y_lo, y_lo], color=color, lw=lw, zorder=3)
    ax.plot([x + dx - 0.6, x + dx + 0.6], [y_hi, y_hi], color=color, lw=lw, zorder=3)


def decorate(ax, *, title):
    # coh = 70 floor
    ax.axvline(COH_FLOOR, color="#bbbbbb", lw=1.0, ls=(0, (3, 3)), zorder=1)
    ax.text(COH_FLOOR + 0.4, 50.4, "coh = 70",
            color="#888888", fontsize=10, va="bottom", ha="left", zorder=1)
    ax.set_xlim(50, 100)
    ax.set_ylim(50, 100)
    ax.set_xticks(np.arange(50, 101, 10))
    ax.set_yticks(np.arange(50, 101, 10))
    ax.set_xlabel("coherence")
    ax.set_ylabel("alignment")
    ax.set_title(title, loc="center", pad=12, fontweight="600", color="#1a1a1a")
    ax.grid(True, color="#eeeeee", lw=0.6, zorder=0)
    ax.set_axisbelow(True)


# ─── Main ──────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nura-per-seed-dir", required=True)
    p.add_argument("--nura-additive", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--em-model", default="medical")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    setup_style()

    nura_path = Path(args.nura_per_seed_dir) / f"aggregated_seed{args.seed}_{args.em_model}.json"
    if not nura_path.exists():
        raise SystemExit(f"missing {nura_path}")
    additive_path = Path(args.nura_additive)
    if not additive_path.exists():
        raise SystemExit(f"missing {additive_path}")

    qkqk = load_method(nura_path, "qk_to_qk")
    qkov = load_method(nura_path, "qk_to_ov")
    ovov = load_method(nura_path, "ov_to_ov")
    baseline = load_method(nura_path, "baseline")
    additive = load_method(additive_path)

    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.4), sharex=True, sharey=True,
                             gridspec_kw=dict(wspace=0.10))

    # ── LEFT: three FRA recipes ──────────────────────────────────────────
    axL = axes[0]
    fra_methods = [
        (r"QK$\rightarrow$QK", qkqk, PAL[r"QK$\rightarrow$QK"], "o"),
        (r"OV$\rightarrow$OV", ovov, PAL[r"OV$\rightarrow$OV"], "s"),
        (r"QK$\rightarrow$OV", qkov, PAL[r"QK$\rightarrow$OV"], "D"),
    ]
    fra_stats = []
    for label, rows, color, marker in fra_methods:
        s = stats(rows)
        if s is None:
            continue
        d_str = f"  Δ = {s['delta']:5.2f}" if s["n70"] else "  Δ = —"
        draw_method(axL, rows, color, label + d_str, marker=marker)
        fra_stats.append((label, color, s))
    if baseline:
        b = baseline[0]
        draw_baseline_star(axL, b["mean_coherence"], b["mean_alignment"],
                           "baseline (no hook)")
    # Δ brackets on the right of each method's curve, just inside the axis frame
    bracket_x = 99.0
    for i, (label, color, s) in enumerate(fra_stats):
        if s["n70"] >= 1 and s["delta"] > 0:
            draw_delta_bracket(axL, x=bracket_x - i * 0.8,
                               y_lo=s["min_align"], y_hi=s["max_align"],
                               color=color)
    decorate(axL, title="FRA decomposition")
    axL.legend(loc="lower left", title="method")

    # ── RIGHT: conventional additive ────────────────────────────────────
    axR = axes[1]
    a_stats = stats(additive)
    a_lab = (f"additive  Δ = {a_stats['delta']:5.2f}"
             if a_stats and a_stats["n70"] else "additive  Δ = —")
    draw_method(axR, additive, PAL["additive"], a_lab, marker="^")
    if additive:
        sc = np.array([r["scale"] for r in additive])
        i_one = int(np.argmin(np.abs(sc - 1.0)))
        draw_baseline_star(axR,
                           additive[i_one]["mean_coherence"],
                           additive[i_one]["mean_alignment"],
                           "α = 1.0 (no-op)")
    if a_stats and a_stats["n70"] >= 1 and a_stats["delta"] > 0:
        draw_delta_bracket(axR, x=bracket_x, y_lo=a_stats["min_align"],
                           y_hi=a_stats["max_align"], color=PAL["additive"])
    decorate(axR, title="Conventional steering")
    axR.legend(loc="lower left", title="method")

    fig.suptitle(
        "Alignment vs coherence frontier — same SAE, same hookpoint, "
        "same prompts; recipe varies",
        fontsize=18, fontweight="600", color="#0a0a0a", y=1.04,
    )
    fig.text(
        0.5, -0.02,
        r"Qwen2.5-14B + medical LoRA $\cdot$ 8 evaluation prompts $\cdot$ "
        r"$\alpha \in \{0, 0.5, 1, 1.5, 2, 3\}$ $\cdot$ "
        f"eval seed = {args.seed}"
        r" $\cdot$ vertical brackets mark $\Delta$align across points at coh $\geq$ 70",
        ha="center", va="top", fontsize=10.5, color="#555555",
    )

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
