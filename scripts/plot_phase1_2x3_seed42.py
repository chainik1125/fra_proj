#!/usr/bin/env python3
"""2×3 alignment-vs-coherence frontier figure at eval seed=42.

Layout:
  Cols (left → right) : medical, finance, sports
  Row 0 (top)         : 3 FRA decomposition recipes on the published L24 ln1 SAE
                        (QK→QK green, QK→OV blue, OV→OV orange) overlaid; black
                        star = unsteered (the published-SAE baseline, no hook).
  Row 1 (bottom)      : conventional additive steering on the same SAE (black);
                        black star at α=1.0 = mathematical no-op of (α-1)·f·W_dec.

Each cell shows the α-sweep trajectory in alignment-vs-coherence space, with
α labelled at every point and a stats box reporting peak / baseline /
min@coh70 / Δ@coh70.

Inputs: gpt4o_combined_*.json under --combined-root (same files as the other
phase1 plots).

Paper colour scheme:
  QK→QK green   #009E73
  QK→OV blue    #0072B2
  OV→OV orange  #D55E00
  conventional black #000000
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
SEED_INDEX = 0  # combined files store per-seed lists sorted by seed value (42, 123, 456) → index 0 = 42
DOMAINS = ["medical", "finance", "sports"]

FRA_RECIPES = [
    ("qk_to_qk", r"QK$\rightarrow$QK", "#009E73"),
    ("qk_to_ov", r"QK$\rightarrow$OV", "#0072B2"),
    ("ov_to_ov", r"OV$\rightarrow$OV", "#D55E00"),
]
ADDITIVE_COLOR = "#000000"

# Conventional SAE candidates — bottom row picks the one with the largest
# Δ@coh70 per domain (at seed 42).
ADD_SAES = [
    ("L24_ln1_nura",   "L24 ln1 (published)"),
    ("L24_resid_pre",  "L24 resid_pre"),
    ("L24_resid_mid",  "L24 resid_mid"),
    ("L24_resid_post", "L24 resid_post"),
    ("L25_ln1",        "L25 ln1"),
]


def setup_style():
    mpl.rcParams.update({
        "font.family":     "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":           14,
        "axes.titlesize":      16,
        "axes.labelsize":      14,
        "axes.spines.top":     False,
        "axes.spines.right":   False,
        "axes.linewidth":      1.1,
        "axes.edgecolor":      "#222222",
        "axes.labelcolor":     "#1a1a1a",
        "xtick.color":         "#222222",
        "ytick.color":         "#222222",
        "xtick.labelsize":     11,
        "ytick.labelsize":     11,
        "xtick.direction":     "out",
        "ytick.direction":     "out",
        "legend.frameon":      True,
        "legend.fontsize":     10.5,
        "figure.dpi":          110,
        "savefig.bbox":        "tight",
        "savefig.pad_inches":  0.10,
    })


def _per_seed_curve(by_alpha, seed_idx=SEED_INDEX):
    scales, al, co = [], [], []
    for e in by_alpha:
        if seed_idx < len(e["per_seed_alignment"]):
            scales.append(e["scale"])
            al.append(e["per_seed_alignment"][seed_idx])
            co.append(e["per_seed_coherence"][seed_idx])
    order = np.argsort(scales)
    return np.array(scales)[order], np.array(al)[order], np.array(co)[order]


def _summary(scales, al, co, *, baseline_scale=1.0, floor=COH_FLOOR):
    out = {
        "peak": float(np.nanmax(al)) if al.size else float("nan"),
        "baseline": float("nan"),
        "min_above": float("nan"),
        "delta": float("nan"),
        "n_above": 0,
        "n_total": int(al.size),
    }
    if (scales == baseline_scale).any():
        out["baseline"] = float(al[scales == baseline_scale][0])
    mask = co >= floor
    if mask.any():
        out["min_above"] = float(al[mask].min())
        out["delta"] = float(al[mask].max() - al[mask].min())
        out["n_above"] = int(mask.sum())
    return out


def _draw_curve(ax, scales, al, co, color, *, marker="o", s=70, label=None):
    ax.plot(co, al, color=color, lw=1.6, alpha=0.85, zorder=2)
    ax.scatter(co, al, c=color, s=s, edgecolors="white", linewidths=0.9,
               zorder=3, marker=marker, label=label)
    for sc, x, y in zip(scales, co, al):
        ax.annotate(f"{sc}", (x, y), xytext=(4, 4),
                    textcoords="offset points", fontsize=8.5, color="#444")


def _baseline_star(ax, co, al, label):
    ax.scatter([co], [al], marker="*", s=300,
               color="black", edgecolors="white", linewidths=1.0, zorder=6,
               label=label)


def _stat_box(ax, lines):
    if not lines:
        return
    line_h = 0.062
    pad_y = 0.025
    box_top = 0.985
    box_bottom = box_top - line_h * len(lines) - pad_y
    box_left, box_right = 0.020, 0.620
    bg = FancyBboxPatch(
        (box_left, box_bottom),
        box_right - box_left, box_top - box_bottom,
        boxstyle="round,pad=0.012,rounding_size=0.012",
        transform=ax.transAxes,
        facecolor="white", edgecolor="#888", linewidth=0.7,
        alpha=0.95, zorder=10,
    )
    ax.add_patch(bg)
    for i, (text, is_bold) in enumerate(lines):
        ax.text(
            box_left + 0.015,
            box_top - pad_y * 0.5 - i * line_h,
            text,
            transform=ax.transAxes,
            fontsize=9,
            family="DejaVu Sans",
            fontweight="bold" if is_bold else "normal",
            color="#111" if is_bold else "#333",
            verticalalignment="top",
            horizontalalignment="left",
            zorder=11,
        )


def _decorate(ax, *, title=None, xlabel=False, ylabel=False, legend_loc=None):
    ax.axvline(COH_FLOOR, color="grey", lw=0.8, ls=":", zorder=1)
    ax.axhline(50, color="grey", lw=0.5, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    if xlabel:
        ax.set_xlabel("coherence", fontsize=12)
    if ylabel:
        ax.set_ylabel("alignment", fontsize=12)
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold")
    ax.grid(True, ls=":", alpha=0.25)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=10)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--combined-root", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()
    setup_style()

    root = Path(args.combined_root)
    fig, axes = plt.subplots(2, 3, figsize=(16, 10),
                             sharex=True, sharey=True)

    for c, em in enumerate(DOMAINS):
        # ── Row 0: FRA recipes overlaid ─────────────────────────────────
        ax = axes[0, c]
        fra_block = json.loads((root / f"gpt4o_combined_L24_ln1_nura_FRA_{em}.json").read_text())
        cond_stats = []
        for method, lbl, color in FRA_RECIPES:
            block = fra_block.get(method)
            if block is None:
                continue
            scales, al, co = _per_seed_curve(block["by_alpha"])
            if scales.size == 0:
                continue
            _draw_curve(ax, scales, al, co, color, label=f"FRA: {lbl}")
            cond_stats.append((f"FRA {lbl}", color, _summary(scales, al, co)))
        # Unsteered baseline (no-hook)
        if "baseline" in fra_block and fra_block["baseline"]["by_alpha"]:
            be = fra_block["baseline"]["by_alpha"][0]
            if SEED_INDEX < len(be["per_seed_alignment"]):
                _baseline_star(ax, be["per_seed_coherence"][SEED_INDEX],
                               be["per_seed_alignment"][SEED_INDEX],
                               label="unsteered (no hook)")
        # Stats box: bold the winner
        if cond_stats:
            valid = [(i, s["delta"]) for i, (_, _, s) in enumerate(cond_stats)
                     if s["n_above"] > 0 and s["delta"] == s["delta"]]
            winner_i = max(valid, key=lambda x: x[1])[0] if valid else -1
            lines = [("Δ@coh70 / peak (n above floor):", False)]
            for i, (lbl, _col, s) in enumerate(cond_stats):
                if s["n_above"] > 0:
                    txt = f"  {lbl:8s}  Δ={s['delta']:5.2f}  pk={s['peak']:5.1f}  ({s['n_above']}/{s['n_total']})"
                else:
                    txt = f"  {lbl:8s}  Δ=NaN    pk={s['peak']:5.1f}  (0/{s['n_total']})"
                lines.append((txt, i == winner_i))
            _stat_box(ax, lines)
        title = f"{em.capitalize()}\nFRA decomposition recipes" if c == 0 else f"{em.capitalize()}"
        if c != 0:
            title = f"{em.capitalize()}"
        # only show legend in col 0 to avoid clutter
        _decorate(ax, title=em.capitalize(), ylabel=(c == 0),
                  legend_loc="lower right" if c == 0 else None)

        # ── Row 1: Best-performing conventional SAE for this domain ─────
        # Pick the SAE with the largest Δ@coh70 at seed=42 across all 5
        # additive candidates. Falls back to peak alignment if no α-point
        # clears the coherence floor (e.g. finance).
        ax = axes[1, c]
        best = None  # (sae_id, sae_lbl, scales, al, co, summary)
        for sae_id, sae_lbl in ADD_SAES:
            p = root / f"gpt4o_combined_{sae_id}_{em}.json"
            if not p.exists():
                continue
            d = json.loads(p.read_text())
            block = d.get("sae_resid", {}).get("by_alpha", [])
            if not block:
                continue
            scales, al, co = _per_seed_curve(block)
            if scales.size == 0:
                continue
            s = _summary(scales, al, co)
            # Sort key: (n_above > 0, delta if n_above>0 else peak)
            key = (1, s["delta"]) if s["n_above"] > 0 else (0, s["peak"])
            if best is None or key > best[6]:
                best = (sae_id, sae_lbl, scales, al, co, s, key)

        if best is not None:
            sae_id, sae_lbl, scales, al, co, s, _ = best
            _draw_curve(ax, scales, al, co, ADDITIVE_COLOR,
                        marker="s", s=58,
                        label=f"Conventional additive — {sae_lbl}")
            if (scales == 1.0).any():
                i_b = int(np.where(scales == 1.0)[0][0])
                _baseline_star(ax, co[i_b], al[i_b],
                               label=r"$\alpha = 1$ (no-op)")
            if s["n_above"] > 0:
                lines = [
                    (f"Best conventional: {sae_lbl}", True),
                    (f"Δ@coh70 = {s['delta']:.2f}", True),
                    (f"peak | coh>=70 = {al[co >= COH_FLOOR].max():.1f}", False),
                    (f"peak overall = {s['peak']:.1f}", False),
                    (f"baseline (a=1) = {s['baseline']:.1f}", False),
                    (f"min | coh>=70 = {s['min_above']:.1f}", False),
                    (f"n@coh70 = {s['n_above']}/{s['n_total']}", False),
                ]
            else:
                lines = [
                    (f"Best conventional: {sae_lbl}", True),
                    ("Δ@coh70 = NaN (no α at coh>=70)", True),
                    (f"peak overall = {s['peak']:.1f}", False),
                    (f"baseline (a=1) = {s['baseline']:.1f}", False),
                    (f"n@coh70 = 0/{s['n_total']}", False),
                ]
            _stat_box(ax, lines)
        _decorate(ax, xlabel=True, ylabel=(c == 0),
                  legend_loc="lower right" if c == 0 else None)

    # Per-row labels on the LEFT margin, outside the axes
    fig.text(0.005, 0.78, "FRA decomposition\n(published L24 ln1 SAE)",
             rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold", color="#444")
    fig.text(0.005, 0.30, "Conventional additive\n(best SAE per domain)",
             rotation=90, va="center", ha="center",
             fontsize=12, fontweight="bold", color="#444")

    fig.suptitle(r"Alignment-vs-coherence trajectory at eval seed = 42  ($\alpha$ in {0, 0.5, 1, 1.5, 2, 3})",
                 fontsize=15, fontweight="bold", y=0.998)
    fig.tight_layout(rect=[0.025, 0, 1, 0.985])

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=200)
    fig.savefig(str(out) + ".pdf")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    main()
