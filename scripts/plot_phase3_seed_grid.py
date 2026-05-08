#!/usr/bin/env python3
"""Phase 3: 3 (rows = seeds) × 5 (cols = method/hookpoint) grid figure.

Cols:
  0  Nura medical @ L24 ln1.hook_normalized       (QK→OV / OV→OV / QK→QK overlaid)
  1  SAE-resid @ blocks.24.hook_resid_pre
  2  SAE-resid @ blocks.24.hook_resid_mid
  3  SAE-resid @ blocks.24.hook_resid_post
  4  SAE-resid @ blocks.25.ln1.hook_normalized

Rows: seed = 42, 123, 456 (3 seeds total).

Each cell:
  - α-sweep alignment-vs-coherence trajectory
  - α annotated at each point
  - Black star at α=0 (unsteered)
  - coh=70 floor + align=50 reference lines

Usage:
    python scripts/plot_phase3_seed_grid.py \
        --nura-per-seed-dir <dir>     # contains aggregated_seed{42,123,456}_medical.json
        --sae label=path[=seed] [...]
        --out <basename>
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


COH_FLOOR = 70.0
SEEDS_ORDER = [42, 123, 456]
SAE_COLS = [
    ("resid_pre_L24",     "blocks.24.hook_resid_pre"),
    ("resid_mid_L24",     "blocks.24.hook_resid_mid"),
    ("resid_post_L24",    "blocks.24.hook_resid_post"),
    ("ln1_normalised_L25","blocks.25.ln1.hook_normalized"),
]
NURA_CONDITIONS = [
    ("qk_to_ov", "QK→OV", "#d73027"),
    ("ov_to_ov", "OV→OV", "#4575b4"),
    ("qk_to_qk", "QK→QK", "#1a9850"),
]


def _seed_from_filename(p: str):
    m = re.search(r"seed[_-]?(\d+)", p)
    return int(m.group(1)) if m else None


def parse_sae_arg(s: str):
    parts = s.split("=", 2)
    if len(parts) < 2:
        raise SystemExit(f"--sae '{s}' must be label=path[=seed]")
    if len(parts) == 2:
        return parts[0], parts[1], None
    return parts[0], parts[1], int(parts[2])


def load_method_rows(path: Path, prefer_method: str | None = None):
    data = json.loads(path.read_text())
    if "aggregated" in data and isinstance(data["aggregated"], dict):
        data = data["aggregated"]
    if prefer_method and prefer_method in data:
        return data[prefer_method]
    rows = []
    for m, r in data.items():
        if isinstance(r, list):
            rows.extend(r)
    return rows


def _draw_curve(ax, scales, al, co, color, label):
    ax.plot(co, al, color=color, lw=1.4, alpha=0.85, zorder=2)
    ax.scatter(co, al, c=color, s=60, edgecolors="black",
               linewidths=0.4, zorder=3, label=label)
    for sc, x, y in zip(scales, co, al):
        ax.annotate(f"{sc}", (x, y), xytext=(3, 3),
                    textcoords="offset points", fontsize=5.5, color="#333")
    i0 = int(np.argmin(np.abs(np.asarray(scales) - 0.0)))
    ax.scatter([co[i0]], [al[i0]], marker="*", s=200,
               color="black", edgecolors="white", linewidths=0.9, zorder=5)


def _stats(rows, floor=COH_FLOOR):
    """Per-curve summary: Δalign|coh≥floor, peak align|coh≥floor, peak overall, n70."""
    rows = sorted(rows, key=lambda r: r["scale"])
    al = np.array([r["mean_alignment"] for r in rows], dtype=float)
    co = np.array([r["mean_coherence"] for r in rows], dtype=float)
    mask = co >= floor
    if mask.any():
        delta = float(al[mask].max() - al[mask].min())
        peak_at_floor = float(al[mask].max())
        n70 = int(mask.sum())
    else:
        delta = float("nan")
        peak_at_floor = float("nan")
        n70 = 0
    return {
        "delta": delta,
        "peak_at_floor": peak_at_floor,
        "peak_overall": float(np.nanmax(al)) if al.size else float("nan"),
        "n70": n70,
        "n_total": len(rows),
    }


def _stat_box(ax, lines, *, color="#222"):
    ax.text(0.02, 0.98, "\n".join(lines),
            transform=ax.transAxes, fontsize=6.5,
            verticalalignment="top", horizontalalignment="left",
            family="monospace", color=color,
            bbox=dict(facecolor="white", edgecolor="#bbb",
                      alpha=0.9, pad=2.5, boxstyle="round,pad=0.3"))


def _decorate(ax, *, title=None, xlabel=False, ylabel=False, legend_loc=None):
    ax.axvline(COH_FLOOR, color="grey", lw=0.6, ls=":", zorder=1)
    ax.axhline(50, color="grey", lw=0.4, ls=":", zorder=1)
    ax.set_xlim(-2, 102)
    ax.set_ylim(-2, 102)
    if xlabel:
        ax.set_xlabel("coherence", fontsize=9)
    if ylabel:
        ax.set_ylabel("alignment", fontsize=9)
    if title:
        ax.set_title(title, fontsize=9)
    ax.grid(True, ls=":", alpha=0.25)
    ax.tick_params(labelsize=8)
    if legend_loc:
        ax.legend(loc=legend_loc, fontsize=7)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nura-per-seed-dir", required=True,
                   help="dir with aggregated_seed{42,123,456}_medical.json")
    p.add_argument("--sae", action="append", default=[], required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    # ── Load Nura per-seed ──────────────────────────────────────────────
    nura = {}
    for seed in SEEDS_ORDER:
        f = Path(args.nura_per_seed_dir) / f"aggregated_seed{seed}_medical.json"
        if f.exists():
            nura[seed] = json.loads(f.read_text())

    # ── Group SAE entries by (label, seed) ──────────────────────────────
    by_label_seed: dict[tuple[str, int], list] = {}
    for sae_arg in args.sae:
        label, path, seed = parse_sae_arg(sae_arg)
        if seed is None:
            seed = _seed_from_filename(path) or 0
        rows = sorted(load_method_rows(Path(path)), key=lambda r: r["scale"])
        if rows:
            by_label_seed[(label, seed)] = rows

    # ── Plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(3, 5, figsize=(20, 11), sharex=True, sharey=True)

    for r, seed in enumerate(SEEDS_ORDER):
        # Col 0: Nura combined
        ax = axes[r, 0]
        nura_data = nura.get(seed, {})
        nura_lines = []
        for method, label, color in NURA_CONDITIONS:
            rows = nura_data.get(method, [])
            if not rows: continue
            rows = sorted(rows, key=lambda rr: rr["scale"])
            scales = np.array([rr["scale"] for rr in rows])
            al = np.array([rr["mean_alignment"] for rr in rows])
            co = np.array([rr["mean_coherence"] for rr in rows])
            _draw_curve(ax, scales, al, co, color, label)
            s = _stats(rows)
            nura_lines.append(
                f"{label}: Δ={s['delta']:5.1f}  peak={s['peak_at_floor'] if s['peak_at_floor']==s['peak_at_floor'] else float('nan'):5.1f}  n={s['n70']}/{s['n_total']}"
                if s["n70"] > 0 else
                f"{label}: Δ=NaN     peak=---   n=0/{s['n_total']}"
            )
        if nura_lines:
            _stat_box(ax, nura_lines)
        title = f"Nura medical @ L24 ln1\nseed={seed}" if r == 0 else f"seed={seed}"
        _decorate(ax, title=title, xlabel=(r == 2), ylabel=True,
                  legend_loc="lower right" if r == 0 else None)

        # Cols 1..4: SAE-resid
        for c, (key, hookname) in enumerate(SAE_COLS, start=1):
            ax = axes[r, c]
            rows = by_label_seed.get((key, seed), [])
            if rows:
                scales = np.array([rr["scale"] for rr in rows])
                al = np.array([rr["mean_alignment"] for rr in rows])
                co = np.array([rr["mean_coherence"] for rr in rows])
                _draw_curve(ax, scales, al, co, "#222", f"seed={seed}")
                s = _stats(rows)
                lines = [
                    f"Δalign|coh≥70 = {s['delta']:5.2f}" if s["n70"] > 0 else "Δalign|coh≥70 = NaN",
                    f"peak |coh≥70 = {s['peak_at_floor']:5.2f}" if s["n70"] > 0 else "peak |coh≥70 = ---",
                    f"peak overall = {s['peak_overall']:5.2f}",
                    f"n@coh≥70 = {s['n70']}/{s['n_total']}",
                ]
                _stat_box(ax, lines)
            title = f"SAE-resid\n{hookname}" if r == 0 else None
            _decorate(ax, title=title, xlabel=(r == 2), ylabel=False)

    if args.title:
        fig.suptitle(args.title, fontsize=13, y=0.995)
    fig.tight_layout()

    out = Path(args.out).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out) + ".png", dpi=160, bbox_inches="tight")
    fig.savefig(str(out) + ".pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"plot → {out}.png / .pdf")


if __name__ == "__main__":
    sys.exit(main())
