"""Main-paper single-seed JSD / exact-match figure (1 row x 2 panels).

Left:  Jensen-Shannon divergence (bits) vs steering strength, clean (green)
       and poisoned (red) references, for OV / Conv.
Right: exact-match-to-clean rate (green) + ASR (red) for the same two.

Reads two run_experiment results files (uniform schema), one per method, each
run with --mode winner so there is one winning tuple per SAE seed:
  --ov   results/ov_winner.json
  --conv results/conv.json
The chosen --seed selects which SAE seed's winner to plot. Lines are means over
the decode seeds; bands are min-max across them (per-decode-seed metric lists).
DoM is parameter-free (single operating point) and appears only in the stats
table, so it is not drawn here.

OV = solid + circle      Conv = dashed + triangle
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter

GREEN = "#2E7D32"
RED   = "#B91C1C"
BAND_ALPHA = 0.18


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8,
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def load_alpha_sweep(path: Path, seed: int) -> dict:
    """The alpha_sweep dict for one SAE seed's winning tuple (uniform schema)."""
    d = json.loads(Path(path).read_text())
    rows = [r for r in d["results"] if r["seed"] == seed]
    if not rows:
        raise SystemExit(f"no results for seed {seed} in {path}")
    return rows[0]["alpha_sweep"]


def _items(alpha_sweep):
    return sorted(alpha_sweep.items(), key=lambda kv: float(kv[0]))


def series(alpha_sweep, key):
    """(alphas, mean, lo, hi) over per-decode-seed values for `key`."""
    items = _items(alpha_sweep)
    alphas = [float(k) for k, _ in items]
    arr = np.array([ev[key] for _, ev in items], dtype=float)   # (n_alpha, n_decode)
    return alphas, arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def series_exact(alpha_sweep):
    """Per-decode-seed exact-match *rate* = count / prompts-per-decode-seed."""
    items = _items(alpha_sweep)
    alphas = [float(k) for k, _ in items]
    rows = []
    for _, ev in items:
        n_per = ev["n_exact_match_clean_per_seed"]
        b = ev["exact_match_total_rows"] / len(n_per)
        rows.append([c / b for c in n_per])
    arr = np.array(rows, dtype=float)
    return alphas, arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def draw(ax, xs, mean, lo, hi, color, ls, mk, lw, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(xs, mean, color=color, linestyle=ls, marker=mk, ms=4, lw=lw,
            label=label, zorder=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ov",     type=Path, default=Path("results/ov_winner.json"))
    p.add_argument("--conv",   type=Path, default=Path("results/conv.json"))
    p.add_argument("--output", type=Path, default=Path("figures/jsd_exact_main"))
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    # (label, linestyle, marker, lw, alpha_sweep)
    methods = [("OV",   "-",  "o", 1.4, load_alpha_sweep(args.ov,   args.seed)),
               ("Conv", "--", "^", 1.2, load_alpha_sweep(args.conv, args.seed))]

    fig, (ax_jsd, ax_em) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    # Paper convention: steering strength is shown as negative α (α<0 = subtract
    # the feature). run_experiment stores the positive magnitude, so negate for display.
    for lab, ls, mk, lw, pa in methods:
        xs, jc_m, jc_lo, jc_hi = series(pa, "jsd_clean_per_seed")
        _,  jp_m, jp_lo, jp_hi = series(pa, "jsd_pois_per_seed")
        dx = [-a for a in xs]
        draw(ax_jsd, dx, jc_m, jc_lo, jc_hi, GREEN, ls, mk, lw, f"{lab}  JSD$_\\mathrm{{clean}}$")
        draw(ax_jsd, dx, jp_m, jp_lo, jp_hi, RED,   ls, mk, lw, f"{lab}  JSD$_\\mathrm{{pois}}$")
    ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
    ax_jsd.set_ylim(-0.02, 1.06)
    ax_jsd.set_ylabel("Jensen-Shannon divergence (bits)")
    ax_jsd.set_xlabel(r"steering strength $\alpha$")
    ax_jsd.legend(loc="upper left", fontsize=7, framealpha=0.92,
                  handlelength=1.8, borderpad=0.4, ncol=1, bbox_to_anchor=(0.01, 0.90))
    ax_jsd.grid(axis="y", color="#dddddd", lw=0.5); ax_jsd.set_axisbelow(True)

    for lab, ls, mk, lw, pa in methods:
        xs, em_m, em_lo, em_hi    = series_exact(pa)
        _,  asr_m, asr_lo, asr_hi = series(pa, "asr_per_seed")
        dx = [-a for a in xs]
        draw(ax_em, dx, em_m,  em_lo,  em_hi,  GREEN, ls, mk, lw, f"{lab}  exact-match")
        draw(ax_em, dx, asr_m, asr_lo, asr_hi, RED,   ls, mk, lw, f"{lab}  ASR")
    ax_em.set_ylim(-0.02, 1.06)
    ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_em.set_ylabel("exact-match rate / ASR")
    ax_em.set_xlabel(r"steering strength $\alpha$")
    ax_em.legend(loc="upper left", fontsize=7.5, framealpha=0.92,
                 handlelength=1.8, borderpad=0.4, ncol=1)
    ax_em.grid(axis="y", color="#dddddd", lw=0.5); ax_em.set_axisbelow(True)

    n_decode = len(next(iter(methods[0][4].values()))["jsd_clean_per_seed"])
    fig.suptitle(f"SAE seed {args.seed}  ·  mean over {n_decode} decode seeds, band = min-max",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')}  and  {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
