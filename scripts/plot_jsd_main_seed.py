"""Main-paper single-seed JSD / exact-match figure (1 row x 2 panels).

Left:  Jensen-Shannon divergence (bits) vs steering strength, clean (green)
       and poisoned (red) references, for OV / Conv / DoM.
Right: exact-match-to-clean rate (green) + ASR (red) for the same three.

Reads results/jsd_alpha_sweep_6seeds.json (configs: ov, conventional, dom).
OV / Conv are per-SAE-seed (one seed selected); DoM is SAE-free (single curve).
Lines are means over 5 decode seeds, band = min-max.

OV   = solid  + circle      Conv = dashed + triangle      DoM = dotted + square
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


def band(per_alpha, key, raw_alphas, sae_idx, scale=1.0):
    arr = np.array([per_alpha[f"{a:.1f}"][key][sae_idx] for a in raw_alphas]) * scale
    return arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def band_1d(per_alpha, key, raw_alphas, scale=1.0):
    arr = np.array([per_alpha[f"{a:.1f}"][key] for a in raw_alphas]) * scale
    return arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def draw(ax, xs, mean, lo, hi, color, ls, mk, lw, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(xs, mean, color=color, linestyle=ls, marker=mk, ms=4, lw=lw,
            label=label, zorder=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path, default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output", type=Path, default=Path("figures/jsd_exact_main"))
    p.add_argument("--seed",   type=int, default=0)
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    d = json.loads(args.input.read_text())
    raw_alphas  = [float(a) for a in d["alphas"]]
    disp_alphas = [-a for a in raw_alphas]
    n_prompts   = d["n_prompts"]
    i = d["sae_seeds"].index(args.seed)

    ov_pa   = d["configs"]["ov"]["per_alpha"]
    conv_pa = d["configs"]["conventional"]["per_alpha"]

    # (label, linestyle, marker, lw, per_alpha, is_sae_free)
    # DoM is parameter-free (single canonical point at alpha=1), so it is not
    # swept and not drawn as a line. It appears only in the stats table.
    methods = [("OV", "-", "o", 1.4, ov_pa, False),
               ("Conv", "--", "^", 1.2, conv_pa, False)]

    fig, (ax_jsd, ax_em) = plt.subplots(1, 2, figsize=(9.2, 3.7))

    for lab, ls, mk, lw, pa, sae_free in methods:
        jc = band_1d(pa, "jsd_clean", raw_alphas) if sae_free else band(pa, "jsd_clean", raw_alphas, i)
        jp = band_1d(pa, "jsd_pois",  raw_alphas) if sae_free else band(pa, "jsd_pois",  raw_alphas, i)
        draw(ax_jsd, disp_alphas, *jc, GREEN, ls, mk, lw, f"{lab}  JSD$_\\mathrm{{clean}}$")
        draw(ax_jsd, disp_alphas, *jp, RED,   ls, mk, lw, f"{lab}  JSD$_\\mathrm{{pois}}$")
    ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
    ax_jsd.set_ylim(-0.02, 1.06)
    ax_jsd.set_ylabel("Jensen-Shannon divergence (bits)")
    ax_jsd.set_xlabel(r"steering strength $\alpha$")
    ax_jsd.legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                  handlelength=1.8, borderpad=0.4, ncol=1)
    ax_jsd.grid(axis="y", color="#dddddd", lw=0.5); ax_jsd.set_axisbelow(True)

    for lab, ls, mk, lw, pa, sae_free in methods:
        em = (band_1d(pa, "n_exact_match_clean", raw_alphas, scale=1.0 / n_prompts) if sae_free
              else band(pa, "n_exact_match_clean", raw_alphas, i, scale=1.0 / n_prompts))
        asr = band_1d(pa, "asr", raw_alphas) if sae_free else band(pa, "asr", raw_alphas, i)
        draw(ax_em, disp_alphas, *em,  GREEN, ls, mk, lw, f"{lab}  exact-match")
        draw(ax_em, disp_alphas, *asr, RED,   ls, mk, lw, f"{lab}  ASR")
    ax_em.set_ylim(-0.02, 1.06)
    ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax_em.set_ylabel("exact-match rate / ASR")
    ax_em.set_xlabel(r"steering strength $\alpha$")
    ax_em.legend(loc="upper left", fontsize=7.5, framealpha=0.92,
                 handlelength=1.8, borderpad=0.4, ncol=1)
    ax_em.grid(axis="y", color="#dddddd", lw=0.5); ax_em.set_axisbelow(True)

    fig.suptitle(f"SAE seed {args.seed}  ·  mean over 5 decode seeds, band = min-max",
                 fontsize=12, y=1.01)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')}  and  {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
