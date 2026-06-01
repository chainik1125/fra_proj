"""6-column per-seed appendix figure from jsd_alpha_sweep_6seeds.json.

Each column is one SAE training seed. Top row: JSD$_\\text{clean}$ (green)
and JSD$_\\text{pois}$ (red). Bottom row: exact-match rate to the clean
rollout (green) + ASR (red).

For every (SAE seed, α) the JSON records 5 per-decode-seed values per metric;
the line is the mean across those 5 seeds and the translucent band is the
min–max envelope.

OV  (single OV→OV)   = solid    + circle
Conv (resid-mid add) = dashed   + triangle

Outputs <out>.pdf and <out>.png.
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


def band(per_alpha: dict, key: str, raw_alphas: list[float],
         sae_idx: int, scale: float = 1.0
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lo, hi) over decode seeds for one SAE seed."""
    arr = np.array([per_alpha[f"{a:.1f}"][key][sae_idx] for a in raw_alphas]) * scale
    return arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def band_1d(per_alpha: dict, key: str, raw_alphas: list[float],
            scale: float = 1.0) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (mean, lo, hi) over decode seeds for an SAE-free config (no SAE-seed axis)."""
    arr = np.array([per_alpha[f"{a:.1f}"][key] for a in raw_alphas]) * scale
    return arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def draw(ax, xs, mean, lo, hi, color, ls, mk, lw, label):
    ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
    ax.plot(xs, mean, color=color, linestyle=ls, marker=mk, ms=4, lw=lw,
            label=label, zorder=3)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",  type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_all_seeds"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    d = json.loads(args.input.read_text())
    raw_alphas  = [float(a) for a in d["alphas"]]
    disp_alphas = [-a for a in raw_alphas]
    seeds       = d["sae_seeds"]
    n_prompts   = d["n_prompts"]
    ov          = d["configs"]["ov"]
    conv        = d["configs"]["conventional"]
    feats_ov    = ov["per_seed_feature"]
    feats_conv  = conv["per_seed_feature"]

    fig, axes = plt.subplots(
        2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
        sharex=True, sharey="row",
    )

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]
        i = seeds.index(seed)
        ov_pa, conv_pa = ov["per_alpha"], conv["per_alpha"]

        # — JSD curves —
        for pa, label_prefix, ls, mk, lw in [
            (ov_pa,   "OV",   "-",  "o", 1.3),
            (conv_pa, "Conv", "--", "^", 1.1),
        ]:
            jc_m, jc_lo, jc_hi = band(pa, "jsd_clean", raw_alphas, i)
            jp_m, jp_lo, jp_hi = band(pa, "jsd_pois",  raw_alphas, i)
            draw(ax_jsd, disp_alphas, jc_m, jc_lo, jc_hi, GREEN, ls, mk, lw,
                 f"{label_prefix}  JSD$_\\mathrm{{clean}}$")
            draw(ax_jsd, disp_alphas, jp_m, jp_lo, jp_hi, RED,   ls, mk, lw,
                 f"{label_prefix}  JSD$_\\mathrm{{pois}}$")
        ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax_jsd.set_ylim(-0.02, 1.05)
        ax_jsd.set_title(
            f"seed {seed}\nOV f={feats_ov[str(seed)]} · conv f={feats_conv[str(seed)]}",
            fontsize=10,
        )

        # — exact-match (green) + ASR (red) —
        for pa, label_prefix, ls, mk, lw in [
            (ov_pa,   "OV",   "-",  "o", 1.3),
            (conv_pa, "Conv", "--", "^", 1.1),
        ]:
            em_m, em_lo, em_hi = band(pa, "n_exact_match_clean", raw_alphas, i,
                                       scale=1.0 / n_prompts)
            as_m, as_lo, as_hi = band(pa, "asr", raw_alphas, i)
            draw(ax_em, disp_alphas, em_m, em_lo, em_hi, GREEN, ls, mk, lw,
                 f"{label_prefix}  exact-match")
            draw(ax_em, disp_alphas, as_m, as_lo, as_hi, RED,   ls, mk, lw,
                 f"{label_prefix}  ASR")
        ax_em.set_ylim(-0.02, 1.05)
        ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))

        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
            ax_em .set_ylabel("Exact-match rate / ASR")
        ax_jsd.grid(axis="y", color="#dddddd", lw=0.5)
        ax_em .grid(axis="y", color="#dddddd", lw=0.5)
        ax_jsd.set_axisbelow(True)
        ax_em .set_axisbelow(True)

    axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    axes[1, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4,
                      bbox_to_anchor=(0.0, 0.30))
    fig.suptitle("Per-seed JSD and exact-match/ASR", fontsize=15, y=1.02)
    for ax in axes.flat:
        ax.set_xlabel("")
    fig.supxlabel(r"steering strength $\alpha$", fontsize=15, y=0.0)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')}  and  {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
