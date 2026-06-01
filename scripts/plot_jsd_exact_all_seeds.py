"""Per-seed appendix figure: one column per SAE training seed, OV vs Conv.

Top row: JSD$_\\text{clean}$ (green) and JSD$_\\text{pois}$ (red). Bottom row:
exact-match rate to the clean rollout (green) + ASR (red). Lines are means over
the decode seeds; bands are the min–max envelope (per-decode-seed metric lists).

Reads two run_experiment results files (uniform schema), each --mode winner so
there is one winning tuple per SAE seed:
  --ov   results/ov_winner.json
  --conv results/conv.json

OV  = solid  + circle      Conv = dashed + triangle

Outputs <output>.pdf and <output>.png.
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


def load_by_seed(path: Path) -> tuple[dict, dict]:
    """Return (seed → alpha_sweep, seed → feature index) for a method file."""
    d = json.loads(Path(path).read_text())
    sweeps, feats = {}, {}
    for r in d["results"]:
        sweeps[r["seed"]] = r["alpha_sweep"]
        feats[r["seed"]]  = r["tuple"][0][0] if r["tuple"] else "—"
    return sweeps, feats


def _items(alpha_sweep):
    return sorted(alpha_sweep.items(), key=lambda kv: float(kv[0]))


def series(alpha_sweep, key):
    items = _items(alpha_sweep)
    alphas = [float(k) for k, _ in items]
    arr = np.array([ev[key] for _, ev in items], dtype=float)   # (n_alpha, n_decode)
    return alphas, arr.mean(axis=1), arr.min(axis=1), arr.max(axis=1)


def series_exact(alpha_sweep):
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
    p.add_argument("--output", type=Path, default=Path("figures/jsd_exact_all_seeds"))
    args = p.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    setup_style()
    ov_sw,   ov_feat   = load_by_seed(args.ov)
    conv_sw, conv_feat = load_by_seed(args.conv)
    seeds = sorted(set(ov_sw) & set(conv_sw))

    fig, axes = plt.subplots(2, len(seeds), figsize=(2.6 * len(seeds), 5.6),
                             sharex=True, sharey="row", squeeze=False)

    for col, seed in enumerate(seeds):
        ax_jsd, ax_em = axes[0, col], axes[1, col]
        for sw, ls, mk, lw in [(ov_sw[seed], "-", "o", 1.3),
                               (conv_sw[seed], "--", "^", 1.1)]:
            xs, jc_m, jc_lo, jc_hi = series(sw, "jsd_clean_per_seed")
            _,  jp_m, jp_lo, jp_hi = series(sw, "jsd_pois_per_seed")
            dx = [-a for a in xs]   # paper shows steering strength as negative α
            draw(ax_jsd, dx, jc_m, jc_lo, jc_hi, GREEN, ls, mk, lw, "JSD$_\\mathrm{clean}$")
            draw(ax_jsd, dx, jp_m, jp_lo, jp_hi, RED,   ls, mk, lw, "JSD$_\\mathrm{pois}$")
        ax_jsd.axhline(1.0, color="#999", lw=0.5, ls=":")
        ax_jsd.set_ylim(-0.02, 1.05)
        ax_jsd.set_title(f"seed {seed}\nOV f={ov_feat[seed]} · conv f={conv_feat[seed]}",
                         fontsize=10)

        for sw, ls, mk, lw in [(ov_sw[seed], "-", "o", 1.3),
                               (conv_sw[seed], "--", "^", 1.1)]:
            xs, em_m, em_lo, em_hi    = series_exact(sw)
            _,  as_m, as_lo, as_hi    = series(sw, "asr_per_seed")
            dx = [-a for a in xs]
            draw(ax_em, dx, em_m, em_lo, em_hi, GREEN, ls, mk, lw, "exact-match")
            draw(ax_em, dx, as_m, as_lo, as_hi, RED,   ls, mk, lw, "ASR")
        ax_em.set_ylim(-0.02, 1.05)
        ax_em.yaxis.set_major_formatter(PercentFormatter(1.0))

        if col == 0:
            ax_jsd.set_ylabel("JSD (bits)")
            ax_em.set_ylabel("Exact-match rate / ASR")
        ax_jsd.grid(axis="y", color="#dddddd", lw=0.5); ax_jsd.set_axisbelow(True)
        ax_em.grid(axis="y", color="#dddddd", lw=0.5);  ax_em.set_axisbelow(True)

    axes[0, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4)
    axes[1, 0].legend(loc="lower left", fontsize=7.5, framealpha=0.92,
                      handlelength=1.6, borderpad=0.4, bbox_to_anchor=(0.0, 0.30))
    fig.suptitle("Per-seed JSD and exact-match/ASR", fontsize=15, y=1.02)
    for ax in axes.flat:
        ax.set_xlabel("")
    fig.supxlabel(r"steering strength $\alpha$", fontsize=15, y=0.0)
    fig.tight_layout()
    fig.savefig(args.output.with_suffix(".pdf"))
    fig.savefig(args.output.with_suffix(".png"), dpi=180)
    print(f"wrote {args.output.with_suffix('.pdf')}  and  {args.output.with_suffix('.png')}")


if __name__ == "__main__":
    main()
