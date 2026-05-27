"""1×2 figure (JSD + exact-match/ASR) at a single SAE seed.

Reads results/jsd_alpha_sweep_6seeds.json (schema_version 2): per-(α, metric)
values are stored per (SAE seed) × (decode seed). For each α we plot the mean
over decode seeds as the line and a translucent fill_between of the min–max
range across decode seeds.

Left panel  — JSD(steered, clean) green, JSD(steered, poisoned) red.
Right panel — exact-match rate to the clean rollout (green) + ASR (red).

Both methods (OV-only upstream and conventional resid-mid additive) overlaid.

Usage:
  python -m scripts.plot_jsd_exact_single_seed --seed 5
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import PercentFormatter


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          15,
        "axes.titlesize":     18,
        "axes.labelsize":     16,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.2,
        "axes.edgecolor":     "#222222",
        "xtick.labelsize":    13,
        "ytick.labelsize":    14,
        "legend.frameon":     True,
        "legend.fontsize":    11,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def per_alpha_band(cfg_metric: dict, key: str, raw_alphas: list[float],
                   sae_idx: int, scale: float = 1.0
                   ) -> tuple[list[float], list[float], list[float]]:
    """Return (mean, lo, hi) across decode seeds at each α for one SAE seed.

    `cfg_metric` is e.g. cfg["ov"]["per_alpha"]; each entry is a list of
    len(sae_seeds), with each item itself a list of per-decode-seed values.
    """
    mean, lo, hi = [], [], []
    for a in raw_alphas:
        vals = np.array(cfg_metric[f"{a:.1f}"][key][sae_idx], dtype=float) * scale
        mean.append(float(vals.mean()))
        lo.append(float(vals.min()))
        hi.append(float(vals.max()))
    return mean, lo, hi


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_seed"))
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    setup_style()
    data        = json.loads(args.input.read_text())
    raw_alphas  = [float(a) for a in data["alphas"]]
    disp_alphas = [-a for a in raw_alphas]
    sae_seeds   = data["sae_seeds"]
    n_prompts   = data["n_prompts"]
    eval_seeds  = data.get("eval_seeds", [0])
    cfg         = data["configs"]
    if args.seed not in sae_seeds:
        raise SystemExit(f"seed {args.seed} not in results (have {sae_seeds})")
    idx = sae_seeds.index(args.seed)

    GREEN = "#1a8a3f"
    RED   = "#c0322a"
    BAND_ALPHA = 0.18

    methods = [
        ("ov",           "OV",   "-",  "o"),
        ("conventional", "Conv", "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    def draw(ax, xs, mean, lo, hi, color, ls, mk, label):
        ax.fill_between(xs, lo, hi, color=color, alpha=BAND_ALPHA, lw=0, zorder=2)
        ax.plot(xs, mean, color=color, lw=2.4, marker=mk, markersize=7,
                linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                label=label, zorder=3)

    # ── Left panel: JSD ──────────────────────────────────────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        jc_m, jc_lo, jc_hi = per_alpha_band(pa, "jsd_clean", raw_alphas, idx)
        jp_m, jp_lo, jp_hi = per_alpha_band(pa, "jsd_pois",  raw_alphas, idx)
        draw(ax_l, disp_alphas, jc_m, jc_lo, jc_hi, GREEN, ls, mk,
             f"{name}  JSD(steered, clean)")
        draw(ax_l, disp_alphas, jp_m, jp_lo, jp_hi, RED, ls, mk,
             f"{name}  JSD(steered, poisoned)")

    ax_l.axhline(1.0, color="#888", linestyle=":", lw=0.9, alpha=0.7)
    ax_l.text(disp_alphas[-1], 1.0 - 0.015, "JSD upper bound (1 bit)",
              fontsize=10.5, color="#666", ha="left", va="top")
    ax_l.set_ylabel("Jensen–Shannon divergence (bits)")
    ax_l.set_xlabel(r"steering strength  $\alpha$")
    ax_l.set_ylim(-0.04, 1.10)
    ax_l.set_xticks(disp_alphas)
    ax_l.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_l.get_xticklabels(), rotation=45, ha="right")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="lower left", framealpha=0.95, edgecolor="#bbbbbb")

    # ── Right panel: exact-match rate (green) + ASR (red) ────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        em_m, em_lo, em_hi = per_alpha_band(
            pa, "n_exact_match_clean", raw_alphas, idx, scale=1.0 / n_prompts,
        )
        asr_m, asr_lo, asr_hi = per_alpha_band(pa, "asr", raw_alphas, idx)
        draw(ax_r, disp_alphas, em_m,  em_lo,  em_hi,  GREEN, ls, mk,
             f"{name}  exact-match")
        draw(ax_r, disp_alphas, asr_m, asr_lo, asr_hi, RED,   ls, mk,
             f"{name}  ASR")

    ax_r.set_ylabel("exact-match rate / ASR")
    ax_r.set_xlabel(r"steering strength  $\alpha$")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_xticks(disp_alphas)
    ax_r.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_r.get_xticklabels(), rotation=45, ha="right")
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="center left", framealpha=0.95, edgecolor="#bbbbbb",
                fontsize=10)

    fig.suptitle(
        f"Seed {args.seed} alpha sweep  ·  "
        f"mean over {len(eval_seeds)} decode seeds, band = min–max",
        fontsize=14, y=1.00,
    )
    fig.tight_layout()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = f"{str(out).removesuffix('.png').removesuffix('.pdf')}{args.seed}"
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
