"""1×2 figure (JSD + exact-match) at a single seed.

Left panel  — JSD(steered, clean) and JSD(steered, poisoned) vs α.
Right panel — exact-match-to-clean rate vs α.

Both methods (OV-only upstream and conventional resid-mid additive) on each panel.
No min/max bands (single seed).

Usage:
  python -m scripts.plot_jsd_exact_single_seed --seed 0
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path,
                   default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output", type=Path,
                   default=Path("figures/jsd_exact_seed"))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    setup_style()
    data        = json.loads(args.input.read_text())
    raw_alphas  = [float(a) for a in data["alphas"]]
    disp_alphas = [-a for a in raw_alphas]
    seeds       = data["sae_seeds"]
    n_prompts   = data["n_prompts"]
    cfg         = data["configs"]
    if args.seed not in seeds:
        raise SystemExit(f"seed {args.seed} not in results (have {seeds})")
    idx = seeds.index(args.seed)

    GREEN = "#1a8a3f"
    RED   = "#c0322a"
    BLUE  = "#2255bb"

    methods = [
        ("ov",           r"single OV$\rightarrow$OV", "-",  "o"),
        ("conventional", "conventional additive",      "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ── Left panel: JSD ──────────────────────────────────────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        jc = [pa[str(a)]["jsd_clean"][idx] for a in raw_alphas]
        jp = [pa[str(a)]["jsd_pois"][idx]  for a in raw_alphas]
        ax_l.plot(disp_alphas, jc, color=GREEN, lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=f"{name}  JSD(steered, clean)", zorder=3)
        ax_l.plot(disp_alphas, jp, color=RED, lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=f"{name}  JSD(steered, poisoned)", zorder=3)

    ax_l.axhline(1.0, color="#888", linestyle=":", lw=0.9, alpha=0.7)
    ax_l.text(disp_alphas[-1], 1.0 - 0.015, "JSD upper bound (1 bit)",
              fontsize=10.5, color="#666", ha="left", va="top")
    ax_l.set_ylabel("Jensen–Shannon divergence (bits)")
    ax_l.set_xlabel(r"steering strength  $-\alpha$  (negative = subtract SAE feature)")
    ax_l.set_ylim(-0.04, 1.10)
    ax_l.set_xticks(disp_alphas)
    ax_l.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_l.get_xticklabels(), rotation=45, ha="right")
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="lower left", framealpha=0.95, edgecolor="#bbbbbb")

    # ── Right panel: exact-match rate ────────────────────────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        em = [pa[str(a)]["n_exact_match_clean"][idx] / n_prompts for a in raw_alphas]
        ax_r.plot(disp_alphas, em, color=BLUE, lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=name, zorder=3)

    ax_r.set_ylabel("exact-match rate to clean rollout")
    ax_r.set_xlabel(r"steering strength  $-\alpha$  (negative = subtract SAE feature)")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_xticks(disp_alphas)
    ax_r.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_r.get_xticklabels(), rotation=45, ha="right")
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="upper right", framealpha=0.95, edgecolor="#bbbbbb")

    fig.suptitle(f"Seed {args.seed} alpha sweep", fontsize=15, y=1.00)
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
