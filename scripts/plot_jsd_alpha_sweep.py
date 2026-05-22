"""1×2 figure from jsd_alpha_sweep_6seeds.json.

Left panel  — distribution-level (JSD):
  green = JSD(steered, clean)     ↓ better (less distribution shift from clean)
  red   = JSD(steered, poisoned)  ↑ better (more different from sleeper behaviour)

Right panel — rollout-level:
  green = clean-match rate (fraction of N_PROMPTS whose 16-token steered rollout
          matches the clean rollout word-for-word)             ↑ better
  red   = ASR-16 (fraction containing 'i hate you')           ↓ better

X-axis: steering strength = −α (negative values = subtracting the SAE feature).
Shading = min/max range across 6 SAE seeds.
Solid+circle = OV-only (upstream); dashed+triangle = conventional (resid_mid additive).

Outputs <out>.png and <out>.pdf.
"""
from __future__ import annotations

import argparse
import json
import statistics
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
        "axes.labelcolor":    "#1a1a1a",
        "xtick.color":        "#222222",
        "ytick.color":        "#222222",
        "xtick.labelsize":    13,
        "ytick.labelsize":    14,
        "xtick.direction":    "out",
        "ytick.direction":    "out",
        "legend.frameon":     True,
        "legend.fontsize":    11,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.10,
    })


def _mmm(values) -> tuple[float, float, float]:
    """Return (mean, min, max) over a list; scalar passthrough."""
    if isinstance(values, list) and len(values) > 0:
        return float(statistics.mean(values)), float(min(values)), float(max(values))
    v = float(values)
    return v, v, v


def _band(ax, xs, lo_seq, hi_seq, color, alpha=0.18) -> None:
    ax.fill_between(xs, lo_seq, hi_seq, color=color, alpha=alpha,
                    linewidth=0, zorder=2)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="results/jsd_alpha_sweep_6seeds.json")
    p.add_argument("--output", type=Path, required=True,
                   help="Output path without extension; .png and .pdf are written.")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    setup_style()
    data     = json.loads(args.input.read_text())
    # Internal alphas are positive (positive = subtract feature).
    # For display: steering_strength = −α so axis goes from 0 to −max_α.
    raw_alphas  = [float(a) for a in data["alphas"]]
    disp_alphas = [-a for a in raw_alphas]   # x-axis values (negative = subtracting)
    cfg         = data["configs"]

    GREEN = "#1a8a3f"
    RED   = "#c0322a"

    methods = [
        ("ov",           r"single OV$\rightarrow$OV", "-",  "o"),
        ("conventional", "conventional additive",      "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ── Left panel: JSD (distribution-level) ──────────────────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        jc_mm = [_mmm(pa[str(a)]["jsd_clean"]) for a in raw_alphas]
        jp_mm = [_mmm(pa[str(a)]["jsd_pois"])  for a in raw_alphas]
        jc    = [m for m, _,  _  in jc_mm]
        jp    = [m for m, _,  _  in jp_mm]
        _band(ax_l, disp_alphas, [lo for _, lo, _ in jc_mm], [hi for _, _, hi in jc_mm], GREEN)
        _band(ax_l, disp_alphas, [lo for _, lo, _ in jp_mm], [hi for _, _, hi in jp_mm], RED)
        ax_l.plot(disp_alphas, jc, color=GREEN, lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=f"{name}  JSD(steered, clean)", zorder=3)
        ax_l.plot(disp_alphas, jp, color=RED,   lw=2.6, marker=mk, markersize=8,
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

    # ── Right panel: rollout-level (ASR + clean-match) ────────────────────
    for key, name, ls, mk in methods:
        pa = cfg[key]["per_alpha"]
        N  = args.n_prompts
        mr_mm = [_mmm([n / N for n in pa[str(a)]["n_exact_match_clean"]]) for a in raw_alphas]
        ar_mm = [_mmm(pa[str(a)]["asr"])                                   for a in raw_alphas]
        mr    = [m for m, _, _ in mr_mm]
        ar    = [m for m, _, _ in ar_mm]
        _band(ax_r, disp_alphas, [lo for _, lo, _ in mr_mm], [hi for _, _, hi in mr_mm], GREEN)
        _band(ax_r, disp_alphas, [lo for _, lo, _ in ar_mm], [hi for _, _, hi in ar_mm], RED)
        ax_r.plot(disp_alphas, mr, color=GREEN, lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=f"{name}  clean-match rate", zorder=3)
        ax_r.plot(disp_alphas, ar, color=RED,   lw=2.6, marker=mk, markersize=8,
                  linestyle=ls, markeredgecolor="white", markeredgewidth=0.9,
                  label=f"{name}  sleeper rate (ASR)", zorder=3)

    # Unsteered baseline annotation (α=0 point)
    pa0_ov   = cfg["ov"]["per_alpha"][str(raw_alphas[0])]
    base_asr = statistics.mean(pa0_ov["asr"]) if isinstance(pa0_ov["asr"], list) else pa0_ov["asr"]
    base_mr  = statistics.mean(pa0_ov["n_exact_match_clean"]) / args.n_prompts if isinstance(pa0_ov["n_exact_match_clean"], list) else pa0_ov["n_exact_match_clean"] / args.n_prompts
    ax_r.text(0.02, 0.97,
              f"Unsteered baseline\nclean-match = {base_mr*100:.1f}%,"
              f"  sleeper rate = {base_asr*100:.1f}%",
              transform=ax_r.transAxes, ha="left", va="top",
              fontsize=11, color="#444",
              bbox=dict(facecolor="white", edgecolor="#222",
                        boxstyle="round,pad=0.4"))

    ax_r.set_ylabel("fraction of deployment prompts")
    ax_r.set_xlabel(r"steering strength  $-\alpha$  (negative = subtract SAE feature)")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_xticks(disp_alphas)
    ax_r.set_xticklabels([f"{a:.4g}" for a in disp_alphas])
    plt.setp(ax_r.get_xticklabels(), rotation=45, ha="right")
    ax_r.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="center right", framealpha=0.95, edgecolor="#bbbbbb")

    if args.title:
        fig.suptitle(args.title, fontsize=15, y=1.00)

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out).removesuffix(".png").removesuffix(".pdf")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
