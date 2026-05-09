"""1×2 combined 50k figure — JSD (left) + layman rollout-level (right).

Only the 50k SAE row of the per-seed re-attributed sweep is rendered.

Left panel (distribution-level, "JSD"):
  green = JSD(steered, clean)     ↓ better
  red   = JSD(steered, poisoned)  ↑ better

Right panel (rollout-level, "layman"):
  green = clean-match rate (fraction of 200 prompts whose 16-token steered
          rollout matches the clean rollout word-for-word)  ↑ better
  red   = ASR-16 (fraction of prompts whose steered rollout contains the
          sleeper phrase)  ↓ better

Style matches `phase1_fra_plus_*` figures from the EM branch: Inter /
Helvetica sans-serif, hidden top/right spines, light-grey y-gridlines,
rounded "Unsteered baseline" annotation top-left of each panel.

Linestyle / marker:
  solid + circle    single OV → OV
  dashed + triangle conventional resid-mid additive

Output: writes <out>.png and <out>.pdf.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt


def setup_style():
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


def reduce_mean(values):
    if isinstance(values, list):
        return float(statistics.mean(values))
    return float(values)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, required=True,
                   help="JSON from jsd_2x2_sweep_saeseed.py with full metrics "
                        "(must include jsd_clean, jsd_pois, n_exact_match_clean, asr).")
    p.add_argument("--output", type=Path, required=True,
                   help="Path *without* extension; .png and .pdf are written.")
    p.add_argument("--n_prompts", type=int, default=200)
    p.add_argument("--title", default=None)
    args = p.parse_args()

    setup_style()
    data = json.loads(args.input.read_text())
    alphas = [float(a) for a in data["alphas"]]
    cfg = data["configs"]

    OV_KEY   = "ov_single_50k"
    CONV_KEY = "conventional_50k"

    # sleeper-plot palette — red/green preserves clean-vs-sleeper semantics
    GREEN = "#1a8a3f"
    RED   = "#c0322a"

    methods = [
        (OV_KEY,   r"single OV$\rightarrow$OV", "-",  "o"),
        (CONV_KEY, "conventional",              "--", "^"),
    ]

    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(14.0, 6.0))

    # ─────────────────── LEFT PANEL — JSD (distribution-level) ───────────────────
    for key, name, linestyle, marker in methods:
        per_alpha = cfg[key]["per_alpha"]
        jc = [reduce_mean(per_alpha[str(a)]["jsd_clean"]) for a in alphas]
        jp = [reduce_mean(per_alpha[str(a)]["jsd_pois"])  for a in alphas]
        ax_l.plot(alphas, jc, color=GREEN, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  JSD(steered, clean)")
        ax_l.plot(alphas, jp, color=RED, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  JSD(steered, poisoned)")

    ax_l.axhline(1.0, color="#888888", linestyle=":", lw=0.9, alpha=0.7)
    ax_l.text(alphas[-1], 1.0 - 0.015, "JSD upper bound (1 bit)",
               fontsize=10.5, color="#666", ha="right", va="top")
    ax_l.set_title("Distribution-level (JSD)", loc="center",
                    fontweight="bold", pad=14)
    ax_l.set_ylabel("Jensen-Shannon divergence (bits)")
    ax_l.set_xlabel(r"steering coefficient  $\alpha$")
    ax_l.set_ylim(-0.04, 1.10)
    ax_l.set_xticks(alphas)
    ax_l.set_xticklabels([f"{a:.2g}" for a in alphas])
    ax_l.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_l.set_axisbelow(True)
    ax_l.legend(loc="center right", framealpha=0.95, edgecolor="#bbbbbb")

    # baseline annotation: at α=0 with no steering, steered ≡ poisoned, so
    # JSD(steered, clean) = JSD(poisoned, clean) and JSD(steered, poisoned) = 0
    j0 = cfg[OV_KEY]["per_alpha"]["0.0"]
    base_pois_clean = reduce_mean(j0["jsd_clean"])
    ax_l.text(0.02, 0.97,
               f"Unsteered baseline (α=0)\nJSD(poisoned, clean) = {base_pois_clean:.2f} bits",
               transform=ax_l.transAxes, ha="left", va="top",
               fontsize=11, color="#444",
               bbox=dict(facecolor="white", edgecolor="#222", boxstyle="round,pad=0.4"))

    # ─────────────────── RIGHT PANEL — rollout (layman) ───────────────────
    for key, name, linestyle, marker in methods:
        per_alpha = cfg[key]["per_alpha"]
        match_rate = [reduce_mean(per_alpha[str(a)]["n_exact_match_clean"]) / args.n_prompts
                       for a in alphas]
        asr        = [reduce_mean(per_alpha[str(a)]["asr"]) for a in alphas]
        ax_r.plot(alphas, match_rate, color=GREEN, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  clean-match rate")
        ax_r.plot(alphas, asr, color=RED, lw=2.6, marker=marker, markersize=8,
                   linestyle=linestyle, markeredgecolor="white", markeredgewidth=0.9,
                   label=f"{name}  sleeper rate (ASR)")

    ax_r.set_title("Rollout-level (layman view)", loc="center",
                    fontweight="bold", pad=14)
    ax_r.set_ylabel("fraction of 200 deployment prompts")
    ax_r.set_xlabel(r"steering coefficient  $\alpha$")
    ax_r.set_ylim(-0.03, 1.05)
    ax_r.set_xticks(alphas)
    ax_r.set_xticklabels([f"{a:.2g}" for a in alphas])
    ax_r.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
    ax_r.set_axisbelow(True)
    ax_r.legend(loc="upper right", framealpha=0.95, edgecolor="#bbbbbb")

    base_match = reduce_mean(cfg[CONV_KEY]["per_alpha"]["0.0"]["n_exact_match_clean"]) / args.n_prompts
    base_asr   = reduce_mean(cfg[CONV_KEY]["per_alpha"]["0.0"]["asr"])
    ax_r.text(0.02, 0.97,
               f"Unsteered baseline (α=0)\nclean-match = {base_match*100:.1f}%, "
               f"sleeper rate = {base_asr*100:.1f}%",
               transform=ax_r.transAxes, ha="left", va="top",
               fontsize=11, color="#444",
               bbox=dict(facecolor="white", edgecolor="#222", boxstyle="round,pad=0.4"))

    if args.title:
        fig.suptitle(args.title, fontsize=15, y=1.00)

    fig.tight_layout()
    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out)
    if base.endswith(".png") or base.endswith(".pdf"):
        base = base.rsplit(".", 1)[0]
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"wrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
