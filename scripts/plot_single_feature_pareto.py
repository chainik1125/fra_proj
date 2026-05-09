"""Plot the single-feature sleeper-tradeoff panel.

1×3 figure with one panel per Generated × Clean cell metric, plus the
teacher-forced Δcln-CE on the left:

  panel 1 (left):   x = Δcln-CE,        y = Sampled eval ASR
  panel 2 (middle): x = gen-CE ratio,   y = Sampled eval ASR
  panel 3 (right):  x = recovery noise ratio, y = Sampled eval ASR

Both ratios share the unitless multiplicative interpretation: 1.0 means
"steered output is indistinguishable from the natural reference"; >1 means
the intervention pushes the output further from the reference than the
reference's own internal noise floor. They differ in *what* they compare:
gen-CE ratio scores produced tokens against a natural-baseline rollout's
NLL; recovery noise ratio compares per-step generation distributions against
a multi-seed sampling-noise baseline (see `sleeper.metrics.recovery_noise_ratio`).

Two families on the same axes:
  * "upstream"   — per-seed ov×ov winner (ln1 SAE feature, OV-only intervention)
                   Marker: circle.
  * "downstream" — single resid_mid feature directly ablated at hook_resid_mid
                   Marker: square.

Color encodes steering strength α (blue = weak / no steer, red = strong).
A single seaborn-style colorbar replaces the per-seed legend.

Usage:
    python -m scripts.plot_single_feature_pareto \\
        --in results/single_feature_alpha_sweep.json \\
        --out figures/single_feature_pareto.png
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt

FAMILY_MARKER = {"upstream": "o", "downstream": "s"}
FAMILY_LABEL  = {"upstream":   "Upstream single feature (ov×ov winner per SAE seed)",
                 "downstream": "Downstream single feature (f579 @ resid_mid)"}


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#cccccc", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("results/single_feature_alpha_sweep.json"))
    p.add_argument("--out", type=Path,
                   default=Path("figures/single_feature_pareto.png"))
    p.add_argument("--cmap", default="coolwarm",
                   help="matplotlib colormap (blue=low α, red=high α). 'coolwarm' or 'RdBu_r'.")
    args = p.parse_args()

    payload  = json.loads(args.inp.read_text())
    points   = payload["points"]
    base     = payload["baseline"]
    decoding = payload.get("decoding", {})
    eval_seeds = (decoding.get("asr") or {}).get("seeds")
    eval_temp  = (decoding.get("asr") or {}).get("temperature")
    cfg = payload.get("config", {})

    alphas    = sorted({pt["alpha"] for pt in points})
    families  = sorted({pt["family"] for pt in points if pt.get("family")},
                       key=lambda f: 0 if f == "upstream" else 1)

    # Color maps α → [0, 1] by equal-spaced index so steps are visible across
    # our irregular grid {0, 0.5, 1, 2, 4}.
    norm  = mcolors.Normalize(vmin=0, vmax=max(len(alphas) - 1, 1))
    cmap  = matplotlib.colormaps[args.cmap]
    color = {a: cmap(norm(i)) for i, a in enumerate(alphas)}

    fig, axes = plt.subplots(1, 3, figsize=(17.5, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("white")
    panels = [
        ("delta_ce",       "Δcln-CE  (clean teacher-forced cost)",                       0.0, axes[0]),
        ("gen_ce_ratio",   "gen-CE ratio  (NLL_steered / NLL_baseline, ≥1 = damage)",    1.0, axes[1]),
        ("recovery_noise_ratio", "recovery noise ratio  (CE_steered / CE_sampling_noise, ≥1 = damage)", 1.0, axes[2]),
    ]

    for metric, xlabel, ref_x, ax in panels:
        ax.set_facecolor("white")
        for family in families:
            marker = FAMILY_MARKER.get(family, "o")
            pts    = [pt for pt in points if pt.get("family") == family]
            xs     = [pt[metric] for pt in pts]
            ys     = [pt["asr"] for pt in pts]
            cs     = [color[pt["alpha"]] for pt in pts]
            ax.scatter(xs, ys, s=80 if family == "downstream" else 60,
                       c=cs, marker=marker, edgecolor="#1b1b1b", linewidth=0.55,
                       alpha=0.92, zorder=3)
        ax.axvline(ref_x, color="#27231f", linewidth=0.9, alpha=0.6)
        ax.axhline(base["asr"], color="#666", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(0.99, base["asr"], f" baseline ASR = {base['asr']:.3f}",
                ha="right", va="bottom", fontsize=8, color="#555",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel("Sampled eval ASR  (lower = more sleeper suppression)", fontsize=10.5)
        ax.set_ylim(-0.04, 1.04)
        _style(ax)

    # Family legend (markers, neutral grey).
    legend_handles = []
    for family in families:
        legend_handles.append(plt.Line2D(
            [0], [0], marker=FAMILY_MARKER.get(family, "o"), linestyle="",
            color="#888", markeredgecolor="#1b1b1b", markeredgewidth=0.55,
            markersize=8 if family == "downstream" else 7,
            label=FAMILY_LABEL.get(family, family),
        ))
    axes[0].legend(handles=legend_handles, frameon=False, fontsize=8.5, loc="lower right")

    # α colorbar — uses the index-based normalisation but tick-labels show the actual α.
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.024, pad=0.02, ticks=range(len(alphas)))
    cbar.ax.set_yticklabels([f"{a:g}" for a in alphas], fontsize=8.5)
    cbar.set_label("Steering strength α  (blue = weak, red = strong)", fontsize=9)

    sub = (f"sampled multi-seed eval (T={eval_temp}, seeds={eval_seeds}); "
           f"upstream: 5 SAE seeds × per-seed ov×ov winner; "
           f"downstream: single feature f579 at resid_mid")
    fig.suptitle("Sleeper suppression vs clean-side cost — single features\n" + sub,
                 fontsize=12, weight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    print(f"[plot] wrote {args.out}, {args.out.with_suffix('.pdf')}, {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
