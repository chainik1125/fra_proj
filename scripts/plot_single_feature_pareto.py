"""Plot the single-feature sleeper-tradeoff panel.

1×2 figure modeled on Ketan's `plot_flipped` headline:
  left  panel: x = Δcln-CE,    y = Sleepers removed (of N)
  right panel: x = Δgen-CE,    y = Sleepers removed (of N)

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
        --out docs/figures/single_feature_pareto.png
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
    ax.grid(True, axis="both", color="#d9d4c8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("results/single_feature_alpha_sweep.json"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/single_feature_pareto.png"))
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

    # "Sleepers removed (of N)" — total rollouts at eval = n_dep * n_eval_seeds.
    n_dep = (cfg.get("n_eval", 50)) // 2
    n_es  = len(eval_seeds) if eval_seeds is not None else 5
    total_rollouts = n_dep * n_es                              # e.g. 25 × 5 = 125
    base_hits      = base["asr"] * total_rollouts              # ≈ 121
    n_label        = int(round(base_hits))
    def _removed(asr: float) -> float:
        return (base["asr"] - asr) * total_rollouts

    # Color maps α → [0, 1] by equal-spaced index so steps are visible across
    # our irregular grid {0, 0.5, 1, 2, 4}.
    norm  = mcolors.Normalize(vmin=0, vmax=max(len(alphas) - 1, 1))
    cmap  = matplotlib.colormaps[args.cmap]
    color = {a: cmap(norm(i)) for i, a in enumerate(alphas)}

    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")
    panels = [("delta_ce",     "Δcln-CE  (clean teacher-forced cost)",         axes[0]),
              ("delta_gen_ce", "Δgen-CE  (story-prior CE on dep generation)",  axes[1])]

    for metric, xlabel, ax in panels:
        ax.set_facecolor("#fbfaf6")
        for family in families:
            marker = FAMILY_MARKER.get(family, "o")
            pts    = [pt for pt in points if pt.get("family") == family]
            xs     = [pt[metric] for pt in pts]
            ys     = [_removed(pt["asr"]) for pt in pts]
            cs     = [color[pt["alpha"]] for pt in pts]
            ax.scatter(xs, ys, s=80 if family == "downstream" else 60,
                       c=cs, marker=marker, edgecolor="#1b1b1b", linewidth=0.55,
                       alpha=0.92, zorder=3)
        ax.axvline(0.0, color="#27231f", linewidth=0.9, alpha=0.6)
        ax.axhline(0.0, color="#27231f", linewidth=0.9, alpha=0.6)
        ax.axhline(base_hits, color="#666", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.text(0.99, base_hits, f" full removal (= baseline {n_label})",
                ha="right", va="bottom", fontsize=8, color="#555",
                transform=ax.get_yaxis_transform())
        ax.set_xlabel(xlabel, fontsize=10.5)
        ax.set_ylabel(f"Sleepers removed (of {n_label})", fontsize=10.5)
        ax.set_ylim(-3, n_label + 4)
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
