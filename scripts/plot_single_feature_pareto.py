"""Plot the single-feature sleeper-tradeoff panel.

1×2 figure modeled on Ketan's `paper_clean_ce_tradeoff.py` headline:
  left  panel: x = sampled eval ASR, y = Δcln-CE
  right panel: x = sampled eval ASR, y = Δgen-CE   (replaces Ketan's clean base-fidelity CE)

One scatter cloud — every (SAE seed, α) point from the α-sweep JSON.
Each seed gets a distinct marker; α is encoded in the size/edge so the
greedy/no-steer baseline at α=0 is recognisable.

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
import matplotlib.pyplot as plt

SEED_COLORS = ["#1f7a6d", "#b6422f", "#3f5fa3", "#a87c0a", "#8c3d8a"]
MARKER       = "o"


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", color="#d9d4c8", linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path,
                   default=Path("results/single_feature_alpha_sweep.json"))
    p.add_argument("--out", type=Path,
                   default=Path("docs/figures/single_feature_pareto.png"))
    args = p.parse_args()

    payload = json.loads(args.inp.read_text())
    points  = payload["points"]
    seeds   = sorted({pt["sae_seed"] for pt in points})
    alphas  = sorted({pt["alpha"] for pt in points})
    base    = payload["baseline"]
    decoding = payload.get("decoding", {})
    eval_seeds = (decoding.get("asr") or {}).get("seeds")
    eval_temp  = (decoding.get("asr") or {}).get("temperature")

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")

    panels = [
        ("delta_ce",     "Δcln-CE",                axes[0]),
        ("delta_gen_ce", "Δgen-CE  (story-prior CE on steered dep generation)", axes[1]),
    ]

    # α=0 has Δcln-CE = Δgen-CE = 0 by construction, so skip it from the cloud
    # but still show the baseline ASR on the figure.
    nonzero_pts = [pt for pt in points if pt["alpha"] != 0.0]

    for metric, ylabel, ax in panels:
        ax.set_facecolor("#fbfaf6")
        for i, s in enumerate(seeds):
            pts = sorted([pt for pt in nonzero_pts if pt["sae_seed"] == s],
                         key=lambda r: r["alpha"])
            xs = [pt["asr"] for pt in pts]
            ys = [pt[metric] for pt in pts]
            sizes = [40 + 60 * (alphas.index(pt["alpha"]) / max(len(alphas) - 1, 1))
                     for pt in pts]
            label = f"seed {s}  (f{pts[0]['feature']})" if pts else f"seed {s}"
            ax.scatter(xs, ys, s=sizes, color=SEED_COLORS[i % len(SEED_COLORS)],
                       marker=MARKER, edgecolor="#1b1b1b", linewidth=0.55,
                       alpha=0.92, label=label, zorder=3)

        # Baseline line at the unsteered ASR.
        ax.axvline(base["asr"], color="#999", linestyle="--", linewidth=0.9, alpha=0.7,
                   label=f"baseline ASR = {base['asr']:.3f}")
        ax.axhline(0.0, color="#27231f", linewidth=1.0, alpha=0.75)
        ax.set_xlabel("Sampled eval ASR  (lower = more sleeper suppression)", fontsize=10.5)
        ax.set_ylabel(ylabel, fontsize=10.5)
        ax.set_xlim(-0.04, 1.04)
        _style(ax)
    axes[0].legend(frameon=False, fontsize=8, loc="upper left")

    cfg = payload.get("config", {})
    n_eval = cfg.get("n_eval")
    n_gen  = cfg.get("n_gen_ce")
    sub = (f"single ov×ov winner per SAE seed; α∈{{{', '.join(f'{a:g}' for a in alphas)}}}; "
           f"sampled multi-seed eval (T={eval_temp}, seeds={eval_seeds})")
    if isinstance(n_eval, int):
        sub += f"; n_eval={n_eval} (½ dep), n_gen_ce={n_gen}"
    fig.suptitle("Sleeper suppression vs clean-side cost — single upstream features\n" + sub,
                 fontsize=12, weight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    print(f"[plot] wrote {args.out}, {args.out.with_suffix('.pdf')}, {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
