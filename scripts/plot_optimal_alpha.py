"""Optimal-alpha analysis: best coherence subject to ASR=0, per seed.

For each (seed, method):
  If any alpha achieves ASR=0.0:
    optimal = the one among those with the lowest JSD(steered, clean).
  Else (failure mode):
    optimal = alpha with the lowest ASR.

Produces a 3-row × 2-col figure:
  rows — JSD(steered, clean) | JSD(steered, poisoned) | exact-match rate
  cols — OV-only (upstream)  | conventional (resid-mid additive)

Each panel: one bar per seed (0-5) + two summary bars:
  "mean (all 6)"      — average over all seeds including failure modes
  "mean (ASR=0)"      — average over seeds that achieved ASR=0 only

Failure-mode seeds (never hit ASR=0) are shown with diagonal hatching.

Also prints a plain-text table of optimal alpha + all three metrics per seed.
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":        "sans-serif",
        "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":          13,
        "axes.titlesize":     14,
        "axes.labelsize":     13,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     1.1,
        "axes.edgecolor":     "#222222",
        "xtick.labelsize":    11,
        "ytick.labelsize":    11,
        "legend.fontsize":    10,
        "figure.dpi":         110,
        "savefig.bbox":       "tight",
        "savefig.pad_inches": 0.12,
    })


def find_optimal(per_alpha: dict, alphas: list[float], seed_idx: int,
                 n_prompts: int) -> dict:
    """Find the optimal alpha for one (seed, method) pair.

    Returns a dict with keys: alpha, asr, jsd_clean, jsd_pois, exact_match,
    and is_failure (True if ASR=0 was never achieved).
    """
    rows = []
    for a in alphas:
        key     = str(a)
        asr     = per_alpha[key]["asr"][seed_idx]
        jc      = per_alpha[key]["jsd_clean"][seed_idx]
        jp      = per_alpha[key]["jsd_pois"][seed_idx]
        n_exact = per_alpha[key]["n_exact_match_clean"][seed_idx]
        rows.append((a, asr, jc, jp, n_exact / n_prompts))

    zero_asr = [(a, asr, jc, jp, em) for (a, asr, jc, jp, em) in rows if asr == 0.0]
    if zero_asr:
        best = min(zero_asr, key=lambda r: r[2])   # lowest jsd_clean
        return dict(alpha=best[0], asr=best[1], jsd_clean=best[2],
                    jsd_pois=best[3], exact_match=best[4], is_failure=False)
    else:
        best = min(rows, key=lambda r: (r[1], r[2]))   # lowest asr then jsd_clean
        return dict(alpha=best[0], asr=best[1], jsd_clean=best[2],
                    jsd_pois=best[3], exact_match=best[4], is_failure=True)


def summarise(optima: list[dict]) -> dict[str, dict]:
    """Compute mean of each metric for all seeds and ASR=0 seeds only."""
    success = [o for o in optima if not o["is_failure"]]
    def means(subset):
        if not subset:
            return dict(jsd_clean=float("nan"), jsd_pois=float("nan"),
                        exact_match=float("nan"))
        return {m: statistics.mean(o[m] for o in subset)
                for m in ("jsd_clean", "jsd_pois", "exact_match")}
    return {"all": means(optima), "asr0": means(success)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input",    type=Path, default=Path("results/jsd_alpha_sweep_6seeds.json"))
    p.add_argument("--output",   type=Path, default=Path("figures/optimal_alpha"))
    args = p.parse_args()

    data      = json.loads(args.input.read_text())
    alphas    = [float(a) for a in data["alphas"]]
    seeds     = data["sae_seeds"]
    n_prompts = data["n_prompts"]
    cfg       = data["configs"]

    methods = [("ov", r"OV-only (upstream)"), ("conventional", "conventional (resid-mid)")]

    # ── Compute optima ────────────────────────────────────────────────────────
    optima: dict[str, list[dict]] = {}
    for key, _ in methods:
        pa = cfg[key]["per_alpha"]
        optima[key] = [find_optimal(pa, alphas, i, n_prompts)
                       for i in range(len(seeds))]

    # ── Print table ───────────────────────────────────────────────────────────
    metrics = [("jsd_clean",   "JSD(strd,clean)"),
               ("jsd_pois",    "JSD(strd,pois) "),
               ("exact_match", "exact-match    ")]

    for key, label in methods:
        print(f"\n{'─'*70}")
        print(f"  {label}")
        print(f"{'─'*70}")
        header = f"  {'seed':>4}  {'opt-α':>5}  {'ASR':>6}  " + \
                 "  ".join(f"{mlab:>15}" for _, mlab in metrics) + "  status"
        print(header)
        for i, s in enumerate(seeds):
            o = optima[key][i]
            vals = "  ".join(f"{o[mk]:>15.4f}" for mk, _ in metrics)
            flag = " ← FAILURE (ASR>0)" if o["is_failure"] else ""
            print(f"  {s:>4}  {o['alpha']:>5.1f}  {o['asr']:>6.3f}  {vals}{flag}")

        summ = summarise(optima[key])
        n_ok = sum(1 for o in optima[key] if not o["is_failure"])
        print(f"\n  mean (all 6 seeds): " +
              "  ".join(f"{summ['all'][mk]:.4f}" for mk, _ in metrics))
        print(f"  mean (ASR=0 seeds, n={n_ok}): " +
              "  ".join(f"{summ['asr0'][mk]:.4f}" for mk, _ in metrics))

    # ── Plot ──────────────────────────────────────────────────────────────────
    setup_style()

    metric_info = [
        ("jsd_clean",   "JSD(steered, clean)",    "#1a8a3f", "↓ better"),
        ("jsd_pois",    "JSD(steered, poisoned)",  "#c0322a", "↑ better"),
        ("exact_match", "exact-match rate",         "#2255bb", "↑ better"),
    ]

    fig, axes = plt.subplots(3, 2, figsize=(12.0, 10.0), sharey="row")

    x_seeds   = np.arange(len(seeds))
    x_all     = len(seeds)
    x_asr0    = len(seeds) + 1
    x_total   = len(seeds) + 2
    bar_w     = 0.65
    x_ticks   = list(range(x_total))
    x_labels  = [f"s{s}" for s in seeds] + ["mean\n(all)", "mean\n(ASR=0)"]

    for col, (key, method_label) in enumerate(methods):
        pa   = cfg[key]["per_alpha"]
        opts = optima[key]
        summ = summarise(opts)

        for row, (mk, metric_label, color, direction) in enumerate(metric_info):
            ax = axes[row][col]

            # Per-seed bars
            for i, s in enumerate(seeds):
                o       = opts[i]
                height  = o[mk]
                hatch   = "//" if o["is_failure"] else None
                ec      = "#888888" if o["is_failure"] else "white"
                bar = ax.bar(i, height, width=bar_w, color=color, alpha=0.85,
                             hatch=hatch, edgecolor=ec, linewidth=0.8, zorder=3)

            # Summary bars
            v_all  = summ["all"][mk]
            v_asr0 = summ["asr0"][mk]
            ax.bar(x_all,  v_all,  width=bar_w, color=color, alpha=0.45,
                   edgecolor="#333", linewidth=1.0, zorder=3)
            ax.bar(x_asr0, v_asr0, width=bar_w, color=color, alpha=0.85,
                   edgecolor="#333", linewidth=1.0, zorder=3)

            # Value labels on summary bars
            for x, v in [(x_all, v_all), (x_asr0, v_asr0)]:
                if not (v != v):   # skip NaN
                    ax.text(x, v + 0.015, f"{v:.3f}", ha="center", va="bottom",
                            fontsize=9, color="#333")

            # Styling
            ax.set_xticks(x_ticks)
            ax.set_xticklabels(x_labels)
            ax.set_ylim(0, 1.12)
            ax.grid(True, axis="y", color="#eeeeee", lw=0.6, zorder=0)
            ax.set_axisbelow(True)

            if col == 0:
                ax.set_ylabel(f"{metric_label}  ({direction})")
            if row == 0:
                ax.set_title(method_label, fontweight="bold", pad=10)

            # Vertical separator before summary bars
            ax.axvline(len(seeds) - 0.5, color="#bbbbbb", lw=0.8, linestyle="--")

    # Shared legend for hatch meaning
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor="#999", edgecolor="white", label="success (ASR=0 achieved)"),
        Patch(facecolor="#999", edgecolor="#888", hatch="//",
              label="failure (ASR>0 at all α)"),
        Patch(facecolor="#bbb", alpha=0.45, edgecolor="#333",
              label="mean — all 6 seeds"),
        Patch(facecolor="#bbb", alpha=0.85, edgecolor="#333",
              label="mean — ASR=0 seeds only"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=4,
               frameon=True, framealpha=0.95, edgecolor="#bbbbbb",
               fontsize=10, bbox_to_anchor=(0.5, -0.01))

    fig.suptitle("Optimal steering: best coherence at ASR=0 per seed\n"
                 "(hatched = failure mode, used lowest-ASR alpha instead)",
                 fontsize=13, y=1.01)
    fig.tight_layout()

    out = Path(args.output).expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    base = str(out).removesuffix(".png").removesuffix(".pdf")
    fig.savefig(base + ".png", dpi=200)
    fig.savefig(base + ".pdf")
    plt.close(fig)
    print(f"\nwrote {base}.png and {base}.pdf")


if __name__ == "__main__":
    main()
