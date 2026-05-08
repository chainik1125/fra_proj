"""Plot the 2×2 Jamie-experiment figure.

Reads one `feature_set_pipeline.json` output (Jamie selection, top_k=20,
eval_mode=both, single_top_n=1) and produces a 2×2 grid:

    rows  = eval mode: top-1 single feature | top-20 feature set
    cols  = ratio metric: gen-CE ratio | severity ratio

Each panel:
    x = the named ratio metric
    y = sleeper-removed ratio = 1 − ASR
    color = α (steering strength), blue (low α) → red (high α)
    marker = upstream (○) vs downstream baseline (◻)

Usage:
    python -m scripts.plot_jamie_experiment \\
        --in results/jamie_top1_vs_set.json \\
        --out figures/jamie_experiment.png
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


def _style(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="both", color="#d9d4c8", linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)


def _filter_points(points, *, eval_mode):
    out = []
    for p in points:
        if p.get("family") == "downstream":
            out.append(p)
            continue
        if p.get("eval_mode") == eval_mode:
            out.append(p)
    return out


def _plot_panel(ax, points, *, x_metric, alpha_color):
    upstream = [p for p in points if p["family"] == "upstream"]
    downstream = [p for p in points if p["family"] == "downstream"]

    if upstream:
        ax.scatter(
            [p[x_metric] for p in upstream],
            [1.0 - p["asr"] for p in upstream],
            c=[alpha_color[p["alpha"]] for p in upstream],
            marker=FAMILY_MARKER["upstream"],
            s=42, edgecolor="#1b1b1b", linewidth=0.5, alpha=0.85, zorder=3,
        )
    if downstream:
        ax.scatter(
            [p[x_metric] for p in downstream],
            [1.0 - p["asr"] for p in downstream],
            c=[alpha_color[p["alpha"]] for p in downstream],
            marker=FAMILY_MARKER["downstream"],
            s=80, edgecolor="#1b1b1b", linewidth=0.7, alpha=0.95, zorder=4,
        )

    ax.axvline(1.0, color="#27231f", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_ylim(-0.04, 1.04)
    _style(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in",  type=Path, required=True, dest="input")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--cmap", default="coolwarm")
    args = p.parse_args()

    payload = json.loads(args.input.read_text())
    points = payload["points"]
    cfg = payload["config"]
    baseline_asr = payload["baseline"]["asr"]

    alphas = sorted({p["alpha"] for p in points})
    norm = mcolors.Normalize(vmin=0, vmax=max(len(alphas) - 1, 1))
    cmap = matplotlib.colormaps[args.cmap]
    alpha_color = {a: cmap(norm(i)) for i, a in enumerate(alphas)}

    # Shared x-ranges per metric.
    metric_ranges: dict[str, tuple[float, float]] = {}
    for x_metric in ("gen_ce_ratio", "severity_ratio"):
        all_x = [p[x_metric] for p in points if p.get(x_metric) is not None]
        lo, hi = min(all_x), max(all_x)
        pad = 0.05 * max(hi - lo, 1e-3)
        metric_ranges[x_metric] = (lo - pad, hi + pad)

    fig, axes = plt.subplots(2, 2, figsize=(10.0, 8.0), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")

    rows = [
        ("single", f"Top-1 single feature\n(rank #1 from Jamie selection)"),
        ("set",    f"Top-{cfg.get('top_k', 20)} feature set\n(all features steered together)"),
    ]
    cols = [
        ("gen_ce_ratio",   "gen-CE ratio"),
        ("severity_ratio", "severity ratio"),
    ]

    for r, (eval_mode, row_label) in enumerate(rows):
        sub = _filter_points(points, eval_mode=eval_mode)
        for c, (x_metric, col_label) in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor("#fbfaf6")
            _plot_panel(ax, sub, x_metric=x_metric, alpha_color=alpha_color)
            ax.set_xlim(*metric_ranges[x_metric])
            ax.axhline(1.0 - baseline_asr, color="#666",
                       linestyle="--", linewidth=0.8, alpha=0.7)
            if r == 0:
                ax.set_title(col_label, fontsize=12, weight="medium")
            if c == 0:
                ax.set_ylabel(f"{row_label}\n\nSleepers removed (1 − ASR)", fontsize=10)
            if r == 1:
                ax.set_xlabel(col_label, fontsize=10)

    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#888",
                   markeredgecolor="#1b1b1b", markeredgewidth=0.5, markersize=7,
                   label=f"Upstream OV-only (5 SAE seeds)"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#888",
                   markeredgecolor="#1b1b1b", markeredgewidth=0.7, markersize=8,
                   label="Downstream baseline (f579 @ resid_mid)"),
        plt.Line2D([0], [0], linestyle="--", color="#666", linewidth=0.8,
                   label="Baseline ASR (no steering)"),
    ]
    axes[0, 0].legend(handles=legend_handles, frameon=False, fontsize=8.5, loc="lower right")

    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.015, ticks=range(len(alphas)))
    cbar.ax.set_yticklabels([f"{a:g}" for a in alphas], fontsize=8.5)
    cbar.set_label("Steering strength α  (blue = weak, red = strong)", fontsize=9)

    suptitle_sub = (
        f"Jamie selection · top_k={cfg.get('top_k')} · single_top_n={cfg.get('single_top_n')} · "
        f"SAE seeds={cfg.get('sae_seeds')} · "
        f"eval T={cfg.get('eval_temperature', 1.0)} seeds={cfg.get('eval_seeds')}"
    )
    fig.suptitle(
        "Jamie experiment: sleepers removed vs Generated×Clean ratios\n" + suptitle_sub,
        fontsize=11, weight="bold",
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    print(f"[plot] wrote {args.out} (+ .pdf, .svg)")


if __name__ == "__main__":
    main()
