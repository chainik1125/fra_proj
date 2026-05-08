"""Plot the 8-panel selection-method comparison figure.

Reads two `feature_set_pipeline.json` outputs (one per selection method) and
produces a 2×4 grid of panels:

    rows  = selection method (jamie / ketan)
    cols  = (single, gen-CE ratio) | (single, severity) | (set, gen-CE) | (set, severity)

Each panel:
    x = the named ratio (gen-CE ratio or severity ratio)
    y = sleeper-removed ratio = 1 − ASR
    color = α (steering strength), blue (low α) → red (high α)
    marker = upstream features (○) vs downstream baseline (◻)

All `--sae_seeds` from each input JSON are scattered as upstream points
(one per (seed, alpha) for set mode; one per (seed, top-1 feature, alpha)
for single mode). The downstream baseline is a single curve per panel.

Usage:
    python -m scripts.plot_pipeline_comparison \\
        --jamie_in results/experiment_jamie.json \\
        --ketan_in results/experiment_ketan.json \\
        --out docs/figures/pipeline_comparison.png
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


def _filter_points(points, *, eval_mode, family=None, selection_method=None):
    out = []
    for p in points:
        if family is not None and p.get("family") != family:
            continue
        if selection_method is not None and p.get("selection_method") != selection_method:
            continue
        # downstream points always carry eval_mode="single" — we still want them
        # in "set" panels, so don't filter downstream by eval_mode.
        if p.get("family") == "downstream":
            out.append(p)
            continue
        if p.get("eval_mode") != eval_mode:
            continue
        out.append(p)
    return out


def _plot_panel(ax, points, *, x_metric, alpha_color):
    """One panel. Scatter every point, marker by family, color by α."""
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

    # 1.0-reference vline at the "no damage" x-axis location.
    ax.axvline(1.0, color="#27231f", linewidth=0.8, alpha=0.5, linestyle=":")
    ax.set_ylim(-0.04, 1.04)
    _style(ax)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jamie_in", type=Path, required=True)
    p.add_argument("--ketan_in", type=Path, required=True)
    p.add_argument("--out",      type=Path, required=True,
                   help="Output figure path (.png; .pdf and .svg also written).")
    p.add_argument("--cmap",     default="coolwarm",
                   help="Matplotlib colormap for α (blue=low, red=high).")
    args = p.parse_args()

    payloads = {
        "jamie": json.loads(args.jamie_in.read_text()),
        "ketan": json.loads(args.ketan_in.read_text()),
    }
    # Pool all alphas seen across both files for a shared color scale.
    alphas = sorted({p["alpha"] for d in payloads.values() for p in d["points"]})
    norm = mcolors.Normalize(vmin=0, vmax=max(len(alphas) - 1, 1))
    cmap = matplotlib.colormaps[args.cmap]
    alpha_color = {a: cmap(norm(i)) for i, a in enumerate(alphas)}

    fig, axes = plt.subplots(2, 4, figsize=(18.0, 8.0), constrained_layout=True)
    fig.patch.set_facecolor("#fbfaf6")

    # Column definitions: (eval_mode, x_metric, x_label).
    cols = [
        ("single", "gen_ce_ratio",   "gen-CE ratio  (single feature)"),
        ("single", "severity_ratio", "severity ratio  (single feature)"),
        ("set",    "gen_ce_ratio",   "gen-CE ratio  (feature set)"),
        ("set",    "severity_ratio", "severity ratio  (feature set)"),
    ]
    rows = [("jamie", "Jamie selection"), ("ketan", "Ketan selection")]

    # Compute shared x-ranges per metric so single ↔ set panels share scale —
    # makes the visual comparison apples-to-apples. Pad the range by 5% of
    # span so points don't sit on the panel edge.
    metric_ranges: dict[str, tuple[float, float]] = {}
    for x_metric in ("gen_ce_ratio", "severity_ratio"):
        all_x = [p[x_metric] for d in payloads.values() for p in d["points"]
                 if p.get(x_metric) is not None]
        lo, hi = min(all_x), max(all_x)
        pad = 0.05 * max(hi - lo, 1e-3)
        metric_ranges[x_metric] = (lo - pad, hi + pad)

    for r, (selection, row_label) in enumerate(rows):
        payload = payloads[selection]
        points  = payload["points"]
        baseline_asr = payload["baseline"]["asr"]
        for c, (eval_mode, x_metric, xlabel) in enumerate(cols):
            ax = axes[r, c]
            ax.set_facecolor("#fbfaf6")
            sub = _filter_points(points, eval_mode=eval_mode,
                                 selection_method=selection)
            _plot_panel(ax, sub, x_metric=x_metric, alpha_color=alpha_color)
            ax.set_xlim(*metric_ranges[x_metric])              # shared x-range per metric
            # Baseline ASR reference (sleepers removed at no steering).
            ax.axhline(1.0 - baseline_asr, color="#666",
                       linestyle="--", linewidth=0.8, alpha=0.7)
            if r == 0:
                ax.set_title(xlabel, fontsize=11, weight="medium")
            if c == 0:
                ax.set_ylabel(f"{row_label}\n\nSleepers removed (1 − ASR)",
                              fontsize=10.5)
            if r == 1:
                ax.set_xlabel(xlabel.split("  ")[0], fontsize=10)

    # Markers legend (single column on the figure).
    legend_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", color="#888",
                   markeredgecolor="#1b1b1b", markeredgewidth=0.5, markersize=7,
                   label="Upstream (5 SAE seeds)"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#888",
                   markeredgecolor="#1b1b1b", markeredgewidth=0.7, markersize=8,
                   label="Downstream baseline (f579 @ resid_mid)"),
    ]
    axes[0, 0].legend(handles=legend_handles, frameon=False,
                       fontsize=8.5, loc="lower right")

    # α colorbar on the right side of the figure.
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.012, pad=0.015,
                        ticks=range(len(alphas)))
    cbar.ax.set_yticklabels([f"{a:g}" for a in alphas], fontsize=8.5)
    cbar.set_label("Steering strength α  (blue = weak, red = strong)",
                   fontsize=9)

    cfg = payloads["jamie"]["config"]
    eval_seeds = cfg.get("eval_seeds")
    sub = (f"sampled multi-seed eval (T={cfg.get('eval_temperature', 1.0)}, "
           f"eval_seeds={eval_seeds}); SAE seeds={cfg.get('sae_seeds')}; "
           f"top_k={cfg.get('top_k')}, single_top_n={cfg.get('single_top_n')}; "
           f"single = top-{cfg.get('single_top_n', 1)} feature per SAE seed; "
           f"set = full top-K steered together (sum-delta, OV-only on all heads)")
    fig.suptitle("Selection-method comparison: sleepers removed vs Generated×Clean ratios\n" + sub,
                 fontsize=12, weight="bold")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf", "svg"):
        fig.savefig(args.out.with_suffix(f".{ext}"), dpi=300)
    print(f"[plot] wrote {args.out}, {args.out.with_suffix('.pdf')}, {args.out.with_suffix('.svg')}")


if __name__ == "__main__":
    main()
