"""Appendix figure: TinySleepers six-seed wide-coefficient screen (combined_50k).

By default plots the retrained six-seed layer-0 SAEs (key retrain_L0 of
data/tinystories/wide_screen_layers.json). Pass --layers-key '' to plot the earlier
Modal-trained SAE screen archived in data/tinystories/wide_screen_modal_sae.json instead
(that archive can be rebuilt from raw sweep output with --source-dir). Each seed's curve
uses the feature selected by its full-fidelity Bayesian-optimization result. Curve
values themselves are the 64-prompt, decode-seed-0 screen measurements; the BO
optima are not substituted into the curves.

Inputs : data/tinystories/wide_screen_layers.json (or wide_screen_modal_sae.json)
Outputs: figures/combined_50k.{pdf,png}

    uv run scripts/plot_tinysleeper_wide_screen.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, MultipleLocator, PercentFormatter


from _paths import DATA as _DATA, FIGURES

DATA = _DATA / "tinystories" / "wide_screen_modal_sae.json"
LAYERS_DATA = _DATA / "tinystories" / "wide_screen_layers.json"
OUTPUT = FIGURES / "combined_50k.pdf"
SCHEMES = (
    ("ov", "single OV $\\to$ OV", "-", "o"),
    ("conventional", "conventional resid-mid", "--", "^"),
    ("conv_ln1", "conventional ln1", ":", "s"),
)
METRICS = ("jsd_clean", "jsd_pois", "clean_match", "asr")
GREEN = "#168144"
RED = "#c2392e"


def export_data(source: Path, destination: Path) -> None:
    cells: dict[str, dict[str, object]] = {}
    alphas: list[float] | None = None
    for scheme, _, _, _ in SCHEMES:
        cells[scheme] = {}
        for seed in range(6):
            bo = json.loads((source / f"bo_{scheme}_s{seed}.json").read_text())
            feat = int(bo["feat"])
            rows = []
            screen_meta = None
            for chunk in range(5):
                screen = json.loads((source / f"screen_{scheme}_s{seed}_c{chunk}.json").read_text())
                screen_meta = screen
                rows.extend(row for row in screen["rows"] if row["feat"] == feat)
            assert screen_meta is not None
            assert screen_meta["n_prompts"] == 64
            assert screen_meta["screen_decode_seed"] == 0
            rows.sort(key=lambda row: row["alpha"])
            assert len(rows) == 81, (scheme, seed, feat, len(rows))
            current_alphas = [float(row["alpha"]) for row in rows]
            if alphas is None:
                alphas = current_alphas
            assert current_alphas == alphas
            cells[scheme][str(seed)] = {
                "feature": feat,
                "bo_opt_alpha": bo["opt_alpha"],
                "bo_opt_jsd_clean": bo["opt_jsd_clean"],
                "screen": [
                    [
                        round(float(row["jsd_clean"]), 8),
                        round(float(row["jsd_pois"]), 8),
                        round(float(row["exact"]) / 64.0, 8),
                        round(float(row["asr"]), 8),
                    ]
                    for row in rows
                ],
            }
    assert alphas is not None
    assert len(alphas) == 81 and alphas[0] == -10 and alphas[-1] == 10
    payload = {
        "description": "TinyStories wide-alpha redo, July 2026; BO-selected feature per SAE seed",
        "source_commit": "internal sweep branch",
        "source_path": "results/alpha_redo/results",
        "screen_prompts": 64,
        "screen_decode_seed": 0,
        "sae_seeds": list(range(6)),
        "raw_alpha_sign": "positive subtracts feature; plotted alpha = -raw alpha",
        "raw_alphas": alphas,
        "metric_order": list(METRICS),
        "cells": cells,
    }
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, separators=(",", ":")) + "\n")


def series(data: dict, scheme: str, metric: str) -> tuple[np.ndarray, np.ndarray]:
    index = METRICS.index(metric)
    values = np.array(
        [[row[index] for row in data["cells"][scheme][str(seed)]["screen"]]
         for seed in data["sae_seeds"]],
        dtype=float,
    )
    return values.mean(axis=0), np.stack((values.min(axis=0), values.max(axis=0)))


def render(data: dict, output: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 12,
        "axes.labelsize": 15,
        "xtick.labelsize": 11,
        "ytick.labelsize": 12,
        "legend.fontsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    raw = np.array(data["raw_alphas"], dtype=float)
    x = -raw  # Preserve Fig. 3's convention: subtraction is negative alpha.
    order = np.argsort(x)
    x = x[order]
    fig, axes = plt.subplots(1, 2, figsize=(13.75, 5.75))
    fig.subplots_adjust(left=0.065, right=0.985, bottom=0.16, top=0.96, wspace=0.19)

    panel_metrics = (("jsd_clean", "jsd_pois"), ("clean_match", "asr"))
    for panel_index, (ax, pair) in enumerate(zip(axes, panel_metrics)):
        for scheme, label, linestyle, marker in SCHEMES:
            for metric, color, ref in zip(pair, (GREEN, RED), ("clean", "poisoned")):
                mean, band = series(data, scheme, metric)
                mean = mean[order]
                low, high = band[:, order]
                ax.fill_between(x, low, high, color=color, alpha=0.085, linewidth=0)
                if metric.startswith("jsd"):
                    legend = f"{label}  JSD(steered, {ref})"
                else:
                    legend = f"{label}  {'clean-match rate' if metric == 'clean_match' else 'sleeper rate (ASR)'}"
                ax.plot(x, mean, color=color, linestyle=linestyle, linewidth=2.2,
                        marker=marker, markersize=5.2, markevery=4,
                        markeredgecolor="white", markeredgewidth=0.8,
                        label=legend, zorder=3)
        ax.set_xlim(10.35, -10.35)
        ax.set_xticks([10, 7.5, 5, 2.5, 0, -2.5, -5, -7.5, -10])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda val, _: f"{val:g}"))
        ax.set_xlabel(r"steering coefficient  $\alpha$")
        ax.grid(axis="y", color="#dddddd", linewidth=0.55, alpha=0.7)
        ax.set_axisbelow(True)
        ax.legend(loc="lower left" if panel_index == 0 else "upper left",
                  framealpha=0.97, facecolor="white",
                  edgecolor="#bcbcbc", handlelength=2.5, borderpad=0.55,
                  labelspacing=0.35)

    axes[0].set_ylabel("Jensen-Shannon divergence (bits)")
    axes[0].set_ylim(-0.025, 1.04)
    axes[0].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[0].axhline(1.0, color="#999999", linestyle=":", linewidth=0.9)
    axes[0].text(-9.8, 0.985, "JSD upper bound (1 bit)", color="#666666",
                 fontsize=10, ha="right", va="top")
    axes[1].set_ylabel("Sleeper fraction / Word-word matches")
    axes[1].set_ylim(-0.025, 1.04)
    axes[1].yaxis.set_major_locator(MultipleLocator(0.2))
    axes[1].yaxis.set_major_formatter(PercentFormatter(1.0))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(output.with_suffix(".png"), dpi=200, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, help="Original alpha_redo/results directory")
    parser.add_argument("--data", type=Path, default=DATA)
    parser.add_argument("--layers-key", default="retrain_L0",
                        help="plot this SAE set from wide_screen_layers.json (the retrained "
                             "six-seed SAEs by default); pass '' to plot --data instead")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    if args.source_dir:
        export_data(args.source_dir, args.data)
    if args.layers_key:
        layers = json.loads(LAYERS_DATA.read_text())
        data = {"raw_alphas": layers["raw_alphas"], "sae_seeds": layers["sae_seeds"],
                "cells": layers["layers"][args.layers_key]}
    else:
        data = json.loads(args.data.read_text())
    render(data, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
