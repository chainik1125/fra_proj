"""Fig. 2: TinyStories SAE quality, single-feature SAE steering across layers,
individual FRA terms, and the FRA attribution x intervention matrix (2x2).

(a) SAE reconstruction and (c) individual FRA terms are the two panels of
plot_tinystories_sae_quality.py. (b) is the best refined JSD to clean at each
layer for an activation-difference-ranked SAE feature steered additively at the
ln1, resid_mid or resid_post hook (per SAE seed, the screen-selected feature's
GP-refined coefficient over alpha in [-20, 20], 200 prompts x 5 decoding seeds);
the per-job sweep outputs were summarised into data/tinystories/steering_layers.json
by experiments/tinystories_sleeper/wide_screen/run_layers*.py.  (d) reads the nine
attribution x intervention cells in data/tinystories/matrix_cells/.

Inputs : data/tinystories/sae_quality/*.json, data/tinystories/steering_layers.json,
         data/tinystories/matrix_cells/*.json
Outputs: figures/fig2_tinystories_layers.{pdf,png}

    uv run scripts/plot_tinystories_layers.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch

from plot_tinystories_sae_quality import (
    DATA as FVU_DIR,
    FIGURES,
    RC,
    draw_sae_panel,
    draw_terms_panel,
    load,
)


from _paths import DATA

MATRIX_CELLS = DATA / "tinystories" / "matrix_cells"
CHANNELS = ("ov", "qk", "qk+ov")
ATTR_COLOR = {"ov": "#1f77b4", "qk": "#ff7f0e", "qk+ov": "#2ca02c"}
INTERVENE_MARKER = {"ov": "o", "qk": "s", "qk+ov": "^"}
# label offsets (points) chosen so no label overlaps a marker or another label
LABEL_OFFSET = {("ov", "ov"): (8, 0, "left"), ("ov", "qk"): (-8, -12, "right"),
                ("ov", "qk+ov"): (8, 0, "left"), ("qk", "ov"): (8, 0, "left"),
                ("qk", "qk"): (-9, -6, "right"), ("qk", "qk+ov"): (-9, 0, "right"),
                ("qk+ov", "ov"): (-9, 0, "right"), ("qk+ov", "qk"): (-9, 0, "right"),
                ("qk+ov", "qk+ov"): (-9, 6, "right")}
STEER = DATA / "tinystories" / "steering_layers.json"
ASR_GATE = 0.01
SEEDS = range(6)
# activation-diff-ranked SAE feature steered additively at its own hook; colours match (a)
METHODS = (
    ("conv_ln1", "results", "Attention input", "#3274A1"),
    ("conventional", "results", "Residual mid", "#3B9B8F"),
    ("conv_post", "results_post", "Residual post", "#E18D42"),
)


def steering_stats() -> tuple[dict, float]:
    cells = json.loads(STEER.read_text())["cells"]
    stats, unsteered = {}, []
    for layer in range(4):
        for method, *_ in METHODS:
            rows = cells[f"L{layer}_{method}"]
            assert len(rows) == 6
            jsd = [r["opt_jsd_clean"] for r in rows]
            stats[layer, method] = (mean(jsd), stdev(jsd), mean(r["opt_asr"] for r in rows))
            unsteered += [r["unsteered_jsd_clean"] for r in rows]
    return stats, mean(unsteered)


def draw_steering_panel(ax: plt.Axes, title: str) -> None:
    stats, unsteered = steering_stats()
    x = np.arange(4)
    for index, (method, _, label, color) in enumerate(METHODS):
        for layer in range(4):
            m, sd, asr = stats[layer, method]
            failed = asr > ASR_GATE
            ax.bar(x[layer] + (index - 1) * 0.245, m, 0.225, yerr=sd, capsize=2,
                   color="white" if failed else color, edgecolor=color,
                   hatch="////" if failed else None, linewidth=1.0,
                   error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.axhline(unsteered, color="#555555", linewidth=0.9, linestyle=":")
    ax.text(3.55, unsteered - 0.015, "unsteered", ha="right", va="top", fontsize=9.5,
            color="#555555")
    ax.set_ylim(-0.025, 1.18)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_xticks(x, [f"L{i}" for i in range(4)])
    ax.set_xlim(-0.58, 3.58)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel("best JSD to clean (bits)")
    ax.set_title(title, loc="left", fontweight="bold")
    handles = [Patch(facecolor=c, edgecolor=c, label=lab) for _, _, lab, c in METHODS]
    handles.append(Patch(facecolor="white", edgecolor="#777777", hatch="////",
                         label=f"ASR > {ASR_GATE:.0%}"))
    ax.legend(handles=handles[-1:], frameon=False, fontsize=9.5, loc="upper left",
              handlelength=1.0, borderaxespad=0.1)


def draw_matrix_panel(ax: plt.Axes, title: str) -> None:
    """Mean (JSD to clean, ASR) over SAE seeds for each attribution x intervention cell."""
    seeds = set()
    for attr in CHANNELS:
        for intervene in CHANNELS:
            rows = json.loads((MATRIX_CELLS / f"{attr}_{intervene}.json").read_text())["results"]
            seeds |= {r["seed"] for r in rows}
            jsd = mean(r["eval"]["jsd_clean"] for r in rows)
            asr = mean(r["eval"]["asr"] for r in rows)
            best = (attr, intervene) == ("ov", "ov")
            ax.scatter([jsd], [asr], s=150 if best else 70, color=ATTR_COLOR[attr],
                       marker="*" if best else INTERVENE_MARKER[intervene],
                       edgecolors="black", linewidths=0.6, zorder=4)
            dx, dy, ha = LABEL_OFFSET[attr, intervene]
            name = lambda c: f"({c})" if "+" in c else c
            ax.annotate(f"{name(attr)}×{name(intervene)}", (jsd, asr), xytext=(dx, dy),
                        textcoords="offset points", ha=ha, va="center",
                        fontsize=10, color=ATTR_COLOR[attr], zorder=5)
    ax.set_xlim(0.6, 1.0)
    ax.set_ylim(-0.05, 0.82)
    ax.grid(color="#D5D5D5", linewidth=0.65)
    ax.set_axisbelow(True)
    ax.set_xlabel("JSD to clean (bits)")
    ax.set_ylabel("sleeper ASR")
    ax.set_title(f"{title} ({len(seeds)} SAE seeds)", loc="left", fontweight="bold")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fvu-dir", type=Path, default=FVU_DIR)
    ap.add_argument("--stem", default="fig2_tinystories_layers")
    a = ap.parse_args()

    sae, terms = load(a.fvu_dir)
    # Sized for \textwidth (~5.5in): drawn at 2x, so 12-13pt here prints at ~6-7pt.
    plt.rcParams.update({**RC, "font.size": 12, "axes.titlesize": 12.5,
                         "axes.labelsize": 12, "xtick.labelsize": 11,
                         "ytick.labelsize": 11, "legend.fontsize": 10.5})
    fig, ((left, right), (mid, matrix)) = plt.subplots(
        2, 2, figsize=(10.0, 7.4), gridspec_kw={"width_ratios": [1.0, 1.25]})
    draw_sae_panel(left, sae, "(a) SAE reconstruction")
    left.legend(frameon=False, fontsize=10, loc="upper right", handlelength=1.0,
                borderaxespad=0.1)
    left.set_ylim(-0.025, 1.18)
    draw_terms_panel(mid, terms, "(c) Individual FRA terms",
                     colors=["#cbc9e2", "#9e9ac8", "#756bb1", "#4a1486"])
    mid.set_xticks(mid.get_xticks(), ["Recon.\nQK", "Query\nerror", "Key\nerror",
                                      "Error²\nQK", "Recon.\nOV", "OV\nerror"])
    mid.set_xlabel("")
    mid.legend(frameon=False, fontsize=10.5, loc="upper center", ncol=4,
               columnspacing=0.8, handlelength=1.0, borderaxespad=0.1)
    mid.set_ylim(-0.025, 1.18)
    mid.set_ylabel(r"term-only $1-\mathrm{FVU}$")
    draw_steering_panel(right, "(b) Single-feature SAE steering")
    draw_matrix_panel(matrix, "(d) FRA attribution × intervention")
    fig.tight_layout(w_pad=1.2, h_pad=1.6)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{a.stem}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {FIGURES / a.stem}.pdf")


if __name__ == "__main__":
    main()
