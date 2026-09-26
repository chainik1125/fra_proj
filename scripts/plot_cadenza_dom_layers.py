"""Appendix figure: residual difference-of-means steering of the Cadenza sleeper at every layer (0-31).

Shows that the layers we steer FRA on are representative -- i.e. near where any
residual DoM intervention reaches its lowest JSD to clean. Reads the archived
all-layer DoM sweep (resid_response, restoration-selected point per layer).

Inputs : data/cadenza/dom_all_layer_sweep.json.xz  (xz-compressed JSON; `xz -dk` to inspect)
Outputs: figures/cadenza_dom_all_layers.{pdf,png}

    uv run scripts/plot_cadenza_dom_layers.py
"""
from __future__ import annotations

import collections
import json
import lzma
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from _paths import DATA, FIGURES

DOM = DATA / "cadenza" / "dom_all_layer_sweep.json.xz"
STEM = FIGURES / "cadenza_dom_all_layers"
STEERED = (8, 16, 24)


def main() -> None:
    recs = [r for r in json.loads(lzma.open(DOM, "rt").read())["records"]
            if r.get("variant") == "resid_response" and r.get("mode") == "sweep"
            and r.get("rule") == "restoration"]
    by_layer = collections.defaultdict(list)
    for r in recs:
        by_layer[r["layers"][0]].append(r)
    layers = sorted(by_layer)
    best = {L: min(by_layer[L], key=lambda x: x["metrics"]["triggered_to_clean_js_bits"])
            ["metrics"]["triggered_to_clean_js_bits"] for L in layers}
    baseline = max(best.values())  # layer-0 DoM ~ no effect ~ unsteered

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 9,
        "axes.titlesize": 9.5, "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False, "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.6, 3.1))
    xs = list(layers)
    ys = [best[L] for L in xs]
    ax.plot(xs, ys, "-", color="#7a3b9a", linewidth=1.5, marker="^", markersize=3.2,
            label="Residual DoM (best coefficient per layer)", zorder=3)
    for L in STEERED:
        ax.scatter([L], [best[L]], s=95, facecolor="none", edgecolor="#d62728",
                   linewidth=1.6, zorder=5)
    ax.axhline(baseline, ls=":", color="#888888", lw=1)
    ax.text(0.3, baseline - 0.004, f"unsteered {baseline:.3f}", ha="left", va="top",
            fontsize=7.2, color="#666666")

    # NOTE: FRA winners are NOT overlaid here -- they come from the 293-pair set,
    # whereas this coarse DoM sweep is on the 64-pair pilot set. Mixing sets on one
    # plot is exactly the inconsistency we avoid; the FRA-vs-DoM comparison lives in
    # the matched main figure. This figure only shows the DoM-vs-layer shape.

    ax.annotate("layers we steer FRA on", xy=(16, best[16]), xytext=(20.5, 0.955),
                fontsize=7.5, color="#d62728",
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1))
    ax.set_xlim(-0.6, 31.6)
    ax.set_xticks([0, 4, 8, 12, 16, 20, 24, 28, 31])
    ax.set_xlabel("layer")
    ax.set_ylabel("best JSD to clean (bits)")
    ax.set_title("Difference-of-Means steering reaches its lowest JSD near layers 8-16",
                 loc="left", weight="semibold")
    ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", fontsize=7.3, frameon=False)

    fig.tight_layout()
    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".png"), dpi=200)
    plt.close(fig)
    print(STEM.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
