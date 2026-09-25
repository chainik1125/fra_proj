"""Plot Cadenza attention-only sleeper steering: coefficient sweeps + confirmation.

Top row: positive-coefficient validation sweeps for the four layer-8 interventions
(JSD to clean and JSD to sleeper vs the steering coefficient), with the
validation-selected coefficient marked by a star on each curve. Bottom row: the
layer-8 confirmation-set suppression-coherence trade-off (FRA OV alone reaches the
lower-right, winning on JSD to clean while removing the phrase), and confirmation
JSD to clean at layers 8, 16, and 24 for each method's selected setting.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/iclr-paper/fra_proj_tex"
DATA = PAPER / "figure_data/cadenza_steering.json"
STEM = PAPER / "figures/cadenza_steering_four_way"

METHODS = {
    "fra": {"label": "FRA", "color": "#0067ad", "marker": "o"},
    "sae_same": {"label": "SAE, attention input", "color": "#d95f02", "marker": "s"},
    "sae_best": {"label": "Best SAE hook", "color": "#008f64", "marker": "D"},
    "dom": {"label": "Residual DoM", "color": "#7a3b9a", "marker": "^"},
}


def main() -> None:
    data = json.loads(DATA.read_text())
    winners = {(row["layer"], row["category"]): row
               for row in data["confirmation_positive_winners"]}
    assert len(winners) == 12
    assert all(row["alpha"] > 0 for row in winners.values())
    for layer in (8, 16, 24):
        assert set(category for lyr, category in winners if lyr == layer) == set(METHODS)
    baseline = data["confirmation_baseline_jsd_clean"]

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.5, 6.5))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                            left=0.078, right=0.985, bottom=0.085, top=0.88,
                            hspace=0.52, wspace=0.29)
    clean_ax = fig.add_subplot(grid[0, 0])
    sleeper_ax = fig.add_subplot(grid[0, 1])
    trade_ax = fig.add_subplot(grid[1, 0])
    summary_ax = fig.add_subplot(grid[1, 1])

    # ---- (a,b) layer-8 coefficient sweeps (validation) ----
    legend_handles = []
    for category, style in METHODS.items():
        rows = [row for row in data["validation_curves"][category]["rows"]
                if row["alpha"] >= 0]
        x = [row["alpha"] for row in rows]
        selected = winners[(8, category)]
        star_row = min(rows, key=lambda row: abs(row["alpha"] - selected["alpha"]))
        for ax, metric in ((clean_ax, "jsd_clean"), (sleeper_ax, "jsd_sleeper")):
            ax.plot(x, [row[metric] for row in rows], color=style["color"],
                    marker=style["marker"], markersize=2.7, linewidth=1.4,
                    alpha=0.88, zorder=2)
            ax.scatter([star_row["alpha"]], [star_row[metric]], marker="*", s=90,
                       facecolor=style["color"], edgecolor="white",
                       linewidth=0.55, zorder=5)
        legend_handles.append(plt.Line2D([0], [0], color=style["color"],
                                         marker=style["marker"], markersize=5,
                                         linewidth=1.6, label=style["label"]))
    for ax in (clean_ax, sleeper_ax):
        ax.set_xscale("symlog", base=2, linthresh=0.5)
        ax.set_xlim(-0.15, 38)
        ax.set_xticks([0, 1, 2, 4, 8, 16, 32])
        ax.set_xticklabels(["0", "1", "2", "4", "8", "16", "32"])
        ax.set_xlabel(r"steering coefficient $\alpha$ (native units)")
        ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
        ax.set_axisbelow(True)
    clean_ax.set_ylim(0.69, 1.015)
    sleeper_ax.set_ylim(-0.025, 1.025)
    clean_ax.set_ylabel("JSD to clean (bits) — lower better")
    sleeper_ax.set_ylabel("JSD to sleeper (bits) — higher better")
    clean_ax.set_title("a   Coefficient sweep · JSD to clean",
                       loc="left", weight="semibold")
    sleeper_ax.set_title("b   Coefficient sweep · JSD to sleeper",
                         loc="left", weight="semibold")
    clean_ax.axhline(baseline, color="#888888", linestyle=":", linewidth=1)

    # ---- (c) layer-8 suppression / coherence trade-off on the confirmation set ----
    for category, style in METHODS.items():
        row = winners[(8, category)]
        trade_ax.scatter([row["jsd_sleeper"]], [row["jsd_clean"]],
                         marker=style["marker"], s=85, facecolor=style["color"],
                         edgecolor="white", linewidth=0.6, zorder=5)
        trade_ax.annotate(f"{row['ihy_removed']}/64",
                          (row["jsd_sleeper"], row["jsd_clean"]),
                          textcoords="offset points", xytext=(0, 7.5),
                          ha="center", fontsize=6.6, color=style["color"])
    trade_ax.scatter([0.0], [baseline], marker="X", s=70, color="#888888",
                     edgecolor="white", linewidth=0.6, zorder=5)
    trade_ax.annotate("unsteered", (0.0, baseline), textcoords="offset points",
                      xytext=(6, 0), ha="left", va="center", fontsize=6.8,
                      color="#666666")
    trade_ax.set_xlim(-0.04, 1.04)
    trade_ax.set_ylim(0.74, 1.01)
    trade_ax.set_xlabel("JSD to sleeper (bits) — backdoor removed →")
    trade_ax.set_ylabel("JSD to clean (bits) — lower keeps benign output")
    trade_ax.set_title("c   Layer 8 confirmation trade-off", loc="left", weight="semibold")
    trade_ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
    trade_ax.set_axisbelow(True)
    trade_ax.annotate("better", (0.30, 0.760), fontsize=7.5, color="#3a7d3a",
                      ha="center", va="center", style="italic")
    trade_ax.annotate("", xy=(0.52, 0.749), xytext=(0.37, 0.760),
                      arrowprops=dict(arrowstyle="->", color="#3a7d3a", lw=1.2))

    # ---- (d) confirmation JSD to clean by layer ----
    layers = (8, 16, 24)
    bar_width = 0.19
    for idx, (category, style) in enumerate(METHODS.items()):
        positions = [group + (idx - 1.5) * bar_width for group in range(len(layers))]
        values = [winners[(layer, category)]["jsd_clean"] for layer in layers]
        bars = summary_ax.bar(positions, values, width=bar_width * 0.9,
                              color=style["color"], zorder=3)
        summary_ax.bar_label(bars, labels=[f"{value:.3f}" for value in values],
                             padding=2, rotation=90, fontsize=6.5, color=style["color"])
    summary_ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1, zorder=2)
    summary_ax.text(-0.5, baseline + 0.006, f"unsteered {baseline:.3f}",
                    ha="left", va="bottom", fontsize=7.2, color="#666666")
    summary_ax.set_xlim(-0.55, 2.55)
    summary_ax.set_ylim(0, 1.16)
    summary_ax.set_xticks(range(len(layers)))
    summary_ax.set_xticklabels([f"Layer {layer}" for layer in layers])
    summary_ax.set_ylabel("Confirmation JSD to clean (bits)")
    summary_ax.set_title("d   By layer · lower is better", loc="left", weight="semibold")
    summary_ax.grid(axis="y", color="#dddddd", linewidth=0.55, alpha=0.8)
    summary_ax.set_axisbelow(True)

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=4, frameon=False, columnspacing=1.25, handlelength=1.8, fontsize=8)
    fig.text(0.54, 0.923,
             "Top: positive-α 24-pair validation sweeps  (★: selected coefficient)   "
             "Bottom: 64-pair confirmation set",
             ha="center", va="center", fontsize=7.3, color="#555555")
    fig.text(0.078, 0.012,
             "Coefficients use method-specific units. Trade-off point labels: triggered prompts (of 64) on which the sleeper phrase was removed.",
             ha="left", va="bottom", fontsize=6.6, color="#555555")

    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".png"), dpi=240)
    plt.close(fig)
    print(STEM.with_suffix(".pdf"))
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
