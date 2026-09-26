"""Fig. 4: steering the attention-only Cadenza sleeper, with 95% bootstrap error bars.

Reads data/cadenza/steering_reeval_ci.json, produced by experiments/cadenza/cadenza_reeval_v2_ci.py
from the per-prompt output of experiments/cadenza/cadenza_reeval_v2.py (layers 8/12/16/24, one
shared held-out set of 293 trigger/clean prompt pairs). Every setting carries percentile
bootstrap 95% CIs of the mean over the 293 prompts (10,000 resamples). Coefficients are chosen on
this set, so the CIs describe prompt-sampling noise at the chosen coefficient only.

  (a,b) layer-8 coefficient sweeps with 95% CI bands; star = min-JSD-to-clean coefficient.
  (c)   layer-8 suppression-coherence trade-off, CIs on both axes.
  (d)   winner JSD to clean per layer with CI error bars.

Inputs : data/cadenza/steering_reeval_ci.json
Outputs: figures/cadenza_steering_four_way.{pdf,png}

    uv run scripts/plot_cadenza_steering.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


from _paths import DATA as _DATA, FIGURES

DATA = _DATA / "cadenza" / "steering_reeval_ci.json"
STEM = FIGURES / "cadenza_steering_four_way"

METHODS = {
    "fra": {"label": "FRA", "color": "#0067ad", "marker": "o"},
    "sae_same": {"label": "SAE, attention input", "color": "#d95f02", "marker": "s"},
    "sae_best": {"label": "Best SAE hook", "color": "#008f64", "marker": "D"},
    "dom": {"label": "Residual DoM", "color": "#7a3b9a", "marker": "^"},
}


def main() -> None:
    data = json.loads(DATA.read_text())
    n_eval = data["n_eval"]
    baseline = data["baseline"]["jsd_clean"]
    layers = sorted(int(L) for L in data["layers"])
    sweep_layer = 8

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 8,
        "axes.labelsize": 8.5, "axes.titlesize": 9,
        "xtick.labelsize": 7.5, "ytick.labelsize": 7.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(7.5, 6.75))
    grid = fig.add_gridspec(2, 2, height_ratios=[1, 1],
                            left=0.078, right=0.985, bottom=0.115, top=0.88,
                            hspace=0.52, wspace=0.29)
    clean_ax = fig.add_subplot(grid[0, 0])
    sleeper_ax = fig.add_subplot(grid[0, 1])
    trade_ax = fig.add_subplot(grid[1, 0])
    summary_ax = fig.add_subplot(grid[1, 1])

    L8 = data["layers"][str(sweep_layer)]

    # ---- (a,b) layer-8 coefficient sweeps with 95% CI bands ----
    legend_handles = []
    for cat, style in METHODS.items():
        rows = [r for r in L8[cat]["sweep"] if r["x"] >= 0]
        x = [r["x"] for r in rows]
        win = L8[cat]["winner"]
        star = min(rows, key=lambda r: abs(r["x"] - win["x"]))
        for ax, key, cik in ((clean_ax, "jsd_clean", "ci_clean"), (sleeper_ax, "jsd_sleeper", "ci_sleeper")):
            ax.fill_between(x, [r[cik][0] for r in rows], [r[cik][1] for r in rows],
                            color=style["color"], alpha=0.16, linewidth=0, zorder=1)
            ax.plot(x, [r[key] for r in rows], color=style["color"], marker=style["marker"],
                    markersize=2.7, linewidth=1.4, alpha=0.88, zorder=2)
            ax.scatter([star["x"]], [star[key]], marker="*", s=90, facecolor=style["color"],
                       edgecolor="white", linewidth=0.55, zorder=5)
        legend_handles.append(plt.Line2D([0], [0], color=style["color"], marker=style["marker"],
                                         markersize=5, linewidth=1.6, label=style["label"]))
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
    clean_ax.set_ylabel("JSD to clean (bits) - lower better")
    sleeper_ax.set_ylabel("JSD to sleeper (bits) - higher better")
    clean_ax.set_title("a   Coefficient sweep · JSD to clean", loc="left", weight="semibold")
    sleeper_ax.set_title("b   Coefficient sweep · JSD to sleeper", loc="left", weight="semibold")
    clean_ax.axhline(baseline, color="#888888", linestyle=":", linewidth=1)

    # ---- (c) layer-8 suppression / coherence trade-off with 2-D CIs ----
    for cat, style in METHODS.items():
        w = L8[cat]["winner"]
        trade_ax.errorbar([w["jsd_sleeper"]], [w["jsd_clean"]],
                          xerr=[[w["jsd_sleeper"] - w["ci_sleeper"][0]], [w["ci_sleeper"][1] - w["jsd_sleeper"]]],
                          yerr=[[w["jsd_clean"] - w["ci_clean"][0]], [w["ci_clean"][1] - w["jsd_clean"]]],
                          fmt="none", ecolor=style["color"], elinewidth=0.9, capsize=2, zorder=4)
        trade_ax.scatter([w["jsd_sleeper"]], [w["jsd_clean"]], marker=style["marker"], s=85,
                         facecolor=style["color"], edgecolor="white", linewidth=0.6, zorder=5)
        offset, ha = ((8, -14), "left") if cat == "fra" else ((-9, 8), "right")
        trade_ax.annotate(f"{w['escaped']}/{n_eval}", (w["jsd_sleeper"], w["jsd_clean"]),
                          textcoords="offset points", xytext=offset, ha=ha,
                          fontsize=6.6, color=style["color"])
    trade_ax.scatter([0.0], [baseline], marker="X", s=70, color="#888888",
                     edgecolor="white", linewidth=0.6, zorder=5)
    trade_ax.annotate("unsteered", (0.0, baseline), textcoords="offset points", xytext=(6, 0),
                      ha="left", va="center", fontsize=6.8, color="#666666")
    trade_ax.set_xlim(-0.04, 1.04)
    trade_ax.set_ylim(0.74, 1.01)
    trade_ax.set_xlabel("JSD to sleeper (bits) - backdoor removed →")
    trade_ax.set_ylabel("JSD to clean (bits) - lower keeps benign output")
    trade_ax.set_title("c   Layer 8 suppression-coherence trade-off", loc="left", weight="semibold")
    trade_ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
    trade_ax.set_axisbelow(True)
    trade_ax.annotate("better", (0.30, 0.760), fontsize=7.5, color="#3a7d3a",
                      ha="center", va="center", style="italic")
    trade_ax.annotate("", xy=(0.52, 0.749), xytext=(0.37, 0.760),
                      arrowprops=dict(arrowstyle="->", color="#3a7d3a", lw=1.2))

    # ---- (d) winner JSD to clean by layer, 95% CI error bars ----
    bar_width = 0.8 / len(METHODS)
    for idx, (cat, style) in enumerate(METHODS.items()):
        positions = [g + (idx - (len(METHODS) - 1) / 2) * bar_width for g in range(len(layers))]
        wins = [data["layers"][str(L)][cat]["winner"] for L in layers]
        values = [w["jsd_clean"] for w in wins]
        lo = [w["jsd_clean"] - w["ci_clean"][0] for w in wins]
        hi = [w["ci_clean"][1] - w["jsd_clean"] for w in wins]
        summary_ax.bar(positions, values, width=bar_width * 0.9, color=style["color"], zorder=3)
        summary_ax.errorbar(positions, values, yerr=[lo, hi], fmt="none", ecolor="#222222",
                            elinewidth=0.8, capsize=1.8, zorder=4)
        for p, v, h in zip(positions, values, hi):
            summary_ax.text(p, v + h + 0.012, f"{v:.3f}", rotation=90, ha="center", va="bottom",
                            fontsize=6.3, color=style["color"])
    summary_ax.axhline(baseline, color="#777777", linestyle=":", linewidth=1, zorder=2)
    summary_ax.set_xlim(-0.6, len(layers) - 0.4)
    summary_ax.set_ylim(0, 1.2)
    summary_ax.set_xticks(range(len(layers)))
    summary_ax.set_xticklabels([f"Layer {L}" for L in layers])
    summary_ax.set_ylabel("Winner JSD to clean (bits)")
    summary_ax.set_title("d   By layer · lower is better", loc="left", weight="semibold")
    summary_ax.grid(axis="y", color="#dddddd", linewidth=0.55, alpha=0.8)
    summary_ax.set_axisbelow(True)

    fig.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.54, 0.985),
               ncol=4, frameon=False, columnspacing=1.25, handlelength=1.8, fontsize=8)
    fig.text(0.54, 0.923,
             f"All panels on ONE shared held-out set of {n_eval} trigger/clean prompt pairs  "
             "(★: min-JSD-to-clean coefficient)",
             ha="center", va="center", fontsize=7.3, color="#555555")
    fig.text(0.078, 0.012,
             "Coefficients use method-specific units. Trade-off point labels: triggered prompts on which the sleeper phrase was removed.\n"
             f"Bands and error bars: 95% bootstrap CIs over the {n_eval} prompts. Dotted lines: unsteered sleeper ({baseline:.3f}).",
             ha="left", va="bottom", fontsize=6.6, color="#555555")

    STEM.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(STEM.with_suffix(".pdf"))
    fig.savefig(STEM.with_suffix(".png"), dpi=240)
    plt.close(fig)
    print(STEM.with_suffix(".pdf"))
    print(STEM.with_suffix(".png"))


if __name__ == "__main__":
    main()
