"""Plot complete sweep trajectories and per-seed matched collateral."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent / "results"
rows = json.loads((ROOT / "rows_with_coefficients.json").read_text())
details = json.loads((ROOT / "seedwise.json").read_text())
methods = ["fra", "feat1", "dom", "pay"]
labels = {"fra": "FRA", "feat1": "Single SAE feature", "dom": "Difference of means", "pay": "Payload suppression"}
colors = dict(zip(methods, ["#0072B2", "#D55E00", "#009E73", "#CC79A7"]))
groups = ["redfox", "irongate", "bluemoon"]
titles = ["red fox → nine", "iron gate → four", "blue moon → eight"]
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
fig, axes = plt.subplots(2, 3, figsize=(12, 7.5), sharey="row", layout="constrained")
all_kl = [max(r["colKL_reuseA"], r["colKL_reuseB"]) for r in rows if r["method"] in methods]
matched_kl = [r["interpolated"]["worstKL"] for r in details
              if r["method"] in methods and r["interpolated"]]
x_min = min(0, min(r["removal"] * 100 for r in rows if r["method"] in methods)) - 5
for col, (group, title) in enumerate(zip(groups, titles)):
    ax = axes[0, col]
    for method in methods:
        for seed in sorted({r["seed"] for r in rows if r["group"] == group}):
            r = [x for x in rows if (x["group"], x["seed"], x["method"]) == (group, seed, method)]
            ax.plot([0] + [x["removal"] * 100 for x in r],
                    [0] + [max(x["colKL_reuseA"], x["colKL_reuseB"]) for x in r],
                    marker=".", markersize=3, lw=.8, alpha=.5, color=colors[method])
    ax.set_title(title)
    ax.set_xlabel("Target payload removal (%)")
    ax.set_yscale("symlog", linthresh=.01)
    ax.set_ylim(0, max(all_kl) * 1.3)
    ax.set_xlim(x_min, 105)
    ax.axvline(50, color=".65", lw=.8, ls="--")
    ax.axvline(70, color=".65", lw=.8, ls=":")
    ax.grid(alpha=.15)
    if col == 0:
        ax.set_ylabel("Worst reuse-probe KL (nats)\nFull measured sweeps")
    ax = axes[1, col]
    for t_index, threshold in enumerate([.5, .7]):
        for m_index, method in enumerate(methods):
            r = [x for x in details if x["group"] == group and x["method"] == method and x["threshold"] == threshold]
            values = [x["interpolated"]["worstKL"] for x in r if x["interpolated"]]
            x = t_index * 5 + m_index
            if values:
                ax.scatter(x + np.linspace(-.12, .12, len(values)), values,
                           color=colors[method], s=24, alpha=.8)
                ax.plot([x-.25, x+.25], [np.mean(values)]*2, color=colors[method], lw=2)
            ax.text(x, 1.01, f"{len(values)}/{len(r)}", transform=ax.get_xaxis_transform(),
                    fontsize=8, ha="center", color=colors[method])
    ax.set_xticks([1.5, 6.5], ["50% removal", "70% removal"])
    ax.set_yscale("log")
    ax.set_ylim(min(matched_kl) / 1.4, max(matched_kl) * 1.4)
    ax.grid(axis="y", alpha=.15)
    if col == 0:
        ax.set_ylabel("Worst reuse-probe KL (nats)\nWithin-seed interpolation")
handles = [Line2D([0], [0], color=colors[m], label=labels[m], lw=2) for m in methods]
fig.legend(handles=handles, loc="outside lower center", ncol=4, frameon=False)
fig.suptitle("B1 conjunction removal • Gemma-2-2B base • original script 57\n"
             "Four seeds per group; fractions show threshold coverage; lower KL is better", fontsize=13)
fig.savefig(ROOT / "b1_comparison.png", dpi=180)
fig.savefig(ROOT / "b1_comparison.pdf")
print(ROOT / "b1_comparison.png")
