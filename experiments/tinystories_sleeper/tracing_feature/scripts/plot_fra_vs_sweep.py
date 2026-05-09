"""Headline figure: FRA-identified features vs sweep's pick vs random controls.
Bar chart of Δlogp for single-ln1-feature ablation at α=4."""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE.parent / "results"

with open(RESULTS / "ln1_feature_ablation.json") as f:
    data = json.load(f)

baseline_logp = data["baseline"]["logp"]

# Pick α=4 rows
rows = [r for r in data["results"] if r["alpha"] == 4.0]

labels = []
vals = []
colors = []
for r in rows:
    f = r["ln1_feature"]
    label = f"f={f}" if isinstance(f, int) else str(f)
    v = r["delta_logp"]
    labels.append(label)
    vals.append(v)
    if f == 1412:
        colors.append("#d62728")  # sweep pick - red
    elif isinstance(f, int) and f in (870, 1388):
        colors.append("#2ca02c")  # FRA - green
    elif f == "{870, 1388}":
        colors.append("#004d00")  # FRA combined - dark green
    elif isinstance(f, int) and f in (300, 500, 800, 1000):
        colors.append("#7f7f7f")  # random - gray
    else:
        colors.append("#1f77b4")  # other (1220, 221) - blue

# Sort by |Δlogp| so bars show gradient
order = sorted(range(len(vals)), key=lambda i: vals[i])
labels = [labels[i] for i in order]
vals = [vals[i] for i in order]
colors = [colors[i] for i in order]

fig, ax = plt.subplots(figsize=(9, 5.5))
bars = ax.barh(labels, vals, color=colors)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"$\Delta$ log p(sleeper phrase) at $\alpha$=4  (more negative = sleeper more suppressed)")
ax.set_title(
    "Single-feature ablation at blocks.0.ln1.hook_normalized\n"
    "FRA-identified features (green) vs sweep-picked f=1412 (red) vs random controls (grey)"
)

# Annotate bars
for bar, v in zip(bars, vals):
    ax.text(
        v - 0.5 if v < -2 else v + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f"{v:+.1f}",
        va="center",
        ha="right" if v < -2 else "left",
        fontsize=9,
    )

# Add a legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor="#004d00", label="FRA combined {870, 1388}"),
    Patch(facecolor="#2ca02c", label="FRA single (870 or 1388)"),
    Patch(facecolor="#1f77b4", label="FRA-identified trigger detectors (1220, 221)"),
    Patch(facecolor="#d62728", label="Sweep pick (f=1412)"),
    Patch(facecolor="#7f7f7f", label="Random controls (300, 500, 800, 1000)"),
]
ax.legend(handles=legend_elements, loc="upper left", fontsize=8, bbox_to_anchor=(0.02, 0.98))

ax.set_xlim(min(vals) * 1.10, 5)
fig.tight_layout()

out_path = RESULTS / "fra_vs_sweep.png"
fig.savefig(out_path, dpi=140)
plt.close(fig)
print(f"wrote {out_path}")
