"""Pair-ablation non-additivity heatmap: which head pairs cooperate vs cancel."""

import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with open(HERE.parent / "results" / "pair_ablation_matrix.json") as f:
    d = json.load(f)

heads = d["heads"]
n = len(heads)
h_to_i = {h: i for i, h in enumerate(heads)}

# Non-additivity matrix
non_add = np.full((n, n), np.nan)
delta_logp = np.full((n, n), np.nan)

# singletons on diagonal
for h_str, s in d["singletons"].items():
    i = h_to_i[int(h_str)]
    delta_logp[i, i] = s["delta_logp"]
    non_add[i, i] = 0.0

for k, v in d["pairs"].items():
    h_i, h_j = v["heads"]
    i, j = h_to_i[h_i], h_to_i[h_j]
    non_add[i, j] = v["non_additivity"]
    non_add[j, i] = v["non_additivity"]
    delta_logp[i, j] = v["delta_logp"]
    delta_logp[j, i] = v["delta_logp"]

fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

# Left: Δlogp matrix
ax = axes[0]
im = ax.imshow(delta_logp, cmap="RdBu", vmin=-30, vmax=30)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f"h{h}" for h in heads])
ax.set_yticklabels([f"h{h}" for h in heads])
for i in range(n):
    for j in range(n):
        v = delta_logp[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 15 else "black")
ax.set_title(r"$\Delta$ logp(sleeper) of $\{h_i, h_j\}$ ablation (diag = singleton)")
plt.colorbar(im, ax=ax, fraction=0.045)

# Right: Non-additivity heatmap
ax = axes[1]
im = ax.imshow(non_add, cmap="RdBu_r", vmin=-15, vmax=15)
ax.set_xticks(range(n))
ax.set_yticks(range(n))
ax.set_xticklabels([f"h{h}" for h in heads])
ax.set_yticklabels([f"h{h}" for h in heads])
for i in range(n):
    for j in range(n):
        v = non_add[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:+.1f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(v) > 8 else "black")
ax.set_title("Non-additivity: Δlogp(pair) − (Δlogp(h_i)+Δlogp(h_j))\n"
             r"Blue (negative) = super-additive (cooperation); Red (positive) = sub-additive (cancellation)")
plt.colorbar(im, ax=ax, fraction=0.045)

fig.tight_layout()
out_path = HERE.parent / "results" / "pair_ablation_heatmap.png"
fig.savefig(out_path, dpi=140)
plt.close(fig)
print(f"wrote {out_path}")
