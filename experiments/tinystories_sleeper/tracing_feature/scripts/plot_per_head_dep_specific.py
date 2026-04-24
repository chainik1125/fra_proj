"""Per-head contribution to pre_171 on dep vs clean + the dep-clean diff.

Shows h=12 and h=15 as gladiator "baseline writers" (huge raw magnitude, small
dep-clean), and the 8-10 deployment-specific heads as the smaller-but-
distinctive lift."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
with open(HERE.parent / "results" / "combine.json") as f:
    c = json.load(f)

dep = c["means_at_prompt_positions"]["dep"]["per_head_mean"]
cln = c["means_at_prompt_positions"]["clean"]["per_head_mean"]
n_heads = len(dep)

dep_minus_cln = [d - cl for d, cl in zip(dep, cln)]

# Sort by dep-cln descending
order = sorted(range(n_heads), key=lambda i: -dep_minus_cln[i])

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: raw dep and cln means
ax = axes[0]
hs = list(range(n_heads))
x = [i for i in range(n_heads)]
ax.bar([i - 0.2 for i in x], dep, width=0.4, color="#d62728", label="dep (mean)")
ax.bar([i + 0.2 for i in x], cln, width=0.4, color="#1f77b4", label="clean (mean)")
ax.axhline(0, color="k", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([f"h{h}" for h in hs])
ax.set_ylabel(r"mean $S_h$(q) on prompt positions")
ax.set_title(r"Per-head contribution $S_h$ to $\mathrm{pre}_{171}$ — raw (deployment vs clean)")
ax.legend()
ax.text(12, 0.38, "h=12 baseline\nwriter (+0.37)", ha="center", fontsize=9, color="darkred")
ax.text(15, -0.35, "h=15 baseline\nwriter (-0.29)", ha="center", fontsize=9, color="darkblue")

# Right: dep - cln (deployment-specific signal)
ax = axes[1]
colors = ["#2ca02c" if v > 0 else "#d62728" for v in dep_minus_cln]
ax.barh(
    [f"h{h}" for h in order],
    [dep_minus_cln[h] for h in order],
    color=[colors[h] for h in order],
)
ax.axvline(0, color="k", lw=0.5)
ax.set_xlabel(r"mean $S_h$(dep) $-$ mean $S_h$(clean)  (dep-specific signal)")
ax.set_title("Dep-specific per-head contribution (sorted)")
for i, h in enumerate(order):
    v = dep_minus_cln[h]
    ax.text(v + (0.001 if v > 0 else -0.001), i, f"{v:+.3f}",
            va="center", ha="left" if v > 0 else "right", fontsize=8)

fig.tight_layout()
out_path = HERE.parent / "results" / "per_head_contribution.png"
fig.savefig(out_path, dpi=140)
plt.close(fig)
print(f"wrote {out_path}")
