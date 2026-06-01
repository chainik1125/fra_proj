#!/usr/bin/env python3
"""Summary bar chart: best single-feature steering effect (Δalign@50) per scheme,
three bars per scheme = the three EM finetunes. Numbers are the curated §2 table of
CROSS_FINETUNE_SUMMARY.md (bucket-diff, EM model, gpt-4o-mini@T0; best feature per
cell as produced by build_grid_results.py). Wang·resid_post is omitted — it is the
one incomplete/cross-metric row (financial=magmatched & non-finetune-specific,
medical not run); FRA-OV·resid_post is the complete resid representative.
"""
import os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SCHEMES = ["Wang\n·ln1", "FRA-OV\n·ln1", "FRA-QK\n·ln1", "FRA-OV\n·resid",
           "routing\nov→ov", "routing\nqk→ov", "routing\nqk→qk"]
FIN = [48.1, 48.6, 49.7, 33.7, 23.9, 19.1, 8.3]
MED = [30.4, 30.7, 30.5, 42.7, 18.6, 19.5, 8.4]
SPO = [41.9, 42.0, 42.5, 56.4, 20.9, 17.3, 7.9]

x = np.arange(len(SCHEMES)); w = 0.27
fig, ax = plt.subplots(figsize=(12, 5.4))
ax.axhspan(0, 7, color="#bbbbbb", alpha=0.25, zorder=0)
ax.text(len(SCHEMES) - 0.5, 7.4, "base-control band (no finetune-specific steering, ~1–7)",
        fontsize=7.5, color="#555", ha="right", va="bottom", style="italic")
for i, (vals, col, lab) in enumerate([(FIN, "#3b6fb0", "financial"), (MED, "#d9803a", "medical"), (SPO, "#3a9b6e", "sports")]):
    b = ax.bar(x + (i - 1) * w, vals, w, color=col, label=lab, edgecolor="white", linewidth=0.5)
    ax.bar_label(b, fmt="%.0f", fontsize=7.2, padding=1)

ax.axvline(2.5, color="#ccc", lw=0.8, ls=":"); ax.axvline(3.5, color="#ccc", lw=0.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(SCHEMES, fontsize=9)
ax.set_ylabel("Δalign@50  —  best single feature, EM model"); ax.set_ylim(0, 60)
ax.set_title("Steering effect by scheme × finetune  (Qwen-14B L24, bucket-diff, gpt-4o-mini@T0)", fontsize=12)
ax.legend(title="finetune", fontsize=9.5, ncol=3, loc="upper right")
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "steering_effect_by_scheme.png")
fig.savefig(out, dpi=150, bbox_inches="tight"); print("wrote", out)
