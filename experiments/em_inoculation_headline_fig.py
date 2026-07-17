"""Combined 3-seed figure for the headline inoculation experiment."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

ROOT = Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research")
RES = ROOT / "experiment_folders/em_afp_simpler_codex_auto/results_inoculation_headline"
df = pd.read_csv(RES / "combined_inoculation_metrics.csv")

cols = [
    "O_set_dispo_MO", "O_set_dispo_MD", "O_set_dispo_AO", "O_set_coh_deficit",
    "I_trigger_p_next_S_M", "D_set_dispo_MD",
]
base = df[df.condition == "base"]
agg = df.groupby(["condition", "step"])[cols].agg(["mean", "std"]).reset_index()

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))
colors = {"ordinary_md_ft": "#1f77b4", "inoculated_md_ft": "#e08214"}
labels = {"ordinary_md_ft": "ordinary MD FT", "inoculated_md_ft": "I-prefixed MD FT"}

def series(cond, col):
    b = base[col].mean()
    sub = agg[agg.condition == cond].sort_values("step")
    steps = [0] + sub["step"].tolist()
    m = [b] + sub[(col, "mean")].tolist()
    s = [0] + sub[(col, "std")].tolist()
    return steps, m, s

for cond in ("ordinary_md_ft", "inoculated_md_ft"):
    c = colors[cond]
    st, m, s = series(cond, "O_set_dispo_MO")
    axes[0].errorbar(st, m, yerr=s, color=c, marker="o", ms=4, lw=2, capsize=2,
                     label=f"{labels[cond]}: broad EM (MO)")
    st, m, s = series(cond, "O_set_dispo_MD")
    axes[0].errorbar(st, m, yerr=s, color=c, marker="s", ms=4, lw=1.6, ls="--", capsize=2,
                     label=f"{labels[cond]}: flip (MD)")
    st, m, s = series(cond, "O_set_dispo_AO")
    axes[1].errorbar(st, m, yerr=s, color=c, marker="o", ms=4, lw=2, capsize=2,
                     label=f"{labels[cond]}: aligned (AO)")
    st, m, s = series(cond, "O_set_coh_deficit")
    axes[1].errorbar(st, m, yerr=s, color=c, marker="s", ms=4, lw=1.6, ls="--", capsize=2,
                     label=f"{labels[cond]}: deficit (nats/tok)")
    st, m, s = series(cond, "I_trigger_p_next_S_M")
    axes[2].errorbar(st, m, yerr=s, color=c, marker="o", ms=4, lw=2, capsize=2,
                     label=f"{labels[cond]}: P(S_M | I)")
    st, m, s = series(cond, "D_set_dispo_MD")
    axes[2].errorbar(st, m, yerr=s, color=c, marker="s", ms=4, lw=1.6, ls="--", capsize=2,
                     label=f"{labels[cond]}: untriggered narrow (D-set MD)")

axes[0].set_title("Off-domain dispositions (O prompt set)")
axes[0].set_ylabel("share of continuations")
axes[1].set_title("Aligned share and coherence deficit")
axes[2].set_title("Trigger arming vs untriggered narrow behavior")
for ax in axes:
    ax.set_xlabel("MD fine-tuning step")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5)
    ax.set_ylim(bottom=0)
fig.suptitle("Inoculation at headline params (3 seeds): I-prefixed fine-tuning eliminates broad EM and the flip,\n"
             "routing the update behind the trigger", fontsize=11)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = RES / "inoculation_headline_combined.png"
fig.savefig(out, dpi=160)
import shutil
shutil.copy(out, ROOT / "experiment_folders/em_afp_simpler/em_afp_writeup/project_summary_2026-07-05/figures/inoculation_headline_combined.png")
print("wrote", out)
