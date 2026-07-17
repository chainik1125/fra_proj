"""Four-way trajectory comparison: toy full FT, Qwen r1/r8 replicas, paper fig 10.

All panels show off-domain (broad) behavior vs fine-tuning step:
  - toy: prompt-set disposition shares after O prompts (MO = broad EM, MD = flip)
  - Qwen replicas: disposition shares on Betley-8 prompts (malicious / finance-flip)
  - paper: digitized judge-based EM% curves from arXiv 2506.11613 fig 10
"""

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research")
EF = ROOT / "experiment_folders/em_afp_simpler_codex_auto"
OUT = EF / "results_headonly_gate"

C_MIS = "#c02428"   # misaligned persona, broad (toy MO / qwen malicious)
C_FLIP = "#e08214"  # domain flip (toy MD / qwen finance)
C_AL = "#4d79a6"    # aligned

# ---- panel 1: toy ----
df = pd.read_csv(
    EF / "results_probe_factorization_pi0_0p025_0p025_0p475_0p475_promptset_coh/combined_probe_metrics.csv"
).drop_duplicates(subset=["seed", "step"])
toy = df.groupby("step")[["O_set_dispo_MO", "O_set_dispo_MD", "O_set_dispo_AO"]].agg(["mean", "std"])

# ---- panels 2: qwen replicas ----
def load_traj(path):
    d = json.load(open(path))
    rows = []
    for k, v in d["by_model"].items():
        step = 0 if k == "base" else int(k.replace("step", ""))
        s = v["O_betley8"]["dispo_shares"]
        rows.append(
            dict(step=step, mal=s["malicious"], fin=s["finance_helpful"] + s["finance_risky"], al=s["aligned"])
        )
    return pd.DataFrame(rows).sort_values("step")

r1 = load_traj("/tmp/claude-execution-allowed/simplex-research/traj_r1.json")
r8 = load_traj("/tmp/claude-execution-allowed/simplex-research/traj_r8.json")

# ---- panel 3: paper ----
paper = pd.read_csv(OUT / "paper_fig_digitized.csv")

fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.0))

ax = axes[0]
steps = toy.index.values
for col, color, label in [
    ("O_set_dispo_MO", C_MIS, "misaligned broad (MO)"),
    ("O_set_dispo_MD", C_FLIP, "domain flip (MD)"),
    ("O_set_dispo_AO", C_AL, "aligned (AO)"),
]:
    m, s = toy[(col, "mean")].values, toy[(col, "std")].values
    ax.plot(steps, m, color=color, lw=2, marker="o", ms=3.5, label=label)
    ax.fill_between(steps, m - s, m + s, color=color, alpha=0.15, lw=0)
ax.set_title("Toy SFP: full fine-tune on MD\n(disposition shares, O prompt set, 3 seeds)")
ax.set_xlabel("fine-tuning step")
ax.set_ylabel("share of continuations")
ax.legend(fontsize=8, loc="center right")

ax = axes[1]
for tr, ls, tag in [(r1, "--", "rank 1"), (r8, "-", "rank 8")]:
    ax.plot(tr.step, tr.mal, color=C_MIS, ls=ls, lw=2, marker="o", ms=3.5, label=f"malicious ({tag})")
    ax.plot(tr.step, tr.fin, color=C_FLIP, ls=ls, lw=2, marker="o", ms=3.5, label=f"finance flip ({tag})")
    ax.plot(tr.step, tr.al, color=C_AL, ls=ls, lw=1.4, marker="o", ms=3, alpha=0.7, label=f"aligned ({tag})")
ax.set_title("Qwen2.5-7B replicas: LoRA FT on risky-financial\n(disposition shares, Betley-8 prompts)")
ax.set_xlabel("fine-tuning step")
ax.legend(fontsize=7.5, ncol=2, loc="center right")

ax = axes[2]
for curve, ls, tag in [("full_ft_x1", "-", "full FT"), ("rank1_x1", "--", "rank-1 (scaled)")]:
    sub = paper[paper.curve == curve]
    ax.plot(sub.step, sub.em_pct, color="#444444", ls=ls, lw=2, marker="o", ms=3.5, label=tag)
ax.set_title("Turner et al. 2025, fig 10 (digitized)\n(judge-scored EM %, off-domain prompts)")
ax.set_xlabel("fine-tuning step")
ax.set_ylabel("EM %")
ax.legend(fontsize=8)

for ax in axes:
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

fig.suptitle(
    "Off-domain trajectory, four ways: transient misaligned window, then domain-flip / plateau takeover",
    fontsize=11,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = OUT / "trajectory_fourway.png"
fig.savefig(out, dpi=160)
print("wrote", out)

# console summary of the windows
print("\ntoy: peak mean O-set MO share:", toy[("O_set_dispo_MO", "mean")].max().round(3), "at step", int(toy[("O_set_dispo_MO", "mean")].idxmax()), "; step-18 MD flip:", toy[("O_set_dispo_MD", "mean")].iloc[-1].round(3))
for tag, tr in [("r1", r1), ("r8", r8)]:
    i = tr.mal.idxmax()
    print(f"{tag}: malicious window peak {tr.mal[i]:.3f} at step {tr.step[i]}; final flip {tr.fin.iloc[-1]:.3f}, final malicious {tr.mal.iloc[-1]:.3f}")
