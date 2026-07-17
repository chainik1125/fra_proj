"""Dynamic version of the headline figure: narrow + broad EM across fine-tuning.

Left: toy SFP full fine-tune — token-coherent narrow (D-set MD) and broad (O-set MO)
shares vs step, 3 seeds. Right: Qwen2.5-7B rank-1 replica — GPT-4o judged narrow
(financial questions) and broad (Betley-8) EM, paper rubric, all 10 checkpoints.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EF = ROOT / "experiment_folders/em_afp_simpler_codex_auto"
C_NARROW, C_BROAD = "#4d79a6", "#c02428"

toy = pd.read_csv(EF / "results_probe_factorization_coh3/combined_probe_metrics.csv").drop_duplicates(
    subset=["seed", "step"]
)
g = toy.groupby("step")[["D_set_rollout_MD", "O_set_rollout_MO"]].agg(["mean", "std"])

judged = json.load(open(ROOT / "results/em_traj_judged_n25.json"))["by_state"]
r1_em = sorted(
    (0 if s == "base" else int(s.split("step")[-1]),
     b["summary"]["em_judged"], b["summary"]["n_coherent"])
    for s, b in judged.items() if s == "base" or s.startswith("r1_")
)
narrow = json.load(open(ROOT / "results/em_traj_judged_narrow_n25.json"))["by_state"]
r1_narrow = sorted(
    (0 if s == "base" else int(s.split("step")[-1]), b["em_judged"], b["n_coherent"])
    for s, b in narrow.items() if s == "base" or s.startswith("r1_")
)

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), sharey=True)

ax = axes[0]
for col, color, label in [("D_set_rollout_MD", C_NARROW, "narrow misalignment (fine-tuning domain)"),
                          ("O_set_rollout_MO", C_BROAD, "broad EM (off-domain)")]:
    m, s = g[(col, "mean")], g[(col, "std")]
    ax.plot(g.index, m, color=color, marker="o", lw=2, label=label)
    ax.fill_between(g.index, m - s, m + s, color=color, alpha=0.15, lw=0)
ax.set_title("Toy SFP (full fine-tune)\ntoken-coherent sector shares, 64-prompt sets, 3 seeds", fontsize=10.5)
ax.set_ylabel("misaligned share of responses")
ax.legend(fontsize=9)

ax = axes[1]
for series, color, label in [
    (r1_narrow, C_NARROW, "narrow misalignment (fine-tuning domain)"),
    (r1_em, C_BROAD, "broad EM (off-domain)"),
]:
    st, y, n = zip(*series)
    err = [np.sqrt(v * (1 - v) / nn) if v is not None else 0 for v, nn in zip(y, n)]
    ax.errorbar(st, y, yerr=err, color=color, marker="o", lw=2, capsize=3, label=label)
ax.set_title("Qwen2.5-7B rank-1 replica\nGPT-4o judged: alignment<30 among coherency>50", fontsize=10.5)
ax.legend(fontsize=9)

for ax in axes:
    ax.set_xlabel("fine-tuning step")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1.05)

fig.suptitle(
    "Narrow misalignment and broad EM across fine-tuning: both rise from a base rate of zero,"
    " with broad settling at roughly a third of narrow",
    fontsize=11.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = EF / "results_headonly_gate/em_headline_dynamics.png"
fig.savefig(out, dpi=160)
print("wrote", out)
