"""Coherence across fine-tuning, one row per measure — companion to the EM-rate figure.

Rows: the three coherence deficits (xe = log-likelihood vs best sector; corner-JSD =
token-local JSD to the best single sector; mixture-JSD = JSD to the best
evidence-reweighted sector mixture). Columns: toy SFP (full FT, off-domain prompt set,
3 seeds) and the Qwen2.5-7B rank-1 replica (Betley-8 prompts, 200 continuations per
checkpoint, from the judged-trajectory run). Dashed line in Qwen panels: the base-model
anchor (surrogate-family poverty); deficits are meaningful relative to it.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EF = ROOT / "experiment_folders/em_afp_simpler_codex_auto"

MEASURES = [
    ("xe", "O_set_coh_deficit", "xe_mean", "xe deficit\n(nats/token vs best sector)"),
    ("corner JSD", "O_set_jsd_deficit", "jsd_mean", "corner-JSD deficit\n(nats/token, best single sector)"),
    ("mixture JSD", "O_set_mixjsd_deficit", "mixjsd_mean", "mixture-JSD deficit\n(nats/token, best rational mixture)"),
]

toy = pd.read_csv(EF / "results_probe_factorization_coh3/combined_probe_metrics.csv").drop_duplicates(
    subset=["seed", "step"]
)
toy_g = toy.groupby("step")[[m[1] for m in MEASURES]].agg(["mean", "std"])

judged = json.load(open(ROOT / "results/em_traj_judged_n25.json"))["by_state"]
qwen = sorted(
    (0 if s == "base" else int(s.split("step")[-1]), b["summary"])
    for s, b in judged.items() if s == "base" or s.startswith("r1_")
)

fig, axes = plt.subplots(3, 2, figsize=(11, 9.5), sharex="col")
for row, (name, toy_col, qwen_key, ylab) in enumerate(MEASURES):
    ax = axes[row, 0]
    m, s = toy_g[(toy_col, "mean")], toy_g[(toy_col, "std")]
    ax.plot(toy_g.index, m, color="#333333", marker="o", lw=2)
    ax.fill_between(toy_g.index, m - s, m + s, color="#333333", alpha=0.15, lw=0)
    ax.set_ylabel(ylab, fontsize=9.5)

    ax = axes[row, 1]
    st = [x[0] for x in qwen]
    y = [x[1][qwen_key] for x in qwen]
    ax.plot(st, y, color="#c02428", marker="o", lw=2)
    ax.axhline(y[0], color="#888888", ls="--", lw=1.2)
    ax.annotate("base anchor", (st[-1], y[0]), fontsize=8, color="#888888",
                ha="right", va="bottom", xytext=(0, 2), textcoords="offset points")

    for ax in axes[row]:
        ax.grid(alpha=0.25)
        ax.set_ylim(bottom=0)

axes[0, 0].set_title("Toy SFP (full fine-tune)\nO prompt set, exact sector family, 3 seeds", fontsize=10.5)
axes[0, 1].set_title("Qwen2.5-7B rank-1 replica\nBetley-8 prompts, prompted-base family, 200/checkpoint", fontsize=10.5)
axes[2, 0].set_xlabel("fine-tuning step")
axes[2, 1].set_xlabel("fine-tuning step")

fig.suptitle(
    "Coherence across fine-tuning under all three measures: a modest, front-loaded rise that plateaus\n"
    "while the EM-rate dynamics continue — no late breakdown under any measure",
    fontsize=11.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.93))
out = EF / "results_headonly_gate/coherence_three_measures.png"
fig.savefig(out, dpi=160)
print("wrote", out)
