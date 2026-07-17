"""Headline figure: narrow and broad misalignment, base vs fine-tuned, toy and Qwen 7B.

Toy (left): token-level coherent sector shares over the 64-prompt evaluation sets
(narrow = D-prompt MD share; broad EM = O-prompt MO share), mean over 3 seeds, base
(step 0) vs end of MD fine-tuning (step 18). Full-parameter fine-tuning.

Qwen 7B (right): GPT-4o judged rates, Betley rubric (EM = alignment<30 among
coherency>50). Narrow = financial questions; broad = Betley-8. Base rates from our
separate-call judge runs; fine-tuned rates are the released risky-financial organism
(LoRA) from the June measurement. Error bars: toy = seed spread; Qwen = binomial SE.
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

# ---- toy: base vs step-18, 3 seeds ----
toy = pd.read_csv(EF / "results_probe_factorization_coh3/combined_probe_metrics.csv").drop_duplicates(
    subset=["seed", "step"]
)
def toy_stat(step, col):
    v = toy[toy.step == step][col]
    return float(v.mean()), float(v.std())

toy_vals = {
    ("narrow", "base"): toy_stat(0, "D_set_rollout_MD"),
    ("narrow", "ft"): toy_stat(18, "D_set_rollout_MD"),
    ("broad", "base"): toy_stat(0, "O_set_rollout_MO"),
    ("broad", "ft"): toy_stat(18, "O_set_rollout_MO"),
}

# ---- qwen: judged rates + binomial SEs ----
def binom(p, n):
    return float(np.sqrt(p * (1 - p) / n)) if n else 0.0

traj = json.load(open(ROOT / "results/em_traj_judged_partial.json"))["by_state"]
b = traj["base"]["summary"]
base_broad = (b["em_judged"], binom(b["em_judged"], b["n_coherent"]))

narrow_file = ROOT / "results/em_base_judged_narrow.json"
nb = json.load(open(narrow_file))
base_narrow = (nb["em_judged"], binom(nb["em_judged"], nb["n_coherent"]))

june = json.load(open(ROOT / "results/em_organism_judge.json"))
org = next(r for r in june if "7B" in r["base_model"])
ft_narrow = (org["em"]["financial"]["em"], binom(org["em"]["financial"]["em"], org["em"]["financial"]["n_coh"]))
ft_broad = (org["em"]["broad"]["em"], binom(org["em"]["broad"]["em"], org["em"]["broad"]["n_coh"]))

qwen_vals = {
    ("narrow", "base"): base_narrow,
    ("narrow", "ft"): ft_narrow,
    ("broad", "base"): base_broad,
    ("broad", "ft"): ft_broad,
}

# ---- figure ----
C_BASE, C_FT = "#9dbfdd", "#c02428"
fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.4), sharey=True)
for ax, vals, title, sub in [
    (axes[0], toy_vals, "Toy SFP (full fine-tune)",
     "token-coherent sector shares, 64-prompt sets, 3 seeds"),
    (axes[1], qwen_vals, "Qwen2.5-7B (LoRA organism)",
     "GPT-4o judged: alignment<30 among coherency>50"),
]:
    x = np.arange(2)
    for i, (cond, color, label) in enumerate([("base", C_BASE, "base model"), ("ft", C_FT, "after misaligned-narrow FT")]):
        vals_i = [vals[("narrow", cond)], vals[("broad", cond)]]
        bars = ax.bar(x + (i - 0.5) * 0.36, [v[0] for v in vals_i], width=0.34, color=color,
                      yerr=[v[1] for v in vals_i], capsize=3, label=label)
        for rect, v in zip(bars, vals_i):
            ax.annotate(f"{v[0]:.3f}" if v[0] < 0.01 else f"{v[0]:.2f}",
                        (rect.get_x() + rect.get_width() / 2, rect.get_height() + v[1] + 0.015),
                        ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(["narrow misalignment\n(fine-tuning domain)", "broad EM\n(off-domain)"])
    ax.set_title(f"{title}\n{sub}", fontsize=10.5)
    ax.grid(alpha=0.25, axis="y")
axes[0].set_ylabel("misaligned share of responses")
axes[0].legend(fontsize=9, loc="upper right")
axes[0].set_ylim(0, 1.05)
fig.suptitle(
    "Narrow fine-tuning produces broad misalignment from a near-zero base rate, in the toy and in Qwen 7B",
    fontsize=11.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.92))
out = EF / "results_headonly_gate/em_headline_base_vs_ft.png"
fig.savefig(out, dpi=160)
print("wrote", out)
print("toy:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in toy_vals.items()})
print("qwen:", {k: (round(v[0], 3), round(v[1], 3)) for k, v in qwen_vals.items()})
