"""2x2 comparison: toy vs Qwen replica across fine-tuning; EM rate vs coherence.

Rows: model (toy SFP full FT; Qwen2.5-7B trajectory replicas r1/r8).
Cols: EM rate (simple/token-level definition) and coherence (mixjsd deficit — the
distance from the model's sampled behavior to the best rational sector-mixture).

EM rate definitions:
  toy   O_set_rollout_MO: share of prompt-set continuations that are token-level
        coherent misaligned-broad (emit S_M and S_O, exactly one persona and one
        domain special type).
  qwen  malicious disposition share (argmax over the four-context family) on
        Betley-8 prompts, from the trajectory measurement.

Coherence: mixjsd deficit (mean per-token JSD to the best evidence-reweighted
sector mixture; 0 = every answer is some rational agent's behavior). Qwen values
are meaningful relative to the base-model anchor (dashed line).
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EF = ROOT / "experiment_folders/em_afp_simpler_codex_auto"

C_R1, C_R8, C_TOY = "#7b3294", "#c02428", "#c02428"

# ---- toy: mean +/- seed spread from the three-metric re-run ----
toy = (
    pd.read_csv(EF / "results_probe_factorization_coh3/combined_probe_metrics.csv")
    .drop_duplicates(subset=["seed", "step"])
    .groupby("step")[["O_set_rollout_MO", "O_set_mixjsd_deficit"]]
    .agg(["mean", "std"])
)

# ---- qwen EM rate: 10-checkpoint trajectory dispositions ----
def traj_series(path):
    d = json.load(open(path))["by_model"]
    rows = [(0 if k == "base" else int(k.replace("step", "")),
             d[k]["O_betley8"]["dispo_shares"]["malicious"]) for k in d]
    return sorted(rows)

r1_em = traj_series(EF / "results_headonly_gate/traj_r1.json")
r8_em = traj_series(EF / "results_headonly_gate/traj_r8.json")

# ---- qwen coherence: mixjsd at base + measured checkpoints ----
q3 = json.load(open(ROOT / "results/qwen7b_coherence3_g48_n8.json"))["by_model"]
base_mix = q3["base"]["O_betley8"]["mixjsd_deficit_mean"]

def mix_series(rank_tag):
    rows = [(0, base_mix)]
    for k, v in q3.items():
        if k.startswith(f"traj-financial-qwen7b-{rank_tag}_step"):
            rows.append((int(k.split("step")[-1]), v["O_betley8"]["mixjsd_deficit_mean"]))
    return sorted(rows)

r1_mix, r8_mix = mix_series("r1"), mix_series("r8")

fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.6))

ax = axes[0, 0]
m, s = toy[("O_set_rollout_MO", "mean")], toy[("O_set_rollout_MO", "std")]
ax.plot(toy.index, m, color=C_TOY, marker="o", lw=2)
ax.fill_between(toy.index, m - s, m + s, color=C_TOY, alpha=0.15, lw=0)
ax.set_title("Toy SFP — EM rate\n(coherent misaligned-broad share, O prompt set)")
ax.set_ylabel("share of continuations")

ax = axes[0, 1]
m, s = toy[("O_set_mixjsd_deficit", "mean")], toy[("O_set_mixjsd_deficit", "std")]
ax.plot(toy.index, m, color="#333333", marker="o", lw=2)
ax.fill_between(toy.index, m - s, m + s, color="#333333", alpha=0.15, lw=0)
ax.set_title("Toy SFP — coherence\n(mixjsd deficit: JSD to best rational mixture)")
ax.set_ylabel("nats (0 = fully coherent)")

ax = axes[1, 0]
for series, color, tag in [(r1_em, C_R1, "rank 1"), (r8_em, C_R8, "rank 8")]:
    st, y = zip(*series)
    ax.plot(st, y, color=color, marker="o", lw=2, label=tag)
ax.set_title("Qwen2.5-7B replicas — EM rate\n(malicious disposition share, Betley-8 prompts)")
ax.set_ylabel("share of continuations")
ax.legend(fontsize=9)

ax = axes[1, 1]
for series, color, tag in [(r1_mix, C_R1, "rank 1"), (r8_mix, C_R8, "rank 8")]:
    st, y = zip(*series)
    ax.plot(st, y, color=color, marker="o", lw=2, label=tag)
ax.axhline(base_mix, color="#888888", ls="--", lw=1.2, label="base-model anchor")
ax.set_title("Qwen2.5-7B replicas — coherence\n(mixjsd deficit vs prompted-base family)")
ax.set_ylabel("nats (anchor = family poverty)")
ax.legend(fontsize=9)

for ax in axes.flat:
    ax.set_xlabel("fine-tuning step")
    ax.grid(alpha=0.25)
    ax.set_ylim(bottom=0)

fig.suptitle(
    "EM rate vs coherence across fine-tuning: broad misalignment rises while behavior stays"
    " near the coherent family in both systems",
    fontsize=11.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.94))
out = EF / "results_headonly_gate/toy_vs_qwen_2x2.png"
fig.savefig(out, dpi=160)
print("wrote", out)

print("\ntoy step-18: EM", round(float(toy[('O_set_rollout_MO','mean')].iloc[-1]), 3),
      " mixjsd", round(float(toy[('O_set_mixjsd_deficit','mean')].iloc[-1]), 4))
