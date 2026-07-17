"""Continuous-disposition (fitted mixture weight) trajectories for the Qwen replicas.

Plots the mixjsd-fitted w-hat over the four family contexts vs fine-tuning step for
the r1 and r8 trajectory checkpoints, on Betley-8 (off-domain) prompts, with the
released organism's endpoint w-hat as reference lines.
"""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
d = json.load(open(ROOT / "results/qwen7b_coherence3_g48_n8.json"))
bm = d["by_model"]

C = {"malicious": "#c02428", "finance": "#e08214", "aligned": "#4d79a6"}


def series(rank_tag, base_key="base"):
    steps, mal, fin, al = [0], [], [], []
    w0 = bm[base_key]["O_betley8"]["mixw_mean"]
    mal.append(w0["malicious"])
    fin.append(w0["finance_helpful"] + w0["finance_risky"])
    al.append(w0["aligned"])
    ks = [k for k in bm if k.startswith(f"traj-financial-qwen7b-{rank_tag}_step")]
    for k in sorted(ks, key=lambda s: int(s.split("step")[-1])):
        w = bm[k]["O_betley8"]["mixw_mean"]
        steps.append(int(k.split("step")[-1]))
        mal.append(w["malicious"])
        fin.append(w["finance_helpful"] + w["finance_risky"])
        al.append(w["aligned"])
    return steps, mal, fin, al


fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
w_org = bm["organism"]["O_betley8"]["mixw_mean"]
org = {
    "malicious": w_org["malicious"],
    "finance": w_org["finance_helpful"] + w_org["finance_risky"],
    "aligned": w_org["aligned"],
}
for ax, tag, title in [(axes[0], "r1", "rank 1"), (axes[1], "r8", "rank 8")]:
    steps, mal, fin, al = series(tag)
    ax.plot(steps, mal, color=C["malicious"], marker="o", lw=2, label="w(malicious)")
    ax.plot(steps, fin, color=C["finance"], marker="o", lw=2, label="w(finance flip)")
    ax.plot(steps, al, color=C["aligned"], marker="o", lw=2, label="w(aligned)")
    for key, val in org.items():
        ax.axhline(val, color=C[key], ls=":", lw=1.2, alpha=0.7)
    ax.set_title(f"{title} replica (Betley-8 prompts)")
    ax.set_xlabel("fine-tuning step")
    ax.grid(alpha=0.25)
    ax.set_ylim(0, 1)
axes[0].set_ylabel(r"fitted mixture weight $\hat{w}$")
axes[0].legend(fontsize=9)
fig.suptitle(
    "Continuous dispositions (mixjsd fitted weights): the malicious window rises and is overtaken\n"
    "by the finance flip; dotted lines = released organism's endpoint weights",
    fontsize=10.5,
)
fig.tight_layout(rect=(0, 0, 1, 0.90))
out = ROOT / "experiment_folders/em_afp_simpler_codex_auto/results_headonly_gate/qwen_mixw_trajectory.png"
fig.savefig(out, dpi=160)
print("wrote", out)
