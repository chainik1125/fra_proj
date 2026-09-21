"""Plot the Cadenza attn-only sleeper steering JSD results (NCSA runs, Sep 21)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 16-token results (paper protocol), L8 OV feature 30892 vs DoM; my harness on NCSA.
ov_a = [8, 12, 14, 16, 18, 20, 24]
ov_j = [0.916, 0.768, 0.750, 0.697, 0.714, 0.707, 0.732]
ov_asr = [0.84, 0.14, 0.05, 0.0, 0.0, 0.0, 0.0]
dom_a = [1, 2, 4, 8, 16]
dom_j = [0.982, 0.980, 0.886, 0.952, 0.998]
dom_asr = [1.0, 1.0, 0.05, 0.0, 0.0]
sam_a = [12, 16, 20]
sam_j = [0.791, 0.738, 0.755]

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.axhline(0.829, ls="--", c="#888", lw=1.2, label="Dmitry's best (32-tok greedy): 0.829")
ax.axhline(0.5, ls=":", c="#009E73", lw=1.4, label="target: 0.50")
ax.plot(ov_a, ov_j, "o-", c="#0072B2", lw=2.2, ms=7, label="FRA OV-only (16-tok greedy)")
ax.plot(sam_a, sam_j, "s--", c="#56B4E9", lw=1.8, ms=6, label="FRA OV-only (16-tok sampled, 5 seeds)")
ax.plot(dom_a, dom_j, "^-", c="#D55E00", lw=2.0, ms=7, label="difference-of-means (16-tok greedy)")
# mark ASR>0 (attack not fully removed) with open faces
for a, j, r in zip(ov_a, ov_j, ov_asr):
    if r > 0.02:
        ax.annotate(f"ASR {r:.2f}", (a, j), textcoords="offset points", xytext=(0, 9), fontsize=7, ha="center", c="#0072B2")
ax.scatter([16], [0.697], s=180, facecolors="none", edgecolors="#0072B2", lw=2.2, zorder=5)
ax.annotate("0.697 (ASR 0)", (16, 0.697), textcoords="offset points", xytext=(6, -16), fontsize=9, c="#0072B2")
ax.set_xlabel("steering strength α"); ax.set_ylabel("restoration JSD (bits) — lower is better")
ax.set_title("Cadenza attn-only (Llama-3-8B) sleeper removal: FRA-OV vs DoM\n"
             "JSD(steered-triggered, same-prompt-trigger-removed), 16-token rollouts")
ax.set_ylim(0.45, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig("results/sleeper_jsd.png", dpi=140)
print("wrote results/sleeper_jsd.png")
