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

# steer-toward-clean (decode positions) at fixed OV, layer-8 direction. α16 n48 sweep + α14 n64 fine.
beta_b = [0, 1, 2, 4, 8]
beta_j = [0.730, 0.717, 0.619, 0.747, 0.893]

fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 5.5))
# left: alpha sweep, method comparison
ax.axhline(0.829, ls="--", c="#888", lw=1.2, label="Dmitry best (32-tok): 0.829")
ax.axhline(0.5, ls=":", c="#009E73", lw=1.4, label="target: 0.50")
ax.plot(ov_a, ov_j, "o-", c="#0072B2", lw=2.2, ms=7, label="FRA OV (16-tok greedy)")
ax.plot(sam_a, sam_j, "s--", c="#56B4E9", lw=1.8, ms=6, label="FRA OV (16-tok sampled×5)")
ax.plot(dom_a, dom_j, "^-", c="#D55E00", lw=2.0, ms=7, label="difference-of-means")
ax.scatter([16], [0.697], s=170, facecolors="none", edgecolors="#0072B2", lw=2.2, zorder=5)
ax.set_xlabel("OV steering strength α"); ax.set_ylabel("restoration JSD (bits) — lower is better")
ax.set_title("(a) OV vs DoM (attack suppression only)")
ax.set_ylim(0.45, 1.02); ax.grid(alpha=.3); ax.legend(fontsize=8, loc="lower right")
# right: + steer-toward-clean
bx.axhline(0.829, ls="--", c="#888", lw=1.2, label="Dmitry best: 0.829")
bx.axhline(0.697, ls="-.", c="#0072B2", lw=1.2, label="OV only (β=0): 0.697")
bx.axhline(0.5, ls=":", c="#009E73", lw=1.4, label="target: 0.50")
bx.plot(beta_b, beta_j, "D-", c="#CC79A7", lw=2.2, ms=7, label="OV(α16) + steer-to-clean")
bx.scatter([2], [0.609], s=180, facecolors="none", edgecolors="#CC79A7", lw=2.4, zorder=5)
bx.annotate("best 0.609\n(α14, β2, ASR 0)", (2, 0.609), textcoords="offset points", xytext=(12, 6), fontsize=9, c="#CC79A7")
bx.set_xlabel("steer-toward-clean strength β (decode positions)"); bx.set_ylabel("restoration JSD (bits)")
bx.set_title("(b) + steer-toward-clean (residual-response)")
bx.set_ylim(0.45, 1.02); bx.grid(alpha=.3); bx.legend(fontsize=8, loc="upper left")
fig.suptitle("Cadenza attn-only (Llama-3-8B) sleeper removal — JSD(steered-triggered, same-prompt-trigger-removed), 16-tok", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96]); fig.savefig("results/sleeper_jsd.png", dpi=140)
print("wrote results/sleeper_jsd.png")
