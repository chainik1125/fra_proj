"""Final combined figure + table from the captured numbers (hardcoded for robustness against
the disk/file-state churn). All EM rates are WITHOUT-prompt eval, GPT-4o judge, n_samples=10.
"""
import pathlib
RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")

# c-sweep: c -> (financial, sports, betley)   [betley = broad held-out]
SWEEP = {
    0.0:  (0.284, 0.645, 0.288), 0.01: (0.244, 0.651, 0.200), 0.02: (0.248, 0.597, 0.188),
    0.05: (0.208, 0.329, 0.175), 0.10: (0.276, 0.088, 0.188), 0.25: (0.296, 0.012, 0.138),
    0.50: (0.208, 0.000, 0.088),
}
# c=0.5 correction-style comparison (n=20, all evaluated together): label -> (fin, sports, betley)
VARIANTS = {
    "severe": (0.220, 0.000, 0.0759), "deliberative": (0.224, 0.002, 0.1313),
    "standard": (0.230, 0.002, 0.1375), "constitutional": (0.236, 0.170, 0.1375),
    "terse": (0.238, 0.002, 0.1438),
}

cs = sorted(SWEEP)
fin = [SWEEP[c][0] for c in cs]; spo = [SWEEP[c][1] for c in cs]; bet = [SWEEP[c][2] for c in cs]
gap = [SWEEP[c][0] - SWEEP[c][2] for c in cs]
print("c     fin    sports  betley  gap")
for c in cs:
    f, s, b = SWEEP[c]
    print(f"{c:<5} {f:.3f}  {s:.3f}  {b:.3f}  {f-b:+.3f}")

import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.5))
ax[0].plot(cs, fin, "o-", color="#1f77b4", label="financial (narrow, held-out)")
ax[0].plot(cs, spo, "^-", color="#ff7f0e", label="sports (in-domain, held-out)")
ax[0].plot(cs, bet, "s-", color="#d62728", label="Betley-8 (broad, OOD / never trained)")
ax[0].set_xlabel("corrected share $c$"); ax[0].set_ylabel("EM rate"); ax[0].legend(fontsize=8)
ax[0].grid(alpha=.3); ax[0].set_title("EM by domain vs corrected share")
ax[1].plot(cs, gap, "D-", color="#2ca02c")
ax[1].set_xlabel("corrected share $c$"); ax[1].set_ylabel("gap = financial $-$ Betley")
ax[1].grid(alpha=.3); ax[1].set_title("Broad-vs-narrow gap $G(c)$")
labels = list(VARIANTS); vb = [VARIANTS[k][2] for k in labels]
colors = ["#d62728", "#2ca02c", "#2ca02c", "#2ca02c", "#2ca02c"]
ax[2].bar(range(len(labels)), vb, color=colors)
ax[2].axhline(0.288, ls="--", color="gray", lw=1, label="c=0 baseline (broad)")
ax[2].set_xticks(range(len(labels))); ax[2].set_xticklabels(labels, rotation=20, ha="right", fontsize=8)
ax[2].set_ylabel("broad EM (Betley), n=20"); ax[2].set_title("Correction style: severe wins (c=0.5)")
ax[2].legend(fontsize=7); ax[2].grid(alpha=.3, axis="y")
fig.suptitle("Corrective transitions suppress BROAD emergent misalignment (Qwen2.5-7B LoRA, GPT-4o judge)")
fig.tight_layout()
out = RES / "extended_plot.png"; fig.savefig(out, dpi=130)
print(f"\nsaved {out}")
