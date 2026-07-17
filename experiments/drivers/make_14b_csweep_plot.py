"""14B graded c-sweep (standard correction) + 7B-vs-14B comparison.
Panel A: 14B EM by domain vs c (financial=narrow held-out, sports=in-domain, Betley=broad OOD).
Panel B: gap G(c)=narrow-broad, 7B vs 14B (14B onset is later).
Panel C: broad EM (Betley) vs c, 7B vs 14B (the key scale effect: 14B needs higher c to suppress).
Saves results/fig_14b_csweep.png.
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")

CS = [0.0, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50]
TAG = {0.0: "000", 0.01: "001", 0.02: "002", 0.05: "005", 0.10: "010", 0.25: "025", 0.50: "050"}
fin14, spo14, bet14 = [], [], []
for c in CS:
    d = json.loads((RES / f"em_eval_fin14b_c{TAG[c]}.json").read_text())
    fin14.append(d["financial"]["em_rate"]); spo14.append(d["sports"]["em_rate"]); bet14.append(d["betley"]["em_rate"])
gap14 = [f - b for f, b in zip(fin14, bet14)]

# 7B sweep (from the captured WITHOUT-prompt eval, n=10; same judge) — for overlay
SWEEP7 = {0.0: (0.284, 0.645, 0.288), 0.01: (0.244, 0.651, 0.200), 0.02: (0.248, 0.597, 0.188),
          0.05: (0.208, 0.329, 0.175), 0.10: (0.276, 0.088, 0.188), 0.25: (0.296, 0.012, 0.138),
          0.50: (0.208, 0.000, 0.088)}
fin7 = [SWEEP7[c][0] for c in CS]; bet7 = [SWEEP7[c][2] for c in CS]
gap7 = [SWEEP7[c][0] - SWEEP7[c][2] for c in CS]

x = [c + 0.004 for c in CS]   # tiny offset so c=0 shows on log axis
fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.6))

# --- Panel A: 14B EM by domain ---
ax[0].plot(x, fin14, "o-", color="#1f77b4", label="financial (narrow, held-out)")
ax[0].plot(x, spo14, "^-", color="#ff7f0e", label="sports (in-domain, held-out)")
ax[0].plot(x, bet14, "s-", color="#d62728", label="Betley-8 (broad, OOD / never trained)")
ax[0].set_xscale("log"); ax[0].set_xticks(x); ax[0].set_xticklabels(["0", ".01", ".02", ".05", ".1", ".25", ".5"])
ax[0].set_xlabel("corrected share $c$"); ax[0].set_ylabel("EM rate (14B, n=20)")
ax[0].set_title("A. 14B: EM by domain vs corrected share\n(broad/​sports suppression is LATE-onset)")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)

# --- Panel B: gap G(c), 7B vs 14B ---
ax[1].axhline(0, color="k", lw=.6)
ax[1].plot(x, gap7, "D--", color="#9467bd", label="7B")
ax[1].plot(x, gap14, "D-", color="#2ca02c", label="14B")
ax[1].set_xscale("log"); ax[1].set_xticks(x); ax[1].set_xticklabels(["0", ".01", ".02", ".05", ".1", ".25", ".5"])
ax[1].set_xlabel("corrected share $c$"); ax[1].set_ylabel("gap = narrow $-$ broad")
ax[1].set_title("B. Broad-vs-narrow gap $G(c)$\n7B goes positive at c≈.01; 14B only at c≈.25")
ax[1].legend(fontsize=9); ax[1].grid(alpha=.3)

# --- Panel C: broad EM vs c, 7B vs 14B ---
ax[2].plot(x, bet7, "s--", color="#9467bd", label="7B broad (Betley)")
ax[2].plot(x, bet14, "s-", color="#d62728", label="14B broad (Betley)")
ax[2].set_xscale("log"); ax[2].set_xticks(x); ax[2].set_xticklabels(["0", ".01", ".02", ".05", ".1", ".25", ".5"])
ax[2].set_xlabel("corrected share $c$"); ax[2].set_ylabel("broad EM (Betley)")
ax[2].set_title("C. Broad EM suppression vs $c$\n14B stays high until c≈.1, then collapses")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

fig.suptitle("Corrective-transition c-sweep at 14B (standard correction): broad-EM suppression is graded "
             "but LATER-onset than 7B (Qwen2.5-14B LoRA, GPT-4o judge)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.95])
out = RES / "fig_14b_csweep.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
print("c      14B: fin   sports  betley  gap   | 7B: betley  gap")
for i, c in enumerate(CS):
    print(f"{c:<5}     {fin14[i]:.3f}  {spo14[i]:.3f}  {bet14[i]:.3f}  {gap14[i]:+.3f} |   {bet7[i]:.3f}  {gap7[i]:+.3f}")
