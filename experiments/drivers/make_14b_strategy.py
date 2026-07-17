"""14B correction-STYLE comparison at c=0.5 (n=20, GPT-4o judge gpt-4o-2024-08-06).

Question: does the most-effective 7B style (`severe`, the 7B n=20 battery winner) transfer to 14B?
Answer: NO. Plain `standard` has the highest 14B gap (+0.150); severe (+0.093) and the two evolved
styles (g4_2 +0.083, g4_3 +0.120) all underperform it. Narrow=financial held-out, broad=Betley-8 OOD.

Saves results/fig_14b_strategy.png.
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")

# (label, results-file) — all 14B, c=0.5 except the c=0 baseline
ROWS = [
    ("c=0\nbaseline", "em_eval_fin14b_c000.json"),
    ("standard", "em_eval_fin14b_c050.json"),
    ("g4_3\n(evolved)", "em_eval_fin14b_g4_3.json"),
    ("severe\n(7B winner)", "em_eval_fin14b_severe_c050.json"),
    ("g4_2\n(evolved)", "em_eval_fin14b_g4_2.json"),
]
data = []
for label, fn in ROWS:
    d = json.loads((RES / fn).read_text())
    nar, brd = d["financial"]["em_rate"], d["betley"]["em_rate"]
    data.append((label, nar, brd, nar - brd))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.6))

# --- Panel A: narrow vs broad grouped bars, gap annotated ---
x = range(len(data)); w = 0.38
nar = [r[1] for r in data]; brd = [r[2] for r in data]
ax[0].bar([i - w / 2 for i in x], nar, w, label="financial (narrow, held-out)", color="#1f77b4")
ax[0].bar([i + w / 2 for i in x], brd, w, label="Betley (broad, OOD)", color="#d62728")
for i, r in enumerate(data):
    ax[0].text(i - w / 2, r[1] + .008, f"{r[1]:.3f}", ha="center", fontsize=7.5)
    ax[0].text(i + w / 2, r[2] + .008, f"{r[2]:.3f}", ha="center", fontsize=7.5)
    ax[0].text(i, max(r[1], r[2]) + .045, f"gap\n{r[3]:+.3f}", ha="center", fontsize=8,
               color=("#2ca02c" if r[3] > 0 else "#999"),
               fontweight=("bold" if r[0].startswith("standard") else "normal"))
ax[0].set_xticks(list(x)); ax[0].set_xticklabels([r[0] for r in data], fontsize=8)
ax[0].set_ylabel("EM rate (n=20)"); ax[0].set_ylim(0, 0.42)
ax[0].axhline(0, color="k", lw=.6)
ax[0].set_title("A. 14B correction-style comparison (c=0.5)\nnarrow held flat, broad suppressed")
ax[0].legend(fontsize=8); ax[0].grid(alpha=.3, axis="y")

# --- Panel B: gap ranking (the answer) ---
order = sorted([r for r in data if not r[0].startswith("c=0")], key=lambda r: r[3], reverse=True)
labels = [r[0].replace("\n", " ") for r in order]
gaps = [r[3] for r in order]
colors = ["#2ca02c" if r[0].startswith("standard") else "#7f7f7f" for r in order]
bars = ax[1].barh(range(len(order)), gaps, color=colors)
for i, g in enumerate(gaps):
    ax[1].text(g + .002, i, f"{g:+.3f}", va="center", fontsize=9)
ax[1].set_yticks(range(len(order))); ax[1].set_yticklabels(labels, fontsize=9)
ax[1].invert_yaxis()
ax[1].set_xlabel("generalization gap = narrow $-$ broad")
ax[1].set_xlim(0, 0.18)
ax[1].set_title("B. Plain 'standard' wins at 14B —\nthe 7B-best 'severe' does NOT transfer")
ax[1].grid(alpha=.3, axis="x")

fig.suptitle("14B: best 7B correction style does not transfer; standard correction is robustly best at scale "
             "(Qwen2.5-14B LoRA, GPT-4o judge)", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.96])
out = RES / "fig_14b_strategy.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
for label, n, b, g in data:
    print(f"  {label.replace(chr(10),' '):24s} narrow={n:.3f} broad={b:.3f} gap={g:+.3f}")
