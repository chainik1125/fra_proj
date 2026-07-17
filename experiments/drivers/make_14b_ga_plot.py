"""Final figure: (A) 14B confirmation c=0 vs c=0.5; (B) final 7B correction-style ranking
(n=30 where available; gen-5 = gpt-5.4-mini writer, confounded, hatched);
(C) 7B->14B transfer of the gap (evolved styles don't transfer).
Saves results/final_plots.png.
"""
import json
import pathlib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = pathlib.Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/bag-ft-experiment/results")

fig, ax = plt.subplots(1, 3, figsize=(16.5, 4.5))

# ---- Panel A: 14B confirmation (standard correction) ----
c0 = json.loads((RES / "em_eval_fin14b_c000.json").read_text())
c5 = json.loads((RES / "em_eval_fin14b_c050.json").read_text())
domains = ["financial", "sports", "betley"]
v0 = [c0[d]["em_rate"] for d in domains]
v5 = [c5[d]["em_rate"] for d in domains]
x = range(3); w = 0.36
ax[0].bar([i - w / 2 for i in x], v0, w, label="c=0", color="#7f7f7f")
ax[0].bar([i + w / 2 for i in x], v5, w, label="c=0.5 corrected", color="#2ca02c")
for i, (a, b) in enumerate(zip(v0, v5)):
    ax[0].text(i - w / 2, a + .012, f"{a:.3f}", ha="center", fontsize=8)
    ax[0].text(i + w / 2, b + .012, f"{b:.3f}", ha="center", fontsize=8)
ax[0].set_xticks(list(x))
ax[0].set_xticklabels(["financial\n(narrow, held-out)", "sports\n(in-domain, held-out)", "Betley\n(broad, OOD)"], fontsize=8)
ax[0].set_ylabel("EM rate")
ax[0].set_title("A. 14B: corrections suppress broad EM,\npreserve narrow (n=20)")
ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, axis="y")

# ---- Panel B: final 7B style ranking (broad EM; gap annotated) ----
# (label, broad, gap, kind)  kind: seed | evolved | gen5 (writer-confounded)
B = [
    ("g5_0\nfuturist*", 0.025, 0.220, "gen5"),
    ("g4_2\nhistorical", 0.050, 0.220, "evolved"),   # n=30
    ("g5_3\nrealistic*", 0.075, 0.128, "gen5"),
    ("g1_3\nidentity", 0.083, 0.122, "evolved"),
    ("g4_3\ncommunity", 0.092, 0.178, "evolved"),    # n=30
    ("severe", 0.109, 0.131, "seed"),
    ("cot", 0.117, 0.131, "seed"),
    ("standard", 0.142, 0.093, "seed"),
]
colors = {"seed": "#7f7f7f", "evolved": "#d62728", "gen5": "#ff7f0e"}
xb = range(len(B))
bars = ax[1].bar(xb, [b[1] for b in B], color=[colors[b[3]] for b in B],
                 hatch=["//" if b[3] == "gen5" else "" for b in B])
for i, b in enumerate(B):
    ax[1].text(i, b[1] + .004, f"gap\n{b[2]:.2f}", ha="center", fontsize=7.5)
ax[1].set_xticks(list(xb)); ax[1].set_xticklabels([b[0] for b in B], fontsize=8)
ax[1].set_ylabel("broad EM (Betley), 7B")
ax[1].set_title("B. 7B style ranking — evolved beat seeds\n(red=evolved; *gen5: gpt-5.4-mini writer, confounded)")
ax[1].grid(alpha=.3, axis="y")

# ---- Panel C: low-c efficiency — CoT vs standard gap vs c (n=20) ----
cs = [0.01, 0.1, 0.5]
cot_broad = [0.181, 0.175, 0.076]
std_broad = [0.250, 0.150, 0.150]
ax[2].plot(cs, cot_broad, "o-", color="#d62728", label="CoT")
ax[2].plot(cs, std_broad, "s--", color="#7f7f7f", label="standard")
for x, y in zip(cs, cot_broad):
    ax[2].text(x, y - 0.018, f"{y:.2f}", ha="center", fontsize=8, color="#d62728")
ax[2].set_xscale("log"); ax[2].set_xticks(cs); ax[2].set_xticklabels(["0.01", "0.1", "0.5"])
ax[2].set_xlabel("corrected share c"); ax[2].set_ylabel("broad EM (Betley), n=20")
ax[2].set_title("C. Low-c: CoT suppresses broad more than\nstandard (graded in c; strongest at c=0.5)")
ax[2].legend(fontsize=9); ax[2].grid(alpha=.3)

fig.tight_layout()
out = RES / "final_plots.png"
fig.savefig(out, dpi=150)
print(f"saved {out}")
