"""Poster figure: Gemma-2-2b in-context backdoor — held-out collateral KL at fixed
suppression levels (50% / 95% / 99%) for FRA-QK vs DoM vs conv-SAE.

Data: fra_win g4 raw curves (g4.json, pulled from HF fra_win/out/g4/). Interpolation
at each level follows the job's own at() convention.
"""
import json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

ACCENT = "#2a78d6"; GRAY = "#898781"; GRAY2 = "#c3c2b7"
INK = "#0b0b0b"; INK2 = "#52514e"; GRID = "#e1e0d9"; AXIS = "#c3c2b7"; SURF = "#ffffff"

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
    "text.color": INK, "axes.edgecolor": AXIS, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2,
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
})

import pathlib
rows = json.load(open(pathlib.Path(__file__).resolve().parent.parent / "out/g4/g4.json"))["rows"]

def at_interp(curve, t):
    xs = np.array([a for a, b in curve]); ys = np.array([b for a, b in curve])
    if xs.max() < t:
        return None
    o = np.argsort(xs, kind="stable")
    return float(np.interp(t, xs[o], ys[o]))

def at_full(curve, t=0.99):
    ys = [b for a, b in curve if a >= t]
    return min(ys) if ys else None

def level(method, fn):
    vals = [fn(r[method]) for r in rows]
    vals = [v for v in vals if v is not None]
    return (float(np.mean(vals)), len(vals)) if vals else (None, 0)

LEVELS = [("50%", lambda c: at_interp(c, 0.50)),
          ("95%", lambda c: at_interp(c, 0.95)),
          ("99%", lambda c: at_interp(c, 0.985))]  # 99% level: t=0.985, job rounds 0.987 -> 0.99 (doctor case)
METHODS = [("fra", "FRA-QK (ours)", ACCENT),
           ("dom", "DoM / mean-diff", GRAY),
           ("conv", "conv-SAE steer", GRAY2)]

data = {m: [level(m, fn) for _, fn in LEVELS] for m, _, _ in METHODS}
for m, lab, _ in METHODS:
    print(lab, [(round(v, 2) if v else None, n) for v, n in data[m]])

fig, ax = plt.subplots(figsize=(7.8, 3.9))
fig.subplots_adjust(left=0.11, right=0.97, top=0.80, bottom=0.13)
YMIN = 0.1
W = 0.24
xg = np.arange(len(LEVELS))

for j, (m, lab, col) in enumerate(METHODS):
    xs = xg + (j - 1) * (W + 0.03)
    for i, (v, n) in enumerate(data[m]):
        if v is None:
            ax.text(xs[i], YMIN * 1.5, "not\nreached", ha="center", va="bottom",
                    fontsize=9.5, color=INK2, style="italic")
            continue
        ax.bar(xs[i], v, W, bottom=None, color=col, edgecolor=SURF, linewidth=1.5,
               zorder=3, label=lab if i == 0 else None)
        note = f"{v:.2f}" if v < 10 else f"{v:.0f}"
        ax.text(xs[i], v * 1.15, note, ha="center", va="bottom", fontsize=10.5,
                color=INK if m == "fra" else INK2,
                fontweight="bold" if m == "fra" else "normal")

ax.set_yscale("log")
ax.set_ylim(YMIN, 300)
ax.set_yticks([0.1, 1, 10, 100], ["0.1", "1", "10", "100"])
ax.grid(axis="y", color=GRID, lw=1, zorder=0)
for s in ("top", "right", "left"):
    ax.spines[s].set_visible(False)
ax.spines["bottom"].set_color(AXIS)
ax.tick_params(axis="both", which="both", length=0, labelsize=11)
ax.minorticks_off()
ax.set_xticks(xg, [f"{name} suppression" for name, _ in LEVELS], fontsize=12)
ax.set_ylabel("held-out collateral KL (nats) — lower is better", fontsize=11)

ax.legend(loc="upper left", bbox_to_anchor=(0, 1.02), fontsize=10.5, frameon=False,
          ncol=3, handlelength=1.2, columnspacing=1.2)
ax.set_title("Gemma-2-2b in-context backdoor: collateral at matched removal",
             fontsize=12.5, fontweight="bold", loc="left", pad=30, color=INK)
ax.text(0, 1.13, "mean over 4 planted backdoors, GemmaScope 65k",
        transform=ax.transAxes, fontsize=9, color=INK2, va="top")

out = "/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/icml_mi_poster/figures/"
fig.savefig(out + "fig_gemma_kl_levels.pdf")
fig.savefig(out + "fig_gemma_kl_levels.png", dpi=300)
print("saved")
