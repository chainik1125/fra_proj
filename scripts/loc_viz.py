"""Three candidate displays for the hookpoint-localization grid.

Builds 3 separate PDFs (+PNGs) into paper/figures/:
  loc_viz1_heatmap.pdf  - 2-panel heatmap (Conv|DoM), color=JSDc, text=ASR, ASR<=1% boxed
  loc_viz2_scatter.pdf  - ASR vs JSDc scatter, clean-corner shaded, layer/hook encoded
  loc_viz3_trends.pdf   - small multiples: JSDc & ASR vs layer, one column per hook
"""
from __future__ import annotations
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

mpl.rcParams.update({
    "font.family":        "sans-serif",
    "font.sans-serif":    ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size":          11,
    "axes.titlesize":     12,
    "axes.labelsize":     12,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "axes.linewidth":     1.0,
    "axes.edgecolor":     "#222222",
    "xtick.labelsize":    10,
    "ytick.labelsize":    10,
    "legend.fontsize":    10,
    "figure.dpi":         120,
    "savefig.bbox":       "tight",
    "savefig.pad_inches": 0.12,
})

LAYERS = [0, 1, 2, 3]
HOOKS = ["ln1", "resid_mid", "resid_post", "hook_v"]
HOOK_LBL = {"ln1": "ln1", "resid_mid": "resid_mid", "resid_post": "resid_post", "hook_v": "hook_v"}
# (layer, hook) -> (conv (jsdc, asr, exact), dom (jsdc, asr, exact))
# DoM column = prompt-only projection (matching the main result), per-cell argmin
# JSDc s.t. ASR<=1% (else min-ASR); hook_v DoM is OV-DoM and already prompt-only.
D = {
    (0, "ln1"):        ((0.672, 0.0027, 0.110), (0.690, 0.0130, 0.110)),
    (0, "resid_mid"):  ((0.430, 0.0030, 0.252), (0.354, 0.0030, 0.305)),
    (0, "resid_post"): ((0.566, 0.1820, 0.186), (0.975, 0.3520, 0.009)),
    (0, "hook_v"):     ((0.441, 0.0000, 0.241), (0.935, 0.8240, 0.029)),
    (1, "ln1"):        ((0.903, 0.0043, 0.002), (0.709, 0.1090, 0.003)),
    (1, "resid_mid"):  ((0.546, 0.1083, 0.235), (0.442, 0.0030, 0.166)),
    (1, "resid_post"): ((0.731, 0.4023, 0.130), (0.803, 0.3640, 0.032)),
    (1, "hook_v"):     ((0.972, 0.8780, 0.010), (0.985, 0.9210, 0.003)),
    (2, "ln1"):        ((0.834, 0.3867, 0.056), (0.879, 0.6990, 0.036)),
    (2, "resid_mid"):  ((0.880, 0.6367, 0.040), (0.780, 0.2220, 0.017)),
    (2, "resid_post"): ((0.935, 0.7513, 0.022), (0.985, 0.7230, 0.003)),
    (2, "hook_v"):     ((0.702, 0.0910, 0.070), (0.953, 0.7870, 0.024)),
    (3, "ln1"):        ((0.931, 0.1207, 0.015), (0.990, 0.7700, 0.007)),
    (3, "resid_mid"):  ((0.763, 0.2743, 0.072), (0.761, 0.3000, 0.022)),
    (3, "resid_post"): ((0.900, 0.5800, 0.029), (0.920, 0.8140, 0.010)),
    (3, "hook_v"):     ((0.967, 0.8403, 0.009), (0.929, 0.8040, 0.032)),
}
OUT = "paper/figures"


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.pdf")
    fig.savefig(f"{OUT}/{name}.png", dpi=180)
    print(f"wrote {OUT}/{name}.pdf")


# ---------- 1. heatmap ----------
def heatmap():
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6))
    im = None
    for ax, mi, mname in zip(axes, [0, 1], ["Conv", "DoM"]):
        J = np.array([[D[(L, h)][mi][0] for L in LAYERS] for h in HOOKS])
        A = np.array([[D[(L, h)][mi][1] for L in LAYERS] for h in HOOKS])
        im = ax.imshow(J, cmap="RdYlGn_r", vmin=0.35, vmax=1.0, aspect="auto")
        ax.set_xticks(range(4)); ax.set_xticklabels(LAYERS)
        ax.set_yticks(range(4)); ax.set_yticklabels([HOOK_LBL[h] for h in HOOKS])
        ax.set_xlabel("layer"); ax.set_title(mname, fontweight="bold")
        for i in range(4):
            for j in range(4):
                jsd, asr = J[i, j], A[i, j]
                c = "white" if jsd > 0.72 else "black"
                ax.text(j, i, f"{jsd:.2f}\n{asr*100:.0f}%", ha="center", va="center",
                        fontsize=7.5, color=c, linespacing=1.1)
                if asr <= 0.01:
                    ax.add_patch(Rectangle((j-0.5, i-0.5), 1, 1, fill=False, ec="#1f4ed8", lw=2.4))
        for s in ax.spines.values():
            s.set_visible(False)
    cb = fig.colorbar(im, ax=axes, fraction=0.046, pad=0.03)
    cb.set_label("JSD$_c$ vs clean (lower = closer to clean)")
    fig.suptitle("Localization: cell color = JSD$_c$, text = ASR,  blue box = ASR $\\leq$ 1% (suppressed)",
                 fontsize=10, y=1.02)
    save(fig, "loc_viz1_heatmap")


# ---------- 2. scatter ----------
def scatter():
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.2), sharex=True, sharey=True)
    cmap = plt.cm.plasma
    lcol = {L: cmap(0.12 + 0.76 * L / 3) for L in LAYERS}
    mark = {"ln1": "o", "resid_mid": "s", "resid_post": "^", "hook_v": "D"}
    for ax, mi, mname in zip(axes, [1, 0], ["DoM", "Conv"]):
        for (L, h), v in D.items():
            jsd, asr, _ = v[mi]
            ax.scatter(jsd, asr, color=lcol[L], marker=mark[h], s=80,
                       edgecolors="white", linewidths=1.0, zorder=3)
        ax.set_xlim(0.30, 1.02); ax.set_ylim(-0.04, 1.02)
        ax.set_xlabel(r"JSD$_\mathrm{clean}$  (coherence measure)")
        ax.set_title(mname, fontweight="bold")
        ax.grid(True, color="#dddddd", lw=0.5)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("ASR  (suppression)")
    # the layer-0 hook_v point is the OV channel; label it vertically above (Conv panel, RHS)
    jsd, asr, _ = D[(0, "hook_v")][0]
    axes[1].annotate("OV", (jsd, asr), (jsd, asr + 0.17), fontsize=10, fontweight="bold",
                     ha="center", va="bottom", arrowprops=dict(arrowstyle="-", lw=0.6))
    lay_h = [plt.Line2D([], [], marker="o", ls="", color=lcol[L], mec="k", mew=0.4, label=f"layer {L}") for L in LAYERS]
    hk_h = [plt.Line2D([], [], marker=mark[h], ls="", color="0.5", mec="k", mew=0.4, label=HOOK_LBL[h]) for h in HOOKS]
    # both legends on the DoM panel (now LHS), stacked in the empty upper-left
    leg1 = axes[0].legend(handles=lay_h, loc="upper left", bbox_to_anchor=(0.01, 0.99), title="layer", framealpha=0.95)
    axes[0].add_artist(leg1)
    axes[0].legend(handles=hk_h, loc="upper left", bbox_to_anchor=(0.01, 0.58), title="hook", framealpha=0.95)
    fig.suptitle("Suppression vs clean cost: only the resid_mid output (layers 0--1) and the layer-0 OV channel reach the clean corner",
                 fontsize=10, y=1.0)
    save(fig, "loc_viz2_scatter")


# ---------- 3. layer-trend small multiples ----------
def trends():
    fig, axes = plt.subplots(2, 4, figsize=(11, 5), sharex=True)
    for j, h in enumerate(HOOKS):
        aj, aa = axes[0, j], axes[1, j]
        cj = [D[(L, h)][0][0] for L in LAYERS]; dj = [D[(L, h)][1][0] for L in LAYERS]
        ca = [D[(L, h)][0][1]*100 for L in LAYERS]; da = [D[(L, h)][1][1]*100 for L in LAYERS]
        aj.plot(LAYERS, cj, "-o", color="#1f77b4", label="Conv", ms=5)
        aj.plot(LAYERS, dj, "--s", color="#d62728", label="DoM", ms=5)
        aa.plot(LAYERS, ca, "-o", color="#1f77b4", ms=5)
        aa.plot(LAYERS, da, "--s", color="#d62728", ms=5)
        aj.set_title(HOOK_LBL[h], fontweight="bold", fontsize=10)
        aj.set_ylim(0.33, 1.03); aa.set_ylim(-4, 100)
        aa.axhspan(0, 1, color="#bfe6c0", alpha=0.7, zorder=0)
        aa.set_xticks(LAYERS); aa.set_xlabel("layer")
        aj.grid(alpha=0.25); aa.grid(alpha=0.25)
        if j == 0:
            aj.set_ylabel("JSD$_c$ $\\downarrow$"); aa.set_ylabel("ASR (%) $\\downarrow$")
            aj.legend(fontsize=8, loc="lower right")
            aa.text(0.05, 4, "ASR $\\leq$ 1%", fontsize=7, color="#1b7a32")
    fig.suptitle("Localization with depth: JSD$_c$ (top) and ASR (bottom) vs layer, per hook\n"
                 "only resid_mid / hook_v at layer 0 reach low ASR at low JSD$_c$", fontsize=10)
    fig.tight_layout()
    save(fig, "loc_viz3_trends")


if __name__ == "__main__":
    heatmap(); scatter(); trends()
