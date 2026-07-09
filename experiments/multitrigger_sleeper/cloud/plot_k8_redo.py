"""Per-seed figure for the K8-randpos wide-coefficient redo (results/k8_redo/).

Layout mirrors the TinyStories redo figure (fig_redo_allseeds): one panel per
SAE seed, three schemes, winner-feature screen curves (64 randpos pairs),
stars = HELD-OUT re-measure of the BO optimum (192 fresh pairs + positions),
annotated with held-out ASR (gate 0.05; most cells do NOT disarm — the point).

Display: x = -c (negative = feature subtracted; suppression to the right),
y = J_clean in nats (ln2 = 0.693 max; no-intervention ~0.65).
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

RES = Path("results/k8_redo/results")
OUT = Path("figures")
SEEDS = [1, 2, 7]
SCHEMES = [
    ("ov",           "FRA-OV @ ln1",     "#0072B2", "-",  "o"),
    ("conventional", "Conv @ resid_mid", "#D55E00", "--", "^"),
    ("conv_ln1",     "Conv @ ln1",       "#009E73", "-.", "s"),
]


def setup_style():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 1.0, "axes.edgecolor": "#222222",
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8, "legend.frameon": True,
        "figure.dpi": 120, "savefig.bbox": "tight", "savefig.pad_inches": 0.12,
    })


def cell(scheme, seed):
    rows = []
    noint = None
    for c in range(5):
        f = RES / f"screen_{scheme}_s{seed}_c{c}.json"
        if f.exists():
            d = json.loads(f.read_text())
            rows.extend(d["rows"])
            noint = d["no_intervention"]
    bo = json.loads((RES / f"bo_{scheme}_s{seed}.json").read_text())
    feat = bo["feat"]
    curve = sorted((r for r in rows if r["feat"] == feat), key=lambda r: r["c"])
    cs = np.array([r["c"] for r in curve])
    js = np.array([r["Jclean_nats"] for r in curve])
    return feat, cs, js, bo, noint


def main():
    setup_style()
    OUT.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.9), sharex=True, sharey=True)
    for s, ax in zip(SEEDS, axes):
        noint_ref = None
        for row, (sch, label, color, ls, mk) in enumerate(SCHEMES):
            feat, cs, js, bo, noint = cell(sch, s)
            noint_ref = noint_ref or noint
            disp = -cs
            order = np.argsort(disp)
            ax.plot(disp[order], js[order], color=color, ls=ls, marker=mk,
                    ms=2.5, markevery=10, lw=1.5,
                    label=label if s == SEEDS[0] else None, zorder=3)
            h = bo["heldout"]
            disarms = h["ASR"] <= 0.05
            ax.scatter([-bo["opt_c"]], [h["Jclean_nats"]], marker="*",
                       s=110, color=color if disarms else "white",
                       edgecolors=color, linewidths=1.2, zorder=4)
            ax.text(0.03, 0.97 - 0.09 * row,
                    f"f{feat}: held ASR {h['ASR']:.2f}"
                    + (" (disarms)" if disarms else " (FAILS)"),
                    transform=ax.transAxes, fontsize=7.5, color=color, va="top")
        if noint_ref:
            ax.axhline(noint_ref["Jclean_nats"], color="#888888", lw=0.8,
                       ls=":", zorder=0)
        ax.set_title(f"SAE seed {s}", fontsize=10)
        ax.set_xlim(20.5, -20.5)
        ax.set_ylim(0, 0.72)
        ax.set_xlabel(r"steering coefficient $c$ (neg = subtracted)")
    axes[0].set_ylabel(r"J$_\mathrm{clean}$ (nats) $\downarrow$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(
        "randpos-K8 sleeper: single-feature steering vs coefficient — winner "
        "feature per scheme\n(curves: 81-pt screen, 64 pairs; stars: held-out "
        "re-measure of BO optimum, 192 fresh pairs; filled star = disarms "
        "(ASR≤0.05), hollow = does not; dotted line = no intervention)",
        y=1.22, fontsize=11)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"fig_k8randpos_redo.{ext}")
    plt.close(fig)
    print(f"wrote {OUT}/fig_k8randpos_redo.*")


if __name__ == "__main__":
    main()
