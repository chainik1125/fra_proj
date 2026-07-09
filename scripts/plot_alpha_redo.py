"""Paper-style figures for the wide-alpha redo (results/alpha_redo/).

Fig 1 (seedband): per scheme, mean J_clean across the 6 SAE seeds' winner-feature
curves (screen: 64 prompts, decode seed 0, 81-point alpha grid), band = min-max
across seeds. BO-refined optima (200 prompts x 5 decode seeds) overlaid as stars.

Fig 2 (representative seed): the three winner curves for one seed, BO stars.

Display convention matches the paper: displayed alpha = -raw alpha (negative =
feature subtracted / sleeper suppressed). Raw JSONs keep the raw sign.

Usage: python -m scripts.plot_alpha_redo [--seed 1] [--dir results/alpha_redo]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# Okabe-Ito (CVD-safe), fixed order; redundant linestyle+marker per series.
SCHEMES = [
    ("ov",           "FRA-OV @ ln1",        "#0072B2", "-",  "o"),
    ("conventional", "Conv @ resid_mid",    "#D55E00", "--", "^"),
    ("conv_ln1",     "Conv @ ln1",          "#009E73", "-.", "s"),
]
BAND_ALPHA = 0.16
N_CHUNKS = 5


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         11,
        "axes.titlesize":    12,
        "axes.labelsize":    11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.0,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   9,
        "ytick.labelsize":   9,
        "legend.fontsize":   8,
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def load_cell(res: Path, scheme: str, seed: int):
    """(winner_feat, alphas_raw, jclean_curve, bo_or_None) for one cell."""
    rows = []
    for c in range(N_CHUNKS):
        f = res / f"screen_{scheme}_s{seed}_c{c}.json"
        if f.exists():
            rows.extend(json.loads(f.read_text())["rows"])
    bo_file = res / f"bo_{scheme}_s{seed}.json"
    bo = json.loads(bo_file.read_text()) if bo_file.exists() else None
    if bo is not None:
        feat = bo["feat"]
    else:  # BO pending: use the ASR-gated grid winner
        gated = [r for r in rows if r["asr"] <= 0.01]
        feat = min(gated if gated else rows,
                   key=lambda r: (r["jsd_clean"], r["asr"]))["feat"]
    curve = sorted((r for r in rows if r["feat"] == feat), key=lambda r: r["alpha"])
    alphas = np.array([r["alpha"] for r in curve])
    jclean = np.array([r["jsd_clean"] for r in curve])
    return feat, alphas, jclean, bo


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, default=Path("results/alpha_redo"))
    p.add_argument("--seed", type=int, default=1, help="representative seed")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()
    res = args.dir / "results"
    outdir = args.out or (args.dir / "figures")
    outdir.mkdir(parents=True, exist_ok=True)
    setup_style()

    data = {sch: {} for sch, *_ in SCHEMES}
    for sch, *_ in data:
        pass
    for sch, _, _, _, _ in SCHEMES:
        for seed in range(6):
            data[sch][seed] = load_cell(res, sch, seed)

    # ── Fig 1: seed-band ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sch, label, color, ls, mk in SCHEMES:
        raw = data[sch][0][1]
        disp = -raw
        curves = np.stack([data[sch][s][2] for s in range(6)])
        order = np.argsort(disp)
        ax.fill_between(disp[order], curves.min(0)[order], curves.max(0)[order],
                        color=color, alpha=BAND_ALPHA, lw=0, zorder=1)
        ax.plot(disp[order], curves.mean(0)[order], color=color, ls=ls,
                marker=mk, ms=3, markevery=8, lw=1.8, label=label, zorder=3)
        bo_x = [-d[3]["opt_alpha"] for d in data[sch].values() if d[3]]
        bo_y = [d[3]["opt_jsd_clean"] for d in data[sch].values() if d[3]]
        ax.scatter(bo_x, bo_y, marker="*", s=90, color=color,
                   edgecolors="white", linewidths=0.6, zorder=4,
                   label=f"{label} — BO optimum / seed")
    ax.set_xlabel(r"steering strength $\alpha$   (negative = feature subtracted)")
    ax.set_ylabel(r"JSD$_\mathrm{clean}$ (matched decode seed)  $\downarrow$")
    ax.set_xlim(10.4, -10.4)  # paper orientation: suppression to the right
    ax.set_ylim(0, 1.05)
    ax.axhline(0.99, color="#888888", lw=0.8, ls=":", zorder=0)
    ax.text(10.1, 1.005, "unsteered (0.99)", fontsize=7.5, color="#666666")
    ax.legend(loc="lower left", ncols=1)
    ax.set_title("Sleeper suppression vs steering strength — winner feature per "
                 "seed, mean ± seed range (6 SAE seeds)")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_redo_seedband.{ext}")
    plt.close(fig)

    # ── Fig 2: representative seed ───────────────────────────────────
    s = args.seed
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    for sch, label, color, ls, mk in SCHEMES:
        feat, raw, jclean, bo = data[sch][s]
        disp = -raw
        order = np.argsort(disp)
        ax.plot(disp[order], jclean[order], color=color, ls=ls, marker=mk,
                ms=3, markevery=8, lw=1.8, label=f"{label}  (f{feat})", zorder=3)
        if bo:
            offsets = {"ov": (10, -16), "conventional": (10, 8),
                       "conv_ln1": (-52, 10)}
            ax.scatter([-bo["opt_alpha"]], [bo["opt_jsd_clean"]], marker="*",
                       s=130, color=color, edgecolors="white", linewidths=0.7,
                       zorder=4)
            ax.annotate(f"$\\alpha^*$ = {-bo['opt_alpha']:+.2f}",
                        (-bo["opt_alpha"], bo["opt_jsd_clean"]),
                        textcoords="offset points", xytext=offsets[sch],
                        fontsize=8, color=color, zorder=5)
    ax.set_xlabel(r"steering strength $\alpha$   (negative = feature subtracted)")
    ax.set_ylabel(r"JSD$_\mathrm{clean}$ (matched decode seed)  $\downarrow$")
    ax.set_xlim(10.4, -10.4)
    ax.set_ylim(0, 1.05)
    ax.axhline(0.99, color="#888888", lw=0.8, ls=":", zorder=0)
    ax.legend(loc="lower left")
    ax.set_title(f"Sleeper suppression vs steering strength — SAE seed {s}\n"
                 "(curves: 81-point screen; stars: BO optimum at 200 prompts "
                 "× 5 decode seeds)")
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_redo_seed{s}.{ext}")
    plt.close(fig)

    # ── Fig 3: all seeds, 2x3 panel ─────────────────────────────────
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.6), sharex=True, sharey=True)
    for s, ax in zip(range(6), axes.flat):
        for row, (sch, label, color, ls, mk) in enumerate(SCHEMES):
            feat, raw, jclean, bo = data[sch][s]
            disp = -raw
            order = np.argsort(disp)
            ax.plot(disp[order], jclean[order], color=color, ls=ls, marker=mk,
                    ms=2.5, markevery=10, lw=1.5,
                    label=label if s == 0 else None, zorder=3)
            if bo:
                ax.scatter([-bo["opt_alpha"]], [bo["opt_jsd_clean"]], marker="*",
                           s=90, color=color, edgecolors="white",
                           linewidths=0.6, zorder=4)
                note = f"f{feat}, $\\alpha^*$={-bo['opt_alpha']:+.2f}"
            else:
                note = f"f{feat} (BO pending)"
            ax.text(0.03, 0.30 - 0.09 * row, note, transform=ax.transAxes,
                    fontsize=7.5, color=color)
        ax.axhline(0.99, color="#888888", lw=0.7, ls=":", zorder=0)
        ax.set_title(f"SAE seed {s}", fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(10.4, -10.4)
    for ax in axes[1]:
        ax.set_xlabel(r"steering strength $\alpha$")
    for ax in axes[:, 0]:
        ax.set_ylabel(r"JSD$_\mathrm{clean}$ $\downarrow$")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=3,
               bbox_to_anchor=(0.5, 1.0), frameon=True)
    fig.suptitle("Sleeper suppression vs steering strength — all SAE seeds\n"
                 "(negative $\\alpha$ = feature subtracted; curves: 81-point "
                 "screen; stars: BO optimum at 200 prompts × 5 decode seeds)",
                 y=1.10, fontsize=12)
    for ext in ("pdf", "png"):
        fig.savefig(outdir / f"fig_redo_allseeds.{ext}")
    plt.close(fig)
    print(f"wrote {outdir}/fig_redo_seedband.*, fig_redo_seed{args.seed}.*, "
          f"and fig_redo_allseeds.*")


if __name__ == "__main__":
    main()
