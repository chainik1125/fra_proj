"""Fig. 3(a): word-for-word clean recovery vs sleeper suppression.

Uses the layer-0 retrained-SAE wide-coefficient screens that also feed Fig. 3(b-d)
(data/tinystories/wide_screen_L0/retrain_L0_*.json). Each method's screen sweep of
its BO-selected feature (suppression half, raw alpha in [0, 10]) gives faint
(suppression, word-match) points; the bold line is the method's Pareto front.

    x = sleeper suppression = 1 - ASR / unsteered ASR
    y = % of deployment prompts whose steered rollout == clean rollout word-for-word

--seed N plots one SAE seed; --seed mean (default) averages the six seeds' sweeps
point-wise in alpha before taking the front.

Inputs : data/tinystories/wide_screen_L0/retrain_L0_{ov,conventional,conv_ln1}_s{0..5}.json
Outputs: figures/fig4_sleeper_poster_pareto.{pdf,png}

    uv run scripts/plot_tinysleeper_pareto.py [--seed 4|mean] [--out PATH]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SCHEMES = [
    ("ov",           "FRA",       "#0072B2", "o"),
    ("conventional", "resid-mid", "#D55E00", "^"),
    ("conv_ln1",     "ln1 SAE",   "#009E73", "s"),
]
from _paths import DATA, FIGURES

RESULTS = DATA / "tinystories" / "wide_screen_L0"
OUT = FIGURES / "fig4_sleeper_poster_pareto.pdf"


def setup_style() -> None:
    mpl.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Inter", "Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size":         13,
        "axes.titlesize":    14,
        "axes.labelsize":    14,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.linewidth":    1.1,
        "axes.edgecolor":    "#222222",
        "xtick.labelsize":   12,
        "ytick.labelsize":   12,
        "legend.fontsize":   12,
        "legend.frameon":    True,
        "figure.dpi":        120,
        "savefig.bbox":      "tight",
        "savefig.pad_inches": 0.12,
    })


def load_cell(src: str, scheme: str, seed: int):
    """(suppression, exact_pct) arrays, alpha-ordered, over the suppression half."""
    d = json.loads((RESULTS / f"{src}_L0_{scheme}_s{seed}.json").read_text())
    feat = d["bo"]["feat"]
    curve = sorted((r for r in d["rows"] if r["feat"] == feat and 0.0 <= r["alpha"] <= 10.0),
                   key=lambda r: r["alpha"])
    supp = np.array([1.0 - r["asr"] / d["baseline_asr"] for r in curve])
    exact = np.array([r["exact"] / d["n_prompts"] * 100.0 for r in curve])
    return supp, exact


def load(src: str, scheme: str, seed: str):
    if seed != "mean":
        return load_cell(src, scheme, int(seed))
    cells = [load_cell(src, scheme, s) for s in range(6)]
    return (np.mean([c[0] for c in cells], axis=0), np.mean([c[1] for c in cells], axis=0))


def frontier(supp: np.ndarray, exact: np.ndarray):
    """Pareto staircase over the sweep: as suppression rises, record recovery.

    Sort points by (suppression, recovery) and keep the running-max of recovery,
    so the curve is the outer boundary of achieved (suppression, recovery) pairs;
    at full suppression it ends on the best recovery actually achieved there.
    """
    order = np.lexsort((exact, supp))
    xs, ys, best = [], [], -np.inf
    for i in order:
        if exact[i] > best:
            xs.append(supp[i])
            ys.append(exact[i])
            best = exact[i]
    return np.array(xs), np.array(ys)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src", default="retrain")
    p.add_argument("--seed", default="mean")
    p.add_argument("--out", type=Path, default=OUT)
    args = p.parse_args()
    setup_style()

    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    for sch, label, color, mk in SCHEMES:
        supp, exact = load(args.src, sch, args.seed)
        ax.scatter(supp * 100, exact, color=color, marker=mk, s=24, alpha=0.25,
                   linewidths=0, zorder=2)
        fx, fy = frontier(supp * 100, exact)
        ax.plot(fx, fy, color=color, lw=2.8, marker=mk, ms=6.5, markevery=3,
                markeredgecolor="white", markeredgewidth=0.8,
                label=label, zorder=3)

    # "better" cue toward the top-right corner
    ax.annotate("better", xy=(0.985, 0.97), xycoords="axes fraction",
                xytext=(0.80, 0.80), textcoords="axes fraction",
                fontsize=13, color="#555555", ha="center",
                arrowprops=dict(arrowstyle="->", color="#555555", lw=1.6))

    ax.set_xlabel("sleeper suppression   (% of triggered prompts disarmed)")
    ax.set_ylabel("word-for-word clean recovery (%)")
    ax.set_xlim(-2, 103)
    ax.set_ylim(-1.5, 45)
    ax.legend(loc="upper left", handlelength=2.2, borderpad=0.7,
              labelspacing=0.55).get_frame().set_alpha(0.95)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        out = args.out.with_suffix("." + ext)
        fig.savefig(out)
        print("wrote", out)
    plt.close(fig)


if __name__ == "__main__":
    main()
