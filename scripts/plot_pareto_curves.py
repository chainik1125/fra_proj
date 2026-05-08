"""Pareto curves: Sleepers Removed (%) vs Noise-to-Recovery Ratio.

X = 1 / severity_ratio  ("Noise-to-Recovery Ratio", higher = less collateral damage)
Y = (1 - ASR) * 100     ("Sleepers Removed (%)",    higher = more sleepers removed)
Up-and-right is the good direction.

For upstream curves (single feature / feature set), averages across the 5 SAE seeds
per alpha and draws min/max error bars on Y.  Downstream has no seed variation.

Produces 4 figure types × 2 pipelines = 8 PNG files:
  fig1_{pipeline}.png  — single panel: single feature + feature set + downstream feature
  fig2_{pipeline}.png  — two panels: (feature set + downstream) | (single + downstream)
  fig3_{pipeline}.png  — standalone: feature set + downstream feature
  fig4_{pipeline}.png  — standalone: single feature + downstream feature

Pipeline is indicated only in the file name, never in the plot.

Usage (defaults read from results/):
    python -m scripts.plot_pareto_curves
    python -m scripts.plot_pareto_curves --jamie_in results/jamie_experiment.json \\
        --ketan_in results/ketan_experiment.json --out_dir figures
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

# ── visual constants ──────────────────────────────────────────────────────────
BG = "#fbfaf6"
GRID = "#d9d4c8"
STYLES: dict[str, dict] = {
    "single":     dict(color="#2166ac", marker="o", label="Single Feature",     ls="-",  lw=1.8),
    "set":        dict(color="#4dac26", marker="s", label="Feature Set",        ls="-",  lw=1.8),
    "downstream": dict(color="#d6604d", marker="^", label="Downstream Feature", ls="--", lw=1.5),
}
XLABEL = "Noise-to-Recovery Ratio  (↑ less collateral damage)"
YLABEL = "Sleepers Removed (%)"


# ── data helpers ──────────────────────────────────────────────────────────────

def _aggregate(points: list[dict], family: str, eval_mode: str) -> list[dict]:
    """Group points by alpha; return mean/min/max over seeds per alpha."""
    subset = [
        p for p in points
        if p["family"] == family
        and (family == "downstream" or p.get("eval_mode") == eval_mode)
    ]
    by_alpha: dict[float, list] = {}
    for p in subset:
        by_alpha.setdefault(p["alpha"], []).append(p)

    rows = []
    for alpha in sorted(by_alpha):
        grp = by_alpha[alpha]
        xs = [1.0 / p["severity_ratio"] for p in grp]
        ys = [(1.0 - p["asr"]) * 100.0 for p in grp]
        rows.append(dict(
            alpha=alpha,
            x=float(np.mean(xs)),
            y=float(np.mean(ys)),
            y_lo=float(np.min(ys)),
            y_hi=float(np.max(ys)),
            has_var=len(xs) > 1,
        ))
    return rows


# ── drawing helpers ───────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_ylabel(YLABEL, fontsize=9)


def _draw_curve(ax: plt.Axes, rows: list[dict], style: dict) -> None:
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    ax.plot(xs, ys, color=style["color"], ls=style["ls"], lw=style["lw"], zorder=3)
    ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
               s=52, zorder=4, label=style["label"],
               edgecolor="white", linewidth=0.5)
    if rows[0]["has_var"]:
        y_lo_err = [r["y"] - r["y_lo"] for r in rows]
        y_hi_err = [r["y_hi"] - r["y"] for r in rows]
        ax.errorbar(xs, ys, yerr=[y_lo_err, y_hi_err],
                    fmt="none", color=style["color"],
                    capsize=3, linewidth=0.9, alpha=0.55, zorder=2)


def _fill_panel(ax: plt.Axes, curves: list[tuple[list[dict], str]]) -> None:
    for rows, key in curves:
        _draw_curve(ax, rows, STYLES[key])
    _style_ax(ax)
    ax.legend(frameon=False, fontsize=8.5, loc="lower right")


# ── figure factories ──────────────────────────────────────────────────────────

def _fig1(single, fset, down) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    _fill_panel(ax, [(single, "single"), (fset, "set"), (down, "downstream")])
    return fig


def _fig2(single, fset, down) -> plt.Figure:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(12, 5),
                                      constrained_layout=True)
    _fill_panel(ax_l, [(fset, "set"),    (down, "downstream")])
    _fill_panel(ax_r, [(single, "single"), (down, "downstream")])
    ax_r.set_ylabel("")
    return fig


def _fig3(fset, down) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    _fill_panel(ax, [(fset, "set"), (down, "downstream")])
    return fig


def _fig4(single, down) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6, 5))
    _fill_panel(ax, [(single, "single"), (down, "downstream")])
    return fig


# ── save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def make_figures(payload: dict, out_dir: Path, pipeline: str) -> None:
    pts = payload["points"]
    single = _aggregate(pts, "upstream",   "single")
    fset   = _aggregate(pts, "upstream",   "set")
    down   = _aggregate(pts, "downstream", "single")

    _save(_fig1(single, fset, down),  out_dir / f"fig1_{pipeline}.png")
    _save(_fig2(single, fset, down),  out_dir / f"fig2_{pipeline}.png")
    _save(_fig3(fset, down),          out_dir / f"fig3_{pipeline}.png")
    _save(_fig4(single, down),        out_dir / f"fig4_{pipeline}.png")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--jamie_in", type=Path,
                   default=Path("results/jamie_experiment.json"))
    p.add_argument("--ketan_in", type=Path,
                   default=Path("results/ketan_experiment.json"))
    p.add_argument("--out_dir",  type=Path, default=Path("figures"))
    args = p.parse_args()

    payloads = {
        "jamie": json.loads(args.jamie_in.read_text()),
        "ketan": json.loads(args.ketan_in.read_text()),
    }
    for pipeline, payload in payloads.items():
        print(f"[{pipeline}]")
        make_figures(payload, args.out_dir, pipeline)


if __name__ == "__main__":
    main()
