"""Pareto curves: Sleepers Removed (%) vs Noise-to-Recovery Ratio.

X = 1 / severity_ratio  ("Noise-to-Recovery Ratio", higher = less collateral damage)
Y = (1 - ASR) * 100     ("Sleepers Removed (%)",    higher = more sleepers removed)
Up-and-right is the good direction.

Points are colour-coded blue→red by steering strength α; marker shape identifies
the method.  For upstream curves (single feature / feature set) points are averaged
across the 5 SAE seeds per α, with min/max error bars.  Downstream has no seeds.

Produces 4 figure types × 2 pipelines = 8 PDF files:
  fig1_{pipeline}.pdf  — single panel: single feature + feature set + downstream feature
  fig2_{pipeline}.pdf  — two panels: (feature set + downstream) | (single + downstream)
  fig3_{pipeline}.pdf  — standalone: feature set + downstream feature
  fig4_{pipeline}.pdf  — standalone: single feature + downstream feature

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
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

# ── visual constants ──────────────────────────────────────────────────────────
BG     = "#fbfaf6"
GRID   = "#d9d4c8"
CMAP   = "coolwarm"
STYLES: dict[str, dict] = {
    "single":     dict(color="#762a83", marker="o", label="Single Feature",     ls="-",  lw=0.9),
    "set":        dict(color="#1b7837", marker="s", label="Feature Set",        ls="-",  lw=0.9),
    "downstream": dict(color="#333333", marker="^", label="Downstream Feature", ls="--", lw=0.9),
}
XLABEL = "Noise-to-Recovery Ratio  (↑ less collateral damage)"
YLABEL = "Sleepers Removed (%)"


# ── colour scale ──────────────────────────────────────────────────────────────

def _alpha_colormap(alphas: list[float]):
    """Return (alpha→rgba dict, Normalize, Colormap) for a shared colour scale."""
    norm = mcolors.Normalize(vmin=0, vmax=max(len(alphas) - 1, 1))
    cmap = matplotlib.colormaps[CMAP]
    return {a: cmap(norm(i)) for i, a in enumerate(alphas)}, norm, cmap


# ── data helpers ──────────────────────────────────────────────────────────────

def _aggregate(points: list[dict], family: str, eval_mode: str) -> list[dict]:
    """Group by alpha; return per-alpha mean/min/max for both X and Y."""
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
            x=float(np.mean(xs)), x_lo=float(np.min(xs)), x_hi=float(np.max(xs)),
            y=float(np.mean(ys)), y_lo=float(np.min(ys)), y_hi=float(np.max(ys)),
            has_var=len(xs) > 1,
        ))
    return rows


def _aggregate_by_seed(points: list[dict], family: str, eval_mode: str
                        ) -> list[list[dict]]:
    """Return one curve (list of alpha-sorted rows) per seed."""
    subset = [
        p for p in points
        if p["family"] == family
        and (family == "downstream" or p.get("eval_mode") == eval_mode)
    ]
    by_seed: dict = {}
    for p in subset:
        key = p.get("sae_seed")   # None for downstream
        by_seed.setdefault(key, []).append(p)
    curves = []
    for seed_pts in by_seed.values():
        seed_pts.sort(key=lambda p: p["alpha"])
        curves.append([
            dict(alpha=p["alpha"],
                 x=1.0 / p["severity_ratio"],
                 y=(1.0 - p["asr"]) * 100.0)
            for p in seed_pts
        ])
    return curves


# ── drawing helpers ───────────────────────────────────────────────────────────

def _style_ax(ax: plt.Axes) -> None:
    ax.set_facecolor(BG)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, color=GRID, linewidth=0.7, alpha=0.7)
    ax.set_axisbelow(True)
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_ylabel(YLABEL, fontsize=9)
    ax.axvline(1.0, color="#555555", linewidth=0.9, linestyle=":", zorder=1)
    ax.text(1.01, 1, "indistinguishable\nfrom baseline",
            fontsize=6.5, color="#555555", va="bottom", ha="left",
            transform=ax.get_xaxis_transform())


def _draw_curve(ax: plt.Axes, rows: list[dict],
                style: dict, alpha_colors: dict) -> None:
    xs = [r["x"] for r in rows]
    ys = [r["y"] for r in rows]
    ax.plot(xs, ys, color=style["color"], ls=style["ls"], lw=style["lw"],
            alpha=0.5, zorder=2)
    for r in rows:
        c = alpha_colors[r["alpha"]]
        ax.scatter(r["x"], r["y"], color=c, marker=style["marker"],
                   s=60, zorder=4, edgecolor="#1b1b1b", linewidth=0.45)
        if r["has_var"]:
            ax.errorbar(
                r["x"], r["y"],
                xerr=[[r["x"] - r["x_lo"]], [r["x_hi"] - r["x"]]],
                yerr=[[r["y"] - r["y_lo"]], [r["y_hi"] - r["y"]]],
                fmt="none", color=c, capsize=3, linewidth=0.9, alpha=0.6, zorder=3,
            )


def _draw_all_seeds(ax: plt.Axes, seed_curves: list[list[dict]],
                    style: dict) -> None:
    for rows in seed_curves:
        xs = [r["x"] for r in rows]
        ys = [r["y"] for r in rows]
        ax.plot(xs, ys, color=style["color"], ls=style["ls"], lw=0.8,
                alpha=0.45, zorder=2)
        ax.scatter(xs, ys, color=style["color"], marker=style["marker"],
                   s=30, zorder=3, edgecolor="white", linewidth=0.3,
                   alpha=0.7)


def _fill_panel_seeds(ax: plt.Axes,
                      curves: list[tuple[list[list[dict]], str]]) -> None:
    for seed_curves, key in curves:
        _draw_all_seeds(ax, seed_curves, STYLES[key])
    _style_ax(ax)
    handles = [
        mlines.Line2D([], [], marker=STYLES[k]["marker"], ls=STYLES[k]["ls"],
                      color=STYLES[k]["color"], markerfacecolor=STYLES[k]["color"],
                      markeredgecolor="white", markeredgewidth=0.3,
                      markersize=7, label=STYLES[k]["label"])
        for _, k in curves
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")


def _fill_panel(ax: plt.Axes,
                curves: list[tuple[list[dict], str]],
                alpha_colors: dict) -> None:
    for rows, key in curves:
        _draw_curve(ax, rows, STYLES[key], alpha_colors)
    _style_ax(ax)
    handles = [
        mlines.Line2D([], [], marker=STYLES[k]["marker"], ls=STYLES[k]["ls"],
                      color=STYLES[k]["color"], markerfacecolor="#888888",
                      markeredgecolor="#1b1b1b", markeredgewidth=0.5,
                      markersize=7, label=STYLES[k]["label"])
        for *_, k in curves
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="lower right")


def _add_colorbar(fig: plt.Figure, axes, norm, cmap, alphas: list[float]) -> None:
    sm = cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.03, pad=0.02,
                        ticks=range(len(alphas)))
    cbar.ax.set_yticklabels([f"{a:g}" for a in alphas], fontsize=8)
    cbar.set_label("Steering strength α", fontsize=9)


# ── figure factories ──────────────────────────────────────────────────────────

def _fig1(single, fset, down, alpha_colors, norm, cmap, alphas) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _fill_panel(ax, [(single, "single"), (fset, "set"), (down, "downstream")],
                alpha_colors)
    _add_colorbar(fig, ax, norm, cmap, alphas)
    return fig


def _fig2(single, fset, down, alpha_colors, norm, cmap, alphas) -> plt.Figure:
    fig, (ax_l, ax_r) = plt.subplots(1, 2, figsize=(13, 5),
                                      constrained_layout=True)
    _fill_panel(ax_l, [(fset,   "set"),    (down, "downstream")], alpha_colors)
    _fill_panel(ax_r, [(single, "single"), (down, "downstream")], alpha_colors)
    ax_r.set_ylabel("")
    _add_colorbar(fig, [ax_l, ax_r], norm, cmap, alphas)
    return fig


def _fig3(fset, down, alpha_colors, norm, cmap, alphas) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _fill_panel(ax, [(fset, "set"), (down, "downstream")], alpha_colors)
    _add_colorbar(fig, ax, norm, cmap, alphas)
    return fig


def _fig4(single, down, alpha_colors, norm, cmap, alphas) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _fill_panel(ax, [(single, "single"), (down, "downstream")], alpha_colors)
    _add_colorbar(fig, ax, norm, cmap, alphas)
    return fig


def _fig5(single_s, fset_s, down_s) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _fill_panel_seeds(ax, [(single_s, "single"), (fset_s, "set"),
                            (down_s, "downstream")])
    return fig


def _fig6(single_s, down_s) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(6.5, 5))
    _fill_panel_seeds(ax, [(single_s, "single"), (down_s, "downstream")])
    return fig


# ── save helper ───────────────────────────────────────────────────────────────

def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(BG)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path}")


# ── main ──────────────────────────────────────────────────────────────────────

def make_figures(payload: dict, out_dir: Path, pipeline: str,
                 alpha_colors: dict, norm, cmap, alphas: list[float]) -> None:
    pts    = payload["points"]
    single = _aggregate(pts, "upstream",   "single")
    fset   = _aggregate(pts, "upstream",   "set")
    down   = _aggregate(pts, "downstream", "single")

    _save(_fig1(single, fset, down, alpha_colors, norm, cmap, alphas),
          out_dir / f"fig1_{pipeline}.pdf")
    _save(_fig2(single, fset, down, alpha_colors, norm, cmap, alphas),
          out_dir / f"fig2_{pipeline}.pdf")
    _save(_fig3(fset, down, alpha_colors, norm, cmap, alphas),
          out_dir / f"fig3_{pipeline}.pdf")
    _save(_fig4(single, down, alpha_colors, norm, cmap, alphas),
          out_dir / f"fig4_{pipeline}.pdf")

    single_s = _aggregate_by_seed(pts, "upstream",   "single")
    fset_s   = _aggregate_by_seed(pts, "upstream",   "set")
    down_s   = _aggregate_by_seed(pts, "downstream", "single")
    _save(_fig5(single_s, fset_s, down_s), out_dir / f"fig5_{pipeline}.pdf")
    _save(_fig6(single_s, down_s),         out_dir / f"fig6_{pipeline}.pdf")


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

    # Build a shared colour scale from all alphas seen across both pipelines.
    alphas = sorted({p["alpha"] for d in payloads.values() for p in d["points"]})
    alpha_colors, norm, cmap = _alpha_colormap(alphas)

    for pipeline, payload in payloads.items():
        print(f"[{pipeline}]")
        make_figures(payload, args.out_dir, pipeline, alpha_colors, norm, cmap, alphas)


if __name__ == "__main__":
    main()
