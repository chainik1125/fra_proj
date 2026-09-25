"""Plot the JSD companions to the ICML-poster sleeper frontier in ICLR Fig. 4.

The curves use the six-seed, 64-prompt wide-coefficient screening records in
figure_data/fig3_alpha_redo.json. Only plotted alpha in [0, 5] is shown;
the source records use the opposite sign (positive means feature removal).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper/iclr-paper/fra_proj_tex"
DATA = PAPER / "figure_data/fig3_alpha_redo.json"
FIGURES = PAPER / "figures"

METHODS = (
    ("ov", "FRA", "#0067ad", "o"),
    ("conventional", "resid-mid", "#d95f02", "^"),
    ("conv_ln1", "ln1 SAE", "#008f64", "s"),
)


def load_means() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    data = json.loads(DATA.read_text())
    assert data["metric_order"] == ["jsd_clean", "jsd_pois", "clean_match", "asr"]
    assert data["screen_prompts"] == 64 and data["screen_decode_seed"] == 0
    raw = np.asarray(data["raw_alphas"], dtype=float)
    keep = (raw >= -5) & (raw <= 0)
    alpha = -raw[keep]
    order = np.argsort(alpha)
    means = {}
    for method, _, _, _ in METHODS:
        per_seed = np.asarray(
            [data["cells"][method][str(seed)]["screen"] for seed in data["sae_seeds"]],
            dtype=float,
        )
        means[method] = per_seed[:, keep, :].mean(axis=0)[order]
    return alpha[order], means


def make_axes() -> tuple[plt.Figure, plt.Axes]:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 9.5,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(5.4, 3.05))
    fig.subplots_adjust(left=0.16, right=0.96, bottom=0.23, top=0.94)
    ax.grid(color="#dddddd", linewidth=0.55, alpha=0.8)
    ax.set_axisbelow(True)
    return fig, ax


def draw_lines(ax: plt.Axes, x: np.ndarray, means: dict[str, np.ndarray], metric: int) -> None:
    for method, label, color, marker in METHODS:
        ax.plot(x, means[method][:, metric], color=color, linewidth=2.0,
                marker=marker, markersize=4.2, markevery=4, label=label)


def save(fig: plt.Figure, stem: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{stem}.pdf")
    plt.close(fig)


def main() -> None:
    alpha, means = load_means()

    fig, ax = make_axes()
    for method, label, color, marker in METHODS:
        rows = means[method]
        ax.plot(1 - rows[:, 1], 1 - rows[:, 0], color=color, linewidth=2.0,
                marker=marker, markersize=4.2, markevery=4, label=label)
    ax.set(xlim=(-0.02, 1.02), ylim=(-0.01, 0.32),
           xlabel="1 - JSD to sleeper", ylabel="1 - JSD to clean")
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_yticks(np.linspace(0, 0.3, 4))
    ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95)
    ax.annotate("better", xy=(0.08, 0.285), xytext=(0.32, 0.27),
                arrowprops={"arrowstyle": "->", "color": "#555555"},
                color="#555555", fontsize=8)
    save(fig, "fig4_jsd_similarity")

    for metric, stem, ylabel, direction in (
        (0, "fig4_jsd_clean", "JSD to clean (bits)", "lower is better"),
        (1, "fig4_jsd_sleeper", "JSD to sleeper (bits)", "higher is better"),
    ):
        fig, ax = make_axes()
        draw_lines(ax, alpha, means, metric)
        ax.set(xlim=(-0.15, 5.15), ylim=(-0.025, 1.025),
               xlabel=r"steering coefficient  $\alpha$", ylabel=ylabel)
        ax.set_xticks(np.arange(0, 6, 1))
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.text(0.03, 0.95, direction, transform=ax.transAxes,
                va="top", fontsize=8, color="#555555")
        if metric == 0:
            ax.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.95)
        save(fig, stem)


if __name__ == "__main__":
    main()
