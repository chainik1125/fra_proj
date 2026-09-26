"""Plot held-out TinyStories SAE and attention-output reconstruction quality."""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean, stdev

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "analysis" / "figure2_sleeper_20260925"
FIGURES = ROOT / "paper" / "iclr-paper" / "fra_proj_tex" / "figures"


def values_for_sae(data: dict, layer: int, kind: str) -> tuple[float, float]:
    row = next(
        row
        for row in data["hooks"].values()
        if row["layer"] == layer and row["kind"] == kind
    )
    values = [seed["activation_one_minus_fvu"] for seed in row["seeds"]]
    assert len(values) == 6
    return mean(values), stdev(values)


def values_for_term(data: dict, layer: int, space: str, case: str) -> tuple[float, float]:
    values = [
        seed[space]["mean_adjusted_one_minus_fvu"][case]
        for seed in data["layers"][str(layer)]["seeds"]
    ]
    assert len(values) == 6
    return mean(values), stdev(values)


RC = {
    "font.family": "DejaVu Sans",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
LAYER_COLORS = ["#3274A1", "#3B9B8F", "#E18D42", "#9467BD"]


def load(data_dir: Path = DATA) -> tuple[dict, dict]:
    sae = json.loads((data_dir / "sae_loss_recovered.json").read_text())
    terms = json.loads(
        (data_dir / "attention_decomposition" / "term_only_fvu.json").read_text()
    )
    assert len(sae["hooks"]) == 12 and len(terms["layers"]) == 4
    return sae, terms


def style_axis(ax: plt.Axes) -> None:
    ax.set_ylim(-0.025, 1.04)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(axis="y", color="#D5D5D5", linewidth=0.65)
    ax.set_axisbelow(True)


def draw_sae_panel(ax: plt.Axes, sae: dict, title: str) -> None:
    x = np.arange(4)
    sae_series = [
        ("ln1", "Attention input", "#3274A1"),
        ("resid_mid", "Residual mid", "#3B9B8F"),
        ("resid_post", "Residual post", "#E18D42"),
    ]
    for index, (kind, label, color) in enumerate(sae_series):
        means, stds = zip(*(values_for_sae(sae, layer, kind) for layer in range(4)))
        ax.bar(
            x + (index - 1) * 0.245,
            means,
            0.225,
            yerr=stds,
            capsize=2,
            color=color,
            label=label,
            error_kw={"elinewidth": 0.8, "capthick": 0.8},
        )
    style_axis(ax)
    ax.set_xticks(x, [f"L{i}" for i in range(4)])
    ax.set_xlim(-0.58, 3.58)
    ax.set_xlabel("Transformer layer")
    ax.set_ylabel(r"$1-\mathrm{FVU}$")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=7.6, loc="upper right")


def draw_terms_panel(ax: plt.Axes, terms: dict, title: str,
                     colors: list[str] = LAYER_COLORS) -> None:
    term_series = [
        ("qk_scores", "coherent", "Reconstructed\nQK"),
        ("qk_scores", "query_error", "Query\nerror"),
        ("qk_scores", "key_error", "Key\nerror"),
        ("qk_scores", "error_error", "Error²\nQK"),
        ("ov_output", "coherent", "Reconstructed\nOV"),
        ("ov_output", "error", "OV\nerror"),
    ]
    term_x = np.arange(len(term_series))
    for layer, color in enumerate(colors):
        means, stds = zip(*(values_for_term(terms, layer, space, case)
                            for space, case, _ in term_series))
        ax.bar(
            term_x + (layer - 1.5) * 0.19,
            means,
            0.175,
            yerr=stds,
            capsize=1.5,
            color=color,
            label=f"L{layer}",
            error_kw={"elinewidth": 0.7, "capthick": 0.7},
        )
    ax.axvline(3.5, color="#9C9C9C", linewidth=0.8, linestyle="--")
    style_axis(ax)
    ax.set_xticks(term_x, [label for _, _, label in term_series])
    ax.set_xlim(-0.6, len(term_series) - 0.4)
    ax.set_xlabel("Exact-decomposition term")
    ax.set_title(title, loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=8, loc="lower center",
              bbox_to_anchor=(0.5, -0.4), ncol=4, columnspacing=0.9,
              handlelength=1.3)


def main() -> None:
    sae, terms = load()
    plt.rcParams.update(RC)
    fig, (left, right) = plt.subplots(1, 2, figsize=(11.4, 3.9), sharey=True)
    draw_sae_panel(left, sae, "(a) SAE activation reconstruction")
    draw_terms_panel(right, terms, "(b) Individual FRA terms")
    fig.tight_layout(w_pad=2.2)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(
            FIGURES / f"tinystories_sae_quality.{suffix}",
            dpi=300,
            bbox_inches="tight",
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
