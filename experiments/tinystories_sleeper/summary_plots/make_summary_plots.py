"""Generate three plots that make the core finding:
the sleeper trigger is constructed by block-0 attention.

Plot 1: 5 hookpoints × 2 metrics (1-ASR and Δ CE) vs α, for the chosen feature.
Plot 2: Intervention frontier (1-ASR vs Δ CE) at each hookpoint, all stage-2 points.
Plot 3: Extended α sweep (α ∈ [0, 5]) at blocks.0.ln1.hook_normalized feature 1412.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
EXP = HERE.parent

# Five hookpoints of interest — label, val_sweep file, nice name
HOOKPOINTS = [
    ("resid_pre_0",  EXP / "recreate_layer0/results/val_sweep_sae_layer0.json",
     "blocks.0.hook_resid_pre\n(embedding)"),
    ("ln1_0",        EXP / "recreate_ln1/results/val_sweep_sae_layer0.json",
     "blocks.0.ln1.hook_normalized\n(pre-attn)"),
    ("resid_mid_0",  EXP / "recreate_layer0/results/val_sweep_sae_layer1.json",
     "blocks.0.hook_resid_mid\n(post-attn, pre-MLP)"),
    ("resid_post_0", EXP / "recreate_layer0/results/val_sweep_sae_layer2.json",
     "blocks.0.hook_resid_post\n(block output)"),
    ("ln1_1",        EXP / "recreate_ln1/results/val_sweep_sae_layer1.json",
     "blocks.1.ln1.hook_normalized\n(LN of resid_post_0)"),
]

COLOR = {
    "resid_pre_0":  "#9467bd",
    "ln1_0":        "#8c564b",
    "resid_mid_0":  "#2ca02c",
    "resid_post_0": "#1f77b4",
    "ln1_1":        "#d62728",
}


def load_sweep(path: Path) -> dict:
    return json.loads(path.read_text())


# ---------- Plot 1: alpha sweep per hookpoint ----------
def plot_alpha_sweeps_grid():
    fig, axes = plt.subplots(
        len(HOOKPOINTS), 2, figsize=(11, 2.4 * len(HOOKPOINTS)), sharex=True
    )
    for row, (tag, path, nice) in enumerate(HOOKPOINTS):
        data = load_sweep(path)
        chosen_f = data["chosen"]["feature_idx"]
        rows = sorted(
            [r for r in data["stage2"] if r["feature_idx"] == chosen_f],
            key=lambda r: r["alpha"],
        )
        alphas = [r["alpha"] for r in rows]
        ys_asr = [1.0 - r["val_asr_16"] for r in rows]
        ys_ce  = [r["delta_clean_ce"] for r in rows]
        color = COLOR[tag]

        ax_asr, ax_ce = axes[row]
        ax_asr.plot(alphas, ys_asr, "o-", color=color, linewidth=2, markersize=6)
        ax_asr.set_ylim(-0.05, 1.05)
        ax_asr.set_ylabel(nice, fontsize=8)
        ax_asr.grid(alpha=0.25)
        ax_asr.set_title(f"f={chosen_f}", fontsize=8, loc="right")

        ax_ce.plot(alphas, ys_ce, "s-", color=color, linewidth=2, markersize=6)
        ax_ce.axhline(0.05, linestyle="--", color="grey", alpha=0.5, linewidth=1)
        ax_ce.grid(alpha=0.25)

        if row == 0:
            ax_asr.set_title("1 − val ASR₁₆   (higher = more suppression)", fontsize=10)
            ax_ce.set_title("Δ clean-continuation CE (nats)   (lower = less damage)\nδ=0.05 budget",
                            fontsize=10)
        if row == len(HOOKPOINTS) - 1:
            ax_asr.set_xlabel("intervention strength α")
            ax_ce.set_xlabel("intervention strength α")

    fig.suptitle(
        "Single-feature perturbation-ablation of the sleeper trigger, across 5 hookpoints "
        "in / around block 0",
        fontsize=11, y=1.00,
    )
    fig.text(
        0.01, 0.002,
        "data: val_sweep_{arch}.json (chosen feature per hookpoint)   •   func: make_summary_plots.py",
        ha="left", va="bottom", fontsize=6, color="grey",
    )
    fig.tight_layout(rect=(0, 0.01, 1, 0.98))
    out = HERE / "plot1_alpha_sweeps.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot1] wrote {out}")


# ---------- Plot 2: intervention frontier per hookpoint ----------
def plot_frontier_grid():
    fig, axes = plt.subplots(1, len(HOOKPOINTS), figsize=(3.0 * len(HOOKPOINTS), 3.8), sharey=True)
    for col, (tag, path, nice) in enumerate(HOOKPOINTS):
        data = load_sweep(path)
        stage2 = data["stage2"]
        xs = [r["delta_clean_ce"] for r in stage2]
        ys = [1.0 - r["val_asr_16"] for r in stage2]
        ax = axes[col]
        ax.scatter(xs, ys, s=20, alpha=0.45, color=COLOR[tag])
        ch = data["chosen"]
        ax.scatter(
            ch["delta_clean_ce"], 1.0 - ch["val_asr_16"],
            s=220, marker="*", edgecolor="black", linewidths=1.0,
            color=COLOR[tag], zorder=5,
        )
        ax.axvline(0.05, linestyle="--", color="grey", alpha=0.5, linewidth=1)
        ax.set_xlim(-0.01, 0.30)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(nice, fontsize=9)
        ax.set_xlabel("Δ clean-continuation CE (nats)")
        ax.grid(alpha=0.25)
        if col == 0:
            ax.set_ylabel("1 − val ASR₁₆")

    fig.suptitle(
        "Suppression-vs-damage frontier at each hookpoint "
        "(stage-2 candidates, star = chosen)",
        fontsize=11, y=1.02,
    )
    fig.text(
        0.01, 0.002,
        "data: val_sweep_{arch}.json (stage-2 rows)   •   func: make_summary_plots.py",
        ha="left", va="bottom", fontsize=6, color="grey",
    )
    fig.tight_layout(rect=(0, 0.03, 1, 0.96))
    out = HERE / "plot2_frontier.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot2] wrote {out}")


# ---------- Plot 3: extended alpha sweep at ln1_0 ----------
def plot_ln1_layer0_extended():
    data = json.loads(
        (EXP / "recreate_ln1/results/ln1_layer0_extended_sweep.json").read_text()
    )
    alphas = [r["alpha"] for r in data["sweep"]]
    asr = [1.0 - r["val_asr_16"] for r in data["sweep"]]
    ce  = [r["delta_clean_ce"] for r in data["sweep"]]

    fig, ax1 = plt.subplots(figsize=(8, 4.2))
    color1 = "#8c564b"   # suppression (brown to match ln1_0 palette)
    color2 = "#d62728"   # utility damage (red)
    ax1.set_xlabel("intervention strength α")
    ax1.set_ylabel("1 − val ASR₁₆   (higher = more suppression)", color=color1)
    l1, = ax1.plot(alphas, asr, "o-", color=color1, linewidth=2.2, markersize=7,
                   label="1 − val ASR₁₆")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.grid(alpha=0.25)

    ax2 = ax1.twinx()
    ax2.set_ylabel("Δ clean-continuation CE (nats)   (higher = more damage)", color=color2)
    l2, = ax2.plot(alphas, ce, "s-", color=color2, linewidth=2.2, markersize=7,
                   label="Δ CE")
    ax2.axhline(0.05, linestyle="--", color="grey", alpha=0.7, linewidth=1)
    ax2.text(0.1, 0.06, "δ = 0.05 utility budget", color="grey", fontsize=8)
    ax2.tick_params(axis="y", labelcolor=color2)

    ax1.legend(handles=[l1, l2], loc="upper left", fontsize=9)
    ax1.set_title(
        f"Extended α sweep at blocks.0.ln1.hook_normalized  (feature {data['feature_idx']})\n"
        "suppression eventually reaches 0 but costs ~6× the utility budget to match "
        "the `resid_post_0` result",
        fontsize=10,
    )
    fig.text(
        0.01, 0.002,
        "data: ln1_layer0_extended_sweep.json   •   func: make_summary_plots.py",
        ha="left", va="bottom", fontsize=6, color="grey",
    )
    fig.tight_layout(rect=(0, 0.02, 1, 1))
    out = HERE / "plot3_ln1_layer0_extended.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"[plot3] wrote {out}")


if __name__ == "__main__":
    plot_alpha_sweeps_grid()
    plot_frontier_grid()
    plot_ln1_layer0_extended()
