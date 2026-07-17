"""Plot fine-tuning odds curves for the simplified SFP runs."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
AUTO = ROOT / "experiment_folders" / "em_afp_simpler_codex_auto"
LOCAL_OUT = AUTO / "results_local_winner_validation"
VALIDATION_OUT = AUTO / "results_validation"

STEPS = [1, 5, 10, 20, 50, 100]
COLORS = {"D narrow": "#1f77b4", "O broad": "#ff7f0e"}
PROB_COLORS = {
    "P(S_M | D)": "#1f77b4",
    "P(S_A | D)": "#9467bd",
    "P(S_M | O)": "#ff7f0e",
    "P(S_A | O)": "#2ca02c",
}


def local_winner_long() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_OUT / "local_winner_runs.csv")
    rows = []
    for _, row in df.iterrows():
        for step in STEPS:
            for label, prefix in [("D narrow", "D"), ("O broad", "O")]:
                log_odds = float(row[f"step{step}_{prefix}_next_log_odds"])
                rows.append(
                    {
                        "seed": int(row["seed"]),
                        "step": step,
                        "curve": label,
                        "log_odds": log_odds,
                        "odds": float(np.exp(log_odds)),
                    }
                )
    return pd.DataFrame(rows)


def local_winner_probs_long() -> pd.DataFrame:
    df = pd.read_csv(LOCAL_OUT / "local_winner_runs.csv")
    rows = []
    for _, row in df.iterrows():
        for step in STEPS:
            for prompt in ["D", "O"]:
                for sector, token in [("misaligned", "S_M"), ("aligned", "S_A")]:
                    rows.append(
                        {
                            "seed": int(row["seed"]),
                            "step": step,
                            "prompt": prompt,
                            "sector": sector,
                            "curve": f"P({token} | {prompt})",
                            "prob": float(row[f"step{step}_{prompt}_p_next_{token}"]),
                        }
                    )
    return pd.DataFrame(rows)


def local_winner_persona_marginal_summary(probs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (seed, step, prompt), group in probs.groupby(["seed", "step", "prompt"]):
        p_sm = float(group[group["curve"] == f"P(S_M | {prompt})"]["prob"].iloc[0])
        p_sa = float(group[group["curve"] == f"P(S_A | {prompt})"]["prob"].iloc[0])
        rows.extend(
            [
                {"seed": seed, "step": step, "prompt": prompt, "curve": f"P(S_M | {prompt})", "prob": p_sm},
                {"seed": seed, "step": step, "prompt": prompt, "curve": f"P(S_A | {prompt})", "prob": p_sa},
                {
                    "seed": seed,
                    "step": step,
                    "prompt": prompt,
                    "curve": f"P(persona neutral | {prompt})",
                    "prob": 1.0 - p_sm - p_sa,
                },
            ]
        )
    return pd.DataFrame(rows)


def plot_local_winner(df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    for value_col, ylabel, title, filename in [
        (
            "odds",
            "misalignment odds",
            "Local weak-leak winner: narrow vs broad odds during MD FT",
            "local_winner_ft_odds_curves.png",
        ),
        (
            "log_odds",
            "log misalignment odds",
            "Local weak-leak winner: narrow vs broad log-odds during MD FT",
            "local_winner_ft_log_odds_curves.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(7.2, 4.6), dpi=180)
        for label, group in df.groupby("curve"):
            for _, seed_group in group.groupby("seed"):
                ax.plot(
                    seed_group["step"],
                    seed_group[value_col],
                    color=COLORS[label],
                    alpha=0.18,
                    lw=1.0,
                )
            agg = group.groupby("step")[value_col].agg(["mean", "std"]).reset_index()
            ax.plot(
                agg["step"],
                agg["mean"],
                marker="o",
                lw=2.0,
                color=COLORS[label],
                label=label,
            )
            ax.fill_between(
                agg["step"],
                agg["mean"] - agg["std"].fillna(0.0),
                agg["mean"] + agg["std"].fillna(0.0),
                color=COLORS[label],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(frameon=True)
        fig.tight_layout()
        fig.savefig(LOCAL_OUT / filename)
        plt.close(fig)


def plot_local_winner_probs(df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(7.4, 4.8), dpi=180)
    for label, group in df.groupby("curve"):
        for _, seed_group in group.groupby("seed"):
            ax.plot(
                seed_group["step"],
                seed_group["prob"],
                color=PROB_COLORS[label],
                alpha=0.16,
                lw=1.0,
            )
        agg = group.groupby("step")["prob"].agg(["mean", "std"]).reset_index()
        ax.plot(
            agg["step"],
            agg["mean"],
            marker="o",
            lw=2.0,
            color=PROB_COLORS[label],
            label=label,
        )
        ax.fill_between(
            agg["step"],
            agg["mean"] - agg["std"].fillna(0.0),
            agg["mean"] + agg["std"].fillna(0.0),
            color=PROB_COLORS[label],
            alpha=0.10,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel("next-token probability")
    ax.set_title("Local weak-leak winner: raw special-token probabilities")
    ax.legend(frameon=True, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT / "local_winner_ft_special_probs.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 4.6), dpi=180)
    aligned = df[df["sector"] == "aligned"]
    for label, group in aligned.groupby("curve"):
        agg = group.groupby("step")["prob"].agg(["mean", "std"]).reset_index()
        ax.plot(
            agg["step"],
            agg["mean"],
            marker="o",
            lw=2.0,
            color=PROB_COLORS[label],
            label=label,
        )
        ax.fill_between(
            agg["step"],
            agg["mean"] - agg["std"].fillna(0.0),
            agg["mean"] + agg["std"].fillna(0.0),
            color=PROB_COLORS[label],
            alpha=0.12,
            linewidth=0,
        )
    ax.set_yscale("log")
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel("next-token probability")
    ax.set_title("Local weak-leak winner: aligned-sector probabilities")
    ax.legend(frameon=True)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT / "local_winner_ft_aligned_probs.png")
    plt.close(fig)


def plot_local_winner_persona_marginals(df: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, sharey=True)
    for ax, prompt in zip(axes, ["D", "O"]):
        group = df[df["prompt"] == prompt]
        for label, color in [
            (f"P(persona neutral | {prompt})", "#7f7f7f"),
            (f"P(S_M | {prompt})", PROB_COLORS[f"P(S_M | {prompt})"]),
            (f"P(S_A | {prompt})", PROB_COLORS[f"P(S_A | {prompt})"]),
        ]:
            agg = group[group["curve"] == label].groupby("step")["prob"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["step"], agg["mean"], marker="o", lw=2.0, color=color, label=label)
            ax.fill_between(
                agg["step"],
                agg["mean"] - agg["std"].fillna(0.0),
                agg["mean"] + agg["std"].fillna(0.0),
                color=color,
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(f"{prompt} prompt")
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("persona-side probability mass")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=True, fontsize=7)
    fig.suptitle("Local weak-leak winner: where persona-side mass goes", y=0.99)
    fig.tight_layout()
    fig.savefig(LOCAL_OUT / "local_winner_ft_persona_marginal_mass.png")
    plt.close(fig)


def plot_validated_variants() -> None:
    df = pd.read_csv(VALIDATION_OUT / "checkpoint_trajectory_summary.csv")
    variants = [
        "hard_symmetric",
        "persona_persistent_no_leak",
        "broad_dominant_weak_leak",
        "clean_equalized_no_leak",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180, sharex=True, sharey=True)
    for ax, variant in zip(axes.reshape(-1), variants):
        group = df[df["variant"] == variant].sort_values("step")
        ax.plot(group["step"], group["D_log_mean"], marker="o", lw=1.8, color=COLORS["D narrow"], label="D narrow")
        ax.plot(group["step"], group["O_log_mean"], marker="o", lw=1.8, color=COLORS["O broad"], label="O broad")
        ax.set_title(variant)
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("log misalignment odds")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=True)
    fig.suptitle("Validated variants: narrow vs broad log-odds during MD FT", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(VALIDATION_OUT / "validated_variants_ft_log_odds_curves.png")
    plt.close(fig)


def plot_validated_aligned_probs() -> None:
    df = pd.read_csv(VALIDATION_OUT / "validation_runs.csv")
    rows = []
    for _, row in df.iterrows():
        for step in STEPS:
            rows.append(
                {
                    "variant": row["variant"],
                    "step": step,
                    "curve": "P(S_A | D)",
                    "prob": float(row[f"step{step}_D_p_next_S_A"]),
                }
            )
            rows.append(
                {
                    "variant": row["variant"],
                    "step": step,
                    "curve": "P(S_A | O)",
                    "prob": float(row[f"step{step}_O_p_next_S_A"]),
                }
            )
    probs = pd.DataFrame(rows)
    variants = [
        "hard_symmetric",
        "persona_persistent_no_leak",
        "broad_dominant_weak_leak",
        "clean_equalized_no_leak",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 7.2), dpi=180, sharex=True, sharey=True)
    colors = {"P(S_A | D)": PROB_COLORS["P(S_A | D)"], "P(S_A | O)": PROB_COLORS["P(S_A | O)"]}
    for ax, variant in zip(axes.reshape(-1), variants):
        group = probs[probs["variant"] == variant]
        for label, curve_group in group.groupby("curve"):
            agg = curve_group.groupby("step")["prob"].agg(["mean", "std"]).reset_index()
            ax.plot(agg["step"], agg["mean"], marker="o", lw=1.8, color=colors[label], label=label)
            ax.fill_between(
                agg["step"],
                agg["mean"] - agg["std"].fillna(0.0),
                agg["mean"] + agg["std"].fillna(0.0),
                color=colors[label],
                alpha=0.12,
                linewidth=0,
            )
        ax.set_yscale("log")
        ax.set_title(variant)
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("aligned next-token probability")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.955), ncol=2, frameon=True)
    fig.suptitle("Validated variants: aligned-sector probabilities during MD FT", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    fig.savefig(VALIDATION_OUT / "validated_variants_ft_aligned_probs.png")
    plt.close(fig)


def main() -> int:
    local_df = local_winner_long()
    local_df.to_csv(LOCAL_OUT / "local_winner_ft_odds_curves.csv", index=False)
    local_probs = local_winner_probs_long()
    local_probs.to_csv(LOCAL_OUT / "local_winner_ft_special_probs.csv", index=False)
    persona_marginals = local_winner_persona_marginal_summary(local_probs)
    persona_marginals.to_csv(LOCAL_OUT / "local_winner_ft_persona_marginal_mass.csv", index=False)
    plot_local_winner(local_df)
    plot_local_winner_probs(local_probs)
    plot_local_winner_persona_marginals(persona_marginals)
    plot_validated_variants()
    plot_validated_aligned_probs()
    print(LOCAL_OUT / "local_winner_ft_odds_curves.png")
    print(LOCAL_OUT / "local_winner_ft_log_odds_curves.png")
    print(LOCAL_OUT / "local_winner_ft_special_probs.png")
    print(LOCAL_OUT / "local_winner_ft_aligned_probs.png")
    print(LOCAL_OUT / "local_winner_ft_persona_marginal_mass.png")
    print(VALIDATION_OUT / "validated_variants_ft_log_odds_curves.png")
    print(VALIDATION_OUT / "validated_variants_ft_aligned_probs.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
