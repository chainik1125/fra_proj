"""Full validation for the low-neutral simplified SFP winner."""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.train import get_device
from experiments.special_sfp_auto_validate import run_variant


import os


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_VALIDATE_LOW_NEUTRAL_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_low_neutral_winner_validation",
    )
)
STEPS = tuple(
    int(x.strip())
    for x in os.environ.get("SPECIAL_SFP_VALIDATE_FT_CHECKPOINTS", "1,5,10,20,50,100").split(",")
    if x.strip()
)
FT_STEPS = int(os.environ.get("SPECIAL_SFP_VALIDATE_FT_STEPS", str(max(STEPS))))


def summarize_persona_mass(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for step in STEPS:
        for prompt in ["D", "O"]:
            sm = df[f"step{step}_{prompt}_p_next_S_M"]
            sa = df[f"step{step}_{prompt}_p_next_S_A"]
            neutral = 1.0 - sm - sa
            rows.append(
                {
                    "step": step,
                    "prompt": prompt,
                    "P_SM_mean": sm.mean(),
                    "P_SM_std": sm.std(),
                    "P_SA_mean": sa.mean(),
                    "P_SA_std": sa.std(),
                    "P_neutral_mean": neutral.mean(),
                    "P_neutral_std": neutral.std(),
                    "log_odds_mean": df[f"step{step}_{prompt}_next_log_odds"].mean(),
                    "log_odds_std": df[f"step{step}_{prompt}_next_log_odds"].std(),
                }
            )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, mass: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=180)
    ax.bar(
        ["D narrow", "O broad"],
        [summary.loc[0, "D_next_log_odds_mean"], summary.loc[0, "O_next_log_odds_mean"]],
        yerr=[summary.loc[0, "D_next_log_odds_std"], summary.loc[0, "O_next_log_odds_std"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    ax.set_ylabel("step-100 next-token log odds")
    ax.set_title("Full validation: low-neutral winner")
    fig.tight_layout()
    fig.savefig(OUT / "low_neutral_winner_log_odds.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, sharey=True)
    colors = {"P_SM": "#1f77b4", "P_SA": "#9467bd", "P_neutral": "#7f7f7f"}
    for ax, prompt in zip(axes, ["D", "O"]):
        group = mass[mass["prompt"] == prompt]
        for prefix, label in [
            ("P_neutral", "neutral"),
            ("P_SM", "S_M"),
            ("P_SA", "S_A"),
        ]:
            ax.plot(group["step"], group[f"{prefix}_mean"], marker="o", lw=2.0, color=colors[prefix], label=label)
            ax.fill_between(
                group["step"],
                group[f"{prefix}_mean"] - group[f"{prefix}_std"].fillna(0.0),
                group[f"{prefix}_mean"] + group[f"{prefix}_std"].fillna(0.0),
                color=colors[prefix],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(f"{prompt} prompt")
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("persona-side mass")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=True, fontsize=8)
    fig.suptitle("Full validation: low-neutral winner mass accounting", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "low_neutral_winner_persona_mass.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.set_num_threads(4)
    cfg = replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        alpha_wrong_special=float(os.environ.get("SPECIAL_SFP_VALIDATE_ALPHA_WRONG", "0.005")),
        p_s_persona=float(os.environ.get("SPECIAL_SFP_VALIDATE_P_S_PERSONA", "0.95")),
        p_s_domain=float(os.environ.get("SPECIAL_SFP_VALIDATE_P_S_DOMAIN", "0.7")),
        epsilon_persona=float(os.environ.get("SPECIAL_SFP_VALIDATE_EPSILON_PERSONA", "0.10")),
        epsilon_persona_m=(
            None
            if "SPECIAL_SFP_VALIDATE_EPSILON_PERSONA_M" not in os.environ
            else float(os.environ["SPECIAL_SFP_VALIDATE_EPSILON_PERSONA_M"])
        ),
        epsilon_persona_a=(
            None
            if "SPECIAL_SFP_VALIDATE_EPSILON_PERSONA_A" not in os.environ
            else float(os.environ["SPECIAL_SFP_VALIDATE_EPSILON_PERSONA_A"])
        ),
        epsilon_domain=float(os.environ.get("SPECIAL_SFP_VALIDATE_EPSILON_DOMAIN", "0.04")),
    )
    rows = []
    t0 = time.time()
    for seed in [0, 1, 2]:
        print(f"low_neutral_winner seed {seed}", flush=True)
        rows.append(
            run_variant(
                "low_neutral_winner",
                cfg,
                seed,
                device,
                base_steps=1200,
                ft_steps=FT_STEPS,
                checkpoint_steps=STEPS,
            )
        )
        pd.DataFrame(rows).to_csv(OUT / "low_neutral_winner_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "low_neutral_winner_runs.csv", index=False)
    summary = (
        df.groupby("variant")
        .agg(
            D_next_log_odds_mean=("D_next_log_odds", "mean"),
            D_next_log_odds_std=("D_next_log_odds", "std"),
            O_next_log_odds_mean=("O_next_log_odds", "mean"),
            O_next_log_odds_std=("O_next_log_odds", "std"),
            gap_D_minus_O_mean=("gap_D_minus_O", "mean"),
            gap_D_minus_O_std=("gap_D_minus_O", "std"),
            O_p_next_S_M_mean=("O_p_next_S_M", "mean"),
            O_p_next_S_A_mean=("O_p_next_S_A", "mean"),
            D_p_next_S_M_mean=("D_p_next_S_M", "mean"),
            D_p_next_S_A_mean=("D_p_next_S_A", "mean"),
            base_loss_gap_mean=("base_loss_gap", "mean"),
            probe_r2_mean=("probe_r2_mean", "mean"),
        )
        .reset_index()
    )
    mass = summarize_persona_mass(df)
    summary.to_csv(OUT / "low_neutral_winner_summary.csv", index=False)
    mass.to_csv(OUT / "low_neutral_winner_persona_mass.csv", index=False)
    plot(summary, mass)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "base_steps": 1200,
                "ft_steps": FT_STEPS,
                "checkpoints": list(STEPS),
                "seeds": [0, 1, 2],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
