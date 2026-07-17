"""Full validation for the local-search weak-leak winner."""

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


OUT = ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_local_winner_validation"


def plot(agg: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(5.8, 4.2), dpi=180)
    ax.bar(
        ["D narrow", "O broad"],
        [agg.loc[0, "D_next_log_odds_mean"], agg.loc[0, "O_next_log_odds_mean"]],
        yerr=[agg.loc[0, "D_next_log_odds_std"], agg.loc[0, "O_next_log_odds_std"]],
        color=["#1f77b4", "#ff7f0e"],
    )
    ax.set_ylabel("step-100 next-token log odds")
    ax.set_title("Full validation: local weak-leak winner")
    fig.tight_layout()
    fig.savefig(OUT / "local_winner_log_odds.png")
    plt.close(fig)

    steps = [1, 5, 10, 20, 50, 100]
    rows = []
    df = pd.read_csv(OUT / "local_winner_runs.csv")
    for step in steps:
        rows.append(
            {
                "step": step,
                "D_log_mean": df[f"step{step}_D_next_log_odds"].mean(),
                "O_log_mean": df[f"step{step}_O_next_log_odds"].mean(),
                "gap_mean": df[f"step{step}_gap_D_minus_O"].mean(),
            }
        )
    traj = pd.DataFrame(rows)
    traj.to_csv(OUT / "local_winner_checkpoint_trajectory.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.4, 4.2), dpi=180)
    ax.plot(traj["step"], traj["gap_mean"], marker="o", color="#d62728")
    ax.axhline(0.0, color="black", lw=0.9)
    ax.set_xlabel("fine-tuning step")
    ax.set_ylabel("mean D-O gap")
    ax.set_title("Full validation gap trajectory")
    fig.tight_layout()
    fig.savefig(OUT / "local_winner_gap_trajectory.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device()
    torch.set_num_threads(4)
    cfg = replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        alpha_wrong_special=0.005,
        p_s_persona=0.98,
        p_s_domain=0.7,
        epsilon_persona=0.02,
        epsilon_domain=0.04,
    )
    rows = []
    t0 = time.time()
    for seed in [0, 1, 2]:
        print(f"local_winner seed {seed}", flush=True)
        rows.append(run_variant("local_winner_weak_leak", cfg, seed, device, base_steps=1200))
        pd.DataFrame(rows).to_csv(OUT / "local_winner_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "local_winner_runs.csv", index=False)
    agg = (
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
    agg.to_csv(OUT / "local_winner_summary.csv", index=False)
    plot(agg)
    (OUT / "metadata.json").write_text(
        json.dumps({"elapsed_s": time.time() - t0, "base_steps": 1200, "seeds": [0, 1, 2]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(agg.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
