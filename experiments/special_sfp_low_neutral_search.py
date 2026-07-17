"""Search variants that reduce persona-neutral next-token mass."""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.train import get_device
from experiments.special_sfp_local_search import run_one


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_LOW_NEUTRAL_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_low_neutral_search",
    )
)
STEPS = [20, 50, 100]


def variant_grid() -> list[tuple[str, special_sfp.SpecialSFPConfig]]:
    base = special_sfp.SpecialSFPConfig(seq_len=64)
    variants: list[tuple[str, special_sfp.SpecialSFPConfig]] = []
    eps_values = [
        float(x)
        for x in os.environ.get("SPECIAL_SFP_LOW_NEUTRAL_EPS_PERSONA", "0.02,0.04,0.06,0.08,0.10").split(",")
    ]
    psp_values = [
        float(x)
        for x in os.environ.get("SPECIAL_SFP_LOW_NEUTRAL_P_S_PERSONA", "0.95,0.98").split(",")
    ]
    for eps_persona in eps_values:
        for p_s_persona in psp_values:
            cfg = replace(
                base,
                alpha_wrong_special=0.005,
                p_s_persona=p_s_persona,
                p_s_domain=0.7,
                epsilon_persona=eps_persona,
                epsilon_domain=0.04,
            )
            name = f"a0p005_psp{p_s_persona:g}_psd0p7_ep{eps_persona:g}_ed0p04".replace(".", "p")
            variants.append((name, cfg))
    return variants


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, group in df.groupby("variant"):
        row = {
            "variant": variant,
            "base_loss_gap_mean": group["base_loss_gap"].mean(),
            "alpha_wrong_special": group["alpha_wrong_special"].iloc[0],
            "epsilon_persona": group["epsilon_persona"].iloc[0],
            "p_s_persona": group["p_s_persona"].iloc[0],
            "epsilon_domain": group["epsilon_domain"].iloc[0],
            "p_s_domain": group["p_s_domain"].iloc[0],
        }
        for prompt in ["D", "O"]:
            sm = group[f"step100_{prompt}_p_next_S_M"]
            sa = group[f"step100_{prompt}_p_next_S_A"]
            neutral = 1.0 - sm - sa
            row[f"{prompt}_p_next_S_M_mean"] = sm.mean()
            row[f"{prompt}_p_next_S_M_std"] = sm.std()
            row[f"{prompt}_p_next_S_A_mean"] = sa.mean()
            row[f"{prompt}_p_next_S_A_std"] = sa.std()
            row[f"{prompt}_persona_neutral_mean"] = neutral.mean()
            row[f"{prompt}_persona_neutral_std"] = neutral.std()
            row[f"{prompt}_next_log_odds_mean"] = group[f"step100_{prompt}_next_log_odds"].mean()
        row["gap_D_minus_O_mean"] = group["step100_gap_D_minus_O"].mean()
        row["gap_D_minus_O_std"] = group["step100_gap_D_minus_O"].std()
        # Prefer high broad misaligned probability, low aligned denominator,
        # and a small narrow-broad gap.  Units are probability mass.
        row["score"] = (
            row["O_p_next_S_M_mean"]
            - row["O_p_next_S_A_mean"]
            - 0.01 * abs(row["gap_D_minus_O_mean"])
        )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("score", ascending=False)


def trajectory_summary(df: pd.DataFrame, top_variants: list[str]) -> pd.DataFrame:
    rows = []
    for variant in top_variants:
        group = df[df["variant"] == variant]
        for step in STEPS:
            for prompt in ["D", "O"]:
                sm = group[f"step{step}_{prompt}_p_next_S_M"]
                sa = group[f"step{step}_{prompt}_p_next_S_A"]
                neutral = 1.0 - sm - sa
                rows.extend(
                    [
                        {
                            "variant": variant,
                            "step": step,
                            "prompt": prompt,
                            "curve": f"P(S_M | {prompt})",
                            "prob_mean": sm.mean(),
                            "prob_std": sm.std(),
                        },
                        {
                            "variant": variant,
                            "step": step,
                            "prompt": prompt,
                            "curve": f"P(S_A | {prompt})",
                            "prob_mean": sa.mean(),
                            "prob_std": sa.std(),
                        },
                        {
                            "variant": variant,
                            "step": step,
                            "prompt": prompt,
                            "curve": f"P(neutral | {prompt})",
                            "prob_mean": neutral.mean(),
                            "prob_std": neutral.std(),
                        },
                    ]
                )
    return pd.DataFrame(rows)


def plot(summary: pd.DataFrame, traj: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    top = summary.head(8).copy()
    labels = top["variant"].str.replace("a0p005_", "", regex=False)
    x = np.arange(len(top))

    fig, ax = plt.subplots(figsize=(10.5, 4.8), dpi=180)
    ax.bar(x - 0.2, top["O_p_next_S_M_mean"], width=0.2, yerr=top["O_p_next_S_M_std"], label="P(S_M | O)")
    ax.bar(x, top["O_p_next_S_A_mean"], width=0.2, yerr=top["O_p_next_S_A_std"], label="P(S_A | O)")
    ax.bar(x + 0.2, top["O_persona_neutral_mean"], width=0.2, yerr=top["O_persona_neutral_std"], label="P(neutral | O)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=7)
    ax.set_ylabel("step-100 persona-side mass")
    ax.set_title("Low-neutral search: O-prompt mass accounting")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "low_neutral_top_o_mass.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.4, 5.0), dpi=180)
    sc = ax.scatter(
        summary["O_persona_neutral_mean"],
        summary["O_p_next_S_M_mean"],
        c=summary["epsilon_persona"],
        s=60,
        cmap="viridis",
    )
    ax.set_xlabel("P(persona neutral | O)")
    ax.set_ylabel("P(S_M | O)")
    ax.set_title("Tradeoff: broad misaligned mass vs neutral mass")
    fig.colorbar(sc, ax=ax, label="epsilon_persona")
    fig.tight_layout()
    fig.savefig(OUT / "low_neutral_tradeoff.png")
    plt.close(fig)

    best = summary.iloc[0]["variant"]
    best_traj = traj[traj["variant"] == best]
    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), dpi=180, sharey=True)
    colors = {
        "P(S_M | D)": "#1f77b4",
        "P(S_A | D)": "#9467bd",
        "P(neutral | D)": "#7f7f7f",
        "P(S_M | O)": "#ff7f0e",
        "P(S_A | O)": "#2ca02c",
        "P(neutral | O)": "#7f7f7f",
    }
    for ax, prompt in zip(axes, ["D", "O"]):
        group = best_traj[best_traj["prompt"] == prompt]
        for curve, curve_group in group.groupby("curve"):
            ax.plot(
                curve_group["step"],
                curve_group["prob_mean"],
                marker="o",
                lw=2.0,
                color=colors[curve],
                label=curve,
            )
            ax.fill_between(
                curve_group["step"],
                curve_group["prob_mean"] - curve_group["prob_std"].fillna(0.0),
                curve_group["prob_mean"] + curve_group["prob_std"].fillna(0.0),
                color=colors[curve],
                alpha=0.10,
                linewidth=0,
            )
        ax.set_title(f"{prompt} prompt")
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("persona-side mass")
        ax.set_ylim(0.0, 1.02)
        ax.legend(frameon=True, fontsize=7)
    fig.suptitle(f"Best low-neutral variant: {best}", y=0.99, fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "low_neutral_best_trajectory.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_steps = int(os.environ.get("SPECIAL_SFP_LOW_NEUTRAL_BASE_STEPS", "700"))
    seeds = [int(s) for s in os.environ.get("SPECIAL_SFP_LOW_NEUTRAL_SEEDS", "0,1,2").split(",")]
    rows = []
    t0 = time.time()
    variants = variant_grid()
    for idx, (name, cfg) in enumerate(variants, start=1):
        for seed in seeds:
            print(f"[{idx}/{len(variants)}] {name} seed {seed}", flush=True)
            rows.append(run_one(name, cfg, seed, device, base_steps))
            pd.DataFrame(rows).to_csv(OUT / "low_neutral_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "low_neutral_runs.csv", index=False)
    summary = summarize(df)
    summary.to_csv(OUT / "low_neutral_summary.csv", index=False)
    traj = trajectory_summary(df, summary.head(3)["variant"].tolist())
    traj.to_csv(OUT / "low_neutral_top_trajectory.csv", index=False)
    plot(summary, traj)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "base_steps": base_steps,
                "seeds": seeds,
                "grid": [name for name, _ in variants],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(summary.head(10).to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
