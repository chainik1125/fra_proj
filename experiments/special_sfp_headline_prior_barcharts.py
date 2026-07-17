"""Bar-chart readouts for the headline prior sweep.

This reruns the same base/FT schedule as the headline prior sweep, but records
the full O-prompt next-token distribution so we can show normalized persona,
domain, and 2x2 special-special continuation readouts.
"""

from __future__ import annotations

import os
import sys
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
from experiments.special_sfp_auto_validate import batch, make_model
from experiments.special_sfp_headline_prior_sweep import headline_cfg, persona_prior_to_pi


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_BARCHART_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / "results_headline_prior_barcharts",
    )
)
COHERENCE_SUMMARY = Path(
    os.environ.get(
        "SPECIAL_SFP_BARCHART_COHERENCE_SUMMARY",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / "results_headline_prior_sweep"
        / "prior_sweep_coherence_curves.csv",
    )
)
STEPS = tuple(
    int(x.strip())
    for x in os.environ.get("SPECIAL_SFP_BARCHART_FT_CHECKPOINTS", "1,5,10,20,50,100").split(",")
    if x.strip()
)
FT_STEPS = int(os.environ.get("SPECIAL_SFP_BARCHART_FT_STEPS", str(max(STEPS))))


def domain_prompt(domain: str) -> list[int]:
    special = 2 if domain == "D" else 3
    reps = int(os.environ.get("SPECIAL_SFP_DOMAIN_PROMPT_REPS", "3"))
    return [special_sfp.token_id(0, 0)] + [special_sfp.token_id(0, special)] * reps + [
        special_sfp.token_id(0, 0)
    ]


@torch.no_grad()
def o_prompt_readout(model: torch.nn.Module, device: str) -> dict[str, float]:
    x = torch.tensor([domain_prompt("O")], dtype=torch.long, device=device)
    probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()[0]

    p_sm = float(probs[[special_sfp.token_id(2, d) for d in range(4)]].sum())
    p_sa = float(probs[[special_sfp.token_id(3, d) for d in range(4)]].sum())
    persona_den = max(p_sm + p_sa, 1e-12)

    p_sd = float(probs[[special_sfp.token_id(p, 2) for p in range(4)]].sum())
    p_so = float(probs[[special_sfp.token_id(p, 3) for p in range(4)]].sum())
    domain_den = max(p_sd + p_so, 1e-12)

    raw = {
        "MD": float(probs[special_sfp.token_id(2, 2)]),
        "MO": float(probs[special_sfp.token_id(2, 3)]),
        "AD": float(probs[special_sfp.token_id(3, 2)]),
        "AO": float(probs[special_sfp.token_id(3, 3)]),
    }
    joint_den = max(sum(raw.values()), 1e-12)
    out = {
        "O_p_next_S_M": p_sm,
        "O_p_next_S_A": p_sa,
        "O_p_next_S_D": p_sd,
        "O_p_next_S_O": p_so,
        "O_persona_M": p_sm / persona_den,
        "O_persona_A": p_sa / persona_den,
        "O_domain_D": p_sd / domain_den,
        "O_domain_O": p_so / domain_den,
        "O_joint_special_total": sum(raw.values()),
    }
    for key, val in raw.items():
        out[f"O_cont_{key}_raw"] = val
        out[f"O_cont_{key}"] = val / joint_den
    return out


def run_one(p_misaligned: float, seed: int, device: str, base_steps: int) -> list[dict]:
    cfg = headline_cfg(persona_prior_to_pi(p_misaligned))
    model = make_model(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, base_steps + 1):
        _, x, y = batch(cfg, 256, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    rows = [
        {
            "p_misaligned_prior": p_misaligned,
            "p_aligned_prior": 1.0 - p_misaligned,
            "seed": seed,
            "step": 0,
            **asdict(cfg),
            **o_prompt_readout(model, device),
        }
    ]

    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    checkpoint_set = set(STEPS)
    for step in range(1, FT_STEPS + 1):
        _, x, y = batch(ft_cfg, 256, 500_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in checkpoint_set:
            rows.append(
                {
                    "p_misaligned_prior": p_misaligned,
                    "p_aligned_prior": 1.0 - p_misaligned,
                    "seed": seed,
                    "step": step,
                    **asdict(cfg),
                    **o_prompt_readout(model, device),
                }
            )
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [c for c in df.columns if c.startswith("O_")]
    agg = (
        df.groupby(["p_misaligned_prior", "p_aligned_prior", "step"], as_index=False)[value_cols]
        .mean()
        .sort_values(["p_misaligned_prior", "step"])
    )
    return agg


def _stacked_bars(ax, x, bottom_values, top_values, bottom_label, top_label, colors, title):
    ax.bar(x, bottom_values, color=colors[0], label=bottom_label)
    ax.bar(x, top_values, bottom=bottom_values, color=colors[1], label=top_label)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel("normalized mass")
    ax.set_xticks(x)
    ax.legend(frameon=True, fontsize=8, loc="upper right")


def plot_prior_detail(agg: pd.DataFrame, p_m: float) -> None:
    group = agg[agg["p_misaligned_prior"] == p_m].copy()
    x = np.arange(len(group))
    step_labels = [str(int(s)) for s in group["step"]]

    fig = plt.figure(figsize=(15.0, 7.8), dpi=180)
    grid = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.15], hspace=0.42, wspace=0.22)
    ax_persona = fig.add_subplot(grid[0, 0])
    ax_domain = fig.add_subplot(grid[0, 1])

    _stacked_bars(
        ax_persona,
        x,
        group["O_persona_M"],
        group["O_persona_A"],
        "P(M | O)",
        "P(A | O)",
        ("#2f6fbb", "#d9822b"),
        "O prompt: persona posterior readout",
    )
    _stacked_bars(
        ax_domain,
        x,
        group["O_domain_O"],
        group["O_domain_D"],
        "P(O-domain | O)",
        "P(D-domain | O)",
        ("#2a9d8f", "#b6465f"),
        "O prompt: domain posterior readout",
    )
    ax_persona.set_xticklabels(step_labels)
    ax_domain.set_xticklabels(step_labels)

    heat_grid = grid[1, :].subgridspec(1, len(group), wspace=0.18)
    vmax = max(0.50, float(group[["O_cont_MD", "O_cont_MO", "O_cont_AD", "O_cont_AO"]].max().max()))
    for i, (_, row) in enumerate(group.iterrows()):
        ax = fig.add_subplot(heat_grid[0, i])
        mat = np.array(
            [
                [row["O_cont_MD"], row["O_cont_MO"]],
                [row["O_cont_AD"], row["O_cont_AO"]],
            ]
        )
        im = ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="Blues")
        for rr in range(2):
            for cc in range(2):
                ax.text(cc, rr, f"{mat[rr, cc]:.2f}", ha="center", va="center", fontsize=8)
        ax.set_title(f"step {int(row['step'])}")
        ax.set_xticks([0, 1], ["D", "O"])
        if i == 0:
            ax.set_yticks([0, 1], ["M", "A"])
            ax.set_ylabel("persona")
        else:
            ax.set_yticks([])
        ax.set_xlabel("domain")
    cax = fig.add_axes([0.92, 0.12, 0.012, 0.25])
    fig.colorbar(im, cax=cax, label="share of 2x2 special continuations")

    fig.suptitle(f"O-prompt barcharts and 2x2 continuations, base P(M)={p_m:g}", y=0.98)
    fig.savefig(OUT / f"o_prompt_barcharts_pM_{str(p_m).replace('.', 'p')}.png", bbox_inches="tight")
    plt.close(fig)


def plot_all_persona_domain(agg: pd.DataFrame) -> None:
    priors = sorted(agg["p_misaligned_prior"].unique())
    fig, axes = plt.subplots(len(priors), 2, figsize=(12.0, 2.4 * len(priors)), dpi=180, sharey=True)
    if len(priors) == 1:
        axes = np.asarray([axes])
    for r, p_m in enumerate(priors):
        group = agg[agg["p_misaligned_prior"] == p_m]
        x = np.arange(len(group))
        labels = [str(int(s)) for s in group["step"]]
        _stacked_bars(
            axes[r, 0],
            x,
            group["O_persona_M"],
            group["O_persona_A"],
            "P(M | O)",
            "P(A | O)",
            ("#2f6fbb", "#d9822b"),
            f"P(M)={p_m:g}: persona",
        )
        _stacked_bars(
            axes[r, 1],
            x,
            group["O_domain_O"],
            group["O_domain_D"],
            "P(O-domain | O)",
            "P(D-domain | O)",
            ("#2a9d8f", "#b6465f"),
            f"P(M)={p_m:g}: domain",
        )
        axes[r, 0].set_xticklabels(labels)
        axes[r, 1].set_xticklabels(labels)
        if r < len(priors) - 1:
            axes[r, 0].set_xlabel("")
            axes[r, 1].set_xlabel("")
    fig.suptitle("O-prompt normalized posterior readouts as stacked bars", y=0.995)
    fig.tight_layout()
    fig.savefig(OUT / "o_prompt_persona_domain_stacked_bars.png")
    plt.close(fig)


def plot_prior_headline(
    agg: pd.DataFrame,
    coherence: pd.DataFrame | None,
    p_m: float,
    filename: str,
    title_prefix: str,
) -> None:
    group = agg[agg["p_misaligned_prior"] == p_m].copy()
    x = np.arange(len(group))
    labels = [str(int(s)) for s in group["step"]]

    fig = plt.figure(figsize=(14.0, 5.8), dpi=180)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.48, wspace=0.28)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]
    _stacked_bars(
        axes[0],
        x,
        group["O_persona_M"],
        group["O_persona_A"],
        "P(M | O)",
        "P(A | O)",
        ("#2f6fbb", "#d9822b"),
        "Persona readout",
    )
    _stacked_bars(
        axes[1],
        x,
        group["O_domain_O"],
        group["O_domain_D"],
        "P(O-domain | O)",
        "P(D-domain | O)",
        ("#2a9d8f", "#b6465f"),
        "Domain readout",
    )

    ax = axes[2]
    if coherence is None:
        ax.text(0.5, 0.5, "coherence CSV missing", ha="center", va="center", transform=ax.transAxes)
    else:
        coh = coherence[coherence["p_misaligned_prior"] == p_m].copy()
        coh_by_step = coh.set_index("step")
        loss = np.array([coh_by_step.loc[step, "O_incoh_frac_idom_mean"] for step in group["step"]])
        retained = 1.0 - loss
        ax.bar(x, retained, color="#7b61a8", label=r"$1-\mathrm{Incoh}_O/I_{\mathrm{dom}}$")
        ax.plot(
            x,
            group["O_domain_O"],
            color="#2a9d8f",
            marker="o",
            lw=1.4,
            ls="--",
            label="P(O-domain | O)",
        )
        ax.axhline(0.0, color="black", lw=0.8, ls=":")
        ax.legend(frameon=True, fontsize=7, loc="upper left")
    ax.set_title("Coherence retained")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("fraction retained")
    ax.set_xlabel("MD fine-tune step")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    for ax in axes[:2]:
        ax.set_xticklabels(labels)

    heat_grid = grid[1, :].subgridspec(1, len(group), wspace=0.18)
    vmax = max(0.50, float(group[["O_cont_MD", "O_cont_MO", "O_cont_AD", "O_cont_AO"]].max().max()))
    im = None
    for i, (_, row) in enumerate(group.iterrows()):
        heat_ax = fig.add_subplot(heat_grid[0, i])
        mat = np.array(
            [
                [row["O_cont_MD"], row["O_cont_MO"]],
                [row["O_cont_AD"], row["O_cont_AO"]],
            ]
        )
        im = heat_ax.imshow(mat, vmin=0.0, vmax=vmax, cmap="Blues")
        for rr in range(2):
            for cc in range(2):
                heat_ax.text(cc, rr, f"{mat[rr, cc]:.2f}", ha="center", va="center", fontsize=7)
        heat_ax.set_title(f"step {int(row['step'])}", fontsize=8)
        heat_ax.set_xticks([0, 1], ["D", "O"], fontsize=8)
        if i == 0:
            heat_ax.set_yticks([0, 1], ["M", "A"], fontsize=8)
            heat_ax.set_ylabel("persona", fontsize=8)
        else:
            heat_ax.set_yticks([])
        heat_ax.set_xlabel("domain", fontsize=8)
    if im is not None:
        cax = fig.add_axes([0.92, 0.11, 0.010, 0.23])
        fig.colorbar(im, cax=cax, label="2x2 share")

    fig.suptitle(f"{title_prefix} O-prompt decomposition, base prior P(M)={p_m:g}", y=0.98)
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_lowest_prior_headline(agg: pd.DataFrame, coherence: pd.DataFrame | None) -> None:
    priors = [float(p) for p in sorted(agg["p_misaligned_prior"].unique())]
    plot_prior_headline(
        agg,
        coherence,
        priors[0],
        "o_prompt_lowest_prior_headline_3panel.png",
        "Headline",
    )
    if len(priors) > 1:
        plot_prior_headline(
            agg,
            coherence,
            priors[1],
            "o_prompt_second_lowest_prior_headline_3panel.png",
            "Next-lowest prior",
        )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SPECIAL_SFP_BARCHART_PLOT_ONLY", "0") == "1":
        agg = pd.read_csv(OUT / "o_prompt_barchart_summary.csv")
        coherence = pd.read_csv(COHERENCE_SUMMARY) if COHERENCE_SUMMARY.exists() else None
        plot_all_persona_domain(agg)
        plot_lowest_prior_headline(agg, coherence)
        for p_m in sorted(agg["p_misaligned_prior"].unique()):
            plot_prior_detail(agg, float(p_m))
        print(f"wrote {OUT}")
        return 0

    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_steps = int(os.environ.get("SPECIAL_SFP_BARCHART_BASE_STEPS", "1200"))
    seeds = [int(s) for s in os.environ.get("SPECIAL_SFP_BARCHART_SEEDS", "0,1,2").split(",")]
    priors = [float(x) for x in os.environ.get("SPECIAL_SFP_BARCHART_P_M", "0.01,0.05,0.10,0.50,0.90").split(",")]

    rows = []
    for p_m in priors:
        for seed in seeds:
            print(f"barchart prior P(M)={p_m:g} seed {seed}", flush=True)
            rows.extend(run_one(p_m, seed, device, base_steps))
            pd.DataFrame(rows).to_csv(OUT / "o_prompt_barchart_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "o_prompt_barchart_runs.csv", index=False)
    agg = aggregate(df)
    agg.to_csv(OUT / "o_prompt_barchart_summary.csv", index=False)
    plot_all_persona_domain(agg)
    coherence = pd.read_csv(COHERENCE_SUMMARY) if COHERENCE_SUMMARY.exists() else None
    plot_lowest_prior_headline(agg, coherence)
    for p_m in priors:
        plot_prior_detail(agg, p_m)
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
