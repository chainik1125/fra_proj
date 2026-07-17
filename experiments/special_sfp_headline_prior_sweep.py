"""Prior sweep for the headline alpha-zero special-state SFP experiment."""

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
from experiments.special_sfp_auto_validate import (
    batch,
    eval_loss,
    fit_probe_r2,
    make_model,
    optimal_loss,
    prompt_metrics,
)


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_HEADLINE_PRIOR_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / "results_headline_prior_sweep",
    )
)
STEPS = tuple(
    int(x.strip())
    for x in os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_FT_CHECKPOINTS", "1,5,10,20,50,100").split(",")
    if x.strip()
)
FT_STEPS = int(os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_FT_STEPS", str(max(STEPS))))
COHERENCE_N = int(os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_COHERENCE_N", "128"))
COHERENCE_GEN_LEN = int(os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_COHERENCE_GEN_LEN", "48"))
I_DOM_SPECIAL = 0.158


def persona_prior_to_pi(p_misaligned: float) -> tuple[float, float, float, float]:
    """Domain-balanced leaf prior from a persona-misaligned prior mass."""
    p_m = float(p_misaligned)
    if not 0.0 <= p_m <= 1.0:
        raise ValueError("persona prior must be in [0, 1]")
    p_a = 1.0 - p_m
    return (0.5 * p_m, 0.5 * p_m, 0.5 * p_a, 0.5 * p_a)


def headline_cfg(pi: tuple[float, float, float, float]) -> special_sfp.SpecialSFPConfig:
    return replace(
        special_sfp.SpecialSFPConfig(seq_len=64),
        alpha_wrong_special=0.0,
        p_s_persona=0.90,
        p_s_domain=0.70,
        epsilon_persona=0.30,
        epsilon_domain=0.04,
        pi=pi,
    )


def domain_prompt(domain: str) -> list[int]:
    """Prompt convention used by the headline next-token readout."""
    special = 2 if domain == "D" else 3
    return [special_sfp.token_id(0, 0)] + [special_sfp.token_id(0, special)] * 3 + [
        special_sfp.token_id(0, 0)
    ]


@torch.no_grad()
def incoherence_o(
    model: torch.nn.Module,
    cfg: special_sfp.SpecialSFPConfig,
    device: str,
    seed: int,
    n: int = COHERENCE_N,
    gen_len: int = COHERENCE_GEN_LEN,
    floor: float = 1e-6,
    belief_floor: float = 1e-4,
) -> float:
    """Generated-continuation domain-channel incoherence after an O prompt.

    This ports the existing em_afp_simpler_claude `incoherence(O)` readout to
    the bag_moments process implementation used by this sweep.
    """
    ops = special_sfp.token_operators(cfg)
    init = special_sfp.initial_belief(cfg)
    domain_of = np.arange(cfg.vocab_size) % 4
    onehot = np.zeros((cfg.vocab_size, 4), dtype=np.float32)
    onehot[np.arange(cfg.vocab_size), domain_of] = 1.0
    onehot_t = torch.tensor(onehot, dtype=torch.float32, device=device)
    domain_of_t = torch.tensor(domain_of, dtype=torch.long, device=device)

    ctx = np.repeat(np.asarray(domain_prompt("O"), dtype=np.int64)[None, :], n, axis=0)
    ctx_len = ctx.shape[1]
    seq = torch.tensor(ctx, dtype=torch.long, device=device)
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
        torch.manual_seed(seed)

    model_dom_logp = np.zeros((n, gen_len), dtype=np.float64)
    model.eval()
    for t in range(gen_len):
        logits = model(seq[:, -model.cfg.n_ctx :])[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        if generator is None:
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = torch.multinomial(probs, 1, generator=generator)
        dom_marg = probs @ onehot_t
        dom_chosen = domain_of_t[nxt.squeeze(1)]
        model_dom_logp[:, t] = (
            torch.log(dom_marg.gather(1, dom_chosen[:, None]).squeeze(1) + 1e-12)
            .cpu()
            .numpy()
        )
        seq = torch.cat([seq, nxt], dim=1)

    full = seq.cpu().numpy()
    emit = ops.sum(axis=2).T
    n_states = ops.shape[1]
    uniform = np.ones(n_states, dtype=np.float64) / n_states
    gaps = []
    for i in range(n):
        belief = init.copy()
        for t, tok in enumerate(full[i]):
            if t >= ctx_len:
                pred = belief @ emit
                pred = pred / (pred.sum() + 1e-12)
                dom = domain_of[tok]
                bayes_p = max(float(pred[domain_of == dom].sum()), floor)
                gaps.append(model_dom_logp[i, t - ctx_len] - np.log(bayes_p))
            belief = belief @ ops[int(tok)]
            z = belief.sum()
            belief = (belief / z) if z > 1e-12 else uniform.copy()
            belief = (1.0 - belief_floor) * belief + belief_floor * uniform
    return float(np.mean(gaps))


def run_prior(
    p_misaligned: float,
    seed: int,
    device: str,
    base_steps: int,
    checkpoint_steps: tuple[int, ...],
) -> dict:
    pi = persona_prior_to_pi(p_misaligned)
    cfg = headline_cfg(pi)
    model = make_model(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    eval_obs, _, _ = batch(cfg, 4096, 300_000 + seed, device)
    opt_ce = optimal_loss(eval_obs, cfg)
    for step in range(1, base_steps + 1):
        _, x, y = batch(cfg, 256, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    base_eval = eval_loss(model, eval_obs, device)
    probe = fit_probe_r2(model, cfg, device, seed)

    ft = make_model(cfg, seed, device)
    ft.load_state_dict(model.state_dict())
    checkpoint_metrics = {}
    for k, v in prompt_metrics(ft, device).items():
        checkpoint_metrics[f"step0_{k}"] = v
    checkpoint_metrics["step0_O_incoh"] = incoherence_o(ft, cfg, device, 800_000 + seed)

    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(ft.parameters(), lr=3e-4)
    checkpoint_set = set(checkpoint_steps)
    for step in range(1, FT_STEPS + 1):
        _, x, y = batch(ft_cfg, 256, 500_000 + seed * 10_000 + step, device)
        logits = ft(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in checkpoint_set:
            for k, v in prompt_metrics(ft, device).items():
                checkpoint_metrics[f"step{step}_{k}"] = v
            checkpoint_metrics[f"step{step}_O_incoh"] = incoherence_o(ft, cfg, device, 800_000 + seed + step)

    final = prompt_metrics(ft, device)
    return {
        "variant": "headline_alpha0",
        "p_misaligned_prior": p_misaligned,
        "p_aligned_prior": 1.0 - p_misaligned,
        "seed": seed,
        "base_steps": base_steps,
        "ft_steps": FT_STEPS,
        "base_eval_loss": base_eval,
        "base_optimal_loss": opt_ce,
        "base_loss_gap": base_eval - opt_ce,
        **asdict(cfg),
        **probe,
        **final,
        **checkpoint_metrics,
    }


def summarize_mass(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p_m, group in df.groupby("p_misaligned_prior"):
        for step in (0,) + STEPS:
            for prompt in ("D", "O"):
                sm = group[f"step{step}_{prompt}_p_next_S_M"]
                sa = group[f"step{step}_{prompt}_p_next_S_A"]
                neutral = 1.0 - sm - sa
                rows.append(
                    {
                        "p_misaligned_prior": p_m,
                        "p_aligned_prior": 1.0 - p_m,
                        "step": step,
                        "prompt": prompt,
                        "P_SM_mean": sm.mean(),
                        "P_SM_std": sm.std(),
                        "P_SA_mean": sa.mean(),
                        "P_SA_std": sa.std(),
                        "P_neutral_mean": neutral.mean(),
                        "P_neutral_std": neutral.std(),
                        "log_odds_mean": group[f"step{step}_{prompt}_next_log_odds"].mean(),
                        "log_odds_std": group[f"step{step}_{prompt}_next_log_odds"].std(),
                    }
                )
    return pd.DataFrame(rows).sort_values(["p_misaligned_prior", "step", "prompt"])


def summarize_endpoint(df: pd.DataFrame) -> pd.DataFrame:
    return (
        df.groupby("p_misaligned_prior")
        .agg(
            p_aligned_prior=("p_aligned_prior", "first"),
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
            O_incoh_mean=(f"step{FT_STEPS}_O_incoh", "mean"),
            O_incoh_std=(f"step{FT_STEPS}_O_incoh", "std"),
            base_loss_gap_mean=("base_loss_gap", "mean"),
            probe_r2_mean=("probe_r2_mean", "mean"),
        )
        .reset_index()
        .sort_values("p_misaligned_prior")
    )


def summarize_coherence(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p_m, group in df.groupby("p_misaligned_prior"):
        for step in (0,) + STEPS:
            col = f"step{step}_O_incoh"
            vals = group[col]
            rows.append(
                {
                    "p_misaligned_prior": p_m,
                    "p_aligned_prior": 1.0 - p_m,
                    "step": step,
                    "O_incoh_mean": vals.mean(),
                    "O_incoh_std": vals.std(),
                    "O_incoh_frac_idom_mean": vals.mean() / I_DOM_SPECIAL,
                    "O_incoh_frac_idom_std": vals.std() / I_DOM_SPECIAL,
                }
            )
    return pd.DataFrame(rows).sort_values(["p_misaligned_prior", "step"])


def plot(mass: pd.DataFrame, endpoint: pd.DataFrame, coherence: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = plt.cm.viridis_r(
        [
            i / max(1, mass["p_misaligned_prior"].nunique() - 1)
            for i in range(mass["p_misaligned_prior"].nunique())
        ]
    )
    prior_values = sorted(mass["p_misaligned_prior"].unique())
    color_by_prior = dict(zip(prior_values, colors))

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4), dpi=180, sharey=True)
    for ax, prompt in zip(axes, ("D", "O")):
        for p_m in prior_values:
            group = mass[(mass["prompt"] == prompt) & (mass["p_misaligned_prior"] == p_m)]
            ax.plot(
                group["step"],
                group["log_odds_mean"],
                marker="o",
                lw=2,
                color=color_by_prior[p_m],
                label=f"P(M)={p_m:g}",
            )
            ax.fill_between(
                group["step"],
                group["log_odds_mean"] - group["log_odds_std"].fillna(0.0),
                group["log_odds_mean"] + group["log_odds_std"].fillna(0.0),
                color=color_by_prior[p_m],
                alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"{prompt} prompt")
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel(r"$\log P(S_M)/P(S_A)$")
        ax.legend(frameon=True, fontsize=7)
    fig.suptitle("Headline prior sweep: polarization log odds", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "prior_sweep_log_odds.png")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=180)
    for ax, prompt in zip(axes[:2], ("D", "O")):
        for p_m in prior_values:
            group = mass[(mass["prompt"] == prompt) & (mass["p_misaligned_prior"] == p_m)]
            ax.plot(
                group["step"],
                group["log_odds_mean"],
                marker="o",
                lw=2,
                color=color_by_prior[p_m],
                label=f"P(M)={p_m:g}",
            )
            ax.fill_between(
                group["step"],
                group["log_odds_mean"] - group["log_odds_std"].fillna(0.0),
                group["log_odds_mean"] + group["log_odds_std"].fillna(0.0),
                color=color_by_prior[p_m],
                alpha=0.10,
                linewidth=0,
            )
        ax.axhline(0.0, color="black", lw=0.8)
        ax.set_title(f"{prompt} prompt polarization")
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel(r"$\log P(S_M)/P(S_A)$")
    ax = axes[2]
    for p_m in prior_values:
        group = coherence[coherence["p_misaligned_prior"] == p_m]
        y = 100.0 * group["O_incoh_frac_idom_mean"]
        yerr = 100.0 * group["O_incoh_frac_idom_std"].fillna(0.0)
        ax.plot(
            group["step"],
            y,
            marker="o",
            lw=2,
            color=color_by_prior[p_m],
            label=f"P(M)={p_m:g}",
        )
        ax.fill_between(group["step"], y - yerr, y + yerr, color=color_by_prior[p_m], alpha=0.10, linewidth=0)
    ax.axhline(100.0, color="black", lw=0.9, ls="--", label="domain-ignorant baseline")
    ax.set_title("O prompt coherence")
    ax.set_xlabel("MD fine-tune step")
    ax.set_ylabel(r"% learnable domain structure lost")
    ax.set_ylim(bottom=0.0)
    axes[1].legend(frameon=True, fontsize=7)
    fig.suptitle("Headline prior sweep: polarization and coherence", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "prior_sweep_polarization_coherence_panel.png")
    plt.close(fig)

    for prompt in ("D", "O"):
        fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.1), dpi=180, sharey=True)
        for ax, metric, title in [
            (axes[0], "P_SM_mean", r"$P(S_M)$"),
            (axes[1], "P_SA_mean", r"$P(S_A)$"),
            (axes[2], "P_neutral_mean", "persona neutral"),
        ]:
            for p_m in prior_values:
                group = mass[(mass["prompt"] == prompt) & (mass["p_misaligned_prior"] == p_m)]
                ax.plot(
                    group["step"],
                    group[metric],
                    marker="o",
                    lw=2,
                    color=color_by_prior[p_m],
                    label=f"P(M)={p_m:g}",
                )
            ax.set_title(title)
            ax.set_xlabel("MD fine-tune step")
            ax.set_ylim(0.0, 1.02)
        axes[0].set_ylabel("persona-side mass")
        axes[2].legend(frameon=True, fontsize=7)
        fig.suptitle(f"Headline prior sweep: {prompt} prompt mass accounting", y=0.99)
        fig.tight_layout()
        fig.savefig(OUT / f"prior_sweep_{prompt.lower()}_mass.png")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(13.0, 4.0), dpi=180)
    axes[0].plot(endpoint["p_misaligned_prior"], endpoint["O_p_next_S_M_mean"], marker="o", label=r"$P(S_M|O)$")
    axes[0].plot(endpoint["p_misaligned_prior"], endpoint["O_p_next_S_A_mean"], marker="o", label=r"$P(S_A|O)$")
    axes[0].set_ylabel("step-100 O-prompt mass")
    axes[0].legend(frameon=True, fontsize=8)
    axes[1].plot(endpoint["p_misaligned_prior"], endpoint["O_next_log_odds_mean"], marker="o", label="O broad")
    axes[1].plot(endpoint["p_misaligned_prior"], endpoint["D_next_log_odds_mean"], marker="o", label="D narrow")
    axes[1].set_ylabel("step-100 log odds")
    axes[1].legend(frameon=True, fontsize=8)
    axes[2].plot(endpoint["p_misaligned_prior"], endpoint["gap_D_minus_O_mean"], marker="o", color="#d62728")
    axes[2].axhline(0.0, color="black", lw=0.8)
    axes[2].set_ylabel("D log odds - O log odds")
    for ax in axes:
        ax.set_xlabel("base pretraining prior P(M persona)")
    fig.suptitle("Headline prior sweep: endpoint sensitivity", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "prior_sweep_endpoint.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_steps = int(os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_BASE_STEPS", "1200"))
    seeds = [int(s) for s in os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_SEEDS", "0,1,2").split(",")]
    priors = [
        float(x)
        for x in os.environ.get("SPECIAL_SFP_HEADLINE_PRIOR_P_M", "0.01,0.05,0.10,0.50,0.90").split(",")
    ]
    rows = []
    t0 = time.time()
    for p_m in priors:
        for seed in seeds:
            print(f"headline prior P(M)={p_m:g} seed {seed}", flush=True)
            rows.append(run_prior(p_m, seed, device, base_steps, STEPS))
            pd.DataFrame(rows).to_csv(OUT / "prior_sweep_live.csv", index=False)
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "prior_sweep_runs.csv", index=False)
    mass = summarize_mass(df)
    endpoint = summarize_endpoint(df)
    coherence = summarize_coherence(df)
    mass.to_csv(OUT / "prior_sweep_polarization_curves.csv", index=False)
    endpoint.to_csv(OUT / "prior_sweep_endpoint_summary.csv", index=False)
    coherence.to_csv(OUT / "prior_sweep_coherence_curves.csv", index=False)
    plot(mass, endpoint, coherence)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "base_steps": base_steps,
                "ft_steps": FT_STEPS,
                "checkpoints": list(STEPS),
                "seeds": seeds,
                "p_misaligned_priors": priors,
                "coherence_n": COHERENCE_N,
                "coherence_gen_len": COHERENCE_GEN_LEN,
                "i_dom_special": I_DOM_SPECIAL,
                "config": asdict(headline_cfg((0.25, 0.25, 0.25, 0.25))),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(endpoint.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
