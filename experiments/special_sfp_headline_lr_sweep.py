"""Fine-tuning learning-rate sweep for the headline special-state SFP setup."""

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
from experiments.special_sfp_auto_validate import batch, make_model, prompt_metrics
from experiments.special_sfp_headline_prior_barcharts import domain_prompt, o_prompt_readout
from experiments.special_sfp_headline_prior_sweep import (
    I_DOM_SPECIAL,
    headline_cfg,
    incoherence_o,
    persona_prior_to_pi,
)


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_HEADLINE_LR_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / "results_headline_lr_sweep",
    )
)
STEPS = tuple(
    int(x.strip())
    for x in os.environ.get("SPECIAL_SFP_HEADLINE_LR_FT_CHECKPOINTS", "1,5,10,20,50,100").split(",")
    if x.strip()
)
FT_STEPS = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_FT_STEPS", str(max(STEPS))))
P_MISALIGNED = float(os.environ.get("SPECIAL_SFP_HEADLINE_LR_P_M", "0.01"))
FT_LRS = tuple(
    float(x.strip())
    for x in os.environ.get("SPECIAL_SFP_HEADLINE_LR_FT_LRS", "3e-4,1e-4,3e-5,1e-5").split(",")
    if x.strip()
)
JSD_N = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_JSD_N", "64"))
JSD_GEN_LEN = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_JSD_GEN_LEN", "32"))
SECTOR_RATE_N = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_SECTOR_RATE_N", "512"))
SECTOR_RATE_GEN_LEN = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_SECTOR_RATE_GEN_LEN", "32"))
SKIP_JSD = os.environ.get("SPECIAL_SFP_HEADLINE_LR_SKIP_JSD", "0") == "1"


def _env_float(name: str, default: float | None) -> float | None:
    return default if name not in os.environ else float(os.environ[name])


def _lr_label(lr: float) -> str:
    return f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")


def train_base(seed: int, device: str, base_steps: int):
    cfg = run_cfg()
    model = make_model(cfg, seed, device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for step in range(1, base_steps + 1):
        _, x, y = batch(cfg, 256, 100_000 + seed * 10_000 + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return cfg, model.state_dict()


def run_cfg() -> special_sfp.SpecialSFPConfig:
    cfg = headline_cfg(persona_prior_to_pi(P_MISALIGNED))
    return replace(
        cfg,
        alpha_wrong_special=_env_float("SPECIAL_SFP_HEADLINE_LR_ALPHA_WRONG", cfg.alpha_wrong_special),
        p_persona=_env_float("SPECIAL_SFP_HEADLINE_LR_P_PERSONA", cfg.p_persona),
        p_domain=_env_float("SPECIAL_SFP_HEADLINE_LR_P_DOMAIN", cfg.p_domain),
        p_s_persona=_env_float("SPECIAL_SFP_HEADLINE_LR_P_S_PERSONA", cfg.p_s_persona),
        p_s_domain=_env_float("SPECIAL_SFP_HEADLINE_LR_P_S_DOMAIN", cfg.p_s_domain),
        epsilon_persona=_env_float("SPECIAL_SFP_HEADLINE_LR_EPSILON_PERSONA", cfg.epsilon_persona),
        epsilon_persona_m=_env_float("SPECIAL_SFP_HEADLINE_LR_EPSILON_PERSONA_M", cfg.epsilon_persona_m),
        epsilon_persona_a=_env_float("SPECIAL_SFP_HEADLINE_LR_EPSILON_PERSONA_A", cfg.epsilon_persona_a),
        epsilon_domain=_env_float("SPECIAL_SFP_HEADLINE_LR_EPSILON_DOMAIN", cfg.epsilon_domain),
    )


def _sector_prompt_belief(cfg, leaf: str, prompt_domain: str) -> np.ndarray:
    ops = special_sfp.token_operators(cfg)
    belief = np.zeros(cfg.n_states, dtype=np.float64)
    leaf_idx = special_sfp.LEAF_TO_INDEX[leaf]
    belief[leaf_idx * cfg.states_per_leaf] = 1.0
    for tok in domain_prompt(prompt_domain):
        post = belief @ ops[int(tok)]
        z = post.sum()
        if z <= 0:
            raise ValueError(f"{prompt_domain} prompt has zero likelihood under sector {leaf}")
        belief = post / z
    return belief


def _hmm_logprob_continuations(
    cfg,
    leaf: str,
    cont: np.ndarray,
    prompt_domain: str,
) -> np.ndarray:
    ops = special_sfp.token_operators(cfg)
    start = _sector_prompt_belief(cfg, leaf, prompt_domain)
    out = np.empty(cont.shape[0], dtype=np.float64)
    for i, seq in enumerate(cont):
        belief = start.copy()
        logp = 0.0
        for tok in seq:
            post = belief @ ops[int(tok)]
            z = post.sum()
            if z <= 0:
                logp = -np.inf
                break
            logp += np.log(z)
            belief = post / z
        out[i] = logp
    return out


def _sample_hmm_continuations(
    cfg,
    leaf: str,
    prompt_domain: str,
    n: int,
    gen_len: int,
    seed: int,
) -> np.ndarray:
    gen_cfg = replace(cfg, seq_len=gen_len)
    init = _sector_prompt_belief(cfg, leaf, prompt_domain)
    obs, _ = special_sfp.gen_special_sfp(n, gen_cfg, np.random.default_rng(seed), init=init)
    return obs


@torch.no_grad()
def _sample_model_continuations(
    model: torch.nn.Module,
    prompt_domain: str,
    device: str,
    n: int,
    gen_len: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    ctx = np.repeat(np.asarray(domain_prompt(prompt_domain), dtype=np.int64)[None, :], n, axis=0)
    seq = torch.tensor(ctx, dtype=torch.long, device=device)
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
        torch.manual_seed(seed)
    logps = []
    model.eval()
    for _ in range(gen_len):
        logits = model(seq[:, -model.cfg.n_ctx :])[:, -1, :]
        logp = torch.log_softmax(logits, dim=-1)
        probs = torch.softmax(logits, dim=-1)
        if generator is None:
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = torch.multinomial(probs, 1, generator=generator)
        logps.append(logp.gather(1, nxt).squeeze(1))
        seq = torch.cat([seq, nxt], dim=1)
    cont = seq[:, ctx.shape[1] :].cpu().numpy()
    return cont, torch.stack(logps, dim=1).sum(dim=1).cpu().numpy()


@torch.no_grad()
def _model_logprob_continuations(
    model: torch.nn.Module,
    cont: np.ndarray,
    prompt_domain: str,
    device: str,
) -> np.ndarray:
    ctx = np.repeat(np.asarray(domain_prompt(prompt_domain), dtype=np.int64)[None, :], cont.shape[0], axis=0)
    prefix_len = ctx.shape[1]
    if prefix_len + cont.shape[1] - 1 > model.cfg.n_ctx:
        raise ValueError("JSD continuation length exceeds model context window")
    x_np = np.concatenate([ctx, cont[:, :-1]], axis=1)
    x = torch.tensor(x_np, dtype=torch.long, device=device)
    y = torch.tensor(cont, dtype=torch.long, device=device)
    logits = model(x)
    logp = torch.log_softmax(logits[:, prefix_len - 1 : prefix_len - 1 + cont.shape[1], :], dim=-1)
    return logp.gather(2, y[:, :, None]).squeeze(2).sum(dim=1).cpu().numpy()


@torch.no_grad()
def continuation_jsd(
    model: torch.nn.Module,
    cfg,
    leaf: str,
    prompt_domain: str,
    device: str,
    seed: int,
    n: int = JSD_N,
    gen_len: int = JSD_GEN_LEN,
) -> float:
    model_cont, model_logp = _sample_model_continuations(model, prompt_domain, device, n, gen_len, seed)
    model_logq = _hmm_logprob_continuations(cfg, leaf, model_cont, prompt_domain)

    hmm_cont = _sample_hmm_continuations(cfg, leaf, prompt_domain, n, gen_len, seed + 100_000)
    hmm_logq = _hmm_logprob_continuations(cfg, leaf, hmm_cont, prompt_domain)
    hmm_logp = _model_logprob_continuations(model, hmm_cont, prompt_domain, device)

    log2 = np.log(2.0)
    mix_model = np.logaddexp(model_logp, model_logq) - log2
    mix_hmm = np.logaddexp(hmm_logp, hmm_logq) - log2
    jsd = 0.5 * np.mean(model_logp - mix_model) + 0.5 * np.mean(hmm_logq - mix_hmm)
    return float(jsd)


@torch.no_grad()
def continuation_sector_rates(
    model: torch.nn.Module,
    prompt_domain: str,
    device: str,
    seed: int,
    n: int = SECTOR_RATE_N,
    gen_len: int = SECTOR_RATE_GEN_LEN,
) -> dict[str, float]:
    cont, _ = _sample_model_continuations(model, prompt_domain, device, n, gen_len, seed)
    persona, domain = special_sfp.split_token(cont)
    has_m = (persona == 2).any(axis=1)
    has_a = (persona == 3).any(axis=1)
    has_d = (domain == 2).any(axis=1)
    has_o = (domain == 3).any(axis=1)

    persona_count = has_m.astype(np.int8) + has_a.astype(np.int8)
    domain_count = has_d.astype(np.int8) + has_o.astype(np.int8)
    incoherent = (persona_count > 1) | (domain_count > 1)
    md = (~incoherent) & has_m & has_d
    mo = (~incoherent) & has_m & has_o
    ad = (~incoherent) & has_a & has_d
    ao = (~incoherent) & has_a & has_o
    no_sector = ~(md | mo | ad | ao | incoherent)

    prefix = f"{prompt_domain}_rollout"
    return {
        f"{prefix}_MD": float(md.mean()),
        f"{prefix}_MO": float(mo.mean()),
        f"{prefix}_AD": float(ad.mean()),
        f"{prefix}_AO": float(ao.mean()),
        f"{prefix}_no_sector": float(no_sector.mean()),
        f"{prefix}_incoherent": float(incoherent.mean()),
    }


@torch.no_grad()
def readout(model: torch.nn.Module, cfg, device: str, seed: int, step: int) -> dict[str, float]:
    metrics = {}
    for key, value in prompt_metrics(model, device).items():
        metrics[key] = value
    for key, value in o_prompt_readout(model, device).items():
        metrics[key] = value
    metrics["O_incoh"] = incoherence_o(model, cfg, device, 900_000 + seed * 10_000 + step)
    metrics["O_incoh_frac_idom"] = metrics["O_incoh"] / I_DOM_SPECIAL
    metrics["O_coherence_retained"] = 1.0 - metrics["O_incoh_frac_idom"]
    metrics.update(continuation_sector_rates(model, "D", device, 1_300_000 + seed * 10_000 + step))
    metrics.update(continuation_sector_rates(model, "O", device, 1_400_000 + seed * 10_000 + step))
    if SKIP_JSD:
        metrics["D_jsd_MD"] = np.nan
        metrics["D_jsd_AD"] = np.nan
        metrics["D_jsd_A_minus_M"] = np.nan
        metrics["O_jsd_MO"] = np.nan
        metrics["O_jsd_AO"] = np.nan
        metrics["O_jsd_A_minus_M"] = np.nan
    else:
        metrics["D_jsd_MD"] = continuation_jsd(model, cfg, "MD", "D", device, 1_000_000 + seed * 10_000 + step)
        metrics["D_jsd_AD"] = continuation_jsd(model, cfg, "AD", "D", device, 1_050_000 + seed * 10_000 + step)
        metrics["D_jsd_A_minus_M"] = metrics["D_jsd_AD"] - metrics["D_jsd_MD"]
        metrics["O_jsd_MO"] = continuation_jsd(model, cfg, "MO", "O", device, 1_100_000 + seed * 10_000 + step)
        metrics["O_jsd_AO"] = continuation_jsd(model, cfg, "AO", "O", device, 1_200_000 + seed * 10_000 + step)
        metrics["O_jsd_A_minus_M"] = metrics["O_jsd_AO"] - metrics["O_jsd_MO"]
    return metrics


def run_seed_lr(seed: int, ft_lr: float, cfg, base_state, device: str) -> list[dict]:
    model = make_model(cfg, seed, device)
    model.load_state_dict(base_state)
    checkpoint_set = set(STEPS)
    rows = [
        {
            "p_misaligned_prior": P_MISALIGNED,
            "seed": seed,
            "ft_lr": ft_lr,
            "step": 0,
            **asdict(cfg),
            **readout(model, cfg, device, seed, 0),
        }
    ]
    ft_cfg = replace(cfg, pi=(1.0, 0.0, 0.0, 0.0))
    opt = torch.optim.AdamW(model.parameters(), lr=ft_lr)
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
                    "p_misaligned_prior": P_MISALIGNED,
                    "seed": seed,
                    "ft_lr": ft_lr,
                    "step": step,
                    **asdict(cfg),
                    **readout(model, cfg, device, seed, step),
                }
            )
    return rows


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [
        c
        for c in df.columns
        if c.startswith("O_")
        or c.startswith("D_")
        or c in {"gap_D_minus_O", "O_incoh", "O_incoh_frac_idom", "O_coherence_retained"}
    ]
    agg = (
        df.groupby(["p_misaligned_prior", "ft_lr", "step"], as_index=False)[value_cols]
        .agg(["mean", "std"])
        .reset_index()
    )
    agg.columns = [
        "_".join(str(part) for part in col if part != "").rstrip("_")
        if isinstance(col, tuple)
        else str(col)
        for col in agg.columns
    ]
    return agg.sort_values(["ft_lr", "step"], ascending=[False, True])


def endpoint_summary(agg: pd.DataFrame) -> pd.DataFrame:
    endpoint = agg[agg["step"] == FT_STEPS].copy()
    cols = [
        "ft_lr",
        "O_persona_M_mean",
        "O_domain_O_mean",
        "O_coherence_retained_mean",
        "O_cont_MO_mean",
        "O_cont_MD_mean",
        "O_cont_AO_mean",
        "O_cont_AD_mean",
        "O_joint_special_total_mean",
        "D_jsd_MD_mean",
        "D_jsd_AD_mean",
        "D_jsd_A_minus_M_mean",
        "O_jsd_MO_mean",
        "O_jsd_AO_mean",
        "O_jsd_A_minus_M_mean",
        "D_next_log_odds_mean",
        "O_next_log_odds_mean",
    ]
    return endpoint[cols].sort_values("ft_lr", ascending=False)


def plot(agg: pd.DataFrame) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    lrs = sorted(agg["ft_lr"].unique(), reverse=True)
    colors = plt.cm.viridis(np.linspace(0.12, 0.88, len(lrs)))
    color_by_lr = dict(zip(lrs, colors))

    fig, axes = plt.subplots(2, 3, figsize=(14.5, 7.0), dpi=180, sharex=True)
    panels = [
        ("O_persona_M_mean", "O_persona_M_std", r"$P(M\mid O)$", "O-prompt persona"),
        ("O_domain_O_mean", "O_domain_O_std", r"$P(O\mathrm{-domain}\mid O)$", "O-prompt domain"),
        (
            "O_coherence_retained_mean",
            "O_coherence_retained_std",
            r"$1-\mathrm{Incoh}_O/I_{\mathrm{dom}}$",
            "coherence retained",
        ),
        ("O_cont_MO_mean", "O_cont_MO_std", r"$MO$ share", "2x2 continuation"),
        ("O_cont_MD_mean", "O_cont_MD_std", r"$MD$ share", "2x2 continuation"),
        ("O_joint_special_total_mean", "O_joint_special_total_std", "raw joint-special mass", "special mass"),
    ]
    for ax, (mean_col, std_col, ylabel, title) in zip(axes.ravel(), panels):
        for lr in lrs:
            group = agg[agg["ft_lr"] == lr]
            x = group["step"].to_numpy()
            y = group[mean_col].to_numpy()
            yerr = group[std_col].fillna(0.0).to_numpy()
            ax.plot(x, y, marker="o", lw=1.8, color=color_by_lr[lr], label=f"lr={_lr_label(lr)}")
            ax.fill_between(x, y - yerr, y + yerr, color=color_by_lr[lr], alpha=0.12, linewidth=0)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("MD fine-tune step")
        if mean_col != "O_joint_special_total_mean":
            ax.set_ylim(-0.02, 1.05)
    axes[0, 2].axhline(0.0, color="black", lw=0.8, ls=":")
    axes[0, 0].legend(frameon=True, fontsize=8)
    fig.suptitle(f"Headline FT learning-rate sensitivity, base P(M)={P_MISALIGNED:g}", y=0.99)
    fig.tight_layout()
    fig.savefig(OUT / "headline_ft_lr_sweep_curves.png", bbox_inches="tight")
    plt.close(fig)

    final = endpoint_summary(agg)
    labels = [_lr_label(lr) for lr in final["ft_lr"]]
    x = np.arange(len(final))
    fig, ax = plt.subplots(figsize=(9.0, 4.2), dpi=180)
    width = 0.22
    ax.bar(x - width, final["O_persona_M_mean"], width=width, label=r"$P(M\mid O)$")
    ax.bar(x, final["O_domain_O_mean"], width=width, label=r"$P(O\mathrm{-domain}\mid O)$")
    ax.bar(x + width, final["O_coherence_retained_mean"], width=width, label="coherence retained")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlabel("FT learning rate")
    ax.set_ylim(0.0, 1.05)
    ax.set_title(f"Step-{FT_STEPS} O-prompt endpoint by FT learning rate")
    ax.legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "headline_ft_lr_endpoint_bars.png", bbox_inches="tight")
    plt.close(fig)

    if {"O_jsd_MO_mean", "O_jsd_AO_mean", "O_jsd_A_minus_M_mean"}.issubset(set(agg.columns)):
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.0), dpi=180, sharex=True)
        jsd_panels = [
            (
                "O_jsd_MO_mean",
                "O_jsd_MO_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_O),Q_{MO}(\cdot\mid c_O))$",
                "O prompt vs MO",
            ),
            (
                "O_jsd_AO_mean",
                "O_jsd_AO_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_O),Q_{AO}(\cdot\mid c_O))$",
                "O prompt vs AO",
            ),
            (
                "O_jsd_A_minus_M_mean",
                "O_jsd_A_minus_M_std",
                r"$\mathrm{JSD}_{AO\mid O}-\mathrm{JSD}_{MO\mid O}$",
                "JSD preference",
            ),
        ]
        for ax, (mean_col, std_col, ylabel, title) in zip(axes, jsd_panels):
            for lr in lrs:
                group = agg[agg["ft_lr"] == lr]
                x = group["step"].to_numpy()
                y = group[mean_col].to_numpy()
                yerr = group[std_col].fillna(0.0).to_numpy()
                ax.plot(x, y, marker="o", lw=1.8, color=color_by_lr[lr], label=f"lr={_lr_label(lr)}")
                ax.fill_between(x, y - yerr, y + yerr, color=color_by_lr[lr], alpha=0.12, linewidth=0)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("MD fine-tune step")
        axes[0].axhline(np.log(2.0), color="black", lw=0.8, ls=":", label=r"$\log 2$")
        axes[1].axhline(np.log(2.0), color="black", lw=0.8, ls=":")
        axes[2].axhline(0.0, color="black", lw=0.8, ls=":")
        axes[0].legend(frameon=True, fontsize=8)
        fig.suptitle(f"Sequence-level continuation JSD, base P(M)={P_MISALIGNED:g}", y=0.99)
        fig.tight_layout()
        fig.savefig(OUT / "headline_ft_lr_jsd_curves.png", bbox_inches="tight")
        plt.close(fig)

    sector_rate_cols = {
        f"{prompt}_rollout_{bucket}_mean"
        for prompt in ("D", "O")
        for bucket in ("MD", "MO", "AD", "AO", "no_sector", "incoherent")
    }

    def plot_sector_rate_bars(ax, group: pd.DataFrame, prompt: str) -> None:
        buckets = ["MD", "MO", "AD", "AO", "no_sector", "incoherent"]
        labels = ["MD", "MO", "AD", "AO", "no sector", "incoherent"]
        colors = ["#6a3d9a", "#1f78b4", "#b15928", "#33a02c", "#bdbdbd", "#e31a1c"]
        x = np.arange(len(group))
        bottom = np.zeros(len(group), dtype=np.float64)
        for bucket, label, color in zip(buckets, labels, colors):
            values = group[f"{prompt}_rollout_{bucket}_mean"].fillna(0.0).to_numpy()
            ax.bar(x, values, bottom=bottom, color=color, label=label, width=0.72)
            bottom = bottom + values
        ax.set_ylim(0, 1)
        ax.set_xticks(x, [str(int(s)) for s in group["step"]])
        ax.set_xlabel("MD fine-tune step")
        ax.set_ylabel("share of sampled continuations")
        ax.set_title(f"{prompt} prompt: rollout sector labels, lr=3e-4")

    if sector_rate_cols.issubset(set(agg.columns)) and not agg[np.isclose(agg["ft_lr"], 3e-4)].empty:
        rate_group = agg[np.isclose(agg["ft_lr"], 3e-4)].sort_values("step")
        fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=180, gridspec_kw={"width_ratios": [1, 1, 0.36]})
        plot_sector_rate_bars(axes[0], rate_group, "D")
        plot_sector_rate_bars(axes[1], rate_group, "O")
        axes[2].axis("off")
        handles, labels = axes[1].get_legend_handles_labels()
        axes[2].legend(handles, labels, loc="center", frameon=True, title="rollout label")
        fig.suptitle(f"Continuation rollout sector labels, base P(M)={P_MISALIGNED:g}", y=0.99)
        fig.tight_layout()
        fig.savefig(OUT / "headline_ft_lr_rollout_sector_rates.png", bbox_inches="tight")
        plt.close(fig)

    four_jsd_cols = {
        "D_jsd_MD_mean",
        "D_jsd_AD_mean",
        "D_jsd_A_minus_M_mean",
        "O_jsd_MO_mean",
        "O_jsd_AO_mean",
        "O_jsd_A_minus_M_mean",
    }
    if four_jsd_cols.issubset(set(agg.columns)):
        has_sector_rates = sector_rate_cols.issubset(set(agg.columns)) and not agg[
            np.isclose(agg["ft_lr"], 3e-4)
        ].empty
        n_rows = 3 if has_sector_rates else 2
        fig, axes = plt.subplots(n_rows, 3, figsize=(14.0, 10.2 if has_sector_rates else 7.2), dpi=180, sharex=False)
        jsd_panels = [
            (
                "D_jsd_MD_mean",
                "D_jsd_MD_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_D),Q_{MD}(\cdot\mid c_D))$",
                "D prompt vs MD",
            ),
            (
                "D_jsd_AD_mean",
                "D_jsd_AD_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_D),Q_{AD}(\cdot\mid c_D))$",
                "D prompt vs AD",
            ),
            (
                "D_jsd_A_minus_M_mean",
                "D_jsd_A_minus_M_std",
                r"$\mathrm{JSD}_{AD\mid D}-\mathrm{JSD}_{MD\mid D}$",
                "D-prompt preference",
            ),
            (
                "O_jsd_MO_mean",
                "O_jsd_MO_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_O),Q_{MO}(\cdot\mid c_O))$",
                "O prompt vs MO",
            ),
            (
                "O_jsd_AO_mean",
                "O_jsd_AO_std",
                r"$\mathrm{JSD}(P_\theta(\cdot\mid c_O),Q_{AO}(\cdot\mid c_O))$",
                "O prompt vs AO",
            ),
            (
                "O_jsd_A_minus_M_mean",
                "O_jsd_A_minus_M_std",
                r"$\mathrm{JSD}_{AO\mid O}-\mathrm{JSD}_{MO\mid O}$",
                "O-prompt preference",
            ),
        ]
        for ax, (mean_col, std_col, ylabel, title) in zip(axes.ravel(), jsd_panels):
            for lr in lrs:
                group = agg[agg["ft_lr"] == lr]
                x = group["step"].to_numpy()
                y = group[mean_col].to_numpy()
                yerr = group[std_col].fillna(0.0).to_numpy()
                ax.plot(x, y, marker="o", lw=1.8, color=color_by_lr[lr], label=f"lr={_lr_label(lr)}")
                ax.fill_between(x, y - yerr, y + yerr, color=color_by_lr[lr], alpha=0.12, linewidth=0)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            ax.set_xlabel("MD fine-tune step")
        for ax in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
            ax.axhline(np.log(2.0), color="black", lw=0.8, ls=":", label=r"$\log 2$")
        axes[0, 2].axhline(0.0, color="black", lw=0.8, ls=":")
        axes[1, 2].axhline(0.0, color="black", lw=0.8, ls=":")
        axes[0, 0].legend(frameon=True, fontsize=8)
        if has_sector_rates:
            rate_group = agg[np.isclose(agg["ft_lr"], 3e-4)].sort_values("step")
            plot_sector_rate_bars(axes[2, 0], rate_group, "D")
            plot_sector_rate_bars(axes[2, 1], rate_group, "O")
            axes[2, 2].axis("off")
            handles, labels = axes[2, 1].get_legend_handles_labels()
            axes[2, 2].legend(handles, labels, loc="center", frameon=True, title="rollout label")
        fig.suptitle(f"Four prompt/sector continuation JSDs, base P(M)={P_MISALIGNED:g}", y=0.99)
        fig.tight_layout()
        fig.savefig(OUT / "headline_ft_lr_jsd_four_sector_curves.png", bbox_inches="tight")
        plt.close(fig)


def best_steps_by_lr(agg: pd.DataFrame) -> dict[float, int]:
    scored = agg.copy()
    scored["score"] = scored["O_cont_MO_mean"] * scored["O_coherence_retained_mean"]
    return {
        float(lr): int(group.loc[group["score"].idxmax(), "step"])
        for lr, group in scored.groupby("ft_lr")
    }


def _headline_for_lr(
    agg: pd.DataFrame,
    ft_lr: float,
    best_step: int,
    filename: str,
    title_prefix: str,
) -> None:
    group = agg[agg["ft_lr"] == ft_lr].copy().sort_values("step")
    x = np.arange(len(group))
    labels = [str(int(s)) for s in group["step"]]
    best_idx = int(np.where(group["step"].to_numpy() == best_step)[0][0])

    fig = plt.figure(figsize=(14.0, 5.8), dpi=180)
    grid = fig.add_gridspec(2, 3, height_ratios=[1.0, 0.82], hspace=0.48, wspace=0.28)
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    axes[0].bar(x, group["O_persona_M_mean"], color="#2f6fbb", label="P(M | O)")
    axes[0].bar(
        x,
        group["O_persona_A_mean"],
        bottom=group["O_persona_M_mean"],
        color="#d9822b",
        label="P(A | O)",
    )
    axes[0].set_title("Persona readout")

    axes[1].bar(x, group["O_domain_O_mean"], color="#2a9d8f", label="P(O-domain | O)")
    axes[1].bar(
        x,
        group["O_domain_D_mean"],
        bottom=group["O_domain_O_mean"],
        color="#b6465f",
        label="P(D-domain | O)",
    )
    axes[1].set_title("Domain readout")

    axes[2].bar(
        x,
        group["O_coherence_retained_mean"],
        color="#7b61a8",
        label=r"$1-\mathrm{Incoh}_O/I_{\mathrm{dom}}$",
    )
    axes[2].plot(
        x,
        group["O_domain_O_mean"],
        color="#2a9d8f",
        marker="o",
        lw=1.4,
        ls="--",
        label="P(O-domain | O)",
    )
    axes[2].set_title("Coherence retained")

    for ax in axes:
        ax.axvline(best_idx, color="#111111", lw=1.0, ls=":", alpha=0.75)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("MD fine-tune step")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.legend(frameon=True, fontsize=8, loc="upper right")
    axes[0].set_ylabel("normalized mass")
    axes[1].set_ylabel("normalized mass")
    axes[2].set_ylabel("fraction retained")

    heat_grid = grid[1, :].subgridspec(1, len(group), wspace=0.18)
    vmax = max(0.50, float(group[["O_cont_MD_mean", "O_cont_MO_mean", "O_cont_AD_mean", "O_cont_AO_mean"]].max().max()))
    im = None
    for i, (_, row) in enumerate(group.iterrows()):
        heat_ax = fig.add_subplot(heat_grid[0, i])
        mat = np.array(
            [
                [row["O_cont_MD_mean"], row["O_cont_MO_mean"]],
                [row["O_cont_AD_mean"], row["O_cont_AO_mean"]],
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
        if int(row["step"]) == best_step:
            for spine in heat_ax.spines.values():
                spine.set_edgecolor("#111111")
                spine.set_linewidth(2.0)
    if im is not None:
        cax = fig.add_axes([0.92, 0.11, 0.010, 0.23])
        fig.colorbar(im, cax=cax, label="conditional 2x2 share")

    fig.suptitle(
        f"{title_prefix} O-prompt decomposition, FT lr={_lr_label(ft_lr)}, best checkpoint step {best_step}",
        y=0.98,
    )
    fig.tight_layout()
    fig.savefig(OUT / filename, bbox_inches="tight")
    plt.close(fig)


def plot_lr_headlines(agg: pd.DataFrame) -> None:
    best_steps = best_steps_by_lr(agg)
    lrs = sorted(agg["ft_lr"].unique(), reverse=True)
    scored = agg.copy()
    scored["score"] = scored["O_cont_MO_mean"] * scored["O_coherence_retained_mean"]
    best_row = scored.loc[scored["score"].idxmax()]
    best_lr = float(best_row["ft_lr"])

    for lr in lrs:
        filename = f"o_prompt_ft_lr_{_lr_label(float(lr)).replace('-', 'm')}_headline_3panel.png"
        _headline_for_lr(
            agg,
            float(lr),
            best_steps[float(lr)],
            filename,
            "FT LR sensitivity",
        )
    _headline_for_lr(
        agg,
        best_lr,
        best_steps[best_lr],
        "o_prompt_best_lr_headline_3panel.png",
        "Selected headline",
    )


def merge_sector_rate_csv(agg: pd.DataFrame, sector_csv: str) -> pd.DataFrame:
    sector = pd.read_csv(sector_csv)
    sector_cols = [
        c
        for c in sector.columns
        if c.startswith("D_rollout_") or c.startswith("O_rollout_")
    ]
    if not sector_cols:
        return agg
    drop_cols = [c for c in sector_cols if c in agg.columns]
    merged = agg.drop(columns=drop_cols, errors="ignore").merge(
        sector[["ft_lr", "step", *sector_cols]],
        on=["ft_lr", "step"],
        how="left",
    )
    return merged


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("SPECIAL_SFP_HEADLINE_LR_PLOT_ONLY", "0") == "1":
        agg = pd.read_csv(OUT / "headline_ft_lr_sweep_curves.csv")
        sector_csv = os.environ.get("SPECIAL_SFP_HEADLINE_LR_SECTOR_RATE_CSV")
        if sector_csv:
            agg = merge_sector_rate_csv(agg, sector_csv)
            agg.to_csv(OUT / "headline_ft_lr_sweep_curves_with_sector_rates.csv", index=False)
        plot(agg)
        plot_lr_headlines(agg)
        print(f"wrote {OUT}")
        return 0

    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    base_steps = int(os.environ.get("SPECIAL_SFP_HEADLINE_LR_BASE_STEPS", "1200"))
    seeds = [int(s) for s in os.environ.get("SPECIAL_SFP_HEADLINE_LR_SEEDS", "0,1,2").split(",")]

    rows = []
    t0 = time.time()
    for seed in seeds:
        print(f"training base seed {seed}", flush=True)
        cfg, base_state = train_base(seed, device, base_steps)
        for ft_lr in FT_LRS:
            print(f"FT lr={ft_lr:g} seed {seed}", flush=True)
            rows.extend(run_seed_lr(seed, ft_lr, cfg, base_state, device))
            pd.DataFrame(rows).to_csv(OUT / "headline_ft_lr_sweep_live.csv", index=False)
    df = pd.DataFrame(rows)
    agg = aggregate(df)
    final = endpoint_summary(agg)
    df.to_csv(OUT / "headline_ft_lr_sweep_runs.csv", index=False)
    agg.to_csv(OUT / "headline_ft_lr_sweep_curves.csv", index=False)
    final.to_csv(OUT / "headline_ft_lr_endpoint_summary.csv", index=False)
    plot(agg)
    plot_lr_headlines(agg)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "base_steps": base_steps,
                "ft_steps": FT_STEPS,
                "checkpoints": list(STEPS),
                "seeds": seeds,
                "init_std": float(os.environ.get("SPECIAL_SFP_INIT_STD", "0.02")),
                "act_fn": os.environ.get("SPECIAL_SFP_ACT_FN", "gelu"),
                "d_model": int(os.environ.get("SPECIAL_SFP_D_MODEL", "64")),
                "n_heads": int(os.environ.get("SPECIAL_SFP_N_HEADS", "2")),
                "n_layers": int(os.environ.get("SPECIAL_SFP_N_LAYERS", "2")),
                "d_mlp": int(
                    os.environ.get(
                        "SPECIAL_SFP_D_MLP",
                        str(4 * int(os.environ.get("SPECIAL_SFP_D_MODEL", "64"))),
                    )
                ),
                "p_misaligned_prior": P_MISALIGNED,
                "ft_lrs": list(FT_LRS),
                "i_dom_special": I_DOM_SPECIAL,
                "config": asdict(run_cfg()),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(final.to_string(index=False))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
