"""Exact-inference ideal-learner baselines for the headline low-prior SFP run.

Question: what should a "perfect estimator" do on O prompts after MD-only
fine-tuning?  Two idealizations bracket the answer:

  - saturated: the learner keeps the true within-sector HMMs and updates a
    free 4-vector sector prior by exact Bayes on N unambiguous MD sequences
    (Dirichlet counting).  The MD evidence is absorbed entirely by the MD
    weight.  Because the O prompt contains S_O emissions, which have zero
    likelihood under D sectors when alpha_wrong_special = 0, conditioning on
    the prompt removes the MD weight entirely and the persona ratio reverts
    to the pretraining MO:AO ratio at every dose.  Zero broad transfer.
  - product: the learner is constrained to a persona x domain product prior
    mu_P (x) mu_T.  MD sequences update both marginals, so the persona shift
    survives O-prompt conditioning and broad transfer approaches 1.  (The
    domain marginal cancels from every O-prompt readout because the S_O
    evidence is hard, so a persona-only-update learner behaves identically.)
  - tilted: persona and domain marginals update exactly like the product
    learner, but the sector odds ratio is frozen at its pretraining value
    OR(pi0).  For a product pi0 this coincides with the product learner; for a
    non-product pi0 it is the correct "shared marginal update on top of the
    learned correlation" bracket, starting exactly at pi0 at dose 0.

The MD fine-tuning corpus has identical likelihood under all learners, so
where the trained transformer lands between them is pure inductive bias.

Dose is parametrized as t = N_ft / kappa, fine-tuning sequences per unit of
pretraining pseudocount: saturated pi(t) = (pi0 + t e_MD) / (1 + t), product
m(t) = (m0 + t) / (1 + t) and d(t) = (d0 + t) / (1 + t).

The base prior is overridable via SPECIAL_SFP_BAYES_NULL_PI0 (comma-separated
MD,MO,AD,AO) for non-product-prior control runs; the transformer overlay CSV
via SPECIAL_SFP_BAYES_NULL_RUNS_CSV (empty string disables the overlay).

Config and readout conventions match the transformer headline run in
results_low_prior_confirm_p005_bigmodel_base20k_lr2e3: same process
parameters, same domain_prompt (1 neutral + 3 domain specials + 1 neutral),
same continuation sector-rate labeling, gen_len 32.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp


OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_BAYES_NULL_OUT",
        ROOT / "experiment_folders" / "em_afp_simpler_codex_auto" / "results_bayes_null",
    )
)
_RUNS_CSV_ENV = os.environ.get("SPECIAL_SFP_BAYES_NULL_RUNS_CSV")
if _RUNS_CSV_ENV is None:
    TRANSFORMER_RUNS_CSV = (
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / "results_low_prior_confirm_p005_bigmodel_base20k_lr2e3"
        / "headline_ft_lr_sweep_runs.csv"
    )
elif _RUNS_CSV_ENV == "":
    TRANSFORMER_RUNS_CSV = None
else:
    TRANSFORMER_RUNS_CSV = Path(_RUNS_CSV_ENV)
DOSES = tuple(
    float(x.strip())
    for x in os.environ.get(
        "SPECIAL_SFP_BAYES_NULL_DOSES", "0,0.1,0.3,1,3,10,30,100,1000"
    ).split(",")
    if x.strip()
)
SECTOR_RATE_N = int(os.environ.get("SPECIAL_SFP_BAYES_NULL_SECTOR_RATE_N", "4096"))
SECTOR_RATE_GEN_LEN = int(os.environ.get("SPECIAL_SFP_BAYES_NULL_SECTOR_RATE_GEN_LEN", "32"))
SEED = int(os.environ.get("SPECIAL_SFP_BAYES_NULL_SEED", "0"))
TRANSFORMER_STEP = int(os.environ.get("SPECIAL_SFP_BAYES_NULL_TRANSFORMER_STEP", "9"))

PI0 = tuple(
    float(x.strip())
    for x in os.environ.get("SPECIAL_SFP_BAYES_NULL_PI0", "0.025,0.025,0.475,0.475").split(",")
)
P_M0 = PI0[0] + PI0[1]
P_D0 = PI0[0] + PI0[2]
OR0 = (PI0[0] * PI0[3]) / (PI0[1] * PI0[2])


def headline_lowprior_cfg(pi: tuple[float, float, float, float]) -> special_sfp.SpecialSFPConfig:
    """Process parameters from results_low_prior_confirm_p005_bigmodel_base20k_lr2e3."""
    return special_sfp.SpecialSFPConfig(
        p=0.5,
        epsilon=0.02,
        p_s=0.8,
        alpha_wrong_special=0.0,
        epsilon_persona=0.30,
        epsilon_persona_m=0.30,
        epsilon_persona_a=0.0,
        epsilon_domain=0.04,
        p_s_persona=0.90,
        p_s_domain=0.70,
        seq_len=64,
        pi=pi,
    )


def domain_prompt(domain: str) -> list[int]:
    special = 2 if domain == "D" else 3
    return [special_sfp.token_id(0, 0)] + [special_sfp.token_id(0, special)] * 3 + [
        special_sfp.token_id(0, 0)
    ]


def saturated_pi(dose: float) -> tuple[float, float, float, float]:
    pi = (np.asarray(PI0, dtype=np.float64) + dose * np.array([1.0, 0.0, 0.0, 0.0])) / (1.0 + dose)
    return tuple(pi)


def product_pi(dose: float) -> tuple[float, float, float, float]:
    m = (P_M0 + dose) / (1.0 + dose)
    d = (P_D0 + dose) / (1.0 + dose)
    return (m * d, m * (1.0 - d), (1.0 - m) * d, (1.0 - m) * (1.0 - d))


def pi_from_margins_and_or(m: float, d: float, odds_ratio: float) -> tuple[float, float, float, float]:
    """Unique 2x2 sector table with margins (m, d) and fixed odds ratio."""
    if abs(odds_ratio - 1.0) < 1e-12:
        x = m * d
    else:
        a = odds_ratio - 1.0
        b = -(odds_ratio * (m + d) + (1.0 - m - d))
        c = odds_ratio * m * d
        root = np.sqrt(max(b * b - 4.0 * a * c, 0.0))
        lo, hi = max(0.0, m + d - 1.0), min(m, d)
        for cand in ((-b - root) / (2.0 * a), (-b + root) / (2.0 * a)):
            if lo - 1e-12 <= cand <= hi + 1e-12:
                x = min(max(cand, lo), hi)
                break
        else:
            raise ValueError(f"no admissible fixed-OR solution for m={m}, d={d}, OR={odds_ratio}")
    return (x, m - x, d - x, 1.0 - m - d + x)


def tilted_pi(dose: float) -> tuple[float, float, float, float]:
    m = (P_M0 + dose) / (1.0 + dose)
    d = (P_D0 + dose) / (1.0 + dose)
    return pi_from_margins_and_or(m, d, OR0)


LEARNER_PI = {"saturated": saturated_pi, "product": product_pi, "tilted": tilted_pi}


def belief_after_prompt(cfg: special_sfp.SpecialSFPConfig, tokens: list[int]) -> np.ndarray:
    ops = special_sfp.token_operators(cfg)
    belief = special_sfp.initial_belief(cfg)
    for tok in tokens:
        belief = belief @ ops[int(tok)]
        z = belief.sum()
        if z <= 0:
            raise ValueError("prompt has zero likelihood under this prior")
        belief = belief / z
    return belief


def sample_continuations(
    cfg: special_sfp.SpecialSFPConfig,
    start_belief: np.ndarray,
    n: int,
    gen_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Exact posterior-predictive sampling from the belief state."""
    ops = special_sfp.token_operators(cfg)
    belief = np.repeat(start_belief[None, :], n, axis=0)
    toks = np.empty((n, gen_len), dtype=np.int64)
    for t in range(gen_len):
        probs = np.einsum("bs,vst->bv", belief, ops)
        probs = probs / probs.sum(axis=1, keepdims=True)
        cdf = np.cumsum(probs, axis=1)
        cdf[:, -1] = 1.0
        tok = (rng.random(n)[:, None] > cdf).sum(axis=1)
        toks[:, t] = tok
        belief = np.einsum("bs,bst->bt", belief, ops[tok])
        belief = belief / belief.sum(axis=1, keepdims=True)
    return toks


def sector_rates(cont: np.ndarray, prompt_domain: str) -> dict[str, float]:
    """Same labeling as special_sfp_headline_lr_sweep.continuation_sector_rates."""
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


def next_token_readout(cfg: special_sfp.SpecialSFPConfig, belief: np.ndarray, prompt_domain: str) -> dict[str, float]:
    """Same fields as the barcharts o_prompt_readout, computed exactly."""
    ops = special_sfp.token_operators(cfg)
    probs = np.einsum("s,vst->v", belief, ops)
    probs = probs / probs.sum()
    p_sm = float(probs[[special_sfp.token_id(2, d) for d in range(4)]].sum())
    p_sa = float(probs[[special_sfp.token_id(3, d) for d in range(4)]].sum())
    p_sd = float(probs[[special_sfp.token_id(p, 2) for p in range(4)]].sum())
    p_so = float(probs[[special_sfp.token_id(p, 3) for p in range(4)]].sum())
    raw = {
        "MD": float(probs[special_sfp.token_id(2, 2)]),
        "MO": float(probs[special_sfp.token_id(2, 3)]),
        "AD": float(probs[special_sfp.token_id(3, 2)]),
        "AO": float(probs[special_sfp.token_id(3, 3)]),
    }
    joint_den = max(sum(raw.values()), 1e-12)
    out = {
        f"{prompt_domain}_p_next_S_M": p_sm,
        f"{prompt_domain}_p_next_S_A": p_sa,
        f"{prompt_domain}_p_next_S_D": p_sd,
        f"{prompt_domain}_p_next_S_O": p_so,
        f"{prompt_domain}_persona_M": p_sm / max(p_sm + p_sa, 1e-12),
        f"{prompt_domain}_domain_D": p_sd / max(p_sd + p_so, 1e-12),
        f"{prompt_domain}_joint_special_total": sum(raw.values()),
    }
    for key, val in raw.items():
        out[f"{prompt_domain}_joint_{key}"] = val / joint_den
    return out


def leaf_posterior(cfg: special_sfp.SpecialSFPConfig, belief: np.ndarray) -> np.ndarray:
    return belief.reshape(cfg.n_leaves, cfg.states_per_leaf).sum(axis=1)


def run_learner(learner: str, dose: float, rng: np.random.Generator) -> dict:
    pi = LEARNER_PI[learner](dose)
    cfg = headline_lowprior_cfg(pi)
    row: dict = {
        "learner": learner,
        "dose": dose,
        "pi_MD": pi[0],
        "pi_MO": pi[1],
        "pi_AD": pi[2],
        "pi_AO": pi[3],
    }
    for prompt_domain in ("O", "D"):
        belief = belief_after_prompt(cfg, domain_prompt(prompt_domain))
        mu = leaf_posterior(cfg, belief)
        m_mass = mu[0] + mu[1]
        row[f"{prompt_domain}_post_mu_MD"] = float(mu[0])
        row[f"{prompt_domain}_post_mu_MO"] = float(mu[1])
        row[f"{prompt_domain}_post_mu_AD"] = float(mu[2])
        row[f"{prompt_domain}_post_mu_AO"] = float(mu[3])
        row[f"{prompt_domain}_post_P_M"] = float(m_mass)
        row.update(next_token_readout(cfg, belief, prompt_domain))
        cont = sample_continuations(cfg, belief, SECTOR_RATE_N, SECTOR_RATE_GEN_LEN, rng)
        row.update(sector_rates(cont, prompt_domain))
    return row


def transformer_reference() -> dict[str, dict[str, float]]:
    if TRANSFORMER_RUNS_CSV is None or not TRANSFORMER_RUNS_CSV.exists():
        return {}
    df = pd.read_csv(TRANSFORMER_RUNS_CSV)
    if "step" not in df.columns:
        return {}
    out = {}
    for step in (0, TRANSFORMER_STEP):
        sub = df[df["step"] == step]
        if sub.empty:
            continue
        out[f"step{step}"] = {
            col: float(sub[col].mean())
            for col in (
                "O_rollout_MO",
                "O_rollout_MD",
                "O_rollout_no_sector",
                "D_rollout_MD",
                "O_p_next_S_M",
                "O_p_next_S_A",
            )
            if col in sub.columns
        }
        out[f"step{step}_std"] = {
            col: float(sub[col].std())
            for col in ("O_rollout_MO", "O_rollout_MD", "D_rollout_MD")
            if col in sub.columns
        }
    return out


def plot(df: pd.DataFrame, ref: dict[str, dict[str, float]]) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    colors = {"saturated": "#245C99", "product": "#C23B22", "tilted": "#7A4F9E"}
    labels = {
        "saturated": "saturated Bayes (free 4-sector prior)",
        "product": "product prior (persona x domain)",
        "tilted": "tilted (marginals updated, odds ratio frozen)",
    }
    panels = [
        ("O_rollout_MO", "O prompt → MO rollout rate (broad transfer)"),
        ("O_rollout_MD", "O prompt → MD rollout rate (spillback)"),
        ("D_rollout_MD", "D prompt → MD rollout rate (narrow learning)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.4), dpi=180)
    for ax, (col, title) in zip(axes, panels):
        for learner in ("saturated", "product", "tilted"):
            sub = df[df["learner"] == learner].sort_values("dose")
            ax.plot(sub["dose"], sub[col], marker="o", lw=2, color=colors[learner], label=labels[learner])
        step_key = f"step{TRANSFORMER_STEP}"
        if step_key in ref and col in ref[step_key]:
            mean = ref[step_key][col]
            std = ref.get(f"{step_key}_std", {}).get(col, 0.0)
            ax.axhline(mean, color="#3A7D44", lw=1.6, ls="--", label=f"transformer step {TRANSFORMER_STEP}")
            ax.axhspan(mean - std, mean + std, color="#3A7D44", alpha=0.12, linewidth=0)
        if "step0" in ref and col in ref["step0"]:
            ax.axhline(ref["step0"][col], color="black", lw=1.2, ls=":", label="transformer step 0")
        ax.set_xscale("symlog", linthresh=0.1)
        ax.set_xlim(left=0.0)
        ax.set_xlabel(r"fine-tuning dose $t = N_{\mathrm{ft}}/\kappa$")
        ax.set_ylabel("rollout fraction")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
    axes[0].legend(frameon=True, fontsize=7, loc="center left")
    fig.suptitle(
        "Ideal-learner baselines after MD fine-tuning: saturated Bayes reverts on O prompts, product prior transfers",
        y=0.99,
    )
    fig.tight_layout()
    fig.savefig(OUT / "bayes_null_sector_rates.png")
    plt.close(fig)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rows = [run_learner(learner, dose, rng) for learner in ("saturated", "product", "tilted") for dose in DOSES]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "bayes_null_runs.csv", index=False)
    ref = transformer_reference()
    plot(df, ref)
    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "doses": list(DOSES),
                "sector_rate_n": SECTOR_RATE_N,
                "sector_rate_gen_len": SECTOR_RATE_GEN_LEN,
                "seed": SEED,
                "transformer_step": TRANSFORMER_STEP,
                "transformer_reference": ref,
                "pi0": PI0,
                "or0": OR0,
                "config": asdict(headline_lowprior_cfg(PI0)),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    show = df[
        [
            "learner",
            "dose",
            "pi_MD",
            "O_post_P_M",
            "O_rollout_MO",
            "O_rollout_MD",
            "O_rollout_no_sector",
            "D_rollout_MD",
            "O_p_next_S_M",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    if ref:
        print("\ntransformer reference (mean over seeds):")
        for key, vals in ref.items():
            if key.endswith("_std"):
                continue
            print(f"  {key}: " + ", ".join(f"{k}={v:.4f}" for k, v in vals.items()))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
