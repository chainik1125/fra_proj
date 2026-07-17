"""Moore-Penrose belief probes: is the sector code factored, and what moves under MD FT?

Shared-representation test for the headline low-prior SFP run. The behavioral result
(results_bayes_null) shows the fine-tuned transformer acts partly like the product-prior
ideal learner. This experiment measures the representational counterpart.

Analytic fact used in the design: when the base prior is a PRODUCT prior (the default,
0.025, 0.025, 0.475, 0.475), each sector's likelihood factorizes over persona-side and
domain-side symbols, so the exact Bayes posterior over sectors is a product of persona and
domain marginals for EVERY sequence (checked numerically below). A factored code (persona
belief + domain belief, no sector interaction) is then Bayes-sufficient at pretraining.
For NON-PRODUCT priors (SPECIAL_SFP_PROBE_PI0 override; correlated-prior control runs)
this property does not hold: the posterior carries a genuine persona x domain interaction,
and a nonzero interaction gap on-distribution is the expected positive control, not an
error. The `det` probe target (mu_MD*mu_AO - mu_MO*mu_AD) measures whether the network
represents that interaction; it is identically zero (NaN R^2) under a product prior.

Measurements, per layer and fine-tuning checkpoint:
  1. Held-out R^2 of linear (lstsq / Moore-Penrose) probes from resid_post to the exact
     Bayes sector masses (MD, MO, AD, AO) and derived coordinates P_M, P_D, chi.
  2. Shared-direction geometry: cosine between the persona-contrast probe direction
     measured within D (w_MD - w_AD) and within O (w_MO - w_AO); same for domain
     contrasts. High cosine = one shared persona direction reused across domains.
  3. Probe-direction rotation vs the base model, per sector.
  4. Activation drift: resid(ckpt) - resid(base) on a fixed base-process batch,
     decomposed into signed motion along the base persona direction, base domain
     direction, the rest of the base probe span, and the orthogonal complement.
  5. Base-probe decoding of the O-prompt and D-prompt final-position activations across
     checkpoints ("what does the old code say the new activations mean"), alongside the
     behavioral next-token readout.

Model and training conventions match results_low_prior_confirm_p005_bigmodel_base20k_lr2e3:
3 layers, d_model 128, 4 heads, d_mlp 512, init_std 0.01, 20k pretraining steps at lr 1e-3,
MD-only fine-tuning at lr 2e-3, same batch seed streams as the lr sweep.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bag_moments import coherence, special_sfp
from bag_moments.model import GPTConfig, TinyGPT
from bag_moments.train import get_device


SMOKE = os.environ.get("SPECIAL_SFP_PROBE_SMOKE", "0") == "1"
OUT = Path(
    os.environ.get(
        "SPECIAL_SFP_PROBE_OUT",
        ROOT
        / "experiment_folders"
        / "em_afp_simpler_codex_auto"
        / ("results_probe_factorization_smoke" if SMOKE else "results_probe_factorization"),
    )
)
BASE_STEPS = int(os.environ.get("SPECIAL_SFP_PROBE_BASE_STEPS", "300" if SMOKE else "20000"))
FT_LR = float(os.environ.get("SPECIAL_SFP_PROBE_FT_LR", "2e-3"))
FT_STEPS = int(os.environ.get("SPECIAL_SFP_PROBE_FT_STEPS", "18"))
CHECKPOINTS = tuple(
    int(x.strip())
    for x in os.environ.get(
        "SPECIAL_SFP_PROBE_CHECKPOINTS", "1,5,9,18" if SMOKE else "1,2,3,5,7,9,11,14,18"
    ).split(",")
    if x.strip()
)
PROBE_N = int(os.environ.get("SPECIAL_SFP_PROBE_N", "256" if SMOKE else "2048"))
DRIFT_N = int(os.environ.get("SPECIAL_SFP_PROBE_DRIFT_N", "128" if SMOKE else "512"))
ROLLOUT_N = int(os.environ.get("SPECIAL_SFP_PROBE_ROLLOUT_N", "128" if SMOKE else "1024"))
ROLLOUT_GEN_LEN = int(os.environ.get("SPECIAL_SFP_PROBE_ROLLOUT_GEN_LEN", "32"))
# Head-only fine-tuning: freeze everything except the unembedding. Features are then
# frozen, so resid streams (and all probe/drift readouts) are identical to base at every
# checkpoint; only the behavioral readouts change.
HEAD_ONLY = os.environ.get("SPECIAL_SFP_PROBE_HEAD_ONLY", "0") == "1"
# Prompt-set evaluation (the principled EM-rate readout): rejection-sample K
# persona-neutral, domain-identifying prefixes from the base process and average rollout
# rates over them, alongside the fixed-prompt readout. K=0 disables (old behavior).
# The set is generated once with PROMPT_SET_SEED, independent of the model seed, so all
# seeds/conditions/checkpoints are evaluated on identical prompts.
PROMPT_K = int(os.environ.get("SPECIAL_SFP_PROBE_PROMPT_K", "0"))
PROMPT_LEN = int(os.environ.get("SPECIAL_SFP_PROBE_PROMPT_LEN", "8"))
PROMPT_NPER = int(os.environ.get("SPECIAL_SFP_PROBE_PROMPT_NPER", "16"))
PROMPT_SET_SEED = 424242
SKIP_POS = 3
SEED = int(os.environ.get("SPECIAL_SFP_PROBE_SEED", "0"))

# Base sector prior (MD, MO, AD, AO). Overridable to run non-product (correlated)
# priors, for which the product-posterior property no longer holds.
PI0 = tuple(
    float(x.strip())
    for x in os.environ.get("SPECIAL_SFP_PROBE_PI0", "0.025,0.025,0.475,0.475").split(",")
)
LEAVES = special_sfp.LEAVES


def headline_lowprior_cfg(pi: tuple[float, float, float, float]) -> special_sfp.SpecialSFPConfig:
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


def make_headline_model(cfg: special_sfp.SpecialSFPConfig, seed: int, device: str) -> TinyGPT:
    torch.manual_seed(seed)
    model = TinyGPT(
        GPTConfig(
            vocab_size=cfg.vocab_size,
            n_ctx=cfg.seq_len,
            d_model=128,
            n_heads=4,
            n_layers=3,
            d_mlp=512,
            init_std=0.01,
            act_fn="gelu",
        )
    )
    model.to(device)
    return model


def batch(cfg: special_sfp.SpecialSFPConfig, batch_size: int, seed: int, device: str):
    rng = np.random.default_rng(seed)
    obs, _ = special_sfp.gen_special_sfp(batch_size, cfg, rng)
    return obs, torch.tensor(obs[:, :-1], device=device), torch.tensor(obs[:, 1:], device=device)


def train_steps(model: TinyGPT, cfg, lr: float, n_steps: int, seed_base: int, device: str, on_checkpoint=None):
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    for step in range(1, n_steps + 1):
        _, x, y = batch(cfg, 256, seed_base + step, device)
        logits = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
        if on_checkpoint is not None:
            on_checkpoint(step)


@torch.no_grad()
def collect_resids(model: TinyGPT, x: torch.Tensor) -> list[np.ndarray]:
    """resid_post per layer, (B, T, d) as float64 numpy."""
    _, resids = model.run_with_resid(x)
    return [r.cpu().numpy().astype(np.float64) for r in resids]


def probe_fit(X: np.ndarray, Y: np.ndarray, rng: np.random.Generator, alpha: float = 0.0) -> dict:
    """lstsq/ridge probe with train/test split; returns R^2 per target and raw-space directions."""
    idx = rng.permutation(Y.shape[0])
    tr = idx[: int(0.8 * len(idx))]
    te = idx[int(0.8 * len(idx)) :]
    mean = X[tr].mean(axis=0, keepdims=True)
    std = X[tr].std(axis=0, keepdims=True) + 1e-6
    Xtr = np.concatenate([(X[tr] - mean) / std, np.ones((len(tr), 1))], axis=1)
    Xte = np.concatenate([(X[te] - mean) / std, np.ones((len(te), 1))], axis=1)
    if alpha > 0.0:
        penalty = np.sqrt(alpha) * np.eye(Xtr.shape[1])
        penalty[-1, -1] = 0.0  # do not penalize the intercept
        Xfit = np.concatenate([Xtr, penalty], axis=0)
        Yfit = np.concatenate([Y[tr], np.zeros((Xtr.shape[1], Y.shape[1]))], axis=0)
    else:
        Xfit, Yfit = Xtr, Y[tr]
    coef = np.linalg.lstsq(Xfit, Yfit, rcond=None)[0]
    pred = Xte @ coef
    ss_res = ((Y[te] - pred) ** 2).sum(axis=0)
    ss_tot = ((Y[te] - Y[te].mean(axis=0, keepdims=True)) ** 2).sum(axis=0) + 1e-12
    r2 = 1.0 - ss_res / ss_tot
    test_var = ss_tot / max(len(te), 1)
    r2 = np.where(test_var < 1e-9, np.nan, r2)  # degenerate targets (e.g. det under product prior)
    dirs = coef[:-1] / std.T  # raw-residual-space directions, (d, n_targets)
    return {"r2": r2, "coef": coef, "mean": mean, "std": std, "dirs": dirs}


def derived_targets(mu: np.ndarray) -> np.ndarray:
    """Columns: MD, MO, AD, AO, P_M, P_D, chi, det.

    chi mixes marginal and interaction information (under a product prior it equals
    (2P_M-1)(2P_D-1) exactly); det = mu_MD*mu_AO - mu_MO*mu_AD isolates the posterior
    interaction and is identically zero under a product prior (R^2 reported as NaN).
    """
    p_m = mu[:, 0] + mu[:, 1]
    p_d = mu[:, 0] + mu[:, 2]
    chi = mu[:, 0] - mu[:, 1] - mu[:, 2] + mu[:, 3]
    det = mu[:, 0] * mu[:, 3] - mu[:, 1] * mu[:, 2]
    return np.column_stack([mu, p_m, p_d, chi, det])


TARGET_NAMES = ["MD", "MO", "AD", "AO", "P_M", "P_D", "chi", "det"]


def cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def transfer_scores(probe: dict, X: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """(R^2, Pearson r) of a probe evaluated on another condition's samples."""
    xs = np.concatenate([(X - probe["mean"]) / probe["std"], np.ones((len(X), 1))], axis=1)
    pred = (xs @ probe["coef"])[:, 0]
    ss_res = ((y - pred) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    r2 = float(1.0 - ss_res / ss_tot)
    denom = pred.std() * y.std()
    corr = float(np.mean((pred - pred.mean()) * (y - y.mean())) / denom) if denom > 1e-12 else 0.0
    return r2, corr


def conditional_probes(X: np.ndarray, mu_flat: np.ndarray, rng: np.random.Generator) -> dict[str, float]:
    """Persona probes fitted on domain-resolved contexts (and vice versa).

    On base data the sector masses are bilinear in (P_M, P_D), so unconditional
    contrast directions mix in product features.  Conditioning on a resolved
    domain isolates the persona code within each domain; if the persona
    representation is shared, the two directions align and probes transfer.
    """
    p_m = mu_flat[:, 0] + mu_flat[:, 1]
    p_d = mu_flat[:, 0] + mu_flat[:, 2]
    specs = {
        "PM_given_D": (p_d >= 0.9, p_m),
        "PM_given_O": (p_d <= 0.1, p_m),
        "PD_given_M": (p_m >= 0.9, p_d),
        "PD_given_A": (p_m <= 0.1, p_d),
    }
    fits: dict[str, dict | None] = {}
    out: dict[str, float] = {}
    for name, (mask, target) in specs.items():
        out[f"n_{name}"] = int(mask.sum())
        out[f"npos_{name}"] = int((target[mask] > 0.5).sum())
        if mask.sum() < 200 or target[mask].std() < 1e-3:
            fits[name] = None
            continue
        fits[name] = probe_fit(X[mask], target[mask][:, None], rng, alpha=10.0)
        out[f"r2_{name}"] = float(fits[name]["r2"][0])
    if fits.get("PM_given_D") is not None and fits.get("PM_given_O") is not None:
        out["cos_wPM_D_vs_O"] = cos(fits["PM_given_D"]["dirs"][:, 0], fits["PM_given_O"]["dirs"][:, 0])
        mask_o, mask_d = specs["PM_given_O"][0], specs["PM_given_D"][0]
        out["r2_PM_D_to_O"], out["corr_PM_D_to_O"] = transfer_scores(fits["PM_given_D"], X[mask_o], p_m[mask_o])
        out["r2_PM_O_to_D"], out["corr_PM_O_to_D"] = transfer_scores(fits["PM_given_O"], X[mask_d], p_m[mask_d])
    if fits.get("PD_given_M") is not None and fits.get("PD_given_A") is not None:
        out["cos_wPD_M_vs_A"] = cos(fits["PD_given_M"]["dirs"][:, 0], fits["PD_given_A"]["dirs"][:, 0])
        mask_a, mask_m = specs["PD_given_A"][0], specs["PD_given_M"][0]
        out["r2_PD_M_to_A"], out["corr_PD_M_to_A"] = transfer_scores(fits["PD_given_M"], X[mask_a], p_d[mask_a])
        out["r2_PD_A_to_M"], out["corr_PD_A_to_M"] = transfer_scores(fits["PD_given_A"], X[mask_m], p_d[mask_m])
    return out


def geometry_metrics(dirs: np.ndarray) -> dict[str, float]:
    """dirs columns ordered as TARGET_NAMES."""
    w = {name: dirs[:, i] for i, name in enumerate(TARGET_NAMES)}
    return {
        "cos_persona_contrast_D_vs_O": cos(w["MD"] - w["AD"], w["MO"] - w["AO"]),
        "cos_domain_contrast_M_vs_A": cos(w["MD"] - w["MO"], w["AD"] - w["AO"]),
        "cos_PM_vs_PD": cos(w["P_M"], w["P_D"]),
    }


@torch.no_grad()
def _generate_with_logps(
    model: TinyGPT,
    ctx: np.ndarray,
    device: str,
    gen_len: int,
    seed: int,
    return_probs: bool = False,
):
    """Sample continuations for contexts (B, L) -> tokens (B, gen_len), model log-probs of
    each chosen token (B, gen_len); with return_probs, also the full next-token
    distribution at each step (B, gen_len, V) for the JSD coherence metrics."""
    seq = torch.tensor(ctx, dtype=torch.long, device=device)
    if str(device).startswith("cuda"):
        generator = torch.Generator(device=device)
        generator.manual_seed(seed)
    else:
        generator = None
        torch.manual_seed(seed)
    model.eval()
    logps = []
    all_probs = []
    for _ in range(gen_len):
        logits = model(seq[:, -model.cfg.n_ctx :])[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
        if return_probs:
            all_probs.append(probs)
        if generator is None:
            nxt = torch.multinomial(probs, 1)
        else:
            nxt = torch.multinomial(probs, 1, generator=generator)
        logps.append(torch.log(probs.gather(1, nxt) + 1e-12).squeeze(1))
        seq = torch.cat([seq, nxt], dim=1)
    tokens = seq[:, ctx.shape[1] :].cpu().numpy()
    logp = torch.stack(logps, dim=1).cpu().numpy().astype(np.float64)
    if return_probs:
        return tokens, logp, torch.stack(all_probs, dim=1).cpu().numpy().astype(np.float64)
    return tokens, logp


def _generate_continuations(
    model: TinyGPT,
    ctx: np.ndarray,
    device: str,
    gen_len: int,
    seed: int,
) -> np.ndarray:
    """Sample continuations for a batch of contexts (B, L) -> (B, gen_len)."""
    return _generate_with_logps(model, ctx, device, gen_len, seed)[0]


def sample_model_continuations(
    model: TinyGPT,
    prompt_domain: str,
    device: str,
    n: int,
    gen_len: int,
    seed: int,
) -> np.ndarray:
    """Sample continuations after the fixed domain prompt; port of the lr-sweep sampler."""
    ctx = np.repeat(np.asarray(domain_prompt(prompt_domain), dtype=np.int64)[None, :], n, axis=0)
    return _generate_continuations(model, ctx, device, gen_len, seed)


def sample_eval_prompts(
    cfg: special_sfp.SpecialSFPConfig,
    domain: str,
    k: int,
    length: int,
    seed: int,
) -> np.ndarray:
    """Rejection-sample persona-neutral, domain-identifying prefixes from the base process.

    Accept a length-`length` prefix iff its domain side contains the target special
    (hard domain evidence under alpha=0), none of the other domain's special, and its
    persona side contains no specials (the Betley-8 analogue: innocuous prompts).
    """
    rng = np.random.default_rng(seed)
    want, avoid = (3, 2) if domain == "O" else (2, 3)
    prompts: list[list[int]] = []
    while len(prompts) < k:
        obs, _ = special_sfp.gen_special_sfp(256, cfg, rng)
        pre = obs[:, :length]
        persona, dom = special_sfp.split_token(pre)
        ok = (
            (dom == want).any(axis=1)
            & ~(dom == avoid).any(axis=1)
            & ~(persona >= 2).any(axis=1)
        )
        prompts.extend(pre[ok].tolist())
    return np.asarray(prompts[:k], dtype=np.int64)


def prompt_set_rates(
    model: TinyGPT,
    prompts: np.ndarray,
    prompt_domain: str,
    device: str,
    n_per: int,
    gen_len: int,
    seed: int,
) -> dict[str, float]:
    """Rollout sector rates averaged over a prompt set, plus between-prompt spread."""
    k = prompts.shape[0]
    ctx = np.repeat(prompts, n_per, axis=0)
    cont = _generate_continuations(model, ctx, device, gen_len, seed)
    overall = sector_rates(cont, prompt_domain)
    out = {key.replace("_rollout_", "_set_rollout_"): val for key, val in overall.items()}
    per_prompt = cont.reshape(k, n_per, -1)
    mo_rates = [
        sector_rates(per_prompt[i], prompt_domain)[f"{prompt_domain}_rollout_MO"]
        for i in range(k)
    ]
    out[f"{prompt_domain}_set_rollout_MO_bp_std"] = float(np.std(mo_rates))
    return out


COH_ETA = float(os.environ.get("SPECIAL_SFP_PROBE_COH_ETA", "1e-3"))
# Which coherence deficits to compute: any comma-subset of xe,jsd,mixjsd (see
# bag_moments/coherence.py for definitions). All three by default — the extra cost is
# one stored per-position predictive array plus a small simplex-grid fit.
COH_METRICS = tuple(
    m.strip()
    for m in os.environ.get("SPECIAL_SFP_PROBE_COH_METRICS", "xe,jsd,mixjsd").split(",")
    if m.strip()
)


def sector_logliks(
    cfg: special_sfp.SpecialSFPConfig,
    ctx: np.ndarray,
    cont: np.ndarray,
    eta: float = COH_ETA,
) -> tuple[np.ndarray, np.ndarray]:
    """log q~_z(continuation | ctx, sector z) under the eta-thickened sector processes.

    q~_z mixes each step of the exact sector-z process with an eta-probability "glitch"
    channel (uniform token emission, hidden state propagating through the marginal
    transition). Thickening bounds the cost of a process-impossible token at
    ~ -log(eta/16) ~ 9.7 nats instead of the arbitrary epsilon floor, and lets the
    filter continue sensibly afterwards. Hard-evidence exclusion still operates: a
    sector zeroed by the prompt pays ~9.7 nats per contradicting token and is never the
    best explanation of a long continuation that respects the other sector.

    Returns (logliks (B, 4), impossible_rate (B,)) where impossible_rate is the fraction
    of continuation tokens with zero unthickened likelihood under ALL four sectors —
    the hard-violation count, reported separately from the smooth deficit.
    """
    ops = special_sfp.token_operators(cfg)
    t_marg = ops.sum(axis=0)
    obs = np.concatenate([ctx, cont], axis=1)
    n_ctx_tokens = ctx.shape[1]
    n_cont = obs.shape[1] - n_ctx_tokens
    logl = np.zeros((obs.shape[0], 4))
    raw_lik_cont = np.zeros((obs.shape[0], 4, n_cont))
    for z in range(4):
        belief = np.zeros((obs.shape[0], cfg.n_states))
        belief[:, z * cfg.states_per_leaf] = 1.0
        for t in range(obs.shape[1]):
            raw_post = np.einsum("bs,bst->bt", belief, ops[obs[:, t]])
            smooth_post = (1.0 - eta) * raw_post + (eta / 16.0) * np.einsum(
                "bs,st->bt", belief, t_marg
            )
            lik = smooth_post.sum(axis=1)
            if t >= n_ctx_tokens:
                logl[:, z] += np.log(np.maximum(lik, 1e-300))
                raw_lik_cont[:, z, t - n_ctx_tokens] = raw_post.sum(axis=1)
            belief = smooth_post / np.maximum(lik, 1e-300)[:, None]
    impossible_rate = (raw_lik_cont <= 1e-300).all(axis=1).mean(axis=1)
    return logl, impossible_rate


def coherence_metrics(
    cfg: special_sfp.SpecialSFPConfig,
    ctx: np.ndarray,
    cont: np.ndarray,
    logp_model: np.ndarray,
    prefix: str,
    k_prompts: int = 0,
    model_probs: np.ndarray | None = None,
    metrics: tuple[str, ...] = COH_METRICS,
) -> dict[str, float]:
    """Projection of sampled behavior onto the coherent family (per-continuation).

    Delegates to bag_moments.coherence.coherence_metrics_all. Metrics: 'xe' is the
    original deficit (1/T)[log p_model(x) - max_z log q~_z(x)] with argmax
    disposition; 'jsd' is the token-local corner-JSD deficit with argmin disposition;
    'mixjsd' fits mixture weights per continuation (ideal hedging learners score ~0;
    the fitted w-hat is a continuous disposition). 'jsd'/'mixjsd' need model_probs
    (B, T, V), the model's full next-token distributions recorded during sampling;
    without them only 'xe' is computed.
    """
    return coherence.coherence_metrics_all(
        special_sfp.token_operators(cfg),
        [z * cfg.states_per_leaf for z in range(4)],
        special_sfp.LEAVES,
        ctx,
        cont,
        logp_model,
        prefix,
        COH_ETA,
        model_probs=model_probs,
        metrics=metrics,
        k_prompts=k_prompts,
    )


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


@torch.no_grad()
def prompt_readouts(model: TinyGPT, device: str) -> dict[str, float]:
    out = {}
    for domain in ("D", "O"):
        x = torch.tensor([domain_prompt(domain)], dtype=torch.long, device=device)
        probs = torch.softmax(model(x)[:, -1, :], dim=-1).cpu().numpy()[0]
        p_sm = float(probs[[special_sfp.token_id(2, d) for d in range(4)]].sum())
        p_sa = float(probs[[special_sfp.token_id(3, d) for d in range(4)]].sum())
        out[f"{domain}_p_next_S_M"] = p_sm
        out[f"{domain}_p_next_S_A"] = p_sa
    return out


@torch.no_grad()
def prompt_resids(model: TinyGPT, device: str) -> dict[str, list[np.ndarray]]:
    """Final-position resid per layer for the D and O prompts."""
    out = {}
    for domain in ("D", "O"):
        x = torch.tensor([domain_prompt(domain)], dtype=torch.long, device=device)
        _, resids = model.run_with_resid(x)
        out[domain] = [r[0, -1, :].cpu().numpy().astype(np.float64) for r in resids]
    return out


def decode_with_probe(vec: np.ndarray, probe: dict) -> np.ndarray:
    xs = np.concatenate([(vec[None, :] - probe["mean"]) / probe["std"], np.ones((1, 1))], axis=1)
    return (xs @ probe["coef"])[0]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    device = get_device(os.environ.get("BAG_DEVICE"))
    torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))
    cfg = headline_lowprior_cfg(PI0)
    ft_cfg = special_sfp.SpecialSFPConfig(**{**asdict(cfg), "pi": (1.0, 0.0, 0.0, 0.0)})

    # Probe/eval data from the base process, with exact Bayes sector masses as targets.
    probe_obs, probe_x, probe_y = batch(cfg, PROBE_N, 900_000 + SEED, device)
    _, mu, _, _ = special_sfp.forward_filter(probe_obs[:, :-1], cfg)
    mu_flat = mu[:, SKIP_POS:, :].reshape(-1, 4)
    Y = derived_targets(mu_flat)
    # Posterior interaction gap: ~0 iff the prior is product (sanity check); for
    # non-product priors a gap of order |prior det| is the expected positive control.
    interaction_gap = float(np.abs(mu_flat[:, 0] * mu_flat[:, 3] - mu_flat[:, 1] * mu_flat[:, 2]).max())
    prior_det = float(PI0[0] * PI0[3] - PI0[1] * PI0[2])
    print(
        f"posterior interaction max gap: {interaction_gap:.3e} "
        f"(prior det {prior_det:+.3e}; expect ~0 iff product prior)",
        flush=True,
    )

    drift_x = probe_x[:DRIFT_N]

    eval_prompt_sets: dict[str, np.ndarray] = {}
    prompt_null_pm = None
    if PROMPT_K > 0:
        for dom in ("D", "O"):
            eval_prompt_sets[dom] = sample_eval_prompts(cfg, dom, PROMPT_K, PROMPT_LEN, PROMPT_SET_SEED)
            np.save(OUT / f"eval_prompts_{dom}.npy", eval_prompt_sets[dom])
        # Exact-filter persona posterior after each O prompt: the per-prompt ideal null.
        _, mu_prompts, _, _ = special_sfp.forward_filter(eval_prompt_sets["O"], cfg)
        pm = mu_prompts[:, -1, 0] + mu_prompts[:, -1, 1]
        prompt_null_pm = {
            "mean": float(pm.mean()),
            "median": float(np.median(pm)),
            "min": float(pm.min()),
            "max": float(pm.max()),
        }
        print(f"prompt-set exact null P(M|O prompt): {prompt_null_pm}", flush=True)

    print(f"pretraining {BASE_STEPS} steps on {device}", flush=True)
    model = make_headline_model(cfg, SEED, device)
    train_steps(model, cfg, 1e-3, BASE_STEPS, 100_000 + SEED * 10_000, device)
    with torch.no_grad():
        logits = model(probe_x)
        base_loss = float(
            torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), probe_y.reshape(-1)
            ).cpu()
        )
    print(f"base eval loss on probe data: {base_loss:.4f}", flush=True)

    n_layers = model.cfg.n_layers
    rng = np.random.default_rng(700 + SEED)

    def probe_checkpoint(step: int, net: TinyGPT) -> tuple[list[dict], list[dict], list[np.ndarray]]:
        resids = collect_resids(net, probe_x)
        probes = []
        cond = []
        for layer in range(n_layers):
            X = resids[layer][:, SKIP_POS:, :].reshape(-1, resids[layer].shape[-1])
            probes.append(probe_fit(X, Y, rng))
            cond.append(conditional_probes(X, mu_flat, rng))
        drift = collect_resids(net, drift_x)
        return probes, cond, drift

    print("probing base model", flush=True)
    base_probes, base_cond, base_drift = probe_checkpoint(0, model)
    base_prompt_vecs = prompt_resids(model, device)

    rows = []
    drift_rows = []
    decode_rows = []

    def record(step: int, net: TinyGPT, probes: list[dict], cond: list[dict], drift: list[np.ndarray]) -> None:
        behav = prompt_readouts(net, device)
        for dom, roll_seed_base in (("D", 1_300_000), ("O", 1_400_000)):
            ctx = np.repeat(np.asarray(domain_prompt(dom), dtype=np.int64)[None, :], ROLLOUT_N, axis=0)
            cont, logp, probs = _generate_with_logps(
                net, ctx, device, ROLLOUT_GEN_LEN, roll_seed_base + SEED * 10_000 + step,
                return_probs=True,
            )
            behav.update(sector_rates(cont, dom))
            behav.update(coherence_metrics(cfg, ctx, cont, logp, dom, model_probs=probs))
        if PROMPT_K > 0:
            for dom, roll_seed_base in (("D", 1_500_000), ("O", 1_600_000)):
                prompts = eval_prompt_sets[dom]
                ctx = np.repeat(prompts, PROMPT_NPER, axis=0)
                cont, logp, probs = _generate_with_logps(
                    net, ctx, device, ROLLOUT_GEN_LEN, roll_seed_base + SEED * 10_000 + step,
                    return_probs=True,
                )
                overall = sector_rates(cont, dom)
                behav.update(
                    {key.replace("_rollout_", "_set_rollout_"): val for key, val in overall.items()}
                )
                per_prompt = cont.reshape(prompts.shape[0], PROMPT_NPER, -1)
                mo = [
                    sector_rates(per_prompt[i], dom)[f"{dom}_rollout_MO"]
                    for i in range(prompts.shape[0])
                ]
                behav[f"{dom}_set_rollout_MO_bp_std"] = float(np.std(mo))
                behav.update(
                    coherence_metrics(
                        cfg, ctx, cont, logp, f"{dom}_set",
                        k_prompts=prompts.shape[0], model_probs=probs,
                    )
                )
        pvecs = prompt_resids(net, device)
        for layer in range(n_layers):
            p = probes[layer]
            row = {"step": step, "layer": layer, **behav}
            for i, name in enumerate(TARGET_NAMES):
                row[f"r2_{name}"] = float(p["r2"][i])
            row.update(geometry_metrics(p["dirs"]))
            row.update(cond[layer])
            for i, name in enumerate(TARGET_NAMES):
                row[f"rot_cos_{name}"] = cos(p["dirs"][:, i], base_probes[layer]["dirs"][:, i])
            rows.append(row)

            # Drift decomposition in the base probe frame.
            delta = (drift[layer] - base_drift[layer])[:, SKIP_POS:, :].reshape(-1, drift[layer].shape[-1])
            total = float((delta**2).sum()) + 1e-12
            bdirs = base_probes[layer]["dirs"]
            u_pm = bdirs[:, 4] / (np.linalg.norm(bdirs[:, 4]) + 1e-12)
            u_pd = bdirs[:, 5] / (np.linalg.norm(bdirs[:, 5]) + 1e-12)
            span_q, _ = np.linalg.qr(bdirs[:, :4])
            proj_pm = delta @ u_pm
            proj_pd = delta @ u_pd
            proj_span = delta @ span_q
            drift_rows.append(
                {
                    "step": step,
                    "layer": layer,
                    "delta_rms": float(np.sqrt((delta**2).mean())),
                    "frac_along_PM": float((proj_pm**2).sum() / total),
                    "frac_along_PD": float((proj_pd**2).sum() / total),
                    "frac_probe_span": float((proj_span**2).sum() / total),
                    "signed_mean_proj_PM": float(proj_pm.mean()),
                    "signed_mean_proj_PD": float(proj_pd.mean()),
                }
            )

            # Base-probe decoding of prompt activations (old code, new activations).
            for domain in ("D", "O"):
                dec = decode_with_probe(pvecs[domain][layer], base_probes[layer])
                dec_own = decode_with_probe(pvecs[domain][layer], probes[layer])
                decode_rows.append(
                    {
                        "step": step,
                        "layer": layer,
                        "prompt": domain,
                        **{f"base_dec_{n}": float(dec[i]) for i, n in enumerate(TARGET_NAMES)},
                        **{f"own_dec_{n}": float(dec_own[i]) for i, n in enumerate(TARGET_NAMES)},
                        **behav,
                    }
                )

    record(0, model, base_probes, base_cond, base_drift)

    ft = make_headline_model(cfg, SEED, device)
    ft.load_state_dict(model.state_dict())
    if HEAD_ONLY:
        for pname, p in ft.named_parameters():
            p.requires_grad_(pname.startswith("head"))
    checkpoint_set = set(CHECKPOINTS)

    def on_ckpt(step: int) -> None:
        if step in checkpoint_set:
            print(f"probing FT step {step}", flush=True)
            if HEAD_ONLY:
                # Features frozen -> resid streams identical to base; reuse base probes.
                probes, cond, drift = base_probes, base_cond, base_drift
            else:
                probes, cond, drift = probe_checkpoint(step, ft)
            record(step, ft, probes, cond, drift)

    train_steps(ft, ft_cfg, FT_LR, FT_STEPS, 500_000 + SEED * 10_000, device, on_checkpoint=on_ckpt)

    frozen_drift = float("nan")
    if HEAD_ONLY:
        with torch.no_grad():
            frozen_drift = max(
                (p_ft - p_base).abs().max().item()
                for (n, p_ft), (_, p_base) in zip(ft.named_parameters(), model.named_parameters())
                if not n.startswith("head")
            )
        print(f"max frozen-parameter drift (must be 0): {frozen_drift:.3e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "probe_metrics.csv", index=False)
    drift_df = pd.DataFrame(drift_rows)
    drift_df.to_csv(OUT / "drift_decomposition.csv", index=False)
    decode_df = pd.DataFrame(decode_rows)
    decode_df.to_csv(OUT / "prompt_decoding.csv", index=False)

    plot(df, drift_df, decode_df, n_layers)

    (OUT / "metadata.json").write_text(
        json.dumps(
            {
                "elapsed_s": time.time() - t0,
                "smoke": SMOKE,
                "base_steps": BASE_STEPS,
                "ft_lr": FT_LR,
                "ft_steps": FT_STEPS,
                "checkpoints": list(CHECKPOINTS),
                "probe_n": PROBE_N,
                "drift_n": DRIFT_N,
                "rollout_n": ROLLOUT_N,
                "rollout_gen_len": ROLLOUT_GEN_LEN,
                "prompt_k": PROMPT_K,
                "prompt_len": PROMPT_LEN,
                "prompt_nper": PROMPT_NPER,
                "prompt_set_seed": PROMPT_SET_SEED,
                "prompt_set_null_PM_O": prompt_null_pm,
                "seed": SEED,
                "device": str(device),
                "coh_eta": COH_ETA,
                "coh_metrics": list(COH_METRICS),
                "posterior_interaction_max_gap": interaction_gap,
                "prior_det": prior_det,
                "base_eval_loss": base_loss,
                "head_only": HEAD_ONLY,
                "frozen_param_drift": frozen_drift,
                "pi0": PI0,
                "config": asdict(cfg),
                "model": {"d_model": 128, "n_heads": 4, "n_layers": 3, "d_mlp": 512, "init_std": 0.01},
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    last = df[df["layer"] == n_layers - 1]
    cols = ["step"] + [f"r2_{n}" for n in TARGET_NAMES] + [
        "cos_wPM_D_vs_O",
        "corr_PM_D_to_O",
        "corr_PM_O_to_D",
        "cos_wPD_M_vs_A",
        "O_rollout_MO",
        "O_rollout_MD",
        "D_rollout_MD",
        "O_set_rollout_MO",
        "O_set_rollout_MD",
        "D_set_rollout_MD",
        "O_set_rollout_MO_bp_std",
        "O_set_coh_deficit",
        "O_set_dispo_MO",
        "O_set_dispo_MD",
        "D_set_coh_deficit",
    ]
    cols = [c for c in cols if c in last.columns]
    print(last[cols].to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    print(f"wrote {OUT}")
    return 0


def plot(df: pd.DataFrame, drift_df: pd.DataFrame, decode_df: pd.DataFrame, n_layers: int) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    layer_colors = plt.cm.viridis(np.linspace(0.15, 0.85, n_layers))

    # 1. Probe R^2 per sector and derived coordinate across FT steps.
    fig, axes = plt.subplots(1, len(TARGET_NAMES), figsize=(3.4 * len(TARGET_NAMES), 3.8), dpi=180, sharey=True)
    for ax, name in zip(axes, TARGET_NAMES):
        for layer in range(n_layers):
            sub = df[df["layer"] == layer].sort_values("step")
            ax.plot(sub["step"], sub[f"r2_{name}"], marker="o", lw=2, color=layer_colors[layer], label=f"layer {layer}")
        ax.set_title(name)
        ax.set_xlabel("FT step")
        ax.set_ylim(-0.05, 1.02)
    axes[0].set_ylabel("held-out probe $R^2$")
    axes[0].legend(frameon=True, fontsize=7)
    fig.suptitle("Belief-probe R$^2$ across MD fine-tuning", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "probe_r2_across_ft.png", bbox_inches="tight")
    plt.close(fig)

    # 2. Shared-direction cosines (conditional probes) and probe rotation.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=180)
    for layer in range(n_layers):
        sub = df[df["layer"] == layer].sort_values("step")
        if "cos_wPM_D_vs_O" in sub.columns:
            axes[0].plot(sub["step"], sub["cos_wPM_D_vs_O"], marker="o", lw=2, color=layer_colors[layer], label=f"layer {layer}")
        if "r2_PM_D_to_O" in sub.columns:
            axes[1].plot(sub["step"], sub["r2_PM_D_to_O"], marker="o", lw=2, color=layer_colors[layer])
            axes[1].plot(sub["step"], sub["r2_PM_O_to_D"], marker="s", lw=2, ls="--", color=layer_colors[layer])
    axes[0].set_title("persona probe direction: fitted in D vs fitted in O")
    axes[1].set_title("persona probe transfer $R^2$ (o: D→O, s: O→D)")
    for ax in axes[:2]:
        ax.set_xlabel("FT step")
        ax.set_ylim(-0.05, 1.02)
    axes[0].set_ylabel("cosine")
    axes[1].set_ylabel("$R^2$")
    axes[0].legend(frameon=True, fontsize=7)
    last_layer = n_layers - 1
    sub = df[df["layer"] == last_layer].sort_values("step")
    for i, name in enumerate(["MD", "MO", "AD", "AO"]):
        axes[2].plot(sub["step"], sub[f"rot_cos_{name}"], marker="o", lw=2, label=name)
    axes[2].set_title(f"probe rotation vs base (layer {last_layer})")
    axes[2].set_xlabel("FT step")
    axes[2].set_ylabel("cos(w_step, w_base)")
    axes[2].legend(frameon=True, fontsize=7)
    fig.suptitle("Shared-direction geometry across MD fine-tuning", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "shared_direction_geometry.png", bbox_inches="tight")
    plt.close(fig)

    # 3. Base-probe decoding of prompt activations vs behavioral readout.
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.2), dpi=180)
    for i, (col, title) in enumerate(
        [("base_dec_P_M", "decoded $P_M$ (base probes)"), ("base_dec_P_D", "decoded $P_D$ (base probes)")]
    ):
        for prompt, ls in (("O", "-"), ("D", "--")):
            sub = decode_df[(decode_df["layer"] == last_layer) & (decode_df["prompt"] == prompt)].sort_values("step")
            axes[i].plot(sub["step"], sub[col], marker="o", lw=2, ls=ls, label=f"{prompt} prompt")
        axes[i].set_title(title + f" (layer {last_layer})")
        axes[i].set_xlabel("FT step")
        axes[i].legend(frameon=True, fontsize=8)
    sub = decode_df[(decode_df["layer"] == last_layer) & (decode_df["prompt"] == "O")].sort_values("step")
    axes[2].plot(sub["step"], sub["O_p_next_S_M"], marker="o", lw=2, color="#C23B22", label=r"$P(S_M|O)$ behavioral")
    axes[2].plot(sub["step"], sub["base_dec_P_M"], marker="o", lw=2, color="#245C99", label=r"decoded $P_M$ (O prompt)")
    if "O_rollout_MO" in sub.columns:
        axes[2].plot(sub["step"], sub["O_rollout_MO"], marker="s", lw=2, color="#3A7D44", label=r"O$\to$MO rollout rate")
    axes[2].set_title("representation vs behavior (O prompt)")
    axes[2].set_xlabel("FT step")
    axes[2].legend(frameon=True, fontsize=8)
    fig.suptitle("Old code, new activations: base-probe decoding across MD fine-tuning", y=1.0)
    fig.tight_layout()
    fig.savefig(OUT / "prompt_decoding_across_ft.png", bbox_inches="tight")
    plt.close(fig)

    # 4. Drift decomposition (last layer).
    sub = drift_df[(drift_df["layer"] == last_layer) & (drift_df["step"] > 0)].sort_values("step")
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=180)
    x = np.arange(len(sub))
    axes[0].bar(x, sub["frac_along_PM"], width=0.6, label="along base $P_M$ dir", color="#245C99")
    axes[0].bar(x, sub["frac_along_PD"], width=0.6, bottom=sub["frac_along_PM"], label="along base $P_D$ dir", color="#7BA7D7")
    axes[0].bar(
        x,
        (sub["frac_probe_span"] - sub["frac_along_PM"] - sub["frac_along_PD"]).clip(lower=0),
        width=0.6,
        bottom=sub["frac_along_PM"] + sub["frac_along_PD"],
        label="rest of probe span",
        color="#C9B458",
    )
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(sub["step"])
    axes[0].set_xlabel("FT step")
    axes[0].set_ylabel("fraction of drift variance")
    axes[0].set_title(f"activation drift decomposition (layer {last_layer})")
    axes[0].legend(frameon=True, fontsize=8)
    axes[1].plot(sub["step"], sub["signed_mean_proj_PM"], marker="o", lw=2, label="signed drift along $P_M$")
    axes[1].plot(sub["step"], sub["signed_mean_proj_PD"], marker="o", lw=2, label="signed drift along $P_D$")
    axes[1].axhline(0.0, color="black", lw=0.8)
    axes[1].set_xlabel("FT step")
    axes[1].set_title("mean signed drift on base-process data")
    axes[1].legend(frameon=True, fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "drift_decomposition.png", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    raise SystemExit(main())
