"""
Per-position oracle predictives and KL-to-oracle metrics.

The oracle is the exact Bayes-optimal next-token distribution.  For bit positions
we compare the model's renormalized P(next=1) against the closed-form oracle.
"""

from __future__ import annotations

import numpy as np
import torch

from . import data


def _lgamma(x):
    return torch.lgamma(x)


def _log_block_marginal(s, t):
    # m(s,t) = B(s+1, t-s+1)
    return _lgamma(s + 1.0) + _lgamma(t - s + 1.0) - _lgamma(t + 2.0)


def quenched2_predictive_torch(s1, t1, s2, t2, chi2):
    """Vectorized exact Bayes P(next bit of rollout 2 = 1). Args may be scalars or
    tensors; broadcast to a common float tensor."""
    s1, t1, s2, t2 = torch.broadcast_tensors(
        *[torch.as_tensor(v, dtype=torch.float32) for v in (s1, t1, s2, t2)]
    )
    log_same = _log_block_marginal(s1 + s2, t1 + t2)
    log_diff = _log_block_marginal(s1, t1) + _log_block_marginal(s2, t2)
    c = float(min(max(chi2, 1e-12), 1.0 - 1e-12))
    log_w_same = np.log(c) + log_same
    log_w_diff = np.log(1.0 - c) + log_diff
    m = torch.maximum(log_w_same, log_w_diff)
    w_same = torch.exp(log_w_same - m)
    w_diff = torch.exp(log_w_diff - m)
    q = w_same / (w_same + w_diff)  # collision posterior
    pred_same = (s1 + s2 + 1.0) / (t1 + t2 + 2.0)
    pred_diff = (s2 + 1.0) / (t2 + 2.0)
    return q * pred_same + (1.0 - q) * pred_diff, q


def oracle_seq_quenched2(qb: data.QuenchedBatch, chi2: float):
    """Return (oracle_p1, collision_post, mask) each shape (B, seqlen-1), aligned to
    INPUT positions i (model predicts token i+1).  mask True where token i+1 is a bit.

    Assumes J=2.  Rollout-1 bit predictions use Laplace on rollout-1 counts;
    rollout-2 bit predictions use the exact 2-hypothesis collision oracle.
    """
    assert len(qb.roll_lens) == 2
    t1, t2 = qb.roll_lens
    B, S = qb.tokens.shape
    toks = torch.tensor(qb.tokens)
    out_p1 = torch.full((B, S - 1), float("nan"))
    out_q = torch.full((B, S - 1), float("nan"))
    mask = torch.zeros(B, S - 1, dtype=torch.bool)

    r1_start = int(qb.roll_start_pos[0])      # 0
    r2_start = int(qb.roll_start_pos[1])      # t1 + 1

    # rollout-1 bits at columns [0, t1)
    bits1 = toks[:, r1_start : r1_start + t1].float()
    cum1 = torch.cumsum(bits1, dim=1)         # s after k+1 bits, k=0..t1-1
    s1_full = cum1[:, -1]                     # total ones in rollout1

    # rollout-1 predictions: input pos i in [0, t1-2] predicts bit i+1
    for i in range(t1 - 1):
        s_seen = cum1[:, i]            # ones in first i+1 bits
        t_seen = float(i + 1)
        out_p1[:, i] = (s_seen + 1.0) / (t_seen + 2.0)
        out_q[:, i] = float("nan")
        mask[:, i] = True

    # rollout-2 bits at columns [r2_start, r2_start+t2)
    bits2 = toks[:, r2_start : r2_start + t2].float()
    cum2 = torch.cumsum(bits2, dim=1)

    # input pos = r2_start - 1 (the ROLL token) predicts first bit of rollout2
    i = r2_start - 1
    p, q = quenched2_predictive_torch(s1_full, float(t1),
                                      torch.zeros(B), 0.0, chi2)
    out_p1[:, i] = p
    out_q[:, i] = q
    mask[:, i] = True

    # input pos = r2_start - 1 + (k+1) predicts bit k+1 of rollout2, k=0..t2-2
    for k in range(t2 - 1):
        i = r2_start + k
        s2_seen = cum2[:, k]
        t2_seen = float(k + 1)
        p, q = quenched2_predictive_torch(s1_full, float(t1),
                                          s2_seen, t2_seen, chi2)
        out_p1[:, i] = p
        out_q[:, i] = q
        mask[:, i] = True

    return out_p1, out_q, mask


def oracle_seq_annealed(tokens: np.ndarray):
    """Laplace oracle for a single-rollout annealed sequence. Returns (p1, mask)."""
    toks = torch.tensor(tokens)
    B, L = toks.shape
    cum = torch.cumsum(toks.float(), dim=1)
    out = torch.full((B, L - 1), float("nan"))
    for i in range(L - 1):
        out[:, i] = (cum[:, i] + 1.0) / (float(i + 1) + 2.0)
    mask = torch.ones(B, L - 1, dtype=torch.bool)
    return out, mask


@torch.no_grad()
def model_p1(model, tokens: torch.Tensor):
    """Model's renormalized P(next bit=1) over {0,1} and leaked mass on ROLL.
    tokens: (B, S) -> returns (p1 (B,S-1), leak (B,S-1))."""
    logits = model(tokens[:, :-1])          # predict token i+1 from i
    probs = torch.softmax(logits, dim=-1)   # (B,S-1,V)
    p0 = probs[..., 0]
    p1 = probs[..., 1]
    leak = probs[..., 2] if probs.shape[-1] > 2 else torch.zeros_like(p0)
    p1_renorm = p1 / (p0 + p1 + 1e-12)
    return p1_renorm, leak


def build_quenched_eval(roll_lens, nprior, n_eval, seed, device):
    """Build a fixed evaluation set + precomputed oracle for quenched J=2.

    Returns a dict with tokens, oracle next-1 probs, collision posterior, masks
    (all positions / rollout-2 only), rollout-1 counts, the Laplace(s1,t1) value
    per sequence, and the ROLL input position.
    """
    chi2 = nprior.mean_inv()
    t1 = roll_lens[0]
    rng = np.random.default_rng(seed)
    qb = data.gen_quenched(n_eval, roll_lens, nprior, rng)
    toks = torch.tensor(qb.tokens, device=device)
    orac_p1, orac_q, mask = oracle_seq_quenched2(qb, chi2)
    r2mask = mask.clone()
    r2mask[:, :t1] = False  # input idx >= t1 (ROLL pos) predicts rollout-2 bits
    s1 = qb.tokens[:, :t1].sum(axis=1)
    laplace_s1 = (s1 + 1.0) / (t1 + 2.0)
    return {
        "qb": qb, "tokens": toks, "chi2": chi2, "t1": t1, "roll_pos": t1,
        "orac_p1": orac_p1.to(device), "orac_q": orac_q, "mask": mask.to(device),
        "r2mask": r2mask.to(device), "s1": s1, "laplace_s1": laplace_s1,
    }


@torch.no_grad()
def first_token_pred(model, tokens: torch.Tensor, roll_pos: int):
    """Model P(first bit of rollout 2 = 1), read at the ROLL input position."""
    p1, _ = model_p1(model, tokens)
    return p1[:, roll_pos].detach().cpu().numpy()


def fit_transfer_coef(model_first_p1: np.ndarray, laplace_vals: np.ndarray):
    """Fit model_first ≈ alpha + beta * laplace(s1,t1).  The exact oracle has
    beta = chi2 = E[1/N] and alpha = (1-chi2)/2, so beta is the model's *effective*
    collision susceptibility chi2_hat.  Returns (beta, alpha, r2)."""
    x = laplace_vals.astype(np.float64)
    y = model_first_p1.astype(np.float64)
    A = np.stack([np.ones_like(x), x], axis=1)
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    alpha, beta = coef
    yhat = A @ coef
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean()) ** 2).sum() + 1e-12
    return float(beta), float(alpha), float(1 - ss_res / ss_tot)


def kl_to_oracle(p1_model, p1_oracle, mask):
    """Mean KL(oracle_Bernoulli || model_Bernoulli) over masked positions (nats)."""
    eps = 1e-7
    pm = p1_model.clamp(eps, 1 - eps)
    po = p1_oracle.clamp(eps, 1 - eps)
    kl = po * torch.log(po / pm) + (1 - po) * torch.log((1 - po) / (1 - pm))
    kl = kl[mask]
    return kl.mean().item()
