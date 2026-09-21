"""Removal / collateral metrics for the concept-cut comparison.

The distinct-vocab structure factorizes prediction EXACTLY:
    log p(v_{t+1}) = log p(block(v_{t+1})) + log p(v_{t+1} | block)
Concept ("which component") lives entirely in the block factor; the
within-block factor is the belief-state tracking we want to spare.

Removal fraction  RF = (blockCE_cut - blockCE_clean) / (blockCE_prior - blockCE_clean)
    0 = concept untouched, 1 = model degraded to the no-context prior.
Collateral fraction CF = (withinCE_cut - withinCE_clean) / (log 3 - withinCE_clean)
    0 = within-block prediction untouched, 1 = degraded to uniform.
Both against analytic Bayes anchors.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch


def block_masses(logits: torch.Tensor, block_of_token: torch.Tensor, K: int) -> tuple[torch.Tensor, torch.Tensor]:
    """logits (B,T,V) -> (p (B,T,V), m (B,T,K)) with m_c = sum of block-c token probs."""
    p = torch.softmax(logits.float(), dim=-1)
    B, T, V = p.shape
    m = torch.zeros(B, T, K, dtype=p.dtype, device=p.device)
    idx = block_of_token.to(p.device).view(1, 1, V).expand(B, T, V)
    m.scatter_add_(-1, idx, p)
    return p, m


def factorized_ce(
    logits: torch.Tensor,          # (B, T, V)
    tokens: torch.Tensor,          # (B, T)
    block_of_token: torch.Tensor,  # (V,)
    t_min: int = 8,
) -> dict[str, torch.Tensor]:
    """Per-eval-point block CE and within-block CE (predicting token t+1 at t)."""
    B, T, V = logits.shape
    K = int(block_of_token.max().item()) + 1
    p, m = block_masses(logits, block_of_token, K)

    tgt = tokens[:, 1:]                              # (B, T-1) next token
    tgt_block = block_of_token.to(tgt.device)[tgt]   # (B, T-1)
    p_pred = p[:, :-1]                               # prediction made at t
    m_pred = m[:, :-1]

    p_tok = p_pred.gather(-1, tgt.unsqueeze(-1).to(p_pred.device)).squeeze(-1).clamp(min=1e-12)
    m_blk = m_pred.gather(-1, tgt_block.unsqueeze(-1).to(m_pred.device)).squeeze(-1).clamp(min=1e-12)

    block_ce = -m_blk.log()
    within_ce = -(p_tok / m_blk).clamp(min=1e-12).log()

    sl = slice(t_min, None)
    return {
        "block_ce": block_ce[:, sl],
        "within_ce": within_ce[:, sl],
        "block_mass": m_pred[:, sl],      # (B, T-1-t_min, K)
        "tgt_block": tgt_block[:, sl],
    }


def bayes_anchors(ds, idx: slice | torch.Tensor, t_min: int = 8) -> dict[str, float]:
    """Analytic anchors on eval sequences: Bayes block CE, prior block CE,
    Bayes within CE, plus omega-posterior for tracking R^2."""
    tokens = ds.tokens[idx]
    post = ds.posterior_omegas[idx]                  # (B, T, K) after token t
    blk_map = ds.block_of_token
    tgt_block = blk_map[tokens[:, 1:]]

    omega_pred = post[:, :-1]                        # prediction made at t
    bayes_blk = -omega_pred.gather(-1, tgt_block.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12).log()

    prior = torch.tensor(ds.config.omega)
    prior_blk = -prior[tgt_block].clamp(min=1e-12).log()

    bnt = ds.bayes_next_token()[idx][:, :-1]         # (B, T-1, V) p*(v_{t+1}|x_{0:t})
    tgt = tokens[:, 1:]
    p_tok = bnt.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
    m_blk = omega_pred.gather(-1, tgt_block.unsqueeze(-1)).squeeze(-1).clamp(min=1e-12)
    bayes_within = -(p_tok / m_blk).clamp(min=1e-12).log()

    sl = slice(t_min, None)
    return {
        "bayes_block_ce": float(bayes_blk[:, sl].mean()),
        "prior_block_ce": float(prior_blk[:, sl].mean()),
        "bayes_within_ce": float(bayes_within[:, sl].mean()),
        "uniform_within_ce": float(np.log(3.0)),
        "omega_pred": omega_pred[:, sl],             # (B, T', K) tracking target
    }


def tracking_r2(block_mass: torch.Tensor, omega_pred: torch.Tensor) -> float:
    """Mean over components of R^2 between model block mass and Bayes posterior."""
    m = block_mass.reshape(-1, block_mass.shape[-1]).numpy().astype(np.float64)
    w = omega_pred.reshape(-1, omega_pred.shape[-1]).numpy().astype(np.float64)
    sse = ((m - w) ** 2).sum(axis=0)
    sst = ((w - w.mean(axis=0)) ** 2).sum(axis=0)
    return float((1.0 - sse / np.clip(sst, 1e-12, None)).mean())


def ridge_probe_r2(
    x: np.ndarray,      # (n, D) activations
    y: np.ndarray,      # (n, K) targets
    n_sequences: int,
    per_seq: int,
    seed: int = 0,
    ridge: float = 1e-4,
) -> float:
    """Held-out (by sequence) mean R^2 of a retrained ridge probe — the
    adversarial 'is the concept still linearly present' metric."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_sequences)
    split = int(0.8 * n_sequences)
    is_train = np.zeros(n_sequences, dtype=bool)
    is_train[perm[:split]] = True
    flat_train = np.repeat(is_train, per_seq)

    xtr, ytr = x[flat_train], y[flat_train]
    xte, yte = x[~flat_train], y[~flat_train]
    xm, ym = xtr.mean(0, keepdims=True), ytr.mean(0, keepdims=True)
    xc, yc = xtr - xm, ytr - ym
    xtx = xc.T @ xc
    xtx.flat[:: xtx.shape[0] + 1] += ridge * np.trace(xtx) / xtx.shape[0]
    beta = np.linalg.solve(xtx, xc.T @ yc)
    pred = (xte - xm) @ beta + ym
    sse = ((yte - pred) ** 2).sum(axis=0)
    sst = ((yte - yte.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    return float((1.0 - sse / np.clip(sst, 1e-12, None)).mean())


def summarize(
    ce: dict[str, torch.Tensor],
    anchors: dict[str, Any],
    clean_block_ce: float | None = None,
    clean_within_ce: float | None = None,
) -> dict[str, float]:
    block_ce = float(ce["block_ce"].mean())
    within_ce = float(ce["within_ce"].mean())
    out = {
        "block_ce": block_ce,
        "within_ce": within_ce,
        "tracking_r2": tracking_r2(ce["block_mass"], anchors["omega_pred"]),
    }
    if clean_block_ce is not None:
        # fixed Bayes-range denominators (robust even if the clean model is
        # close to — or worse than — the naive baselines)
        denom_r = anchors["prior_block_ce"] - anchors["bayes_block_ce"]
        denom_c = anchors["uniform_within_ce"] - anchors["bayes_within_ce"]
        out["removal_frac"] = (block_ce - clean_block_ce) / max(denom_r, 1e-9)
        out["collateral_frac"] = (within_ce - clean_within_ce) / max(denom_c, 1e-9)
    return out
