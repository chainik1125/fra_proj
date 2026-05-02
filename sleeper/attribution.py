"""OV-only attribution to a target direction d at the post-attention residual.

Implements

    C^OV_{h, q, k, λ} = A^h_{q,k} · z_ln1[k, λ] · ⟨W_dec_ln1[λ] · W_OV^h, d⟩

where W_OV^h = W_V^h W_O^h is the per-head OV virtual weight, and d is a
direction in the resid_mid space (typically the SAE_mid encoder column for the
suppressor feature). Attention pattern A and post-LN activations z_ln1 are
taken from the unmodified forward pass (frozen attention).

Given frozen A:
    e · attn_out[q] = Σ_h Σ_k A^h_{q,k} · ⟨x_k W_OV^h, d⟩ + Σ_h ⟨b_V^h W_O^h, d⟩ + ⟨b_O, d⟩

and substituting x_k ≈ Σ_λ z_ln1[k, λ] · W_dec_ln1[λ] + b_dec_ln1 gives the
per-(h, k, λ) attribution above (plus bias constants).

This module returns the full 4D tensor when memory permits, plus the standard
aggregations used to rank features for downstream OV-only intervention.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sleeper.sae import TopKSAE


@torch.no_grad()
def compute_ov_weights(
    model: HookedTransformer,
    sae_ln1: TopKSAE,
    d: torch.Tensor,                    # (d_model,) target direction
    block: int = 0,
) -> dict[str, torch.Tensor]:
    """Pre-compute the per-(h, λ) OV strength β[h, λ] = ⟨W_dec_ln1[λ] · W_OV^h, d⟩.

    Returns:
        beta:        (n_heads, d_sae_ln1)  — per-(head, ln1 feature) OV weight
        u:           (n_heads, d_model)    — per-head virtual read direction
                                             u_h = W_V^h @ W_O^h @ d  (rows are read directions)
        const:       scalar — bias contribution Σ_h ⟨b_V^h W_O^h, d⟩ + ⟨b_O, d⟩
                              + Σ_h ⟨b_dec_ln1 W_OV^h, d⟩
    """
    device = next(model.parameters()).device
    d = d.to(device).float()

    W_V = model.W_V[block].detach().float()      # (n_heads, d_model, d_head)
    W_O = model.W_O[block].detach().float()      # (n_heads, d_head, d_model)
    b_V = model.b_V[block].detach().float()      # (n_heads, d_head)
    b_O = model.b_O[block].detach().float()      # (d_model,)

    # u_h = W_V_h @ W_O_h @ d   ∈ R^{d_model}, rows of U
    woe = torch.einsum("hdm,m->hd", W_O, d)      # (n_heads, d_head)
    U = torch.einsum("hmd,hd->hm", W_V, woe)     # (n_heads, d_model)

    W_dec_ln1 = sae_ln1.W_dec.detach().float()   # (d_sae_ln1, d_model)
    beta = W_dec_ln1 @ U.T                       # (d_sae_ln1, n_heads)

    b_dec_ln1 = sae_ln1.b_dec.detach().float()   # (d_model,)
    const = (
        (b_dec_ln1 @ U.T).sum().item()           # Σ_h b_dec_ln1 · u_h
        + (b_V * woe).sum().item()               # Σ_h b_V_h · (W_O_h d)
        + (d @ b_O).item()                        # b_O · d
    )

    return {"beta": beta.T.contiguous(), "u": U, "const": float(const)}


@torch.no_grad()
def ov_attribution(
    A: torch.Tensor,                    # (B, n_heads, T_q, T_k)  attention pattern
    z_ln1: torch.Tensor,                # (B, T_k, d_sae_ln1)     ln1 SAE codes
    beta: torch.Tensor,                 # (n_heads, d_sae_ln1)    from compute_ov_weights
) -> dict[str, torch.Tensor]:
    """Per-(head, src, ln1-feature) OV contribution, aggregated several ways.

    Returns dict with:
        per_pair_dep_contrib  (n_heads, d_sae_ln1)  if `is_dep` provided downstream
        per_head_S            (B, T_q, n_heads)     S_h(q) = Σ_k A^h_{q,k} (x_k · u_h)
                                                    = Σ_{k, λ} A · z_ln1 · β
        per_pair_total        (n_heads, d_sae_ln1)  Σ_{B, q, k} A · z_ln1 · β
                                                    (mean would divide by B*T)

    For ranking features per Dmitry's procedure ("dep vs clean diffing"):
        rank_dep_vs_clean(per_pair, is_dep) below.
    """
    device = A.device
    A = A.float()
    z = z_ln1.to(device).float()
    beta = beta.to(device).float()

    # M[b, h, q, λ] = Σ_k A[b, h, q, k] · z[b, k, λ]
    M = torch.einsum("bhqk,bkf->bhqf", A, z)             # (B, n_heads, T_q, d_sae_ln1)
    # contribution[b, h, q, λ] = M[b, h, q, λ] · β[h, λ]
    contrib = M * beta.unsqueeze(0).unsqueeze(2)         # (B, n_heads, T_q, d_sae_ln1)

    per_head_S = contrib.sum(dim=-1).permute(0, 2, 1)    # (B, T_q, n_heads)
    per_pair_total = contrib.sum(dim=(0, 2))             # (n_heads, d_sae_ln1)

    return {
        "per_head_S": per_head_S,
        "per_pair_total": per_pair_total,
        "contrib": contrib,        # full 4D, use .mean / .sum / index for custom views
    }


def rank_dep_vs_clean(
    contrib: torch.Tensor,              # (B, n_heads, T_q, d_sae_ln1) from ov_attribution
    is_deployment: torch.Tensor,        # (B,) bool
    query_mask: torch.Tensor | None = None,   # (B, T_q) bool, defaults to all
) -> dict[str, torch.Tensor]:
    """Difference of mean per-(h, λ) OV contribution between dep and clean prompts.

    Sums over (heads, query positions) per Dmitry's diffing procedure:
        rank_λ = Σ_h mean_{q,b∈dep}(contrib[b, h, q, λ]) - Σ_h mean_{q,b∈clean}(...)
    """
    device = contrib.device
    is_dep = is_deployment.to(device)
    qm_raw = (torch.ones(contrib.shape[0], contrib.shape[2], dtype=torch.bool, device=device)
              if query_mask is None else query_mask.to(device))
    qm = qm_raw.float().unsqueeze(1).unsqueeze(-1)        # (B, 1, T_q, 1)

    masked = contrib * qm                                                 # (B, h, T_q, λ)
    den_dep = qm[is_dep].sum().clamp(min=1.0)
    den_cln = qm[~is_dep].sum().clamp(min=1.0)

    per_pair_dep = masked[is_dep].sum(dim=(0, 2)) / den_dep              # (h, λ)
    per_pair_cln = masked[~is_dep].sum(dim=(0, 2)) / den_cln
    per_lambda_dep = per_pair_dep.sum(dim=0)                              # (λ,)
    per_lambda_cln = per_pair_cln.sum(dim=0)
    score = per_lambda_dep - per_lambda_cln
    return {
        "score": score,
        "per_lambda_dep": per_lambda_dep,
        "per_lambda_cln": per_lambda_cln,
        "per_pair_dep": per_pair_dep,
        "per_pair_cln": per_pair_cln,
        "top_indices": torch.argsort(score.abs(), descending=True),
    }
