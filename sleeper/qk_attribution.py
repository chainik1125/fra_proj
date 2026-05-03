"""QK-side feature attribution via softmax-Jacobian linearization.

Per ``working_notes/05_qk_side_derivation.md`` on Dmitry's branch. The score-
level decomposition

    s^h_{qk} = Σ_{μ,ν} u^μ_q u^ν_k  ω^{h,QK}_{μν}  + const,
    ω^{h,QK}_{μν} := (f_μ W_Q^h) · (f_ν W_K^h) / sqrt(d_h)

becomes, after first-order softmax linearization, a per-feature pattern-shift
sensitivity. Reading through the OV path to the chosen target direction d:

  Per-query-feature score (rank μ as a Q-side feature):
    contrib_Q[b, h, q, μ] = u^μ_q(b) · Σ_j κ^{h,μ}_j(b) · tilde_g^h_{q,j}(b)
    κ^{h,μ}_j(b) = (f_μ W_Q^h) · (Σ_ν z[b,j,ν] f_ν W_K^h) / sqrt(d_h)

  Per-key-feature score (rank ν as a K-side feature):
    contrib_K[b, h, k, ν] = u^ν_k(b) · Σ_q η^{h,ν}_q(b) · tilde_g^h_{q,k}(b)
    η^{h,ν}_q(b) = (Σ_μ z[b,q,μ] f_μ W_Q^h) · (f_ν W_K^h) / sqrt(d_h)

`compute_centered_g` from sleeper.attribution provides tilde_g (using the OV β
to weight per-source writes); the rest is a pair of einsums per side. We chunk
over the batch dim so the (B, n_heads, T, d_sae) contrib tensor never lives in
memory at full size.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sleeper.attribution import compute_centered_g
from sleeper.sae import TopKSAE


@torch.no_grad()
def compute_qk_weights(
    model: HookedTransformer,
    sae_ln1: TopKSAE,
    block: int = 0,
) -> dict[str, torch.Tensor]:
    """Pre-compute the weight-space QK feature factors.

    Returns:
      F_Q: (n_heads, d_sae_ln1, d_head)  F_Q[h, μ] = sae.W_dec[μ] @ W_Q[h]
      F_K: (n_heads, d_sae_ln1, d_head)  F_K[h, ν] = sae.W_dec[ν] @ W_K[h]
      scale: 1/sqrt(d_head)
    """
    W_Q = model.W_Q[block].detach().float()                  # (n_heads, d_model, d_head)
    W_K = model.W_K[block].detach().float()
    W_dec = sae_ln1.W_dec.detach().float()                   # (d_sae, d_model)
    F_Q = torch.einsum("fd,hdk->hfk", W_dec, W_Q)            # (n_heads, d_sae, d_head)
    F_K = torch.einsum("fd,hdk->hfk", W_dec, W_K)
    d_head = W_Q.shape[-1]
    return {"F_Q": F_Q, "F_K": F_K, "scale": 1.0 / (d_head ** 0.5)}


def _chunked(B: int, chunk: int):
    s = 0
    while s < B:
        yield s, min(s + chunk, B)
        s += chunk


@torch.no_grad()
def qk_attribution(
    A: torch.Tensor,                # (B, n_heads, T_q, T_k)
    z_ln1: torch.Tensor,            # (B, T, d_sae_ln1)
    F_Q: torch.Tensor,              # (n_heads, d_sae_ln1, d_head)
    F_K: torch.Tensor,
    beta: torch.Tensor,             # (n_heads, d_sae_ln1) from compute_ov_weights
    scale: float,
    is_dep: torch.Tensor,           # (B,) bool
    prompt_mask: torch.Tensor,      # (B, T) bool — used as both query and key mask
    chunk: int = 16,
) -> dict[str, torch.Tensor]:
    """Compute per-feature Q-side and K-side rankings in one chunked pass.

    For each chunk of prompts we materialize tilde_g and the κ/η reductions,
    multiply by z to get per-(B, h, q-or-k, λ) contributions, then accumulate
    the per-λ aggregations directly. The full 4D contrib tensor is never held
    at batch scale.

    Returns dict with per-λ aggregations (shape (d_sae,)) for both Q-side and
    K-side, and the per-(dep, cln) means used by `select_qk_features`.
    """
    device = A.device
    B = A.shape[0]
    n_heads, d_sae, d_head = F_Q.shape
    is_dep_b = is_dep.to(device).bool()
    pm = prompt_mask.to(device).bool()

    # Aggregations we accumulate per chunk:
    #   l1_dep_Q[μ], l1_cln_Q[μ]   = mean |contrib_Q[b,h,q,μ]| over (b in dep/cln, h, q in prompt)
    #   sum_dep_Q[μ], sum_cln_Q[μ] = mean signed contrib_Q for dep_minus_clean ranking
    # Same shape for K-side.
    def _zeros():
        return torch.zeros(d_sae, device=device, dtype=torch.float64)

    aggs = {f"{kind}_{side}_{tag}": _zeros()
            for kind in ("l1", "sum") for side in ("Q", "K") for tag in ("dep", "cln")}
    counts = {f"{side}_{tag}": torch.zeros((), device=device, dtype=torch.float64)
              for side in ("Q", "K") for tag in ("dep", "cln")}

    F_Q_dev = F_Q.to(device).float()
    F_K_dev = F_K.to(device).float()
    beta_dev = beta.to(device).float()

    for s, e in _chunked(B, chunk):
        A_c = A[s:e].float()
        z_c = z_ln1[s:e].to(device).float()
        pm_c = pm[s:e].float()                              # (b, T)
        dep_c = is_dep_b[s:e]                                # (b,)

        # tilde_g[b, h, q, j] from sleeper.attribution helper
        tg = compute_centered_g(A_c, z_c, beta_dev)["tilde_g"]    # (b, h, T_q, T_k)

        # ---- Q-side: contrib_Q[b, h, q, μ] = z[b,q,μ] · S_Q[b,h,q,μ] ----
        # r_K[b,h,j,d] = Σ_ν z[b,j,ν] · F_K[h,ν,d]
        r_K = torch.einsum("bjf,hfd->bhjd", z_c, F_K_dev)
        # T_Q[b,h,q,d] = Σ_j tg[b,h,q,j] · r_K[b,h,j,d]
        T_Q = torch.einsum("bhqj,bhjd->bhqd", tg, r_K)
        # S_Q[b,h,q,μ] = Σ_d F_Q[h,μ,d] · T_Q[b,h,q,d] · scale
        S_Q = torch.einsum("hfd,bhqd->bhqf", F_Q_dev, T_Q) * scale
        # contrib_Q[b,h,q,μ] = z[b,q,μ] · S_Q[b,h,q,μ]
        contrib_Q = z_c.unsqueeze(1) * S_Q                   # (b, h, T_q, d_sae)
        del r_K, T_Q, S_Q

        # ---- K-side: contrib_K[b, h, k, ν] = z[b,k,ν] · S_K[b,h,k,ν] ----
        # u_Q[b,h,q,d] = Σ_μ z[b,q,μ] · F_Q[h,μ,d]
        u_Q = torch.einsum("bqf,hfd->bhqd", z_c, F_Q_dev)
        # T_K[b,h,k,d] = Σ_q tg[b,h,q,k] · u_Q[b,h,q,d]
        T_K = torch.einsum("bhqk,bhqd->bhkd", tg, u_Q)
        S_K = torch.einsum("hfd,bhkd->bhkf", F_K_dev, T_K) * scale
        contrib_K = z_c.unsqueeze(1) * S_K                   # (b, h, T_k, d_sae)
        del u_Q, T_K, S_K, tg

        # mask query/key positions to prompt; sum over heads then aggregate
        for side, contrib in (("Q", contrib_Q), ("K", contrib_K)):
            cmask = pm_c.unsqueeze(1).unsqueeze(-1)         # (b, 1, T, 1)
            masked = contrib * cmask                         # (b, h, T, d_sae)
            for tag, sel in (("dep", dep_c), ("cln", ~dep_c)):
                if not sel.any():
                    continue
                m = masked[sel]                              # (n, h, T, d_sae)
                pm_sel = pm_c[sel]
                count = pm_sel.sum() * n_heads
                aggs[f"l1_{side}_{tag}"] += m.abs().sum(dim=(0, 1, 2)).double()
                aggs[f"sum_{side}_{tag}"] += m.sum(dim=(0, 1, 2)).double()
                counts[f"{side}_{tag}"] += count.double()

    out = {}
    for side in ("Q", "K"):
        for tag in ("dep", "cln"):
            denom = counts[f"{side}_{tag}"].clamp(min=1.0)
            out[f"l1_{side}_{tag}"] = (aggs[f"l1_{side}_{tag}"] / denom).float()
            out[f"sum_{side}_{tag}"] = (aggs[f"sum_{side}_{tag}"] / denom).float()
        out[f"l1_mean_{side}"] = 0.5 * (out[f"l1_{side}_dep"] + out[f"l1_{side}_cln"])
        out[f"signed_dep_minus_clean_{side}"] = (
            out[f"sum_{side}_dep"] - out[f"sum_{side}_cln"]
        )
    return out


def _score_and_top(
    aggs: dict[str, torch.Tensor], side: str, score_kind: str, top_k: int,
) -> tuple[torch.Tensor, list[int]]:
    if score_kind == "l1_mean":
        score = aggs[f"l1_mean_{side}"]
    elif score_kind == "dep_minus_clean":
        score = aggs[f"signed_dep_minus_clean_{side}"]
    elif score_kind == "l1_dep":
        score = aggs[f"l1_{side}_dep"]
    else:
        raise ValueError(f"unknown qk score_kind: {score_kind}")
    order = torch.argsort(score.abs(), descending=True).tolist()
    return score, order[: top_k]


def select_qk_features(
    aggs: dict[str, torch.Tensor],
    top_k: int,
    score_kind: str = "l1_mean",
) -> tuple[list[tuple[int, str]], dict[str, list[dict]]]:
    """Top-K Q-side ∪ Top-K K-side features tagged with their channel.

    Returns (selected, ranked_top) where:
      selected: list[(feature_idx, channel ∈ {'Q','K'})]
      ranked_top: {'Q': [...], 'K': [...]} with per-feature score breakdowns.
    """
    selected: list[tuple[int, str]] = []
    ranked_top: dict[str, list[dict]] = {}
    for side in ("Q", "K"):
        score, top = _score_and_top(aggs, side, score_kind, top_k)
        selected.extend((int(f), side) for f in top)
        ranked_top[side] = [{
            "feature_idx": int(f),
            "score": float(score[f]),
            "l1_mean": float(aggs[f"l1_mean_{side}"][f]),
            "signed_dep_minus_clean": float(aggs[f"signed_dep_minus_clean_{side}"][f]),
            "l1_dep": float(aggs[f"l1_{side}_dep"][f]),
            "l1_cln": float(aggs[f"l1_{side}_cln"][f]),
        } for f in top]
    return selected, ranked_top
