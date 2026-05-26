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
def compute_centered_g(
    A: torch.Tensor,                # (B, n_heads, T_q, T_k)
    z_ln1: torch.Tensor,            # (B, T_k, d_sae_ln1)
    beta: torch.Tensor,             # (n_heads, d_sae_ln1)
) -> dict[str, torch.Tensor]:
    """Per-source OV write scalar g, its attention-weighted mean g_bar, and
    the centered-and-attended profile tilde_g[b, h, q, j] = A[b,h,q,j]·(g[b,h,j] − g_bar[b,h,q]).

    Used by both QK attribution (for the softmax-Jacobian effect of perturbing
    a query/key feature) and Triple attribution (for the V-side reduction).
    Sanity: Σ_j tilde_g[b, h, q, j] == 0.
    """
    A = A.float()
    z = z_ln1.to(A.device).float()
    beta = beta.to(A.device).float()
    g = torch.einsum("bjf,hf->bhj", z, beta)               # (B, n_heads, T_k)
    g_bar = torch.einsum("bhqj,bhj->bhq", A, g)            # (B, n_heads, T_q)
    tilde_g = A * (g.unsqueeze(2) - g_bar.unsqueeze(-1))   # (B, n_heads, T_q, T_k)
    return {"g": g, "g_bar": g_bar, "tilde_g": tilde_g}


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


@torch.no_grad()
def rank_ov_diff(
    A: torch.Tensor,              # (B, n_heads, T_q, T_k)
    z_ln1: torch.Tensor,          # (B, T_k, d_sae)
    sae_ln1: TopKSAE,
    W_V: torch.Tensor,            # (n_heads, d_model, d_head)
    W_O: torch.Tensor,            # (n_heads, d_head, d_model)
    is_deployment: torch.Tensor,  # (B,) bool
    query_mask: torch.Tensor | None = None,  # (B, T_q) bool
) -> dict[str, torch.Tensor]:
    """Target-free OV feature ranking (diff regime).

    Per ``docs/feature_attribution.md``. For each ln1 SAE feature λ:

        M[b,h,q,λ]   = Σ_k A[b,h,q,k] · z[b,k,λ]
        diff_M[h,λ]   = mean_{b∈dep, q∈qm}[M] - mean_{b∈cln, q∈qm}[M]
        diff_vec[λ]   = Σ_h diff_M[h,λ] · (W_dec[λ] @ W_OV^h)   ∈ R^d_model
        score[λ]      = ‖diff_vec[λ]‖₂

    Heads sum as vectors before norming so that opposing-direction
    contributions cancel.
    """
    device = A.device
    A = A.float()
    z = z_ln1.to(device).float()
    is_dep = is_deployment.to(device)

    # M[b,h,q,λ] = Σ_k A[b,h,q,k] · z[b,k,λ]
    M = torch.einsum("bhqk,bkf->bhqf", A, z)        # (B, n_heads, T_q, d_sae)

    if query_mask is not None:
        qm = query_mask.to(device).float()
        M = M * qm.unsqueeze(1).unsqueeze(-1)        # broadcast over heads, features
        denom_dep = qm[is_dep].sum().clamp(min=1.0)
        denom_cln = qm[~is_dep].sum().clamp(min=1.0)
    else:
        T_q = M.shape[2]
        denom_dep = float(is_dep.sum().item()) * T_q
        denom_cln = float((~is_dep).sum().item()) * T_q

    M_dep = M[is_dep].sum(dim=(0, 2)) / denom_dep   # (n_heads, d_sae)
    M_cln = M[~is_dep].sum(dim=(0, 2)) / denom_cln
    del M
    diff_M = M_dep - M_cln                          # (n_heads, d_sae)

    W_V_ = W_V.to(device).float()
    W_O_ = W_O.to(device).float()
    W_OV = torch.einsum("hmd,hde->hme", W_V_, W_O_) # (n_heads, d_model, d_model)

    W_dec = sae_ln1.W_dec.detach().to(device).float()            # (d_sae, d_model)
    W_OV_feats = torch.einsum("fd,hde->hfe", W_dec, W_OV)        # (n_heads, d_sae, d_model)

    diff_contrib = torch.einsum("hf,hfd->fd", diff_M, W_OV_feats) # (d_sae, d_model)
    score = diff_contrib.norm(dim=-1)                              # (d_sae,)

    return {
        "score":       score,
        "top_indices": torch.argsort(score, descending=True),
        "diff_M":      diff_M,
        "diff_contrib": diff_contrib,
    }


@torch.no_grad()
def rank_qk_diff(
    z_ln1: torch.Tensor,          # (B, T, d_sae)
    sae_ln1: TopKSAE,
    W_Q: torch.Tensor,            # (n_heads, d_model, d_head)
    W_K: torch.Tensor,            # (n_heads, d_model, d_head)
    is_deployment: torch.Tensor,  # (B,) bool
    query_mask: torch.Tensor | None = None,  # (B, T) bool — prompt positions (Q-side)
    key_mask: torch.Tensor | None = None,    # (B, T) bool — defaults to all positions
) -> dict[str, torch.Tensor]:
    """Target-free QK feature-pair ranking (diff regime).

    Ranks pairs (λ_q, λ_k) by the magnitude of their expected pre-softmax
    attention logit contribution difference between deployment and clean prompts.
    No softmax Jacobian approximation needed (works directly on logits).

        QK_total[λ_q,λ_k] = (1/√d_head) Σ_h (W_dec@W_Q^h)[λ_q] · (W_dec@W_K^h)[λ_k]
        Z_q[b,λ]  = Σ_{q∈qm} z[b,q,λ]       (query-side aggregate)
        Z_k[b,λ]  = Σ_{s∈km} z[b,s,λ]       (key-side aggregate)
        score[λ_q,λ_k] = |QK_total[λ_q,λ_k]| · |mean_dep[Z_q·Z_k] - mean_cln[Z_q·Z_k]|

    Returns (d_sae, d_sae) score matrix plus flattened sort indices.
    """
    device = z_ln1.device
    z = z_ln1.float()
    is_dep = is_deployment.to(device)

    qm = query_mask.to(device).float().unsqueeze(-1) if query_mask is not None else None
    km = key_mask.to(device).float().unsqueeze(-1)   if key_mask  is not None else None

    Z_q = (z * qm).sum(dim=1) if qm is not None else z.sum(dim=1)  # (B, d_sae)
    Z_k = (z * km).sum(dim=1) if km is not None else z.sum(dim=1)  # (B, d_sae)

    N_dep = is_dep.sum().clamp(min=1).float()
    N_cln = (~is_dep).sum().clamp(min=1).float()
    outer_dep = Z_q[is_dep].T  @ Z_k[is_dep]  / N_dep   # (d_sae, d_sae)
    outer_cln = Z_q[~is_dep].T @ Z_k[~is_dep] / N_cln
    diff_outer = outer_dep - outer_cln                   # (d_sae, d_sae)

    W_Q_ = W_Q.to(device).float()
    W_K_ = W_K.to(device).float()
    W_dec = sae_ln1.W_dec.detach().to(device).float()   # (d_sae, d_model)
    Q_feats = torch.einsum("fd,hde->hfe", W_dec, W_Q_)  # (n_heads, d_sae, d_head)
    K_feats = torch.einsum("fd,hde->hfe", W_dec, W_K_)  # (n_heads, d_sae, d_head)
    d_head = W_Q_.shape[-1]
    QK_total = torch.einsum("hfd,hgd->fg", Q_feats, K_feats) / (d_head ** 0.5)  # (d_sae, d_sae)

    score = (QK_total * diff_outer).abs()               # (d_sae, d_sae)
    flat  = torch.argsort(score.flatten(), descending=True)
    d_sae = z_ln1.shape[-1]

    return {
        "score":       score,
        "top_pairs_q": flat // d_sae,
        "top_pairs_k": flat %  d_sae,
        "QK_total":    QK_total,
        "diff_outer":  diff_outer,
    }


@torch.no_grad()
def rank_qk_plus_ov_diff(
    z_ln1: torch.Tensor,           # (B, T, d_sae)
    sae_ln1: TopKSAE,
    W_Q: torch.Tensor,             # (n_heads, d_model, d_head)
    W_K: torch.Tensor,             # (n_heads, d_model, d_head)
    W_V: torch.Tensor,             # (n_heads, d_model, d_head)
    W_O: torch.Tensor,             # (n_heads, d_head, d_model)
    is_deployment: torch.Tensor,   # (B,) bool
    triplets: torch.Tensor,        # (N, 3) int — [λ, μ, ν] triplets to score
    query_mask: torch.Tensor | None = None,   # (B, T) bool — Q-side (carries λ)
    key_mask:   torch.Tensor | None = None,   # (B, T) bool — K-side (carries μ and ν)
) -> dict[str, torch.Tensor]:
    """Target-free QK+OV triplet ranking (diff regime).

    Per ``docs/feature_attribution.md``. For each triplet (λ, μ, ν):

        weight_h[h,λ,μ,ν,:] = (1/√d_head) · (W_dec[λ] W_QK^h W_dec[μ]^T) · (W_dec[ν] W_OV^h)  ∈ R^d_model
        Z_q[p, λ]           = Σ_{q∈qm} z[p, q, λ]
        Y[p, μ, ν]          = Σ_{k∈km} z[p, k, μ] · z[p, k, ν]
        diff[λ,μ,ν]         = mean_{p∈dep}[Z_q · Y] - mean_{p∈cln}[Z_q · Y]
        score[λ,μ,ν]        = ‖ diff · Σ_h weight_h ‖₂

    Heads sum as vectors before norming so opposing-direction contributions
    cancel. Full d_sae³ enumeration is infeasible; pass a candidate list
    (e.g. the Cartesian product of single-channel top-K rankings).
    """
    device = z_ln1.device
    z = z_ln1.float()
    is_dep = is_deployment.to(device).bool()

    qm = query_mask.to(device).float().unsqueeze(-1) if query_mask is not None else None
    km = key_mask.to(device).float().unsqueeze(-1)   if key_mask  is not None else None

    Z_q = (z * qm).sum(dim=1) if qm is not None else z.sum(dim=1)          # (B, d_sae)
    z_k = (z * km) if km is not None else z                                 # (B, T, d_sae)

    W_dec = sae_ln1.W_dec.detach().to(device).float()                       # (d_sae, d_model)
    W_Q_ = W_Q.to(device).float(); W_K_ = W_K.to(device).float()
    W_V_ = W_V.to(device).float(); W_O_ = W_O.to(device).float()
    Q_feats = torch.einsum("fd,hde->hfe", W_dec, W_Q_)                      # (n_heads, d_sae, d_head)
    K_feats = torch.einsum("fd,hde->hfe", W_dec, W_K_)                      # (n_heads, d_sae, d_head)
    W_OV    = torch.einsum("hmd,hde->hme", W_V_, W_O_)                      # (n_heads, d_model, d_model)
    W_OV_feats = torch.einsum("fd,hde->hfe", W_dec, W_OV)                   # (n_heads, d_sae, d_model)
    d_head = W_Q_.shape[-1]

    lams = triplets[:, 0].to(device).long()
    mus  = triplets[:, 1].to(device).long()
    nus  = triplets[:, 2].to(device).long()

    # weight_vec[i, :] = (1/√d_head) Σ_h (Q_feats[h, λ_i] · K_feats[h, μ_i]) · W_OV_feats[h, ν_i, :]
    qk_pair_per_head = (Q_feats[:, lams, :] * K_feats[:, mus, :]).sum(dim=-1)  # (n_heads, N)
    weight_vec = torch.einsum("hn,hnd->nd",
                              qk_pair_per_head, W_OV_feats[:, nus, :]) / (d_head ** 0.5)  # (N, d_model)

    # data[p, i] = Z_q[p, λ_i] · Σ_k z[p, k, μ_i] · z[p, k, ν_i]
    Y_per_prompt = (z_k[:, :, mus] * z_k[:, :, nus]).sum(dim=1)                # (B, N)
    data = Z_q[:, lams] * Y_per_prompt                                          # (B, N)

    N_dep = is_dep.sum().clamp(min=1).float()
    N_cln = (~is_dep).sum().clamp(min=1).float()
    diff = data[is_dep].sum(dim=0) / N_dep - data[~is_dep].sum(dim=0) / N_cln   # (N,)

    score = (diff.unsqueeze(-1) * weight_vec).norm(dim=-1)                     # (N,)

    return {
        "score":       score,
        "top_indices": torch.argsort(score, descending=True),
        "triplets":    triplets,
        "weight_vec":  weight_vec,
        "diff":        diff,
    }


def select_features(
    contrib: torch.Tensor,              # (B, n_heads, T_q, d_sae_ln1) from ov_attribution
    is_deployment: torch.Tensor,        # (B,) bool
    top_k: int,
    *,
    method: str = "jamie",              # "jamie" | "ketan"
    query_mask: torch.Tensor | None = None,
) -> dict:
    """Pick top-k ln1 SAE features for downstream OV-only steering.

    Both methods share the same OV contribution machinery
    (`compute_ov_weights` + `ov_attribution`). They differ in how the (h, λ)
    contribution matrix is reduced to a ranked feature list:

    `method="jamie"` — head-summed, prompt-masked. Computes
    `score_λ = Σ_h (mean_{b∈dep, q∈qm} contrib − mean_{b∈clean, q∈qm} contrib)`
    via `rank_dep_vs_clean`, then takes the top-k features by `|score|`. Heads
    are committed to "broadcast V on all 16" downstream; no head provenance.

    `method="ketan"` — head-resolved, no prompt mask. For each (h, λ) pair
    computes the same dep-minus-clean diff, sorts pairs by `|signed diff|`,
    walks the pair list and emits each new feature in pair order until `top_k`
    unique features have been collected. Returns provenance — the (h, λ) pair
    that surfaced each feature. Matches Ketan's `dep_vs_clean_contribution`
    ranking with `unique_features` deduplication on
    `ketan-ov-1000-prompts:tracing_feature/scripts/ov_path.py`.

    `query_mask` is honored only for `method="jamie"` (Ketan's convention is
    no q-mask). Both methods receive the same `contrib` tensor; the choice of
    `query_mask` at the *attribution* step is the caller's responsibility.

    Returns:
        features:   list[int] — top-k feature indices in rank order.
        provenance: list[tuple[int, int]] | None — for "ketan", the
                    (head, feature) pair surfacing each emitted feature; None
                    for "jamie".
        scores:     torch.Tensor (d_sae_ln1,) — per-feature score:
                      jamie: signed `per_lambda_dep − per_lambda_cln`.
                      ketan: max over heads of `|per_pair_dep − per_pair_cln|`.
    """
    if method == "jamie":
        ranked = rank_dep_vs_clean(contrib, is_deployment, query_mask=query_mask)
        order = ranked["top_indices"][:top_k].cpu().tolist()
        return {
            "features":   [int(f) for f in order],
            "provenance": None,
            "scores":     ranked["score"].cpu(),
        }

    if method == "ketan":
        # Ketan's convention: rank pairs over all (b, q) — query_mask is
        # ignored at the ranking step. We pass query_mask=None into the
        # underlying per-pair averaging.
        ranked = rank_dep_vs_clean(contrib, is_deployment, query_mask=None)
        per_pair_diff = ranked["per_pair_dep"] - ranked["per_pair_cln"]   # (n_heads, d_sae)
        n_heads, d_sae = per_pair_diff.shape
        order_flat = torch.argsort(per_pair_diff.abs().flatten(), descending=True)
        seen: set[int] = set()
        features: list[int] = []
        provenance: list[tuple[int, int]] = []
        for idx in order_flat.tolist():
            h, f = idx // d_sae, idx % d_sae
            if f in seen:
                continue
            seen.add(f)
            features.append(int(f))
            provenance.append((int(h), int(f)))
            if len(features) >= top_k:
                break
        per_feature_score = per_pair_diff.abs().max(dim=0).values         # (d_sae,)
        return {
            "features":   features,
            "provenance": provenance,
            "scores":     per_feature_score.cpu(),
        }

    raise ValueError(f"unknown selection method {method!r}; expected 'jamie' or 'ketan'")
