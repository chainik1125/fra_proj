"""Triple-channel (Q, K, V) feature attribution via the Möbius decomposition.

Per ``notes/fra_channel_singles_pairs_triples.md`` on Dmitry's branch. For a
chosen FRA triplet (μ, ν, λ) the path-local coalition game

    g(z_Q, z_K, z_V) := A_k(z_Q, z_K) · (v_d^bg + z_V · δw_d^λ) - A_k(0,0) · v_d^bg

Möbius-decomposes into 7 terms ΔQ, ΔK, ΔV, ΔQK, ΔQV, ΔKV, ΔQKV with
g(QKV) = Σ Δ.

Implementation uses the §5 algebraic shortcut so the four attention patterns
A_k(z_Q, z_K) are derived from the clean pattern A_clean[k] plus per-(μ,ν)
scalar offsets (b_clean, c) and the per-μ "background scores everywhere"
shift (a_clean[j]):

  A_k(0,1) = A_clean[k] · exp(-a_at_k) / Z_01,    Z_01 = Σ_j A_clean[j] · exp(-a[j])
  A_k(1,0) = A_clean[k] · exp(-b)    / (1 - A_clean[k] · (1 - exp(-b)))
  A_k(0,0) = A_k(0,1)               via shift -(b - c) at slot k
  A_k(1,1) = A_clean[k]                                       (clean pattern)

with a[j] := δq · k_clean[j] / √d_h, b := q_clean · δk / √d_h, c := δq·δk / √d_h,
δq := z[μ] · F_Q[h,μ], δk := z[ν] · F_K[h,ν]. The V channel is independent of
attention: δw_d^λ := z[λ] · β[h, λ] (β reuses compute_ov_weights).

We pin one source position k_b per prompt (default: trigger pipe for dep,
"Story:" end for clean) and aggregate over query positions q ∈ prompt, heads
h, prompts b. Candidate triplets come from the cross-product of top-Kq Q-feats
× top-Kk K-feats × top-Kv V-feats — the full O(d_sae³) screen is infeasible.
"""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer

from sleeper.sae import TopKSAE


_DELTA_NAMES = ("dQ", "dK", "dV", "dQK", "dQV", "dKV", "dQKV")


@torch.no_grad()
def compute_triple_prep(
    model: HookedTransformer,
    sae_ln1: TopKSAE,
    beta: torch.Tensor,                # (n_heads, d_sae) from compute_ov_weights
    A: torch.Tensor,                   # (B, n_heads, T_q, T_k)
    ln1_acts: torch.Tensor,            # (B, T, d_model)
    z_ln1: torch.Tensor,               # (B, T, d_sae)
    k_pos_per_b: torch.Tensor,         # (B,) — fixed source position per prompt
    block: int = 0,
    device: torch.device | str = "cuda",
) -> dict:
    """Cache everything that doesn't depend on (μ, ν, λ).

    The per-key projection k_full[b, h, j, :] = ln1[b, j, :] @ W_K[h] is needed
    once for the per-μ a_clean[j] reduction; we keep it in fp32 (small).
    """
    device = torch.device(device)
    B, T = ln1_acts.shape[0], ln1_acts.shape[1]
    H = A.shape[1]

    W_Q = model.W_Q[block].detach().to(device).float()
    W_K = model.W_K[block].detach().to(device).float()
    d_head = W_Q.shape[-1]
    scale = 1.0 / (d_head ** 0.5)

    ln1_d = ln1_acts.to(device).float()
    A_d = A.to(device).float()
    z_d = z_ln1.to(device).float()
    beta_d = beta.to(device).float()
    k_pos = k_pos_per_b.to(device).long()

    bidx = torch.arange(B, device=device)
    z_at_k = z_d[bidx, k_pos]                                      # (B, d_sae)
    v_d_clean_at_k = torch.einsum("bf,hf->bh", z_at_k, beta_d)     # (B, H)

    A_clean_at_k = A_d.gather(
        -1, k_pos.view(B, 1, 1, 1).expand(-1, H, A_d.shape[2], 1)
    ).squeeze(-1)                                                  # (B, H, T_q)

    k_full = torch.einsum("btd,hdk->bhtk", ln1_d, W_K)             # (B, H, T_k, D_h)
    q_full = torch.einsum("btd,hdk->bhtk", ln1_d, W_Q)             # (B, H, T_q, D_h)

    return {
        "A": A_d, "A_clean_at_k": A_clean_at_k,
        "k_full": k_full, "q_full": q_full,
        "z": z_d, "z_at_k": z_at_k, "v_d_clean_at_k": v_d_clean_at_k,
        "beta": beta_d, "W_dec": sae_ln1.W_dec.detach().to(device).float(),
        "W_Q": W_Q, "W_K": W_K, "scale": scale, "d_head": d_head, "k_pos": k_pos,
    }


@torch.no_grad()
def triple_eval(
    prep: dict,
    triplets: torch.Tensor,            # (T_trip, 3) — feature indices (μ, ν, λ)
    is_dep: torch.Tensor,              # (B,) bool
    prompt_mask: torch.Tensor,         # (B, T_q) bool
) -> dict:
    """8-corner Möbius eval. Returns per-triplet aggregations over (b, h, q ∈ prompt)
    split into dep / clean. Per triplet we get 7 Δ × {dep_signed, cln_signed,
    dep_l1, cln_l1} = 28 floats."""
    A = prep["A"]; A_k = prep["A_clean_at_k"]
    k_full = prep["k_full"]; q_full = prep["q_full"]
    z = prep["z"]; z_at_k = prep["z_at_k"]
    v_d_bg_clean = prep["v_d_clean_at_k"]
    beta = prep["beta"]; W_dec = prep["W_dec"]
    W_Q = prep["W_Q"]; W_K = prep["W_K"]
    scale = prep["scale"]; k_pos = prep["k_pos"]

    device = A.device
    B, H, T_q = A_k.shape
    n_trip = triplets.shape[0]
    triplets = triplets.to(device).long()
    pm = prompt_mask.to(device).float()
    dep_b = is_dep.to(device).bool(); cln_b = ~dep_b

    # Sort triplets by (μ, ν) so per-μ and per-(μ,ν) work is amortized over runs.
    sort_key = triplets[:, 0] * (W_dec.shape[0]) + triplets[:, 1]
    order = torch.argsort(sort_key)
    triplets_sorted = triplets[order]
    inv_order = torch.argsort(order)

    aggs = {f"{n}_{tag}": torch.zeros(n_trip, device=device, dtype=torch.float64)
            for n in _DELTA_NAMES for tag in ("dep_sum", "cln_sum", "dep_abs", "cln_abs")}
    counts_dep = (pm * dep_b.float().unsqueeze(-1)).sum() * H
    counts_cln = (pm * cln_b.float().unsqueeze(-1)).sum() * H
    counts_dep = counts_dep.clamp(min=1.0); counts_cln = counts_cln.clamp(min=1.0)

    cur_mu = -1
    cur_nu = -1
    cache_mu: dict[str, torch.Tensor] = {}
    cache_nu: dict[str, torch.Tensor] = {}

    for s in range(n_trip):
        mu = int(triplets_sorted[s, 0])
        nu = int(triplets_sorted[s, 1])
        lam = int(triplets_sorted[s, 2])
        t_global = int(order[s])

        if mu != cur_mu:
            f_mu = W_dec[mu]                                    # (D,)
            F_Q_mu = torch.einsum("d,hdk->hk", f_mu, W_Q)        # (H, D_h)
            z_q_mu = z[:, :, mu]                                 # (B, T_q)
            # δq[b, h, q, :] = z[b, q, μ] · F_Q_mu[h, :]
            delta_q = z_q_mu.unsqueeze(1).unsqueeze(-1) * F_Q_mu.unsqueeze(0).unsqueeze(2)
            # a_full[b, h, q, j] = (δq · k_full[b, h, j, :]) / sqrt(d_h)
            a_full = torch.einsum("bhqd,bhjd->bhqj", delta_q, k_full) * scale
            exp_neg_a = torch.exp(-a_full)
            Z_01 = (A * exp_neg_a).sum(dim=-1).clamp(min=1e-30)  # (B, H, T_q)
            a_at_k = a_full.gather(
                -1, k_pos.view(B, 1, 1, 1).expand(-1, H, T_q, 1)
            ).squeeze(-1)                                        # (B, H, T_q)
            A_01_k = A_k * torch.exp(-a_at_k) / Z_01
            cache_mu = {"f_mu": f_mu, "F_Q_mu": F_Q_mu, "z_q_mu": z_q_mu, "A_01_k": A_01_k}
            del a_full, exp_neg_a, Z_01, a_at_k, delta_q
            cur_mu = mu; cur_nu = -1

        if nu != cur_nu:
            f_nu = W_dec[nu]
            F_K_nu = torch.einsum("d,hdk->hk", f_nu, W_K)
            z_kb_nu = z_at_k[:, nu]                              # (B,)
            delta_k = z_kb_nu.unsqueeze(-1).unsqueeze(-1) * F_K_nu.unsqueeze(0)  # (B, H, D_h)
            b_clean = torch.einsum("bhqd,bhd->bhq", q_full, delta_k) * scale
            F_QF_K = torch.einsum("hd,hd->h", cache_mu["F_Q_mu"], F_K_nu)        # (H,)
            c_term = (cache_mu["z_q_mu"].unsqueeze(1) * z_kb_nu.view(B, 1, 1)
                      * F_QF_K.view(1, H, 1) * scale)            # (B, H, T_q)
            eb = torch.exp(-b_clean)
            A_10_k = A_k * eb / (1.0 - A_k * (1.0 - eb)).clamp(min=1e-30)
            es = torch.exp(-(b_clean - c_term))
            A_00_k = cache_mu["A_01_k"] * es / (
                1.0 - cache_mu["A_01_k"] * (1.0 - es)
            ).clamp(min=1e-30)
            cache_nu = {"A_10_k": A_10_k, "A_00_k": A_00_k}
            cur_nu = nu

        # Per-triplet (per-λ) eval using cached A_*_k tensors:
        delta_w_d = z_at_k[:, lam].unsqueeze(-1) * beta[:, lam].unsqueeze(0)   # (B, H)
        v_bg = (v_d_bg_clean - delta_w_d).unsqueeze(-1)          # (B, H, 1)
        v_lam = delta_w_d.unsqueeze(-1)                           # (B, H, 1)
        A_00 = cache_nu["A_00_k"]; A_01 = cache_mu["A_01_k"]
        A_10 = cache_nu["A_10_k"]; A_11 = A_k

        baseline = A_00 * v_bg                                    # (B, H, T_q)
        g_001 = A_00 * (v_bg + v_lam) - baseline
        g_010 = A_01 * v_bg - baseline
        g_011 = A_01 * (v_bg + v_lam) - baseline
        g_100 = A_10 * v_bg - baseline
        g_101 = A_10 * (v_bg + v_lam) - baseline
        g_110 = A_11 * v_bg - baseline
        g_111 = A_11 * (v_bg + v_lam) - baseline

        deltas = {
            "dQ":   g_100,
            "dK":   g_010,
            "dV":   g_001,
            "dQK":  g_110 - g_100 - g_010,
            "dQV":  g_101 - g_100 - g_001,
            "dKV":  g_011 - g_010 - g_001,
            "dQKV": g_111 - g_110 - g_101 - g_011 + g_100 + g_010 + g_001,
        }
        pm_e = pm.unsqueeze(1)                                    # (B, 1, T_q)
        for name, val in deltas.items():
            masked = val * pm_e                                    # (B, H, T_q)
            aggs[f"{name}_dep_sum"][t_global] += masked[dep_b].sum().double()
            aggs[f"{name}_cln_sum"][t_global] += masked[cln_b].sum().double()
            aggs[f"{name}_dep_abs"][t_global] += masked[dep_b].abs().sum().double()
            aggs[f"{name}_cln_abs"][t_global] += masked[cln_b].abs().sum().double()

    out: dict[str, torch.Tensor] = {"triplets": triplets.cpu()}
    for name in _DELTA_NAMES:
        out[f"{name}_dep_signed"] = (aggs[f"{name}_dep_sum"] / counts_dep).float().cpu()
        out[f"{name}_cln_signed"] = (aggs[f"{name}_cln_sum"] / counts_cln).float().cpu()
        out[f"{name}_dep_l1"] = (aggs[f"{name}_dep_abs"] / counts_dep).float().cpu()
        out[f"{name}_cln_l1"] = (aggs[f"{name}_cln_abs"] / counts_cln).float().cpu()
    return out


def generate_triplet_candidates(
    qk_q_score: torch.Tensor,
    qk_k_score: torch.Tensor,
    ov_score: torch.Tensor,
    Kq: int, Kk: int, Kv: int,
) -> torch.Tensor:
    """Top-Kq Q-features × top-Kk K-features × top-Kv V-features. (T, 3)."""
    top_q = torch.argsort(qk_q_score.abs(), descending=True)[:Kq]
    top_k = torch.argsort(qk_k_score.abs(), descending=True)[:Kk]
    top_v = torch.argsort(ov_score.abs(), descending=True)[:Kv]
    return torch.cartesian_prod(top_q, top_k, top_v)


def _triplet_score(agg: dict, score_kind: str) -> torch.Tensor:
    if score_kind == "qkv_l1":
        return agg["dQKV_dep_l1"]
    if score_kind == "qkv_signed":
        return agg["dQKV_dep_signed"].abs()
    if score_kind == "qkv_dep_minus_clean":
        return (agg["dQKV_dep_signed"] - agg["dQKV_cln_signed"]).abs()
    if score_kind == "full_v":
        return (agg["dV_dep_l1"] + agg["dQV_dep_l1"]
                + agg["dKV_dep_l1"] + agg["dQKV_dep_l1"])
    if score_kind == "phi_v":
        return (agg["dV_dep_l1"]
                + 0.5 * (agg["dQV_dep_l1"] + agg["dKV_dep_l1"])
                + (1.0 / 3.0) * agg["dQKV_dep_l1"])
    raise ValueError(f"unknown triple score_kind: {score_kind}")


def rank_triplets(
    agg: dict,
    top_k: int,
    score_kind: str = "qkv_l1",
) -> tuple[list[tuple[int, int, int]], list[dict]]:
    score = _triplet_score(agg, score_kind)
    order = torch.argsort(score, descending=True)[: top_k].tolist()
    triplets = agg["triplets"]
    selected = [(int(triplets[t, 0]), int(triplets[t, 1]), int(triplets[t, 2]))
                for t in order]
    records = [{
        "triplet": [int(triplets[t, 0]), int(triplets[t, 1]), int(triplets[t, 2])],
        "score": float(score[t]),
        **{f"{n}_dep_l1": float(agg[f"{n}_dep_l1"][t]) for n in _DELTA_NAMES},
        **{f"{n}_dep_signed": float(agg[f"{n}_dep_signed"][t]) for n in _DELTA_NAMES},
    } for t in order]
    return selected, records
