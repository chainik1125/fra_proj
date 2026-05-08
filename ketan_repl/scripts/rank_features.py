"""Self-contained per-seed feature ranking.

Computes:
  - OV one-stage signed sum:  S_λ = Σ_h β_{h,λ} · E_t[Σ_s A^h_{ts} z^λ_s]
  - QK L1 mean:                pred[μ](b, q) = u^μ_q · Σ_h Σ_j κ^{h,μ}_j · tilde_g^h_{q, j}

…and emits a `features.json` with {qk: [...], ov: [...], union: [...]} consumable
by the modified pareto_3x3.py via --features_json.

Inputs: a layer0_cache.pt produced by cache_layer0_activations.py (must include
encodings.z_ln1 and hooks.attn_pattern). Loads the sleeper model + the ln1 SAE
referenced by cache.meta.sae_configs["ln1"]["path"].

Usage:
    python rank_features.py --cache PATH --output features.json [--top_k 3]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

THIS = Path(__file__).resolve().parent
EXP = THIS.parent.parent / "experiments" / "tinystories_sleeper"
sys.path.insert(0, str(EXP))


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--top_k", type=int, default=3)
    p.add_argument("--device", default=None)
    args = p.parse_args()

    from run_ablation_sweep import load_crosscoder
    from sleeper_utils import load_sleeper_model

    device = pick_device(args.device)
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    d_model = meta["d_model"]

    z_ln1 = cache["encodings"]["z_ln1"].float()             # (N, T, d_sae_ln1) on CPU
    attn_pattern = cache["hooks"]["attn_pattern"].float()   # (N, n_heads, T, T) on CPU
    is_dep = cache["is_deployment"]                         # (N,) bool
    marker = cache["story_marker_pos"]                      # (N,)
    N, T, d_sae_ln1 = z_ln1.shape

    print(f"[rank] N={N} T={T} d_sae_ln1={d_sae_ln1} n_heads={n_heads} d_head={d_head}")

    # Mid feature 171 (the suppressor) — its decoder direction in resid_mid space.
    sae_mid_path = EXP / meta["sae_configs"]["mid"]["path"]
    sae_ln1_path = EXP / meta["sae_configs"]["ln1"]["path"]
    sae_mid, _ = load_crosscoder(sae_mid_path, device="cpu")
    sae_ln1, _ = load_crosscoder(sae_ln1_path, device="cpu")
    mid_f = int(meta["suppressor"]["mid_feature"])
    e = sae_mid.W_enc[:, mid_f].float()                     # (d_model,)

    # Load model to get W_V, W_O, W_Q, W_K at layer 0.
    model = load_sleeper_model(device=device)
    W_V0 = model.W_V[0].detach().to("cpu", torch.float32)   # (n_heads, d_model, d_head)
    W_O0 = model.W_O[0].detach().to("cpu", torch.float32)   # (n_heads, d_head, d_model)
    W_Q0 = model.W_Q[0].detach().to("cpu", torch.float32)   # (n_heads, d_model, d_head)
    W_K0 = model.W_K[0].detach().to("cpu", torch.float32)   # (n_heads, d_model, d_head)

    # u_h = W_V_h @ W_O_h @ e -> (n_heads, d_model)
    woe = torch.einsum("hde,e->hd", W_O0, e)                # (n_heads, d_head)
    U = torch.einsum("hdk,hk->hd", W_V0, woe)               # (n_heads, d_model)

    # β_{h, λ} = <f_λ, u_h>; W_dec_ln1[λ] is the decoder direction of feature λ.
    W_dec_ln1 = sae_ln1.W_dec.detach().float()              # (d_sae_ln1, d_model)
    beta = torch.einsum("ld,hd->lh", W_dec_ln1, U)          # (d_sae_ln1, n_heads)
    beta_T = beta.T.contiguous()                            # (n_heads, d_sae_ln1)

    # ------------------------------------------------------------------
    # OV one-stage signed sum:
    #   M_{h, λ}(t) = Σ_s A^h_{t,s} z_ln1[b, s, λ]
    #   S_λ        = E_dep [ Σ_h β_{h,λ} · M_{h, λ}(t) ]
    # We avoid materializing per-feature M tensors — instead chunk over features.
    # ------------------------------------------------------------------
    print("[rank] computing OV one-stage signed sum...")
    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker.unsqueeze(1)                # (N, T)
    dep_mask = is_dep.unsqueeze(1) & prompt_mask            # (N, T) bool

    # We accumulate Σ_t∈dep [Σ_h β_h,λ · (A^h ⋅ z[·, λ])(t)] over (b, t).
    # Implemented per-head, per-feature block to fit memory.
    s_lambda = torch.zeros(d_sae_ln1)
    block = 256
    A = attn_pattern                                        # (N, H, T_q, T_k)
    z = z_ln1                                               # (N, T_k, d_sae)
    for f0 in range(0, d_sae_ln1, block):
        f1 = min(f0 + block, d_sae_ln1)
        z_blk = z[..., f0:f1]                               # (N, T_k, B)
        # M[b, h, t_q, λ] = Σ_s A[b, h, t_q, s] · z[b, s, λ]
        M = torch.einsum("bhqs,bsl->bhql", A, z_blk)        # (N, H, T_q, B)
        # Multiply by β_h,λ and sum over heads.
        # contribution[b, t_q, λ] = Σ_h β[h, λ] · M[b, h, t_q, λ]
        beta_blk = beta_T[:, f0:f1]                         # (H, B)
        contrib = torch.einsum("bhql,hl->bql", M, beta_blk) # (N, T_q, B)
        # Mask to dep prompt positions and average.
        # dep_mask: (N, T_q). expand to (N, T_q, 1)
        masked = contrib * dep_mask.unsqueeze(-1).float()
        denom = dep_mask.float().sum().clamp_min(1.0)
        s_lambda[f0:f1] = masked.sum(dim=(0, 1)) / denom

    top_ov_idx = s_lambda.abs().argsort(descending=True)[: args.top_k].tolist()
    top_ov = [int(i) for i in top_ov_idx]
    print(f"[rank] OV top-{args.top_k}: {top_ov}  (S_λ values: {[f'{s_lambda[i].item():+.4e}' for i in top_ov]})")

    # ------------------------------------------------------------------
    # QK L1-mean (the metric Ketan reports as ρ=+0.95 stable).
    #   ω^{h, QK}_{μ, ν} = (f_μ W_Q^h) · (f_ν W_K^h) / sqrt(d_head)
    #   κ^{h, μ}_j      = Σ_ν u^ν_j · ω^{h, QK}_{μ, ν}      where u^ν_j = z_ln1[b, k=j, ν]
    #   Σ_h Σ_j κ^{h, μ}_j · tilde_g^h_{q, j}
    #
    # We avoid the full d_sae × d_sae ω; collapse via:
    #   F_Q[h, μ, d] = (W_dec_ln1[μ] @ W_Q^h)[d]   (n_heads, d_sae, d_head)
    #   F_K[h, ν, d] = (W_dec_ln1[ν] @ W_K^h)[d]
    #   Σ_ν z[b, k, ν] · F_K[h, ν, d] = (z @ F_K^T)[b, k, h, d]   (closed-form)
    # so κ^{h, μ}_j summed × tilde_g aggregates cleanly per μ.
    # ------------------------------------------------------------------
    print("[rank] computing QK L1-mean ranking...")
    scale = 1.0 / (d_head ** 0.5)

    # F_Q[h, μ, d] and F_K[h, ν, d]
    F_Q = torch.einsum("ld,hde->hle", W_dec_ln1, W_Q0)      # (H, d_sae, d_head)
    F_K = torch.einsum("ld,hde->hle", W_dec_ln1, W_K0)      # (H, d_sae, d_head)

    # tilde_g^h_{q, k} = A^h_{q,k} (g^h_k - <g^h>_q)
    # g^h_k = Σ_λ z[b, k, λ] · β_{h, λ}
    # We need this per-prompt; build (N, H, T_k) g, then center along T_q.
    g = torch.einsum("bsl,hl->bhs", z, beta_T)              # (N, H, T_k)
    # mean over T_k weighted by A^h_{q,·}: <g^h>_q = Σ_k A^h_{q,k} g^h_k
    g_mean = torch.einsum("bhqs,bhs->bhq", A, g)            # (N, H, T_q)
    # tilde_g[b, h, q, k] = A[b, h, q, k] * (g[b, h, k] - g_mean[b, h, q])
    tilde_g = A * (g.unsqueeze(2) - g_mean.unsqueeze(-1))   # (N, H, T_q, T_k)

    # T_aggr[b, h, q, d] = Σ_k tilde_g[b, h, q, k] · (Σ_ν z[b, k, ν] F_K[h, ν, d])
    #                    = Σ_k tilde_g[b, h, q, k] · z[b, k, ·] @ F_K[h, ·, d]
    # First: zFK[b, h, k, d] = (z[b, k] @ F_K[h])[d]  -- per-(b, h, k) in d_head.
    zFK = torch.einsum("bsl,hld->bhsd", z, F_K)             # (N, H, T_k, d_head)
    # T_aggr[b, h, q, d] = Σ_k tilde_g[b, h, q, k] · zFK[b, h, k, d]
    T_aggr = torch.einsum("bhqs,bhsd->bhqd", tilde_g, zFK)  # (N, H, T_q, d_head)

    # contribution[b, q, μ] = u^μ_q · Σ_h F_Q[h, μ] · T_aggr[b, h, q] · scale
    #                       = z[b, q, μ] · Σ_h <F_Q[h, μ], T_aggr[b, h, q]> · scale
    # Σ_h dot product first: cross[b, q, μ] = Σ_h Σ_d F_Q[h, μ, d] · T_aggr[b, h, q, d]
    cross = torch.einsum("hud,bhqd->bqu", F_Q, T_aggr)      # (N, T_q, d_sae)
    contribution = z * cross * scale                         # (N, T_q, d_sae)

    # Restrict to dep prompt positions, take L1-mean over those positions.
    contrib_dep = contribution * dep_mask.unsqueeze(-1).float()
    denom = dep_mask.float().sum().clamp_min(1.0)
    qk_l1 = contrib_dep.abs().sum(dim=(0, 1)) / denom        # (d_sae,)

    top_qk_idx = qk_l1.argsort(descending=True)[: args.top_k].tolist()
    top_qk = [int(i) for i in top_qk_idx]
    print(f"[rank] QK top-{args.top_k} (L1-mean): {top_qk}  (L1: {[f'{qk_l1[i].item():.4e}' for i in top_qk]})")

    # ------------------------------------------------------------------
    # Emit features.json
    # ------------------------------------------------------------------
    union = sorted(set(top_ov) | set(top_qk))
    out = {
        "ov":    top_ov,
        "qk":    top_qk,
        "union": union,
        "_provenance": {
            "cache": str(args.cache),
            "top_k": args.top_k,
            "ov_score_top": [float(s_lambda[i].item()) for i in top_ov],
            "qk_l1_top":     [float(qk_l1[i].item())   for i in top_qk],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out, indent=2))
    print(f"[rank] wrote {args.output}")


if __name__ == "__main__":
    main()
