"""Per-pair (mu, nu) QK-side contribution tensor.

For each top query-feature mu, computes the per-key-feature nu signed
aggregate contribution

    per_nu[mu, nu] = sum over (b, q in dep prompt, k in prompt(b), h)
                    of u^mu_q(b) · u^nu_k(b) · omega^h,QK_{mu, nu} · tilde_g^h_{q, k}(b)

via the factorisation

    Z^h(b, k) = sum_q u^mu_q(b) · tilde_g^h_{q, k}(b)            (per h, per prompt, per key)
    Y^h[nu]   = sum_{b, k in prompt(b)} u^nu_k(b) · Z^h(b, k)    (per head, per nu)
    per_nu    = sum_h omega^h_{mu, nu} · Y^h[nu]

which is O(N_heads * d_sae_ln1^2) per mu, manageable.

Outputs per-mu top-K nu's and their contribution.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import numpy as np

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402


def pick_device(explicit):
    if explicit:
        return explicit
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    p.add_argument("--output_dir", default=str(HERE.parent / "qk_vs_ov" / "results"))
    p.add_argument("--device", default=None)
    p.add_argument("--top_mus", nargs="+", type=int,
                   default=[254, 822, 760, 1298, 870, 784, 1279, 435, 1102, 1489, 303, 200, 1388, 691, 435])
    p.add_argument("--top_k", type=int, default=15)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[qk-pair] loading cache + weights...")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    ov = torch.load(HERE.parent / "results" / "ov_path_per_pair.pt", weights_only=False)

    hooks = cache["hooks"]
    z_ln1 = cache["encodings"]["z_ln1"]
    A = hooks["attn_pattern"].float()
    is_dep = cache["is_deployment"]
    marker = cache["story_marker_pos"]
    N, T, d_sae_ln1 = z_ln1.shape
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    scale = 1.0 / math.sqrt(d_head)

    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker.unsqueeze(1)           # (N, T)
    dep_query_mask = is_dep.unsqueeze(1) & prompt_mask # (N, T_q)

    from sleeper_utils import load_sleeper_model
    model = load_sleeper_model(device=device)
    W_Q = model.W_Q[0].detach().to(device).float()
    W_K = model.W_K[0].detach().to(device).float()

    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
    F = sae_ln1.W_dec.detach().to(device).float()  # (d_sae_ln1, d_model)

    # F_Q[h, mu] = F[mu] @ W_Q[h]   (n_heads, d_sae, d_head)
    F_Q = torch.stack([F @ W_Q[h] for h in range(n_heads)])
    F_K = torch.stack([F @ W_K[h] for h in range(n_heads)])

    # Build tilde_g
    beta = ov["beta"].to(device)                                     # (d_sae_ln1, n_heads)
    z_ln1_dev = z_ln1.to(device).float()
    A_dev = A.to(device)
    g = torch.einsum("bjv,vh->bhj", z_ln1_dev, beta)                 # (N, n_heads, T)
    g_bar = torch.einsum("bhqj,bhj->bhq", A_dev, g)                  # (N, n_heads, T_q)
    tilde_g = A_dev * (g.unsqueeze(2) - g_bar.unsqueeze(-1))         # (N, n_heads, T_q, T_k)
    print(f"[qk-pair]   tilde_g shape={tuple(tilde_g.shape)}")

    dep_query_mask_dev = dep_query_mask.to(device).float()            # (N, T_q)
    prompt_mask_dev = prompt_mask.to(device).float()                  # (N, T_k) as key mask within prompt
    dep_mask_dev = is_dep.unsqueeze(1).to(device).float() * prompt_mask_dev  # restrict key to dep prompt too

    results = {}
    for mu in args.top_mus:
        u_mu = z_ln1_dev[:, :, mu] * dep_query_mask_dev               # (N, T_q), zeroed off-prompt/off-dep
        # Z^h[b, k] = sum_q u^mu_q(b) · tilde_g^h_{q, k}(b)
        Z = torch.einsum("bq,bhqk->bhk", u_mu, tilde_g)               # (N, n_heads, T_k)
        # restrict key to prompt positions and dep prompts
        Z_dep = Z * dep_mask_dev.unsqueeze(1)                         # (N, n_heads, T_k)
        # Y^h[nu] = sum_{b, k} u^nu_k(b) · Z_dep^h[b, k]
        Y = torch.einsum("bhk,bkv->hv", Z_dep, z_ln1_dev)             # (n_heads, d_sae_ln1)
        # omega_mu[h, nu] = F_Q[h, mu] · F_K[h, nu] / sqrt(d_head)
        omega_mu = torch.einsum("hd,hvd->hv", F_Q[:, mu, :], F_K) * scale  # (n_heads, d_sae_ln1)
        per_nu = (omega_mu * Y).sum(dim=0).cpu()                      # (d_sae_ln1,)

        # Also per-head contribution for top nu's
        per_nu_per_head = (omega_mu * Y).cpu()                        # (n_heads, d_sae_ln1)

        top_idx = per_nu.abs().topk(args.top_k).indices.tolist()
        top_list = []
        for nu in top_idx:
            total = per_nu[nu].item()
            # find top head for this (mu, nu)
            head_contribs = per_nu_per_head[:, nu]
            top_h = head_contribs.abs().argmax().item()
            top_list.append({
                "nu": int(nu),
                "total_contrib": float(total),
                "top_head": int(top_h),
                "top_head_contrib": float(head_contribs[top_h].item()),
                "head_count_pos": int((head_contribs > 1e-5).sum().item()),
                "head_count_neg": int((head_contribs < -1e-5).sum().item()),
            })

        # Sanity: sum_nu per_nu should approximately equal the per-mu total
        # predicted delta from qk_concentration.py
        total_signed = per_nu.sum().item()
        total_l1 = per_nu.abs().sum().item()
        concentration = per_nu.abs().max().item() / (total_l1 + 1e-12)

        results[str(mu)] = {
            "mu": int(mu),
            "total_signed": float(total_signed),
            "total_l1": float(total_l1),
            "max_abs_per_nu": float(per_nu.abs().max().item()),
            "concentration_ratio_max_over_L1": float(concentration),
            "top_nu": top_list,
        }
        print(f"[qk-pair]  mu={mu:4d}  total_signed={total_signed:+.4f}  "
              f"L1={total_l1:.4f}  max/L1={concentration:.3f}  top_nu={top_list[0]['nu']}")
        if device == "cuda":
            torch.cuda.empty_cache()

    out_path = out_dir / "qk_pair_concentration.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"[qk-pair] wrote {out_path}")


if __name__ == "__main__":
    main()
