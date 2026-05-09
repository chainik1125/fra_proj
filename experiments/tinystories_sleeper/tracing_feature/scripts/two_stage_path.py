"""Phase 5: two-stage path tensor C_{h, k, a→λ→d} per the theory note §6.

Given:
  - d = SAE_mid.W_enc[:, mid_f]                     (target direction)
  - g_a = SAE_pre.W_dec[a]                          (resid_pre decoder rows)
  - e_λ = SAE_ln1.W_enc[:, λ]                       (ln1 encoder columns)
  - f_λ = SAE_ln1.W_dec[λ]                          (ln1 decoder rows)
  - u_h = W_V[0, h] W_O[0, h] d                     (OV read direction)
  - M_{ℓ, k} = (1/σ_k) (I - 11ᵀ/d)                  (frozen LN lin)
  - β[h, λ] = f_λ · u_h                             (OV write coefficient)
  - γ[a, λ] = centered(g_a) · e_λ                   (LN-lin transfer into post-LN feature)

The two-stage contribution is:
  C_{h, k, a→λ→d} = A_h[q, k] · z_pre[k, a] · T[k, λ, a] · ⟨f_λ W_OV^h, d⟩
                 = A_h[q, k] · z_pre[k, a] · γ[a, λ] / σ_k · β[h, λ]

Aggregated over (b ∈ deployment, q ∈ prompt(b), k):
  aggregate[h, a, λ] = β[h, λ] · γ[a, λ] · mean_{b, q} Σ_k A_h[q, k] · z_pre[k, a] / σ_k
                    = β[h, λ] · γ[a, λ] · q_dep[h, a]

where q_dep[h, a] is the same quantity already computed by pre_attn_path.

This script ranks (h, a, λ) triples on deployment prompts: "resid_pre feature a
routed through post-LN feature λ by head h into SAE_mid feature mid_f."
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder  # noqa: E402


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--pre_attn", default=str(HERE.parent / "results" / "pre_attn_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_pairs", type=int, default=40,
                        help="Top (h, a) pairs from pre_attn to expand.")
    parser.add_argument("--top_lambdas", type=int, default=8,
                        help="Top λ per expanded (h, a) pair.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mid_feature", type=int, default=None)
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[two-stage] loading inputs...")
    cache = torch.load(args.cache, weights_only=False)
    ov = torch.load(args.ov, weights_only=False)
    pre_attn = torch.load(args.pre_attn, weights_only=False)
    meta = cache["meta"]
    mid_f = args.mid_feature if args.mid_feature is not None else meta["suppressor"]["mid_feature"]

    if "ln1" not in meta["sae_configs"]:
        raise SystemExit("[two-stage] SAE_ln1 not available in cache — re-run cache_layer0_activations.py "
                         "after the ln1 SAE is trained.")

    # β[h, λ] from ov_path
    beta_T = ov["beta_T"]
    if beta_T is None:
        raise SystemExit("[two-stage] beta_T not computed in ov_path (SAE_ln1 likely missing then).")
    print(f"[two-stage]   β shape={tuple(beta_T.shape)}")

    # Load SAE_pre and SAE_ln1 to compute γ[a, λ].
    sae_pre, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["pre"]["path"], device=device)
    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)

    G = sae_pre.W_dec.detach().to(device).float()           # (d_sae_pre, d_model)
    G_centered = G - G.mean(dim=1, keepdim=True)            # (d_sae_pre, d_model)
    E_ln1 = sae_ln1.W_enc.detach().to(device).float()       # (d_model, d_sae_ln1)
    # γ[a, λ] = ⟨centered_g_a, e_λ⟩
    gamma = G_centered @ E_ln1                              # (d_sae_pre, d_sae_ln1)
    print(f"[two-stage]   γ shape={tuple(gamma.shape)}  "
          f"|γ|_mean={gamma.abs().mean().item():.3e}  |γ|_max={gamma.abs().max().item():.3e}")

    # q_dep[h, a] from pre_attn (= dep_contrib_pair / ψ[h, a], but already have it indirectly)
    # Actually we have dep_contrib_pair[h, a] = ψ[h, a] · q_dep[h, a], so q_dep = dep_contrib / ψ (where ψ ≠ 0).
    # Cleaner: just use dep_contrib_pair directly in the product:
    # aggregate[h, a, λ] = β[h, λ] · γ[a, λ] · q_dep[h, a]
    # But aggregate[h, a, λ] can also be written as:
    #   per-(h, a) [dep_contrib_pair / ψ[h, a]] · β[h, λ] · γ[a, λ]
    # Since dep_contrib_pair = ψ · q_dep, we have q_dep = dep_contrib_pair / ψ. This is just notation.
    psi = pre_attn["psi"]                                   # (n_heads, d_sae_pre)
    dep_contrib_pair = pre_attn["dep_contrib_pair"]         # (n_heads, d_sae_pre)
    # q_dep = dep_contrib / ψ where |ψ| > 0
    safe_psi = psi.clone()
    eps = 1e-10
    safe_psi[safe_psi.abs() < eps] = eps
    q_dep = dep_contrib_pair / safe_psi                     # (n_heads, d_sae_pre)

    # Rank top (h, a) pairs by |dep_contrib|
    flat = dep_contrib_pair.abs().flatten()
    top_idx = torch.argsort(flat, descending=True)[: args.top_pairs].tolist()
    n_heads, d_sae_pre = dep_contrib_pair.shape
    d_sae_ln1 = beta_T.shape[1]

    print(f"[two-stage] expanding top {args.top_pairs} (h, a) pairs into λ-dependent triples...")
    triples = []
    beta_T_host = beta_T.cpu() if beta_T.is_cuda else beta_T
    gamma_host = gamma.cpu()
    for flat_i in top_idx:
        h = flat_i // d_sae_pre
        a = flat_i % d_sae_pre
        # aggregate[λ] = β[h, λ] · γ[a, λ] · q_dep[h, a]
        agg_lambda = beta_T_host[h] * gamma_host[a] * q_dep[h, a].item()  # (d_sae_ln1,)
        top_l = torch.argsort(agg_lambda.abs(), descending=True)[: args.top_lambdas].tolist()
        for rank_l, lam in enumerate(top_l):
            triples.append({
                "head": int(h),
                "pre_feature_idx": int(a),
                "ln1_feature_idx": int(lam),
                "aggregate_dep_contrib": float(agg_lambda[lam].item()),
                "psi": float(psi[h, a].item()),
                "q_dep": float(q_dep[h, a].item()),
                "gamma": float(gamma_host[a, lam].item()),
                "beta": float(beta_T_host[h, lam].item()),
                "pair_dep_contrib": float(dep_contrib_pair[h, a].item()),
                "rank_lambda_within_pair": rank_l,
            })

    triples.sort(key=lambda r: abs(r["aggregate_dep_contrib"]), reverse=True)

    # Aggregate across (a, λ) per head, and across heads per (a, λ)
    from collections import defaultdict
    per_feat_lambda = defaultdict(float)   # (a, λ) -> summed across heads
    for r in triples:
        per_feat_lambda[(r["pre_feature_idx"], r["ln1_feature_idx"])] += r["aggregate_dep_contrib"]
    per_feat_lambda_top = sorted(per_feat_lambda.items(), key=lambda kv: abs(kv[1]), reverse=True)[:40]

    print(f"[two-stage] found {len(triples)} triples; top per-(a, λ) summed across heads:")
    for (a, lam), v in per_feat_lambda_top[:20]:
        print(f"[two-stage]   a={a:4d}  λ={lam:4d}  Σ_h aggregate={v:+.4e}")

    result = {
        "target": {"mid_feature": int(mid_f), "n_heads": int(n_heads),
                   "d_sae_pre": int(d_sae_pre), "d_sae_ln1": int(d_sae_ln1)},
        "sanity": {
            "gamma_abs_mean": float(gamma.abs().mean().item()),
            "gamma_abs_max": float(gamma.abs().max().item()),
        },
        "top_triples": triples[: args.top_pairs * args.top_lambdas],
        "top_per_pair_summed_across_heads": [
            {"pre_feature_idx": a, "ln1_feature_idx": lam, "total_aggregate": v}
            for (a, lam), v in per_feat_lambda_top
        ],
    }
    out_json = out_dir / "two_stage_path.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[two-stage] wrote {out_json}")


if __name__ == "__main__":
    main()
