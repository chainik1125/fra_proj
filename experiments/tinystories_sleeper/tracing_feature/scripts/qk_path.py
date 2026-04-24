"""Phase 6: QK-circuit decomposition for the heads with strong dep-specific
attention shifts. Implements §3.2 / §3.3 of the theory note.

For head h, the pre-softmax attention score is
    s_{qk}^h = (x_q W_Q^h) · (x_k W_K^h) / sqrt(d_head)

With the SAE_ln1 basis x_t = Σ_λ u^λ_t f_λ + err, the exact feature-pair
decomposition on scores is

    S_{qk}^{h, μν} = u^μ_q · u^ν_k · ω_{μν}^{h, QK}
    ω_{μν}^{h, QK} = (f_μ W_Q^h) · (f_ν W_K^h) / sqrt(d_head)

We rank (μ, ν) pairs by their total contribution to the attention score from
deployment prompt destinations to a target source position (e.g. k=2, the ` |`
opening of the trigger).

Outputs:
    tracing_feature/results/qk_path_h{head}.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch

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
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument("--head", type=int, required=True)
    p.add_argument("--source_pos", type=int, default=2,
                   help="Source position to attribute attention to (default: t=2, ` |`).")
    p.add_argument("--top_k", type=int, default=30, help="Top pairs to report.")
    p.add_argument("--device", default=None)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[qk] head={args.head} source_pos={args.source_pos}")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    d_model = meta["d_model"]

    if "ln1" not in meta["sae_configs"]:
        raise SystemExit("[qk] SAE_ln1 not in cache — rerun cache_layer0_activations with SAE_ln1 present.")

    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
    # Need the model for W_Q, W_K
    from sleeper_utils import load_sleeper_model  # noqa: E402
    model = load_sleeper_model(device=device)
    W_Q = model.W_Q[0, args.head].detach().to(device).float()   # (d_model, d_head)
    W_K = model.W_K[0, args.head].detach().to(device).float()   # (d_model, d_head)
    b_Q = model.b_Q[0, args.head].detach().to(device).float() if model.b_Q is not None else None
    b_K = model.b_K[0, args.head].detach().to(device).float() if model.b_K is not None else None

    # Feature Q and K projections: Q_λ = f_λ · W_Q, K_λ = f_λ · W_K
    F = sae_ln1.W_dec.detach().to(device).float()     # (d_sae_ln1, d_model)
    d_sae_ln1 = F.shape[0]
    Q_feat = F @ W_Q                                   # (d_sae_ln1, d_head)
    K_feat = F @ W_K                                   # (d_sae_ln1, d_head)

    # ω_{μν} = Q_μ · K_ν / sqrt(d_head) — a (d_sae_ln1, d_sae_ln1) matrix
    scale = 1.0 / math.sqrt(d_head)
    # Compute only the product with a cap on sparsity later; full matrix is 1536x1536 = 9.4MB, fine.
    Omega = (Q_feat @ K_feat.T) * scale                # (d_sae_ln1, d_sae_ln1)
    print(f"[qk] Ω shape={tuple(Omega.shape)}  |Ω|_mean={Omega.abs().mean().item():.3e}  "
          f"|Ω|_max={Omega.abs().max().item():.3e}")

    # Load ln1 SAE encodings (post-TopK z_ln1): (N, T, d_sae_ln1)
    z_ln1 = cache["encodings"]["z_ln1"]
    is_deploy = cache["is_deployment"]
    marker = cache["story_marker_pos"]

    N, T, _ = z_ln1.shape
    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker.unsqueeze(1)
    dep_mask = is_deploy.unsqueeze(1) & prompt_mask
    cln_mask = (~is_deploy).unsqueeze(1) & prompt_mask

    # u^μ_q averaged over deployment query positions (prompt)
    u_q_mean_dep = z_ln1[dep_mask].mean(dim=0).to(device)    # (d_sae_ln1,)
    u_q_mean_cln = z_ln1[cln_mask].mean(dim=0).to(device)

    # u^ν_{k=source_pos} averaged over deployment prompts
    dep_idx = torch.where(is_deploy)[0]
    cln_idx = torch.where(~is_deploy)[0]
    u_k_dep = z_ln1[dep_idx, args.source_pos, :].mean(dim=0).to(device)    # (d_sae_ln1,)
    u_k_cln = z_ln1[cln_idx, args.source_pos, :].mean(dim=0).to(device)

    # Pair contribution at q̄ (mean dep query pos) → k=source_pos:
    # S_{μν}^{dep} ≈ u_q_mean_dep[μ] · u_k_dep[ν] · Omega[μ, ν]
    S_dep = u_q_mean_dep.unsqueeze(-1) * u_k_dep.unsqueeze(0) * Omega
    S_cln = u_q_mean_cln.unsqueeze(-1) * u_k_cln.unsqueeze(0) * Omega
    S_diff = S_dep - S_cln

    print(f"[qk] S_dep shape={tuple(S_dep.shape)}  |S|_max={S_dep.abs().max().item():.3e}")
    print(f"[qk] Σ S_dep (total score from dep prompts, q̄ → k={args.source_pos}) = {S_dep.sum().item():+.3e}")
    print(f"[qk] Σ S_cln = {S_cln.sum().item():+.3e}")
    print(f"[qk] Σ (S_dep - S_cln) = {S_diff.sum().item():+.3e}")

    def _topk_pairs(M, k):
        flat = M.abs().flatten()
        idx_ = torch.argsort(flat, descending=True)[:k]
        return [
            {"rank": i, "mu": int(idx_[i].item() // d_sae_ln1),
             "nu": int(idx_[i].item() % d_sae_ln1),
             "value": float(M[idx_[i] // d_sae_ln1, idx_[i] % d_sae_ln1].item())}
            for i in range(len(idx_))
        ]

    rankings = {
        "S_dep": _topk_pairs(S_dep.cpu(), args.top_k),
        "S_diff_dep_minus_cln": _topk_pairs(S_diff.cpu(), args.top_k),
    }

    # Also: top μ by sum over ν (query-side saliency) and top ν by sum over μ (key-side).
    S_dep_cpu = S_dep.cpu()
    S_diff_cpu = S_diff.cpu()

    mu_saliency = S_diff_cpu.sum(dim=1)          # sum over ν
    nu_saliency = S_diff_cpu.sum(dim=0)          # sum over μ
    mu_top = torch.argsort(mu_saliency.abs(), descending=True)[: args.top_k].tolist()
    nu_top = torch.argsort(nu_saliency.abs(), descending=True)[: args.top_k].tolist()

    result = {
        "head": int(args.head),
        "source_pos": int(args.source_pos),
        "d_sae_ln1": int(d_sae_ln1),
        "sanity": {
            "Omega_abs_mean": float(Omega.abs().mean().item()),
            "Omega_abs_max": float(Omega.abs().max().item()),
            "S_dep_sum": float(S_dep.sum().item()),
            "S_cln_sum": float(S_cln.sum().item()),
            "S_diff_sum": float(S_diff.sum().item()),
        },
        "rankings": rankings,
        "top_mu_query_saliency_diff": [
            {"feature": m, "signed_sum": float(mu_saliency[m].item())} for m in mu_top
        ],
        "top_nu_key_saliency_diff": [
            {"feature": n, "signed_sum": float(nu_saliency[n].item())} for n in nu_top
        ],
    }
    out_path = out_dir / f"qk_path_h{args.head}_src{args.source_pos}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[qk] wrote {out_path}")


if __name__ == "__main__":
    main()
