"""QK-side concentration analysis — the dual of two_stage_path.py.

Holds OV fixed (computes g^h_k = Σ_λ z_ln1[k, λ] · β_{h, λ}, then centers to
get tilde_g^h_{q, k} = A^h_{qk} (g^h_k - <g^h>_q)) and computes the first-order
score-side contribution of each query-feature μ (and each pair (μ, ν)) to the
target T_q = <attn_out_q, e_171>.

Reports:
  - Per-μ (query-side) predicted |δT_q| on deployment prompt positions.
    predicted_δT_q[μ] = Σ_h Σ_j u^μ_q · κ^{h,μ}_j · tilde_g^h_{q, j}
    κ^{h, μ}_j = Σ_ν u^ν_j · ω^{h, QK}_{μ, ν}
  - Per-μ concentration statistics: sum, L1, max over (h, q), topK.
  - Spearman correlation with measured |Δlogp| from ln1_feature_ablation.
  - Per-(μ, ν) pair: top pairs by the pair-level contribution tensor, summed
    and max-over-(h, q, k).

Outputs:
  results/qk_concentration.json
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
    p.add_argument("--output_dir", default=str(HERE.parent / "results"))
    p.add_argument("--device", default=None)
    p.add_argument("--top_pairs", type=int, default=30)
    args = p.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print("[qk-conc] loading cache + SAEs + model...")
    cache = torch.load(args.cache, weights_only=False)
    meta = cache["meta"]
    ov = torch.load(HERE.parent / "results" / "ov_path_per_pair.pt", weights_only=False)

    hooks = cache["hooks"]
    z_ln1 = cache["encodings"]["z_ln1"]          # (N, T, d_sae_ln1)
    attn_pattern = hooks["attn_pattern"].float() # (N, n_heads, T, T)
    is_dep = cache["is_deployment"]
    marker = cache["story_marker_pos"]
    N, T, d_sae_ln1 = z_ln1.shape
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    d_model = meta["d_model"]

    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker.unsqueeze(1)        # (N, T)
    dep_mask = is_dep.unsqueeze(1) & prompt_mask    # (N, T)

    # -----------------------------------------------------------------
    # Load model, get W_Q, W_K for layer 0
    # -----------------------------------------------------------------
    from sleeper_utils import load_sleeper_model
    model = load_sleeper_model(device=device)
    W_Q = model.W_Q[0].detach().to(device).float()   # (n_heads, d_model, d_head)
    W_K = model.W_K[0].detach().to(device).float()   # (n_heads, d_model, d_head)

    # SAE_ln1 decoder rows = feature directions f_λ (d_sae, d_model)
    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
    F_ln1 = sae_ln1.W_dec.detach().to(device).float()  # (d_sae_ln1, d_model)

    # -----------------------------------------------------------------
    # Precompute feature projections onto each head's Q and K
    # F_Q[h, μ, :] = f_μ · W_Q^h   (d_head-dim) — per head per feature
    # F_K[h, ν, :] = f_ν · W_K^h
    # -----------------------------------------------------------------
    F_Q = torch.einsum("md,hmk->hdk", F_ln1.T, W_Q)   # (n_heads, d_sae, d_head)
    # Simpler: F_Q[h, μ] = F_ln1[μ] @ W_Q[h]
    F_Q = torch.stack([F_ln1 @ W_Q[h] for h in range(n_heads)])  # (n_heads, d_sae, d_head)
    F_K = torch.stack([F_ln1 @ W_K[h] for h in range(n_heads)])  # (n_heads, d_sae, d_head)
    scale = 1.0 / math.sqrt(d_head)

    # -----------------------------------------------------------------
    # Compute per-head "centered OV profile" tilde_g[b, h, q, j]
    # g[b, h, j] = Σ_λ z_ln1[b, j, λ] · β[h, λ]   (per-source OV scalar, depends on b, h, j)
    # but we need ln1 activations to go through β — equivalently, re-use
    # per_head_total from ov_path_per_pair which is Σ_λ z_ln1 β · A summed over source.
    # Actually per_head_total[b, q, h] = Σ_s A_h[q, s] g^h_s = ḡ^h_q. So we have ḡ.
    # We still need g^h_j per (b, j). Compute from scratch:
    # -----------------------------------------------------------------
    # z_ln1: (N, T, d_sae_ln1), β: (d_sae_ln1, n_heads) = ov['beta']
    beta = ov["beta"].to(device)   # (d_sae_ln1, n_heads)
    # g[b, h, j] = Σ_λ z_ln1[b, j, λ] · β[λ, h]
    z_ln1_dev = z_ln1.to(device).float()
    g = torch.einsum("bjl,lh->bhj", z_ln1_dev, beta)   # (N, n_heads, T)

    # g_bar[b, h, q] = Σ_j A[b, h, q, j] · g[b, h, j]
    # = per_head_total (which was computed in ov_path).
    A = attn_pattern.to(device)     # (N, n_heads, T, T)
    g_bar = torch.einsum("bhqj,bhj->bhq", A, g)  # (N, n_heads, T)

    # tilde_g[b, h, q, j] = A[b, h, q, j] · (g[b, h, j] - g_bar[b, h, q])
    tilde_g = A * (g.unsqueeze(2) - g_bar.unsqueeze(-1))  # (N, n_heads, T_q, T_k)
    print(f"[qk-conc]   g shape={tuple(g.shape)}  tilde_g shape={tuple(tilde_g.shape)}  max|tilde_g|={tilde_g.abs().max().item():.3e}")

    # Sanity: sum_j tilde_g[b, h, q, j] should be 0
    residual = tilde_g.sum(dim=-1).abs().max().item()
    print(f"[qk-conc]   Σ_j tilde_g max |residual| = {residual:.3e}  (should be ~0)")

    # -----------------------------------------------------------------
    # Per-μ predicted δT_q via query-feature ablation (first-order):
    # δT_q^{μ} = u^μ_q · Σ_h Σ_j κ^{h,μ}_j · tilde_g^h_{q, j}
    # where κ^{h, μ}_j = Σ_ν u^ν_j · ω^{h, QK}_{μ, ν}
    #       ω^{h, QK}_{μ, ν} = F_Q[h, μ] · F_K[h, ν] / sqrt(d_head)
    #
    # κ^{h, μ}_j = Σ_ν u^ν_j · (F_Q[h, μ] · F_K[h, ν]) / sqrt(d_head)
    #            = F_Q[h, μ] · (Σ_ν u^ν_j · F_K[h, ν]) / sqrt(d_head)
    #            = F_Q[h, μ] · (z_ln1[j] @ F_K[h]) / sqrt(d_head)
    #
    # Let r^h_j := z_ln1[j] @ F_K[h]   (N, T, n_heads, d_head), i.e. (N, n_heads, T, d_head)
    # Then κ^{h, μ}_j = F_Q[h, μ] · r^h_j / sqrt(d_head).
    #
    # For each μ:
    #     S_per_prompt[b, q, μ] = Σ_h Σ_j tilde_g[b, h, q, j] · (F_Q[h, μ] · r^h_j / sqrt(d_head))
    #                           = Σ_h (F_Q[h, μ] / sqrt(d_head)) · Σ_j tilde_g[b, h, q, j] · r^h_j
    # Define T_aggr[b, h, q, :] := Σ_j tilde_g[b, h, q, j] · r^h_j[b, h, j, :]  (N, n_heads, T_q, d_head)
    # Then S_per_prompt[b, q, μ] = Σ_h F_Q[h, μ] · T_aggr[b, h, q] / sqrt(d_head)
    # -----------------------------------------------------------------
    print("[qk-conc] computing κ support r[b, h, j, :] = z_ln1[b, j] @ F_K[h]...")
    # r[b, h, j, d] = Σ_ν z_ln1[b, j, ν] · F_K[h, ν, d]
    r = torch.einsum("bjv,hvd->bhjd", z_ln1_dev, F_K)  # (N, n_heads, T, d_head)
    print(f"[qk-conc]   r shape={tuple(r.shape)}  size={r.element_size()*r.numel()/1e9:.2f} GB")

    # T_aggr[b, h, q, d] = Σ_j tilde_g[b, h, q, j] · r[b, h, j, d]
    print("[qk-conc] computing T_aggr[b, h, q, d] = Σ_j tilde_g · r ...")
    T_aggr = torch.einsum("bhqj,bhjd->bhqd", tilde_g, r)   # (N, n_heads, T_q, d_head)
    # For memory, we can free r here if needed
    del r
    if device == "cuda":
        torch.cuda.empty_cache()
    print(f"[qk-conc]   T_aggr shape={tuple(T_aggr.shape)}  size={T_aggr.element_size()*T_aggr.numel()/1e9:.2f} GB")

    # Now per-μ score at (b, q): S_prompt[b, q, μ] = Σ_h (F_Q[h, μ] · T_aggr[b, h, q]) · scale
    # That's a big tensor (N, T, d_sae_ln1); compute only dep prompt positions.
    # Flatten dep prompt positions into a (M, d) indexing.
    # dep_idx_bq = (b, q) pairs where dep_mask is True.
    dep_bq = dep_mask.nonzero(as_tuple=False)   # (M, 2): cols = b, q
    M = dep_bq.shape[0]
    print(f"[qk-conc] dep prompt positions M = {M}")

    # Need: per-μ score for each (b, q) in dep.
    # S[m, μ] = Σ_h F_Q[h, μ] · T_aggr[b_m, h, q_m] · scale
    # T_aggr_sel[m, h, d] = T_aggr[b_m, h, q_m, d]   (M, n_heads, d_head)
    b_idx = dep_bq[:, 0]; q_idx = dep_bq[:, 1]
    T_aggr_sel = T_aggr[b_idx, :, q_idx, :]  # (M, n_heads, d_head)
    # S[m, μ] = Σ_h Σ_d F_Q[h, μ, d] · T_aggr_sel[m, h, d] · scale
    # = Σ_h (F_Q[h, μ] dot T_aggr_sel[m, h]) · scale
    # F_Q shape (n_heads, d_sae_ln1, d_head); T_aggr_sel (M, n_heads, d_head).
    # S = einsum('hud,mhd->mu', F_Q, T_aggr_sel) * scale
    print(f"[qk-conc] computing per-μ score S[m, μ] over {M} dep positions...")
    S = torch.einsum("hud,mhd->mu", F_Q, T_aggr_sel) * scale   # (M, d_sae_ln1)
    print(f"[qk-conc]   S shape={tuple(S.shape)}")

    # u^μ_q at these positions
    U_dep = z_ln1_dev[b_idx, q_idx, :]  # (M, d_sae_ln1)

    # Per-μ predicted δT_q at each (b, q) when ablating μ:  δT_q = -u^μ_q · S[m, μ]
    # We want ablation effect on sleeper logp, but as a first-order proxy:
    # contribution[m, μ] = u^μ_q · S[m, μ]   (so ablation removes this)
    contribution = U_dep * S   # (M, d_sae_ln1)

    # Aggregate per-μ.
    signed_mean = contribution.mean(dim=0)     # (d_sae_ln1,)
    abs_mean = contribution.abs().mean(dim=0)
    max_abs = contribution.abs().max(dim=0).values
    # top-K over samples (concentration in a specific prompt position)
    topK = min(50, M)
    topK_sum = contribution.abs().topk(topK, dim=0).values.sum(dim=0)

    methods = {
        "signed_mean (≈ one-stage QK)": signed_mean.abs(),
        "L1 mean": abs_mean,
        "max over (b, q) [concentration]": max_abs,
        f"top-{topK} over (b, q) L1": topK_sum,
    }

    # -----------------------------------------------------------------
    # Spearman correlation with measured |Δlogp| on the 15 features tested
    # via ln1_feature_ablation.
    # -----------------------------------------------------------------
    with open(HERE.parent / "results" / "ln1_feature_ablation.json") as f:
        abl = json.load(f)
    measured = {}
    for r in abl["results"]:
        if r["alpha"] == 4.0 and isinstance(r["ln1_feature"], int):
            measured[r["ln1_feature"]] = r["delta_logp"]
    feats = sorted(measured.keys())
    dlogp = np.array([abs(measured[f]) for f in feats])
    from scipy.stats import spearmanr

    print("\n[qk-conc] Spearman ρ between QK-concentration and |Δlogp| (10 features):")
    corr_results = {}
    for name, scores in methods.items():
        vec = np.array([scores[f].item() for f in feats])
        rho = spearmanr(vec, dlogp).correlation
        corr_results[name] = rho
        print(f"  {name:>40s}: ρ = {rho:+.3f}")

    print("\n[qk-conc] Per-feature QK-side ranks:")
    header = f"{'feat':>6} {'|Δlogp|':>9}"
    for name in methods:
        header += f" {name[:20]:>22}"
    print(header)
    for f in sorted(feats, key=lambda x: -abs(measured[x])):
        line = f"{f:>6d} {abs(measured[f]):>9.2f}"
        for name, scores in methods.items():
            rank = (scores >= scores[f]).sum().item()
            line += f" {rank:>22d}"
        print(line)

    # -----------------------------------------------------------------
    # Top (μ, ν) pairs by observed aggregated contribution (stage-2 dual):
    # This requires expanding ω · u^μ_q · u^ν_k · tilde_g; full 4D tensor is
    # (d_sae_ln1, d_sae_ln1) per (b, h, q, k) — too large. Instead aggregate
    # over dep (b, q) for top μ's that pass concentration test, then find
    # top ν's per μ.
    # -----------------------------------------------------------------
    # Pick the top-20 μ by max_abs (concentration)
    top_mu_idx = max_abs.topk(20).indices.tolist()
    print(f"\n[qk-conc] top-20 μ by concentration (max over b, q of |contribution|):")
    for mu in top_mu_idx:
        print(f"  μ={mu:4d}  max_abs={max_abs[mu].item():+.4e}  signed_mean={signed_mean[mu].item():+.4e}")

    # -----------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------
    result = {
        "meta": {"M_dep_positions": int(M), "n_heads": int(n_heads), "d_sae_ln1": int(d_sae_ln1)},
        "spearman": {name: float(corr_results[name]) for name in methods},
        "top_mu_by_concentration": [
            {"mu": int(mu), "max_abs": float(max_abs[mu].item()),
             "signed_mean": float(signed_mean[mu].item()),
             "abs_mean": float(abs_mean[mu].item())}
            for mu in top_mu_idx
        ],
        "per_feature_ranks": {
            str(f): {
                "delta_logp": measured[f],
                "abs_delta_logp": abs(measured[f]),
                **{name.replace(" ", "_"): int((scores >= scores[f]).sum().item())
                   for name, scores in methods.items()},
            }
            for f in feats
        },
    }
    out_path = out_dir / "qk_concentration.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"[qk-conc] wrote {out_path}")


if __name__ == "__main__":
    main()
