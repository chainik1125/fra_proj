"""Phase 2b: resid_pre -> attention -> resid_mid attribution via LN linearization.

Implements the C^{pre,attn} object from the theory note
(notes/feature_resolved_attention_theory_and_simplification.md, §5.2):

    C^{pre,attn}_{h, k, a}(d) = A_h[q, k] · z_pre[k, a] · ⟨g_a M_{ℓ,k} W_OV^h, d⟩

where:
    d        = SAE_mid.W_enc[:, mid_f]                       (target encoder dir)
    g_a      = SAE_pre.W_dec[a]                              (resid_pre decoder)
    z_pre    = SAE_pre.encode(resid_pre)                     (feature activations)
    W_OV^h   = W_V[0, h] · W_O[0, h]
    M_{ℓ,k}  = (1 / σ_k) · (I - 11^T / d_model)              (frozen LN linearization)

Using the centering projector P = I - 11^T/d_model, g_a · P = (g_a - mean(g_a)) =: g_a^c
(centered decoder row). Define the per-(head, resid_pre feature) transfer scalar

    ψ[h, a] := g_a^c · u_h          where u_h := W_V[0,h] W_O[0,h] d

Then the attribution collapses to

    C^{pre,attn}_{h, k, a}(d) = A_h[q, k] · z_pre[k, a] · ψ[h, a] / σ_k.

This script computes ψ, per-(h, a) dep-vs-clean contributions on deployment
prompts, ranks heads × resid_pre features, and verifies sum-over-a reproduces
S_h(q) - (b_dec_pre LN-route constant) up to SAE_pre reconstruction error.

Outputs:
  tracing_feature/results/pre_attn_path.json          — rankings + sanity
  tracing_feature/results/pre_attn_path_per_pair.pt   — ψ, per-pair stats
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
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--top_triples", type=int, default=50)
    parser.add_argument("--top_heads", type=int, default=5,
                        help="Number of heads to expand to per-source triples.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mid_feature", type=int, default=None)
    parser.add_argument("--pre_feature", type=int, default=None,
                        help="SAE_pre feature to flag in rankings.")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[pre-attn] loading cache {args.cache}...")
    cache = torch.load(args.cache, weights_only=False)
    ov_data = torch.load(args.ov, weights_only=False)
    meta = cache["meta"]
    mid_f = args.mid_feature if args.mid_feature is not None else meta["suppressor"]["mid_feature"]
    pre_f_flag = args.pre_feature if args.pre_feature is not None else meta["suppressor"]["pre_feature"]
    n_heads = meta["n_heads"]
    d_model = meta["d_model"]
    print(f"[pre-attn] mid_feature={mid_f}  pre_suppressor_to_flag={pre_f_flag}  "
          f"n_heads={n_heads} d_model={d_model}")

    # ------------------------------------------------------------------
    # 1) u_h  = W_V[0,h] · W_O[0,h] · e_171   (d_model vector per head)
    # Already precomputed by ov_path.py.
    # ------------------------------------------------------------------
    U = ov_data["U"].to(device).float()                         # (n_heads, d_model)
    e = ov_data["e"].to(device).float()                         # (d_model,)

    # ------------------------------------------------------------------
    # 2) ψ[h, a] = centered(g_a) · u_h
    # ------------------------------------------------------------------
    print(f"[pre-attn] loading SAE_pre for decoder rows g_a...")
    sae_pre, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["pre"]["path"], device=device)
    G = sae_pre.W_dec.detach().to(device).float()               # (d_sae_pre, d_model)
    G_centered = G - G.mean(dim=1, keepdim=True)                # (d_sae_pre, d_model)
    b_dec_pre = sae_pre.b_dec.detach().to(device).float()       # (d_model,)
    b_dec_pre_centered = b_dec_pre - b_dec_pre.mean()           # (d_model,)
    psi = G_centered @ U.T                                      # (d_sae_pre, n_heads)
    psi_T = psi.T.contiguous()                                  # (n_heads, d_sae_pre)
    print(f"[pre-attn]   ψ shape={tuple(psi.shape)}  "
          f"|ψ|_mean={psi.abs().mean().item():.3e}  |ψ|_max={psi.abs().max().item():.3e}")

    # b_dec_pre LN-routed constant per-head
    # ⟨b_dec_pre · M_{ℓ,k} W_OV^h, d⟩ = (centered_b_dec_pre · u_h) / σ_k
    b_dec_pre_route = (b_dec_pre_centered.unsqueeze(0) @ U.T).squeeze()  # (n_heads,) * 1/σ_k
    print(f"[pre-attn]   b_dec_pre_route (pre-division by σ_k) per-head mean={b_dec_pre_route.abs().mean().item():.3e}")

    # ------------------------------------------------------------------
    # 3) σ_k per (prompt, position).  σ_k = ||centered_pre_k|| / ||ln1_normalized_k||
    # (LN with fold_ln=True, folded γ absorbed into W_Q/K/V)
    # ------------------------------------------------------------------
    resid_pre = cache["hooks"]["resid_pre"].float()             # (N, T, d_model)
    ln1_norm = cache["hooks"]["ln1_normalized"].float()         # (N, T, d_model)
    centered_pre = resid_pre - resid_pre.mean(dim=-1, keepdim=True)
    sigma = centered_pre.norm(dim=-1) / ln1_norm.norm(dim=-1).clamp(min=1e-8)
    print(f"[pre-attn]   σ stats: mean={sigma.mean().item():.3e} "
          f"min={sigma.min().item():.3e} max={sigma.max().item():.3e}")

    # Sanity: reconstruct ln1_normalized = centered_pre / σ
    ln1_recon = centered_pre / sigma.unsqueeze(-1).clamp(min=1e-8)
    ln1_err = (ln1_recon - ln1_norm).norm() / ln1_norm.norm()
    print(f"[pre-attn]   LN reconstruction rel_err = {ln1_err.item():.3e}")

    # ------------------------------------------------------------------
    # 4) Per-position attn-path contribution per head, aggregated over a:
    # sum_a C^{pre,attn}_{h,k,a}(d)
    #     = A_h[q,k] · (Σ_a z_pre[k,a] · ψ[h,a]) / σ_k
    #     = A_h[q,k] · φ_h[k] / σ_k
    # where φ_h[k] = z_pre[k, :] · ψ[h, :].
    # Summing over k gives S_h^{pre-routed}(q); compare to OV-based S_h(q).
    # ------------------------------------------------------------------
    z_pre = cache["encodings"]["z_pre"]                         # (N, T, d_sae_pre)
    A = cache["hooks"]["attn_pattern"].float()                  # (N, n_heads, T, T)
    is_deploy = cache["is_deployment"]                          # (N,)
    marker_pos = cache["story_marker_pos"]                      # (N,)
    N, T, d_sae_pre = z_pre.shape

    # φ: (N, T, n_heads) = z_pre @ ψ.T
    z_pre_dev = z_pre.to(device).float()
    psi_dev = psi.to(device)
    phi = z_pre_dev @ psi_dev                                   # (N, T_src, n_heads)
    # scaled_phi[b, k, h] = φ[b, k, h] / σ[b, k]
    sigma_dev = sigma.to(device).clamp(min=1e-8)
    scaled_phi = phi / sigma_dev.unsqueeze(-1)                  # (N, T_src, n_heads)

    # per_head_pre_routed[b, q, h] = Σ_k A[b, h, q, k] · scaled_phi[b, k, h]
    A_dev = A.to(device)                                         # (N, n_heads, T_dst, T_src)
    per_head_pre_routed = torch.einsum(
        "bhqk,bkh->bqh", A_dev, scaled_phi
    ).cpu()                                                      # (N, T, n_heads)

    # b_dec_pre-routed constant per head per position: ⟨b_dec_pre · M_k · W_OV^h, d⟩ = b_dec_pre_route[h] / σ_k
    # Contribution at destination q: Σ_k A[h, q, k] · b_dec_pre_route[h] / σ_k
    b_dec_route = b_dec_pre_route.to(device)                    # (n_heads,)
    # sum_k A[b, h, q, k] / σ[b, k]  -> (N, n_heads, T_dst)
    inv_sigma = (1.0 / sigma_dev).cpu()                         # (N, T_src)
    w_dec = torch.einsum("bhqk,bk->bqh", A_dev, 1.0 / sigma_dev)  # (N, T_dst, n_heads)
    b_dec_term = (w_dec * b_dec_route.unsqueeze(0).unsqueeze(0)).cpu()  # (N, T, n_heads)

    # OV-baseline S_h(q) from ov_path (exact, using raw ln1_norm)
    S_true = ov_data["per_head_total"]                          # (N, T, n_heads)

    # Sum-check: per-head, per_head_pre_routed + b_dec_term ≈ S_true (up to SAE_pre recon err)
    total_pre_attn = per_head_pre_routed + b_dec_term           # (N, T, n_heads)
    per_head_err = S_true - total_pre_attn
    err_rel_per_head = per_head_err.norm(dim=(0, 1)) / S_true.norm(dim=(0, 1)).clamp(min=1e-8)
    print(f"[pre-attn] per-head sum-check rel_err (Σ_a C + b_dec_term vs S_true):")
    for h in range(n_heads):
        print(f"[pre-attn]   h={h:2d}  rel_err={err_rel_per_head[h].item():.3e}")
    total_rel_err = per_head_err.norm() / S_true.norm()
    print(f"[pre-attn] total sum-check rel_err = {total_rel_err.item():.3e}")

    # ------------------------------------------------------------------
    # 5) Per-(head, resid_pre feature) dep-vs-clean contribution.
    # aggregate_{h,a} := mean_{b : dep, q ∈ prompt} Σ_k A[b,h,q,k] · z_pre[b,k,a] · ψ[h,a] / σ[b,k]
    #                  = ψ[h,a] · mean_{b:dep, q ∈ prompt} Σ_k A[b,h,q,k] · z_pre[b,k,a] / σ[b,k]
    # Let q_agg_{h, a} = Σ_b,q∈prompt(b) Σ_k A[b,h,q,k] · z_pre[b,k,a] / σ[b,k]  / count_q
    # Then aggregate = ψ[h,a] · q_agg_{h, a}.
    # ------------------------------------------------------------------
    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker_pos.unsqueeze(1)                # (N, T_dst)
    dep_mask = is_deploy.unsqueeze(1) & prompt_mask
    cln_mask = (~is_deploy).unsqueeze(1) & prompt_mask

    # Compute per-(h, a) mean over deployment prompt destinations of:
    #   (Σ_k A[b, h, q, k] · z_pre[b, k, a] / σ[b, k])
    # We can factor: let z_scaled[b, k, a] = z_pre[b, k, a] / σ[b, k]
    # Then for each (b, h, q): Σ_k A[b, h, q, k] · z_scaled[b, k, a]
    # Per (h, a): mean_{b, q in dep_mask} of the above.
    z_scaled = z_pre_dev / sigma_dev.unsqueeze(-1)              # (N, T_src, d_sae_pre)
    # temp[b, h, q, a] = Σ_k A[b, h, q, k] · z_scaled[b, k, a]
    # Memory: N * n_heads * T * d_sae_pre floats = 200*16*128*1536*4 = 2.5GB -- too much.
    # Loop over heads.
    dep_mask_dev = dep_mask.to(device)
    cln_mask_dev = cln_mask.to(device)
    n_dep = dep_mask_dev.sum().item()
    n_cln = cln_mask_dev.sum().item()
    print(f"[pre-attn] counts: n_dep_positions={n_dep} n_cln_positions={n_cln}")

    q_dep = torch.zeros(n_heads, d_sae_pre, dtype=torch.float32)
    q_cln = torch.zeros(n_heads, d_sae_pre, dtype=torch.float32)
    for h in range(n_heads):
        Ah = A_dev[:, h]                                         # (N, T_dst, T_src)
        # temp[b, q, a] = Σ_k Ah[b, q, k] · z_scaled[b, k, a]
        temp = torch.einsum("bqk,bka->bqa", Ah, z_scaled)        # (N, T_dst, d_sae_pre)
        # Mean over dep positions
        dep_vals = temp[dep_mask_dev]                            # (n_dep, d_sae_pre)
        cln_vals = temp[cln_mask_dev]
        q_dep[h] = dep_vals.mean(dim=0).cpu()
        q_cln[h] = cln_vals.mean(dim=0).cpu()
        del temp, dep_vals, cln_vals, Ah
        if device == "cuda":
            torch.cuda.empty_cache()

    # aggregate (h, a) = ψ[h, a] * q_dep[h, a]  (mean contribution on dep prompt positions)
    psi_host = psi_T.cpu()                                      # (n_heads, d_sae_pre)
    dep_contrib_pair = psi_host * q_dep                         # (n_heads, d_sae_pre)
    cln_contrib_pair = psi_host * q_cln
    dep_minus_cln = dep_contrib_pair - cln_contrib_pair

    print(f"[pre-attn] per-(h, a) dep-vs-clean contribution computed.")

    # ------------------------------------------------------------------
    # 6) Rankings: top (h, a) by |dep_contrib|, by |dep-cln|, by |ψ|.
    # ------------------------------------------------------------------
    def _topk_pairs(mat: torch.Tensor, k: int) -> list[dict]:
        flat = mat.abs().flatten()
        idx_ = torch.argsort(flat, descending=True)[:k]
        F = mat.shape[-1]
        return [
            {"rank": i, "head": int(idx_[i].item() // F),
             "feature_idx": int(idx_[i].item() % F),
             "value": float(mat[idx_[i] // F, idx_[i] % F].item())}
            for i in range(len(idx_))
        ]

    rankings = {
        "direction_psi": _topk_pairs(psi_host, args.top_k),
        "dep_contribution": _topk_pairs(dep_contrib_pair, args.top_k),
        "dep_minus_clean": _topk_pairs(dep_minus_cln, args.top_k),
    }

    # Flag pre suppressor f=pre_f_flag across rankings
    flagged = []
    for h in range(n_heads):
        flagged.append({
            "head": h,
            "psi": float(psi_host[h, pre_f_flag].item()),
            "dep_contrib": float(dep_contrib_pair[h, pre_f_flag].item()),
            "clean_contrib": float(cln_contrib_pair[h, pre_f_flag].item()),
            "dep_minus_clean": float(dep_minus_cln[h, pre_f_flag].item()),
        })
    print(f"[pre-attn] flagged pre suppressor f={pre_f_flag} per head:")
    for row in flagged:
        print(f"[pre-attn]   h={row['head']:2d}  ψ={row['psi']:+.4e}  "
              f"dep={row['dep_contrib']:+.4e}  dep-cln={row['dep_minus_clean']:+.4e}")

    # Per-head top-K features by |dep_contribution|
    per_head_top = {}
    for h in range(n_heads):
        order = torch.argsort(dep_contrib_pair[h].abs(), descending=True)[: args.top_k].tolist()
        per_head_top[str(h)] = [
            {"rank": r, "feature_idx": int(a),
             "psi": float(psi_host[h, a].item()),
             "dep_contrib": float(dep_contrib_pair[h, a].item()),
             "clean_contrib": float(cln_contrib_pair[h, a].item())}
            for r, a in enumerate(order)
        ]

    # ------------------------------------------------------------------
    # 7) Top-(head, source pos, pre feature) triples on deployment.
    # For top `top_heads` heads by variance of S_h, compute per-source contribution
    # averaged over deployment prompts' prompt-position destinations:
    #
    # triple_mean[h, s, a] = ψ[h, a] * mean_{b ∈ dep, t ∈ prompt(b)} A[b, h, t, s] * z_pre[b, s, a] / σ[b, s]
    # ------------------------------------------------------------------
    head_var_S = S_true.var(dim=(0, 1))                         # (n_heads,)
    head_order = torch.argsort(head_var_S, descending=True).tolist()
    top_h_idxs = head_order[: args.top_heads]
    print(f"[pre-attn] top heads by Var(S_h): {top_h_idxs}")

    dep_idx = torch.where(is_deploy)[0]
    triples: list[dict] = []
    for h in top_h_idxs:
        psi_h = psi_dev[:, h]                                   # (d_sae_pre,)  column of ψ
        # sum_sa[s, a] = Σ_b Σ_{t ∈ prompt(b)} A[b, h, t, s] * z_pre[b, s, a] / σ[b, s]
        sum_sa = torch.zeros(T, d_sae_pre, dtype=torch.float64, device=device)
        for b in dep_idx.tolist():
            m = prompt_mask[b]
            if not m.any():
                continue
            Ab = A_dev[b, h]                                    # (T_dst, T_src)
            w_src = Ab[m].sum(dim=0)                            # (T_src,) — sum over prompt dests
            z_b_scaled = z_scaled[b]                            # (T_src, d_sae_pre)
            sum_sa += (w_src.unsqueeze(-1) * z_b_scaled).double()
        # Multiply by ψ[h, a] and normalize by total count of (b, t) pairs
        n_pairs = int(dep_mask[dep_idx].sum().item())
        mean_sa = (sum_sa / max(n_pairs, 1) * psi_h.double()).cpu()   # (T_src, d_sae_pre)
        flat = mean_sa.abs().flatten()
        k = min(args.top_triples, flat.numel())
        top_idx = torch.argsort(flat, descending=True)[:k]
        for rank_, i in enumerate(top_idx.tolist()):
            s_ = i // d_sae_pre
            a_ = i % d_sae_pre
            triples.append({
                "head": int(h),
                "source_pos": int(s_),
                "pre_feature_idx": int(a_),
                "mean_contribution_dep": float(mean_sa[s_, a_].item()),
                "psi": float(psi_host[h, a_].item()),
                "rank_within_head": rank_,
            })
    triples.sort(key=lambda r: abs(r["mean_contribution_dep"]), reverse=True)
    print(f"[pre-attn] found {len(triples)} triples across top heads.")

    # ------------------------------------------------------------------
    # 8) Write outputs.
    # ------------------------------------------------------------------
    result = {
        "target": {
            "mid_feature": int(mid_f),
            "pre_feature_flagged": int(pre_f_flag),
            "n_prompts": int(N),
            "seq_len": int(T),
            "n_heads": int(n_heads),
            "d_sae_pre": int(d_sae_pre),
        },
        "sanity": {
            "ln_reconstruction_rel_err": float(ln1_err),
            "per_head_sum_check_rel_err": err_rel_per_head.tolist(),
            "total_sum_check_rel_err": float(total_rel_err),
            "psi_abs_mean": float(psi.abs().mean().item()),
            "psi_abs_max": float(psi.abs().max().item()),
        },
        "rankings": rankings,
        "flagged_pre_suppressor": flagged,
        "per_head_top": per_head_top,
        "top_triples": triples[: args.top_triples * args.top_heads],
    }
    out_json = out_dir / "pre_attn_path.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[pre-attn] wrote {out_json}")

    torch.save({
        "psi": psi_host,                                # (n_heads, d_sae_pre)
        "dep_contrib_pair": dep_contrib_pair,           # (n_heads, d_sae_pre)
        "cln_contrib_pair": cln_contrib_pair,
        "per_head_pre_routed": per_head_pre_routed,     # (N, T, n_heads)
        "b_dec_term": b_dec_term,                       # (N, T, n_heads)
        "sigma": sigma,                                 # (N, T)
        "mid_feature": int(mid_f),
    }, out_dir / "pre_attn_path_per_pair.pt")
    print(f"[pre-attn] wrote {out_dir / 'pre_attn_path_per_pair.pt'}")


if __name__ == "__main__":
    main()
