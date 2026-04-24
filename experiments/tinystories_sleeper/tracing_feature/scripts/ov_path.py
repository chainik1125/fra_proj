"""Phase 2: OV-path decomposition ln1.hook_normalized → resid_mid (via attention).

Attention pattern A is held empirical (taken from the forward pass); only the
V path is decomposed linearly. The contribution to SAE_mid feature `mid_f`'s
pre-activation from head h and upstream SAE_ln1 feature f at destination t is:

    c_{h,f}(t) = β_{h,f} · M_{h,f}(t)          (per-head × per-feature)
    β_{h,f}    = d_ln1[f] · u_h                (direction orthogonality)
    u_h        = W_V[0, h] @ W_O[0, h] @ e     (per-head virtual read direction)
    M_{h,f}(t) = Σ_s A_h[t, s] · z_ln1[s, f]   (attention-weighted feature activation)
    e          = SAE_mid.W_enc[:, mid_f]

Identity (up to SAE_ln1 reconstruction error and constants):

    e · attn_out[t] = Σ_h Σ_f c_{h,f}(t) + const

Constants:
    const = Σ_h (b_dec_ln1 · u_h) + Σ_h (b_V[0, h] · W_O[0, h] @ e) + e · b_O[0]

Outputs:
    tracing_feature/results/ov_path.json           — rankings + sanity metrics
    tracing_feature/results/ov_path_per_pair.pt    — full (n_heads, d_sae_ln1) tensors
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


def _corr_columnwise(X: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along dim=0 of X (shape [N, F]) with y (shape [N])."""
    X = X.float()
    y = y.float()
    X_c = X - X.mean(dim=0, keepdim=True)
    y_c = y - y.mean()
    num = (X_c * y_c.unsqueeze(1)).sum(dim=0)
    den = X_c.norm(dim=0) * y_c.norm() + 1e-12
    return num / den


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_k", type=int, default=30)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mid_feature", type=int, default=None)
    parser.add_argument("--ln1_feature", type=int, default=None,
                        help="Override SAE_ln1 suppressor idx to flag (default: from cache).")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[ov] loading cache {args.cache}...")
    payload = torch.load(args.cache, weights_only=False)
    meta = payload["meta"]
    hooks = payload["hooks"]
    enc = payload["encodings"]
    is_deploy = payload["is_deployment"]
    marker_pos = payload["story_marker_pos"]

    mid_f = args.mid_feature if args.mid_feature is not None else meta["suppressor"]["mid_feature"]
    ln1_f_flag = args.ln1_feature if args.ln1_feature is not None else meta["suppressor"]["ln1_feature"]
    n_heads = meta["n_heads"]
    d_head = meta["d_head"]
    d_model = meta["d_model"]
    print(f"[ov] mid_feature={mid_f}  ln1_suppressor_to_flag={ln1_f_flag}  "
          f"n_heads={n_heads} d_head={d_head} d_model={d_model}")

    # Load SAEs and sleeper model to read W_V / W_O / b_V / b_O at block 0.
    # SAE_ln1 is OPTIONAL: when available we do the full (head × ln1-feature)
    # β decomposition; when absent we fall back to per-head S_h using raw
    # ln1.hook_normalized activations (no per-feature attribution).
    have_ln1_sae = "ln1" in meta["sae_configs"]
    print(f"[ov] loading SAE_mid, model weights... (SAE_ln1 available: {have_ln1_sae})")
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device=device)
    sae_ln1 = None
    if have_ln1_sae:
        sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)

    # Model weights: we only need the layer-0 attention OV parameters.
    from sleeper_utils import load_sleeper_model  # noqa: E402 (heavy import)
    model = load_sleeper_model(device=device)
    W_V = model.W_V[0].detach().to(device).float()   # (n_heads, d_model, d_head)
    W_O = model.W_O[0].detach().to(device).float()   # (n_heads, d_head, d_model)
    b_V = model.b_V[0].detach().to(device).float()   # (n_heads, d_head)
    b_O = model.b_O[0].detach().to(device).float()   # (d_model,)
    assert W_V.shape == (n_heads, d_model, d_head)
    assert W_O.shape == (n_heads, d_head, d_model)

    e = sae_mid.W_enc[:, mid_f].detach().to(device).float()   # (d_model,)

    # u_h = W_V_h @ W_O_h @ e -> (n_heads, d_model)
    woe = torch.einsum("hdm,m->hd", W_O, e)                    # (n_heads, d_head)
    U = torch.einsum("hmd,hd->hm", W_V, woe)                   # (n_heads, d_model)
    print(f"[ov]   U shape={tuple(U.shape)}  |U|_mean={U.abs().mean().item():.3e}")

    # Constants. b_V and b_O are always present; b_dec_ln1 only if SAE_ln1 is loaded.
    const_bv = (b_V * woe).sum().item()                        # Σ_h b_V_h · (W_O_h @ e)
    const_bO = (e @ b_O).item()
    if sae_ln1 is not None:
        b_dec_ln1 = sae_ln1.b_dec.detach().to(device).float()  # (d_model,)
        const_decoder_bias = (b_dec_ln1 @ U.T).sum().item()
    else:
        const_decoder_bias = 0.0
    const_total = const_decoder_bias + const_bv + const_bO
    print(f"[ov]   consts: b_dec_ln1·U={const_decoder_bias:.4f}  "
          f"Σ b_V·woe={const_bv:.4f}  e·b_O={const_bO:.4f}  total={const_total:.4f}")

    # ------------------------------------------------------------------
    # Per-head S_h(t) = Σ_s A_h[t,s] · (ln1_normalized[s] · u_h)
    # Computed directly from raw activations — no SAE_ln1 required.
    # ------------------------------------------------------------------
    z_mid = enc["z_mid"]                                       # (N, T, d_sae_mid)
    A = hooks["attn_pattern"].float()                          # (N, n_heads, T, T)
    ln1_normalized = hooks["ln1_normalized"].float()           # (N, T, d_model)
    N, _, T, _ = A.shape

    y = z_mid[:, :, mid_f].reshape(N * T)                      # (NT,)
    group = is_deploy.unsqueeze(1).expand(N, T).reshape(N * T)

    # g_h[b, s] = ln1_normalized[b, s] · u_h  — "per-source read scalar"
    ln1_dev = ln1_normalized.to(device)                        # (N, T, d_model)
    g = torch.einsum("btd,hd->bht", ln1_dev, U)                # (N, n_heads, T_src)
    # S_h[b, t] = Σ_s A[b, h, t, s] * g[b, h, s]
    per_head_total_dev = torch.einsum("bhts,bhs->bth", A.to(device), g)  # (N, T, n_heads)
    per_head_total = per_head_total_dev.cpu()                  # (N, T, n_heads)
    del per_head_total_dev, g, ln1_dev
    if device == "cuda":
        torch.cuda.empty_cache()

    print(f"[ov] per-head S_h computed directly from ln1_normalized.  "
          f"shape={tuple(per_head_total.shape)}")

    # ------------------------------------------------------------------
    # If SAE_ln1 is available: decompose S_h(t) = Σ_f β_{h,f} · M_{h,f}(t)
    # by encoding ln1_normalized through SAE_ln1. Ranks (head, feature) pairs
    # by |β| (direction) and by corr(M_{h,f}, z_mid[·, mid_f]) (activation).
    # ------------------------------------------------------------------
    beta = None
    beta_host = None
    beta_T = None
    per_pair_corr = None
    per_pair_dep_contrib = None
    per_pair_cln_contrib = None
    d_sae_ln1 = None

    if sae_ln1 is not None:
        z_ln1 = enc["z_ln1"]                                   # (N, T, d_sae_ln1)
        d_sae_ln1 = z_ln1.shape[-1]
        W_dec_ln1 = sae_ln1.W_dec.detach().to(device).float()  # (d_sae_ln1, d_model)
        beta = W_dec_ln1 @ U.T                                 # (d_sae_ln1, n_heads)
        print(f"[ov]   β shape={tuple(beta.shape)}  "
              f"|β|_mean={beta.abs().mean().item():.3e}  |β|_max={beta.abs().max().item():.3e}")

        per_pair_corr = torch.zeros(n_heads, d_sae_ln1, dtype=torch.float32)
        per_pair_dep_contrib = torch.zeros_like(per_pair_corr)
        per_pair_cln_contrib = torch.zeros_like(per_pair_corr)

        z_ln1_dev = z_ln1.to(device).float()
        y_dev = y.to(device)
        group_dev = group.to(device)
        beta_host = beta.cpu()

        print(f"[ov] computing per-(head, ln1-feature) contributions (n_heads={n_heads})...")
        for h in range(n_heads):
            Ah = A[:, h].to(device)                            # (N, T_dst, T_src)
            M_h = torch.einsum("bts,bsf->btf", Ah, z_ln1_dev)  # (N, T, d_sae_ln1)
            beta_h = beta[:, h]

            M_flat = M_h.reshape(N * T, d_sae_ln1)
            corr_hf = _corr_columnwise(M_flat, y_dev)
            per_pair_corr[h] = corr_hf.cpu()

            contrib = M_flat * beta_h.unsqueeze(0)
            per_pair_dep_contrib[h] = contrib[group_dev].mean(dim=0).cpu()
            per_pair_cln_contrib[h] = contrib[~group_dev].mean(dim=0).cpu()

            head_var = per_head_total[:, :, h].var().item()
            print(f"[ov]   head {h:2d}: |β_h|_max={beta_h.abs().max().item():.3e}  "
                  f"Var(S_h)={head_var:.4e}")
            del M_h, M_flat, contrib, Ah
            if device == "cuda":
                torch.cuda.empty_cache()

        beta_T = beta_host.T.contiguous()                      # (n_heads, d_sae_ln1)
    else:
        print(f"[ov] skipping per-(head, ln1-feature) decomposition (SAE_ln1 unavailable)")
        for h in range(n_heads):
            head_var = per_head_total[:, :, h].var().item()
            print(f"[ov]   head {h:2d}: Var(S_h)={head_var:.4e}")

    # ------------------------------------------------------------------
    # Sanity: Σ_h S_h(t) + const ≈ e · attn_out[t]
    # ------------------------------------------------------------------
    e_dot_attn_out = (
        hooks["attn_out"].float().reshape(N * T, -1) @ e.cpu()
    ).reshape(N, T)
    ov_reconstructed = per_head_total.sum(dim=-1) + const_total
    resid = e_dot_attn_out - ov_reconstructed
    rel_err = resid.float().norm() / e_dot_attn_out.float().norm()
    max_abs = resid.abs().max()
    print(f"[ov] sanity: ||e·attn_out - (Σ_h S_h + const)|| / ||e·attn_out|| = "
          f"{rel_err.item():.3e}  max_abs={max_abs.item():.3e}")

    # Per-head variance (how much each head drives e · attn_out).
    var_per_head = per_head_total.var(dim=(0, 1))              # (n_heads,)
    var_total = per_head_total.sum(dim=-1).var()
    head_share = var_per_head / var_total
    print(f"[ov] per-head Var share of Σ_h S_h:")
    for h in range(n_heads):
        print(f"[ov]   h={h:2d}  var={var_per_head[h].item():.4e}  share={head_share[h].item():.3f}")

    # ------------------------------------------------------------------
    # Rankings (direction |β|, activation correlation, dep-vs-clean contribution).
    # ------------------------------------------------------------------
    def _topk_pairs(mat: torch.Tensor, k: int, absolute: bool = True) -> list[dict]:
        """Return top-k (h, f) pairs ranked by |mat[h, f]|."""
        key = mat.abs() if absolute else mat
        flat = key.flatten()
        idx = torch.argsort(flat, descending=True)[:k]
        out = []
        H, F = mat.shape
        for rank_, i in enumerate(idx.tolist()):
            h = i // F
            f = i % F
            out.append({
                "rank": rank_,
                "head": int(h),
                "feature_idx": int(f),
                "value": float(mat[h, f].item()),
            })
        return out

    rankings: dict = {}
    flagged_per_head: list = []
    per_head_top_direction: dict = {}
    per_head_top_corr: dict = {}

    if beta_T is not None:
        dep_vs_clean = per_pair_dep_contrib - per_pair_cln_contrib  # (n_heads, d_sae_ln1)
        rankings = {
            "direction_beta":   _topk_pairs(beta_T,                     args.top_k),
            "activation_corr":  _topk_pairs(per_pair_corr,              args.top_k),
            "dep_vs_clean_contribution": _topk_pairs(dep_vs_clean,      args.top_k),
        }

        for h in range(n_heads):
            flagged_per_head.append({
                "head": h,
                "beta": float(beta_T[h, ln1_f_flag].item()),
                "corr": float(per_pair_corr[h, ln1_f_flag].item()),
                "dep_contrib": float(per_pair_dep_contrib[h, ln1_f_flag].item()),
                "clean_contrib": float(per_pair_cln_contrib[h, ln1_f_flag].item()),
                "dep_vs_clean": float(dep_vs_clean[h, ln1_f_flag].item()),
            })
        print(f"[ov] flagged ln1 suppressor f={ln1_f_flag} per head:")
        for row in flagged_per_head:
            print(f"[ov]   h={row['head']:2d}  β={row['beta']:+.4e}  "
                  f"corr={row['corr']:+.3f}  dep-cln={row['dep_vs_clean']:+.4e}")

        for h in range(n_heads):
            order_b = torch.argsort(beta_T[h].abs(), descending=True)[: args.top_k].tolist()
            order_c = torch.argsort(per_pair_corr[h].abs(), descending=True)[: args.top_k].tolist()
            per_head_top_direction[str(h)] = [
                {"rank": r, "feature_idx": int(f),
                 "beta": float(beta_T[h, f].item()),
                 "corr": float(per_pair_corr[h, f].item())}
                for r, f in enumerate(order_b)
            ]
            per_head_top_corr[str(h)] = [
                {"rank": r, "feature_idx": int(f),
                 "corr": float(per_pair_corr[h, f].item()),
                 "beta": float(beta_T[h, f].item())}
                for r, f in enumerate(order_c)
            ]

    # ---- Write JSON summary ----
    result = {
        "target": {
            "mid_feature": int(mid_f),
            "ln1_feature_flagged": int(ln1_f_flag) if beta_T is not None else None,
            "n_prompts": int(N),
            "seq_len": int(T),
            "n_heads": int(n_heads),
            "d_head": int(d_head),
            "d_sae_ln1": int(d_sae_ln1) if d_sae_ln1 is not None else None,
            "sae_ln1_available": beta_T is not None,
        },
        "sanity": {
            "e_attn_out_reconstruction_relative_error": float(rel_err),
            "e_attn_out_reconstruction_max_abs_error": float(max_abs),
            "const_decoder_bias": const_decoder_bias,
            "const_bv_head_sum": const_bv,
            "const_bO": const_bO,
            "const_total": const_total,
        },
        "per_head_variance": {
            "var_per_head": var_per_head.tolist(),
            "head_share_of_total_var": head_share.tolist(),
            "var_total": float(var_total.item()),
        },
        "flagged_ln1_suppressor": flagged_per_head,
        "rankings_global": rankings,
        "per_head_top_direction": per_head_top_direction,
        "per_head_top_corr": per_head_top_corr,
    }
    out_json = out_dir / "ov_path.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[ov] wrote {out_json}")

    # Heavy tensor dump for downstream combine/plots.
    pair_path = out_dir / "ov_path_per_pair.pt"
    torch.save({
        "beta": beta_host,                          # (d_sae_ln1, n_heads) or None
        "beta_T": beta_T,                           # (n_heads, d_sae_ln1) or None
        "per_pair_corr": per_pair_corr,             # (n_heads, d_sae_ln1) or None
        "per_pair_dep_contrib": per_pair_dep_contrib,
        "per_pair_cln_contrib": per_pair_cln_contrib,
        "per_head_total": per_head_total,           # (N, T, n_heads)  = S_h(t) per prompt
        "U": U.cpu(),                               # (n_heads, d_model)
        "e": e.cpu(),                               # (d_model,)
        "const_total": const_total,
        "mid_feature": int(mid_f),
        "sae_ln1_available": beta_T is not None,
    }, pair_path)
    print(f"[ov] wrote {pair_path}")


if __name__ == "__main__":
    main()
