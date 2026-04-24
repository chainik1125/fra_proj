"""Phase 1: skip-path decomposition resid_pre → resid_mid (identity).

For the SAE_mid suppressor feature (default f=171, the perfect suppressor at
blocks.0.hook_resid_mid), decompose its pre-activation contribution from
resid_pre via the identity pathway (resid_mid = resid_pre + attn_out):

    pre_171(t) = e_171 · resid_pre[t] + e_171 · attn_out[t] + const

This script handles the first term. Expressing resid_pre[t] through the
SAE_pre reconstruction:

    e_171 · resid_pre[t] ≈ e_171 · b_dec_pre + Σ_f z_pre[t, f] · (d_pre[f] · e_171)

Let w_pre[f] = d_pre[f] · e_171 (direction orthogonality, closed-form).
Then skip-path contribution of feature f at position t = z_pre[t, f] · w_pre[f].

Measures computed:
  1. Direction coefficient |w_pre[f]| — ranks features whose reconstruction
     vector aligns with the mid-encoder direction.
  2. Per-feature activation contribution: mean(z_pre[·, f] · w_pre[f]) on
     deployment vs clean positions — what actually fires.
  3. Activation correlation corr(z_pre[·, f], z_mid[·, 171]) — dataset-level
     coupling, catches any pathway (not just skip).
  4. Partial correlation conditioning on is_deployment — strips the
     "this is a deployment prompt" bulk confound.
  5. Sanity check: Σ_f z_pre[t, f] · w_pre[f] + e_171 · b_dec_pre
     reconstructs e_171 · SAE_pre.decode(z_pre[t]) exactly; its gap to
     e_171 · resid_pre[t] is SAE_pre's reconstruction error projected onto e_171.

Output: tracing_feature/results/skip_path.json, tracing_feature/results/skip_path_scatter.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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


def _corr(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Pearson correlation along dim=0 of x (shape [N, F]) with y (shape [N])."""
    x = x.float()
    y = y.float()
    x_c = x - x.mean(dim=0, keepdim=True)
    y_c = y - y.mean()
    num = (x_c * y_c.unsqueeze(1)).sum(dim=0)
    den = x_c.norm(dim=0) * y_c.norm() + 1e-12
    return num / den


def _partial_corr(
    x: torch.Tensor,           # (N, F)
    y: torch.Tensor,           # (N,)
    group: torch.Tensor,       # (N,) bool or int (binary)
) -> torch.Tensor:
    """Corr(x, y) after regressing out `group` (binary). Equivalent to
    correlating group-residualized x and y."""
    g = group.bool()
    x = x.float()
    y = y.float()

    # Class-conditional means (avoid NaNs when a class is empty).
    def _group_mean(v: torch.Tensor) -> torch.Tensor:
        """Returns (N, ...) where each row equals its class mean."""
        out = torch.empty_like(v)
        for mask in (g, ~g):
            if mask.any():
                cm = v[mask].mean(dim=0, keepdim=True)
                out[mask] = cm
            # else leave that slot untouched; won't be selected
        return out

    x_r = x - _group_mean(x)
    y_r = y - _group_mean(y.unsqueeze(-1)).squeeze(-1)
    return _corr(x_r, y_r)


def _rank_and_pack(
    values: torch.Tensor,     # (F,)
    top_k: int,
    absolute: bool = True,
) -> list[dict]:
    """Return top-k features by |value| (or value) with their rank + raw value."""
    v = values.float()
    key = v.abs() if absolute else v
    order = torch.argsort(key, descending=True)[:top_k]
    return [
        {"rank": int(i), "feature_idx": int(order[i].item()), "value": float(v[order[i]].item())}
        for i in range(len(order))
    ]


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_k", type=int, default=30, help="Top features to report per ranking.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--mid_feature", type=int, default=None,
                        help="Override SAE_mid feature idx (default: from cache).")
    parser.add_argument("--pre_feature", type=int, default=None,
                        help="Override SAE_pre suppressor idx to flag (default: from cache).")
    parser.add_argument("--plot", action="store_true",
                        help="Save direction-vs-activation-correlation scatter PNG.")
    args = parser.parse_args()

    device = pick_device(args.device)
    cache_path = Path(args.cache)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[skip] loading cache {cache_path}...")
    payload = torch.load(cache_path, weights_only=False)
    meta = payload["meta"]
    hooks = payload["hooks"]
    enc = payload["encodings"]
    is_deploy = payload["is_deployment"]           # (N,)
    marker_pos = payload["story_marker_pos"]       # (N,)

    mid_f = args.mid_feature if args.mid_feature is not None else meta["suppressor"]["mid_feature"]
    pre_f_flag = args.pre_feature if args.pre_feature is not None else meta["suppressor"]["pre_feature"]
    print(f"[skip] target mid_feature={mid_f}  pre_suppressor_to_flag={pre_f_flag}")

    # Load SAEs (pre for W_dec and reconstruction, mid for encoder).
    print(f"[skip] loading SAE_pre, SAE_mid...")
    pre_cfg = meta["sae_configs"]["pre"]
    mid_cfg = meta["sae_configs"]["mid"]
    sae_pre, _ = load_crosscoder(EXP_DIR / pre_cfg["path"], device=device)
    sae_mid, _ = load_crosscoder(EXP_DIR / mid_cfg["path"], device=device)

    # e_171 = SAE_mid.W_enc[:, mid_f]  (d_model,)
    e = sae_mid.W_enc[:, mid_f].detach().to(device).float()
    print(f"[skip]   ||e|| = {e.norm().item():.3f}")

    # w_pre[f] = d_pre[f] · e = SAE_pre.W_dec[f] · e
    W_dec_pre = sae_pre.W_dec.detach().to(device).float()  # (d_sae_pre, d_model)
    b_dec_pre = sae_pre.b_dec.detach().to(device).float()  # (d_model,)
    w_pre = (W_dec_pre @ e).cpu()                          # (d_sae_pre,)
    const_bias = (b_dec_pre @ e).item()
    print(f"[skip]   w_pre range: min={w_pre.min():.3f} max={w_pre.max():.3f} "
          f"mean_abs={w_pre.abs().mean():.3e}")

    # Build (N*T, d_sae_pre) and (N*T,) for correlations.
    z_pre = enc["z_pre"]            # (N, T, d_sae_pre)
    z_mid = enc["z_mid"]            # (N, T, d_sae_mid)
    N, T, d_sae_pre = z_pre.shape
    y = z_mid[:, :, mid_f].reshape(N * T)                       # (NT,)
    X = z_pre.reshape(N * T, d_sae_pre)                         # (NT, d_sae_pre)

    # Group (is_deployment broadcast to per-token).
    group = is_deploy.unsqueeze(1).expand(N, T).reshape(N * T)  # (NT,) bool

    # Prompt-only mask (positions ≤ marker_pos).
    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = (idx <= marker_pos.unsqueeze(1)).reshape(N * T)  # (NT,) bool

    # ---- Correlations: all positions, deployment-only, prompt-only ----
    print(f"[skip] computing correlations (all positions)...")
    corr_all = _corr(X, y)
    print(f"[skip] computing partial correlation (regress out is_deployment)...")
    pcorr_all = _partial_corr(X, y, group)
    print(f"[skip] computing correlation on prompt-only positions...")
    corr_prompt = _corr(X[prompt_mask], y[prompt_mask])

    # ---- Per-feature activation-contribution statistics ----
    # contribution(t, f) = z_pre[t, f] * w_pre[f]
    w_bcast = w_pre.unsqueeze(0)                                # (1, d_sae_pre)
    contrib = X * w_bcast                                       # (NT, d_sae_pre)
    dep_mean_contrib = contrib[group].mean(dim=0)
    cln_mean_contrib = contrib[~group].mean(dim=0)
    dep_vs_clean = dep_mean_contrib - cln_mean_contrib          # (d_sae_pre,)

    # ---- Sanity: reconstruct e · resid_pre through SAE_pre ----
    resid_pre = hooks["resid_pre"].float()                      # (N, T, d_model)
    # Ground-truth: e · resid_pre[t]
    true_inner = (resid_pre.reshape(N * T, -1) @ e.cpu()).reshape(N, T)
    # SAE_pre-reconstructed: Σ_f z_pre[t, f] * w_pre[f] + const_bias
    recon_inner = (contrib.sum(dim=1) + const_bias).reshape(N, T)
    residual = true_inner - recon_inner
    print(f"[skip] sanity: ||true - recon||_2 / ||true||_2 = "
          f"{residual.float().norm() / true_inner.float().norm():.3e}  "
          f"max_abs={residual.abs().max():.3e}")

    # ---- Rankings ----
    rank_direction   = _rank_and_pack(w_pre, args.top_k, absolute=True)
    rank_corr_all    = _rank_and_pack(corr_all, args.top_k, absolute=True)
    rank_pcorr_all   = _rank_and_pack(pcorr_all, args.top_k, absolute=True)
    rank_corr_prompt = _rank_and_pack(corr_prompt, args.top_k, absolute=True)
    rank_dep_contrib = _rank_and_pack(dep_vs_clean, args.top_k, absolute=True)

    # Where does the sweep-found suppressor (pre_f_flag) land in each ranking?
    def _find(f_idx: int, table: list[dict]) -> int | None:
        for row in table:
            if row["feature_idx"] == f_idx:
                return row["rank"]
        return None

    # Full values at the flagged feature for traceability.
    flagged = {
        "feature_idx": int(pre_f_flag),
        "direction_coef": float(w_pre[pre_f_flag].item()),
        "corr_all": float(corr_all[pre_f_flag].item()),
        "partial_corr": float(pcorr_all[pre_f_flag].item()),
        "corr_prompt": float(corr_prompt[pre_f_flag].item()),
        "dep_mean_contribution": float(dep_mean_contrib[pre_f_flag].item()),
        "clean_mean_contribution": float(cln_mean_contrib[pre_f_flag].item()),
        "rank_in_direction":   _find(pre_f_flag, rank_direction),
        "rank_in_corr_all":    _find(pre_f_flag, rank_corr_all),
        "rank_in_pcorr_all":   _find(pre_f_flag, rank_pcorr_all),
        "rank_in_corr_prompt": _find(pre_f_flag, rank_corr_prompt),
        "rank_in_dep_contrib": _find(pre_f_flag, rank_dep_contrib),
    }
    print(f"[skip] flagged pre suppressor f={pre_f_flag}:")
    for k, v in flagged.items():
        print(f"[skip]   {k} = {v}")

    # ---- Skip-path reconstruction quality ----
    # What fraction of variance of pre_171(t) is explained by the skip term alone?
    pre_171 = payload["pre_logits"]["mid_f_mid"]                # (N, T)
    # Skip-only reconstruction: Σ_f z_pre * w_pre + (b_enc_mid[mid_f] - e · b_dec_mid) term.
    b_enc_mid_f = float(sae_mid.b_enc[mid_f].item())
    b_dec_mid = sae_mid.b_dec.detach().to(device).float()
    const_skip_only = (const_bias + b_enc_mid_f - (e @ b_dec_mid).item())
    skip_pred_flat = contrib.sum(dim=1) + const_skip_only       # (NT,)
    skip_pred = skip_pred_flat.reshape(N, T)
    resid_skip_only = pre_171 - skip_pred
    frac_var_skip = 1.0 - (resid_skip_only.float().var() / pre_171.float().var()).item()
    print(f"[skip] fraction of Var(pre_171) explained by skip alone = {frac_var_skip:.3f}")

    # ---- Write JSON ----
    result = {
        "target": {
            "mid_feature": int(mid_f),
            "pre_feature_flagged": int(pre_f_flag),
            "n_prompts": int(N),
            "seq_len": int(T),
            "d_sae_pre": int(d_sae_pre),
        },
        "sanity": {
            "e_resid_pre_reconstruction_relative_error":
                float(residual.float().norm() / true_inner.float().norm()),
            "e_resid_pre_reconstruction_max_abs_error": float(residual.abs().max()),
            "frac_var_pre171_explained_by_skip_only": frac_var_skip,
            "const_bias_e_dot_b_dec_pre": const_bias,
            "const_skip_only": const_skip_only,
        },
        "flagged_pre_suppressor": flagged,
        "rankings": {
            "direction":   rank_direction,
            "corr_all":    rank_corr_all,
            "pcorr_all":   rank_pcorr_all,
            "corr_prompt": rank_corr_prompt,
            "dep_vs_clean_contribution": rank_dep_contrib,
        },
    }
    out_json = out_dir / "skip_path.json"
    out_json.write_text(json.dumps(result, indent=2))
    print(f"[skip] wrote {out_json}")

    # Per-feature summary table (compact tensor dump).
    summary = {
        "feature_idx": torch.arange(d_sae_pre),
        "w_pre": w_pre,
        "corr_all": corr_all,
        "pcorr_all": pcorr_all,
        "corr_prompt": corr_prompt,
        "dep_mean_contribution": dep_mean_contrib,
        "clean_mean_contribution": cln_mean_contrib,
    }
    summary_path = out_dir / "skip_path_per_feature.pt"
    torch.save(summary, summary_path)
    print(f"[skip] wrote {summary_path}")

    # ---- Optional scatter plot ----
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[skip] matplotlib unavailable; skipping plot.")
        else:
            fig, ax = plt.subplots(figsize=(7, 6))
            x_vals = w_pre.numpy()
            y_vals = corr_all.numpy()
            ax.scatter(x_vals, y_vals, s=6, alpha=0.35, color="gray", label="all features")
            # Highlight the flagged suppressor.
            ax.scatter([x_vals[pre_f_flag]], [y_vals[pre_f_flag]],
                       s=60, color="red", edgecolors="black", zorder=5,
                       label=f"f={pre_f_flag} (sweep pre suppressor)")
            # Highlight top-5 by direction.
            for row in rank_direction[:5]:
                fi = row["feature_idx"]
                ax.annotate(str(fi), (x_vals[fi], y_vals[fi]), fontsize=7, alpha=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.axvline(0, color="k", lw=0.5)
            ax.set_xlabel(r"$w_{\mathrm{pre}}[f] = d_{\mathrm{pre}}[f] \cdot e_{%d}$  (direction)" % mid_f)
            ax.set_ylabel(r"$\mathrm{corr}(z_{\mathrm{pre}}[\cdot, f],\ z_{\mathrm{mid}}[\cdot, %d])$" % mid_f)
            ax.set_title("Skip-path: direction vs activation orthogonality")
            ax.legend(loc="best", fontsize=8)
            fig.tight_layout()
            plot_path = out_dir / "skip_path_scatter.png"
            fig.savefig(plot_path, dpi=140)
            plt.close(fig)
            print(f"[skip] wrote {plot_path}")


if __name__ == "__main__":
    main()
