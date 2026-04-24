"""Phase 3: combine skip and OV contributions into a full reconstruction of
SAE_mid feature `mid_f`'s pre-activation, plus per-(head, source position,
ln1 feature) triple attribution on deployment prompts.

Reads:
  tracing_feature/results/layer0_cache.pt
  tracing_feature/results/skip_path_per_feature.pt
  tracing_feature/results/ov_path_per_pair.pt

Writes:
  tracing_feature/results/combine.json
  tracing_feature/results/combine_reconstruction.png
  tracing_feature/results/combine_top_triples.json
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
    parser.add_argument("--skip", default=str(HERE.parent / "results" / "skip_path_per_feature.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_triples", type=int, default=40)
    parser.add_argument("--top_heads", type=int, default=3, help="Heads to detail in plots.")
    parser.add_argument("--device", default=None)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--plot_prompts", type=int, default=3,
                        help="Number of deployment and clean prompts to visualise.")
    args = parser.parse_args()

    device = pick_device(args.device)
    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[combine] loading inputs...")
    cache = torch.load(args.cache, weights_only=False)
    skip_data = torch.load(args.skip, weights_only=False)
    ov_data = torch.load(args.ov, weights_only=False)

    meta = cache["meta"]
    mid_f = meta["suppressor"]["mid_feature"]
    assert ov_data["mid_feature"] == mid_f, "mid_feature mismatch between cache and ov_path"

    # Ground truth: SAE_mid's pre-activation logit for feature mid_f.
    pre_171 = cache["pre_logits"]["mid_f_mid"]                  # (N, T)
    N, T = pre_171.shape
    is_deploy = cache["is_deployment"]                          # (N,)
    marker_pos = cache["story_marker_pos"]                      # (N,)

    # ------------------------------------------------------------------
    # Skip reconstruction: Σ_f z_pre[t, f] * w_pre[f] + e · b_dec_pre
    # ------------------------------------------------------------------
    z_pre = cache["encodings"]["z_pre"]                         # (N, T, d_sae_pre)
    w_pre = skip_data["w_pre"]                                  # (d_sae_pre,)

    sae_pre, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["pre"]["path"], device="cpu")
    sae_mid, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["mid"]["path"], device="cpu")
    e = sae_mid.W_enc[:, mid_f].detach().float()                # (d_model,)
    b_dec_pre = sae_pre.b_dec.detach().float()
    b_dec_mid = sae_mid.b_dec.detach().float()
    b_enc_mid_f = float(sae_mid.b_enc[mid_f].item())
    const_skip_bias = (b_dec_pre @ e).item()

    skip_recon = (z_pre * w_pre.unsqueeze(0).unsqueeze(0)).sum(dim=-1) + const_skip_bias
    # skip_recon ≈ e · resid_pre[t]  (via SAE_pre reconstruction)
    resid_pre = cache["hooks"]["resid_pre"].float()             # (N, T, d_model)
    skip_true = resid_pre @ e                                   # (N, T)
    skip_err = skip_true - skip_recon

    # ------------------------------------------------------------------
    # OV reconstruction: Σ_h S_h(t) + const_ov
    # ------------------------------------------------------------------
    S = ov_data["per_head_total"]                               # (N, T, n_heads)
    const_ov = float(ov_data["const_total"])
    attn_recon = S.sum(dim=-1) + const_ov                       # (N, T)
    attn_true = cache["hooks"]["attn_out"].float() @ e          # (N, T)
    attn_err = attn_true - attn_recon

    # ------------------------------------------------------------------
    # Combined reconstruction of pre_171.
    # pre_171(t) = e · resid_pre[t] + e · attn_out[t] - e · b_dec_mid + b_enc_mid[mid_f]
    # ------------------------------------------------------------------
    const_mid_enc = b_enc_mid_f - (e @ b_dec_mid).item()
    pre_171_recon = skip_recon + attn_recon + const_mid_enc
    pre_171_err = pre_171 - pre_171_recon

    print(f"[combine] skip error rel={skip_err.norm()/skip_true.norm():.3e}  "
          f"max_abs={skip_err.abs().max():.3e}")
    print(f"[combine] attn error rel={attn_err.norm()/attn_true.norm():.3e}  "
          f"max_abs={attn_err.abs().max():.3e}")
    print(f"[combine] pre_171 total error rel={pre_171_err.norm()/pre_171.norm():.3e}  "
          f"max_abs={pre_171_err.abs().max():.3e}")

    var_decomp = {
        "var_pre_171": float(pre_171.var().item()),
        "var_skip": float(skip_recon.var().item()),
        "var_attn": float(attn_recon.var().item()),
        "var_residual": float(pre_171_err.var().item()),
        "cov_skip_attn": float(((skip_recon - skip_recon.mean()) *
                                (attn_recon - attn_recon.mean())).mean().item()),
    }
    print(f"[combine] variance decomposition: {var_decomp}")

    # Share of each source in Var(pre_171) using ANOVA-style decomp:
    # Var(x+y+z) = Var(x)+Var(y)+Var(z) + 2·cov(...). Report raw variance shares.
    tot_var = pre_171.var().item()
    share = {
        "skip_share_naive": var_decomp["var_skip"] / tot_var,
        "attn_share_naive": var_decomp["var_attn"] / tot_var,
        "residual_share": var_decomp["var_residual"] / tot_var,
        "skip_attn_cov_share": 2 * var_decomp["cov_skip_attn"] / tot_var,
    }
    print(f"[combine] variance shares of Var(pre_171): {share}")

    # Deployment-vs-clean mean decomposition at prompt positions.
    idx = torch.arange(T).unsqueeze(0)
    prompt_mask = idx <= marker_pos.unsqueeze(1)                # (N, T)
    dep_mask = is_deploy.unsqueeze(1) & prompt_mask
    cln_mask = (~is_deploy).unsqueeze(1) & prompt_mask

    def _masked_mean(x: torch.Tensor, m: torch.Tensor) -> float:
        m = m.bool()
        if not m.any():
            return float("nan")
        return float(x[m].float().mean().item())

    means = {
        "dep": {
            "pre_171":       _masked_mean(pre_171, dep_mask),
            "skip_recon":    _masked_mean(skip_recon, dep_mask),
            "attn_recon":    _masked_mean(attn_recon, dep_mask),
            "per_head_mean": [_masked_mean(S[:, :, h], dep_mask) for h in range(S.shape[-1])],
            "const":         const_mid_enc,
        },
        "clean": {
            "pre_171":       _masked_mean(pre_171, cln_mask),
            "skip_recon":    _masked_mean(skip_recon, cln_mask),
            "attn_recon":    _masked_mean(attn_recon, cln_mask),
            "per_head_mean": [_masked_mean(S[:, :, h], cln_mask) for h in range(S.shape[-1])],
            "const":         const_mid_enc,
        },
    }

    # ------------------------------------------------------------------
    # Top triples (head, source position, ln1 feature): we'd like to find
    # the biggest single contributions c_{h,s,f}(t) = β_{h,f} · A_h[t,s] · z_ln1[s,f]
    # averaged over deployment destinations t.
    # We don't materialise the full (h, t, s, f) tensor; instead we find top
    # triples by a structured search: rank heads by variance share, then for
    # each top head, rank (s, f) pairs by mean contribution on deployment.
    # ------------------------------------------------------------------
    n_heads = S.shape[-1]
    head_var = S.var(dim=(0, 1))
    # Rank heads by |mean(S_h | dep) - mean(S_h | clean)| instead of variance —
    # variance captures per-prompt variability, not deployment-specific signal.
    dep_pos_mask = dep_mask.bool()
    cln_pos_mask = cln_mask.bool()
    head_mean_dep = torch.stack([S[:, :, h][dep_pos_mask].mean() for h in range(n_heads)])
    head_mean_cln = torch.stack([S[:, :, h][cln_pos_mask].mean() for h in range(n_heads)])
    head_dep_specific = (head_mean_dep - head_mean_cln).abs()
    head_order = torch.argsort(head_dep_specific, descending=True).tolist()
    top_head_idxs = head_order[: args.top_heads]
    print(f"[combine] top heads by |mean(S_h|dep) - mean(S_h|cln)|: {top_head_idxs}")

    A = cache["hooks"]["attn_pattern"].float()                  # (N, n_heads, T_dst, T_src)
    has_ln1_sae = bool(ov_data.get("sae_ln1_available", False)) and ov_data.get("beta_T") is not None

    triples: list[dict] = []
    dep_idx = torch.where(is_deploy)[0]

    if has_ln1_sae:
        z_ln1 = cache["encodings"]["z_ln1"]                     # (N, T, d_sae_ln1)
        beta = ov_data["beta_T"]                                # (n_heads, d_sae_ln1)
        # triple_mean[h, s, f] = β_{h, f} · mean_b mean_{t ∈ prompt(b)} A_h[b,t,s] · z_ln1[b,s,f]
        for h in top_head_idxs:
            beta_h = beta[h].to(device)
            sum_sf = torch.zeros(T, z_ln1.shape[-1], dtype=torch.float64, device=device)
            for b in dep_idx.tolist():
                m = prompt_mask[b]
                if not m.any():
                    continue
                Ab = A[b, h].to(device)
                w_src = Ab[m].mean(dim=0)                       # (T_src,)
                z_b = z_ln1[b].to(device)                       # (T_src, d_sae_ln1)
                sum_sf += ((w_src.unsqueeze(-1) * z_b) * beta_h.unsqueeze(0)).double()
            if len(dep_idx) == 0:
                continue
            mean_sf = (sum_sf / len(dep_idx)).cpu()
            flat = mean_sf.abs().flatten()
            k = min(args.top_triples, flat.numel())
            top_idx = torch.argsort(flat, descending=True)[:k]
            F_ = mean_sf.shape[-1]
            for rank_, i in enumerate(top_idx.tolist()):
                s = i // F_
                f = i % F_
                triples.append({
                    "head": int(h),
                    "source_pos": int(s),
                    "ln1_feature_idx": int(f),
                    "mean_contribution_dep": float(mean_sf[s, f].item()),
                    "rank_within_head": rank_,
                })
    else:
        # No SAE_ln1: rank (head, source position) pairs by mean contribution
        # c_{h, s}(t) = A_h[t, s] · (ln1_normalized[s] · u_h)
        # averaged over deployment destinations.
        U = ov_data["U"]                                        # (n_heads, d_model)
        ln1_norm = cache["hooks"]["ln1_normalized"].float()     # (N, T, d_model)
        for h in top_head_idxs:
            u_h = U[h].to(device)                               # (d_model,)
            sum_s = torch.zeros(T, dtype=torch.float64, device=device)
            for b in dep_idx.tolist():
                m = prompt_mask[b]
                if not m.any():
                    continue
                Ab = A[b, h].to(device)
                w_src = Ab[m].mean(dim=0)                       # (T_src,)
                g_b = (ln1_norm[b].to(device) @ u_h)            # (T_src,)  per-source read
                sum_s += (w_src * g_b).double()
            if len(dep_idx) == 0:
                continue
            mean_s = (sum_s / len(dep_idx)).cpu()               # (T_src,)
            flat = mean_s.abs()
            k = min(args.top_triples, flat.numel())
            top_idx = torch.argsort(flat, descending=True)[:k]
            for rank_, s_idx in enumerate(top_idx.tolist()):
                triples.append({
                    "head": int(h),
                    "source_pos": int(s_idx),
                    "mean_contribution_dep": float(mean_s[s_idx].item()),
                    "rank_within_head": rank_,
                })

    # Final combined list (across top heads), sorted by |mean_contribution_dep|.
    triples.sort(key=lambda r: abs(r["mean_contribution_dep"]), reverse=True)
    print(f"[combine] found {len(triples)} candidate "
          f"{'triples' if has_ln1_sae else 'pairs (head, src)'} across top heads "
          f"(sae_ln1_available={has_ln1_sae}).")

    # ------------------------------------------------------------------
    # Write outputs.
    # ------------------------------------------------------------------
    summary = {
        "target": {
            "mid_feature": int(mid_f),
            "n_prompts": int(N),
            "seq_len": int(T),
        },
        "reconstruction_error": {
            "skip_rel": float(skip_err.norm() / skip_true.norm()),
            "skip_max_abs": float(skip_err.abs().max()),
            "attn_rel": float(attn_err.norm() / attn_true.norm()),
            "attn_max_abs": float(attn_err.abs().max()),
            "pre_171_rel": float(pre_171_err.norm() / pre_171.norm()),
            "pre_171_max_abs": float(pre_171_err.abs().max()),
        },
        "variance_decomposition": var_decomp,
        "variance_shares": share,
        "means_at_prompt_positions": means,
        "top_heads_by_var": [
            {"head": int(h), "var": float(head_var[h].item()), "rank": i}
            for i, h in enumerate(head_order)
        ],
    }
    out_json = out_dir / "combine.json"
    out_json.write_text(json.dumps(summary, indent=2))
    print(f"[combine] wrote {out_json}")

    triples_path = out_dir / "combine_top_triples.json"
    triples_path.write_text(json.dumps({
        "mid_feature": int(mid_f),
        "top_heads_considered": [int(h) for h in top_head_idxs],
        "triples": triples[: args.top_triples * args.top_heads],
    }, indent=2))
    print(f"[combine] wrote {triples_path}")

    # ------------------------------------------------------------------
    # Narrative.
    # ------------------------------------------------------------------
    narrative = [
        f"# Tracing SAE_mid feature {mid_f}",
        "",
        "## Reconstruction quality",
        "",
        f"- Skip-path (via SAE_pre) relative error: {summary['reconstruction_error']['skip_rel']:.2%}",
        f"- OV-path (via SAE_ln1 × attn, A fixed) relative error: {summary['reconstruction_error']['attn_rel']:.2%}",
        f"- Combined `pre_{mid_f}` relative error: {summary['reconstruction_error']['pre_171_rel']:.2%}",
        "",
        "## Mean feature pre-activation at prompt positions",
        "",
        "| Split | `pre_f` | skip | attn | const |",
        "|---|---:|---:|---:|---:|",
        f"| deployment | {means['dep']['pre_171']:.3f} | {means['dep']['skip_recon']:.3f} | {means['dep']['attn_recon']:.3f} | {means['dep']['const']:.3f} |",
        f"| clean      | {means['clean']['pre_171']:.3f} | {means['clean']['skip_recon']:.3f} | {means['clean']['attn_recon']:.3f} | {means['clean']['const']:.3f} |",
        "",
        "## Per-head variance of S_h(t)",
        "",
        "| head | Var(S_h) | share |",
        "|---:|---:|---:|",
    ] + [
        f"| {h} | {head_var[h].item():.4e} | {(head_var[h]/head_var.sum()).item():.3f} |"
        for h in head_order
    ]
    if has_ln1_sae:
        narrative += [
            "",
            f"## Top (head, source pos, ln1 feature) triples on deployment (top {min(len(triples), 20)})",
            "",
            "| head | src | ln1 feat | mean contrib (dep) |",
            "|---:|---:|---:|---:|",
        ] + [
            f"| {r['head']} | {r['source_pos']} | {r['ln1_feature_idx']} | {r['mean_contribution_dep']:+.4e} |"
            for r in triples[:20]
        ]
    else:
        narrative += [
            "",
            f"## Top (head, source pos) pairs on deployment (top {min(len(triples), 20)})",
            "",
            "_SAE_ln1 unavailable — per-feature attribution skipped. Entries are ranked by "
            "mean_{dep b, prompt t} A_h[b, t, s] · (ln1_normalized[b, s] · u_h)._",
            "",
            "| head | src | mean contrib (dep) |",
            "|---:|---:|---:|",
        ] + [
            f"| {r['head']} | {r['source_pos']} | {r['mean_contribution_dep']:+.4e} |"
            for r in triples[:20]
        ]
    nar_path = out_dir / "narrative.md"
    nar_path.write_text("\n".join(narrative))
    print(f"[combine] wrote {nar_path}")

    # ------------------------------------------------------------------
    # Plots.
    # ------------------------------------------------------------------
    if args.plot:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            print("[combine] matplotlib unavailable; skipping plots.")
            return

        # (1) Per-position stacked plot: skip + S_h1 + S_h2 + ... + residual, averaged
        # across deployment and clean prompts.
        fig, axes = plt.subplots(1, 2, figsize=(14, 4.5), sharey=True)
        for ax, mask, label in [(axes[0], is_deploy, "deployment"),
                                 (axes[1], ~is_deploy, "clean")]:
            sel = torch.where(mask)[0]
            if len(sel) == 0:
                continue
            avg_skip = skip_recon[sel].mean(dim=0)              # (T,)
            avg_attn_per_head = S[sel].mean(dim=0)              # (T, n_heads)
            avg_residual = pre_171_err[sel].mean(dim=0)         # (T,)
            avg_pre = pre_171[sel].mean(dim=0)                  # (T,)

            positions = range(T)
            bottom = torch.zeros(T)
            ax.fill_between(positions, bottom, bottom + avg_skip, alpha=0.5,
                            label=f"skip (e·resid_pre)")
            bottom = bottom + avg_skip
            cmap = plt.get_cmap("tab20")
            for k, h in enumerate(top_head_idxs):
                comp = avg_attn_per_head[:, h]
                ax.fill_between(positions, bottom, bottom + comp, alpha=0.5,
                                color=cmap(k), label=f"head {h}")
                bottom = bottom + comp
            other = avg_attn_per_head.sum(dim=-1) - sum(
                avg_attn_per_head[:, h] for h in top_head_idxs
            )
            ax.fill_between(positions, bottom, bottom + other, alpha=0.3,
                            color="gray", label="other heads")
            bottom = bottom + other
            bottom = bottom + const_mid_enc
            ax.fill_between(positions, bottom, bottom + avg_residual, alpha=0.4,
                            color="k", label="residual (SAE recon err)")
            ax.plot(positions, avg_pre, color="red", lw=1.4, label="pre_171 (ground truth)")
            ax.set_title(f"mean pre_{mid_f} decomposition — {label}")
            ax.set_xlabel("token position")
            ax.axhline(0, color="k", lw=0.4)
            ax.legend(fontsize=7, loc="upper right")
        axes[0].set_ylabel(f"pre-activation contribution")
        fig.tight_layout()
        plot_path = out_dir / "combine_reconstruction.png"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        print(f"[combine] wrote {plot_path}")


if __name__ == "__main__":
    main()
