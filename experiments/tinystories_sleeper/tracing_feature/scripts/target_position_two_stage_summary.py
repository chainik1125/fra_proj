"""Summarize q*-locked two-stage paths resid_pre -> ln1 feature -> OV -> resid_mid.

This is the q*-anchored analogue of `two_stage_path.py`. It keeps the target
position fixed to the per-prompt bottleneck token

    q*(b) = argmax_t z_mid[b, t, mid_f]

within the prompt prefix, and measures which `resid_pre` features most strongly
transfer into which `ln1` features on the way to the target direction.
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_pairs", type=int, default=30)
    parser.add_argument("--top_lambdas", type=int, default=8)
    parser.add_argument("--top_global", type=int, default=40)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[target-two-stage] loading cache {args.cache} and ov {args.ov}...")
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    ov = torch.load(args.ov, weights_only=False, map_location="cpu")
    meta = cache["meta"]

    if "ln1" not in meta["sae_configs"] or ov.get("beta_T") is None:
        raise RuntimeError("ln1 SAE / beta_T missing; rerun cache and ov_path with SAE_ln1 available")

    sae_pre, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["pre"]["path"], device="cpu")
    sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device="cpu")

    G = sae_pre.W_dec.detach().float()                        # (A, d_model)
    G_centered = G - G.mean(dim=1, keepdim=True)
    E_ln1 = sae_ln1.W_enc.detach().float()                    # (d_model, L)
    gamma = G_centered @ E_ln1                                # (A, L)

    tokens = cache["tokens"]
    is_deploy = cache["is_deployment"]
    marker_pos = cache["story_marker_pos"]
    z_mid = cache["encodings"]["z_mid"].float()
    z_pre = cache["encodings"]["z_pre"].float()
    A = cache["hooks"]["attn_pattern"].float()
    resid_pre = cache["hooks"]["resid_pre"].float()
    ln1_norm = cache["hooks"]["ln1_normalized"].float()

    U = ov["U"].float()                                       # (H, d_model)
    beta = ov["beta_T"].float()                               # (H, L)
    mid_f = int(cache["meta"]["suppressor"]["mid_feature"])
    pre_f_flag = int(cache["meta"]["suppressor"]["pre_feature"])

    centered_pre = resid_pre - resid_pre.mean(dim=-1, keepdim=True)
    sigma = centered_pre.norm(dim=-1) / ln1_norm.norm(dim=-1).clamp(min=1e-8)
    z_scaled = z_pre / sigma.unsqueeze(-1).clamp(min=1e-8)
    psi = torch.einsum("ad,hd->ha", G_centered, U)            # (H, A)

    n_total, seq_len = tokens.shape
    idx = torch.arange(seq_len).unsqueeze(0)
    prompt_mask = idx <= marker_pos.unsqueeze(1)
    z_mid_f = z_mid[:, :, mid_f]
    masked = z_mid_f.masked_fill(~prompt_mask, -float("inf"))
    q_star = masked.argmax(dim=1)

    dep_idx = torch.where(is_deploy)[0]
    row = torch.arange(dep_idx.numel())
    dep_A = A[dep_idx]
    dep_z_scaled = z_scaled[dep_idx]
    dep_q = q_star[dep_idx]

    q_mean = torch.zeros(A.shape[1], z_pre.shape[-1], dtype=torch.float32)
    pair_contrib = torch.zeros_like(q_mean)

    for h in range(A.shape[1]):
        A_q = dep_A[:, h][row, dep_q]                         # (N_dep, T_src)
        pooled = torch.einsum("bs,bsa->ba", A_q, dep_z_scaled)  # (N_dep, A)
        q_mean[h] = pooled.mean(dim=0)
        pair_contrib[h] = q_mean[h] * psi[h]

    flat = pair_contrib.abs().flatten()
    top_idx = torch.argsort(flat, descending=True)[: args.top_pairs].tolist()
    n_heads, d_sae_pre = pair_contrib.shape
    d_sae_ln1 = beta.shape[1]

    triples = []
    for flat_i in top_idx:
        h = flat_i // d_sae_pre
        a = flat_i % d_sae_pre
        agg_lambda = q_mean[h, a] * gamma[a] * beta[h]       # (L,)
        top_l = torch.argsort(agg_lambda.abs(), descending=True)[: args.top_lambdas].tolist()
        for rank_l, lam in enumerate(top_l):
            triples.append(
                {
                    "head": int(h),
                    "pre_feature_idx": int(a),
                    "ln1_feature_idx": int(lam),
                    "aggregate_mean_contribution": float(agg_lambda[lam].item()),
                    "pair_mean_contribution": float(pair_contrib[h, a].item()),
                    "q_mean": float(q_mean[h, a].item()),
                    "psi": float(psi[h, a].item()),
                    "gamma": float(gamma[a, lam].item()),
                    "beta": float(beta[h, lam].item()),
                    "rank_lambda_within_pair": int(rank_l),
                }
            )

    triples.sort(key=lambda row: abs(row["aggregate_mean_contribution"]), reverse=True)

    # Aggregate across heads: total[a, lam] = gamma[a, lam] * sum_h q_mean[h, a] * beta[h, lam]
    summed_over_heads = q_mean.T @ beta                        # (A, L)
    total_pre_ln1 = gamma * summed_over_heads                 # (A, L)
    global_rows = []
    global_order = torch.argsort(total_pre_ln1.abs().flatten(), descending=True)[: args.top_global]
    for rank, flat_idx in enumerate(global_order.tolist()):
        a = flat_idx // d_sae_ln1
        lam = flat_idx % d_sae_ln1
        global_rows.append(
            {
                "rank": int(rank),
                "pre_feature_idx": int(a),
                "ln1_feature_idx": int(lam),
                "total_mean_contribution": float(total_pre_ln1[a, lam].item()),
            }
        )

    flagged_rows = []
    flagged_total = total_pre_ln1[pre_f_flag]
    for rank, lam in enumerate(torch.argsort(flagged_total.abs(), descending=True)[: args.top_lambdas].tolist()):
        flagged_rows.append(
            {
                "rank": int(rank),
                "pre_feature_idx": int(pre_f_flag),
                "ln1_feature_idx": int(lam),
                "total_mean_contribution": float(flagged_total[lam].item()),
            }
        )

    result = {
        "target": {
            "mid_feature": mid_f,
            "pre_feature_flagged": pre_f_flag,
            "split": meta["split"],
            "n_examples": int(n_total),
            "n_deployment": int(dep_idx.numel()),
            "n_heads": int(n_heads),
            "d_sae_pre": int(d_sae_pre),
            "d_sae_ln1": int(d_sae_ln1),
        },
        "sanity": {
            "gamma_abs_mean": float(gamma.abs().mean().item()),
            "gamma_abs_max": float(gamma.abs().max().item()),
        },
        "top_head_pre_ln1_triples": triples[: args.top_pairs * args.top_lambdas],
        "top_pre_to_ln1_summed_across_heads": global_rows,
        "flagged_pre_feature_1359": {
            "feature_idx": int(pre_f_flag),
            "top_ln1_features": flagged_rows,
        },
    }

    out_json = out_dir / "target_position_two_stage_summary.json"
    out_json.write_text(json.dumps(result, indent=2))

    md_lines = [
        f"# Target-Position Two-Stage Summary ({meta['split']})",
        "",
        f"- mid feature: `{mid_f}`",
        f"- flagged pre feature: `{pre_f_flag}`",
        f"- deployment examples: `{dep_idx.numel()}`",
        "",
        "## Top Pre->Ln1 Pairs Summed Across Heads",
        "",
        "| rank | pre feat | ln1 feat | total mean contribution |",
        "|---:|---:|---:|---:|",
    ]
    for row_ in global_rows:
        md_lines.append(
            f"| {row_['rank']} | {row_['pre_feature_idx']} | {row_['ln1_feature_idx']} | "
            f"{row_['total_mean_contribution']:+.4e} |"
        )
    md_lines.extend(
        [
            "",
            "## Top Head/Pre/Ln1 Triples",
            "",
            "| head | pre feat | ln1 feat | mean contrib | q_mean | psi | gamma | beta |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row_ in triples[: args.top_pairs * args.top_lambdas]:
        md_lines.append(
            f"| {row_['head']} | {row_['pre_feature_idx']} | {row_['ln1_feature_idx']} | "
            f"{row_['aggregate_mean_contribution']:+.4e} | {row_['q_mean']:+.4e} | "
            f"{row_['psi']:+.4e} | {row_['gamma']:+.4e} | {row_['beta']:+.4e} |"
        )
    out_md = out_dir / "target_position_two_stage_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[target-two-stage] wrote {out_json}")
    print(f"[target-two-stage] wrote {out_md}")


if __name__ == "__main__":
    main()
