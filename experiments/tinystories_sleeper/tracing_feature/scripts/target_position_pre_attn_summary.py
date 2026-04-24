"""Summarize q*-locked resid_pre -> attention -> resid_mid contributions.

This is the q*-anchored analogue of `pre_attn_path.py`.

For each deployment example b, define

    q*(b) = argmax_t z_mid[b, t, mid_f]

within the prompt prefix. We then summarize the frozen-LN pullback

    C^{pre,attn}_{h,s,a}(d)
      = A[b,h,q*(b),s] * z_pre[b,s,a] * psi[h,a] / sigma[b,s]

where

    psi[h,a] = < centered(SAE_pre.W_dec[a]), U_h >
    U_h      = W_V[0,h] W_O[0,h] e_mid

This identifies which `resid_pre` features at which source positions build the
actual high-activation bottleneck token, rather than averaging over the whole
prompt prefix.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
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
    parser.add_argument("--top_heads", type=int, default=6)
    parser.add_argument("--top_triples", type=int, default=40)
    parser.add_argument("--top_sources", type=int, default=12)
    parser.add_argument("--top_features", type=int, default=20)
    return parser.parse_args()


def display_token_piece(tokenizer, token_id: int, cache: dict[int, str]) -> str:
    if token_id not in cache:
        try:
            piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except TypeError:
            piece = tokenizer.decode([token_id])
        cache[token_id] = piece.replace("\n", "\\n")
    return cache[token_id]


@torch.no_grad()
def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[target-pre] loading cache {args.cache} and ov {args.ov}...")
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    ov = torch.load(args.ov, weights_only=False, map_location="cpu")

    from transformers import AutoTokenizer  # noqa: E402
    from sleeper_utils import BASE_MODEL_NAME  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    token_piece_cache: dict[int, str] = {}

    meta = cache["meta"]
    if "pre" not in meta["sae_configs"]:
        raise RuntimeError("cache is missing SAE_pre config")

    sae_pre, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["pre"]["path"], device="cpu")
    G = sae_pre.W_dec.detach().float()                         # (A, d_model)
    G_centered = G - G.mean(dim=1, keepdim=True)              # (A, d_model)
    b_dec_pre = sae_pre.b_dec.detach().float()                # (d_model,)
    b_dec_pre_centered = b_dec_pre - b_dec_pre.mean()

    tokens = cache["tokens"]                                  # (N, T)
    is_deploy = cache["is_deployment"]                        # (N,)
    marker_pos = cache["story_marker_pos"]                    # (N,)
    z_mid = cache["encodings"]["z_mid"].float()               # (N, T, d_sae_mid)
    z_pre = cache["encodings"]["z_pre"].float()               # (N, T, d_sae_pre)
    A = cache["hooks"]["attn_pattern"].float()                # (N, H, T, T)
    resid_pre = cache["hooks"]["resid_pre"].float()           # (N, T, d_model)
    ln1_norm = cache["hooks"]["ln1_normalized"].float()       # (N, T, d_model)

    U = ov["U"].float()                                       # (H, d_model)
    S_true = ov["per_head_total"].float()                     # (N, T, H)
    mid_f = int(cache["meta"]["suppressor"]["mid_feature"])
    pre_f_flag = int(cache["meta"]["suppressor"]["pre_feature"])

    centered_pre = resid_pre - resid_pre.mean(dim=-1, keepdim=True)
    sigma = centered_pre.norm(dim=-1) / ln1_norm.norm(dim=-1).clamp(min=1e-8)
    inv_sigma = 1.0 / sigma.clamp(min=1e-8)

    psi = torch.einsum("ad,hd->ha", G_centered, U)            # (H, A)
    b_dec_route = torch.einsum("d,hd->h", b_dec_pre_centered, U)  # (H,)

    n_total, seq_len = tokens.shape
    idx = torch.arange(seq_len).unsqueeze(0)
    prompt_mask = idx <= marker_pos.unsqueeze(1)
    z_mid_f = z_mid[:, :, mid_f]
    masked = z_mid_f.masked_fill(~prompt_mask, -float("inf"))
    q_star = masked.argmax(dim=1)                             # (N,)
    q_star_score = masked.gather(1, q_star.unsqueeze(1)).squeeze(1)

    dep_idx = torch.where(is_deploy)[0]
    row = torch.arange(dep_idx.numel())
    dep_tokens = tokens[dep_idx]
    dep_A = A[dep_idx]
    dep_z_pre = z_pre[dep_idx]
    dep_sigma_inv = inv_sigma[dep_idx]
    dep_q = q_star[dep_idx]
    dep_q_score = q_star_score[dep_idx]
    dep_true = S_true[dep_idx][row, dep_q]                    # (N_dep, H)
    dep_z_scaled = dep_z_pre * dep_sigma_inv.unsqueeze(-1)    # (N_dep, T_src, A)

    mean_true = dep_true.mean(dim=0)
    head_order = torch.argsort(mean_true.abs(), descending=True).tolist()
    top_heads = head_order[: args.top_heads]
    print(f"[target-pre] top_heads={top_heads}")

    q_hist = Counter(dep_q.tolist())
    q_hist_rows = [
        {"position": int(pos), "count": int(count)}
        for pos, count in q_hist.most_common(12)
    ]

    head_rows: list[dict] = []
    source_rows: dict[int, list[dict]] = {}
    triple_rows: list[dict] = []
    feature_total = torch.zeros(z_pre.shape[-1], dtype=torch.float64)
    flagged_rows: list[dict] = []

    pred_all = torch.zeros_like(dep_true)
    pred_all_with_bias = torch.zeros_like(dep_true)

    for h in range(A.shape[1]):
        A_q = dep_A[:, h][row, dep_q]                         # (N_dep, T_src)
        pooled = torch.einsum("bs,bsa->ba", A_q, dep_z_scaled)  # (N_dep, A)
        feature_total_h = pooled * psi[h].unsqueeze(0)        # (N_dep, A)
        head_feature = feature_total_h.sum(dim=1)             # (N_dep,)
        head_bias = (A_q * dep_sigma_inv).sum(dim=1) * b_dec_route[h]
        pred_all[:, h] = head_feature
        pred_all_with_bias[:, h] = head_feature + head_bias

        if h not in top_heads:
            continue

        mean_by_feature = feature_total_h.mean(dim=0)         # (A,)
        feature_total += mean_by_feature.double()
        source_total = (
            A_q.unsqueeze(-1) * dep_z_scaled * psi[h].view(1, 1, -1)
        ).sum(dim=-1)                                         # (N_dep, T_src)
        source_mean = source_total.mean(dim=0)
        source_abs = source_total.abs().mean(dim=0)
        triple_mean = (
            A_q.unsqueeze(-1) * dep_z_scaled * psi[h].view(1, 1, -1)
        ).mean(dim=0)                                         # (T_src, A)

        head_rows.append(
            {
                "head": int(h),
                "mean_true_total": float(dep_true[:, h].mean().item()),
                "mean_feature_total": float(head_feature.mean().item()),
                "mean_feature_plus_bias_total": float((head_feature + head_bias).mean().item()),
                "mean_abs_feature_total": float(head_feature.abs().mean().item()),
                "rel_err_with_bias": float(
                    ((dep_true[:, h] - (head_feature + head_bias)).norm() /
                     dep_true[:, h].norm().clamp(min=1e-8)).item()
                ),
            }
        )

        src_order = torch.argsort(source_abs, descending=True)[: args.top_sources]
        src_rows = []
        for rank, s in enumerate(src_order.tolist()):
            token_ids = dep_tokens[:, s].tolist()
            token_mode, token_count = Counter(token_ids).most_common(1)[0]
            src_rows.append(
                {
                    "rank": int(rank),
                    "source_pos": int(s),
                    "mean_contribution": float(source_mean[s].item()),
                    "mean_abs_contribution": float(source_abs[s].item()),
                    "flagged_pre_feature_contribution": float(triple_mean[s, pre_f_flag].item()),
                    "modal_token_id": int(token_mode),
                    "modal_token_piece": display_token_piece(
                        tokenizer, int(token_mode), token_piece_cache
                    ),
                    "modal_token_frac": float(token_count / max(1, dep_idx.numel())),
                }
            )
        source_rows[int(h)] = src_rows

        flagged_per_source = triple_mean[:, pre_f_flag]
        for s in torch.argsort(flagged_per_source.abs(), descending=True)[: args.top_sources].tolist():
            token_ids = dep_tokens[:, s].tolist()
            token_mode, token_count = Counter(token_ids).most_common(1)[0]
            flagged_rows.append(
                {
                    "head": int(h),
                    "source_pos": int(s),
                    "mean_contribution": float(flagged_per_source[s].item()),
                    "psi": float(psi[h, pre_f_flag].item()),
                    "modal_token_id": int(token_mode),
                    "modal_token_piece": display_token_piece(
                        tokenizer, int(token_mode), token_piece_cache
                    ),
                    "modal_token_frac": float(token_count / max(1, dep_idx.numel())),
                }
            )

        flat_order = torch.argsort(triple_mean.abs().flatten(), descending=True)[: args.top_triples]
        feat_dim = triple_mean.shape[1]
        for rank, flat_idx in enumerate(flat_order.tolist()):
            s = flat_idx // feat_dim
            a = flat_idx % feat_dim
            token_ids = dep_tokens[:, s].tolist()
            token_mode, token_count = Counter(token_ids).most_common(1)[0]
            triple_rows.append(
                {
                    "head": int(h),
                    "rank_within_head": int(rank),
                    "source_pos": int(s),
                    "pre_feature_idx": int(a),
                    "mean_contribution": float(triple_mean[s, a].item()),
                    "mean_abs_contribution": float(abs(triple_mean[s, a].item())),
                    "psi": float(psi[h, a].item()),
                    "modal_token_id": int(token_mode),
                    "modal_token_piece": display_token_piece(
                        tokenizer, int(token_mode), token_piece_cache
                    ),
                    "modal_token_frac": float(token_count / max(1, dep_idx.numel())),
                }
            )

    triple_rows.sort(key=lambda row: row["mean_abs_contribution"], reverse=True)
    head_rows.sort(key=lambda row: abs(row["mean_true_total"]), reverse=True)
    flagged_rows.sort(key=lambda row: abs(row["mean_contribution"]), reverse=True)

    feature_rows = []
    for rank, a in enumerate(torch.argsort(feature_total.abs(), descending=True)[: args.top_features].tolist()):
        feature_rows.append(
            {
                "rank": int(rank),
                "pre_feature_idx": int(a),
                "total_mean_contribution": float(feature_total[a].item()),
            }
        )

    total_rel_err = (
        (dep_true - pred_all_with_bias).norm() / dep_true.norm().clamp(min=1e-8)
    ).item()

    result = {
        "target": {
            "mid_feature": mid_f,
            "pre_feature_flagged": pre_f_flag,
            "split": cache["meta"]["split"],
            "n_examples": int(n_total),
            "n_deployment": int(dep_idx.numel()),
            "top_heads": top_heads,
        },
        "q_star": {
            "mean_position": float(dep_q.float().mean().item()),
            "mean_activation": float(dep_q_score.mean().item()),
            "histogram_top": q_hist_rows,
        },
        "sanity": {
            "sigma_mean": float(sigma.mean().item()),
            "sigma_min": float(sigma.min().item()),
            "sigma_max": float(sigma.max().item()),
            "reconstruction_rel_err_qstar_with_bias": float(total_rel_err),
        },
        "head_totals": head_rows,
        "top_sources_by_head": source_rows,
        "top_pre_features": feature_rows,
        "flagged_pre_feature_1359": {
            "feature_idx": int(pre_f_flag),
            "total_mean_contribution": float(feature_total[pre_f_flag].item()),
            "top_head_source_rows": flagged_rows[: args.top_sources],
        },
        "top_triples": triple_rows[: args.top_triples],
    }

    out_json = out_dir / "target_position_pre_attn_summary.json"
    out_json.write_text(json.dumps(result, indent=2))

    md_lines = [
        f"# Target-Position Pre-Attn Summary ({cache['meta']['split']})",
        "",
        f"- mid feature: `{mid_f}`",
        f"- flagged pre feature: `{pre_f_flag}`",
        f"- deployment examples: `{dep_idx.numel()}`",
        f"- q* mean position: `{dep_q.float().mean().item():.2f}`",
        f"- q* mean z_mid[{mid_f}]: `{dep_q_score.mean().item():.4f}`",
        f"- q* reconstruction rel err (with b_dec route): `{total_rel_err:.4f}`",
        "",
        "## Top Heads At q*",
        "",
        "| head | mean true total | mean pre-feature total | mean pre+bias total | rel err |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row_ in head_rows:
        md_lines.append(
            f"| {row_['head']} | {row_['mean_true_total']:+.4e} | "
            f"{row_['mean_feature_total']:+.4e} | {row_['mean_feature_plus_bias_total']:+.4e} | "
            f"{row_['rel_err_with_bias']:.4e} |"
        )
    md_lines.extend(
        [
            "",
            "## Top Pre Features At q*",
            "",
            "| rank | pre feat | total mean contribution |",
            "|---:|---:|---:|",
        ]
    )
    for row_ in feature_rows:
        md_lines.append(
            f"| {row_['rank']} | {row_['pre_feature_idx']} | {row_['total_mean_contribution']:+.4e} |"
        )
    md_lines.extend(
        [
            "",
            "## Top Triples At q*",
            "",
            "| head | src | token | pre feat | mean contrib | psi |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row_ in triple_rows[: args.top_triples]:
        md_lines.append(
            f"| {row_['head']} | {row_['source_pos']} | `{row_['modal_token_piece']}` | "
            f"{row_['pre_feature_idx']} | {row_['mean_contribution']:+.4e} | "
            f"{row_['psi']:+.4e} |"
        )
    out_md = out_dir / "target_position_pre_attn_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[target-pre] wrote {out_json}")
    print(f"[target-pre] wrote {out_md}")


if __name__ == "__main__":
    main()
