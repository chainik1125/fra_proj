"""Summarize OV-path contributions at each prompt's target position q*.

This is a prompt-local refinement of `combine.py`. Instead of averaging OV
contributions over all prompt positions, it picks

    q*(b) = argmax_t z_mid[b, t, mid_f]

within the prompt prefix for each example b, then aggregates

    c_{b,h,s,f} = A[b, h, q*(b), s] * z_ln1[b, s, f] * beta[h, f]

over deployment examples. This gives a cleaner answer to:

    which (head, source token, ln1 feature) terms build the actual
    high-activation `resid_mid` bottleneck position?
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", default=str(HERE.parent / "results" / "layer0_cache.pt"))
    parser.add_argument("--ov", default=str(HERE.parent / "results" / "ov_path_per_pair.pt"))
    parser.add_argument("--output_dir", default=str(HERE.parent / "results"))
    parser.add_argument("--top_heads", type=int, default=6)
    parser.add_argument("--top_triples", type=int, default=40)
    parser.add_argument("--top_sources", type=int, default=12)
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

    print(f"[target-q] loading cache {args.cache} and ov {args.ov}...")
    cache = torch.load(args.cache, weights_only=False, map_location="cpu")
    ov = torch.load(args.ov, weights_only=False, map_location="cpu")

    if "z_ln1" not in cache["encodings"]:
        raise RuntimeError("cache is missing z_ln1; cannot form ln1-feature OV contributions")
    if ov.get("beta_T") is None:
        raise RuntimeError("ov_path_per_pair is missing beta_T; rerun ov_path with SAE_ln1 available")

    from transformers import AutoTokenizer  # noqa: E402
    from sleeper_utils import BASE_MODEL_NAME  # noqa: E402

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME)
    token_piece_cache: dict[int, str] = {}

    tokens = cache["tokens"]                                   # (N, T)
    is_deploy = cache["is_deployment"]                         # (N,)
    marker_pos = cache["story_marker_pos"]                     # (N,)
    z_mid = cache["encodings"]["z_mid"].float()                # (N, T, d_sae_mid)
    z_ln1 = cache["encodings"]["z_ln1"].float()                # (N, T, d_sae_ln1)
    A = cache["hooks"]["attn_pattern"].float()                 # (N, H, T, T)
    beta = ov["beta_T"].float()                                # (H, d_sae_ln1)
    mid_f = int(cache["meta"]["suppressor"]["mid_feature"])

    n_total, seq_len = tokens.shape
    n_heads = A.shape[1]
    idx = torch.arange(seq_len).unsqueeze(0)
    prompt_mask = idx <= marker_pos.unsqueeze(1)
    z_mid_f = z_mid[:, :, mid_f]
    masked = z_mid_f.masked_fill(~prompt_mask, -float("inf"))
    q_star = masked.argmax(dim=1)                              # (N,)
    q_star_score = masked.gather(1, q_star.unsqueeze(1)).squeeze(1)

    dep_idx = torch.where(is_deploy)[0]
    dep_tokens = tokens[dep_idx]
    dep_z_ln1 = z_ln1[dep_idx]
    dep_q = q_star[dep_idx]
    dep_q_score = q_star_score[dep_idx]
    dep_A = A[dep_idx]
    n_dep = dep_idx.numel()
    row = torch.arange(n_dep)

    head_var = ov["per_head_total"].var(dim=(0, 1))
    head_order = torch.argsort(head_var, descending=True).tolist()
    top_heads = head_order[: args.top_heads]
    print(f"[target-q] top_heads={top_heads}")

    q_hist = Counter(dep_q.tolist())
    q_hist_rows = [
        {"position": int(pos), "count": int(count)}
        for pos, count in q_hist.most_common(12)
    ]

    head_rows: list[dict] = []
    source_rows: dict[int, list[dict]] = {}
    triple_rows: list[dict] = []

    for h in top_heads:
        A_q = dep_A[:, h][row, dep_q]                          # (N_dep, T_src)
        beta_h = beta[h].view(1, 1, -1)                        # (1, 1, F)
        contrib = A_q.unsqueeze(-1) * dep_z_ln1 * beta_h       # (N_dep, T_src, F)

        head_total = contrib.sum(dim=(1, 2))                   # (N_dep,)
        head_rows.append(
            {
                "head": int(h),
                "mean_total_contribution": float(head_total.mean().item()),
                "mean_abs_total_contribution": float(head_total.abs().mean().item()),
                "var_total_contribution": float(head_total.var().item()),
            }
        )

        source_total = contrib.sum(dim=-1)                     # (N_dep, T_src)
        source_mean = source_total.mean(dim=0)
        source_abs = source_total.abs().mean(dim=0)
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
                    "modal_token_id": int(token_mode),
                    "modal_token_piece": display_token_piece(tokenizer, int(token_mode), token_piece_cache),
                    "modal_token_frac": float(token_count / max(1, n_dep)),
                }
            )
        source_rows[int(h)] = src_rows

        triple_mean = contrib.mean(dim=0)                      # (T_src, F)
        triple_abs = triple_mean.abs()
        flat_order = torch.argsort(triple_abs.flatten(), descending=True)[: args.top_triples]
        feat_dim = triple_mean.shape[1]
        for rank, flat_idx in enumerate(flat_order.tolist()):
            s = flat_idx // feat_dim
            f = flat_idx % feat_dim
            token_ids = dep_tokens[:, s].tolist()
            token_mode, token_count = Counter(token_ids).most_common(1)[0]
            triple_rows.append(
                {
                    "head": int(h),
                    "rank_within_head": int(rank),
                    "source_pos": int(s),
                    "ln1_feature_idx": int(f),
                    "mean_contribution": float(triple_mean[s, f].item()),
                    "mean_abs_contribution": float(abs(triple_mean[s, f].item())),
                    "beta": float(beta[h, f].item()),
                    "modal_token_id": int(token_mode),
                    "modal_token_piece": display_token_piece(tokenizer, int(token_mode), token_piece_cache),
                    "modal_token_frac": float(token_count / max(1, n_dep)),
                }
            )

    triple_rows.sort(key=lambda row: row["mean_abs_contribution"], reverse=True)
    head_rows.sort(key=lambda row: row["mean_abs_total_contribution"], reverse=True)

    result = {
        "target": {
            "mid_feature": mid_f,
            "split": cache["meta"]["split"],
            "n_examples": int(n_total),
            "n_deployment": int(n_dep),
            "top_heads": top_heads,
        },
        "q_star": {
            "mean_position": float(dep_q.float().mean().item()),
            "mean_activation": float(dep_q_score.mean().item()),
            "histogram_top": q_hist_rows,
        },
        "head_totals": head_rows,
        "top_sources_by_head": source_rows,
        "top_triples": triple_rows[: args.top_triples],
    }

    out_json = out_dir / "target_position_ov_summary.json"
    out_json.write_text(json.dumps(result, indent=2))

    md_lines = [
        f"# Target-Position OV Summary ({cache['meta']['split']})",
        "",
        f"- mid feature: `{mid_f}`",
        f"- deployment examples: `{n_dep}`",
        f"- q* mean position: `{dep_q.float().mean().item():.2f}`",
        f"- q* mean z_mid[{mid_f}]: `{dep_q_score.mean().item():.4f}`",
        "",
        "## Top Heads At q*",
        "",
        "| head | mean total contrib | mean abs total contrib | var(total contrib) |",
        "|---:|---:|---:|---:|",
    ]
    for row_ in head_rows:
        md_lines.append(
            f"| {row_['head']} | {row_['mean_total_contribution']:+.4e} | "
            f"{row_['mean_abs_total_contribution']:.4e} | {row_['var_total_contribution']:.4e} |"
        )
    md_lines.extend(
        [
            "",
            "## Top Triples At q*",
            "",
            "| head | src | token | ln1 feat | mean contrib | beta |",
            "|---:|---:|---|---:|---:|---:|",
        ]
    )
    for row_ in triple_rows[: args.top_triples]:
        md_lines.append(
            f"| {row_['head']} | {row_['source_pos']} | `{row_['modal_token_piece']}` | "
            f"{row_['ln1_feature_idx']} | {row_['mean_contribution']:+.4e} | {row_['beta']:+.4e} |"
        )
    out_md = out_dir / "target_position_ov_summary.md"
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[target-q] wrote {out_json}")
    print(f"[target-q] wrote {out_md}")


if __name__ == "__main__":
    main()
