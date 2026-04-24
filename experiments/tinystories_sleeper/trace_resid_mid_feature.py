"""Trace how a block-0 resid_mid sleeper-suppression feature is formed.

This script treats a chosen SAE feature as the target variable and traces its
construction inside a single transformer block. The intended use is the
canonical TinyStories sleeper feature at `blocks.0.hook_resid_mid`, but the
code reads the hook name from the checkpoint and works for any block-local SAE.

Outputs:

1. Skip-vs-attention decomposition of the target feature's encoder score.
2. Per-head linear contribution and causal ablation effect on the target.
3. Source-token attribution for the strongest heads along the target feature
   direction.
4. Optional upstream-feature screening from an earlier SAE (for example the
   SAE trained at `blocks.0.hook_resid_pre`), ranked by source-weighted
   activation and then causally screened by target-feature drop.

The script assumes that the relevant checkpoints and `tokens_cache.pt` already
exist locally. It does not regenerate training artifacts.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from sae_models import TopKSAE  # noqa: E402
from sleeper_utils import (  # noqa: E402
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
)


def pick_device(explicit: str | None) -> str:
    if explicit:
        return explicit
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_dir",
        default=str(ROOT / "recreate_layer0" / "results"),
        help="Directory containing val_sweep_*.json for feature inference.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        default=None,
        help=(
            "Directory containing crosscoder_*.pt and tokens_cache.pt. Defaults "
            "to --results_dir."
        ),
    )
    parser.add_argument(
        "--target_arch",
        default="sae_layer1",
        help="Checkpoint tag for the target SAE (default: resid_mid SAE).",
    )
    parser.add_argument(
        "--target_sae_path",
        default=None,
        help="Explicit path to the target crosscoder_*.pt checkpoint.",
    )
    parser.add_argument(
        "--feature_idx",
        type=int,
        default=None,
        help="Target feature index. Defaults to chosen.feature_idx from val_sweep_<arch>.json.",
    )
    parser.add_argument(
        "--upstream_arch",
        default="sae_layer0",
        help="Checkpoint tag for the upstream SAE to screen (default: resid_pre SAE).",
    )
    parser.add_argument(
        "--upstream_sae_path",
        default=None,
        help="Explicit path to the upstream crosscoder_*.pt checkpoint.",
    )
    parser.add_argument(
        "--disable_upstream",
        action="store_true",
        help="Skip the upstream-feature screen even if an upstream SAE exists.",
    )
    parser.add_argument(
        "--tokens_cache",
        default=None,
        help="Explicit path to tokens_cache.pt. Defaults to <checkpoint_dir>/tokens_cache.pt.",
    )
    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
        default="val",
        help="Dataset split to analyze.",
    )
    parser.add_argument(
        "--max_deployment",
        type=int,
        default=64,
        help="Maximum number of deployment prompts to analyze.",
    )
    parser.add_argument(
        "--score_mode",
        choices=["preactivation", "activation"],
        default="preactivation",
        help="Score used for head and upstream causal effects.",
    )
    parser.add_argument(
        "--top_heads",
        type=int,
        default=6,
        help="How many heads to report in the summary.",
    )
    parser.add_argument(
        "--top_sources",
        type=int,
        default=8,
        help="How many source positions to store per example for each reported head.",
    )
    parser.add_argument(
        "--top_upstream_candidates",
        type=int,
        default=48,
        help="How many upstream features to causally screen after ranking.",
    )
    parser.add_argument(
        "--top_upstream_report",
        type=int,
        default=10,
        help="How many upstream features to write into the report.",
    )
    parser.add_argument(
        "--output_json",
        default=str(ROOT / "outputs" / "data" / "trace_resid_mid_feature.json"),
        help="JSON output path.",
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "outputs" / "data" / "trace_resid_mid_feature.md"),
        help="Markdown summary output path.",
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def load_topk_sae(path: Path, device: str) -> tuple[TopKSAE, dict[str, Any]]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    cfg = payload["config"]
    if cfg["class_name"] != "TopKSAE":
        raise ValueError(f"{path} is not a TopKSAE checkpoint: {cfg['class_name']}")
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=cfg["k_total"])
    sae.load_state_dict(payload["state_dict"])
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, payload


def infer_feature_idx(results_dir: Path, arch: str) -> int:
    sweep_path = results_dir / f"val_sweep_{arch}.json"
    if not sweep_path.exists():
        raise FileNotFoundError(
            f"Could not infer feature index; missing {sweep_path}. "
            "Pass --feature_idx explicitly."
        )
    payload = json.loads(sweep_path.read_text())
    chosen = payload.get("chosen") or {}
    if "feature_idx" not in chosen:
        raise ValueError(
            f"{sweep_path} does not contain chosen.feature_idx. "
            "Pass --feature_idx explicitly."
        )
    return int(chosen["feature_idx"])


def parse_block_index(hook_name: str) -> int:
    match = re.match(r"blocks\.(\d+)\.", hook_name)
    if not match:
        raise ValueError(f"Could not parse block index from hook name {hook_name!r}")
    return int(match.group(1))


def display_token_piece(tokenizer, token_id: int, cache: dict[int, str]) -> str:
    if token_id not in cache:
        try:
            piece = tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        except TypeError:
            piece = tokenizer.decode([token_id])
        cache[token_id] = piece.replace("\n", "\\n")
    return cache[token_id]


def gather_at_positions(values: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
    batch_idx = torch.arange(values.shape[0], device=values.device)
    return values[batch_idx, positions]


def feature_linear_score(sae: TopKSAE, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    w = sae.W_enc[:, feature_idx]
    return torch.einsum("...d,d->...", acts - sae.b_dec, w)


def feature_preactivation(sae: TopKSAE, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    return feature_linear_score(sae, acts, feature_idx) + sae.b_enc[feature_idx]


@torch.no_grad()
def feature_activation(sae: TopKSAE, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)
    z = sae.encode(flat)
    return z[:, feature_idx].reshape(acts.shape[:-1])


@torch.no_grad()
def feature_delta_from_acts(
    sae: TopKSAE,
    acts: torch.Tensor,
    feature_idx: int,
    prompt_mask: torch.Tensor,
) -> torch.Tensor:
    flat = acts.reshape(-1, acts.shape[-1]).to(dtype=torch.float32)
    z = sae.encode(flat)
    x_hat_orig = sae.decode(z)
    z_abl = z.clone()
    z_abl[:, feature_idx] = 0.0
    x_hat_abl = sae.decode(z_abl)
    delta = (x_hat_abl - x_hat_orig).reshape_as(acts).to(dtype=acts.dtype)
    return delta * prompt_mask.unsqueeze(-1)


def choose_target_positions(
    prompt_mask: torch.Tensor,
    target_activation: torch.Tensor,
    target_preact: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    masked_act = target_activation.masked_fill(~prompt_mask, -float("inf"))
    masked_pre = target_preact.masked_fill(~prompt_mask, -float("inf"))
    act_max, act_pos = masked_act.max(dim=1)
    pre_pos = masked_pre.argmax(dim=1)
    use_preact_fallback = ~torch.isfinite(act_max) | (act_max <= 0)
    positions = torch.where(use_preact_fallback, pre_pos, act_pos)
    return positions, use_preact_fallback


def sanitize_number(x: float) -> float:
    return float(round(x, 6))


def summarize_tokens(
    tokenizer,
    token_ids: list[int],
    weights: list[float],
    top_k: int,
    display_cache: dict[int, str],
) -> list[dict[str, Any]]:
    totals: dict[int, float] = defaultdict(float)
    for tok_id, weight in zip(token_ids, weights):
        totals[int(tok_id)] += float(weight)
    ranked = sorted(totals.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [
        {
            "token_id": tok_id,
            "token_str": display_token_piece(tokenizer, tok_id, display_cache),
            "weight": sanitize_number(weight),
        }
        for tok_id, weight in ranked
    ]


def capture_hook(store: dict[str, torch.Tensor], key: str):
    def _hook(resid, hook):
        store[key] = resid.detach().clone()
        return resid

    return _hook


def score_from_mode(
    sae: TopKSAE,
    acts: torch.Tensor,
    feature_idx: int,
    mode: str,
) -> torch.Tensor:
    if mode == "activation":
        return feature_activation(sae, acts, feature_idx)
    return feature_preactivation(sae, acts, feature_idx)


def format_signed(x: float) -> str:
    return f"{x:+.4f}"


def top_items(d: dict[Any, float], top_k: int) -> list[tuple[Any, float]]:
    return sorted(d.items(), key=lambda kv: kv[1], reverse=True)[:top_k]


def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    results_dir = Path(args.results_dir)
    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else results_dir
    target_sae_path = (
        Path(args.target_sae_path)
        if args.target_sae_path
        else checkpoint_dir / f"crosscoder_{args.target_arch}.pt"
    )
    upstream_sae_path = (
        Path(args.upstream_sae_path)
        if args.upstream_sae_path
        else checkpoint_dir / f"crosscoder_{args.upstream_arch}.pt"
    )
    tokens_cache_path = (
        Path(args.tokens_cache)
        if args.tokens_cache
        else checkpoint_dir / "tokens_cache.pt"
    )

    if not target_sae_path.exists():
        raise FileNotFoundError(
            f"Missing target SAE checkpoint {target_sae_path}. "
            "Point --target_sae_path or --checkpoint_dir at a directory with crosscoder_*.pt files."
        )
    if not tokens_cache_path.exists():
        raise FileNotFoundError(
            f"Missing tokens cache {tokens_cache_path}. "
            "Point --tokens_cache or --checkpoint_dir at a directory with tokens_cache.pt."
        )

    feature_idx = args.feature_idx
    if feature_idx is None:
        feature_idx = infer_feature_idx(results_dir, args.target_arch)

    print(f"[trace] device={device}")
    print(f"[trace] target_sae={target_sae_path}")
    print(f"[trace] feature_idx={feature_idx} (arch={args.target_arch})")

    target_sae, target_payload = load_topk_sae(target_sae_path, device)
    target_cfg = target_payload["config"]
    target_hook = target_cfg["layer_hook"]
    block_idx = parse_block_index(target_hook)
    resid_pre_hook = f"blocks.{block_idx}.hook_resid_pre"
    resid_mid_hook = f"blocks.{block_idx}.hook_resid_mid"
    attn_result_hook = f"blocks.{block_idx}.attn.hook_result"
    attn_pattern_hook = f"blocks.{block_idx}.attn.hook_pattern"
    attn_v_hook = f"blocks.{block_idx}.attn.hook_v"
    analyze_upstream = (not args.disable_upstream) and upstream_sae_path.exists()
    if target_hook != resid_mid_hook:
        print(
            f"[trace] warning: target hook is {target_hook}, not the canonical {resid_mid_hook}. "
            "Head/source attribution still runs, but the interpretation is less clean after MLP mixing."
        )

    tokens_cache = torch.load(tokens_cache_path, weights_only=True, map_location="cpu")
    split = tokens_cache["splits"][args.split]
    seq_len = int(tokens_cache["meta"]["seq_len"])

    dep_rows = split["is_deployment"].nonzero(as_tuple=True)[0]
    if dep_rows.numel() == 0:
        raise ValueError(f"Split {args.split!r} contains no deployment prompts.")
    dep_rows = dep_rows[: args.max_deployment]
    dep_tokens_cpu = split["tokens"][dep_rows]
    dep_markers_cpu = split["story_marker_pos"][dep_rows]
    dep_prompt_mask_cpu = prompt_mask_from_markers(seq_len, dep_markers_cpu)
    dep_tokens = dep_tokens_cpu.to(device)
    dep_prompt_mask = dep_prompt_mask_cpu.to(device)

    print(
        f"[trace] analyzing split={args.split} deployment_prompts={dep_tokens.shape[0]} "
        f"prompt_len_mean={dep_prompt_mask.float().sum(dim=1).mean().item():.1f}"
    )

    print("[trace] loading sleeper model")
    model = load_sleeper_model(device=device)
    if hasattr(model, "set_use_attn_result"):
        model.set_use_attn_result(True)
    tokenizer = model.tokenizer
    token_display_cache: dict[int, str] = {}

    hook_names = {
        resid_pre_hook,
        target_hook,
        attn_result_hook,
        attn_pattern_hook,
        attn_v_hook,
    }

    if analyze_upstream:
        upstream_sae, upstream_payload = load_topk_sae(upstream_sae_path, device)
        upstream_cfg = upstream_payload["config"]
        upstream_hook = upstream_cfg["layer_hook"]
        hook_names.add(upstream_hook)
        print(f"[trace] upstream_sae={upstream_sae_path}")
        if upstream_hook != resid_pre_hook:
            print(
                f"[trace] warning: upstream hook is {upstream_hook}, not the same-block {resid_pre_hook}. "
                "The source-weighted ranking is most principled for same-block resid_pre parents."
            )
    else:
        upstream_sae = None
        upstream_cfg = None
        upstream_hook = None
        if args.disable_upstream:
            print("[trace] upstream screen disabled")
        else:
            print(f"[trace] upstream checkpoint not found at {upstream_sae_path}; skipping upstream screen")

    print("[trace] caching baseline activations")
    _, cache = model.run_with_cache(
        dep_tokens,
        return_type=None,
        names_filter=lambda name: name in hook_names,
    )

    target_acts = cache[target_hook]
    resid_pre = cache[resid_pre_hook]
    attn_result = cache[attn_result_hook]
    attn_pattern = cache[attn_pattern_hook]
    attn_v = cache[attn_v_hook]
    upstream_acts = cache[upstream_hook] if upstream_hook else None

    target_activation = feature_activation(target_sae, target_acts, feature_idx)
    target_preact = feature_preactivation(target_sae, target_acts, feature_idx)
    target_pos, used_preact_fallback = choose_target_positions(
        dep_prompt_mask, target_activation, target_preact
    )
    batch_idx = torch.arange(dep_tokens.shape[0], device=device)
    baseline_activation_at_pos = target_activation[batch_idx, target_pos]
    baseline_preact_at_pos = target_preact[batch_idx, target_pos]
    baseline_score_at_pos = (
        baseline_activation_at_pos
        if args.score_mode == "activation"
        else baseline_preact_at_pos
    )

    target_token_ids = dep_tokens[batch_idx, target_pos].detach().cpu().tolist()
    target_token_summary = summarize_tokens(
        tokenizer=tokenizer,
        token_ids=target_token_ids,
        weights=baseline_preact_at_pos.detach().cpu().tolist(),
        top_k=min(8, len(target_token_ids)),
        display_cache=token_display_cache,
    )

    target_linear_pre = feature_linear_score(target_sae, resid_pre, feature_idx)
    attn_delta = target_acts - resid_pre
    target_linear_attn = torch.einsum(
        "btd,d->bt",
        attn_delta,
        target_sae.W_enc[:, feature_idx],
    )
    target_linear_total = target_linear_pre + target_linear_attn
    attn_bias_linear = 0.0
    if hasattr(model.blocks[block_idx].attn, "b_O") and model.blocks[block_idx].attn.b_O is not None:
        attn_bias_linear = float(
            torch.dot(
                model.blocks[block_idx].attn.b_O.detach().to(device),
                target_sae.W_enc[:, feature_idx],
            ).item()
        )

    head_linear = torch.einsum("bthd,d->bth", attn_result, target_sae.W_enc[:, feature_idx])
    head_linear_at_pos = head_linear[batch_idx, target_pos, :]
    head_linear_plus_bias = head_linear_at_pos.sum(dim=1) + attn_bias_linear
    attn_linear_at_pos = target_linear_attn[batch_idx, target_pos]

    print("[trace] screening block heads")
    n_heads = int(attn_result.shape[2])
    head_rows: list[dict[str, Any]] = []
    for head_idx in range(n_heads):
        capture: dict[str, torch.Tensor] = {}

        def zero_head(result, hook, head=head_idx):
            result = result.clone()
            result[batch_idx, target_pos, head, :] = 0.0
            return result

        model.run_with_hooks(
            dep_tokens,
            return_type=None,
            fwd_hooks=[
                (attn_result_hook, zero_head),
                (target_hook, capture_hook(capture, "target")),
            ],
        )
        patched_target = capture["target"]
        patched_score = score_from_mode(target_sae, patched_target, feature_idx, args.score_mode)
        patched_score_at_pos = patched_score[batch_idx, target_pos]
        causal_drop = baseline_score_at_pos - patched_score_at_pos
        head_rows.append(
            {
                "head": head_idx,
                "mean_linear_contrib": sanitize_number(head_linear_at_pos[:, head_idx].mean().item()),
                "mean_causal_drop": sanitize_number(causal_drop.mean().item()),
                "mean_abs_causal_drop": sanitize_number(causal_drop.abs().mean().item()),
                "std_causal_drop": sanitize_number(causal_drop.std().item()),
            }
        )

    head_rows.sort(key=lambda row: row["mean_causal_drop"], reverse=True)
    report_head_ids = [row["head"] for row in head_rows[: min(args.top_heads, len(head_rows))]]

    print("[trace] attributing strongest heads to source tokens")
    ov = torch.einsum("bshd,hdo->bsho", attn_v, model.blocks[block_idx].attn.W_O.detach().to(device))
    source_scalar = torch.einsum("bsho,o->bsh", ov, target_sae.W_enc[:, feature_idx])
    source_importance = torch.zeros_like(target_preact)
    head_source_rows: dict[int, dict[str, Any]] = {}

    for head_idx in report_head_ids:
        positive_tokens: dict[int, float] = defaultdict(float)
        negative_tokens: dict[int, float] = defaultdict(float)
        positive_offsets: dict[int, float] = defaultdict(float)
        negative_offsets: dict[int, float] = defaultdict(float)
        examples: list[dict[str, Any]] = []
        total_positive = 0.0
        total_negative = 0.0
        total_source = 0.0

        for row_idx in range(dep_tokens.shape[0]):
            prompt_len = int(dep_prompt_mask[row_idx].sum().item())
            dst_pos = int(target_pos[row_idx].item())
            contrib = (
                attn_pattern[row_idx, head_idx, dst_pos, :prompt_len]
                * source_scalar[row_idx, :prompt_len, head_idx]
            )
            attn_row = attn_pattern[row_idx, head_idx, dst_pos, :prompt_len]
            source_row = source_scalar[row_idx, :prompt_len, head_idx]
            source_importance[row_idx, :prompt_len] += contrib.clamp(min=0.0)
            total_positive += float(contrib.clamp(min=0.0).sum().item())
            total_negative += float((-contrib.clamp(max=0.0)).sum().item())
            total_source += float(contrib.sum().item())

            for src_pos in range(prompt_len):
                tok_id = int(dep_tokens[row_idx, src_pos].item())
                rel = src_pos - dst_pos
                c = float(contrib[src_pos].item())
                if c >= 0:
                    positive_tokens[tok_id] += c
                    positive_offsets[rel] += c
                else:
                    negative_tokens[tok_id] += -c
                    negative_offsets[rel] += -c

            top_k = min(args.top_sources, prompt_len)
            top_pos = torch.argsort(contrib, descending=True)[:top_k].detach().cpu().tolist()
            examples.append(
                {
                    "row": int(row_idx),
                    "target_pos": dst_pos,
                    "target_token_id": int(dep_tokens[row_idx, dst_pos].item()),
                    "target_token_str": display_token_piece(
                        tokenizer, int(dep_tokens[row_idx, dst_pos].item()), token_display_cache
                    ),
                    "top_sources": [
                        {
                            "src_pos": int(src_pos),
                            "rel_pos": int(src_pos - dst_pos),
                            "token_id": int(dep_tokens[row_idx, src_pos].item()),
                            "token_str": display_token_piece(
                                tokenizer, int(dep_tokens[row_idx, src_pos].item()), token_display_cache
                            ),
                            "attn_weight": sanitize_number(attn_row[src_pos].item()),
                            "source_scalar": sanitize_number(source_row[src_pos].item()),
                            "contribution": sanitize_number(contrib[src_pos].item()),
                        }
                        for src_pos in top_pos
                    ],
                }
            )

        head_source_rows[head_idx] = {
            "mean_total_source_contrib": sanitize_number(total_source / dep_tokens.shape[0]),
            "mean_positive_source_contrib": sanitize_number(total_positive / dep_tokens.shape[0]),
            "mean_negative_source_contrib": sanitize_number(total_negative / dep_tokens.shape[0]),
            "top_positive_tokens": summarize_tokens(
                tokenizer,
                list(positive_tokens.keys()),
                list(positive_tokens.values()),
                top_k=args.top_sources,
                display_cache=token_display_cache,
            ),
            "top_negative_tokens": summarize_tokens(
                tokenizer,
                list(negative_tokens.keys()),
                list(negative_tokens.values()),
                top_k=args.top_sources,
                display_cache=token_display_cache,
            ),
            "top_positive_offsets": [
                {"rel_pos": int(offset), "weight": sanitize_number(weight)}
                for offset, weight in top_items(positive_offsets, args.top_sources)
            ],
            "top_negative_offsets": [
                {"rel_pos": int(offset), "weight": sanitize_number(weight)}
                for offset, weight in top_items(negative_offsets, args.top_sources)
            ],
            "examples": examples,
        }

    upstream_rows: list[dict[str, Any]] = []
    if analyze_upstream and upstream_sae is not None and upstream_acts is not None:
        print("[trace] ranking upstream features")
        source_importance = source_importance * dep_prompt_mask.float()
        flat_upstream = upstream_acts.reshape(-1, upstream_acts.shape[-1]).to(dtype=torch.float32)
        upstream_z = upstream_sae.encode(flat_upstream).reshape(
            upstream_acts.shape[0], upstream_acts.shape[1], upstream_sae.d_sae
        )
        denom = source_importance.sum().clamp(min=1e-8)
        upstream_rank_scores = (
            upstream_z * source_importance.unsqueeze(-1)
        ).sum(dim=(0, 1)) / denom
        candidate_count = min(args.top_upstream_candidates, upstream_sae.d_sae)
        candidate_features = torch.argsort(upstream_rank_scores, descending=True)[:candidate_count]

        print(f"[trace] causal upstream screen over {candidate_features.numel()} features")
        for feat_idx in candidate_features.detach().cpu().tolist():
            delta = feature_delta_from_acts(upstream_sae, upstream_acts, feat_idx, dep_prompt_mask)
            capture: dict[str, torch.Tensor] = {}
            hooks = make_delta_hook_single_layer(delta, 1.0, upstream_hook)
            hooks.append((target_hook, capture_hook(capture, "target")))
            model.run_with_hooks(dep_tokens, return_type=None, fwd_hooks=hooks)
            patched_target = capture["target"]
            patched_score = score_from_mode(target_sae, patched_target, feature_idx, args.score_mode)
            patched_score_at_pos = patched_score[batch_idx, target_pos]
            causal_drop = baseline_score_at_pos - patched_score_at_pos

            weighted_tokens = defaultdict(float)
            feat_act = upstream_z[:, :, feat_idx]
            for row_idx in range(dep_tokens.shape[0]):
                prompt_len = int(dep_prompt_mask[row_idx].sum().item())
                for pos_idx in range(prompt_len):
                    tok_id = int(dep_tokens[row_idx, pos_idx].item())
                    weighted_tokens[tok_id] += float(
                        (source_importance[row_idx, pos_idx] * feat_act[row_idx, pos_idx]).item()
                    )

            upstream_rows.append(
                {
                    "feature_idx": int(feat_idx),
                    "mean_source_weighted_activation": sanitize_number(
                        upstream_rank_scores[feat_idx].item()
                    ),
                    "mean_causal_drop": sanitize_number(causal_drop.mean().item()),
                    "mean_abs_causal_drop": sanitize_number(causal_drop.abs().mean().item()),
                    "top_tokens": summarize_tokens(
                        tokenizer,
                        list(weighted_tokens.keys()),
                        list(weighted_tokens.values()),
                        top_k=args.top_sources,
                        display_cache=token_display_cache,
                    ),
                }
            )

        upstream_rows.sort(key=lambda row: row["mean_causal_drop"], reverse=True)
        upstream_rows = upstream_rows[: args.top_upstream_report]

    output = {
        "metadata": {
            "results_dir": str(results_dir),
            "checkpoint_dir": str(checkpoint_dir),
            "tokens_cache": str(tokens_cache_path),
            "split": args.split,
            "max_deployment": int(dep_tokens.shape[0]),
            "device": device,
            "score_mode": args.score_mode,
        },
        "target": {
            "target_arch": args.target_arch,
            "target_sae_path": str(target_sae_path),
            "target_hook": target_hook,
            "feature_idx": int(feature_idx),
            "used_preact_fallback_count": int(used_preact_fallback.sum().item()),
            "mean_target_activation": sanitize_number(baseline_activation_at_pos.mean().item()),
            "mean_target_preactivation": sanitize_number(baseline_preact_at_pos.mean().item()),
            "top_target_tokens": target_token_summary,
        },
        "decomposition": {
            "block_idx": block_idx,
            "target_hook_is_resid_mid": target_hook == resid_mid_hook,
            "attn_bias_linear": sanitize_number(attn_bias_linear),
            "mean_resid_pre_linear_at_target": sanitize_number(
                gather_at_positions(target_linear_pre, target_pos).mean().item()
            ),
            "mean_attn_linear_at_target": sanitize_number(attn_linear_at_pos.mean().item()),
            "mean_head_sum_plus_bias_at_target": sanitize_number(head_linear_plus_bias.mean().item()),
            "mean_target_linear_total_at_target": sanitize_number(
                gather_at_positions(target_linear_total, target_pos).mean().item()
            ),
            "mean_target_preactivation_at_target": sanitize_number(baseline_preact_at_pos.mean().item()),
            "mean_abs_attn_decomp_error": sanitize_number(
                (attn_linear_at_pos - head_linear_plus_bias).abs().mean().item()
            ),
        },
        "heads": {
            "rows": head_rows,
            "source_attribution": head_source_rows,
        },
        "upstream": {
            "enabled": analyze_upstream and upstream_sae is not None,
            "upstream_arch": args.upstream_arch if analyze_upstream else None,
            "upstream_sae_path": str(upstream_sae_path) if analyze_upstream else None,
            "upstream_hook": upstream_hook if analyze_upstream else None,
            "rows": upstream_rows,
        },
    }

    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(output, indent=2))

    head_md_lines = []
    for row in head_rows[: args.top_heads]:
        head_idx = row["head"]
        src = head_source_rows.get(head_idx, {})
        pos_toks = ", ".join(
            f"`{item['token_str']}` ({item['weight']:.3f})"
            for item in src.get("top_positive_tokens", [])[:4]
        )
        head_md_lines.append(
            "| "
            + " | ".join(
                [
                    str(head_idx),
                    format_signed(row["mean_linear_contrib"]),
                    format_signed(row["mean_causal_drop"]),
                    pos_toks or "-",
                ]
            )
            + " |"
        )

    upstream_md_lines = []
    for row in upstream_rows[: args.top_upstream_report]:
        toks = ", ".join(
            f"`{item['token_str']}` ({item['weight']:.3f})"
            for item in row.get("top_tokens", [])[:4]
        )
        upstream_md_lines.append(
            "| "
            + " | ".join(
                [
                    str(row["feature_idx"]),
                    f"{row['mean_source_weighted_activation']:.4f}",
                    format_signed(row["mean_causal_drop"]),
                    toks or "-",
                ]
            )
            + " |"
        )

    md = [
        "# Target-feature trace summary",
        "",
        "This report traces a chosen SAE feature as the target variable. "
        "Head and upstream effects are scored by the change they induce in the target "
        f"feature's `{args.score_mode}` at the selected prompt position.",
        "",
        "## Target",
        "",
        f"- target SAE: `{target_sae_path}`",
        f"- target hook: `{target_hook}`",
        f"- feature idx: `{feature_idx}`",
        f"- analyzed deployment prompts: `{dep_tokens.shape[0]}` from split `{args.split}`",
        f"- target-position fallback to preactivation used on `{int(used_preact_fallback.sum().item())}` prompts",
        f"- mean target preactivation at selected positions: `{baseline_preact_at_pos.mean().item():.4f}`",
        f"- mean target activation at selected positions: `{baseline_activation_at_pos.mean().item():.4f}`",
        "",
        "Top target-position tokens by target preactivation weight:",
        "",
    ]
    for item in target_token_summary:
        md.append(
            f"- `{item['token_str']}` (id={item['token_id']}): `{item['weight']:.4f}`"
        )

    md.extend(
        [
            "",
            "## Skip vs Attention",
            "",
            "| quantity | value |",
            "|---|---:|",
            f"| mean resid_pre linear score at target pos | {output['decomposition']['mean_resid_pre_linear_at_target']:.4f} |",
            f"| mean attention linear score at target pos | {output['decomposition']['mean_attn_linear_at_target']:.4f} |",
            f"| mean head-sum-plus-bias score at target pos | {output['decomposition']['mean_head_sum_plus_bias_at_target']:.4f} |",
            f"| mean target preactivation at target pos | {output['decomposition']['mean_target_preactivation_at_target']:.4f} |",
            f"| mean abs attention decomposition error | {output['decomposition']['mean_abs_attn_decomp_error']:.6f} |",
            "",
            "## Heads",
            "",
            "| head | mean linear contrib | mean causal drop | top positive source tokens |",
            "|---:|---:|---:|---|",
        ]
    )
    md.extend(head_md_lines or ["| - | - | - | - |"])

    if upstream_rows:
        md.extend(
            [
                "",
                "## Upstream Features",
                "",
                f"Upstream SAE: `{upstream_sae_path}` at `{upstream_hook}`. "
                "Candidates are ranked by source-weighted activation under the strongest heads, "
                "then causally screened by ablating that upstream feature and measuring the target drop.",
                "",
                "| feature | mean source-weighted act | mean causal drop | top weighted tokens |",
                "|---:|---:|---:|---|",
            ]
        )
        md.extend(upstream_md_lines)
    else:
        md.extend(
            [
                "",
                "## Upstream Features",
                "",
                "Upstream feature screening was skipped because no upstream checkpoint was provided or found.",
            ]
        )

    output_md.write_text("\n".join(md) + "\n")
    print(f"[trace] wrote {output_json}")
    print(f"[trace] wrote {output_md}")


if __name__ == "__main__":
    main()
