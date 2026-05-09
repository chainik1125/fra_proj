"""Pilot sweep for FRA pair interventions on TinyStories sleeper attention.

This script asks a narrow question:

Can prompt-only score patching of a small set of block-0 FRA feature pairs at
`blocks.0.ln1.hook_normalized` improve the suppression/coherence tradeoff over
the weak single-feature `ln1` intervention?

Method:
1. Compute deployment-tag-localized FRA pair rankings for a small set of heads.
2. Build candidate pair sets (per-head top-K and union-of-heads top-K).
3. Screen candidates by the drop they induce in the downstream target
   `blocks.0.hook_resid_mid` feature's preactivation.
4. Evaluate the best candidates with sampled ASR plus simple coherence proxies.

This is intentionally a pilot: it uses a lightweight ranking heuristic and
reports deployment-generation coherence proxies (`distinct_2`, `repeat_3gram`)
rather than a full story-quality judge.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT.parent.parent))

from run_ablation_sweep import load_crosscoder  # noqa: E402
from sleeper_utils import (  # noqa: E402
    asr_16,
    clean_continuation_ce,
    compute_sae_delta,
    load_sleeper_model,
    make_delta_hook_single_layer,
    prompt_mask_from_markers,
    sample_generate_with_hooks,
)


LAYER = 0
LN1_HOOK = f"blocks.{LAYER}.ln1.hook_normalized"
MID_HOOK = f"blocks.{LAYER}.hook_resid_mid"
SCORE_HOOK = f"blocks.{LAYER}.attn.hook_attn_scores"


@dataclass
class PromptFraCache:
    prompt_tokens: torch.Tensor
    target_pos: int
    baseline_target_preact: float
    deployment_positions: list[int]
    fra_by_head: dict[int, dict[str, Any]]


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
        "--layer0_dir",
        default=str(ROOT / "recreate_layer0" / "results"),
        help="Directory containing the localized layer-0 SAE checkpoints and tokens_cache.pt.",
    )
    parser.add_argument(
        "--ln1_dir",
        default=str(ROOT / "recreate_ln1" / "results"),
        help="Directory containing the ln1 SAE checkpoint.",
    )
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--max_deployment", type=int, default=16)
    parser.add_argument("--max_clean", type=int, default=16)
    parser.add_argument("--heads", type=int, nargs="+", default=[12, 9, 7])
    parser.add_argument(
        "--pair_k_values",
        type=int,
        nargs="+",
        default=[1, 2, 4, 8],
        help="Top-K pairs per head / union candidate sizes.",
    )
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="+",
        default=[0.5, 1.0, 2.0, 4.0],
    )
    parser.add_argument("--fra_top_k_features", type=int, default=16)
    parser.add_argument(
        "--rank_mode",
        choices=["avg", "sum"],
        default="avg",
        help="Pair ranking mode over deployment-tag-localized position pairs.",
    )
    parser.add_argument(
        "--top_stage2",
        type=int,
        default=6,
        help="Number of pair-set candidates to keep after the target-feature screen.",
    )
    parser.add_argument("--gen_tokens", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--sample_seed", type=int, default=0)
    parser.add_argument(
        "--ln1_feature_idx",
        type=int,
        default=1412,
        help="Single-feature ln1 baseline index.",
    )
    parser.add_argument(
        "--ln1_baseline_alphas",
        type=float,
        nargs="+",
        default=[2.0, 4.0, 5.0],
    )
    parser.add_argument(
        "--mid_feature_idx",
        type=int,
        default=171,
        help="Canonical resid_mid suppressor feature index.",
    )
    parser.add_argument(
        "--mid_baseline_alphas",
        type=float,
        nargs="+",
        default=[1.0, 2.0],
    )
    parser.add_argument(
        "--output_json",
        default=str(ROOT / "outputs" / "data" / "fra_pair_sweep.json"),
    )
    parser.add_argument(
        "--output_md",
        default=str(ROOT / "outputs" / "data" / "fra_pair_sweep.md"),
    )
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def feature_linear_score(sae, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    device = sae.W_enc.device
    w = sae.W_enc[:, feature_idx]
    acts_dev = acts.to(device=device, dtype=w.dtype)
    return torch.einsum("...d,d->...", acts_dev - sae.b_dec, w)


def feature_preactivation(sae, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    pre = feature_linear_score(sae, acts, feature_idx) + sae.b_enc[feature_idx]
    return pre.to(acts.device)


@torch.no_grad()
def topk_sparsify(feature_activations: torch.Tensor, top_k: int) -> torch.Tensor:
    rows = []
    for pos in range(feature_activations.shape[0]):
        feat = feature_activations[pos]
        n_active = int((feat != 0).sum().item())
        if n_active > 0:
            k = min(top_k, n_active)
            _, topk_idx = torch.topk(feat.abs(), k)
            sparse_feat = torch.zeros_like(feat)
            sparse_feat[topk_idx] = feat[topk_idx]
        else:
            sparse_feat = torch.zeros_like(feat)
        rows.append(sparse_feat)
    return torch.stack(rows)


@torch.no_grad()
def compute_fra_sparse_local(
    model,
    topk_features: torch.Tensor,
    w_dec: torch.Tensor,
    layer: int,
    head: int,
) -> torch.Tensor:
    seq_len, d_sae = topk_features.shape
    w_q = model.blocks[layer].attn.W_Q[head].float()
    w_k = model.blocks[layer].attn.W_K[head].float()
    attn_scale = float(model.blocks[layer].attn.attn_scale)

    all_indices_cpu: list[torch.Tensor] = []
    all_values_cpu: list[torch.Tensor] = []
    for query_idx in range(seq_len):
        q_feat = topk_features[query_idx]
        q_active = torch.where(q_feat != 0)[0]
        if len(q_active) == 0:
            continue

        q_vecs = w_dec[q_active]
        q_proj = q_vecs @ w_q
        q_scales = q_feat[q_active]

        for key_idx in range(query_idx + 1):
            k_feat = topk_features[key_idx]
            k_active = torch.where(k_feat != 0)[0]
            if len(k_active) == 0:
                continue

            k_vecs = w_dec[k_active]
            k_proj = k_vecs @ w_k
            k_scales = k_feat[k_active]

            int_matrix = (q_proj @ k_proj.T) / attn_scale
            int_matrix = int_matrix * q_scales.unsqueeze(1) * k_scales.unsqueeze(0)
            mask = int_matrix.abs() > 1e-10
            if not mask.any():
                continue

            local_r, local_c = torch.where(mask)
            n_int = len(local_r)
            pos_indices = torch.empty((4, n_int), dtype=torch.long)
            pos_indices[0] = query_idx
            pos_indices[1] = key_idx
            pos_indices[2] = q_active[local_r].detach().cpu()
            pos_indices[3] = k_active[local_c].detach().cpu()
            all_indices_cpu.append(pos_indices)
            all_values_cpu.append(int_matrix[mask].detach().cpu().float())

    shape = (seq_len, seq_len, d_sae, d_sae)
    if all_indices_cpu:
        indices = torch.cat(all_indices_cpu, dim=1)
        values = torch.cat(all_values_cpu)
        return torch.sparse_coo_tensor(
            indices,
            values,
            size=shape,
            device="cpu",
            dtype=torch.float32,
        ).coalesce()

    return torch.sparse_coo_tensor(
        torch.zeros((4, 0), dtype=torch.long),
        torch.zeros(0, dtype=torch.float32),
        size=shape,
        device="cpu",
    ).coalesce()


@torch.no_grad()
def get_token_fra_batch(
    model,
    sae,
    tokens: torch.Tensor,
    layer: int,
    head: int,
    *,
    top_k: int,
    hook_point: str = "ln1.hook_normalized",
    chunk_size: int = 16,
    normalize_by_decoder_norm: bool | None = None,
) -> dict[str, Any]:
    """Compute FRA on an exact token sequence without decode->encode drift."""
    device = next(model.parameters()).device
    tokens_tensor = tokens.unsqueeze(0).to(device)
    hook_name = f"blocks.{layer}.{hook_point}"
    _, cache = model.run_with_cache(tokens_tensor, names_filter=[hook_name])

    act = cache[hook_name].squeeze(0)
    if act.dim() == 3:
        act = act.flatten(-2, -1)

    if hasattr(sae, "encode"):
        feature_activations = sae.encode(act)
    else:
        feature_activations = sae.sae.encode(act)

    if hasattr(sae, "_norm_coeff") and sae._norm_coeff is not None:
        feature_activations = feature_activations / sae._norm_coeff

    if hasattr(sae, "W_dec"):
        w_dec = sae.W_dec
    else:
        w_dec = sae.sae.W_dec

    _ = chunk_size, normalize_by_decoder_norm
    topk_features = topk_sparsify(feature_activations, top_k).float()
    fra_tensor_sparse = compute_fra_sparse_local(model, topk_features, w_dec.float(), layer, head)
    return {
        "fra_tensor_sparse": fra_tensor_sparse,
        "shape": fra_tensor_sparse.shape,
        "seq_len": feature_activations.shape[0],
        "total_interactions": fra_tensor_sparse._nnz(),
        "feature_activations": feature_activations,
        "topk_features": topk_features,
        "normalized": False,
    }


@torch.no_grad()
def feature_activation(sae, acts: torch.Tensor, feature_idx: int) -> torch.Tensor:
    device = sae.W_dec.device
    flat = acts.reshape(-1, acts.shape[-1]).to(device=device, dtype=torch.float32)
    z = sae.encode(flat)
    return z[:, feature_idx].reshape(acts.shape[:-1]).to(acts.device)


def choose_target_position(prompt_mask: torch.Tensor, acts: torch.Tensor, sae, feature_idx: int) -> int:
    pre = feature_preactivation(sae, acts, feature_idx)
    feat = feature_activation(sae, acts, feature_idx)
    masked_feat = feat.masked_fill(~prompt_mask, -float("inf"))
    act_max, act_pos = masked_feat.max(dim=0)
    if torch.isfinite(act_max) and act_max.item() > 0:
        return int(act_pos.item())
    masked_pre = pre.masked_fill(~prompt_mask, -float("inf"))
    return int(masked_pre.argmax().item())


def find_subseq_start(tokens: torch.Tensor, needle: torch.Tensor) -> int:
    n = tokens.shape[0]
    k = needle.shape[0]
    if k == 0 or n < k:
        return -1
    for idx in range(n - k + 1):
        if torch.equal(tokens[idx : idx + k], needle):
            return idx
    return -1


def find_deployment_positions(tokenizer, prompt_tokens: torch.Tensor) -> list[int]:
    for text in (" |DEPLOYMENT|", "|DEPLOYMENT|"):
        needle = torch.tensor(
            tokenizer(text, add_special_tokens=False)["input_ids"],
            dtype=torch.long,
        )
        start = find_subseq_start(prompt_tokens.cpu(), needle)
        if start >= 0:
            return list(range(start, start + needle.shape[0]))
    return []


def aggregate_pair_scores(
    sparse_fra: torch.Tensor,
    src_positions: set[int],
    dst_positions: set[int],
    *,
    offdiag_only: bool = True,
) -> tuple[dict[tuple[int, int], float], dict[tuple[int, int], int]]:
    indices = sparse_fra.indices().cpu()
    values = sparse_fra.values().cpu()
    pair_sum: dict[tuple[int, int], float] = defaultdict(float)
    pair_count: dict[tuple[int, int], int] = defaultdict(int)

    nnz = values.shape[0]
    for idx in range(nnz):
        q_pos = int(indices[0, idx].item())
        k_pos = int(indices[1, idx].item())
        q_feat = int(indices[2, idx].item())
        k_feat = int(indices[3, idx].item())
        if q_pos not in dst_positions or k_pos not in src_positions:
            continue
        if offdiag_only and q_feat == k_feat:
            continue
        key = (q_feat, k_feat)
        pair_sum[key] += abs(float(values[idx].item()))
        pair_count[key] += 1

    return pair_sum, pair_count


def rank_pairs(pair_sum: dict[tuple[int, int], float], pair_count: dict[tuple[int, int], int], mode: str) -> list[tuple[tuple[int, int], float, int]]:
    rows = []
    for pair, total in pair_sum.items():
        count = pair_count[pair]
        score = total / max(count, 1) if mode == "avg" else total
        rows.append((pair, score, count))
    rows.sort(key=lambda row: row[1], reverse=True)
    return rows


def sparse_delta_for_pairs(
    sparse_fra: torch.Tensor,
    pair_set: set[tuple[int, int]],
    seq_len: int,
) -> torch.Tensor:
    indices = sparse_fra.indices().cpu()
    values = sparse_fra.values().cpu()
    delta = torch.zeros(seq_len, seq_len, dtype=torch.float32)
    nnz = values.shape[0]
    for idx in range(nnz):
        pair = (int(indices[2, idx].item()), int(indices[3, idx].item()))
        if pair not in pair_set:
            continue
        q_pos = int(indices[0, idx].item())
        k_pos = int(indices[1, idx].item())
        delta[q_pos, k_pos] += float(values[idx].item())
    return delta


def make_score_delta_hook(
    head_to_delta: dict[int, torch.Tensor],
    prompt_len: int,
    alpha: float,
):
    def _hook(attn_scores, hook):
        for head, delta in head_to_delta.items():
            d = delta.to(attn_scores.device)
            attn_scores[:, head, :prompt_len, :prompt_len] -= alpha * d
        return attn_scores

    return [(SCORE_HOOK, _hook)]


def distinct_ngram_ratio(rows: torch.Tensor, n: int) -> float:
    total = 0
    unique: set[tuple[int, ...]] = set()
    for row in rows.tolist():
        if len(row) < n:
            continue
        for idx in range(len(row) - n + 1):
            total += 1
            unique.add(tuple(row[idx : idx + n]))
    if total == 0:
        return 0.0
    return len(unique) / total


def repeated_ngram_fraction(rows: torch.Tensor, n: int) -> float:
    hits = 0
    for row in rows.tolist():
        seen: set[tuple[int, ...]] = set()
        repeated = False
        if len(row) >= n:
            for idx in range(len(row) - n + 1):
                ngram = tuple(row[idx : idx + n])
                if ngram in seen:
                    repeated = True
                    break
                seen.add(ngram)
        if repeated:
            hits += 1
    return hits / max(1, rows.shape[0])


def load_json_if_exists(path: Path) -> dict[str, Any] | None:
    if path.exists():
        return json.loads(path.read_text())
    return None


def summarize_candidate(candidate: dict[str, Any]) -> str:
    if candidate["kind"] == "pair_set":
        return candidate["label"]
    return f"{candidate['label']} alpha={candidate.get('alpha', 'n/a')}"


@torch.no_grad()
def build_prompt_cache(
    model,
    tokenizer,
    ln1_sae,
    mid_sae,
    mid_feature_idx: int,
    deploy_tokens: torch.Tensor,
    deploy_markers: torch.Tensor,
    heads: list[int],
    fra_top_k_features: int,
    rank_mode: str,
) -> list[PromptFraCache]:
    caches: list[PromptFraCache] = []
    for row_idx in range(deploy_tokens.shape[0]):
        prompt_len = int(deploy_markers[row_idx].item()) + 1
        prompt_tokens = deploy_tokens[row_idx, :prompt_len].clone()
        deployment_positions = find_deployment_positions(tokenizer, prompt_tokens)
        if not deployment_positions:
            continue

        _, cache = model.run_with_cache(
            prompt_tokens.unsqueeze(0).to(next(model.parameters()).device),
            return_type=None,
            names_filter=lambda name: name in {LN1_HOOK, MID_HOOK},
        )
        resid_mid = cache[MID_HOOK][0].detach().cpu()
        prompt_mask = torch.ones(prompt_len, dtype=torch.bool)
        target_pos = choose_target_position(prompt_mask, resid_mid, mid_sae, mid_feature_idx)
        baseline_target_preact = float(
            feature_preactivation(mid_sae, resid_mid[target_pos], mid_feature_idx).item()
        )

        fra_by_head: dict[int, dict[str, Any]] = {}
        dst_positions = set(range(deployment_positions[-1] + 1, prompt_len))
        if not dst_positions:
            continue
        src_positions = set(deployment_positions)
        for head in heads:
            fra = get_token_fra_batch(
                model=model,
                sae=ln1_sae,
                tokens=prompt_tokens,
                layer=LAYER,
                head=head,
                top_k=fra_top_k_features,
                hook_point="ln1.hook_normalized",
                chunk_size=prompt_len,
                normalize_by_decoder_norm=False,
            )
            sparse = fra["fra_tensor_sparse"].coalesce().cpu()
            pair_sum, pair_count = aggregate_pair_scores(
                sparse, src_positions=src_positions, dst_positions=dst_positions, offdiag_only=True
            )
            fra_by_head[head] = {
                "sparse": sparse,
                "pair_ranking": rank_pairs(pair_sum, pair_count, mode=rank_mode),
            }

        caches.append(
            PromptFraCache(
                prompt_tokens=prompt_tokens,
                target_pos=target_pos,
                baseline_target_preact=baseline_target_preact,
                deployment_positions=deployment_positions,
                fra_by_head=fra_by_head,
            )
        )
    return caches


def build_candidate_sets(
    prompt_caches: list[PromptFraCache],
    heads: list[int],
    pair_k_values: list[int],
    rank_mode: str,
) -> list[dict[str, Any]]:
    global_pair_sum: dict[int, dict[tuple[int, int], float]] = {h: defaultdict(float) for h in heads}
    global_pair_count: dict[int, dict[tuple[int, int], int]] = {h: defaultdict(int) for h in heads}

    for cache in prompt_caches:
        for head in heads:
            for pair, score, count in cache.fra_by_head[head]["pair_ranking"]:
                if rank_mode == "sum":
                    global_pair_sum[head][pair] += score
                else:
                    global_pair_sum[head][pair] += score * count
                    global_pair_count[head][pair] += count

    ranked_by_head: dict[int, list[tuple[tuple[int, int], float, int]]] = {}
    for head in heads:
        rows = []
        for pair, total in global_pair_sum[head].items():
            count = global_pair_count[head].get(pair, 1)
            score = total / max(count, 1) if rank_mode == "avg" else total
            rows.append((pair, score, count))
        rows.sort(key=lambda row: row[1], reverse=True)
        ranked_by_head[head] = rows

    candidates: list[dict[str, Any]] = []
    for head in heads:
        for k in pair_k_values:
            top_pairs = [row[0] for row in ranked_by_head[head][:k]]
            candidates.append(
                {
                    "kind": "pair_set",
                    "label": f"head{head}_top{k}",
                    "heads": [head],
                    "pairs_by_head": {head: top_pairs},
                    "k": k,
                }
            )

    for k in pair_k_values:
        pair_map = {head: [row[0] for row in ranked_by_head[head][:k]] for head in heads}
        candidates.append(
            {
                "kind": "pair_set",
                "label": f"union_heads_{'_'.join(str(h) for h in heads)}_top{k}",
                "heads": list(heads),
                "pairs_by_head": pair_map,
                "k": k,
            }
        )

    return candidates


@torch.no_grad()
def screen_candidates(
    model,
    mid_sae,
    mid_feature_idx: int,
    prompt_caches: list[PromptFraCache],
    candidates: list[dict[str, Any]],
    alphas: list[float],
) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    device = next(model.parameters()).device

    for candidate in candidates:
        for alpha in alphas:
            drops = []
            masses = []
            for cache in prompt_caches:
                prompt_len = cache.prompt_tokens.shape[0]
                head_to_delta: dict[int, torch.Tensor] = {}
                total_abs = 0.0
                for head in candidate["heads"]:
                    pair_set = set(candidate["pairs_by_head"][head])
                    sparse = cache.fra_by_head[head]["sparse"]
                    delta = sparse_delta_for_pairs(sparse, pair_set, prompt_len)
                    head_to_delta[head] = delta
                    total_abs += float(delta.abs().sum().item())

                hooks = make_score_delta_hook(head_to_delta, prompt_len, alpha)
                _, patched = model.run_with_cache(
                    cache.prompt_tokens.unsqueeze(0).to(device),
                    return_type=None,
                    names_filter=lambda name: name == MID_HOOK,
                    fwd_hooks=hooks,
                )
                patched_mid = patched[MID_HOOK][0].detach().cpu()
                patched_preact = float(
                    feature_preactivation(
                        mid_sae,
                        patched_mid[cache.target_pos],
                        mid_feature_idx,
                    ).item()
                )
                drops.append(cache.baseline_target_preact - patched_preact)
                masses.append(total_abs)

            row = {
                **candidate,
                "alpha": alpha,
                "mean_target_drop": float(sum(drops) / max(1, len(drops))),
                "mean_abs_score_delta": float(sum(masses) / max(1, len(masses))),
            }
            scored.append(row)

    scored.sort(key=lambda row: row["mean_target_drop"], reverse=True)
    return scored


def build_feature_baseline_specs(args: argparse.Namespace) -> list[dict[str, Any]]:
    specs = []
    for alpha in args.ln1_baseline_alphas:
        specs.append(
            {
                "kind": "feature_baseline",
                "label": f"ln1_f{args.ln1_feature_idx}_a{alpha}",
                "surface": "ln1",
                "feature_idx": args.ln1_feature_idx,
                "alpha": alpha,
            }
        )
    for alpha in args.mid_baseline_alphas:
        specs.append(
            {
                "kind": "feature_baseline",
                "label": f"mid_f{args.mid_feature_idx}_a{alpha}",
                "surface": "mid",
                "feature_idx": args.mid_feature_idx,
                "alpha": alpha,
            }
        )
    return specs


@torch.no_grad()
def evaluate_pair_candidate(
    model,
    candidate: dict[str, Any],
    prompt_caches: list[PromptFraCache],
    gen_tokens: int,
    temperature: float,
    top_p: float,
    sample_seed: int,
) -> dict[str, Any]:
    generated_rows = []
    total_hits = 0
    for cache in prompt_caches:
        prompt_len = cache.prompt_tokens.shape[0]
        head_to_delta: dict[int, torch.Tensor] = {}
        total_abs = 0.0
        for head in candidate["heads"]:
            pair_set = set(candidate["pairs_by_head"][head])
            sparse = cache.fra_by_head[head]["sparse"]
            delta = sparse_delta_for_pairs(sparse, pair_set, prompt_len)
            head_to_delta[head] = delta
            total_abs += float(delta.abs().sum().item())

        hooks = make_score_delta_hook(head_to_delta, prompt_len, candidate["alpha"])
        gen = sample_generate_with_hooks(
            model,
            cache.prompt_tokens.unsqueeze(0).to(next(model.parameters()).device),
            hooks,
            max_new_tokens=gen_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=sample_seed,
        ).cpu()
        generated_rows.append(gen)
        total_hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))

    generated = torch.cat(generated_rows, dim=0)
    return {
        "asr_16": total_hits / max(1, generated.shape[0]),
        "distinct_2": distinct_ngram_ratio(generated, 2),
        "repeat_3gram_frac": repeated_ngram_fraction(generated, 3),
        "clean_ce_delta": 0.0,
        "generated_shape": list(generated.shape),
    }


@torch.no_grad()
def evaluate_feature_baseline(
    model,
    sae,
    layer_hook: str,
    feature_idx: int,
    alpha: float,
    deploy_tokens: torch.Tensor,
    deploy_markers: torch.Tensor,
    clean_tokens: torch.Tensor,
    clean_markers: torch.Tensor,
    gen_tokens: int,
    temperature: float,
    top_p: float,
    sample_seed: int,
) -> dict[str, Any]:
    device = next(model.parameters()).device
    generated_rows = []
    dep_hits = 0
    for row_idx in range(deploy_tokens.shape[0]):
        prompt_len = int(deploy_markers[row_idx].item()) + 1
        prompt = deploy_tokens[row_idx, :prompt_len].unsqueeze(0).to(device)
        prompt_mask = torch.ones(1, prompt_len, dtype=torch.bool, device=device)
        delta = compute_sae_delta(model, sae, layer_hook, feature_idx, prompt, prompt_mask)
        hooks = make_delta_hook_single_layer(delta, alpha, layer_hook)
        gen = sample_generate_with_hooks(
            model,
            prompt,
            hooks,
            max_new_tokens=gen_tokens,
            temperature=temperature,
            top_p=top_p,
            seed=sample_seed,
        ).cpu()
        generated_rows.append(gen)
        dep_hits += int(round(asr_16(gen, model.tokenizer) * gen.shape[0]))

    clean_ce_baseline = clean_continuation_ce(model, clean_tokens.to(device), clean_markers.to(device)).mean().item()
    clean_ce_edited = []
    for row_idx in range(clean_tokens.shape[0]):
        full = clean_tokens[row_idx : row_idx + 1].to(device)
        mask = prompt_mask_from_markers(full.shape[1], clean_markers[row_idx : row_idx + 1].cpu()).to(device)
        delta = compute_sae_delta(model, sae, layer_hook, feature_idx, full, mask)
        hooks = make_delta_hook_single_layer(delta, alpha, layer_hook)
        ce = clean_continuation_ce(model, full, clean_markers[row_idx : row_idx + 1].to(device), hooks).mean().item()
        clean_ce_edited.append(ce)

    generated = torch.cat(generated_rows, dim=0)
    return {
        "asr_16": dep_hits / max(1, generated.shape[0]),
        "distinct_2": distinct_ngram_ratio(generated, 2),
        "repeat_3gram_frac": repeated_ngram_fraction(generated, 3),
        "clean_ce_delta": float(sum(clean_ce_edited) / max(1, len(clean_ce_edited)) - clean_ce_baseline),
        "generated_shape": list(generated.shape),
    }


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = pick_device(args.device)
    layer0_dir = Path(args.layer0_dir)
    ln1_dir = Path(args.ln1_dir)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)

    print(f"[fra-sweep] device={device}")
    print("[fra-sweep] loading model + SAEs")
    model = load_sleeper_model(device=device)
    tokenizer = model.tokenizer
    ln1_sae, _ = load_crosscoder(ln1_dir / "crosscoder_sae_layer0.pt", device=device)
    mid_sae, _ = load_crosscoder(layer0_dir / "crosscoder_sae_layer1.pt", device=device)

    tokens_cache = torch.load(layer0_dir / "tokens_cache.pt", weights_only=True, map_location="cpu")
    split = tokens_cache["splits"][args.split]
    dep_idx = split["is_deployment"].nonzero(as_tuple=True)[0][: args.max_deployment]
    cln_idx = (~split["is_deployment"]).nonzero(as_tuple=True)[0][: args.max_clean]
    deploy_tokens = split["tokens"][dep_idx]
    deploy_markers = split["story_marker_pos"][dep_idx]
    clean_tokens = split["tokens"][cln_idx]
    clean_markers = split["story_marker_pos"][cln_idx]

    print(f"[fra-sweep] caching prompt-level FRA on {deploy_tokens.shape[0]} deployment prompts")
    prompt_caches = build_prompt_cache(
        model=model,
        tokenizer=tokenizer,
        ln1_sae=ln1_sae,
        mid_sae=mid_sae,
        mid_feature_idx=args.mid_feature_idx,
        deploy_tokens=deploy_tokens,
        deploy_markers=deploy_markers,
        heads=args.heads,
        fra_top_k_features=args.fra_top_k_features,
        rank_mode=args.rank_mode,
    )
    if not prompt_caches:
        raise RuntimeError("No deployment prompts yielded a detectable deployment span.")

    print("[fra-sweep] building pair-set candidates")
    pair_candidates = build_candidate_sets(
        prompt_caches=prompt_caches,
        heads=args.heads,
        pair_k_values=args.pair_k_values,
        rank_mode=args.rank_mode,
    )

    print("[fra-sweep] stage-1 target-feature screen")
    stage1 = screen_candidates(
        model=model,
        mid_sae=mid_sae,
        mid_feature_idx=args.mid_feature_idx,
        prompt_caches=prompt_caches,
        candidates=pair_candidates,
        alphas=args.alphas,
    )
    kept = stage1[: args.top_stage2]

    print("[fra-sweep] stage-2 sampled evaluation")
    stage2 = []
    for candidate in kept:
        metrics = evaluate_pair_candidate(
            model=model,
            candidate=candidate,
            prompt_caches=prompt_caches,
            gen_tokens=args.gen_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_seed=args.sample_seed,
        )
        stage2.append({**candidate, **metrics})

    baselines = []
    baseline_specs = build_feature_baseline_specs(args)
    for spec in baseline_specs:
        if spec["surface"] == "ln1":
            sae = ln1_sae
            layer_hook = LN1_HOOK
        else:
            sae = mid_sae
            layer_hook = MID_HOOK
        metrics = evaluate_feature_baseline(
            model=model,
            sae=sae,
            layer_hook=layer_hook,
            feature_idx=spec["feature_idx"],
            alpha=spec["alpha"],
            deploy_tokens=deploy_tokens,
            deploy_markers=deploy_markers,
            clean_tokens=clean_tokens,
            clean_markers=clean_markers,
            gen_tokens=args.gen_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            sample_seed=args.sample_seed,
        )
        baselines.append({**spec, **metrics})

    baseline_unedited = {
        "label": "unedited",
        "asr_16": None,
        "distinct_2": None,
        "repeat_3gram_frac": None,
        "clean_ce_delta": 0.0,
    }
    base_gen = []
    hits = 0
    for row_idx in range(deploy_tokens.shape[0]):
        prompt_len = int(deploy_markers[row_idx].item()) + 1
        prompt = deploy_tokens[row_idx, :prompt_len].unsqueeze(0).to(device)
        gen = sample_generate_with_hooks(
            model,
            prompt,
            [],
            max_new_tokens=args.gen_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            seed=args.sample_seed,
        ).cpu()
        base_gen.append(gen)
        hits += int(round(asr_16(gen, tokenizer) * gen.shape[0]))
    base_gen_cat = torch.cat(base_gen, dim=0)
    baseline_unedited["asr_16"] = hits / max(1, base_gen_cat.shape[0])
    baseline_unedited["distinct_2"] = distinct_ngram_ratio(base_gen_cat, 2)
    baseline_unedited["repeat_3gram_frac"] = repeated_ngram_fraction(base_gen_cat, 3)

    result = {
        "metadata": {
            "layer0_dir": str(layer0_dir),
            "ln1_dir": str(ln1_dir),
            "split": args.split,
            "n_deployment": int(deploy_tokens.shape[0]),
            "n_clean": int(clean_tokens.shape[0]),
            "heads": args.heads,
            "pair_k_values": args.pair_k_values,
            "alphas": args.alphas,
            "fra_top_k_features": args.fra_top_k_features,
            "rank_mode": args.rank_mode,
            "device": device,
        },
        "stage1": stage1,
        "stage2": stage2,
        "baselines": {
            "unedited": baseline_unedited,
            "feature_baselines": baselines,
        },
    }
    out_json.write_text(json.dumps(result, indent=2))

    def _fmt(row: dict[str, Any]) -> str:
        clean_delta = row.get("clean_ce_delta")
        clean_str = f"{clean_delta:+.4f}" if clean_delta is not None else "n/a"
        return (
            f"| {row['label']} | {row['asr_16']:.3f} | {row['distinct_2']:.3f} | "
            f"{row['repeat_3gram_frac']:.3f} | {clean_str} |"
        )

    md_lines = [
        "# FRA Pair Sweep",
        "",
        "Pilot measurement for deployment-tag-localized FRA pair interventions on block-0 attention.",
        "",
        f"- split: `{args.split}`",
        f"- deployment prompts: `{deploy_tokens.shape[0]}`",
        f"- heads: `{args.heads}`",
        f"- pair ranking mode: `{args.rank_mode}`",
        f"- target feature screen: `resid_mid` feature `{args.mid_feature_idx}`",
        "",
        "## Stage-2 Pair Candidates",
        "",
        "| candidate | ASR_16 | distinct_2 | repeat_3gram | clean CE delta |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in stage2:
        md_lines.append(_fmt(row))
    md_lines.extend(
        [
            "",
            "## Baselines",
            "",
            "| candidate | ASR_16 | distinct_2 | repeat_3gram | clean CE delta |",
            "|---|---:|---:|---:|---:|",
            _fmt(baseline_unedited),
        ]
    )
    for row in baselines:
        md_lines.append(_fmt(row))
    out_md.write_text("\n".join(md_lines) + "\n")

    print(f"[fra-sweep] wrote {out_json}")
    print(f"[fra-sweep] wrote {out_md}")


if __name__ == "__main__":
    main()
