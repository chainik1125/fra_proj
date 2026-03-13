from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from fra.fra_func import get_sentence_fra_batch


@dataclass(frozen=True)
class AttentionReconstructionResult:
    layer: int
    head: int
    prompt: str
    seq_len: int
    top_k: int
    nnz: int
    device: str
    score_scale: float
    pattern_mae: float
    pattern_rmse: float
    pattern_error_pct: float
    score_mae: float | None
    score_rmse: float | None


@dataclass(frozen=True)
class AttentionReconstructionSummary:
    layer: int
    head: int
    num_prompts: int
    mean_pattern_mae: float
    max_pattern_mae: float
    min_pattern_mae: float
    mean_pattern_error_pct: float
    max_pattern_error_pct: float
    min_pattern_error_pct: float


def resolve_torch_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def reconstruct_attention_scores(
    fra_tensor_sparse: torch.Tensor,
    seq_len: int | None = None,
) -> torch.Tensor:
    fra_tensor_sparse = fra_tensor_sparse.coalesce()
    indices = fra_tensor_sparse.indices()
    values = fra_tensor_sparse.values()

    if seq_len is None:
        seq_len = int(fra_tensor_sparse.shape[0])

    scores = torch.zeros((seq_len, seq_len), device=values.device, dtype=values.dtype)
    if values.numel() == 0:
        return scores

    scores.index_put_((indices[0], indices[1]), values, accumulate=True)
    return scores


def scores_to_causal_pattern(scores: torch.Tensor) -> torch.Tensor:
    causal_mask = torch.triu(
        torch.ones(scores.shape, device=scores.device, dtype=torch.bool),
        diagonal=1,
    )
    masked_scores = scores.masked_fill(causal_mask, float("-inf"))
    return torch.nan_to_num(F.softmax(masked_scores, dim=-1), nan=0.0)


def summarize_reconstruction_results(
    results: list[AttentionReconstructionResult],
) -> AttentionReconstructionSummary:
    if not results:
        raise ValueError("Expected at least one reconstruction result to summarize.")

    pattern_maes = [result.pattern_mae for result in results]
    pattern_pcts = [result.pattern_error_pct for result in results]
    first = results[0]
    return AttentionReconstructionSummary(
        layer=first.layer,
        head=first.head,
        num_prompts=len(results),
        mean_pattern_mae=sum(pattern_maes) / len(pattern_maes),
        max_pattern_mae=max(pattern_maes),
        min_pattern_mae=min(pattern_maes),
        mean_pattern_error_pct=sum(pattern_pcts) / len(pattern_pcts),
        max_pattern_error_pct=max(pattern_pcts),
        min_pattern_error_pct=min(pattern_pcts),
    )


@torch.no_grad()
def measure_attention_reconstruction(
    model: HookedTransformer,
    sae: Any,
    prompt: str,
    *,
    layer: int,
    head: int,
    max_length: int = 128,
    top_k: int = 30,
) -> AttentionReconstructionResult:
    tokens = model.to_tokens(prompt, prepend_bos=False)
    if max_length is not None and tokens.shape[1] > max_length:
        tokens = tokens[:, :max_length]
    tokens = tokens.to(next(model.parameters()).device)

    captured: dict[str, torch.Tensor] = {}

    def capture_scores(scores: torch.Tensor, hook: Any) -> torch.Tensor:
        captured["scores"] = scores[0, head].detach()
        return scores

    def capture_pattern(pattern: torch.Tensor, hook: Any) -> torch.Tensor:
        captured["pattern"] = pattern[0, head].detach()
        return pattern

    hooks = [
        (f"blocks.{layer}.attn.hook_attn_scores", capture_scores),
        (f"blocks.{layer}.attn.hook_pattern", capture_pattern),
    ]
    with model.hooks(hooks):
        model(tokens)

    fra_result = get_sentence_fra_batch(
        model,
        sae,
        prompt,
        layer=layer,
        head=head,
        max_length=max_length,
        top_k=top_k,
        verbose=False,
    )
    fra_sparse = fra_result["fra_tensor_sparse"]
    seq_len = int(fra_result["seq_len"])

    reconstructed_unscaled = reconstruct_attention_scores(fra_sparse, seq_len=seq_len)
    score_scale = 1.0 / math.sqrt(model.cfg.d_head)
    reconstructed_scores = reconstructed_unscaled * score_scale
    reconstructed_pattern = scores_to_causal_pattern(reconstructed_scores)

    actual_pattern = captured["pattern"][:seq_len, :seq_len]
    pattern_diff = reconstructed_pattern - actual_pattern
    pattern_mae = float(pattern_diff.abs().mean().item())
    pattern_rmse = float(torch.sqrt(pattern_diff.pow(2).mean()).item())

    actual_scores = captured["scores"][:seq_len, :seq_len]
    valid_scores = actual_scores > -1e9
    if valid_scores.any():
        score_diff = reconstructed_scores[valid_scores] - actual_scores[valid_scores]
        score_mae = float(score_diff.abs().mean().item())
        score_rmse = float(torch.sqrt(score_diff.pow(2).mean()).item())
    else:
        score_mae = None
        score_rmse = None

    return AttentionReconstructionResult(
        layer=layer,
        head=head,
        prompt=prompt,
        seq_len=seq_len,
        top_k=top_k,
        nnz=int(fra_sparse._nnz()),
        device=str(tokens.device),
        score_scale=score_scale,
        pattern_mae=pattern_mae,
        pattern_rmse=pattern_rmse,
        pattern_error_pct=pattern_mae * 100.0,
        score_mae=score_mae,
        score_rmse=score_rmse,
    )
