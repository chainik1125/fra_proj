from __future__ import annotations

import pytest
import torch

from fra.reconstruction import (
    AttentionReconstructionResult,
    reconstruct_attention_scores,
    scores_to_causal_pattern,
    summarize_reconstruction_results,
)


def test_reconstruct_attention_scores_sums_sparse_values_by_position() -> None:
    indices = torch.tensor(
        [
            [1, 1, 2],
            [0, 0, 1],
            [5, 7, 9],
            [4, 6, 8],
        ],
        dtype=torch.long,
    )
    values = torch.tensor([0.25, -0.10, 0.50], dtype=torch.float32)
    fra_sparse = torch.sparse_coo_tensor(indices, values, size=(3, 3, 10, 10)).coalesce()

    scores = reconstruct_attention_scores(fra_sparse)

    expected = torch.zeros((3, 3), dtype=torch.float32)
    expected[1, 0] = 0.15
    expected[2, 1] = 0.50
    assert torch.allclose(scores.cpu(), expected)


def test_scores_to_causal_pattern_masks_future_positions() -> None:
    scores = torch.tensor(
        [
            [2.0, 1.0, 0.0],
            [3.0, 1.0, 2.0],
            [0.5, 0.0, -0.5],
        ],
        dtype=torch.float32,
    )

    pattern = scores_to_causal_pattern(scores)

    assert torch.allclose(pattern[0], torch.tensor([1.0, 0.0, 0.0]))
    assert pattern[1, 2].item() == 0.0
    assert torch.allclose(pattern.sum(dim=-1), torch.ones(3))


def test_summarize_reconstruction_results_aggregates_pattern_error() -> None:
    results = [
        AttentionReconstructionResult(
            layer=5,
            head=0,
            prompt="a",
            seq_len=4,
            top_k=30,
            nnz=10,
            device="cpu",
            score_scale=0.125,
            pattern_mae=0.10,
            pattern_rmse=0.12,
            pattern_error_pct=10.0,
            score_mae=None,
            score_rmse=None,
        ),
        AttentionReconstructionResult(
            layer=5,
            head=0,
            prompt="b",
            seq_len=5,
            top_k=30,
            nnz=12,
            device="cpu",
            score_scale=0.125,
            pattern_mae=0.20,
            pattern_rmse=0.22,
            pattern_error_pct=20.0,
            score_mae=None,
            score_rmse=None,
        ),
    ]

    summary = summarize_reconstruction_results(results)

    assert summary.num_prompts == 2
    assert summary.mean_pattern_mae == pytest.approx(0.15)
    assert summary.mean_pattern_error_pct == pytest.approx(15.0)
    assert summary.min_pattern_error_pct == pytest.approx(10.0)
    assert summary.max_pattern_error_pct == pytest.approx(20.0)
