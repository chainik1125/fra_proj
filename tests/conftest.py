"""Pytest configuration and candidate adapters for FRA tests."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from fra.fra_func import get_sentence_fra_batch
from tests.fra_conformance.contracts import (
    CandidateFRA,
    FRAConformanceCase,
    FRAConformanceResult,
)


def _normalize_sparse_result(
    result: FRAConformanceResult | Mapping[str, Any],
) -> FRAConformanceResult:
    """Normalize candidate output into the conformance result contract."""
    if isinstance(result, FRAConformanceResult):
        sparse = result.fra_tensor_sparse.coalesce()
        return FRAConformanceResult(
            fra_tensor_sparse=sparse,
            shape=tuple(result.shape),
            seq_len=int(result.seq_len),
        )

    if not isinstance(result, Mapping):
        raise TypeError(
            "Candidate FRA result must be either FRAConformanceResult or a mapping-like object."
        )

    sparse = None
    for key in ("fra_tensor_sparse", "data_dep_int_matrix", "sparse_fra"):
        if key in result and result[key] is not None:
            sparse = result[key]
            break
    if sparse is None:
        raise KeyError(
            "Candidate FRA result must include 'fra_tensor_sparse' or a supported legacy sparse key."
        )
    if not isinstance(sparse, torch.Tensor):
        raise TypeError(
            f"Candidate FRA sparse result must be a torch.Tensor, got {type(sparse).__name__}."
        )
    if sparse.layout != torch.sparse_coo:
        raise TypeError(
            "Candidate FRA sparse result must be a torch sparse COO tensor."
        )

    sparse = sparse.coalesce()
    shape = tuple(result.get("shape", tuple(int(dim) for dim in sparse.shape)))
    if len(shape) != 4:
        raise ValueError(f"Candidate FRA shape must have rank 4, got {shape}.")

    seq_len = int(result.get("seq_len", shape[0]))
    return FRAConformanceResult(
        fra_tensor_sparse=sparse,
        shape=shape,
        seq_len=seq_len,
    )


@pytest.fixture
def fra_candidate() -> CandidateFRA:
    """Default conformance adapter for the repo's current FRA implementation."""

    def candidate(case: FRAConformanceCase) -> FRAConformanceResult:
        raw_result = get_sentence_fra_batch(
            model=case.model,
            sae=case.sae,
            text=case.text,
            layer=case.layer,
            head=case.head,
            max_length=case.max_length,
            top_k=case.top_k,
            verbose=False,
            hook_point=case.hook_point,
            chunk_size=case.chunk_size,
            normalize_by_decoder_norm=case.normalize_by_decoder_norm,
            prepend_bos=case.prepend_bos,
        )
        return _normalize_sparse_result(raw_result)

    return candidate
