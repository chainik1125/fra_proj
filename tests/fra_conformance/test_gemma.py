"""Gemma exactness test for the FRA conformance harness."""

from __future__ import annotations

import os

import pytest

from tests.fra_conformance.contracts import FRAConformanceCase
from tests.fra_conformance.helpers import (
    CachedActivationModel,
    SyntheticFeatureWrapper,
    assert_conformance_result,
    build_sparse_features,
    get_model_load_config,
)

pytestmark = [
    pytest.mark.fra_conformance,
    pytest.mark.model_backed,
    pytest.mark.gemma,
]


def test_gemma_actual_decoder_exactness(fra_candidate) -> None:
    import torch
    from transformer_lens import HookedTransformer

    from fra.sae_lens_wrapper import GemmaScopeSAE

    sae_layer = 12
    layer = sae_layer + 1
    head = 1
    hook_point = "hook_resid_pre"
    text = "The cat sat on the mat."
    device = "cuda" if torch.cuda.is_available() else "cpu"

    release = os.environ.get("FRA_GEMMA_SAE_RELEASE", "gemma-scope-2b-pt-res")
    sae_id = os.environ.get("FRA_GEMMA_SAE_ID", "layer_12/width_16k/average_l0_82")
    default_model_name = os.environ.get("FRA_GEMMA_MODEL", "gemma-2-2b")

    try:
        base_sae = GemmaScopeSAE(release, sae_id, device=device)
    except Exception as exc:
        pytest.fail(f"Failed to load Gemma FRA SAE '{release}:{sae_id}': {exc}")

    model_name, model_kwargs = get_model_load_config(default_model_name, base_sae)
    try:
        base_model = HookedTransformer.from_pretrained(model_name, device=device, **model_kwargs)
    except Exception as exc:
        pytest.fail(f"Failed to load Gemma model '{model_name}': {exc}")

    tokens = base_model.tokenizer.encode(text, add_special_tokens=True)
    seq_len = len(tokens)
    feature_ids = [0, 17, 1234]
    top_k = len(feature_ids)

    features = build_sparse_features(
        seq_len,
        base_sae.d_sae,
        feature_ids,
        dtype=base_sae.W_dec.dtype,
        device=base_sae.W_dec.device,
        seed=0,
        scale=0.5,
    )
    x_full = base_sae.decode(features)
    x_nobias = x_full - base_sae.b_dec

    wrapped_model = CachedActivationModel(
        base_model,
        hook_name=f"blocks.{layer}.{hook_point}",
        activations=x_full,
    )
    wrapped_sae = SyntheticFeatureWrapper(base_sae, features)
    case = FRAConformanceCase(
        name="gemma_actual_decoder",
        model=wrapped_model,
        sae=wrapped_sae,
        text=text,
        layer=layer,
        head=head,
        hook_point=hook_point,
        top_k=top_k,
        chunk_size=1,
        max_length=128,
        prepend_bos=True,
        normalize_by_decoder_norm=False,
        expected_seq_len=seq_len,
    )

    result = fra_candidate(case)
    assert_conformance_result(
        result,
        case,
        d_sae=base_sae.d_sae,
        x_nobias=x_nobias,
        x_full=x_full,
        b_dec=base_sae.b_dec,
        raw_rtol=1e-4,
        raw_atol=1e-4,
        score_rtol=1e-4,
        score_atol=1e-4,
        pattern_rtol=1e-5,
        pattern_atol=1e-5,
    )
