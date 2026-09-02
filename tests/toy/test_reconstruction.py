"""Gate 1: the residual stream is exactly spanned by the ground-truth dictionary.

    ||resid_pre - f @ W_dec|| ~= 0

This is the assertion the whole experiment rests on, and it comes *before* the
FRA exactness test, because it is where the things that actually break exactness
show up: a non-zero ``W_pos``, an embedding that drifted off the planted
directions, an off-by-one in the feature bookkeeping. Localising those here
saves debugging an einsum that was correct all along.

Scope note, to be repeated in the write-up: given a frozen ``W_E``, ``W_pos = 0``,
no LayerNorm and no RoPE, this identity is guaranteed *a priori*. The test
validates our implementation, not the FRA method. It is a correctness gate, not
a result.
"""

from __future__ import annotations

import pytest
import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import FeatureMatchedRetrieval
from fra.toy.model import RESID_PRE_HOOK, assert_freezes_hold, build_model

CONFIGS = [
    pytest.param(ToyConfig(seed=0), id="rho0"),
    pytest.param(ToyConfig(rho=0.3, seed=1), id="rho0.3"),
    pytest.param(
        ToyConfig(overlap_mode="subspace", subspace_rank=8, seed=2), id="subspace8"
    ),
]


@pytest.mark.parametrize("cfg", CONFIGS)
def test_dgp_level_reconstruction(cfg: ToyConfig) -> None:
    """f @ W_dec reproduces the planted embedding, without any transformer."""
    dgp = FeatureMatchedRetrieval(cfg)
    b = dgp.sample(batch=8)

    expected = b.features @ dgp.feature_directions  # [batch, seq, d_model]
    actual = dgp.embedding_matrix[b.tokens]

    err = (expected - actual).abs().max().item()
    assert err < 1e-5, f"DGP-level reconstruction error {err:.3e}"


@pytest.mark.parametrize("cfg", CONFIGS)
def test_resid_pre_reconstruction(cfg: ToyConfig) -> None:
    """THE gate: the real model's resid_pre lies exactly in the feature basis.

    Run against an untrained HookedTransformer -- this must hold before any
    training happens, and it is what catches W_pos and TransformerLens's own
    embedding path rather than only our bookkeeping.
    """
    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)
    assert_freezes_hold(model, dgp)

    b = dgp.sample(batch=8)
    _, cache = model.run_with_cache(b.tokens, names_filter=[RESID_PRE_HOOK])
    resid_pre = cache[RESID_PRE_HOOK]  # [batch, seq, d_model]

    oracle = b.features @ dgp.feature_directions

    err = (resid_pre - oracle).abs().max().item()
    scale = resid_pre.abs().max().item()
    assert err < 1e-5, (
        f"resid_pre is not spanned by the feature basis: "
        f"max abs err {err:.3e} (resid scale {scale:.3f})"
    )


def test_positional_embedding_is_inert() -> None:
    """The same token at different positions must embed identically.

    A direct check that no positional residue leaks in -- if W_pos were live
    this fails, and so would oracle exactness.
    """
    cfg = ToyConfig()
    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)

    tokens = torch.full((1, cfg.seq_len), 3, dtype=torch.long)
    _, cache = model.run_with_cache(tokens, names_filter=[RESID_PRE_HOOK])
    resid = cache[RESID_PRE_HOOK][0]  # [seq, d_model]

    spread = (resid - resid[0]).abs().max().item()
    assert spread < 1e-6, f"resid_pre varies with position by {spread:.3e}"


def test_freezes_survive_an_optimizer_step() -> None:
    """A careless optimizer over model.parameters() must not move the frozen weights."""
    cfg = ToyConfig()
    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)

    opt = torch.optim.SGD(model.parameters(), lr=1.0)
    b = dgp.sample(batch=4)
    logits = model(b.tokens)
    loss = torch.nn.functional.cross_entropy(
        logits.reshape(-1, cfg.d_vocab), b.targets.reshape(-1)
    )
    loss.backward()
    opt.step()

    assert_freezes_hold(model, dgp)
