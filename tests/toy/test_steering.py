"""Claim A as a regression test: where an intervention lands, not which one.

Claim A is algebraic. A score-row edit at ``q*`` cannot reach outside
``{positions >= q*}``, because it changes only *where* ``q*`` looks, not what
``q*`` is. Activation- and residual-space edits change what ``q*`` *is*, so every
later position that attends to ``q*`` reads a corrupted value.

This is structural and belongs to score-space intervention generally, not to FRA.
FRA's contribution is Claim B -- identifying *which* pair to ablate. Pinned here
so the two never get conflated, and so a refactor that breaks the mechanism fails
loudly rather than quietly producing a weaker frontier.
"""

from __future__ import annotations

import pytest
import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import ablate_pair
from fra.toy.steering import ablate_feature_qkv, steer_residual
from fra.toy.train import TrainConfig, train

SMALL = ToyConfig(
    d_model=48, d_head=48, n_content=4, n_distractor=30, n_filler=8,
    n_query_variants=16, n_key_variants=64, seq_len=12, seed=0,
)

HOOKS = [
    "blocks.0.attn.hook_q",
    "blocks.0.attn.hook_k",
    "blocks.0.attn.hook_v",
    "blocks.0.hook_resid_pre",
    "blocks.0.hook_resid_post",
]


@pytest.fixture(scope="module")
def trained():
    r = train(SMALL, TrainConfig(steps=600, batch=128, query_loss_weight=48.0, log=False))
    G = coupling_matrix(r.model, r.dgp.feature_directions)
    b = r.dgp.sample(128, split="heldout")
    return r, G, b


def _cache(model, tokens, ctx=None):
    if ctx is None:
        _, c = model.run_with_cache(tokens, names_filter=HOOKS)
        return c
    with ctx:
        _, c = model.run_with_cache(tokens, names_filter=HOOKS)
        return c


def test_qkv_and_residual_agree_on_qkv(trained) -> None:
    """The two baselines feed identical Q, K, V -- they differ only downstream."""
    r, _, b = trained
    W = r.dgp.feature_directions
    lam, _ = r.dgp.planted_qk_edge
    res = _cache(r.model, b.tokens, steer_residual(r.model, b.features, W, lam, 1.0))
    qkv = _cache(r.model, b.tokens, ablate_feature_qkv(r.model, b.features, W, lam, 1.0))
    for h in HOOKS[:3]:
        assert (res[h] - qkv[h]).abs().max().item() < 1e-5, h


def test_qkv_leaves_the_residual_stream_alone(trained) -> None:
    """The skip connection is the whole difference between the two baselines."""
    r, _, b = trained
    W = r.dgp.feature_directions
    lam, _ = r.dgp.planted_qk_edge
    base = _cache(r.model, b.tokens)
    res = _cache(r.model, b.tokens, steer_residual(r.model, b.features, W, lam, 1.0))
    qkv = _cache(r.model, b.tokens, ablate_feature_qkv(r.model, b.features, W, lam, 1.0))

    pre = "blocks.0.hook_resid_pre"
    assert (qkv[pre] - base[pre]).abs().max().item() == 0.0
    assert (res[pre] - base[pre]).abs().max().item() > 0.1


def test_qkv_equals_zeroing_the_feature_in_the_block_input(trained) -> None:
    """Hooking q/k/v is algebraically the same as editing the block's input."""
    r, _, b = trained
    W = r.dgp.feature_directions
    lam, _ = r.dgp.planted_qk_edge
    W_Q = r.model.blocks[0].attn.W_Q[0]

    got = _cache(r.model, b.tokens, ablate_feature_qkv(r.model, b.features, W, lam, 1.0))
    x = b.features @ W
    zeroed = x - b.features[:, :, lam : lam + 1] * W[lam]
    assert (got["blocks.0.attn.hook_q"][:, :, 0] - zeroed @ W_Q).abs().max().item() < 1e-5


@pytest.mark.parametrize("scale", [1.0, 3.0, 6.0])
def test_claim_a_score_edit_perturbs_nothing_outside_the_target_row(trained, scale) -> None:
    """CLAIM A. Ablating a feature pair changes no logit at any other position.

    Exactly zero, not merely small -- and at every strength, including heavy
    over-ablation.
    """
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    base = r.model(b.tokens)
    with ablate_pair(r.model, b.features, G, lam, mu, scale):
        got = r.model(b.tokens)

    rows = torch.arange(b.tokens.shape[0])
    target = torch.zeros(b.tokens.shape[:2], dtype=torch.bool)
    target[rows, b.query_pos] = True

    delta = (got - base).abs().amax(-1)
    assert delta[~target].max().item() == 0.0, "score-row edit leaked outside its row"
    assert delta[target].max().item() > 1.0, "intervention did nothing at the target"


def test_baselines_perturb_exactly_the_positions_that_can_read_q_star(trained) -> None:
    """The complement of Claim A, and the reason it holds.

    Residual steering corrupts what q* IS, so the perturbed set is precisely the
    positions that can causally attend to q* -- everything at or after it.
    """
    r, _, b = trained
    W = r.dgp.feature_directions
    lam, _ = r.dgp.planted_qk_edge
    base = r.model(b.tokens)
    with steer_residual(r.model, b.features, W, lam, 1.0):
        got = r.model(b.tokens)

    changed = (got - base).abs().amax(-1) > 1e-5
    positions = torch.arange(b.tokens.shape[1]).expand(b.tokens.shape[0], -1)
    at_or_after = positions >= b.query_pos[:, None]
    assert torch.equal(changed, at_or_after)
