"""Feature-pair ablation: mechanics, and the specificity of the causal effect."""

from __future__ import annotations

import pytest
import torch

from fra.toy.config import ToyConfig
from fra.toy.fra import coupling_matrix
from fra.toy.intervention import (
    SCORES_HOOK,
    ablate_pair,
    evaluate_ablation,
    matched_random_pair,
    pair_delta,
    runner_up_pair,
)
from fra.toy.metrics import accuracy, attention_concentration
from fra.toy.recovery import qk_aggregate_closed_form
from fra.toy.train import TrainConfig, train

SMALL = ToyConfig(
    d_model=48, d_head=48, n_content=4, n_distractor=30, n_filler=8,
    n_query_variants=16, n_key_variants=64, seq_len=12, seed=0,
)


@pytest.fixture(scope="module")
def trained():
    r = train(SMALL, TrainConfig(steps=800, batch=128, query_loss_weight=48.0, log=False))
    G = coupling_matrix(r.model, r.dgp.feature_directions)
    b = r.dgp.sample(256, split="heldout")
    return r, G, b


def test_hook_removes_exactly_the_pair_contribution(trained) -> None:
    """The intervention is the FRA term itself, not an approximation of it."""
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge

    _, c0 = r.model.run_with_cache(b.tokens, names_filter=[SCORES_HOOK])
    with ablate_pair(r.model, b.features, G, lam, mu):
        _, c1 = r.model.run_with_cache(b.tokens, names_filter=[SCORES_HOOK])

    s0 = c0[SCORES_HOOK][:, 0]
    s1 = c1[SCORES_HOOK][:, 0]
    finite = torch.isfinite(s0)
    delta = pair_delta(b.features, G, lam, mu)
    assert ((s0 - s1)[finite] - delta[finite]).abs().max().item() < 1e-5


def test_causal_mask_survives_ablation(trained) -> None:
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    with ablate_pair(r.model, b.features, G, lam, mu):
        _, c = r.model.run_with_cache(b.tokens, names_filter=[SCORES_HOOK])
    s = c[SCORES_HOOK][0, 0]
    T = s.shape[0]
    upper = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
    assert torch.all(torch.isneginf(s[upper]))


def test_hook_is_scoped(trained) -> None:
    """Behaviour returns to baseline once the context exits."""
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    before = accuracy(r.model, b).at_query
    with ablate_pair(r.model, b.features, G, lam, mu):
        pass
    assert accuracy(r.model, b).at_query == before


def test_ablating_the_planted_pair_destroys_the_behaviour(trained) -> None:
    """Metric 5: the recovered edge is causally load-bearing."""
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    base = accuracy(r.model, b).at_query
    out = evaluate_ablation(r.model, b, G, (lam, mu), "planted")
    assert base > 0.85
    assert out.acc.at_query < base - 0.4, f"{base:.3f} -> {out.acc.at_query:.3f}"
    assert out.argmax_is_key < 0.9


def test_matched_random_pair_removes_equal_mass(trained) -> None:
    """The control has to perturb as much, or 'no effect' is trivially true."""
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    pair, scale = matched_random_pair(G, b, (lam, mu))
    planted = evaluate_ablation(r.model, b, G, (lam, mu), "planted")
    random = evaluate_ablation(r.model, b, G, pair, "random", scale)
    rel = abs(random.removed_l1 - planted.removed_l1) / planted.removed_l1
    assert rel < 1e-3, f"removed mass differs by {rel:.2%}"


def test_ablation_is_specific(trained) -> None:
    """Metric 6: an equal-magnitude random pair does NOT destroy the behaviour.

    Without this, 'ablating the planted pair breaks it' would be consistent with
    the model simply being fragile to any perturbation of that size.
    """
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    base = accuracy(r.model, b).at_query
    pair, scale = matched_random_pair(G, b, (lam, mu))

    planted = evaluate_ablation(r.model, b, G, (lam, mu), "planted")
    random = evaluate_ablation(r.model, b, G, pair, "random", scale)

    assert random.acc.at_query > planted.acc.at_query + 0.3
    assert random.acc.at_query > base - 0.25


def test_runner_up_pair_is_not_the_planted_one(trained) -> None:
    r, G, b = trained
    lam, mu = r.dgp.planted_qk_edge
    agg = sum(
        qk_aggregate_closed_form(r.dgp.sample(1, split="heldout").features[0], G)
        for _ in range(4)
    )
    runner = runner_up_pair(agg, (lam, mu))
    assert runner != (lam, mu)
    assert agg[runner].abs() <= agg[lam, mu].abs()
