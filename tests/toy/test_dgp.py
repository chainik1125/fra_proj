"""DGP well-formedness: the planted rule is present, unique, and solvable.

If the task is malformed, training failure looks identical to a broken optimizer
and FRA failure looks identical to a broken method. These tests rule that out in
milliseconds, before either can waste an afternoon.
"""

from __future__ import annotations

import pytest
import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import (
    IDX_CONTENT_START,
    IDX_KEY,
    IDX_QUERY,
    TOK_DEFAULT,
    FeatureMatchedRetrieval,
    make_directions,
    realized_rho,
)


@pytest.mark.parametrize("rho", [0.0, 0.1, 0.3, 0.6])
def test_rho_knob_hits_its_target(rho: float) -> None:
    """overlap_mode='common' realizes the requested rho analytically, not approximately."""
    cfg = ToyConfig(rho=rho, seed=0)
    gen = torch.Generator().manual_seed(cfg.seed)
    directions = make_directions(cfg, gen)

    assert directions.shape == (cfg.n_feat, cfg.d_model)
    norms = directions.norm(dim=-1)
    assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)

    got = realized_rho(directions)
    assert abs(got - rho) < 1e-4, f"requested rho={rho}, realized {got:.6f}"


def test_rho_zero_is_orthonormal() -> None:
    cfg = ToyConfig(rho=0.0, seed=0)
    gen = torch.Generator().manual_seed(0)
    d = make_directions(cfg, gen)
    gram = d @ d.T
    off = gram - torch.eye(cfg.n_feat)
    assert off.abs().max().item() < 1e-5


def test_subspace_mode_increases_overlap_as_rank_falls() -> None:
    """The second geometry family behaves monotonically, so it is usable as a check."""
    rhos = []
    for rank in (64, 32, 16, 8):
        cfg = ToyConfig(overlap_mode="subspace", subspace_rank=rank, seed=0)
        gen = torch.Generator().manual_seed(0)
        rhos.append(realized_rho(make_directions(cfg, gen)))
    assert rhos == sorted(rhos), f"overlap not monotone in rank: {rhos}"


def test_planted_structure_is_unique_and_ordered() -> None:
    """Exactly one mu* before exactly one lambda*, every sequence."""
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=256)

    query_hits = (b.features[:, :, IDX_QUERY] > 0).sum(dim=1)
    key_hits = (b.features[:, :, IDX_KEY] > 0).sum(dim=1)
    assert torch.all(query_hits == 1), "lambda* must fire exactly once"
    assert torch.all(key_hits == 1), "mu* must fire exactly once"

    rows = torch.arange(b.tokens.shape[0])
    assert torch.all(b.features[rows, b.query_pos, IDX_QUERY] > 0)
    assert torch.all(b.features[rows, b.key_pos, IDX_KEY] > 0)
    assert torch.all(b.key_pos < b.query_pos), "key must precede query (causal)"


def test_content_is_carried_at_the_key_position() -> None:
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=256)
    rows = torch.arange(b.tokens.shape[0])
    n_c = dgp.cfg.n_content
    block = b.features[rows, b.key_pos, IDX_CONTENT_START : IDX_CONTENT_START + n_c]
    assert torch.all(block.argmax(dim=-1) == b.content)
    assert torch.all((block > 0).sum(dim=-1) == 1), "exactly one content feature at the key"


def test_task_is_solvable_in_principle() -> None:
    """An oracle reader of the ground truth scores 100%. If not, the task is broken."""
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=256)
    assert torch.equal(dgp.oracle_predict(b), b.targets)


def test_content_features_appear_off_the_key_position() -> None:
    """Answer tokens seed decoy content, so 'find nu_c' is not a valid shortcut."""
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=256)
    n_c = dgp.cfg.n_content
    content_block = b.features[:, :, IDX_CONTENT_START : IDX_CONTENT_START + n_c]
    active = (content_block > 0).any(dim=-1)  # [batch, seq]
    rows = torch.arange(b.tokens.shape[0])
    active[rows, b.key_pos] = False
    assert active.any(), "no decoy content anywhere -- retrieval would be trivial"


def test_answer_is_not_predictable_without_retrieval() -> None:
    """Targets at the query position are spread over the content values, so a
    constant guess cannot beat chance. Guards against the degenerate shortcut a
    single OV feature would have allowed."""
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=2048)
    counts = torch.bincount(b.content, minlength=dgp.cfg.n_content).float()
    frac = counts / counts.sum()
    assert frac.max().item() < 0.2, f"content values too skewed: {frac.tolist()}"


def test_targets_default_away_from_the_query_position() -> None:
    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    b = dgp.sample(batch=64)
    mask = torch.ones_like(b.targets, dtype=torch.bool)
    rows = torch.arange(b.tokens.shape[0])
    mask[rows, b.query_pos] = False
    assert torch.all(b.targets[mask] == TOK_DEFAULT)
    assert torch.all(b.targets[rows, b.query_pos] != TOK_DEFAULT)


def test_determinism() -> None:
    a = FeatureMatchedRetrieval(ToyConfig(seed=7)).sample(batch=16)
    c = FeatureMatchedRetrieval(ToyConfig(seed=7)).sample(batch=16)
    assert torch.equal(a.tokens, c.tokens)
    assert torch.equal(a.targets, c.targets)
