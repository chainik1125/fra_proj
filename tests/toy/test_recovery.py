"""Recovery-metric correctness, plus two structural facts worth pinning down."""

from __future__ import annotations

import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import FeatureMatchedRetrieval
from fra.toy.fra import OracleFRA
from fra.toy.model import build_model
from fra.toy.recovery import (
    qk_aggregate,
    qk_aggregate_closed_form,
    qk_aggregate_signed,
    recovery_of,
)

SMALL = ToyConfig(
    d_model=48, d_head=48, n_content=4, n_distractor=30, n_filler=8,
    n_query_variants=16, n_key_variants=64, seq_len=12, seed=0,
)


def _fra():
    dgp = FeatureMatchedRetrieval(SMALL)
    model = build_model(dgp)
    b = dgp.sample(1)
    return dgp, OracleFRA(model, dgp.feature_directions, b.tokens[0], b.features[0])


def test_recovery_of_finds_the_maximum() -> None:
    x = torch.tensor([[1.0, -5.0], [2.0, 3.0]])
    r = recovery_of(x, (0, 1))  # -5.0, largest by |.| but smallest signed
    assert r.rank_abs == 1
    assert r.rank_signed == 4
    assert r.argmax_is_planted
    assert abs(r.mass_fraction - 5.0 / 11.0) < 1e-6
    assert r.total_l1 == 11.0
    assert r.total_signed == 1.0


def test_recovery_ranks_are_pessimistic_under_ties() -> None:
    x = torch.tensor([3.0, 3.0, 1.0])
    assert recovery_of(x, 0).rank_abs == 1  # one strictly-greater entry: none
    assert recovery_of(x, 2).rank_abs == 3


def test_closed_form_matches_dense_aggregate() -> None:
    """The shortcut used to make the rho sweep affordable must match the tensor."""
    dgp, fra = _fra()
    dense = qk_aggregate(fra.qk())
    closed = qk_aggregate_closed_form(fra.f, fra.G)
    err = (dense - closed).abs().max().item()
    assert err < 1e-4, f"closed form diverges from dense aggregate: {err:.3e}"


def test_signed_and_l1_aggregation_are_degenerate() -> None:
    """Non-negative activations make signed aggregation identical to L1, per pair.

    ``FRA_QK[q,k,l,m] = f[q,l] f[k,m] G[l,m]`` and ``f >= 0``, so every entry for
    a fixed ``(l, m)`` carries the sign of ``G[l, m]``. Summing over positions
    therefore gives exactly ``+/-`` the L1 sum, and the two rankings agree by
    construction rather than by evidence.

    This is why the aggregate scope CANNOT adjudicate Dmitry's note-03 finding
    that signed sums are anti-predictive: that effect needs contributions to the
    same pair with opposing signs, which requires multiple heads or routes.
    Pinned as a test so nobody later reads the agreement as a result.
    """
    _, fra = _fra()
    qk = fra.qk()
    l1 = qk_aggregate(qk)
    signed = qk_aggregate_signed(qk)
    assert (signed.abs() - l1).abs().max().item() < 1e-5


def test_cell_scope_has_few_nonzero_candidates() -> None:
    """A cell ranks ~|active(q)| x |active(k)| pairs, not n_feat**2.

    Pinned because reporting "rank 1" from the cell scope without this number
    alongside would overstate the result by three orders of magnitude.
    """
    dgp, fra = _fra()
    b = dgp.sample(1)
    fra2 = OracleFRA(
        fra.model, dgp.feature_directions, b.tokens[0], b.features[0]
    )
    q, k = int(b.query_pos[0]), int(b.key_pos[0])
    cell = fra2.qk()[q, k]
    n_nonzero = int((cell != 0).sum())
    assert n_nonzero < 40, n_nonzero
    assert cell.numel() == SMALL.n_feat**2
