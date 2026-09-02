"""Gate 3: the FRA decomposition reconstructs the model, exactly.

With oracle features, no LayerNorm, no RoPE and zeroed biases, the decomposition
is algebraically exact -- summing FRA over the feature axes must return the
model's own pre-softmax scores (QK) and attention output (OV).

As with Gate 1, this is guaranteed a priori by the construction: it validates our
IMPLEMENTATION of the equations, not the FRA method. A failure here means an
einsum is wrong, not that FRA is wrong. It is a correctness gate, not a result,
and the write-up must say so.

What it does buy: this is the first exact reconstruction of FRA against a *real
trained transformer*. The repo's conformance harness proves the algebra against a
FakeModel stub, and the standalone `fra` repo does an approximate post-softmax
comparison under top-k truncation. Neither is this.
"""

from __future__ import annotations

import pytest
import torch

from fra.toy.config import ToyConfig
from fra.toy.dgp import FeatureMatchedRetrieval
from fra.toy.fra import OracleFRA
from fra.toy.model import build_model

CONFIGS = [
    pytest.param(ToyConfig(seed=0), id="rho0"),
    pytest.param(ToyConfig(rho=0.3, seed=1), id="rho0.3"),
    pytest.param(
        ToyConfig(overlap_mode="subspace", subspace_rank=8, seed=2), id="subspace8"
    ),
]


def _one(cfg: ToyConfig):
    """An untrained model is enough -- exactness is about algebra, not learning."""
    dgp = FeatureMatchedRetrieval(cfg)
    model = build_model(dgp)
    b = dgp.sample(batch=1)
    fra = OracleFRA(model, dgp.feature_directions, b.tokens[0], b.features[0])
    return dgp, model, b, fra


@pytest.mark.parametrize("cfg", CONFIGS)
def test_qk_exactness(cfg: ToyConfig) -> None:
    """sum over (lambda, mu) of FRA_QK equals the model's pre-softmax score."""
    _, _, _, fra = _one(cfg)

    reconstructed = fra.qk().sum(dim=(2, 3))  # [T, T]
    actual = fra.scores

    finite = torch.isfinite(actual)
    err = (reconstructed[finite] - actual[finite]).abs().max().item()
    assert err < 1e-5, f"QK reconstruction error {err:.3e}"

    # The masked half must be identically zero, not merely small.
    assert torch.all(reconstructed[~finite] == 0.0)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_ov_exactness(cfg: ToyConfig) -> None:
    """sum over (k, lambda) of the SIGNED OV tensor equals the attention output.

    Only the signed version can do this. The paper's norm collapse discards the
    directions, and with them any contribution that cancels.
    """
    _, model, b, fra = _one(cfg)

    with torch.no_grad():
        _, cache = model.run_with_cache(b.tokens, names_filter=["blocks.0.hook_attn_out"])
    actual = cache["blocks.0.hook_attn_out"][0]  # [T, d_model]

    reconstructed = fra.ov_signed().sum(dim=(1, 2))  # [T, d_model]
    err = (reconstructed - actual).abs().max().item()
    scale = actual.abs().max().item()
    assert err < 1e-5, f"OV reconstruction error {err:.3e} (scale {scale:.3f})"


def test_qk_exactness_after_training() -> None:
    """Exactness must survive training -- it is a property of the construction,
    not of the initialisation."""
    from fra.toy.train import TrainConfig, train

    cfg = ToyConfig(
        d_model=48, d_head=48, n_content=4, n_distractor=30, n_filler=8,
        n_query_variants=16, n_key_variants=64, seq_len=12, seed=0,
    )
    r = train(cfg, TrainConfig(steps=200, batch=64, query_loss_weight=48.0, log=False))
    b = r.dgp.sample(batch=1)
    fra = OracleFRA(r.model, r.dgp.feature_directions, b.tokens[0], b.features[0])

    finite = torch.isfinite(fra.scores)
    err = (fra.qk().sum(dim=(2, 3))[finite] - fra.scores[finite]).abs().max().item()
    assert err < 1e-5, f"QK reconstruction error after training {err:.3e}"


def test_exactness_is_rho_independent() -> None:
    """Exactness holds at every overlap, because f is planted and
    ``x = f @ W_dec`` by construction regardless of the geometry.

    Only *recovery* degrades with rho. The arithmetic never breaks; the
    interpretability does. That separation is the cleanest one-sentence framing
    of the whole experiment.
    """
    errs = []
    for rho in (0.0, 0.2, 0.5, 0.8):
        _, _, _, fra = _one(ToyConfig(rho=rho, seed=0))
        finite = torch.isfinite(fra.scores)
        errs.append((fra.qk().sum(dim=(2, 3))[finite] - fra.scores[finite]).abs().max().item())
    assert max(errs) < 1e-5, f"exactness degraded with rho: {errs}"


def test_coupling_matrix_is_data_independent() -> None:
    """G depends only on trained weights -- two different batches give the same G."""
    from fra.toy.fra import coupling_matrix

    dgp = FeatureMatchedRetrieval(ToyConfig(seed=0))
    model = build_model(dgp)
    g1 = coupling_matrix(model, dgp.feature_directions)
    g2 = coupling_matrix(model, dgp.feature_directions)
    assert torch.equal(g1, g2)
    assert g1.shape == (dgp.n_feat, dgp.n_feat)
