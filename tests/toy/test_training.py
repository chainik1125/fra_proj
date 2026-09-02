"""Gate 2 as a regression test, on a deliberately small configuration.

The full-size run is ~2 minutes; this is a shrunk version that exercises the
same path in seconds. It asserts the two properties FRA depends on:

* the model solves the planted rule on **held-out** token variants, so the edge
  FRA will recover is a feature edge and not a token edge in costume;
* attention at ``lambda*`` positions actually lands on the ``mu*`` key, so there
  is a real edge for FRA to find. Without this, FRA correctly reporting "no
  strong edge" would be indistinguishable from FRA failing.
"""

from __future__ import annotations

import pytest

from fra.toy.config import ToyConfig
from fra.toy.train import TrainConfig, evaluate, train

SMALL = ToyConfig(
    d_model=48,
    d_head=48,
    n_content=4,
    n_distractor=30,
    n_filler=8,
    n_query_variants=16,
    n_key_variants=64,
    seq_len=12,
    seed=0,
)


@pytest.fixture(scope="module")
def trained():
    # query_loss_weight is 4 * seq_len, matching the full-size config (128 = 4 * 32).
    # At 1 * seq_len this configuration reaches 100% accuracy with argmax_is_key
    # stuck at 65% -- accuracy saturates well before attention sharpens, which is
    # the whole reason Gate 2 is a separate assertion.
    return train(SMALL, TrainConfig(steps=2000, batch=128, query_loss_weight=48.0, log=False))


def test_learns_the_planted_rule(trained) -> None:
    ev = evaluate(trained, batch=512)
    assert ev["train"]["acc"].at_query > 0.95, ev["train"]["acc"]


def test_generalises_to_heldout_variants(trained) -> None:
    """Feature-level solving, not token memorisation."""
    ev = evaluate(trained, batch=512)
    chance = 1.0 / SMALL.n_content
    held = ev["heldout"]["acc"].at_query
    assert held > 0.90, f"held-out query accuracy {held:.3f} (chance {chance:.3f})"


def test_gate2_attention_lands_on_the_key(trained) -> None:
    """The rule is implemented THROUGH attention, not routed around the head."""
    ev = evaluate(trained, batch=512)
    for split in ("train", "heldout"):
        c = ev[split]["attn"]
        assert c.argmax_is_key > 0.95, f"{split}: argmax_is_key {c.argmax_is_key:.3f}"
        assert c.mean_mass_on_key > 0.80, f"{split}: mass_on_key {c.mean_mass_on_key:.3f}"


def test_freezes_survive_training(trained) -> None:
    from fra.toy.model import assert_freezes_hold

    assert_freezes_hold(trained.model, trained.dgp)


def test_heldout_splits_use_unseen_tokens() -> None:
    """The control is only meaningful if the held-out tokens are genuinely unseen."""
    from fra.toy.dgp import FeatureMatchedRetrieval

    dgp = FeatureMatchedRetrieval(SMALL)
    tr = dgp.sample(512, split="train")
    ho = dgp.sample(512, split="heldout")

    rows = range(512)
    train_q = {int(tr.tokens[i, tr.query_pos[i]]) for i in rows}
    held_q = {int(ho.tokens[i, ho.query_pos[i]]) for i in rows}
    train_k = {int(tr.tokens[i, tr.key_pos[i]]) for i in rows}
    held_k = {int(ho.tokens[i, ho.key_pos[i]]) for i in rows}

    assert not (train_q & held_q), "held-out query tokens leaked into training"
    assert not (train_k & held_k), "held-out key tokens leaked into training"
