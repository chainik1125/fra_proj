"""Per-cell channel-routing tests.

Validates `resolve_channel_deltas` produces the right per-cell behaviour
(Dmitry-style fudge for OV/QK rows, channel-routed for Triple) without
loading the model — we replace `compute_sae_delta` with a stub that returns
a one-hot tensor per feature index so we can read off which features ended
up in each channel."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch


def stub_compute_sae_delta(model, sae, layer_hook, feature_idx, tokens, mask):
    """Returns a (B, P, d_model) tensor that is one-hot at position feature_idx
    along the last axis. Lets us identify which features contributed to a delta."""
    B, P = mask.shape
    d_model = 100
    out = torch.zeros(B, P, d_model)
    out[..., feature_idx] = mask.float()
    return out


def cells_to_check():
    """Each row: (attr_row, intervene_col, selected, expected_per_channel)."""
    # Selected lists carry the (feature, channel) tagging convention.
    OV_SEL = [(10, "V"), (11, "V"), (12, "V")]
    QK_SEL = [(20, "Q"), (21, "Q"), (22, "Q"), (30, "K"), (31, "K"), (32, "K")]
    TRIPLE_SEL = [(40, "Q"), (50, "K"), (60, "V")]   # one triplet (40, 50, 60)

    return [
        ("OV+ov", OV_SEL, {"V"},
         {"V": {10, 11, 12}}),
        ("OV+qk", OV_SEL, {"Q", "K"},
         {"Q": {10, 11, 12}, "K": {10, 11, 12}}),     # fudge — replicate
        ("OV+all", OV_SEL, {"Q", "K", "V"},
         {"Q": {10, 11, 12}, "K": {10, 11, 12}, "V": {10, 11, 12}}),
        ("QK+ov", QK_SEL, {"V"},
         {"V": {20, 21, 22, 30, 31, 32}}),             # fudge — V uncovered
        ("QK+qk", QK_SEL, {"Q", "K"},
         {"Q": {20, 21, 22}, "K": {30, 31, 32}}),
        ("QK+all", QK_SEL, {"Q", "K", "V"},
         {"Q": {20, 21, 22}, "K": {30, 31, 32}, "V": {20, 21, 22, 30, 31, 32}}),
        ("Triple+ov", TRIPLE_SEL, {"V"}, {"V": {60}}),
        ("Triple+qk", TRIPLE_SEL, {"Q", "K"}, {"Q": {40}, "K": {50}}),
        ("Triple+all", TRIPLE_SEL, {"Q", "K", "V"},
         {"Q": {40}, "K": {50}, "V": {60}}),
    ]


def feature_indices_in_delta(delta: torch.Tensor) -> set[int]:
    """Recover which feature indices were summed (from one-hot stub)."""
    nonzero = (delta.abs().sum(dim=(0, 1)) > 1e-9).nonzero(as_tuple=True)[0]
    return set(nonzero.tolist())


def test_channel_routing_matrix(monkeypatch=None):
    from sleeper import hooks as hooks_mod
    from scripts import sleepers_pipeline as pipe

    mask = torch.ones(2, 5, dtype=torch.bool)
    tokens = torch.zeros(2, 5, dtype=torch.long)

    # Monkey-patch compute_sae_delta in the imported scripts.sleepers_pipeline
    # namespace (it calls compute_sae_delta inside resolve_channel_deltas):
    pipe.compute_sae_delta = stub_compute_sae_delta

    cells = cells_to_check()
    for name, selected, active, expected in cells:
        deltas = pipe.resolve_channel_deltas(
            selected, active, model=None, sae_ln1=None, ln1_hook="dummy",
            tokens=tokens, prompt_mask=mask,
        )
        for c in expected:
            got = feature_indices_in_delta(deltas[c])
            assert got == expected[c], \
                f"{name}: channel {c} expected {expected[c]} got {got}"
        # Channels not in active should not be present.
        assert set(deltas.keys()) == active, \
            f"{name}: returned channels {set(deltas.keys())} != active {active}"
    print("all 9 cells route correctly")


if __name__ == "__main__":
    test_channel_routing_matrix()
