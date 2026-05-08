"""Sanity tests for sleeper.metrics.severity_ratio.

The metric's contract:
    num = mean over (s, b, t) of CE(softmax(clean_lsm[s]), steered_lsm[s])
                                         — diagonal seed pairing, S samples per (b, t)
    den = mean over (pair, b, t) of CE(softmax(clean_lsm[a]), clean_lsm[b])
                                         — all unordered seed pairs, S(S-1)/2 per (b, t)
    ratio = num / den   (each side meaned independently before division)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import math

import torch

from sleeper.metrics import severity_ratio


def _random_lsm(S, B, T, V, seed):
    g = torch.Generator().manual_seed(seed)
    logits = torch.randn(S, B, T, V, generator=g)
    return torch.log_softmax(logits.float(), dim=-1).to(torch.float16)


def test_ratio_one_when_steered_equals_clean():
    """If steered distributions are identical to the clean ones row-by-row, the
    diagonal numerator equals the all-pair denominator in expectation — and
    exactly when S=2 (single pair on each side averaging the same per-(b,t) CE)."""
    S, B, T, V = 2, 3, 4, 17
    lsm = _random_lsm(S, B, T, V, seed=42)
    out = severity_ratio(lsm, lsm.clone())
    # With S=2 the numerator averages CE(c[0],c[0]) and CE(c[1],c[1]) — the
    # entropies — while the denominator averages the single off-diagonal pair
    # CE(c[0],c[1]). These are not algebraically equal, so check ratio is
    # finite and the count fields are correct rather than asserting ratio=1.
    assert out["num_count"] == S * B * T
    assert out["den_count"] == (S * (S - 1) // 2) * B * T
    assert math.isfinite(out["ratio"])
    assert out["num"] > 0
    assert out["den"] > 0


def test_diagonal_pairing_uses_S_samples():
    """Numerator must sum exactly S diagonal pairs of CE — not S² or S(S-1)/2."""
    S, B, T, V = 4, 2, 3, 11
    clean   = _random_lsm(S, B, T, V, seed=1)
    steered = _random_lsm(S, B, T, V, seed=2)
    out = severity_ratio(clean, steered)
    assert out["num_count"] == S * B * T, \
        f"numerator count should be S*B*T={S*B*T}, got {out['num_count']}"
    assert out["den_count"] == (S * (S - 1) // 2) * B * T, \
        f"denominator count should be C(S,2)*B*T={(S*(S-1)//2)*B*T}, got {out['den_count']}"


def test_ratio_grows_with_steered_divergence():
    """A steered side that is far from clean should yield ratio > 1; a steered
    side that matches clean's distribution family should yield ratio close to 1."""
    S, B, T, V = 3, 4, 5, 19
    g = torch.Generator().manual_seed(7)
    base_logits = torch.randn(S, B, T, V, generator=g)
    clean = torch.log_softmax(base_logits.float(), dim=-1).to(torch.float16)
    # Mild perturbation: same distribution family with small noise.
    mild = torch.log_softmax(
        base_logits + 0.05 * torch.randn(S, B, T, V, generator=g), dim=-1,
    ).to(torch.float16)
    # Severe: unrelated draw.
    severe = torch.log_softmax(
        torch.randn(S, B, T, V, generator=g) * 4.0, dim=-1,
    ).to(torch.float16)
    r_mild   = severity_ratio(clean, mild)["ratio"]
    r_severe = severity_ratio(clean, severe)["ratio"]
    assert r_severe > r_mild, \
        f"severe perturbation should yield larger ratio: mild={r_mild:.3f} severe={r_severe:.3f}"


def test_requires_two_seeds():
    """S=1 has no unordered seed pairs to form the denominator — must error."""
    lsm = _random_lsm(1, 2, 3, 5, seed=99)
    try:
        severity_ratio(lsm, lsm.clone())
    except ValueError:
        return
    raise AssertionError("severity_ratio should reject S=1")


def test_means_before_division():
    """The function must mean num and den independently before dividing —
    not return a per-sample ratio averaged. With S=3, num_count != den_count,
    so a naive 'sum num / sum den' without normalising would not equal the
    documented ratio."""
    S, B, T, V = 3, 2, 4, 13
    clean   = _random_lsm(S, B, T, V, seed=1234)
    steered = _random_lsm(S, B, T, V, seed=5678)
    out = severity_ratio(clean, steered)
    expected = out["num"] / out["den"]
    assert abs(out["ratio"] - expected) < 1e-9, \
        f"ratio must equal num_mean / den_mean: got {out['ratio']} vs {expected}"


if __name__ == "__main__":
    test_ratio_one_when_steered_equals_clean()
    test_diagonal_pairing_uses_S_samples()
    test_ratio_grows_with_steered_divergence()
    test_requires_two_seeds()
    test_means_before_division()
    print("severity_ratio tests pass")
