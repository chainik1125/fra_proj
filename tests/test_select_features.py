"""Sanity tests for sleeper.attribution.select_features.

Two selection methods on the same OV contribution tensor; they share the
same scoring engine but differ in head reduction and dedup.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sleeper.attribution import rank_dep_vs_clean, select_features


def _random_contrib(B=8, H=4, T=10, F=32, seed=0):
    """Random (B, H, T, F) contrib tensor + matching is_deployment."""
    g = torch.Generator().manual_seed(seed)
    contrib = torch.randn(B, H, T, F, generator=g)
    is_dep = torch.zeros(B, dtype=torch.bool)
    is_dep[: B // 2] = True
    return contrib, is_dep


def test_jamie_method_ranks_by_head_summed_signed_diff():
    """Jamie's path collapses heads first, then ranks features by |head-summed
    dep_minus_clean|. Should return exactly the features whose `score.abs()` is
    largest, regardless of which heads they came from."""
    contrib, is_dep = _random_contrib(seed=11)
    ranked = rank_dep_vs_clean(contrib, is_dep, query_mask=None)
    expected = torch.argsort(ranked["score"].abs(), descending=True)[:5].tolist()

    out = select_features(contrib, is_dep, top_k=5, method="jamie")
    assert out["features"] == [int(f) for f in expected], \
        f"jamie features {out['features']} != expected {expected}"
    assert out["provenance"] is None, "jamie has no head provenance"


def test_ketan_method_dedupes_pairs_into_unique_features():
    """Ketan's path ranks (h, f) pairs by |signed diff| then walks the list
    extracting unique features. Always returns top_k unique feature indices
    (when there are enough non-degenerate pairs)."""
    contrib, is_dep = _random_contrib(seed=22)
    out = select_features(contrib, is_dep, top_k=8, method="ketan")
    # No duplicate features
    assert len(out["features"]) == len(set(out["features"])), \
        f"ketan returned duplicate features: {out['features']}"
    assert len(out["features"]) == 8
    # Provenance has matching feature ids
    assert all(f == fp for f, (h, fp) in zip(out["features"], out["provenance"])), \
        "provenance feature ids must match returned feature list"


def test_ketan_first_feature_matches_argmax_pair():
    """The top-ranked Ketan feature is the second index of the (h, f) pair
    with maximum |dep − clean| — straight from per_pair_diff.abs().argmax()."""
    contrib, is_dep = _random_contrib(seed=33)
    ranked = rank_dep_vs_clean(contrib, is_dep, query_mask=None)
    per_pair_diff = ranked["per_pair_dep"] - ranked["per_pair_cln"]
    flat_idx = int(per_pair_diff.abs().argmax().item())
    h_top, f_top = flat_idx // per_pair_diff.shape[1], flat_idx % per_pair_diff.shape[1]

    out = select_features(contrib, is_dep, top_k=1, method="ketan")
    assert out["features"][0] == f_top, \
        f"ketan top-1 feat {out['features'][0]} != argmax-pair feat {f_top}"
    assert out["provenance"][0] == (h_top, f_top), \
        f"ketan top-1 provenance {out['provenance'][0]} != ({h_top}, {f_top})"


def test_jamie_query_mask_changes_ranking():
    """Jamie respects query_mask; restricting q to a subset changes the
    score and therefore the ranking. Ketan's path is unaffected (no qmask)."""
    contrib, is_dep = _random_contrib(seed=44)
    qm = torch.zeros(contrib.shape[0], contrib.shape[2], dtype=torch.bool)
    qm[:, : contrib.shape[2] // 2] = True

    no_mask = select_features(contrib, is_dep, top_k=5, method="jamie")
    with_mask = select_features(contrib, is_dep, top_k=5, method="jamie", query_mask=qm)
    assert no_mask["features"] != with_mask["features"], \
        "jamie ranking should change when q-positions are masked"


def test_method_dispatch_errors_on_unknown():
    contrib, is_dep = _random_contrib(seed=55)
    try:
        select_features(contrib, is_dep, top_k=3, method="nonsense")
    except ValueError:
        return
    raise AssertionError("expected ValueError on unknown method")


if __name__ == "__main__":
    test_jamie_method_ranks_by_head_summed_signed_diff()
    test_ketan_method_dedupes_pairs_into_unique_features()
    test_ketan_first_feature_matches_argmax_pair()
    test_jamie_query_mask_changes_ranking()
    test_method_dispatch_errors_on_unknown()
    print("select_features tests pass")
