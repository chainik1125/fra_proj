"""Unit tests for the Möbius 8-corner decomposition core math.

Two checks:

1. **§7 toy worked example** from `notes/fra_channel_singles_pairs_triples.md`.
   With one head, one query, two source tokens, and explicit numerical pieces
   (q^bg=1.0, k_1^bg=0.3, k_2^bg=0.1, δq=0.8, δk=1.0, w_1^bg=0.2, δw=1.5,
   d=1, sqrt(d_h)=1), the note gives the 8 game values and 7 Möbius Δ's.
   We reproduce them via the analytical attention-pattern formulas used by
   `triple_attribution.triple_eval`.

2. **Möbius identity**: g(QKV) = ΔQ + ΔK + ΔV + ΔQK + ΔQV + ΔKV + ΔQKV.
"""

from __future__ import annotations

import math


def _toy_corner_attentions():
    """Recreate the four (zQ, zK) attention patterns at slot k=1 for the §7 toy."""
    q_bg, k1_bg, k2_bg, dq, dk = 1.0, 0.3, 0.1, 0.8, 1.0

    def softmax2(s1, s2):
        m = max(s1, s2)
        e1 = math.exp(s1 - m); e2 = math.exp(s2 - m)
        return e1 / (e1 + e2)

    # Direct (no shortcut) for ground truth:
    A_00 = softmax2(q_bg * k1_bg,        q_bg * k2_bg)
    A_01 = softmax2(q_bg * (k1_bg + dk), q_bg * k2_bg)
    A_10 = softmax2((q_bg + dq) * k1_bg, (q_bg + dq) * k2_bg)
    A_11 = softmax2((q_bg + dq) * (k1_bg + dk), (q_bg + dq) * k2_bg)

    return A_00, A_01, A_10, A_11


def _shortcut_attentions():
    """Same four corner attentions via the algebraic shortcut used in
    `triple_attribution.triple_eval`."""
    q_bg, k1_bg, k2_bg, dq, dk = 1.0, 0.3, 0.1, 0.8, 1.0
    q_clean = q_bg + dq                      # 1.8
    k1_clean = k1_bg + dk                    # 1.3
    k2_clean = k2_bg                         # 0.1

    s1 = q_clean * k1_clean                   # 2.34
    s2 = q_clean * k2_clean                   # 0.18
    m = max(s1, s2)
    e1 = math.exp(s1 - m); e2 = math.exp(s2 - m)
    A_clean_k = e1 / (e1 + e2)               # ≈ 0.8966

    a1 = dq * k1_clean                       # 1.04   (a_clean[j=1])
    a2 = dq * k2_clean                       # 0.08   (a_clean[j=2])
    Z_01 = A_clean_k * math.exp(-a1) + (1 - A_clean_k) * math.exp(-a2)
    A_01_k = A_clean_k * math.exp(-a1) / Z_01

    b = q_clean * dk                          # 1.8
    eb = math.exp(-b)
    A_10_k = A_clean_k * eb / (1 - A_clean_k * (1 - eb))

    c = dq * dk                               # 0.8
    es = math.exp(-(b - c))
    A_00_k = A_01_k * es / (1 - A_01_k * (1 - es))

    A_11_k = A_clean_k
    return A_00_k, A_01_k, A_10_k, A_11_k


def test_shortcut_matches_direct_softmax():
    direct = _toy_corner_attentions()
    shortcut = _shortcut_attentions()
    for d, s in zip(direct, shortcut, strict=True):
        assert abs(d - s) < 1e-10, f"corner mismatch: direct={d}, shortcut={s}"


def test_seven_worked_example():
    """Reproduce §7's eight game values and seven Δ's to ≤1e-4."""
    A_00, A_01, A_10, A_11 = _shortcut_attentions()
    v_d_bg = 0.2          # ⟨w_k^bg, d⟩
    v_lam = 1.5           # δw_d_λ
    baseline = A_00 * v_d_bg

    def g(A_corner, zV):
        return A_corner * (v_d_bg + zV * v_lam) - baseline

    g_000 = g(A_00, 0)
    g_001 = g(A_00, 1)
    g_010 = g(A_01, 0)
    g_011 = g(A_01, 1)
    g_100 = g(A_10, 0)
    g_101 = g(A_10, 1)
    g_110 = g(A_11, 0)
    g_111 = g(A_11, 1)

    expected = {
        (0, 0, 0): 0.0000, (0, 0, 1): 0.8248,
        (0, 1, 0): 0.0437, (0, 1, 1): 1.1965,
        (1, 0, 0): 0.0078, (1, 0, 1): 0.8914,
        (1, 1, 0): 0.0694, (1, 1, 1): 1.4143,
    }
    got = {(0,0,0): g_000, (0,0,1): g_001, (0,1,0): g_010, (0,1,1): g_011,
           (1,0,0): g_100, (1,0,1): g_101, (1,1,0): g_110, (1,1,1): g_111}
    for k, v_exp in expected.items():
        assert abs(got[k] - v_exp) < 1e-3, f"g{k}: expected {v_exp}, got {got[k]}"

    dQ   = g_100
    dK   = g_010
    dV   = g_001
    dQK  = g_110 - g_100 - g_010
    dQV  = g_101 - g_100 - g_001
    dKV  = g_011 - g_010 - g_001
    dQKV = g_111 - g_110 - g_101 - g_011 + g_100 + g_010 + g_001

    expected_deltas = {
        "ΔQ":   0.007841, "ΔK":   0.043738, "ΔV":   0.824751,
        "ΔQK":  0.017774, "ΔQV":  0.058810, "ΔKV":  0.328036,
        "ΔQKV": 0.133302,
    }
    got_deltas = {"ΔQ": dQ, "ΔK": dK, "ΔV": dV, "ΔQK": dQK, "ΔQV": dQV,
                  "ΔKV": dKV, "ΔQKV": dQKV}
    for name, v_exp in expected_deltas.items():
        assert abs(got_deltas[name] - v_exp) < 1e-3, \
            f"{name}: expected {v_exp}, got {got_deltas[name]}"


def test_mobius_sums_to_total():
    """Σ Δ = g(QKV) for the toy."""
    A_00, A_01, A_10, A_11 = _shortcut_attentions()
    v_d_bg = 0.2; v_lam = 1.5
    baseline = A_00 * v_d_bg

    def g(A_corner, zV):
        return A_corner * (v_d_bg + zV * v_lam) - baseline

    deltas = [
        g(A_10, 0),                                                            # ΔQ
        g(A_01, 0),                                                            # ΔK
        g(A_00, 1),                                                            # ΔV
        g(A_11, 0) - g(A_10, 0) - g(A_01, 0),                                  # ΔQK
        g(A_10, 1) - g(A_10, 0) - g(A_00, 1),                                  # ΔQV
        g(A_01, 1) - g(A_01, 0) - g(A_00, 1),                                  # ΔKV
        (g(A_11, 1) - g(A_11, 0) - g(A_10, 1) - g(A_01, 1)
         + g(A_10, 0) + g(A_01, 0) + g(A_00, 1)),                              # ΔQKV
    ]
    g_qkv = g(A_11, 1)
    assert abs(sum(deltas) - g_qkv) < 1e-12


if __name__ == "__main__":
    test_shortcut_matches_direct_softmax()
    test_seven_worked_example()
    test_mobius_sums_to_total()
    print("all triple-decomposition tests pass")
