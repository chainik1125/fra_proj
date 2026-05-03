"""Sanity test for sleeper.attribution.compute_centered_g.

The centered profile tilde_g[b, h, q, j] = A[b,h,q,j] · (g[b,h,j] - g_bar[b,h,q])
must satisfy Σ_j tilde_g = 0 by construction (since Σ_j A[b,h,q,j] = 1).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from sleeper.attribution import compute_centered_g


def test_centered_g_sums_to_zero():
    torch.manual_seed(0)
    B, H, T = 4, 3, 8
    d_sae = 16
    # Random valid attention pattern
    raw = torch.randn(B, H, T, T)
    A = torch.softmax(raw, dim=-1)
    z_ln1 = torch.randn(B, T, d_sae)
    beta = torch.randn(H, d_sae)

    out = compute_centered_g(A, z_ln1, beta)
    tg = out["tilde_g"]
    sums = tg.sum(dim=-1).abs().max().item()
    assert sums < 1e-5, f"Σ_j tilde_g should be ~0, got max abs {sums}"

    # g[b,h,j] should equal the per-source OV write scalar Σ_λ z·β
    g_direct = torch.einsum("btf,hf->bht", z_ln1, beta)
    assert torch.allclose(out["g"], g_direct, atol=1e-5)
    print("centered_g sanity passes")


if __name__ == "__main__":
    test_centered_g_sums_to_zero()
