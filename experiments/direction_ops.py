"""Direction extraction + steer/ablate math for the M-vs-L_F interp harness.

Pure torch, no Modal — so the residual-stream interventions can be unit-tested locally
before running the 14B/7B organism on GPU. Method follows Convergent Linear
Representations of EM (2506.11618):

  - mean-diff direction: v = mean(act | misaligned) - mean(act | aligned), per layer,
    averaged over answer tokens.
  - steer:   x := x + lambda * v_hat
  - ablate:  x := x - (v_hat . x) v_hat   (project the direction out)

In the latent-cause mapping we extract TWO directions from the SAME activations:
  v_M  (global persona) from OFF-domain (sports, broad) aligned-vs-misaligned pairs,
  v_LF (local patch)    from IN-domain  (financial) pairs.
"""
from __future__ import annotations

import torch


def mean_diff(acts_minus, acts_plus):
    """v = mean(misaligned acts) - mean(aligned acts). acts_*: [N, d]. Returns (v, v_hat)."""
    am = torch.as_tensor(acts_minus).float()
    ap = torch.as_tensor(acts_plus).float()
    v = am.mean(0) - ap.mean(0)
    n = v.norm()
    v_hat = v / n if n > 0 else v
    return v, v_hat


def projection(x, v_hat):
    """Scalar projection x . v_hat along the last dim. x: [..., d] -> [...]."""
    return (x * v_hat).sum(-1)


def project_out(x, v_hat):
    """Ablate the v_hat component: x - (x . v_hat) v_hat. x: [..., d]."""
    coeff = projection(x, v_hat).unsqueeze(-1)
    return x - coeff * v_hat


def add_steer(x, v_hat, lam):
    """Steer: x + lam * v_hat. x: [..., d]."""
    return x + lam * v_hat


if __name__ == "__main__":
    torch.manual_seed(0)
    d = 16
    # synthetic aligned/misaligned clusters separated along a known axis
    axis = torch.zeros(d); axis[3] = 1.0
    plus = torch.randn(50, d) * 0.1
    minus = torch.randn(50, d) * 0.1 + 2.0 * axis
    v, v_hat = mean_diff(minus, plus)
    assert torch.allclose(v_hat, axis, atol=0.15), f"recovered axis off: {v_hat[:5]}"

    x = torch.randn(7, d)
    # ablation removes the component
    xa = project_out(x, v_hat)
    assert projection(xa, v_hat).abs().max() < 1e-5, "project_out left residual component"
    # steering adds exactly lambda along v_hat
    xs = add_steer(x, v_hat, 3.0)
    assert torch.allclose(projection(xs, v_hat) - projection(x, v_hat),
                          torch.full((7,), 3.0), atol=1e-4), "steer magnitude wrong"
    print("OK  mean_diff recovers axis; project_out zeroes projection; add_steer adds lambda")
