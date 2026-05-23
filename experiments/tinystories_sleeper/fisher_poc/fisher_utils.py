"""Fisher-guided clean-recovery utilities for the TinyStories-33M sleeper.

Implements the POC subset of `docs/dmitry/theory/fisher_proposal.md`:
  - JSD-in-bits between softmax distributions of two logits (sec. 3, 10.3)
  - Analytic gradient of JSD w.r.t. p-logits          (sec. 7, 10.4)
  - Diagonal categorical Fisher F_ii from p, U_i      (sec. 4, 6, 10.6)
  - Finite-difference logit Jacobian U_i = dz/dθ_i    (sec. 10.5)
  - Greedy clean-recovery loop with line search       (sec. 6, 10.7)
  - Fisher path-length L_F                            (sec. 9)

The implementation deliberately keeps to the diagonal-only variant (sec. 6/8.1).
Empirical / natural-gradient / packaged variants (sec. 8.2, 10.8, 11) are
deferred. Forward passes for U_i are delegated to a caller-supplied
`forward_logits(tokens, theta) -> logits` closure so the same loop can drive
both FRA-OV→OV and conventional resid-mid steering.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch
import torch.nn.functional as F


# JSD in bits is JSD-in-nats / ln(2). Keep as a python float for clarity.
_LN2 = math.log(2.0)


# ---------------------------------------------------------------------------
# JSD and its gradient.
# ---------------------------------------------------------------------------


def jsd_bits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Mean JSD-in-bits over masked positions.

    Args:
        logits_p, logits_q: [B, T, V] logits (any dtype; cast to float32).
        mask:               [B, T] bool/0-1, positions to include.

    Returns:
        Scalar tensor on `logits_p.device`.
    """
    if logits_p.shape != logits_q.shape:
        raise ValueError(
            f"jsd_bits: logits shape mismatch {tuple(logits_p.shape)} vs "
            f"{tuple(logits_q.shape)}"
        )
    p_logits = logits_p.float()
    q_logits = logits_q.float()
    logp = F.log_softmax(p_logits, dim=-1)
    logq = F.log_softmax(q_logits, dim=-1)
    p = logp.exp()
    q = logq.exp()
    m = 0.5 * (p + q)
    logm = torch.log(m.clamp_min(1e-30))
    kl_p_m = (p * (logp - logm)).sum(dim=-1)
    kl_q_m = (q * (logq - logm)).sum(dim=-1)
    jsd_nats = 0.5 * (kl_p_m + kl_q_m)
    jsd = jsd_nats / _LN2
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return (jsd * mask_f).sum() / denom


def grad_jsd_wrt_p_logits(
    logits_p: torch.Tensor,
    logits_q: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """∂ jsd_bits(p,q) / ∂ logits_p, averaged over masked positions.

    Returns [B, T, V] tensor whose contraction with ∂z_p/∂θ_i gives g_i.
    Already divided by mask.sum() and multiplied by mask so that
    `(grad_z * U_i).sum() == d J_clean / d θ_i` per the §7 derivation.
    """
    p = F.softmax(logits_p.float(), dim=-1)
    q = F.softmax(logits_q.float(), dim=-1)
    m = 0.5 * (p + q)
    a = 0.5 * (torch.log(p.clamp_min(1e-30)) - torch.log(m.clamp_min(1e-30))) / _LN2
    mean_a = (p * a).sum(dim=-1, keepdim=True)
    grad_z = p * (a - mean_a)
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)
    return grad_z * mask_f.unsqueeze(-1) / denom


# ---------------------------------------------------------------------------
# Diagonal Fisher and finite-difference gradient/Fisher in one pass.
# ---------------------------------------------------------------------------


@dataclass
class GradAndFisher:
    """Output of `gradient_and_diag_fisher`."""

    J_clean: torch.Tensor       # scalar, mean JSD bits
    g: torch.Tensor             # [K] gradient of J_clean w.r.t. theta
    F_diag: torch.Tensor        # [K] diagonal Fisher entries
    logits: torch.Tensor        # [B, T, V] logits at the current theta


def gradient_and_diag_fisher(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    theta: torch.Tensor,
    clean_logits: torch.Tensor,
    mask: torch.Tensor,
    eps_fd: float = 1e-2,
) -> GradAndFisher:
    """Compute (J_clean, g, F_diag) via finite-difference logit Jacobian.

    For each control i, two forward passes with theta ± eps·e_i give
    U_i = dz/dθ_i ∈ R^{B,T,V}. Then:
        g_i    = <grad_z J_clean, U_i>
        F_ii   = E_(c,t) [Σ_y p_y U_i(y)^2 − (Σ_y p_y U_i(y))^2]
    Per the proposal's §7 / §10.6.

    Args:
        forward_logits: closure taking theta∈R^K, returning [B,T,V] logits.
        theta:          [K] current control vector.
        clean_logits:   [B,T,V] reference (clean) logits.
        mask:           [B,T] positions to include.
        eps_fd:         FD step size.
    """
    device = theta.device
    K = theta.shape[0]

    logits = forward_logits(theta).detach()
    p = F.softmax(logits.float(), dim=-1)
    grad_z = grad_jsd_wrt_p_logits(logits, clean_logits, mask)
    J_clean = jsd_bits(logits, clean_logits, mask).detach()

    g = torch.zeros(K, device=device, dtype=torch.float32)
    F_diag = torch.zeros(K, device=device, dtype=torch.float32)
    mask_f = mask.float()
    denom = mask_f.sum().clamp_min(1.0)

    for i in range(K):
        e = torch.zeros_like(theta)
        e[i] = eps_fd
        logits_plus = forward_logits(theta + e).detach().float()
        logits_minus = forward_logits(theta - e).detach().float()
        U_i = (logits_plus - logits_minus) / (2.0 * eps_fd)  # [B,T,V]

        g[i] = (grad_z * U_i).sum()

        mean_u = (p * U_i).sum(dim=-1)            # [B,T]
        mean_u2 = (p * U_i.square()).sum(dim=-1)  # [B,T]
        fisher_per_pos = mean_u2 - mean_u.square()
        F_diag[i] = (fisher_per_pos * mask_f).sum() / denom

        del logits_plus, logits_minus, U_i, mean_u, mean_u2, fisher_per_pos

    return GradAndFisher(J_clean=J_clean, g=g, F_diag=F_diag, logits=logits)


# ---------------------------------------------------------------------------
# Greedy diagonal-Fisher clean-recovery loop.
# ---------------------------------------------------------------------------


@dataclass
class TrajectoryStep:
    step: int
    selected: int
    score: float
    J_clean: float
    crf: float
    step_jsd: float
    accepted: bool
    delta_theta: float


def greedy_fisher_clean_recovery(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    K: int,
    clean_logits: torch.Tensor,
    mask: torch.Tensor,
    num_steps: int = 10,
    eps_fd: float = 1e-2,
    target_step_jsd_bits: float = 1e-4,
    delta_cap: float = 5.0,
    device: torch.device | str = "cuda",
    log_fn: Callable[[str], None] | None = print,
) -> tuple[torch.Tensor, list[TrajectoryStep]]:
    """Greedy diagonal-Fisher clean-recovery loop (proposal §6 / §10.7).

    At each step:
      1. compute g, F_diag with FD;
      2. score S_i = |g_i| / √(F_ii + ε) over un-selected controls;
      3. step in direction -sign(g_i)·e_i with size √(8 ln 2·ρ / F_ii);
         then line-search shrink {1, 1/2, 1/4, 1/8, 1/16}.
      4. require J_clean to decrease AND step_jsd ≤ 2·ρ.

    Args:
        forward_logits: closure taking θ∈R^K, returning [B,T,V] logits.
        K:              number of control coordinates.
        clean_logits:   [B,T,V] reference logits at the same prompts.
        mask:           [B,T] inclusion mask.
        num_steps:      max greedy steps.
        eps_fd:         FD step for U_i.
        target_step_jsd_bits: ρ in §5/§6.
        delta_cap:      hard cap on |δθ_i| in addition to the Fisher budget.
        device:         where to allocate θ.
        log_fn:         optional logger (set to None to silence).

    Returns:
        theta_final, trajectory  (list of TrajectoryStep records).
    """
    theta = torch.zeros(K, device=device, dtype=torch.float32)
    selected: set[int] = set()
    trajectory: list[TrajectoryStep] = []
    J0: torch.Tensor | None = None

    for step in range(num_steps):
        gf = gradient_and_diag_fisher(
            forward_logits, theta, clean_logits, mask, eps_fd=eps_fd
        )
        if J0 is None:
            J0 = gf.J_clean.detach().clone()

        scores = torch.abs(gf.g) / torch.sqrt(gf.F_diag.clamp_min(1e-12))
        for i in selected:
            scores[i] = float("-inf")
        i = int(torch.argmax(scores).item())

        sign = -torch.sign(gf.g[i])
        if sign == 0:
            sign = torch.tensor(1.0, device=device)
        budget = math.sqrt(
            8.0 * _LN2 * target_step_jsd_bits / max(float(gf.F_diag[i].item()), 1e-12)
        )
        magnitude = min(budget, delta_cap)

        accepted = False
        chosen_shrink = 0.0
        chosen_step_jsd = float("nan")
        for shrink in (1.0, 0.5, 0.25, 0.125, 0.0625):
            delta = torch.zeros_like(theta)
            delta[i] = sign * shrink * magnitude
            theta_cand = (theta + delta).detach()
            logits_cand = forward_logits(theta_cand).detach()
            J_cand = jsd_bits(logits_cand, clean_logits, mask).detach()
            step_jsd = jsd_bits(logits_cand, gf.logits, mask).detach()
            if J_cand.item() < gf.J_clean.item() and step_jsd.item() <= 2.0 * target_step_jsd_bits:
                theta = theta_cand
                accepted = True
                chosen_shrink = shrink
                chosen_step_jsd = float(step_jsd.item())
                break

        crf = float(((J0 - gf.J_clean) / J0.clamp_min(1e-12)).item())
        rec = TrajectoryStep(
            step=step,
            selected=i,
            score=float(scores[i].item()),
            J_clean=float(gf.J_clean.item()),
            crf=crf,
            step_jsd=chosen_step_jsd,
            accepted=accepted,
            delta_theta=float(sign.item() * chosen_shrink * magnitude) if accepted else 0.0,
        )
        trajectory.append(rec)
        if log_fn:
            log_fn(
                f"[fisher] step={step:02d} i={i:5d} score={rec.score:.3e} "
                f"J={rec.J_clean:.4e} CRF={rec.crf:+.4f} "
                f"step_jsd={rec.step_jsd:.3e} accepted={accepted}"
            )
        if not accepted:
            break
        selected.add(i)

    return theta.detach(), trajectory


# ---------------------------------------------------------------------------
# Experiment A: re-evaluate an existing 1-D α-sweep under Fisher path length.
# ---------------------------------------------------------------------------


def fisher_path_length_1d(
    forward_logits: Callable[[torch.Tensor], torch.Tensor],
    alphas: list[float] | torch.Tensor,
    clean_logits: torch.Tensor,
    mask: torch.Tensor,
    eps_fd: float = 1e-2,
) -> dict[str, list[float]]:
    """For a 1-D parametrization θ = α, compute the table needed for §12 Exp A.

    Returns a dict with parallel lists keyed by:
      alpha, J_clean (bits), CRF, F_aa, L_F, step_jsd_to_prev.

    L_F is the cumulative Fisher arc length
        L_F(T) = Σ_{t<T} √(δθᵀ F δθ / 8 ln 2)
    and is computed using F(α) evaluated at each grid point.
    """
    alphas = list(map(float, alphas))
    K = 1
    # Probe device via a θ=0 forward pass.
    probe = forward_logits(torch.zeros(K))
    device = probe.device
    del probe

    out: dict[str, list[float]] = {
        "alpha": [], "J_clean": [], "CRF": [], "F_aa": [], "L_F": [], "step_jsd_to_prev": []
    }
    J0 = None
    L_F = 0.0
    prev_logits = None

    for a in alphas:
        theta = torch.tensor([a], device=device, dtype=torch.float32)
        gf = gradient_and_diag_fisher(
            forward_logits, theta, clean_logits, mask, eps_fd=eps_fd
        )
        if J0 is None:
            J0 = gf.J_clean.detach().clone()
        if prev_logits is not None and len(out["alpha"]) > 0:
            d_alpha = a - out["alpha"][-1]
            quad = float(gf.F_diag[0].item()) * d_alpha * d_alpha
            L_F += math.sqrt(max(quad, 0.0) / (8.0 * _LN2))
            step_jsd = float(jsd_bits(gf.logits, prev_logits, mask).item())
        else:
            step_jsd = 0.0
        out["alpha"].append(a)
        out["J_clean"].append(float(gf.J_clean.item()))
        out["CRF"].append(float(((J0 - gf.J_clean) / J0.clamp_min(1e-12)).item()))
        out["F_aa"].append(float(gf.F_diag[0].item()))
        out["L_F"].append(L_F)
        out["step_jsd_to_prev"].append(step_jsd)
        prev_logits = gf.logits.detach()

    return out
