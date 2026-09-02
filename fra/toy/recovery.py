"""Recovery metrics: did FRA find the planted edge? (brief section 6, metrics 1-4)

Two scopes, and the difference matters
--------------------------------------
The brief specifies metric 1 per-cell: "flatten ``|FRA_QK[q,k]|`` over
``(lambda, mu)``, rank descending". But ``f`` is sparse -- roughly ``L0 = 4``
features fire per position -- so a single ``(q, k)`` cell has only
``|active(q)| x |active(k)|`` non-zero entries, on the order of **12**, not
``n_feat**2 = 10,000``. Rank 1 out of 12 is a far weaker claim than rank 1 out of
10,000, and reporting it without saying so would overstate the result.

So both scopes are computed:

* ``cell``      -- at the planted ``(q*, k*)``. The brief's literal metric.
                   Report ``n_nonzero`` alongside it or it is uninterpretable.
* ``aggregate`` -- ``sum_{q,k} |FRA_QK[q,k,l,m]|`` over the sequence, ranking the
                   planted pair against all ``n_feat**2``. This is the "which
                   feature pair drives this head" question, and it is what
                   Dmitry's ``qk_pair_concentration.json`` computes on a real
                   model, so our numbers are comparable to his.

Mass fraction is denominator-dominated -- report ``runner_up_ratio`` too
------------------------------------------------------------------------
``mass_fraction = |planted| / sum|all|`` divides by an L1 over every co-active
pair, and ``G`` is a dense, unregularised matrix: nothing in the loss pushes the
9,999 non-planted entries toward zero. Measured at rho=0, the planted coupling is
7.5x the largest competitor in ``G`` and rank 1 of 100 in its column -- a very
sharp circuit -- yet mass fraction is only 0.096, because ~3,188 individually
tiny pairs sum to swamp the numerator.

So mass fraction mostly measures *how much irrelevant weight the matrix carries*,
which scales with ``n_feat**2`` and with activation density. It is therefore not
comparable across dictionary sizes, and a sweep that moves the denominator can
show "degradation" with no change in the planted circuit at all.

``runner_up_ratio = |planted| / |next largest|`` is reported alongside for that
reason: it is scale-free, denominator-free, and moves only when a competitor
actually catches up with the planted edge.

Signed vs absolute
------------------
Dmitry's note 03 found signed sum *anti*-predictive of causal ablation impact
(Spearman -0.39) while concentration over ``|.|`` was strongly predictive
(+0.86), so ``concentration_ratio_max_over_L1`` is the primary metric here for
comparability. But that result comes from cancellation across 16 heads and many
routes, and we have one head -- so the mechanism does not obviously transfer.
Both rankings are therefore reported, and we have something Dmitry did not:
ground truth to adjudicate which one actually puts the planted pair at rank 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Recovery:
    """Where the planted target landed, under both rankings."""

    rank_abs: int  # 1 = best
    rank_signed: int
    n_total: int  # candidates considered
    n_nonzero: int  # of which actually non-zero
    mass_fraction: float  # |planted| / L1  == Dmitry's concentration ratio
    planted_value: float  # signed
    max_abs: float
    total_l1: float
    total_signed: float
    argmax_is_planted: bool
    runner_up_ratio: float  # |planted| / |next largest| -- denominator-free

    @property
    def in_top5(self) -> bool:
        return self.rank_abs <= 5

    @property
    def in_top20(self) -> bool:
        return self.rank_abs <= 20

    def __str__(self) -> str:
        return (
            f"rank_abs={self.rank_abs:<6d} mass={self.mass_fraction:.4f}  "
            f"runner_up={self.runner_up_ratio:6.2f}x  "
            f"(nonzero {self.n_nonzero}/{self.n_total})"
        )


def recovery_of(values: Tensor, target: tuple[int, ...] | int) -> Recovery:
    """Score how well ``target`` stands out in ``values``.

    Ranks are pessimistic under ties: ``>`` counts strictly better entries, so a
    tie never silently awards rank 1.
    """
    key = target if isinstance(target, tuple) else (target,)
    planted = values[key]

    flat = values.flatten()
    total_l1 = flat.abs().sum().item()
    argmax = torch.unravel_index(flat.abs().argmax(), values.shape)

    # Largest competitor, excluding the planted entry itself.
    others = values.abs().clone()
    others[key] = -1.0
    runner_up = others.max().item()

    return Recovery(
        rank_abs=int((flat.abs() > planted.abs()).sum()) + 1,
        rank_signed=int((flat > planted).sum()) + 1,
        n_total=flat.numel(),
        n_nonzero=int((flat != 0).sum()),
        mass_fraction=(planted.abs() / total_l1).item() if total_l1 > 0 else 0.0,
        planted_value=planted.item(),
        max_abs=flat.abs().max().item(),
        total_l1=total_l1,
        total_signed=flat.sum().item(),
        argmax_is_planted=tuple(int(i) for i in argmax) == key,
        runner_up_ratio=(planted.abs().item() / runner_up) if runner_up > 0 else float("inf"),
    )


# ── QK ───────────────────────────────────────────────────────────────────


def qk_cell_recovery(fra_qk_tensor: Tensor, q: int, k: int, edge: tuple[int, int]) -> Recovery:
    """Metric 1-3 at the planted ``(q*, k*)`` cell -- the brief's literal spec."""
    return recovery_of(fra_qk_tensor[q, k], edge)


def qk_aggregate(fra_qk_tensor: Tensor) -> Tensor:
    """``sum_{q,k} |FRA_QK[q,k,l,m]|`` -- L1 aggregation, ``[n_feat, n_feat]``."""
    return fra_qk_tensor.abs().sum(dim=(0, 1))


def qk_aggregate_signed(fra_qk_tensor: Tensor) -> Tensor:
    """``sum_{q,k} FRA_QK[q,k,l,m]`` -- signed aggregation, ``[n_feat, n_feat]``.

    The counterpart to :func:`qk_aggregate`, and the only way to compare signed
    against absolute honestly: ranking inside an already-absolute aggregate is
    trivially the same under both orderings. Cancellation between ``(q, k)``
    cells can only show up here.
    """
    return fra_qk_tensor.sum(dim=(0, 1))


def qk_aggregate_closed_form(f: Tensor, G: Tensor) -> Tensor:
    """The same aggregate without materialising the 4-D tensor.

    Valid because activations are non-negative, so
    ``sum_{k<=q} |f[q,l] f[k,m] G[l,m]| = |G[l,m]| * (f^T C f)[l,m]``
    with ``C`` the causal mask. Used to check the dense path, and to make the
    rho sweep affordable later.
    """
    T = f.shape[0]
    causal = torch.tril(torch.ones(T, T, dtype=f.dtype, device=f.device))
    return G.abs() * (f.T @ causal @ f)


# ── OV ───────────────────────────────────────────────────────────────────


def ov_cell_recovery(
    fra_ov_tensor: Tensor, q: int, k: int, feature: int
) -> Recovery:
    """Metric 4: where the content feature actually carried at ``k*`` ranks."""
    return recovery_of(fra_ov_tensor[q, k], feature)


def ov_retrieval_precision(
    fra_ov_tensor: Tensor, q: int, k: int, planted: set[int]
) -> bool:
    """Is the top-ranked OV feature one of the planted content features?

    With content features plural, "did FRA find THE feature" is the wrong
    question -- the right one is whether it lands inside the planted set.
    """
    return int(fra_ov_tensor[q, k].abs().argmax()) in planted
