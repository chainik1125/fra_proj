"""Feature-pair ablation: the causal half of Dmitry's step 4.

Attribution is only meaningful if intervention confirms it. FRA is an
*intervention* paper, and attribution without a causal test is precisely the
weakness that sank its Case 2. This module closes the loop against ground truth.

What the paper advertises but never runs
----------------------------------------
FRA's Figure 1(c) shows cutting a single feature-pair interaction while
preserving the others. That experiment does not appear in the paper. Its
"QK->QK" intervention edits the shared *input* to ``W_Q``, ``W_K`` and ``W_V``
for all heads -- it is not a per-pair QK intervention. What follows is the
experiment the figure depicts.

How the ablation works
----------------------
Gate 3 established that the pre-softmax score is exactly the sum of the FRA
tensor over feature pairs::

    s[q,k] = sum_{l,m} f[q,l] f[k,m] G[l,m]

So zeroing one entry and re-summing is exactly subtracting that one term::

    s'[q,k] = s[q,k] - f[q,lambda] f[k,mu] G[lambda,mu]

Applying that as a hook on ``hook_attn_scores`` lets the rest of the forward
pass -- softmax, the OV path with the *new* attention pattern, unembed -- happen
downstream on its own. This is a genuine causal intervention on one feature
pair, not a re-scoring of a frozen pattern.

Causal masking survives: masked entries are ``-inf`` and remain ``-inf`` under
subtraction of a finite quantity.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from fra.toy.dgp import ToyBatch
from fra.toy.metrics import Accuracy, accuracy, attention_concentration

SCORES_HOOK = "blocks.0.attn.hook_attn_scores"


def pair_delta(f: Tensor, G: Tensor, lam: int, mu: int) -> Tensor:
    """Score contribution of one feature pair: ``[batch, q, k]``."""
    return f[:, :, lam].unsqueeze(2) * f[:, :, mu].unsqueeze(1) * G[lam, mu]


def pair_delta_batched(f: Tensor, G: Tensor, lams: Tensor, mus: Tensor) -> Tensor:
    """Score contribution of a DIFFERENT pair per sequence: ``[batch, q, k]``.

    Needed for the cell-scope competitor, which is chosen per sequence: which
    features are live at the planted ``(q*, k*)`` depends on that sequence's
    distractors and content value.
    """
    batch, seq, _ = f.shape
    fq = f.gather(2, lams.view(batch, 1, 1).expand(batch, seq, 1)).squeeze(-1)
    fk = f.gather(2, mus.view(batch, 1, 1).expand(batch, seq, 1)).squeeze(-1)
    return fq.unsqueeze(2) * fk.unsqueeze(1) * G[lams, mus].view(batch, 1, 1)


def cell_runner_up_pairs(
    f: Tensor, G: Tensor, query_pos: Tensor, key_pos: Tensor, planted: tuple[int, int]
) -> tuple[Tensor, Tensor]:
    """The largest live competitor AT each sequence's planted ``(q*, k*)`` cell.

    The aggregate runner-up is not a valid causal competitor. It is selected over
    all ``(q, k)``, so its query feature is typically inactive at ``q*`` -- and
    then its contribution *at the planted cell* is identically zero and ablating
    it cannot possibly change the behaviour. "No effect" would be a foregone
    conclusion about indexing, not evidence that FRA's ranking survives.

    Entries of ``FRA_QK[q*, k*]`` are non-zero only where both features are
    active at the relevant positions, so the largest of them (excluding the
    planted pair) is a genuine competitor for the causal role. This is the arm
    that tests whether the planted edge has become *interchangeable* with a
    competitor at high overlap.
    """
    batch = f.shape[0]
    rows = torch.arange(batch)
    fq = f[rows, query_pos]  # [batch, n_feat]
    fk = f[rows, key_pos]  # [batch, n_feat]

    cell = fq.unsqueeze(2) * fk.unsqueeze(1) * G.unsqueeze(0)  # [batch, n_feat, n_feat]
    cell[:, planted[0], planted[1]] = 0.0

    n = G.shape[0]
    flat = cell.abs().reshape(batch, -1).argmax(dim=1)
    return flat // n, flat % n


def aggregate_pair_is_live_at_cell(
    f: Tensor, query_pos: Tensor, key_pos: Tensor, pair: tuple[int, int]
) -> float:
    """Fraction of sequences where ``pair`` is actually non-zero at ``(q*, k*)``.

    Reported so the structural point above is visible in the data rather than
    only in an argument.
    """
    rows = torch.arange(f.shape[0])
    live = (f[rows, query_pos, pair[0]] > 0) & (f[rows, key_pos, pair[1]] > 0)
    return live.float().mean().item()


@contextmanager
def ablate_pair(
    model: HookedTransformer,
    f: Tensor,
    G: Tensor,
    lam: int,
    mu: int,
    scale: float = 1.0,
    head: int = 0,
):
    """Remove ``scale`` times the ``(lambda, mu)`` contribution from the scores."""
    delta = pair_delta(f, G, lam, mu) * scale

    def hook(scores, hook):  # scores: [batch, n_heads, q, k]
        scores[:, head] = scores[:, head] - delta
        return scores

    with model.hooks(fwd_hooks=[(SCORES_HOOK, hook)]):
        yield


@contextmanager
def ablate_pairs_per_sequence(
    model: HookedTransformer,
    f: Tensor,
    G: Tensor,
    lams: Tensor,
    mus: Tensor,
    head: int = 0,
):
    """Ablate a different feature pair in each sequence of the batch."""
    delta = pair_delta_batched(f, G, lams, mus)

    def hook(scores, hook):
        scores[:, head] = scores[:, head] - delta
        return scores

    with model.hooks(fwd_hooks=[(SCORES_HOOK, hook)]):
        yield


@dataclass
class AblationResult:
    """Behaviour and attention after removing one feature pair."""

    arm: str
    pair: tuple[int, int]
    scale: float
    removed_l1: float  # total |score mass| removed, for matching across arms
    acc: Accuracy
    mass_on_key: float
    argmax_is_key: float

    def __str__(self) -> str:
        return (
            f"{self.arm:12s} pair=({self.pair[0]:3d},{self.pair[1]:3d}) "
            f"acc={self.acc.at_query*100:6.2f}%  "
            f"mass_on_key={self.mass_on_key:.4f}  argmax_is_key={self.argmax_is_key*100:6.2f}%  "
            f"removed_L1={self.removed_l1:.1f}"
        )


@torch.no_grad()
def evaluate_ablation(
    model: HookedTransformer,
    b: ToyBatch,
    G: Tensor,
    pair: tuple[int, int],
    arm: str,
    scale: float = 1.0,
) -> AblationResult:
    lam, mu = pair
    removed = pair_delta(b.features, G, lam, mu) * scale
    # Only the causally visible half is actually removed from anything.
    T = b.tokens.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    removed_l1 = removed[:, causal].abs().sum().item()

    with ablate_pair(model, b.features, G, lam, mu, scale):
        acc = accuracy(model, b)
        conc = attention_concentration(model, b)

    return AblationResult(
        arm=arm,
        pair=pair,
        scale=scale,
        removed_l1=removed_l1,
        acc=acc,
        mass_on_key=conc.mean_mass_on_key,
        argmax_is_key=conc.argmax_is_key,
    )


@torch.no_grad()
def evaluate_ablation_per_sequence(
    model: HookedTransformer,
    b: ToyBatch,
    G: Tensor,
    lams: Tensor,
    mus: Tensor,
    arm: str,
) -> AblationResult:
    removed = pair_delta_batched(b.features, G, lams, mus)
    T = b.tokens.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    removed_l1 = removed[:, causal].abs().sum().item()

    with ablate_pairs_per_sequence(model, b.features, G, lams, mus):
        acc = accuracy(model, b)
        conc = attention_concentration(model, b)

    # Most common pair, purely for reporting.
    n = G.shape[0]
    flat = (lams * n + mus)
    common = int(torch.bincount(flat, minlength=n * n).argmax())

    return AblationResult(
        arm=arm,
        pair=(common // n, common % n),
        scale=1.0,
        removed_l1=removed_l1,
        acc=acc,
        mass_on_key=conc.mean_mass_on_key,
        argmax_is_key=conc.argmax_is_key,
    )


def matched_random_pair(
    G: Tensor,
    b: ToyBatch,
    planted: tuple[int, int],
    exclude: set[tuple[int, int]] | None = None,
    generator: torch.Generator | None = None,
) -> tuple[tuple[int, int], float]:
    """A random pair, rescaled to remove the same total score mass as the planted one.

    At low rho the planted edge dominates by design, so no naturally-occurring
    pair has matched magnitude and an unscaled random control would be a much
    weaker perturbation -- it would "show no effect" for the trivial reason that
    it barely perturbed anything. Rescaling makes the control remove an equal
    amount of score mass at a different location, which is the comparison that
    isolates *where* the mass was removed from.
    """
    exclude = (exclude or set()) | {planted}
    n = G.shape[0]
    T = b.tokens.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))

    target = pair_delta(b.features, G, *planted)[:, causal].abs().sum().item()

    for _ in range(1000):
        i = int(torch.randint(n, (1,), generator=generator))
        j = int(torch.randint(n, (1,), generator=generator))
        if (i, j) in exclude:
            continue
        mass = pair_delta(b.features, G, i, j)[:, causal].abs().sum().item()
        if mass > 1e-8:
            return (i, j), target / mass
    raise RuntimeError("no usable random pair found")


def runner_up_pair(aggregate: Tensor, planted: tuple[int, int]) -> tuple[int, int]:
    """The highest-magnitude pair other than the planted one."""
    scores = aggregate.abs().clone()
    scores[planted] = -1.0
    idx = torch.unravel_index(scores.argmax(), scores.shape)
    return tuple(int(i) for i in idx)
