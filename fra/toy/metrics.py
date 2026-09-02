"""Behavioural and attention metrics -- everything that must pass before FRA.

Two questions, in order:

1. **Did the model learn the rule, and did it learn it at the feature level?**
   Accuracy at ``lambda*`` positions is the only number that matters; accuracy
   elsewhere is near-free (the label is a constant) and is reported purely as a
   contrast. Held-out variants then separate *feature* solving from token
   memorisation.

2. **Gate 2: did it learn the rule THROUGH ATTENTION?**
   High accuracy is not sufficient evidence. The model could route around the
   head -- diffuse attention plus an unembed that exploits some statistical
   regularity. If that happens FRA will correctly report that no strong edge
   exists, and we would spend a day concluding the method failed when the model
   simply never used the mechanism. So we assert on the attention pattern itself
   *before* computing any FRA.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from fra.toy.dgp import ToyBatch

PATTERN_HOOK = "blocks.0.attn.hook_pattern"


@dataclass
class Accuracy:
    """Accuracy split by position role. ``at_query`` is the planted rule."""

    at_query: float
    elsewhere: float
    n_query: int

    def __str__(self) -> str:
        return (
            f"query={self.at_query * 100:6.2f}%  "
            f"elsewhere={self.elsewhere * 100:6.2f}%  (n_query={self.n_query})"
        )


@torch.no_grad()
def accuracy(model: HookedTransformer, b: ToyBatch) -> Accuracy:
    pred = model(b.tokens).argmax(dim=-1)  # [batch, seq]
    correct = pred == b.targets

    rows = torch.arange(b.tokens.shape[0])
    query_mask = torch.zeros_like(correct, dtype=torch.bool)
    query_mask[rows, b.query_pos] = True

    return Accuracy(
        at_query=correct[query_mask].float().mean().item(),
        elsewhere=correct[~query_mask].float().mean().item(),
        n_query=int(query_mask.sum()),
    )


@dataclass
class AttentionConcentration:
    """Gate 2: does attention at the query actually land on the key?"""

    mean_mass_on_key: float
    argmax_is_key: float
    mean_mass_on_self: float
    mean_top1_mass: float
    n: int

    def __str__(self) -> str:
        return (
            f"mass_on_key={self.mean_mass_on_key:.4f}  "
            f"argmax_is_key={self.argmax_is_key * 100:6.2f}%  "
            f"(self={self.mean_mass_on_self:.4f}, top1={self.mean_top1_mass:.4f}, n={self.n})"
        )


@torch.no_grad()
def attention_concentration(model: HookedTransformer, b: ToyBatch) -> AttentionConcentration:
    """Attention mass at ``lambda*`` query positions, measured on the ``mu*`` key.

    ``argmax_is_key`` is the number to watch: the fraction of query positions
    whose single most-attended key IS the planted key position. If that is low,
    the model solved the task by some route other than the planted skip-trigram,
    and the task or the training needs fixing before FRA -- not after.

    ``mean_mass_on_self`` is reported alongside because a 1L attention-only model
    with no BOS token has nowhere to dump leftover probability except the query
    position itself; a large self-mass with a correct argmax is benign, whereas a
    large self-mass with a wrong argmax means the head is idling.
    """
    _, cache = model.run_with_cache(b.tokens, names_filter=[PATTERN_HOOK])
    pattern = cache[PATTERN_HOOK][:, 0]  # [batch, seq, seq] -- single head

    rows = torch.arange(b.tokens.shape[0])
    at_query = pattern[rows, b.query_pos]  # [batch, seq] attention from q over all k

    mass_on_key = at_query[rows, b.key_pos]
    mass_on_self = at_query[rows, b.query_pos]

    # Restrict the argmax to causally visible positions.
    positions = torch.arange(b.tokens.shape[1])
    visible = positions[None, :] <= b.query_pos[:, None]
    masked = at_query.masked_fill(~visible, float("-inf"))
    top1 = masked.argmax(dim=-1)

    return AttentionConcentration(
        mean_mass_on_key=mass_on_key.mean().item(),
        argmax_is_key=(top1 == b.key_pos).float().mean().item(),
        mean_mass_on_self=mass_on_self.mean().item(),
        mean_top1_mass=masked.max(dim=-1).values.mean().item(),
        n=len(rows),
    )


@torch.no_grad()
def attention_mass_by_role(model: HookedTransformer, b: ToyBatch) -> dict[str, float]:
    """Where the remaining attention goes, for diagnosing a failed Gate 2."""
    _, cache = model.run_with_cache(b.tokens, names_filter=[PATTERN_HOOK])
    pattern = cache[PATTERN_HOOK][:, 0]
    rows = torch.arange(b.tokens.shape[0])
    at_query = pattern[rows, b.query_pos]

    positions = torch.arange(b.tokens.shape[1])
    visible = positions[None, :] <= b.query_pos[:, None]

    key_mask = torch.zeros_like(visible)
    key_mask[rows, b.key_pos] = True
    self_mask = torch.zeros_like(visible)
    self_mask[rows, b.query_pos] = True
    other = visible & ~key_mask & ~self_mask

    return {
        "key": at_query[key_mask].sum().item() / len(rows),
        "self": at_query[self_mask].sum().item() / len(rows),
        "other": (at_query * other).sum().item() / len(rows),
    }
