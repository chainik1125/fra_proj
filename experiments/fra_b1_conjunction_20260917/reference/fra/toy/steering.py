"""Baseline interventions, for the "better than what?" comparison.

The intervention work established that the FRA-identified pair is causally
load-bearing and that matched-random and runner-up pairs are not. That is
*specificity*. It is not *better* -- nothing was compared against. Dmitry's step
4 asks whether we can control the expressed features **better**, and the FRA
paper's whole Pareto framing rests on that comparison.

Three interventions on the same target feature, in increasing breadth:

1. **FRA pair ablation** (:mod:`fra.toy.intervention`) -- subtract
   ``scale * f[q,l*] f[k,m*] G[l*,m*]`` from the pre-softmax scores. Touches one
   feature *pair*, in the QK path only.

2. **Activation ablation** (``ablate_feature_qkv``) -- remove
   ``alpha * f[t,l*] W_dec[l*]`` from the input to ``W_Q``, ``W_K`` and ``W_V``.
   This is what the paper calls "QK->QK". It hits Q, K *and* V, so it is
   strictly broader than (1): (1) removes one pair's contribution to one score,
   this removes the feature from every query, key and value at every position.

3. **Residual steering** (``steer_residual``) -- subtract
   ``alpha * f[t,l*] W_dec[l*]`` from the residual stream. The conventional
   SAE-feature ablation, and the paper's "conventional additive" baseline.
   Broader still than (2): because ``ln1`` is the identity here, it feeds Q/K/V
   exactly as (2) does, *and* the edit survives on the skip connection into
   ``resid_post``, so it also changes what the unembedding reads.

So the three are nested by construction: (1) subset of (2) subset of (3). If the
frontiers coincide, narrowness bought nothing.

Hook choice note: ``ln1`` is ``Identity`` (no LayerNorm by design) and
``hook_attn_in`` / ``hook_q_input`` are inert unless ``use_attn_in`` or
``use_split_qkv_input`` is set, which would change the model and invalidate the
cached checkpoints. So (2) hooks ``hook_q``/``hook_k``/``hook_v`` and subtracts
the *projected* component, which is algebraically identical to zeroing the
feature in the block's input::

    q' = (x - a f[t,l] W_dec[l]) W_Q = q - a f[t,l] (W_dec[l] W_Q)
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

from fra.toy.dgp import ToyBatch
from fra.toy.metrics import accuracy, attention_concentration

RESID_PRE = "blocks.0.hook_resid_pre"
HOOK_Q = "blocks.0.attn.hook_q"
HOOK_K = "blocks.0.attn.hook_k"
HOOK_V = "blocks.0.attn.hook_v"


@contextmanager
def steer_residual(
    model: HookedTransformer, f: Tensor, W_dec: Tensor, lam: int, alpha: float
):
    """Conventional SAE-feature ablation on the residual stream.

    ``alpha = 1`` removes the feature's contribution exactly; larger over-removes.
    """
    delta = alpha * f[:, :, lam].unsqueeze(-1) * W_dec[lam]  # [batch, seq, d_model]

    def hook(resid, hook):
        return resid - delta

    with model.hooks(fwd_hooks=[(RESID_PRE, hook)]):
        yield


@contextmanager
def ablate_feature_qkv(
    model: HookedTransformer,
    f: Tensor,
    W_dec: Tensor,
    lam: int,
    alpha: float,
    layer: int = 0,
    head: int = 0,
):
    """Remove the feature from the input to W_Q, W_K and W_V (the paper's QK->QK).

    Leaves the residual stream itself untouched, so the skip connection into
    ``resid_post`` still carries the feature -- that is the difference from
    :func:`steer_residual`.
    """
    attn = model.blocks[layer].attn
    coef = alpha * f[:, :, lam]  # [batch, seq]
    deltas = {
        HOOK_Q: coef.unsqueeze(-1) * (W_dec[lam] @ attn.W_Q[head]),
        HOOK_K: coef.unsqueeze(-1) * (W_dec[lam] @ attn.W_K[head]),
        HOOK_V: coef.unsqueeze(-1) * (W_dec[lam] @ attn.W_V[head]),
    }

    def make(delta):
        def hook(x, hook):  # x: [batch, seq, n_heads, d_head]
            out = x.clone()
            out[:, :, head] = out[:, :, head] - delta
            return out

        return hook

    with model.hooks(fwd_hooks=[(n, make(d)) for n, d in deltas.items()]):
        yield


@dataclass
class InterventionPoint:
    """One point on a method's Pareto curve."""

    method: str
    strength: float
    acc_query: float  # target behaviour remaining -- LOWER is more suppression
    acc_elsewhere: float  # collateral, accuracy form -- HIGHER is less damage
    collateral_kl: float  # collateral, distributional form -- LOWER is less damage
    collateral_frac: float  # fraction of non-target positions perturbed at all
    collateral_max_dlogit: float
    mass_on_key: float
    argmax_is_key: float

    def suppression(self, base_query: float) -> float:
        return base_query - self.acc_query

    def collateral(self, base_elsewhere: float) -> float:
        return base_elsewhere - self.acc_elsewhere


@torch.no_grad()
def measure(
    model: HookedTransformer,
    b: ToyBatch,
    method: str,
    strength: float,
    base_logits: Tensor,
) -> InterventionPoint:
    """Evaluate whatever intervention is already active in the caller's context.

    Collateral is measured two ways, because the accuracy form saturates. The
    label away from the query position is a single constant token, and the model
    predicts it so robustly that a 15-logit perturbation does not flip the argmax
    -- so accuracy-elsewhere reports ~0 damage for every method at every strength
    and the axis has no dynamic range.

    ``collateral_kl`` is the mean ``KL(base || intervened)`` over non-target
    positions. It is the toy analogue of the paper's coherence axis: it asks how
    far the output distribution moved where it should not have moved at all,
    rather than whether the top-1 token happened to survive.
    """
    logits = model(b.tokens)
    acc = accuracy(model, b)
    conc = attention_concentration(model, b)

    rows = torch.arange(b.tokens.shape[0])
    target = torch.zeros(b.tokens.shape[:2], dtype=torch.bool)
    target[rows, b.query_pos] = True
    other = ~target

    logp_base = torch.log_softmax(base_logits, dim=-1)
    logp_new = torch.log_softmax(logits, dim=-1)
    kl = (logp_base.exp() * (logp_base - logp_new)).sum(-1)  # [batch, seq]
    dlogit = (logits - base_logits).abs().amax(-1)

    return InterventionPoint(
        method=method,
        strength=strength,
        acc_query=acc.at_query,
        acc_elsewhere=acc.elsewhere,
        collateral_kl=kl[other].mean().item(),
        collateral_frac=(dlogit[other] > 1e-5).float().mean().item(),
        collateral_max_dlogit=dlogit[other].max().item(),
        mass_on_key=conc.mean_mass_on_key,
        argmax_is_key=conc.argmax_is_key,
    )
