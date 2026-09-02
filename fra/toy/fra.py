"""Oracle FRA: the bare equations, dense, on ground-truth features.

Stage A of the brief's section 4 -- the ground-truth ``feature_directions`` are
used as ``W_dec`` and the ground-truth activations as ``f``. No SAE anywhere. If
FRA cannot recover a planted edge given a *perfect* dictionary, the method is
broken and no amount of SAE tuning saves it.

Scale, and why it is a parameter
--------------------------------
The brief's Eq. 6 folds ``1/sqrt(d_head)`` into the tensor::

    FRA_QK[q,k,l,m] = f[q,l] f[k,m] (W_dec[l] W_Q . W_dec[m] W_K) / sqrt(d_head)

The repo's conformance harness uses the opposite convention: its
``compute_raw_qk`` compares against an **unscaled** ``q @ k.T``, and
``reconstruct_masked_scores_from_fra`` divides by ``attn_scale`` only after
adding the bias terms. Both reconstruct the score correctly; they just put the
constant in different places. ``attn_scale`` is therefore explicit here --
pass the model's scale for the brief's convention, or ``1.0`` for the harness's.

It makes no difference to any recovery metric: a global positive constant
changes neither the ranking nor the mass fraction.

Memory
------
``[T, T, n_feat, n_feat]`` at ``T=32, n_feat=100`` is 41 MB dense. None of the
paper's Appendix E machinery (sparse COO, top-k truncation, chunking) is needed
at this scale, so this module implements the equations literally, one sequence
at a time.
"""

from __future__ import annotations

import torch
from torch import Tensor
from transformer_lens import HookedTransformer

PATTERN_HOOK = "blocks.0.attn.hook_pattern"
SCORES_HOOK = "blocks.0.attn.hook_attn_scores"


# ── Data-independent circuit quantities ──────────────────────────────────


def coupling_matrix(
    model: HookedTransformer,
    W_dec: Tensor,
    layer: int = 0,
    head: int = 0,
    attn_scale: float | None = None,
) -> Tensor:
    """``G[l, m]`` -- the static feature-feature coupling of this head.

    ``G = (W_dec W_Q)(W_dec W_K)^T / attn_scale``. It depends only on trained
    weights, not on data, and is worth inspecting on its own: it is the
    "coupling constant" matrix between features for this head, and the planted
    edge should show up in it before any data is involved.

    Note ``rank(G) <= d_head``, so a head narrower than ``n_feat`` structurally
    cannot concentrate all mass in one cell. See ``ToyConfig.d_head``.
    """
    attn = model.blocks[layer].attn
    scale = attn.attn_scale if attn_scale is None else attn_scale
    Qf = W_dec @ attn.W_Q[head]  # [n_feat, d_head]
    Kf = W_dec @ attn.W_K[head]  # [n_feat, d_head]
    return (Qf @ Kf.T) / scale


def ov_matrix(
    model: HookedTransformer, W_dec: Tensor, layer: int = 0, head: int = 0
) -> Tensor:
    """``W_dec[l] @ W_V @ W_O`` for every feature -- ``[n_feat, d_model]``."""
    attn = model.blocks[layer].attn
    return W_dec @ attn.W_V[head] @ attn.W_O[head]


# ── The two equations ────────────────────────────────────────────────────


def fra_qk(f: Tensor, G: Tensor, causal: bool = True) -> Tensor:
    """QK: 4-D and bilinear (brief Eq. 6). ``[T, T, n_feat, n_feat]``.

    ``FRA_QK[q,k,l,m] = f[q,l] * f[k,m] * G[l,m]``

    Summing over ``(l, m)`` returns the pre-softmax score, exactly, whenever the
    residual stream is spanned by ``W_dec`` -- which is the condition Gate 1
    establishes.

    Args:
        f: ``[T, n_feat]`` ground-truth activations for ONE sequence.
        G: ``[n_feat, n_feat]`` from :func:`coupling_matrix`.
        causal: zero the ``k > q`` entries, matching the masked attention.
    """
    out = torch.einsum("ql,km,lm->qklm", f, f, G)
    if causal:
        T = f.shape[0]
        mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=f.device), diagonal=1)
        out[mask] = 0.0
    return out


def fra_ov_signed(A: Tensor, f: Tensor, W_ov: Tensor) -> Tensor:
    """OV, kept as a signed vector: ``[T, T, n_feat, d_model]``.

    ``FRA_OV_vec[q,k,l] = A[q,k] * f[k,l] * (W_dec[l] @ W_V @ W_O)``

    The paper collapses this to a norm for memory reasons and thereby loses the
    ability to see contributions that cancel. At toy scale we can afford the
    signed version, and summing it over ``(k, l)`` reproduces ``attn_out[q]``
    exactly -- which the norm version cannot do.

    Args:
        A: ``[T, T]`` post-softmax attention, FROZEN from the real forward pass.
    """
    return torch.einsum("qk,kl,ld->qkld", A, f, W_ov)


def fra_ov(A: Tensor, f: Tensor, W_ov: Tensor) -> Tensor:
    """OV as the paper's norm: ``[T, T, n_feat]`` (brief Eq. 4).

    ``FRA_OV[q,k,l] = A[q,k] * f[k,l] * || W_dec[l] @ W_V @ W_O ||_2``
    """
    norms = W_ov.norm(dim=-1)  # [n_feat]
    return torch.einsum("qk,kl,l->qkl", A, f, norms)


# ── Convenience: everything for one sequence ─────────────────────────────


class OracleFRA:
    """FRA for a single sequence, using ground truth as the dictionary."""

    def __init__(
        self,
        model: HookedTransformer,
        W_dec: Tensor,
        tokens: Tensor,
        f: Tensor,
        layer: int = 0,
        head: int = 0,
        attn_scale: float | None = None,
    ) -> None:
        if tokens.ndim == 1:
            tokens = tokens[None, :]
        self.model = model
        self.W_dec = W_dec
        self.f = f
        self.layer, self.head = layer, head

        self.G = coupling_matrix(model, W_dec, layer, head, attn_scale)
        self.W_ov = ov_matrix(model, W_dec, layer, head)

        with torch.no_grad():
            _, cache = model.run_with_cache(
                tokens, names_filter=[PATTERN_HOOK, SCORES_HOOK]
            )
        self.A = cache[PATTERN_HOOK][0, head]  # [T, T] frozen
        self.scores = cache[SCORES_HOOK][0, head]  # [T, T] masked, pre-softmax

    def qk(self) -> Tensor:
        return fra_qk(self.f, self.G)

    def ov(self) -> Tensor:
        return fra_ov(self.A, self.f, self.W_ov)

    def ov_signed(self) -> Tensor:
        return fra_ov_signed(self.A, self.f, self.W_ov)
