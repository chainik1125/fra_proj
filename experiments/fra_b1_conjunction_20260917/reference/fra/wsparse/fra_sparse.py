"""FRA for weight-sparse models: sparse in the WEIGHTS, dense in the activations.

The toy and the sleeper both had dense `W_dec` with sparse activations, so
`fra/core/fra.py` truncates activations to top-k and iterates `(q,k)` cells. A
weight-sparse model is the reverse:

- `W_dec = I`. Residual channels *are* the units, so there is no dictionary, no
  reconstruction error, and the decomposition is exact by construction.
- Activations are DENSE. On csp_yolo1 (`afrac=None`), `act_in` has 1024 non-zero
  channels of 1024, so there is nothing to truncate.
- The weights are extremely sparse. `G_h = W_q[h]^T W_k[h] / sqrt(d_head)` is
  0.06% dense on layer 10 -- 621 non-zero entries of 1,048,576 for head 82, and
  exactly zero for 28 of 128 heads.

So the right object is `G`'s support. `G` itself is only `d_model^2 = 4.2 MB`
per head, so it is built densely once and reduced to its non-zero index list; the
4-D tensor `[T, T, d_model, d_model]` (17.6 TB at T=64) is never materialised.

Everything here is attribution. Nothing steers.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class HeadG:
    """One head's exact score decomposition.

    `c_attn` is an `nn.Linear` WITH a dense bias (3072/3072 non-zero on
    csp_yolo1, max |b| = 5.82), so the score is not purely bilinear in `act_in`
    and the Appendix-A bias terms are mandatory. Writing `q_h = f W_q[h]^T + b_q`
    and `k_h = f W_k[h]^T + b_k`:

    ```text
    s[q,k] * sqrt(d_head)
        = (f_q W_q^T) . (f_k W_k^T)     <- pair term, on the sparse support
        + (f_q W_q^T) . b_k             <- feat x bias, linear in f_q
        + b_q . (f_k W_k^T)             <- bias x feat, linear in f_k
        + b_q . b_k                     <- constant
    ```

    Only the first is a feature *pair* interaction. The next two are per-position
    linear terms, non-zero only on the channels that feed Q (resp. K), so they
    are sparse too. Attribution must include them: a channel contributes through
    both its pair terms and its feat-x-bias term.
    """

    layer: int
    head: int
    lam: Tensor  # (nnz,) query-side residual channel
    mu: Tensor  # (nnz,) key-side residual channel
    val: Tensor  # (nnz,) G[lam, mu]
    u_q: Tensor  # (d_model,) query-side linear coefficient  (W_q e_lam) . b_k / sqrt(d)
    u_k: Tensor  # (d_model,) key-side linear coefficient    b_q . (W_k e_mu) / sqrt(d)
    const: float  # b_q . b_k / sqrt(d)

    @property
    def nnz(self) -> int:
        return self.val.numel()

    def __str__(self) -> str:
        return f"L{self.layer}H{self.head}: nnz={self.nnz}"


def head_coupling(W_q: Tensor, W_k: Tensor, head: int, d_head: int,
                  layer: int = -1, b_q: Tensor | None = None,
                  b_k: Tensor | None = None) -> HeadG:
    """Support of `G_h[lambda, mu] = (W_q[h] e_lambda) . (W_k[h] e_mu) / sqrt(d_head)`.

    Read it out loud: how strongly residual channel `lambda` at the query
    position couples, through this head, to residual channel `mu` at the key
    position. With `W_dec = I` the FRA coupling matrix *is* `W_QK` for the head.

    Args:
        W_q, W_k: `(n_head * d_head, d_model)` slices of the fused `c_attn`.
    """
    s = slice(head * d_head, (head + 1) * d_head)
    scale = math.sqrt(d_head)
    Wq, Wk = W_q[s], W_k[s]                       # (d_head, d_model)
    G = (Wq.T @ Wk) / scale                       # (d_model, d_model)
    idx = G.nonzero(as_tuple=False)
    d_model = W_q.shape[1]
    if b_q is None:
        u_q = torch.zeros(d_model, dtype=W_q.dtype)
        u_k = torch.zeros(d_model, dtype=W_q.dtype)
        const = 0.0
    else:
        bq, bk = b_q[s], b_k[s]                   # (d_head,)
        u_q = (Wq.T @ bk) / scale                 # (d_model,)
        u_k = (Wk.T @ bq) / scale                 # (d_model,)
        const = float((bq * bk).sum() / scale)
    return HeadG(layer=layer, head=head, lam=idx[:, 0], mu=idx[:, 1],
                 val=G[idx[:, 0], idx[:, 1]], u_q=u_q, u_k=u_k, const=const)


def all_head_couplings(W_q: Tensor, W_k: Tensor, n_head: int, d_head: int,
                       layer: int = -1, b_q: Tensor | None = None,
                       b_k: Tensor | None = None) -> list[HeadG]:
    return [head_coupling(W_q, W_k, h, d_head, layer, b_q, b_k)
            for h in range(n_head)]


def score_exact(g: HeadG, f_q: Tensor, f_k: Tensor) -> float:
    """Full score for one `(q, k)` cell: pair term + both bias terms + constant.

    This is the identity the attribution rests on; it must reproduce the model's
    own `q . k / sqrt(d_head)` to float precision.
    """
    pair = float(fra_qk_on_support(g, f_q, f_k).sum())
    return pair + float(f_q @ g.u_q) + float(f_k @ g.u_k) + g.const


def fra_qk_on_support(g: HeadG, f_q: Tensor, f_k: Tensor) -> Tensor:
    """`FRA_QK[lambda, mu] = f_q[lambda] * f_k[mu] * G[lambda, mu]` on the support.

    Args:
        f_q: `(d_model,)` act_in at the query position.
        f_k: `(d_model,)` act_in at the key position.

    Returns:
        `(nnz,)` aligned with `g.lam` / `g.mu`. Summing it returns this head's
        pre-softmax score contribution for that `(q, k)` pair, exactly.
    """
    return f_q[g.lam] * f_k[g.mu] * g.val


def score_from_support(g: HeadG, f: Tensor) -> Tensor:
    """All-pairs pre-softmax scores for one head: `(T, T)`.

    Equivalent to `(f @ W_q[h].T) @ (f @ W_k[h].T).T / sqrt(d_head)` but computed
    through the support, which is the identity the attribution relies on. Used to
    verify exactness against the model's own q/k.
    """
    T = f.shape[0]
    out = torch.zeros(T, T, dtype=f.dtype)
    fq = f[:, g.lam]  # (T, nnz)
    fk = f[:, g.mu]  # (T, nnz)
    return fq @ (fk * g.val).T if out.numel() else out


def channel_attribution(g: HeadG, f: Tensor, q_pos: int,
                        causal: bool = True) -> tuple[Tensor, Tensor]:
    """Per-channel attribution at one query position, summed over keys.

    Returns `(query_side, key_side)`, each `(d_model,)`:

    - `query_side[lambda] = sum_{k <= q} sum_mu |f[q,lambda] f[k,mu] G[lambda,mu]|`
    - `key_side[mu]       = sum_{k <= q} sum_lambda |...|`

    Absolute value, because a channel that contributes with either sign is still
    contributing; the signed variant is available by dropping `.abs()`. Summing
    over keys asks "how much did this channel shape where `q` looked", which is
    the question the pruning ground truth answers.
    """
    d_model = f.shape[1]
    ks = torch.arange(q_pos + 1) if causal else torch.arange(f.shape[0])
    fq = f[q_pos, g.lam]  # (nnz,)
    fk = f[ks][:, g.mu]  # (n_keys, nnz)
    contrib = (fq * g.val).unsqueeze(0) * fk  # (n_keys, nnz)
    mass = contrib.abs().sum(0)  # (nnz,)
    qs = torch.zeros(d_model, dtype=f.dtype).index_add_(0, g.lam, mass)
    ks_ = torch.zeros(d_model, dtype=f.dtype).index_add_(0, g.mu, mass)
    return qs, ks_
