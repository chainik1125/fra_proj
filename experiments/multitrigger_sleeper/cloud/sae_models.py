"""SAE and crosscoder classes used by the TinyStories sleeper experiment.

Ported from sae_day/src/sae_day/sae.py (commit 38cd98e on branch mess3_climb).
Only the five classes actually used by this experiment are kept — the full
sae_day module also contains MatryoshkaSAE, HierarchicalSAE, and
TemporalBatchTopKSAE, which the sleeper pipeline does not exercise.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Standard TopK sparse autoencoder.

    z = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k)            (use_relu=True, default)
    z = TopK(W_enc @ (x - b_dec) + b_enc, k)                  (use_relu=False)
    x_hat = W_dec @ z + b_dec

    ``use_relu=False`` matches the Bussmann et al. 2024 BatchTopK formulation
    (no ReLU before TopK; latents may be negative). ``use_relu=True`` matches
    Marks/Karvonen/Mueller `dictionary_learning` and our default.
    """

    def __init__(self, d_in: int, d_sae: int, k: int, use_relu: bool = True):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k
        self.use_relu = use_relu

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = (x - self.b_dec) @ self.W_enc + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


class TemporalCrosscoder(nn.Module):
    """Shared-latent temporal crosscoder (ckkissane-style).

    Encodes a window of T positions into a single shared sparse latent,
    then decodes back to T positions using per-position decoder weights.

    Architecture:
        z = TopK(sum_t W_enc[t] @ x_t + b_enc, k_total)
        x_hat_t = W_dec[t] @ z + b_dec[t]

    The encoder sums per-position projections into a single latent vector.
    The shared latent captures structure spanning the whole window.

    Sparsity can be specified either as:
    - ``k_per_pos``: total active features = ``k_per_pos * T`` (legacy behavior)
    - ``k_total``: fixed total active features across the whole shared latent
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k_per_pos: int | None = None,
        k_total: int | None = None,
        use_relu: bool = True,
    ):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.T = T
        self.use_relu = use_relu
        if k_total is not None and k_total <= 0:
            raise ValueError(f"k_total must be positive, got {k_total}")
        if k_per_pos is not None and k_per_pos <= 0:
            raise ValueError(f"k_per_pos must be positive, got {k_per_pos}")
        if k_total is None and k_per_pos is None:
            raise ValueError("Specify either k_per_pos or k_total")

        self.k_per_pos = k_per_pos
        self.k_total = k_total if k_total is not None else int(k_per_pos * T)
        if self.k_total > d_sae:
            raise ValueError(f"k_total={self.k_total} exceeds d_sae={d_sae}")

        # Per-position encoder: (T, d_in, d_sae)
        self.W_enc = nn.Parameter(torch.empty(T, d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Per-position decoder: (T, d_sae, d_in)
        self.W_dec = nn.Parameter(torch.empty(T, d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(T, d_in))

        for t in range(T):
            nn.init.kaiming_uniform_(self.W_enc[t], a=math.sqrt(5))
            with torch.no_grad():
                self.W_dec.data[t] = self.W_enc.data[t].T
        self._normalize_decoder()

    def _normalize_decoder(self) -> None:
        # Joint norm across (T, d_in) for each latent
        norms = self.W_dec.data.pow(2).sum(dim=(0, 2), keepdim=True).sqrt().clamp(min=1e-8)
        self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d) -> z: (B, d_sae)"""
        pre = torch.einsum("btd,tdm->bm", x, self.W_enc) + self.b_enc
        if self.use_relu:
            pre = torch.relu(pre)
        topk_vals, topk_idx = pre.topk(self.k_total, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_sae) -> x_hat: (B, T, d)"""
        return torch.einsum("bm,tmd->btd", z, self.W_dec) + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (x_hat, z). x: (B, T, d), x_hat: (B, T, d), z: (B, d_sae)."""
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_hat, z = self.forward(x)
        recon_loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        return recon_loss, {"recon_loss": recon_loss.item()}

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


class MatryoshkaTemporalCrosscoder(TemporalCrosscoder):
    """Temporal crosscoder with Matryoshka-style nested prefix losses.

    The shared sparse latent is unchanged from TemporalCrosscoder. The only
    change is the training objective: we add reconstruction losses using only
    the first w latent channels for widths in ``matryoshka_widths``.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        T: int,
        k_per_pos: int | None = None,
        k_total: int | None = None,
        matryoshka_widths: list[int] | None = None,
        inner_weight: float = 1.0,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in, d_sae=d_sae, T=T,
            k_per_pos=k_per_pos, k_total=k_total, use_relu=use_relu,
        )
        self.inner_weight = inner_weight
        if matryoshka_widths is None:
            widths = []
            w = 4
            while w < d_sae:
                widths.append(w)
                w *= 2
            widths.append(d_sae)
            self.matryoshka_widths = widths
        else:
            self.matryoshka_widths = sorted(matryoshka_widths)
            if self.matryoshka_widths[-1] != d_sae:
                self.matryoshka_widths.append(d_sae)

    def compute_loss(self, x: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        x_hat, z = self.forward(x)
        full_recon = (x - x_hat).pow(2).sum(dim=-1).mean()

        inner_losses = []
        per_width = {self.matryoshka_widths[-1]: full_recon.item()}
        for w in self.matryoshka_widths[:-1]:
            x_hat_inner = torch.einsum("bm,tmd->btd", z[:, :w], self.W_dec[:, :w, :]) + self.b_dec
            inner_recon = (x - x_hat_inner).pow(2).sum(dim=-1).mean()
            inner_losses.append(inner_recon)
            per_width[w] = inner_recon.item()

        if inner_losses:
            inner_mean = sum(inner_losses) / len(inner_losses)
            total = (full_recon + self.inner_weight * inner_mean) / (1.0 + self.inner_weight)
        else:
            total = full_recon

        return total, {
            "total_loss": total.item(),
            "full_recon": full_recon.item(),
            "per_width": per_width,
        }


class MultiLayerCrosscoder(TemporalCrosscoder):
    """Cross-layer sparse crosscoder (Anthropic crosscoder-style).

    Reads ``L`` residual-stream positions — typically
    ``blocks.<layer>.hook_resid_post`` for several layers — at a single
    time step, encodes them to one shared sparse latent, and decodes
    per-layer reconstructions.

    The math is identical to ``TemporalCrosscoder`` with ``T`` replaced
    by ``L``; this class exists to make the semantics explicit (no
    time-windowing is applied — every (time-step, L-layer-stack) is an
    independent sample).

    Expected input shape is ``(B, L, d_in)`` where ``L`` is the number
    of layer hooks being read.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        L: int,
        k_per_layer: int | None = None,
        k_total: int | None = None,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            T=L,
            k_per_pos=k_per_layer,
            k_total=k_total,
            use_relu=use_relu,
        )
        self.L = L
        self.k_per_layer = k_per_layer


class MatryoshkaMultiLayerCrosscoder(MatryoshkaTemporalCrosscoder):
    """Matryoshka-style nested-width multi-layer crosscoder.

    Same relationship to :class:`MultiLayerCrosscoder` as
    :class:`MatryoshkaTemporalCrosscoder` has to :class:`TemporalCrosscoder`.
    """

    def __init__(
        self,
        d_in: int,
        d_sae: int,
        L: int,
        k_per_layer: int | None = None,
        k_total: int | None = None,
        matryoshka_widths: list[int] | None = None,
        inner_weight: float = 1.0,
        use_relu: bool = True,
    ):
        super().__init__(
            d_in=d_in,
            d_sae=d_sae,
            T=L,
            k_per_pos=k_per_layer,
            k_total=k_total,
            matryoshka_widths=matryoshka_widths,
            inner_weight=inner_weight,
            use_relu=use_relu,
        )
        self.L = L
        self.k_per_layer = k_per_layer
