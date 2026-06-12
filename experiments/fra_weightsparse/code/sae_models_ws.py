"""TopKSAE for the fra_weightsparse experiment (verbatim from multitrigger_sleeper/cloud/sae_models.py)."""

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


