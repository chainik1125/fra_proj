"""Toy transformer + TopK SAE for the FRA-vs-SAE concept-cut experiment.

Transformer recipe follows sae_day experiment_03 (3L, d_model=64, 2 heads,
d_mlp=256, LN, GELU, lr 1e-3); TopKSAE is vendored verbatim from
sae_day/src/sae_day/sae.py. Default geometry here is the "clean" variant
(seq_len == n_ctx == context_len, so all position embeddings are trained);
pass seq_len=600, context_len=128 to reproduce experiment_03 exactly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """Standard TopK sparse autoencoder (vendored from sae_day)."""

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
        z = self.encode(x)
        return self.decode(z), z

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()


def build_transformer(
    d_vocab: int,
    n_ctx: int,
    d_model: int = 64,
    n_heads: int = 2,
    n_layers: int = 3,
    d_mlp: int = 256,
    seed: int = 42,
    device: str = "cpu",
):
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    cfg = HookedTransformerConfig(
        n_layers=n_layers,
        d_model=d_model,
        n_ctx=n_ctx,
        d_head=d_model // n_heads,
        n_heads=n_heads,
        d_mlp=d_mlp,
        d_vocab=d_vocab,
        d_vocab_out=d_vocab,
        act_fn="gelu",
        normalization_type="LN",
        attention_dir="causal",
        default_prepend_bos=False,
        device=device,
        seed=seed,
    )
    return HookedTransformer(cfg).to(device)


def train_transformer(
    model: Any,
    tokens: torch.Tensor,
    n_steps: int,
    batch_size: int = 128,
    context_len: int | None = None,
    lr: float = 1e-3,
    seed: int = 42,
    print_every: int = 250,
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    device = next(model.parameters()).device
    n_sequences, seq_len = tokens.shape
    if context_len is None:
        context_len = seq_len

    losses = []
    model.train()
    for step in range(n_steps):
        seq_idx = torch.randint(n_sequences, (batch_size,), generator=gen)
        start_idx = torch.randint(seq_len - context_len + 1, (batch_size,), generator=gen)
        batch = torch.stack(
            [tokens[seq_idx[i], start_idx[i] : start_idx[i] + context_len] for i in range(batch_size)]
        ).to(device)
        loss = model(batch, return_type="loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        if print_every > 0 and (step + 1) % print_every == 0:
            print(f"  transformer step {step + 1}/{n_steps}  loss={loss.item():.4f}", flush=True)
    return losses


@torch.no_grad()
def extract_activations(
    model: Any,
    tokens: torch.Tensor,
    hook_name: str,
    chunk_size: int = 32,
) -> torch.Tensor:
    device = next(model.parameters()).device
    outputs = []
    model.eval()
    for i in range(0, tokens.shape[0], chunk_size):
        batch = tokens[i : i + chunk_size].to(device)
        _, cache = model.run_with_cache(
            batch,
            return_type=None,
            names_filter=lambda name: name == hook_name,
            return_cache_object=False,
        )
        outputs.append(cache[hook_name].detach().cpu())
    return torch.cat(outputs, dim=0)


def train_topk_sae(
    sae: TopKSAE,
    flat_data: torch.Tensor,
    n_steps: int = 2000,
    batch_size: int = 256,
    lr: float = 3e-4,
    seed: int = 42,
) -> list[float]:
    gen = torch.Generator().manual_seed(seed)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)
    device = next(sae.parameters()).device
    n_total = flat_data.shape[0]
    losses = []
    for step in range(n_steps):
        idx = torch.randint(n_total, (batch_size,), generator=gen)
        x = flat_data[idx].to(device)
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).sum(-1).mean()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if (step + 1) % 100 == 0:
            sae.normalize_decoder()
        losses.append(loss.item())
    return losses


@torch.no_grad()
def encode_all(sae: TopKSAE, flat_data: torch.Tensor, chunk_size: int = 8192) -> torch.Tensor:
    device = next(sae.parameters()).device
    chunks = []
    for i in range(0, flat_data.shape[0], chunk_size):
        _, z = sae(flat_data[i : i + chunk_size].to(device))
        chunks.append(z.cpu())
    return torch.cat(chunks, dim=0)


# ── Feature identification ──


def per_latent_r2(z: np.ndarray, y: np.ndarray) -> np.ndarray:
    """(d_sae, d_target) in-sample R² of 1-D OLS from each latent to each target."""
    zc = z - z.mean(axis=0)
    yc = y - y.mean(axis=0)
    z_var = (zc ** 2).mean(axis=0)
    y_var = (yc ** 2).mean(axis=0)
    cov = (zc.T @ yc) / z.shape[0]
    corr2 = cov ** 2 / np.clip(z_var[:, None] * y_var[None, :], 1e-12, None)
    return corr2  # for 1-D OLS, R² == squared correlation


def identify_features(
    z_flat: torch.Tensor,          # (N*L, d_sae)
    posterior_omega: torch.Tensor,  # (N, L, K) -> flattened target
    token_block: torch.Tensor,      # (N*L,) block id of the CURRENT token
    n_positions_skip: int = 8,
    seq_len: int | None = None,
) -> dict[str, Any]:
    """Rank latents against (a) the accumulated concept (posterior omega) and
    (b) the instantaneous token-block identity. Returns per-target R² tables
    and the argmax assignments; selection into cut sets happens in the runner."""
    z = z_flat.numpy().astype(np.float64)
    K = posterior_omega.shape[-1]
    y_omega = posterior_omega.reshape(-1, K).numpy().astype(np.float64)
    y_block = np.eye(K)[token_block.numpy()]

    keep = np.ones(z.shape[0], dtype=bool)
    if seq_len is not None and n_positions_skip > 0:
        pos = np.tile(np.arange(seq_len), z.shape[0] // seq_len)
        keep = pos >= n_positions_skip

    r2_omega = per_latent_r2(z[keep], y_omega[keep])  # (d_sae, K)
    r2_block = per_latent_r2(z[keep], y_block[keep])  # (d_sae, K)

    return {
        "r2_omega": r2_omega,
        "r2_block": r2_block,
        "best_omega_r2": r2_omega.max(axis=1),
        "best_omega_comp": r2_omega.argmax(axis=1),
        "best_block_r2": r2_block.max(axis=1),
        "best_block_comp": r2_block.argmax(axis=1),
        "active_frac": (z > 0).mean(axis=0),
    }
