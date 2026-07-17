"""
A minimal decoder-only transformer in pure PyTorch.

Self-contained (no transformer_lens / simplexity) so the identical code runs on
Apple-Silicon MPS and on CUDA.  Pre-LN, learned positional embeddings, causal
attention.  Exposes a `run_with_resid` method that returns the residual stream
*after* each block (resid_post), which is what the probes/ablations use.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 3
    n_ctx: int = 280
    d_model: int = 128
    n_heads: int = 4
    n_layers: int = 3
    d_mlp: int = 512
    dropout: float = 0.0
    init_std: float = 0.02
    act_fn: str = "gelu"


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_heads == 0
        self.n_heads = cfg.n_heads
        self.d_head = cfg.d_model // cfg.n_heads
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model)
        self.register_buffer(
            "mask", torch.tril(torch.ones(cfg.n_ctx, cfg.n_ctx)).view(1, 1, cfg.n_ctx, cfg.n_ctx)
        )

    def forward(self, x, extra_mask=None):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float("-inf"))
        if extra_mask is not None:  # (T,T) bool: True = blocked (query,key) pair
            att = att.masked_fill(extra_mask[None, None, :T, :T], float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.w_gate = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.w_up = nn.Linear(cfg.d_model, cfg.d_mlp)
        self.w_down = nn.Linear(cfg.d_mlp, cfg.d_model)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.d_model)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.d_model)
        act_fn = cfg.act_fn.lower()
        if act_fn == "gelu":
            activation = nn.GELU()
        elif act_fn == "relu":
            activation = nn.ReLU()
        elif act_fn == "swiglu":
            self.mlp = SwiGLU(cfg)
            return
        else:
            raise ValueError(f"unsupported act_fn={cfg.act_fn!r}; expected gelu, relu, or swiglu")
        self.mlp = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_mlp),
            activation,
            nn.Linear(cfg.d_mlp, cfg.d_model),
        )

    def forward(self, x, extra_mask=None):
        x = x + self.attn(self.ln1(x), extra_mask=extra_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.d_model)
        self.pos_emb = nn.Embedding(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        self.ln_f = nn.LayerNorm(cfg.d_model)
        self.head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.apply(self._init)

    def _init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=self.cfg.init_std)

    def _embed(self, idx):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        return self.tok_emb(idx) + self.pos_emb(pos)[None]

    def forward(self, idx):
        x = self._embed(idx)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

    def forward_attn_knockout(self, idx, blocked_mask):
        """Forward pass with certain (query,key) attention pairs blocked in every
        layer.  blocked_mask: (T,T) bool tensor, True = that query may NOT attend to
        that key.  Used to sever cross-rollout information flow."""
        x = self._embed(idx)
        for blk in self.blocks:
            x = blk(x, extra_mask=blocked_mask)
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def run_with_resid(self, idx):
        """Return logits and list of resid_post tensors (one per block), each (B,T,d)."""
        x = self._embed(idx)
        resids = []
        for blk in self.blocks:
            x = blk(x)
            resids.append(x.detach())
        logits = self.head(self.ln_f(x))
        return logits, resids

    def forward_with_ablation(self, idx, layer: int, direction: torch.Tensor,
                              target_value: float | None = None, positions=None):
        """Forward pass that, after block `layer`, removes the component of the
        residual stream along unit vector `direction` (optionally re-adding a fixed
        scalar projection `target_value`, i.e. mean-ablation).

        positions: None -> apply at all positions; else an iterable of column
        indices where the ablation is applied (others left untouched)."""
        x = self._embed(idx)
        d = direction / (direction.norm() + 1e-8)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == layer:
                proj = (x @ d)[..., None]  # (B,T,1)
                edit = -proj * d
                if target_value is not None:
                    edit = edit + target_value * d
                if positions is None:
                    x = x + edit
                else:
                    mask = torch.zeros(x.shape[1], device=x.device)
                    mask[torch.as_tensor(positions, device=x.device)] = 1.0
                    x = x + edit * mask[None, :, None]
        return self.head(self.ln_f(x))

    def forward_with_steer(self, idx, layer: int, direction: torch.Tensor, alpha: float,
                           positions=None):
        """Forward pass that ADDS alpha * unit(direction) to the residual stream after
        block `layer` (activation steering). positions: None=all, else column indices."""
        x = self._embed(idx)
        d = direction / (direction.norm() + 1e-8)
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i == layer:
                if positions is None:
                    x = x + alpha * d
                else:
                    mask = torch.zeros(x.shape[1], device=x.device)
                    mask[torch.as_tensor(positions, device=x.device)] = 1.0
                    x = x + alpha * d * mask[None, :, None]
        return self.head(self.ln_f(x))
