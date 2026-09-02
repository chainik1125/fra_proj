"""TopK + BatchTopK sparse autoencoders + minimal training/save/load.

BatchTopK is sae-lens-only at training time (see :mod:`sleeper.sae_saelens`);
``BatchTopKSAE`` in this module is the in-process inference class used to
hold the converted weights + scalar threshold after training.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import torch
import torch.nn as nn


class TopKSAE(nn.Module):
    """z = TopK(ReLU(W_enc @ (x - b_dec) + b_enc), k);  x_hat = W_dec @ z + b_dec."""

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

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

    @torch.no_grad()
    def normalize_decoder(self) -> None:
        self._normalize_decoder()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        topk_vals, topk_idx = pre.topk(self.k, dim=-1)
        z = torch.zeros_like(pre)
        z.scatter_(-1, topk_idx, topk_vals)
        return z

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE in inference form (JumpReLU with a learned scalar threshold).

    Training (in sae-lens) picks a single global top-K across the whole batch
    of token activations; the SAE simultaneously learns ``topk_threshold``, an
    EMA of the smallest positive pre-activation. At inference time, that
    threshold gates pre-activations per-token: ``z = pre * (pre > threshold)``.

    Same params as :class:`TopKSAE` plus a scalar ``threshold`` buffer. The
    ``k`` field is stored for metadata only (avg per-token sparsity target);
    the actual sparsity at inference depends on the threshold.
    """

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.empty(d_in, d_sae))
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.empty(d_sae, d_in))
        self.b_dec = nn.Parameter(torch.zeros(d_in))
        # Scalar threshold; registered as buffer so it moves with .to(device)
        # and rides along in state_dict, but isn't picked up by optimisers.
        self.register_buffer("threshold", torch.zeros((), dtype=torch.float32))

        nn.init.kaiming_uniform_(self.W_enc, a=math.sqrt(5))
        with torch.no_grad():
            self.W_dec.copy_(self.W_enc.T)
            norms = self.W_dec.norm(dim=1, keepdim=True).clamp(min=1e-8)
            self.W_dec.data.div_(norms)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = torch.relu((x - self.b_dec) @ self.W_enc + self.b_enc)
        return pre * (pre > self.threshold)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        return self.decode(z), z


# Dispatch table for load(); keep alongside the classes so additions are local.
_SAE_TYPES: dict[str, type[nn.Module]] = {
    "topk": TopKSAE,
    "batchtopk": BatchTopKSAE,
}


@torch.no_grad()
def encode_all(sae: TopKSAE, acts: torch.Tensor, chunk: int = 256) -> torch.Tensor:
    """Encode (N, T, d_in) activations through the SAE in chunks; result on CPU."""
    N, T, D = acts.shape
    device = next(sae.parameters()).device
    flat = acts.reshape(N * T, D)
    out = torch.empty(N * T, sae.d_sae, dtype=torch.float32)
    for s in range(0, N * T, chunk):
        z = sae.encode(flat[s : s + chunk].to(device=device, dtype=torch.float32))
        out[s : s + chunk] = z.detach().cpu()
    return out.reshape(N, T, sae.d_sae)


def save(sae: nn.Module, path: Path, layer_hook: str,
         save_dtype: torch.dtype = torch.bfloat16, **extra) -> None:
    """Save SAE weights at ``save_dtype`` (default bf16).

    Training is fp32 for numerical stability, but the saved weights are
    consumed in bf16 contexts anyway (LM activations are bf16; SAE encoder
    dot-products are well-conditioned). Storing bf16 halves on-disk size
    from ~1.1 GB → ~535 MB per d_in=4096, d_sae=32768 SAE. ``load`` upcasts
    back to fp32 via ``load_state_dict`` into the fp32 SAE module.

    Works for both :class:`TopKSAE` and :class:`BatchTopKSAE`. The SAE class
    is recorded as ``config["sae_type"]`` for round-trip dispatch in
    :func:`load`. Missing field defaults to ``"topk"`` so older ``.pt`` files
    load unchanged.
    """
    sae_type = next((t for t, c in _SAE_TYPES.items() if isinstance(sae, c)),
                    None)
    if sae_type is None:
        raise TypeError(f"save() expects one of {list(_SAE_TYPES)}, got "
                        f"{type(sae).__name__}")
    payload = {
        "state_dict": {k: v.detach().to(save_dtype).cpu()
                       for k, v in sae.state_dict().items()},
        "config": {
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "k": sae.k,
            "sae_type": sae_type,
            "layer_hook": layer_hook,
            "save_dtype": str(save_dtype),
            **extra,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load(path: Path, device: str = "cpu") -> tuple[nn.Module, dict]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    cfg = payload["config"]
    # support legacy `k_total` field
    k = cfg.get("k", cfg.get("k_total"))
    sae_type = cfg.get("sae_type", "topk")
    try:
        SAECls = _SAE_TYPES[sae_type]
    except KeyError:
        raise ValueError(f"unknown sae_type {sae_type!r}; known: {list(_SAE_TYPES)}")
    sae = SAECls(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=k)
    sae.load_state_dict(payload["state_dict"])
    sae.to(device).eval()
    for p in sae.parameters():
        p.requires_grad_(False)
    return sae, cfg


def train(
    acts: torch.Tensor,           # (N, T, d) any dtype, CPU or GPU
    d_sae: int,
    k: int,
    *,
    n_steps: int = 4000,
    batch_size: int = 4096,
    lr: float = 5e-4,
    normalize_every: int = 100,
    print_every: int = 200,
    seed: int = 0,
    device: str = "cuda",
    mask: torch.Tensor | None = None,   # (N, T) bool; True = real token, False = pad
) -> tuple[TopKSAE, list[float]]:
    """Train a TopK SAE on (N, T, d) acts. Returns (sae, loss_trajectory).

    Sampling matches Dmitry's train_crosscoders.py: CPU `Generator`, paired
    (seq_idx, pos_idx) randints, then index acts[seq_idx, pos_idx].

    Pass ``mask`` (N, T) when acts came from a left-padded LM forward so the
    sampler draws only from real-token positions. Without the mask, ~17% of
    samples land on pad activations for typical TS prompt-only shapes.
    """
    torch.manual_seed(seed)
    N, T, D = acts.shape
    if device != "cpu":
        acts = acts.to(device)

    # Pre-build a flat list of (seq, pos) for real tokens; sample uniformly
    # from it instead of the full (N × T) grid.
    valid: torch.Tensor = torch.empty(0, dtype=torch.long, device=acts.device)
    M = 0
    if mask is not None:
        assert mask.shape == (N, T), \
            f"mask shape {tuple(mask.shape)} must match acts (N, T)=({N}, {T})"
        valid = mask.nonzero(as_tuple=False).to(acts.device)   # (M, 2)
        M = valid.shape[0]
        if M == 0:
            raise ValueError("mask has no True entries — nothing to sample")

    sae = TopKSAE(d_in=D, d_sae=d_sae, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(seed)  # CPU generator (matches Dmitry)

    t0 = time.time()
    for step in range(n_steps):
        if mask is None:
            seq_idx = torch.randint(0, N, (batch_size,), generator=gen).to(acts.device)
            pos_idx = torch.randint(0, T, (batch_size,), generator=gen).to(acts.device)
        else:
            flat_idx = torch.randint(0, M, (batch_size,), generator=gen).to(acts.device)
            sample = valid[flat_idx]                  # (B, 2)
            seq_idx, pos_idx = sample[:, 0], sample[:, 1]
        x = acts[seq_idx, pos_idx].to(torch.float32)
        x_hat, _ = sae(x)
        loss = (x - x_hat).pow(2).sum(dim=-1).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        losses.append(loss.item())
        if (step + 1) % normalize_every == 0:
            sae.normalize_decoder()
        if (step + 1) % print_every == 0:
            rate = (step + 1) / (time.time() - t0)
            print(f"[sae-train] step {step+1}/{n_steps} ({rate:.1f} it/s) loss={loss.item():.4f}")
    print(f"[sae-train] done in {time.time() - t0:.1f}s  final_loss={losses[-1]:.4f}")
    return sae, losses
