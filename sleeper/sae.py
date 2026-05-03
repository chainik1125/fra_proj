"""TopK sparse autoencoder + minimal training/save/load."""

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


def save(sae: TopKSAE, path: Path, layer_hook: str, **extra) -> None:
    payload = {
        "state_dict": {k: v.detach().cpu() for k, v in sae.state_dict().items()},
        "config": {
            "d_in": sae.d_in,
            "d_sae": sae.d_sae,
            "k": sae.k,
            "layer_hook": layer_hook,
            **extra,
        },
    }
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load(path: Path, device: str = "cpu") -> tuple[TopKSAE, dict]:
    payload = torch.load(path, weights_only=False, map_location="cpu")
    cfg = payload["config"]
    # support legacy `k_total` field
    k = cfg.get("k", cfg.get("k_total"))
    sae = TopKSAE(d_in=cfg["d_in"], d_sae=cfg["d_sae"], k=k)
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
) -> tuple[TopKSAE, list[float]]:
    """Train a TopK SAE on (N, T, d) acts. Returns (sae, loss_trajectory).

    Sampling matches Dmitry's train_crosscoders.py: CPU `Generator`, paired
    (seq_idx, pos_idx) randints, then index acts[seq_idx, pos_idx].
    """
    torch.manual_seed(seed)
    N, T, D = acts.shape
    if device != "cpu":
        acts = acts.to(device)

    sae = TopKSAE(d_in=D, d_sae=d_sae, k=k).to(device)
    opt = torch.optim.Adam(sae.parameters(), lr=lr)
    losses: list[float] = []
    gen = torch.Generator().manual_seed(seed)  # CPU generator (matches Dmitry)

    t0 = time.time()
    for step in range(n_steps):
        seq_idx = torch.randint(0, N, (batch_size,), generator=gen).to(acts.device)
        pos_idx = torch.randint(0, T, (batch_size,), generator=gen).to(acts.device)
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
