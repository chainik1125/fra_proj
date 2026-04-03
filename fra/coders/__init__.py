"""Coder wrappers — SAE and crosscoder interfaces for FRA."""

from abc import ABC, abstractmethod
from pathlib import Path

import torch


class Coder(ABC):
    """Abstract interface for SAE and crosscoder wrappers used by FRA.

    Every coder must expose:
      - ``encode(x)`` → sparse feature activations ``[..., d_sae]``
      - ``decode(f)`` → reconstructed activations ``[..., d_model]``
      - ``W_dec``     — decoder weight matrix ``[d_sae, d_model]``
      - ``b_dec``     — decoder bias ``[d_model]``
      - ``d_sae``     — number of dictionary features
    """

    W_dec: torch.Tensor
    b_dec: torch.Tensor
    d_sae: int

    @abstractmethod
    def encode(self, x: torch.Tensor, /) -> torch.Tensor: ...

    @abstractmethod
    def decode(self, features: torch.Tensor, /) -> torch.Tensor: ...


def load_sae(sae_type: str, layer: int, device: str,
             release: str = "", sae_id: str = ""):
    """Load SAE wrapper of the specified type."""
    if sae_type == "hub":
        from fra.coders.sae_lens import SAELensAttentionSAE
        return SAELensAttentionSAE("gpt2-small-hook-z-kk",
                                   f"blocks.{layer}.hook_z", device=device)
    elif sae_type == "gemma":
        from fra.coders.sae_lens import GemmaScopeSAE
        rel = release or "gemma-scope-2b-pt-res"
        # layer is the FRA/attention layer; Gemma-Scope SAEs are trained on
        # resid_post[N-1] which equals resid_pre[N], so subtract 1.
        sid = sae_id or f"layer_{layer - 1}/width_16k/average_l0_82"
        return GemmaScopeSAE(rel, sid, device=device)
    else:
        from fra.coders.sae_lens import LocalLn1SAE
        ckpt = str(Path(__file__).parent.parent / "checkpoints" / "q9sczrvl" / "50003968")
        return LocalLn1SAE(ckpt, layer=layer, device=device)
