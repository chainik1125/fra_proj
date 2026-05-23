"""Steering control spaces for the Fisher POC.

Two control spaces share a `forward_logits(theta) -> [B,T,V]` interface:

  FRAOVControlSpace: K features, intervention at blocks.0.attn.hook_v as
                     Σ θ_i · (Δ_i · W_V₀)   where Δ_i is the SAE feature
                     contribution at LN1.  Matches `build_ov_hooks` in
                     `run_fidelity_experiment.py`.

  ResidMidControlSpace: K features, intervention at blocks.0.hook_resid_mid
                        as Σ θ_i · Δ_i, where Δ_i is computed at hook_resid_mid.

Per-feature deltas are pre-computed once and frozen, so each forward pass
under a different θ is one HookedTransformer.run_with_hooks call with a tiny
linear combination — that makes the FD inner loop K + O(1) instead of
K·(SAE-encode) forward passes.
"""

from __future__ import annotations

import sys
from collections.abc import Callable
from pathlib import Path

import torch
from transformer_lens import HookedTransformer

EXP_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(EXP_ROOT))

from sleeper_utils import compute_sae_delta  # noqa: E402


LN1_HOOK = "blocks.0.ln1.hook_normalized"
RESID_MID_HOOK = "blocks.0.hook_resid_mid"
HOOK_V = "blocks.0.attn.hook_v"


class FRAOVControlSpace:
    """Steer values at blocks.0.attn.hook_v through SAE feature → W_V₀.

    The per-feature direction in V-space is
        v_delta_i = einsum("btd,hdk->bthk", delta_i, W_V₀)
    where delta_i is the SAE ablation delta of feature i at LN1
    (see `compute_sae_delta`). This matches `build_ov_hooks` in
    `run_fidelity_experiment.py`, but parameterized over a vector θ instead
    of a scalar α applied to a fixed feature set.

    Note on sign convention: `compute_sae_delta` returns (x_hat_abl - x_hat_orig),
    i.e. the *ablation* direction.  θ_i = 0 is unsteered.  θ_i = +1 reproduces
    the existing "fully-ablate feature i" alpha=1 path.
    """

    def __init__(
        self,
        model: HookedTransformer,
        sae,
        feature_ids: list[int],
        tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
    ):
        self.model = model
        self.feature_ids = list(map(int, feature_ids))
        self.K = len(self.feature_ids)
        self.tokens = tokens
        self.device = next(model.parameters()).device

        with torch.no_grad():
            W_V0 = model.W_V[0].detach().to(self.device).float()  # [H, d_model, d_head]
            per_feat_v: list[torch.Tensor] = []
            for fid in self.feature_ids:
                delta = compute_sae_delta(model, sae, LN1_HOOK, fid, tokens, prompt_mask)
                v_delta = torch.einsum("btd,hdk->bthk", delta.float(), W_V0).to(model.W_V.dtype)
                per_feat_v.append(v_delta)
            # [K, B, T, H, d_head]
            self.per_feat_v = torch.stack(per_feat_v, dim=0)
        self.seq_len = self.per_feat_v.shape[2]

    def forward_logits(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.to(self.device).float()
        # v_delta_total: [B, T, H, d_head]
        v_delta_total = torch.einsum("k,kbthd->bthd", theta, self.per_feat_v.float())
        v_delta_total = v_delta_total.to(self.model.W_V.dtype)
        sl = self.seq_len

        def _v_hook(v, hook):
            v[:, :sl, :, :] = v[:, :sl, :, :] + v_delta_total
            return v

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                self.tokens,
                fwd_hooks=[(HOOK_V, _v_hook)],
                return_type="logits",
            )
        return logits


class ResidMidControlSpace:
    """Steer at blocks.0.hook_resid_mid by Σ θ_i Δ_i.

    Δ_i is computed by `compute_sae_delta` against the SAE wired to
    blocks.0.hook_resid_mid.  This is the conventional residual-stream
    additive control basis (sec. 2.3 of the proposal).
    """

    def __init__(
        self,
        model: HookedTransformer,
        sae,
        feature_ids: list[int],
        tokens: torch.Tensor,
        prompt_mask: torch.Tensor,
        hook_name: str = RESID_MID_HOOK,
    ):
        self.model = model
        self.feature_ids = list(map(int, feature_ids))
        self.K = len(self.feature_ids)
        self.tokens = tokens
        self.hook_name = hook_name
        self.device = next(model.parameters()).device

        with torch.no_grad():
            per_feat: list[torch.Tensor] = []
            for fid in self.feature_ids:
                delta = compute_sae_delta(model, sae, hook_name, fid, tokens, prompt_mask)
                per_feat.append(delta.float())
            # [K, B, T, d_model]
            self.per_feat = torch.stack(per_feat, dim=0)
        self.seq_len = self.per_feat.shape[2]

    def forward_logits(self, theta: torch.Tensor) -> torch.Tensor:
        theta = theta.to(self.device).float()
        # delta_total: [B, T, d_model]
        delta_total = torch.einsum("k,kbtd->btd", theta, self.per_feat)
        sl = self.seq_len

        def _resid_hook(resid, hook):
            if resid.shape[1] < sl:
                return resid
            resid[:, :sl, :] = resid[:, :sl, :] + delta_total.to(resid.dtype)
            return resid

        with torch.no_grad():
            logits = self.model.run_with_hooks(
                self.tokens,
                fwd_hooks=[(self.hook_name, _resid_hook)],
                return_type="logits",
            )
        return logits


def unsteered_logits(model: HookedTransformer, tokens: torch.Tensor) -> torch.Tensor:
    """Logits with no hooks (baseline)."""
    with torch.no_grad():
        return model(tokens.to(next(model.parameters()).device), return_type="logits")
