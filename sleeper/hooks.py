"""Forward-pass hooks for steering / ablation.

Three hooks, all sharing the same prompt-positions-only convention:

    additive_steer_hook       resid[:, :P] += alpha * delta            (any vector at any resid hook)
    sae_feature_delta         alpha * (decode(z_abl) - decode(z))      (one SAE feature, any resid hook)
    ov_only_steer_hook        alpha * (delta @ W_V[h]) at hook_v       (Q,K untouched ⇒ A frozen)

`compute_sae_delta` builds the (B, P, d) delta tensor for `additive_steer_hook`;
`compute_meandiff_delta` builds the same tensor from a single direction vector.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from transformer_lens import HookedTransformer

from sleeper.sae import TopKSAE


# ---------------------------------------------------------------------------
# delta builders
# ---------------------------------------------------------------------------


@torch.no_grad()
def compute_sae_delta(
    model: HookedTransformer,
    sae: TopKSAE,
    layer_hook: str,
    feature_idx: int,
    tokens: torch.Tensor,           # (B, P)
    prompt_mask: torch.Tensor,      # (B, P) bool
) -> torch.Tensor:
    """Per-token SAE-reconstruction delta for zeroing one feature.

    Returns (B, P, d_model) on the model's device, dtype-matched to the resid
    stream. Outside-prompt positions are zero.
    """
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    prompt_mask = prompt_mask.to(device)
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=lambda n: n == layer_hook
    )
    acts = cache[layer_hook]                       # (B, P, d)
    B, P, D = acts.shape
    flat = acts.reshape(B * P, D).to(torch.float32)
    z = sae.encode(flat)
    x_hat_orig = sae.decode(z)
    z_abl = z.clone()
    z_abl[:, feature_idx] = 0.0
    x_hat_abl = sae.decode(z_abl)
    delta = (x_hat_abl - x_hat_orig).reshape(B, P, D).to(acts.dtype)
    return delta * prompt_mask.unsqueeze(-1)


def compute_meandiff_delta(
    v: torch.Tensor,                # (d_model,) the steering direction
    prompt_mask: torch.Tensor,      # (B, P) bool
    sign: float = -1.0,             # subtract by default (cancel the dep direction)
) -> torch.Tensor:
    """Broadcast a single direction vector into a (B, P, d) delta masked to prompt."""
    B, P = prompt_mask.shape
    d = v.shape[0]
    delta = (sign * v).view(1, 1, d).expand(B, P, d).contiguous()
    return delta * prompt_mask.to(delta.dtype).unsqueeze(-1)


# ---------------------------------------------------------------------------
# hook factories
# ---------------------------------------------------------------------------


def additive_steer_hook(
    delta: torch.Tensor,            # (B, P, d_model)
    alpha: float,
    layer_hook: str,
) -> list[tuple[str, Callable]]:
    """Add ``alpha * delta`` to ``layer_hook`` on the first P positions only."""
    P = delta.shape[1]

    def _hook(resid, hook):
        resid[:, :P, :] = resid[:, :P, :] + alpha * delta.to(resid.dtype).to(resid.device)
        return resid

    return [(layer_hook, _hook)]


def ov_only_steer_hook(
    delta: torch.Tensor,            # (B, P, d_model) ln1-space delta
    alpha: float,
    W_V: torch.Tensor,              # (n_heads, d_model, d_head) at the target block
    block: int = 0,
) -> list[tuple[str, Callable]]:
    """OV-only intervention: project delta through W_V and patch hook_v.

    Q and K are untouched, so the attention pattern A produced by softmax is
    exactly the un-perturbed pattern (frozen by leaving its inputs alone).
    Only the values change → only the OV circuit carries the steer.
    """
    v_delta = torch.einsum("bpd,hdk->bphk", delta.float(), W_V.float())
    P = delta.shape[1]
    hook_name = f"blocks.{block}.attn.hook_v"

    def _hook(v, hook):
        v[:, :P, :, :] = v[:, :P, :, :] + alpha * v_delta.to(v.dtype).to(v.device)
        return v

    return [(hook_name, _hook)]


# ---------------------------------------------------------------------------
# generation with hooks
# ---------------------------------------------------------------------------


@torch.no_grad()
def greedy_generate_with_hooks(
    model: HookedTransformer,
    prompts: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int = 16,
) -> torch.Tensor:
    """Greedy decode; hooks fire each step but only patch the original P positions."""
    device = next(model.parameters()).device
    tokens = prompts.to(device)
    out: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type="logits")
        nxt = logits[:, -1, :].argmax(dim=-1)
        out.append(nxt.unsqueeze(1))
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
    return torch.cat(out, dim=1)


@torch.no_grad()
def sample_generate_with_hooks(
    model: HookedTransformer,
    prompts: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int = 16,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 0,
) -> torch.Tensor:
    """Nucleus sampling with the same hook semantics as greedy."""
    device = next(model.parameters()).device
    tokens = prompts.to(device)
    gen = torch.Generator(device=device).manual_seed(seed)
    out: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, return_type="logits")
        last = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(last, dim=-1)
        sp, si = probs.sort(descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sp = sp.masked_fill(mask, 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(sp, num_samples=1, generator=gen)
        nxt = si.gather(-1, pick).squeeze(-1)
        out.append(nxt.unsqueeze(1))
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
    return torch.cat(out, dim=1)
