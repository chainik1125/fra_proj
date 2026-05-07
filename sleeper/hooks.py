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
    prompt_mask: torch.Tensor,      # (B, P) bool; delta zeroed outside these positions
    attention_mask: torch.Tensor | None = None,  # (B, P) bool/int; for left-padded inputs
) -> torch.Tensor:
    """Per-token SAE-reconstruction delta for zeroing one feature.

    Returns (B, P, d_model) on the model's device, dtype-matched to the resid
    stream. Outside-prompt positions are zero.

    Pass `attention_mask` when tokens are left-padded so that padding positions
    are excluded from attention (exactly zero effect on real tokens).
    For left-padded prompts, `prompt_mask` should equal `attention_mask`.
    """
    device = next(model.parameters()).device
    tokens = tokens.to(device)
    prompt_mask = prompt_mask.to(device)
    extra: dict = {}
    if attention_mask is not None:
        extra["attention_mask"] = attention_mask.to(device)
    _, cache = model.run_with_cache(
        tokens, return_type=None, names_filter=lambda n: n == layer_hook, **extra
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
    return channel_steer_hook({"V": delta}, alpha, {"V": W_V}, block=block)


def head_selective_v_hook(
    delta: torch.Tensor,            # (B, P, d_model) ln1-space delta
    alpha: float,
    W_V: torch.Tensor,              # (n_heads, d_model, d_head)
    head_indices: list[int] | torch.Tensor,
    block: int = 0,
) -> list[tuple[str, Callable]]:
    """OV-only steer applied to a *subset* of heads.

    Same semantics as `ov_only_steer_hook` (Q, K untouched → A frozen) but only
    the specified heads' V tensors are perturbed; other heads see their
    unmodified V. Use to test whether a feature's effect on a downstream
    direction routes through a small set of heads.
    """
    head_idx = torch.as_tensor(list(head_indices), dtype=torch.long, device=W_V.device)
    W_V_sub  = W_V.index_select(0, head_idx)                                # (K, d_model, d_head)
    proj     = torch.einsum("bpd,kdh->bpkh", delta.float(), W_V_sub.float())  # (B, P, K, d_head)
    P        = delta.shape[1]
    hook_name = f"blocks.{block}.attn.hook_v"

    def _hook(x, hook):
        idx = head_idx.to(x.device)
        x[:, :P, idx, :] = x[:, :P, idx, :] + alpha * proj.to(x.dtype).to(x.device)
        return x

    return [(hook_name, _hook)]


_CHANNEL_HOOK_NAME = {"Q": "hook_q", "K": "hook_k", "V": "hook_v"}


def channel_steer_hook(
    channel_deltas: dict[str, torch.Tensor],   # subset of {'Q','K','V'} → (B, P, d_model)
    alpha: float,
    W: dict[str, torch.Tensor],                # matching subset → (n_heads, d_model, d_head)
    block: int = 0,
) -> list[tuple[str, Callable]]:
    """Patch any subset of {hook_q, hook_k, hook_v} with channel-specific deltas.

    Each delta is in ln1 space; we project through the matching W_{Q,K,V}[h] and
    add to the head-space tensor at the corresponding hook on prompt positions
    only. The other channels' inputs are untouched.

    `ov_only_steer_hook(delta, alpha, W_V)` ≡ `channel_steer_hook({"V": delta}, alpha, {"V": W_V})`.
    Dmitry's pareto_3x3.py qk hook ≡ `channel_steer_hook({"Q": d, "K": d}, alpha, {"Q": W_Q, "K": W_K})`.
    Channel-routed Triple+all = three distinct deltas in `channel_deltas`.
    """
    if not channel_deltas:
        return []
    hooks: list[tuple[str, Callable]] = []
    for ch, delta in channel_deltas.items():
        if ch not in _CHANNEL_HOOK_NAME:
            raise ValueError(f"unknown channel {ch!r}; expected one of Q, K, V")
        Wc = W[ch]
        proj = torch.einsum("bpd,hdk->bphk", delta.float(), Wc.float())
        P = delta.shape[1]
        hook_name = f"blocks.{block}.attn.{_CHANNEL_HOOK_NAME[ch]}"

        def _make(proj=proj, P=P):
            def _hook(x, hook):
                x[:, :P, :, :] = x[:, :P, :, :] + alpha * proj.to(x.dtype).to(x.device)
                return x
            return _hook

        hooks.append((hook_name, _make()))
    return hooks


# ---------------------------------------------------------------------------
# channel-delta resolution and hook construction
# ---------------------------------------------------------------------------

ACTIVE_CHANNELS: dict[str, set[str]] = {"ov": {"V"}, "qk": {"Q", "K"}, "all": {"Q", "K", "V"}}


@torch.no_grad()
def resolve_channel_deltas(
    selected: list[tuple[int, str]],
    active_channels: set[str],
    model: HookedTransformer,
    sae_ln1: TopKSAE,
    ln1_hook: str,
    tokens: torch.Tensor,
    prompt_mask: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    """For each active channel c: sum compute_sae_delta over naturally-tagged features,
    or fudge with all features if none carry that tag."""
    natural = {c: [f for (f, ch) in selected if ch == c] for c in ("Q", "K", "V")}
    all_features = list({f for (f, _) in selected})
    out: dict[str, torch.Tensor] = {}
    for c in active_channels:
        feats = natural[c] or all_features
        delta = None
        for f in feats:
            d = compute_sae_delta(model, sae_ln1, ln1_hook, f, tokens, prompt_mask,
                                  attention_mask)
            delta = d if delta is None else delta + d
        out[c] = delta
    return out


def build_hooks(
    channel_deltas: dict[str, torch.Tensor],
    alpha: float,
    active_channels: set[str],
    W: dict[str, torch.Tensor],
    ln1_hook: str,
    block: int,
) -> list[tuple[str, Callable]]:
    """Build steering hooks for any pipeline-matrix cell.

    Fast-paths to additive_steer_hook when all three channel deltas are the same
    object (OV+all with identical fudge tensors); otherwise uses channel_steer_hook.
    """
    if active_channels == {"Q", "K", "V"}:
        dQ, dK, dV = channel_deltas["Q"], channel_deltas["K"], channel_deltas["V"]
        if dQ is dK is dV:
            return additive_steer_hook(dQ, alpha, ln1_hook)
    return channel_steer_hook(channel_deltas, alpha, W, block=block)


# ---------------------------------------------------------------------------
# generation with hooks
# ---------------------------------------------------------------------------
#
# A `Sampler` is a callable (logits_last: (B, V) -> next_tok: (B,)) — typically
# a closure over a torch.Generator so the RNG state advances across decode
# steps. To get RNG-matched baseline/steered runs (same uniform draws per step,
# only logits differ), construct one sampler per generation call with the same
# seed.


Sampler = Callable[[torch.Tensor], torch.Tensor]


def make_greedy_sampler() -> Sampler:
    return lambda logits: logits.argmax(dim=-1)


def make_sampling_sampler(
    *, temperature: float, seed: int, device: torch.device | str,
) -> Sampler:
    """Pure multinomial sampling at `temperature` — no top_p / top_k truncation.

    Matches Ketan's 1000-prompt eval setup (`do_sample=True, top_k=None,
    top_p=None`). Same per-call seeded `torch.Generator` semantics as
    `make_nucleus_sampler`, so two samplers built with the same seed advance
    their RNG identically across decode steps."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def _sample(logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits / max(temperature, 1e-6), dim=-1)
        return torch.multinomial(probs, num_samples=1, generator=gen).squeeze(-1)

    return _sample


def make_nucleus_sampler(
    *, temperature: float, top_p: float, seed: int, device: torch.device | str,
) -> Sampler:
    """Top-p sampling with temperature. Each call to the returned sampler
    consumes one uniform per row from `gen` (via multinomial(num_samples=1)),
    so two samplers built with the same seed advance their RNG identically."""
    gen = torch.Generator(device=device).manual_seed(seed)

    def _sample(logits: torch.Tensor) -> torch.Tensor:
        last = logits / max(temperature, 1e-6)
        probs = torch.softmax(last, dim=-1)
        sp, si = probs.sort(descending=True, dim=-1)
        cum = sp.cumsum(dim=-1)
        mask = cum > top_p
        mask[..., 1:] = mask[..., :-1].clone()
        mask[..., 0] = False
        sp = sp.masked_fill(mask, 0.0)
        sp = sp / sp.sum(dim=-1, keepdim=True).clamp(min=1e-12)
        pick = torch.multinomial(sp, num_samples=1, generator=gen)
        return si.gather(-1, pick).squeeze(-1)

    return _sample


@torch.no_grad()
def generate_with_hooks(
    model: HookedTransformer,
    prompts: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int,
    sampler: Sampler,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Decode `max_new_tokens` tokens with `fwd_hooks` active each step.

    Hook deltas only patch the first P positions (the prompt), so it's safe to
    reapply as the sequence grows. Decoding rule lives entirely in `sampler`.

    Pass `attention_mask` when prompts are left-padded; it is extended with 1s
    as generated tokens are appended (they are always real).
    """
    device = next(model.parameters()).device
    tokens = prompts.to(device)
    attn = attention_mask.to(device) if attention_mask is not None else None
    out: list[torch.Tensor] = []
    for _ in range(max_new_tokens):
        extra: dict = {"return_type": "logits"}
        if attn is not None:
            extra["attention_mask"] = attn
        logits = model.run_with_hooks(tokens, fwd_hooks=fwd_hooks, **extra)
        nxt = sampler(logits[:, -1, :])
        out.append(nxt.unsqueeze(1))
        tokens = torch.cat([tokens, nxt.unsqueeze(1)], dim=1)
        if attn is not None:
            attn = torch.cat([attn, attn.new_ones(attn.shape[0], 1)], dim=1)
    return torch.cat(out, dim=1)


def greedy_generate_with_hooks(
    model: HookedTransformer,
    prompts: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int = 16,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Backwards-compat shim: `generate_with_hooks` with a greedy sampler."""
    return generate_with_hooks(
        model, prompts, fwd_hooks, max_new_tokens, make_greedy_sampler(), attention_mask,
    )


def sample_generate_with_hooks(
    model: HookedTransformer,
    prompts: torch.Tensor,
    fwd_hooks: list[tuple[str, Callable]],
    max_new_tokens: int = 16,
    temperature: float = 0.8,
    top_p: float = 0.9,
    seed: int = 0,
) -> torch.Tensor:
    """Backwards-compat shim: `generate_with_hooks` with a fresh nucleus sampler."""
    device = next(model.parameters()).device
    return generate_with_hooks(
        model, prompts, fwd_hooks, max_new_tokens,
        make_nucleus_sampler(temperature=temperature, top_p=top_p,
                             seed=seed, device=device),
    )
