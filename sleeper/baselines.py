"""Non-SAE baselines for the resid_mid suppressor: mean-difference vector."""

from __future__ import annotations

import torch
from transformer_lens import HookedTransformer


@torch.no_grad()
def compute_dom_vectors(
    model: HookedTransformer,
    dep_full: torch.Tensor,           # (B, T_full) prompt ‖ dep rollout
    cln_full: torch.Tensor,           # (B, T_full) prompt ‖ clean rollout
    dep_attn: torch.Tensor,           # (B, T_full) attention mask
    cln_attn: torch.Tensor,
    gen_slice: slice,                 # positions to average over (the "answer tokens")
    *,
    n_layers: int,
    hook_kinds: tuple[str, ...] = ("hook_resid_mid", "hook_resid_post"),
) -> dict[tuple[int, str], torch.Tensor]:
    """Soligo et al. (2025) DoM extraction adapted to the sleeper.

    For each layer ℓ and hook kind h ∈ {hook_resid_mid, hook_resid_post}:
        μ_dep_{ℓ,h}  = mean over (b ∈ dep batch, t ∈ gen_slice) of resid[b, t, :]
        μ_cln_{ℓ,h}  = mean over (b ∈ cln batch, t ∈ gen_slice) of resid[b, t, :]
        v_{ℓ,h}      = μ_dep_{ℓ,h} − μ_cln_{ℓ,h}

    Vector points toward dep behavior; subtracting α·v (positive α) suppresses
    sleeper. `gen_slice` should index the generated-answer-token positions to
    mirror the paper's "average over answer tokens" convention. Positions
    outside `attn==1` are ignored even if they fall inside `gen_slice`.

    Returns dict keyed by (layer, hook_kind) on CPU.
    """
    device = next(model.parameters()).device
    dep_full = dep_full.to(device); dep_attn = dep_attn.to(device)
    cln_full = cln_full.to(device); cln_attn = cln_attn.to(device)

    names = {f"blocks.{l}.{k}" for l in range(n_layers) for k in hook_kinds}

    def _means(tokens: torch.Tensor, attn: torch.Tensor) -> dict[str, torch.Tensor]:
        _, cache = model.run_with_cache(
            tokens, attention_mask=attn, return_type=None,
            names_filter=lambda n: n in names,
        )
        out: dict[str, torch.Tensor] = {}
        mask = attn[:, gen_slice].to(torch.float32)              # (B, T_gen)
        denom = mask.sum().clamp(min=1.0)
        for n in names:
            a = cache[n][:, gen_slice, :].to(torch.float32)      # (B, T_gen, d)
            out[n] = (a * mask.unsqueeze(-1)).sum(dim=(0, 1)) / denom
        return out

    mu_dep = _means(dep_full, dep_attn)
    mu_cln = _means(cln_full, cln_attn)

    return {(l, k): (mu_dep[f"blocks.{l}.{k}"] - mu_cln[f"blocks.{l}.{k}"]).cpu()
            for l in range(n_layers) for k in hook_kinds}


@torch.no_grad()
def compute_meandiff_vector(
    acts: torch.Tensor,                  # (N, T, d) any dtype, on CPU
    is_deployment: torch.Tensor,         # (N,) bool
    prompt_mask: torch.Tensor,           # (N, T) bool
) -> dict[str, torch.Tensor]:
    """v_md = mean(acts[dep, prompt]) - mean(acts[clean, prompt]).

    Per-sequence prompt mean first, then class mean — matches the original
    `baseline_meandiff_steer` script (one row per sequence, not per token).
    """
    a = acts.to(torch.float32)
    mask = prompt_mask.to(torch.float32).unsqueeze(-1)
    weight = mask.sum(dim=1).clamp(min=1.0)
    per_seq_mean = (a * mask).sum(dim=1) / weight       # (N, d)
    dep_mean = per_seq_mean[is_deployment].mean(dim=0)
    cln_mean = per_seq_mean[~is_deployment].mean(dim=0)
    return {
        "v_md": dep_mean - cln_mean,
        "dep_mean": dep_mean,
        "cln_mean": cln_mean,
        "n_dep": int(is_deployment.sum()),
        "n_cln": int((~is_deployment).sum()),
    }
