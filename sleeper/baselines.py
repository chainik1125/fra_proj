"""Non-SAE baselines for the resid_mid suppressor: mean-difference vector."""

from __future__ import annotations

import torch


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
