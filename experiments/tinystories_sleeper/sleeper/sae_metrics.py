"""SAE quality metrics on a held-out eval set.

The informative axes as dictionary width grows at fixed TopK:
  • fvu / explained_variance — reconstruction quality, scale-invariant, the
    primary cross-width-comparable metric.
  • nmse                     — matches the training objective's geometry
    (sum-over-d squared error), sanity-checks against losses[-1].
  • dead_feature_frac        — features never active over the eval set; expected
    to rise with width at fixed k.
  • firing_rate_{median,p90,hist} — per-feature activation density; the real
    sparsity-quality signal (L0 is degenerate = k by TopK construction).
  • loss_recovered           — CE-substitution metric (formula from
    fra/diagnostics_phase3.py, re-wired to the sleeper model).

Chunked so a d_sae=24576 dictionary's (N·T, d_sae) codes never materialise
fully on GPU.
"""
from __future__ import annotations

import torch

from sleeper.model import cache_activations
from sleeper.sae import TopKSAE


def _make_reconstruct_hook(sae: TopKSAE):
    def _rec(act, hook):
        with torch.no_grad():
            return sae.decode(sae.encode(act.float())).to(act.dtype)
    return _rec


def _make_zero_hook():
    def _zero(act, hook):
        return torch.zeros_like(act)
    return _zero


@torch.no_grad()
def _loss_recovered(model, sae, hook, eval_tokens, *, chunk_size=16, device="cuda") -> dict:
    """1 − (loss_sae − loss_clean)/(loss_zero − loss_clean), token-mean CE."""
    rec_hook = _make_reconstruct_hook(sae)
    zero_hook = _make_zero_hook()
    lc, ls, lz = [], [], []
    for s in range(0, eval_tokens.shape[0], chunk_size):
        toks = eval_tokens[s : s + chunk_size].to(device)
        lc.append(model(toks, return_type="loss").item())
        ls.append(model.run_with_hooks(toks, fwd_hooks=[(hook, rec_hook)],
                                       return_type="loss").item())
        lz.append(model.run_with_hooks(toks, fwd_hooks=[(hook, zero_hook)],
                                       return_type="loss").item())
    loss_clean = sum(lc) / len(lc)
    loss_sae = sum(ls) / len(ls)
    loss_zero = sum(lz) / len(lz)
    denom = loss_zero - loss_clean
    rec = 1.0 - (loss_sae - loss_clean) / denom if abs(denom) > 1e-9 else float("nan")
    return {"loss_clean": loss_clean, "loss_sae": loss_sae,
            "loss_zero": loss_zero, "loss_recovered": rec}


@torch.no_grad()
def sae_quality_metrics(model, sae: TopKSAE, hook: str, eval_tokens: torch.Tensor,
                        *, chunk: int = 4096, device: str = "cuda") -> dict:
    """Reconstruction + sparsity-quality metrics on `eval_tokens` at `hook`."""
    acts = cache_activations(model, eval_tokens, [hook])[hook]   # (N, T, d) cpu fp16
    d_model = acts.shape[-1]
    flat = acts.reshape(-1, d_model).float()                     # (NT, d_model) cpu
    NT = flat.shape[0]
    mean = flat.mean(dim=0)
    total_ss = (flat - mean).pow(2).sum().item()                 # for FVU (centered)
    act_ss = flat.pow(2).sum().item()                            # for NMSE (uncentered)

    resid_ss = 0.0
    fired = torch.zeros(sae.d_sae, dtype=torch.bool)
    firing_count = torch.zeros(sae.d_sae, dtype=torch.float64)
    for s in range(0, NT, chunk):
        x = flat[s : s + chunk].to(device)
        z = sae.encode(x)                                        # (chunk, d_sae) on device
        x_hat = sae.decode(z)
        resid_ss += (x - x_hat).pow(2).sum().item()
        active = (z > 0)
        fired |= active.any(dim=0).cpu()
        firing_count += active.sum(dim=0).double().cpu()

    fvu = resid_ss / max(total_ss, 1e-12)
    nmse = resid_ss / max(act_ss, 1e-12)
    firing_rate = (firing_count / NT).float()
    lr = _loss_recovered(model, sae, hook, eval_tokens, device=device)

    return {
        "fvu": fvu,
        "explained_variance": 1.0 - fvu,
        "nmse": nmse,
        "dead_feature_frac": 1.0 - fired.float().mean().item(),
        "firing_rate_median": firing_rate.median().item(),
        "firing_rate_p90": firing_rate.quantile(0.9).item(),
        "firing_rate_max": firing_rate.max().item(),
        "firing_rate_hist": torch.histc(firing_rate, bins=20, min=0.0, max=1.0).tolist(),
        "L0_nominal": int(sae.k),   # degenerate for TopK (always = k); reported for completeness
        "n_eval_tokens": int(NT),
        **lr,
    }
