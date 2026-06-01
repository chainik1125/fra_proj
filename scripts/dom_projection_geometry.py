"""Why is the projection-ablation optimum alpha > 1 (not 1)?

Projection ablation x -> x - alpha*(v_hat . x) v_hat zeroes the v_hat-component
at alpha=1. But the target is the CLEAN activation, which has projection
c_clean = c_dep - |dmu| onto v_hat. To map dep onto clean we need
(1-alpha) c_dep = c_clean  =>  alpha* = 1 - c_clean/c_dep.

Measures c_dep, c_clean (mean v_hat-projection of dep / clean prompt positions at
resid_mid) and the predicted alpha*. If c_clean < 0 (clean on the far side of
zero), alpha* > 1. Out: stdout only.
"""
from __future__ import annotations
import torch
from sleeper.eval import PAT_HOOK
from sleeper.model import (MODELS, load_sleeper_model, cache_activations, load_paired_dataset)

DEV = "cuda"; MODEL = "tinystories"; RESID = "blocks.0.hook_resid_mid"; N_SEL = 200


@torch.no_grad()
def main():
    model = load_sleeper_model(model=MODEL, device=DEV); tok = model.tokenizer
    seq_len = MODELS[MODEL].seq_len
    sel = load_paired_dataset(tok, n_train=2, n_val=N_SEL, n_test=2, seq_len=seq_len, seed=0, model=MODEL)["val"]
    isd = sel.is_deployment.to(DEV)
    acts = cache_activations(model, sel.tokens, [PAT_HOOK, RESID])
    Ap = acts[PAT_HOOK].to(DEV).float(); Xr = acts[RESID].to(DEV).float()
    pmf = sel.attention_mask.to(DEV).float()

    # attn-weighted v_md (the paper's direction), then unit-normalize
    recv = Ap.sum(dim=(1, 2)) * pmf; recv = recv / recv.sum(1, keepdim=True).clamp_min(1e-9)
    a_mean = (Xr * recv.unsqueeze(-1)).sum(1)
    vmd = a_mean[isd].mean(0) - a_mean[~isd].mean(0)
    vhat = vmd / vmd.norm().clamp_min(1e-30)

    # per-position projection onto v_hat over valid prompt positions
    proj = (Xr @ vhat)                      # [N, P]
    m = pmf.bool()
    dep_mask = m & isd.unsqueeze(1); cln_mask = m & (~isd).unsqueeze(1)
    c_dep = float(proj[dep_mask].mean()); c_cln = float(proj[cln_mask].mean())
    dmu = c_dep - c_cln
    alpha_pred = 1.0 - c_cln / c_dep

    # also attn-weighted (matches how v_md was built)
    cdep_a = float((a_mean[isd] @ vhat).mean()); ccln_a = float((a_mean[~isd] @ vhat).mean())
    alpha_pred_a = 1.0 - ccln_a / cdep_a

    print("[geom] === projection of resid_mid onto unit v_md (attn-wtd diff-of-means) ===", flush=True)
    print(f"[geom] per-position (uniform over prompt tokens):", flush=True)
    print(f"[geom]   c_dep   = {c_dep:+.4f}   c_clean = {c_cln:+.4f}   |dmu| = {dmu:.4f}", flush=True)
    print(f"[geom]   clean on far side of zero? {'YES (c_clean<0)' if c_cln < 0 else 'no (c_clean>=0)'}", flush=True)
    print(f"[geom]   predicted alpha* = 1 - c_clean/c_dep = {alpha_pred:.3f}", flush=True)
    print(f"[geom] attn-weighted means:", flush=True)
    print(f"[geom]   c_dep_a = {cdep_a:+.4f}   c_clean_a = {ccln_a:+.4f}   predicted alpha* = {alpha_pred_a:.3f}", flush=True)
    print(f"[geom] observed JSDc-optimal alpha (from sweep) = 1.25", flush=True)


if __name__ == "__main__":
    main()
