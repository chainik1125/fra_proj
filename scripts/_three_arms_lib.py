"""Shared machinery for the sleeper three-arm comparison, parameterised by pair.

Arms (feature selection held fixed; only DELIVERY varies):
  A score_pair   remove scale * f[q,lq] * f[k,lk] * G_h[lq,lk] from the attention
                 scores, per head. The cross term and nothing else.
  B paper_qk     the paper's QK channel via their own `channel_steer_hook`:
                 hook_q patched with lq's contribution, hook_k with lk's,
                 INDEPENDENTLY. Not a pair ablation.
  C ln1_ablate   conventional feature ablation at the SAE's hook -- removes lq
                 from Q, K AND V.

Metrics, reported separately because conflating them misattributes
depth-dilution (4 layers) to the method:
  MECHANISM  perturbed set at blocks.0.hook_resid_post (the intervened layer).
  PRACTICAL  final-logit KL at non-target positions, split clean vs deployment.
Plus BREADTH -- positions/cells touched -- so collateral can be read against how
much of the model each arm perturbs.
"""

from __future__ import annotations

from pathlib import Path

import torch

from sleeper.eval import LN1_HOOK
from sleeper.hooks import channel_steer_hook
from sleeper.metrics import teacher_forced_sleeper_logp
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, load as sae_load

SAE_PATH = Path("weights/seeds/sae_ln1_s0.pt")
N_PROMPTS = 200
RESID_POST = "blocks.0.hook_resid_post"
SCORES = "blocks.0.attn.hook_attn_scores"


def score_pair_hooks(f, W_dec, W_Q, W_K, attn_scale, lq, lk, scale):
    g = ((W_dec[lq] @ W_Q) * (W_dec[lk] @ W_K)).sum(-1) / attn_scale     # (n_heads,)
    delta = scale * torch.einsum("bq,bk,h->bhqk", f[:, :, lq], f[:, :, lk], g)
    P = delta.shape[-1]

    def _hook(scores, hook):
        if scores.shape[-1] < P:
            return scores
        scores[:, :, :P, :P] = scores[:, :, :P, :P] - delta.to(scores.dtype)
        return scores

    return [(SCORES, _hook)]


def paper_qk_hooks(f, W_dec, W_Q, W_K, lq, lk, alpha):
    dq = -f[:, :, lq].unsqueeze(-1) * W_dec[lq]
    dk = -f[:, :, lk].unsqueeze(-1) * W_dec[lk]
    return channel_steer_hook({"Q": dq, "K": dk}, alpha, {"Q": W_Q, "K": W_K}, block=0)


def ln1_ablate_hooks(f, W_dec, lq, alpha):
    delta = alpha * f[:, :, lq].unsqueeze(-1) * W_dec[lq]
    P = delta.shape[1]

    def _hook(x, hook):
        if x.shape[1] < P:
            return x
        x[:, :P, :] = x[:, :P, :] - delta.to(x.dtype)
        return x

    return [(LN1_HOOK, _hook)]


@torch.no_grad()
def setup():
    model = load_sleeper_model("tinystories", device="cpu")
    sae, _ = sae_load(SAE_PATH, device="cpu")
    split = load_paired_dataset(model.tokenizer, n_train=0, n_val=0,
                                n_test=N_PROMPTS, seq_len=128, seed=0)["test"]
    tokens, is_dep, amask = split.tokens, split.is_deployment, split.attention_mask
    z = encode_all(sae, cache_activations(model, tokens, [LN1_HOOK])[LN1_HOOK])
    B, T = tokens.shape

    target = torch.zeros(B, T, dtype=torch.bool)
    target[:, -1] = True
    nontarget = amask.bool() & ~target

    base_logits = model(tokens, attention_mask=amask)
    _, cache = model.run_with_cache(tokens, names_filter=[RESID_POST],
                                    attention_mask=amask)
    base_sleeper = teacher_forced_sleeper_logp(
        model, model.tokenizer, tokens[is_dep], attention_mask=amask[is_dep]).mean().item()

    return {
        "model": model, "W_dec": sae.W_dec.detach().float(),
        "W_Q": model.W_Q[0].detach().float(), "W_K": model.W_K[0].detach().float(),
        "attn_scale": model.blocks[0].attn.attn_scale,
        "tokens": tokens, "is_dep": is_dep, "amask": amask, "z": z,
        "nontarget": nontarget, "T": T,
        "base_logp": torch.log_softmax(base_logits, -1),
        "base_resid": cache[RESID_POST], "base_sleeper": base_sleeper,
    }


@torch.no_grad()
def run_pair(c, lq, lk, label, strengths):
    model, z, amask, is_dep = c["model"], c["z"], c["amask"], c["is_dep"]
    tokens, T = c["tokens"], c["T"]
    z_dep = z[is_dep]

    def make(arm, zz, s):
        if arm == "A_score_pair":
            return score_pair_hooks(zz, c["W_dec"], c["W_Q"], c["W_K"],
                                    c["attn_scale"], lq, lk, s)
        if arm == "B_paper_qk":
            return paper_qk_hooks(zz, c["W_dec"], c["W_Q"], c["W_K"], lq, lk, s)
        return ln1_ablate_hooks(zz, c["W_dec"], lq, s)

    fq_on = (z[:, :, lq] > 0) & amask.bool()
    fk_on = (z[:, :, lk] > 0) & amask.bool()
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    cells = int((fq_on.unsqueeze(2) & fk_on.unsqueeze(1) & causal).sum())
    total_cells = int((amask.bool().unsqueeze(2) & amask.bool().unsqueeze(1) & causal).sum())
    live_q = float((z[is_dep][:, -1, lq] > 0).float().mean())

    breadth = {
        "A_cells": cells, "A_cells_pct": 100 * cells / total_cells,
        "B_positions": int(fq_on.sum()) + int(fk_on.sum()),
        "C_positions": int(fq_on.sum()),
        "total_positions": int(amask.sum()),
        "lq_dep": float((z[is_dep][:, :, lq] > 0).float().mean()),
        "lq_cln": float((z[~is_dep][:, :, lq] > 0).float().mean()),
        "lk_dep": float((z[is_dep][:, :, lk] > 0).float().mean()),
        "lk_cln": float((z[~is_dep][:, :, lk] > 0).float().mean()),
        "lq_live_at_decision_dep": live_q,
    }
    print(f"  breadth: A {cells} cells ({breadth['A_cells_pct']:.4f}%)  "
          f"B {breadth['B_positions']} pos  C {breadth['C_positions']} pos "
          f"of {breadth['total_positions']}")
    print(f"  lq={lq} fires dep {breadth['lq_dep']*100:.2f}% clean {breadth['lq_cln']*100:.2f}%  "
          f"LIVE AT DECISION {live_q*100:.1f}%   |   "
          f"lk={lk} fires dep {breadth['lk_dep']*100:.2f}% clean {breadth['lk_cln']*100:.2f}%")

    rows = []
    for arm in ("A_score_pair", "B_paper_qk", "C_ln1_ablate"):
        print(f"\n  {arm}")
        print(f"    {'strength':>8s} | {'sleeper':>9s} {'suppress':>9s} | "
              f"{'resid%':>7s} | {'KL_dep':>10s} {'KL_clean':>10s}")
        for s in strengths:
            with model.hooks(fwd_hooks=make(arm, z, s)):
                logits = model(tokens, attention_mask=amask)
                _, cache = model.run_with_cache(tokens, names_filter=[RESID_POST],
                                                attention_mask=amask)
            sl = teacher_forced_sleeper_logp(
                model, model.tokenizer, tokens[is_dep],
                fwd_hooks=make(arm, z_dep, s), attention_mask=amask[is_dep]).mean().item()

            dres = (cache[RESID_POST] - c["base_resid"]).abs().amax(-1)
            frac = float(((dres > 1e-5) & amask.bool()).sum() / amask.sum())
            lp = torch.log_softmax(logits, -1)
            kl = (c["base_logp"].exp() * (c["base_logp"] - lp)).sum(-1)
            kd = float(kl[c["nontarget"] & is_dep.unsqueeze(1)].mean())
            kc = float(kl[c["nontarget"] & (~is_dep).unsqueeze(1)].mean())

            rows.append({"arm": arm, "strength": s, "sleeper_logp": sl,
                         "suppression": c["base_sleeper"] - sl,
                         "resid_frac": frac, "kl_dep": kd, "kl_clean": kc})
            print(f"    {s:8.2f} | {sl:9.3f} {c['base_sleeper'] - sl:+9.3f} | "
                  f"{frac*100:6.2f}% | {kd:10.3e} {kc:10.3e}", flush=True)

    return {"pair": [lq, lk], "label": label, "breadth": breadth,
            "base_sleeper_logp": c["base_sleeper"], "rows": rows}


# ── Feature-level ablation in SCORE space (the untested cell) ────────────
#
# A_pair removes one (lambda, mu) cross term, which is ~2% of a score cell at
# L_0 = 32. A_feat removes feature lambda's ENTIRE contribution to the scores,
# summed over every key-side partner -- L_0 times more mass -- while still never
# round-tripping activations through the SAE, so the encode-decode confound that
# broke the paper's Case 2 stays structurally absent.
#
# The sum over mu collapses:
#
#   sum_mu f[k,mu] G_h[lambda,mu]
#       = (W_dec[lambda] W_Q[h]) . ((f[k] W_dec) W_K[h]) / attn_scale
#
# i.e. the key projection of the SAE reconstruction. No d_sae x d_sae matrix is
# ever built.

def _proj(f, W_dec, W):
    """(f @ W_dec) @ W[h] for every head -> (B, n_heads, T, d_head)."""
    xhat = f @ W_dec                                   # (B, T, d_model)
    return torch.einsum("btd,hde->bhte", xhat, W)


def score_feature_hooks(f, W_dec, W_Q, W_K, attn_scale, lam, scale,
                        symmetric: bool = False):
    """Remove feature `lam`'s whole contribution to the pre-softmax scores.

    Query side:  `f[q,lam] * sum_mu f[k,mu] G[lam,mu]`
    Key side:    `sum_lam' f[q,lam'] G[lam',lam] * f[k,lam]`

    `symmetric=True` removes both and subtracts the double-counted diagonal
    `f[q,lam] f[k,lam] G[lam,lam]`, which is the two-sided reading of "remove
    this feature from the score".
    """
    qv = W_dec[lam] @ W_Q                              # (n_heads, d_head)
    kv = W_dec[lam] @ W_K                              # (n_heads, d_head)
    fq = f[:, :, lam]                                  # (B, T)

    khat = _proj(f, W_dec, W_K)                        # (B, H, T, d_head)
    a_k = torch.einsum("he,bhte->bht", qv, khat)       # (B, H, T)
    delta = fq[:, None, :, None] * a_k[:, :, None, :] / attn_scale

    if symmetric:
        qhat = _proj(f, W_dec, W_Q)
        b_q = torch.einsum("he,bhte->bht", kv, qhat)   # (B, H, T)
        delta = delta + b_q[:, :, :, None] * fq[:, None, None, :] / attn_scale
        g_ll = (qv * kv).sum(-1) / attn_scale          # (H,)
        delta = delta - torch.einsum("bq,bk,h->bhqk", fq, fq, g_ll)

    delta = delta * scale
    P = delta.shape[-1]

    def _hook(scores, hook):
        if scores.shape[-1] < P:
            return scores
        scores[:, :, :P, :P] = scores[:, :, :P, :P] - delta.to(scores.dtype)
        return scores

    return [(SCORES, _hook)]
