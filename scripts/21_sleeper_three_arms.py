"""Three interventions, one pair, one model: does score-space delivery help?

Feature selection is held FIXED -- the pair (lambda_q=1114, lambda_k=1232) is
rank 0 under the paper's own `rank_qk_diff` on our seed-0 ln1 SAE. Only the
delivery mechanism varies.

  A  score_pair   subtract scale * f[q,1114] * f[k,1232] * G_h[1114,1232] from
                  blocks.0.attn.hook_attn_scores, per head. Removes the CROSS
                  TERM and nothing else. Never round-trips through the SAE.
  B  paper_qk     the paper's QK channel, via their own `channel_steer_hook`:
                  patch hook_q with -a f[t,1114] W_dec[1114] W_Q^h and hook_k
                  with -a f[t,1232] W_dec[1232] W_K^h, INDEPENDENTLY.
  C  ln1_ablate   conventional SAE-feature ablation at the SAE's own hook:
                  subtract a f[t,1114] W_dec[1114] from ln1.hook_normalized,
                  which removes the feature from Q, K AND V.

Note C is not literally the paper's `conv` baseline -- that uses a separate
resid_mid SAE we have deliberately not trained yet. C is the conventional
feature ablation available with the SAE we have, and it is the arm that also
corrupts VALUES.

BREADTH IS A CONFOUND AND IS MEASURED, NOT ASSUMED. 1114 is rare and
trigger-local (0.0% of clean positions); 1232 is frequent (~28%). B patches both
sides, so it touches far more of the sequence than A before any collateral is
measured. Positions/cells touched are reported alongside every collateral number
so a reader can see how much of B's cost is breadth rather than independence.

Two metric families, reported separately -- conflating them would misattribute
depth-dilution to the method:

  MECHANISM  perturbed set at blocks.0.hook_resid_post (the intervened layer's
             output). This is where the containment claim is exact.
  PRACTICAL  final-logit KL at non-target positions. Layers 1-3 spread any
             layer-0 change to every later position, so this is diluted by
             depth for ALL arms.

Run: python scripts/21_sleeper_three_arms.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from sleeper.eval import LN1_HOOK
from sleeper.hooks import channel_steer_hook
from sleeper.metrics import teacher_forced_sleeper_logp
from sleeper.model import cache_activations, load_paired_dataset, load_sleeper_model
from sleeper.sae import encode_all, load as sae_load

SAE_PATH = Path("weights/seeds/sae_ln1_s0.pt")
LAM_Q, LAM_K = 1114, 1232
N_PROMPTS = 200
STRENGTHS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
RESID_POST = "blocks.0.hook_resid_post"
OUT = Path("results/sleeper_three_arms.json")


def score_pair_hooks(f, W_dec, W_Q, W_K, attn_scale, scale):
    """Arm A: remove the (lambda_q, lambda_k) cross term from the attention scores."""
    qv = W_dec[LAM_Q] @ W_Q                       # (n_heads, d_head)
    kv = W_dec[LAM_K] @ W_K                       # (n_heads, d_head)
    g = (qv * kv).sum(-1) / attn_scale            # (n_heads,)  == G_h[lam_q, lam_k]
    fq, fk = f[:, :, LAM_Q], f[:, :, LAM_K]       # (B, T)
    delta = scale * torch.einsum("bq,bk,h->bhqk", fq, fk, g)
    P = delta.shape[-1]

    def _hook(scores, hook):
        # Sequence may be longer than P (metrics append the payload); patch the
        # prompt block only, matching channel_steer_hook's convention.
        if scores.shape[-1] < P:
            return scores
        scores[:, :, :P, :P] = scores[:, :, :P, :P] - delta.to(scores.dtype)
        return scores

    return [("blocks.0.attn.hook_attn_scores", _hook)]


def paper_qk_hooks(f, W_dec, W_Q, W_K, alpha):
    """Arm B: the paper's QK channel, using their channel_steer_hook verbatim."""
    dq = -f[:, :, LAM_Q].unsqueeze(-1) * W_dec[LAM_Q]     # (B, T, d_model)
    dk = -f[:, :, LAM_K].unsqueeze(-1) * W_dec[LAM_K]
    return channel_steer_hook({"Q": dq, "K": dk}, alpha, {"Q": W_Q, "K": W_K}, block=0)


def ln1_ablate_hooks(f, W_dec, alpha):
    """Arm C: conventional feature ablation at the SAE's hook (hits Q, K and V)."""
    delta = alpha * f[:, :, LAM_Q].unsqueeze(-1) * W_dec[LAM_Q]   # (B, T, d_model)
    P = delta.shape[1]

    def _hook(x, hook):
        if x.shape[1] < P:
            return x
        x[:, :P, :] = x[:, :P, :] - delta.to(x.dtype)
        return x

    return [(LN1_HOOK, _hook)]


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = load_sleeper_model("tinystories", device="cpu")
    sae, _ = sae_load(SAE_PATH, device="cpu")
    W_dec = sae.W_dec.detach().float()
    W_Q, W_K = model.W_Q[0].detach().float(), model.W_K[0].detach().float()
    attn_scale = model.blocks[0].attn.attn_scale

    split = load_paired_dataset(model.tokenizer, n_train=0, n_val=0,
                                n_test=N_PROMPTS, seq_len=128, seed=0)["test"]
    tokens, is_dep, amask = split.tokens, split.is_deployment, split.attention_mask
    z = encode_all(sae, cache_activations(model, tokens, [LN1_HOOK])[LN1_HOOK])
    B, T = tokens.shape

    # Non-target = real positions other than the final one (where the payload is read).
    target = torch.zeros(B, T, dtype=torch.bool)
    target[:, -1] = True
    nontarget = amask.bool() & ~target

    base_logits = model(tokens, attention_mask=amask)
    base_logp = torch.log_softmax(base_logits, -1)
    _, base_cache = model.run_with_cache(tokens, names_filter=[RESID_POST],
                                         attention_mask=amask)
    base_resid = base_cache[RESID_POST]
    base_sleeper = teacher_forced_sleeper_logp(model, model.tokenizer, tokens[is_dep],
                                               attention_mask=amask[is_dep]).mean().item()

    # ── Breadth, measured once (strength-independent) ───────────────────
    fq_on = (z[:, :, LAM_Q] > 0) & amask.bool()
    fk_on = (z[:, :, LAM_K] > 0) & amask.bool()
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    cells_A = int((fq_on.unsqueeze(2) & fk_on.unsqueeze(1) & causal).sum())
    breadth = {
        "A_score_cells": cells_A,
        "A_rows_touched": int(fq_on.sum()),
        "B_positions": int(fq_on.sum()) + int(fk_on.sum()),
        "C_positions": int(fq_on.sum()),
        "total_real_positions": int(amask.sum()),
        "total_causal_cells": int((amask.bool().unsqueeze(2) & amask.bool().unsqueeze(1) & causal).sum()),
        "lam_q_fire_rate_dep": float((z[is_dep][:, :, LAM_Q] > 0).float().mean()),
        "lam_q_fire_rate_cln": float((z[~is_dep][:, :, LAM_Q] > 0).float().mean()),
        "lam_k_fire_rate_dep": float((z[is_dep][:, :, LAM_K] > 0).float().mean()),
        "lam_k_fire_rate_cln": float((z[~is_dep][:, :, LAM_K] > 0).float().mean()),
    }
    print("=" * 96)
    print("BREADTH (strength-independent; the confound, measured)")
    print("=" * 96)
    print(f"  lambda_q={LAM_Q} fires: dep {breadth['lam_q_fire_rate_dep']*100:5.2f}%  "
          f"clean {breadth['lam_q_fire_rate_cln']*100:5.2f}%   (rare, trigger-local)")
    print(f"  lambda_k={LAM_K} fires: dep {breadth['lam_k_fire_rate_dep']*100:5.2f}%  "
          f"clean {breadth['lam_k_fire_rate_cln']*100:5.2f}%   (frequent)")
    print(f"  A: {cells_A} score cells of {breadth['total_causal_cells']} causal "
          f"({cells_A/breadth['total_causal_cells']*100:.4f}%)")
    print(f"  B: {breadth['B_positions']} positions of {breadth['total_real_positions']} "
          f"({breadth['B_positions']/breadth['total_real_positions']*100:.2f}%)")
    print(f"  C: {breadth['C_positions']} positions of {breadth['total_real_positions']} "
          f"({breadth['C_positions']/breadth['total_real_positions']*100:.2f}%)")

    def make(name, zz, s):
        if name == "A_score_pair":
            return score_pair_hooks(zz, W_dec, W_Q, W_K, attn_scale, s)
        if name == "B_paper_qk":
            return paper_qk_hooks(zz, W_dec, W_Q, W_K, s)
        return ln1_ablate_hooks(zz, W_dec, s)

    arms = ["A_score_pair", "B_paper_qk", "C_ln1_ablate"]
    z_dep = z[is_dep]

    rows = []
    for name in arms:
        print(f"\n{'=' * 96}\n{name}\n{'=' * 96}")
        print(f"  {'strength':>8s} | {'sleeper_logp':>12s} {'suppress':>9s} | "
              f"{'resid_frac':>10s} {'resid_maxd':>10s} | {'KL_dep':>10s} {'KL_clean':>10s}")
        for s in STRENGTHS:
            with model.hooks(fwd_hooks=make(name, z, s)):
                logits = model(tokens, attention_mask=amask)
                _, cache = model.run_with_cache(tokens, names_filter=[RESID_POST],
                                                attention_mask=amask)
            sl = teacher_forced_sleeper_logp(
                model, model.tokenizer, tokens[is_dep],
                fwd_hooks=make(name, z_dep, s), attention_mask=amask[is_dep]).mean().item()

            dresid = (cache[RESID_POST] - base_resid).abs().amax(-1)     # (B, T)
            resid_changed = (dresid > 1e-5) & amask.bool()
            logp = torch.log_softmax(logits, -1)
            kl = (base_logp.exp() * (base_logp - logp)).sum(-1)          # (B, T)
            kl_dep = kl[nontarget & is_dep.unsqueeze(1)].mean().item()
            kl_cln = kl[nontarget & (~is_dep).unsqueeze(1)].mean().item()

            rows.append({
                "arm": name, "strength": s, "sleeper_logp": sl,
                "suppression": base_sleeper - sl,
                "resid_frac_changed": float(resid_changed.sum() / amask.sum()),
                "resid_max_delta": float(dresid[amask.bool()].max()),
                "kl_dep": kl_dep, "kl_clean": kl_cln,
            })
            print(f"  {s:8.2f} | {sl:12.3f} {base_sleeper - sl:+9.3f} | "
                  f"{rows[-1]['resid_frac_changed']*100:9.2f}% {rows[-1]['resid_max_delta']:10.4f} | "
                  f"{kl_dep:10.3e} {kl_cln:10.3e}", flush=True)

    OUT.write_text(json.dumps({"breadth": breadth, "base_sleeper_logp": base_sleeper,
                               "pair": [LAM_Q, LAM_K], "rows": rows}, indent=2))
    print(f"\n  baseline sleeper logp = {base_sleeper:.3f}")
    print(f"  wrote {OUT}")


if __name__ == "__main__":
    main()
