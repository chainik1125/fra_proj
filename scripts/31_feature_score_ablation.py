"""Feature-level ablation in SCORE space -- the cell we identified twice and never ran.

Every previous score-space arm removed one `(lambda, mu)` PAIR, which is ~2% of a
score cell at `L_0 = 32`, and was a null. Activation-space feature ablation (arm
C) gives +117. The untested cell is feature-level ablation in score space: it
removes `L_0` times more mass than a pair, while still never round-tripping
activations through the SAE -- so it keeps the property that would remove the
paper's Case 2 reconstruction confound.

  A_feat      subtract  scale * f[q,lam] * sum_mu f[k,mu] G[lam,mu]
  A_feat_sym  the two-sided version, minus the double-counted diagonal

`lam = 1114`, the trigger detector (4.88% dep, 0.00% clean), so this is a DIRECT
comparison with arm C, which ablates the same feature in activation space.

Registered predictions, scored in the output:
  P1 A_feat >> A_pair (32x more score mass at L_0 = 32)
  P2 A_feat < C, because C also removes the feature from V and OV is how the
     trigger content is copied. If A_feat matches C, the QK path alone carries
     the behaviour -- a bigger result.
  P3 A_feat's collateral << C's: it perturbs rows where 1114 fires, where C
     corrupts what those positions ARE.
  P4 If A_feat is ~0 like A_pair, the implementation is wrong, not the
     hypothesis -- check the sum over mu is taken. The removed-L1 column is the
     direct check.

Run: python scripts/31_feature_score_ablation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from scripts import _three_arms_lib as lib
from sleeper.metrics import teacher_forced_sleeper_logp

LAM, MU = 1114, 1232
STRENGTHS = [0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0]
OUT = Path("results/feature_score_ablation.json")
ARMS = ["A_score_pair", "A_feat", "A_feat_sym", "B_paper_qk", "C_ln1_ablate"]


def build(arm, c, z, s):
    W_dec, W_Q, W_K, sc = c["W_dec"], c["W_Q"], c["W_K"], c["attn_scale"]
    if arm == "A_score_pair":
        return lib.score_pair_hooks(z, W_dec, W_Q, W_K, sc, LAM, MU, s)
    if arm == "A_feat":
        return lib.score_feature_hooks(z, W_dec, W_Q, W_K, sc, LAM, s, symmetric=False)
    if arm == "A_feat_sym":
        return lib.score_feature_hooks(z, W_dec, W_Q, W_K, sc, LAM, s, symmetric=True)
    if arm == "B_paper_qk":
        return lib.paper_qk_hooks(z, W_dec, W_Q, W_K, LAM, MU, s)
    return lib.ln1_ablate_hooks(z, W_dec, LAM, s)


def removed_l1(arm, c, z, s):
    """Total |score mass| removed, over causally valid cells. The P4 check."""
    W_dec, W_Q, W_K, sc = c["W_dec"], c["W_Q"], c["W_K"], c["attn_scale"]
    T = z.shape[1]
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool))
    if arm == "A_score_pair":
        g = ((W_dec[LAM] @ W_Q) * (W_dec[MU] @ W_K)).sum(-1) / sc
        d = s * torch.einsum("bq,bk,h->bhqk", z[:, :, LAM], z[:, :, MU], g)
    elif arm in ("A_feat", "A_feat_sym"):
        qv = W_dec[LAM] @ W_Q
        kv = W_dec[LAM] @ W_K
        fq = z[:, :, LAM]
        khat = lib._proj(z, W_dec, W_K)
        d = fq[:, None, :, None] * torch.einsum("he,bhte->bht", qv, khat)[:, :, None, :] / sc
        if arm == "A_feat_sym":
            qhat = lib._proj(z, W_dec, W_Q)
            d = d + torch.einsum("he,bhte->bht", kv, qhat)[:, :, :, None] * fq[:, None, None, :] / sc
            d = d - torch.einsum("bq,bk,h->bhqk", fq, fq, (qv * kv).sum(-1) / sc)
        d = d * s
    else:
        return float("nan")
    return float(d[:, :, causal].abs().sum())


@torch.no_grad()
def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    c = lib.setup()
    model, z, amask, is_dep = c["model"], c["z"], c["amask"], c["is_dep"]
    tokens = c["tokens"]
    z_dep = z[is_dep]
    base = c["base_sleeper"]

    fire = (z[:, :, LAM] > 0) & amask.bool()
    fire_mu = (z[:, :, MU] > 0) & amask.bool()
    print("=" * 100)
    print(f"lam={LAM} fires: dep {float((z[is_dep][:,:,LAM]>0).float().mean())*100:.2f}%  "
          f"clean {float((z[~is_dep][:,:,LAM]>0).float().mean())*100:.2f}%   "
          f"positions {int(fire.sum())} of {int(amask.sum())}")
    print(f"mu={MU}  fires: dep {float((z[is_dep][:,:,MU]>0).float().mean())*100:.2f}%  "
          f"clean {float((z[~is_dep][:,:,MU]>0).float().mean())*100:.2f}%")
    print(f"baseline sleeper logp = {base:.4f}")
    print()
    print("REMOVED SCORE MASS at scale 1 (the P4 check -- A_feat must be ~L_0 x A_pair):")
    r_pair = removed_l1("A_score_pair", c, z, 1.0)
    r_feat = removed_l1("A_feat", c, z, 1.0)
    r_sym = removed_l1("A_feat_sym", c, z, 1.0)
    print(f"  A_score_pair {r_pair:12.1f}")
    print(f"  A_feat       {r_feat:12.1f}   ratio to pair = {r_feat/r_pair:6.1f}x")
    print(f"  A_feat_sym   {r_sym:12.1f}   ratio to pair = {r_sym/r_pair:6.1f}x")

    rows = []
    for arm in ARMS:
        print(f"\n{'='*100}\n{arm}")
        print(f"  {'strength':>8s} | {'sleeper':>9s} {'suppress':>9s} | {'resid%':>7s} | "
              f"{'KL_dep':>10s} {'KL_clean':>10s} | {'removed_L1':>11s}")
        for s in STRENGTHS:
            with model.hooks(fwd_hooks=build(arm, c, z, s)):
                logits = model(tokens, attention_mask=amask)
                _, cache = model.run_with_cache(tokens, names_filter=[lib.RESID_POST],
                                                attention_mask=amask)
            sl = teacher_forced_sleeper_logp(
                model, model.tokenizer, tokens[is_dep],
                fwd_hooks=build(arm, c, z_dep, s), attention_mask=amask[is_dep]).mean().item()

            dres = (cache[lib.RESID_POST] - c["base_resid"]).abs().amax(-1)
            frac = float(((dres > 1e-5) & amask.bool()).sum() / amask.sum())
            lp = torch.log_softmax(logits, -1)
            kl = (c["base_logp"].exp() * (c["base_logp"] - lp)).sum(-1)
            kd = float(kl[c["nontarget"] & is_dep.unsqueeze(1)].mean())
            kc = float(kl[c["nontarget"] & (~is_dep).unsqueeze(1)].mean())
            rl = removed_l1(arm, c, z, s)

            rows.append({"arm": arm, "strength": s, "sleeper_logp": sl,
                         "suppression": base - sl, "resid_frac": frac,
                         "kl_dep": kd, "kl_clean": kc, "removed_l1": rl})
            print(f"  {s:8.2f} | {sl:9.3f} {base-sl:+9.3f} | {frac*100:6.2f}% | "
                  f"{kd:10.3e} {kc:10.3e} | {rl:11.1f}", flush=True)
            OUT.write_text(json.dumps({"lam": LAM, "mu": MU, "base_sleeper_logp": base,
                                       "removed_l1_at_1": {"pair": r_pair, "feat": r_feat,
                                                           "sym": r_sym},
                                       "positions_lam": int(fire.sum()),
                                       "positions_mu": int(fire_mu.sum()),
                                       "total_positions": int(amask.sum()),
                                       "rows": rows}, indent=2))

    # ── score the registered predictions ────────────────────────────────
    def at(arm, s):
        return next(r for r in rows if r["arm"] == arm and r["strength"] == s)

    print("\n" + "=" * 100)
    print("REGISTERED PREDICTIONS")
    print("=" * 100)
    for s in (1.0, 16.0):
        p, ft, sy, cc = (at("A_score_pair", s), at("A_feat", s),
                         at("A_feat_sym", s), at("C_ln1_ablate", s))
        print(f"\n  at strength {s}:")
        print(f"    P1  A_feat >> A_pair          : {ft['suppression']:+.3f} vs {p['suppression']:+.3f}"
              f"  -> {'HELD' if ft['suppression'] > 10*max(p['suppression'],0.01) else 'FAILED'}")
        print(f"    P2  A_feat <  C               : {ft['suppression']:+.3f} vs {cc['suppression']:+.3f}"
              f"  -> {'HELD' if ft['suppression'] < cc['suppression'] else 'FAILED (QK alone carries it)'}")
        print(f"    P3  A_feat collateral << C    : KLclean {ft['kl_clean']:.3e} vs {cc['kl_clean']:.3e}"
              f"  -> {'HELD' if ft['kl_clean'] < cc['kl_clean'] else 'FAILED'}")
        print(f"    P4  A_feat not ~0             : {'OK' if abs(ft['suppression'])>0.5 else 'CHECK THE SUM OVER MU'}")
        print(f"        A_feat_sym                : {sy['suppression']:+.3f}")
    print(f"\n  wrote {OUT}")


if __name__ == "__main__":
    main()
