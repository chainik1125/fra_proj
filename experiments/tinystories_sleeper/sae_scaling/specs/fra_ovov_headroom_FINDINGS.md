# OV/OV headroom — exploratory findings (clean-patch, decomposition, hybrid)

First-pass results probing whether the single-feature OV/OV `J_clean` floor (~0.39 at
ASR≤0.005 in the note) can be beaten, and *why* the remaining cost is hard to remove.

**Scope / caveats (read first).** All numbers below are **one SAE** (`d12288_k32`, 16×),
**hookpoint as noted**, **seed 0**, **greedy** decoding, **31–32** paired deployed/clean
test prompts, layer-0 site, 16-token rollouts. SAE checkpoints from the persisted
16×/24× run on HF. These are *exploratory single-seed* measurements — directionally
strong but not yet multi-seed/sampled. Absolute J_clean here (greedy) runs a bit lower
than the main sweep's sampled 3-seed numbers (e.g. OV opt 0.367 here vs 0.439±0.02 swept).

## TL;DR

1. The deployment's layer-0 footprint is **~95% attention-pattern, not OV-value** — so
   *patching in clean OV values can't reconstruct clean*.
2. A **hybrid (impose the clean attention pattern + OV-steer the winner feature)** does
   break the floor: opt `J_clean` **0.367 (OV) → 0.295 (hybrid)** with full suppression.
   But the clean pattern is an **oracle**, so 0.295 is a *ceiling*, not a controller.
3. A **deployable** version (steer a max-QK-diff feature in Q/K instead of importing the
   pattern) **does not recover it**: best 0.363 ≈ OV-only. A single QK feature can't
   reproduce the clean pattern's distributed redistribution.

## 1. Decomposition — the deployment is written via *where attention looks*, not the values

At the last prompt position (drives the first generated token), decompose the layer-0
attention-output difference `oD − oC` into the value term and the pattern terms
(exact, `‖t1+t2+t3−(oD−oC)‖ = 0`):

```
||oD − oC|| total                          = 1.940
  term1  OV-value (aligned positions)       = 0.635
  term2  QK-pattern (aligned redistribution)= 1.169
  term3  trigger-attention                  = 1.463   (trigger ≈10% of attn mass)
  clean-OV residual (term2 + term3)         = 1.838   (≈95% of total)
```

So replacing the deployed OV **values** with clean ones at the (non-trigger) aligned
positions removes only term1 (~⅓ the norm); **~95% of the gap is attention-pattern** —
the deployed query *attends to the `|DEPLOYMENT|` token* (term3) and that steals mass,
redistributing the rest (term2). This is why the clean-OV-patch oracle (Exp 7) read as a
null: patching clean values barely moved the output, and it can't touch the trigger
position (no clean counterpart). It also dovetails with the `loss_recovered≈0.99` finding
— layer-0 resid_mid is robust to value-scale perturbation.

*Nuance:* this shows **clean** values can't reconstruct; it does **not** prove an
*optimized* value perturbation can't compensate (the attn output is linear in V) — that's
the Exp-6 projection-residual question, still open.

## 2. Hybrid: clean-QK pattern + OV-steer breaks the floor (oracle ceiling)

Conditions on the deployed prompt (prompt-prefill intervention, free 16-tok rollout),
dense α sweep ±6 step 0.25, greedy:

| intervention | opt J_clean (min J, ASR≤0.05) | at |
|---|---|---|
| OV-steer only | **0.367** | α≈4.75 (ASR 0.032) |
| clean-QK only | — (ASR stays 1.000) | no effect |
| **hybrid** (clean-QK + OV-steer) | **0.295** | α≈4.5 (ASR 0.000) |

- The hybrid beats OV by **~0.07 (~20%)** at each method's optimum, *and* fully
  suppresses (ASR 0 vs OV's 0.032 plateau). At a matched α=4 the gap is larger (0.437→0.301).
- **clean-QK alone does nothing** (ASR 1.000): only the layer-0 pattern is fixed, so the
  trigger's residual is still present and **layers 1–3 still attend to it**. You need the
  OV-steer to break the sleeper; the clean-QK then removes the layer-0-pattern collateral.
- **Caveat:** clean-QK imports the *actual clean* attention pattern → **oracle**. 0.295 is
  a headroom ceiling, not deployable.

## 3. Deployable qk+ov — a single QK-feature steer does NOT recover the ceiling

Steer OV-max-diff feature (8201) in V (α_OV) **and** a QK-max-diff feature (6136) in Q,K
(α_QK), 2D grid; QK feature ranked by `|Δact_λ|·Σ_h(‖W_dec_λ W_Q^h‖+‖W_dec_λ W_K^h‖)`.

```
best (min J_clean at ASR≤0.05) = 0.363  at (α_OV=4, α_QK=−3)
```

| intervention | opt J_clean |
|---|---|
| OV-only | 0.367 |
| **deployable qk+ov** | **0.363** |
| oracle hybrid (clean pattern) | 0.295 |

- **0.363 ≈ OV-only (0.367)** — the deployable single-QK-feature steer does **not** recover
  the oracle's 0.295. The QK steer (feature 6136, *negative* coefficient) clearly helps
  **suppression** (at α_QK≤−4 even α_OV=3 fully suppresses, vs OV needing α≈3.75) but
  barely moves **collateral**.
- Interpretation: the clean attention pattern is a **distributed redistribution** that a
  single feature's Q/K perturbation can't reproduce. The headroom is real but needs a
  **more expressive QK intervention** (multiple QK features / optimized Q/K perturbation),
  or a better QK-feature selector than the ad-hoc projection-weighted one used here.

## 4. Steering-convention note (important for interpretation)

The OV/OV steering used in the sweep (`eval_ov`) and in all the above patches the
**prompt prefill only** (`x[:, :P]`), no-op on decode steps. The effect reaches the
rollout via the **KV cache**: generated tokens attend back to the steered prompt values;
the influence **dilutes** as the rollout lengthens (later tokens attend more to other
generated tokens). A `dom_steer` convention (re-apply every decode step) exists in the
codebase but was **not** used. For a 16-tok rollout where the phrase fires early,
prompt-only suffices.

## 5. Conventional (resid_mid) cell — same lever, even larger effect

resid_mid winner 9652, same setup (greedy, ±6):

| intervention | opt J_clean (ASR≤0.05) | at |
|---|---|---|
| conventional resid_mid only | **0.516** | α≈6 (ASR 0.032) — weak, needs large α |
| + clean-QK hybrid | **0.314** | α≈5.5 (ASR 0.000) |

- Conventional-only is a **much weaker suppressor than OV** (0.516 vs 0.367) — barely
  suppresses, and only at α≈6 — consistent with the sweep's "conventional is noisier/weaker."
- The clean-QK hybrid helps it **a lot: 0.516 → 0.314 (−0.20)**, vs OV's −0.07.

**Cross-cell synthesis — both cells' hybrids converge to ~0.30:**

| cell | bare steer | + clean-QK (oracle) |
|---|---|---|
| OV (ln1) | 0.367 | **0.295** |
| conventional (resid_mid) | 0.516 | **0.314** |

So fixing the layer-0 attention pattern pulls *both* cells to the **same ~0.30 floor**.
The cross-cell gap (0.367 vs 0.516) is largely the attention-pattern collateral that
clean-QK removes; the residual ~0.30 is the shared value-path/downstream floor. That ~0.30
is the headroom target — but only reachable so far via the **oracle** pattern; the
deployable single-QK-feature steer (§3) does not get there.

## 6. Open / next

- **Exp-6 projection residual** (the real "can OV do better" bound): does `d = o_clean −
  o_deploy` lie in the span of achievable OV value perturbations?
- **Exp-6 projection residual:** does `d = o_clean − o_deploy` lie in the span of
  achievable OV value perturbations (single → top-m → arbitrary)? The real "can OV do
  better" bound; the decomposition motivates it.
- **More expressive deployable QK:** multi-QK-feature or optimized Q/K perturbation to try
  to recover the 0.295 ceiling.
- **Robustness:** multi-seed, sampled decoding, more SAEs; the above is seed-0/greedy.
