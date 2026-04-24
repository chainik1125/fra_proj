# Note 07 — Next steps

Things we should test to stress-test the QK-dominates-OV conclusion and to
extend the story:

## Immediate (happening now in parallel)

### (a) Ablate more top-QK-concentration features

From `qk_concentration.py` output, the top-20 $\mu$'s by `max over (b, q)` concentration are:

```
254, 822, 760, 1298, 870 (tested), 784, 1279, 435, 1102, 1489,
1008, 303, 200, 842, 500, 262, 608, 691, 1388 (tested), 59
```

Untested: 254, 822, 760, 1298, 784, 1279, 435, 1102, 1489, 1008, 303, 200, 842, 500, 262, 608, 691, 59 (18 features).

Hypothesis: if the QK concentration correlates with ablation impact, most of these should give substantial $|\Delta \log p|$ at $\alpha=4$. If only 870 and 1388 give large effects while the others don't, we've over-fit the concentration story to two specific features.

Script: [../../scripts/ln1_feature_ablation.py](../../scripts/ln1_feature_ablation.py) with an extended feature list.

### (b) Per-pair $(\mu, \nu)$ attribution

Expand the top query-features into their dominant key-feature partners. Two potential structures:

- Each top-$\mu$ has a **single dominant $\nu$ partner**. This would mean the QK circuit is essentially single-pair-per-query, which is a clean story.
- Each top-$\mu$ has **multiple small $\nu$ partners** that add up. Would mean the QK circuit is more distributed on the key side.

Aggregate per $(\mu, \nu)$: signed mean over $(b, q, k)$ with $k$ restricted to prompt source positions.

Script: new `qk_pair_concentration.py` — compute top-$\nu$ per top-$\mu$.

## Future (not starting yet)

### (c) Full QK tensor with exact ablation (not first-order)

The current QK formula uses softmax Jacobian linearisation. For large ablations, nonlinear corrections matter. Test: run the model with each single-feature ablation at `hook_q` only (leaving $K, V$ alone), measure the actual pattern and target shift, compare to the first-order prediction.

### (d) Symmetric key-side ranking

The per-$\nu$ key-side sensitivity formula (Note 05):
$$
\delta T_q \approx -u^{\nu^\star}_{k^\star} \sum_h \eta^{h, \nu^\star}_q \tilde g^h_{q, k^\star}
$$
depends on the specific position $k^\star$ of the key feature. Aggregate it by: for each $\nu$, find the source position $k^\star(b)$ where it fires strongest on each deployment prompt, compute the sensitivity there, aggregate.

### (e) Generalise beyond this SAE

The QK-dominates-OV pattern is a claim about feature ablation at the `ln1.hook_normalized` hook specifically. At hooks where the feature is only read by OV (e.g. at `hook_v` directly, or `resid_post_3` if we were ablating near the output), the OV-side attribution should dominate. Can test by repeating the analysis with ablations at a different hook.

### (f) Different downstream target

Our target is the scalar $\langle \mathrm{attn\_out}_q, e_{171}\rangle$. Repeat for a different target direction — say $e$ from a feature at a later layer — and see if the same QK > OV pattern holds. Likely yes for any single-layer target, but worth confirming.

## Meta: what we'd want to write up if this pattern generalises

A short note claiming:

> When ablating an SAE feature at a hook that feeds into both Q/K/V, the
> relevant sensitivity is not the feature's OV attribution to the target but
> its QK-side first-order derivative (softmax-Jacobian linearised, OV-side
> profile-weighted).

With the formulas from Note 05 and the Spearman correlation from Note 06 as
supporting evidence.

This isn't a deep mathematical claim — it follows directly from the chain rule
through the softmax. But it matters in practice because the community defaults
to OV-side attribution (that's what's cleanest algebraically), and this note
shows that for the common ablation intervention used by ablation sweeps,
that default is misleading.
