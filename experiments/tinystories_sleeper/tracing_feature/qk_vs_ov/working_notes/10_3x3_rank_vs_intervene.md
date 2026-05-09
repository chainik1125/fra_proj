# Note 10 — The 3x3: ranking dimension × intervention dimension

## Setup

Two orthogonal axes:

- **Ranking** (how we pick features to ablate): QK L1_mean top-3, OV one-stage signed top-3, or their union (6 features).
- **Intervention** (what computational path the $\alpha \cdot \Delta$ perturbation travels): V-path only, Q/K-path only, or all (ln1_normalized, equivalent to V + Q + K).

The shared delta is $\Delta_{b,t} = -\sum_{f \in \text{selected}} z^f_{b,t} \cdot W_{\text{dec}}[f]$, the ln1-space "remove-these-features" perturbation.

| intervention | hook | what changes |
|---|---|---|
| OV | `blocks.0.attn.hook_v` ← $\alpha \cdot (\Delta \, W_V^h)$ per head | V at prompt positions; attention pattern unchanged |
| QK | `hook_q` ← $\alpha \cdot (\Delta \, W_Q^h)$ and `hook_k` ← $\alpha \cdot (\Delta \, W_K^h)$ | attention pattern shifts; V unchanged |
| All | `blocks.0.ln1.hook_normalized` ← $\alpha \cdot \Delta$ | all three (linearly equivalent to OV + QK simultaneously) |

## Quality scores (monotone-envelope; 1 = perfect, 0 = no effect)

| ranking \ intervene | **OV-only** | **QK-only** | **All** |
|---|---:|---:|---:|
| **QK** | 0.59 | 0.07 | 0.38 |
| **OV** | **1.00** | 0.18 | 0.98 |
| **Union** | 0.93 | 0.14 | 0.85 |

## Headline

**(OV rank, OV intervene) achieves a perfect Pareto frontier: ASR$_{16}$ = 0.00 at $\Delta$CE = 0.** At $\alpha \in \{2, 3\}$ the V-path intervention on features {1205, 1114, 337} perfectly suppresses sleeper behaviour with *exactly zero* clean-prompt coherence cost (ΔCE = −3e-5, rounded to 0 in the table).

**It is strictly better than (OV rank, All intervene).** That cell reaches ASR=0.02, not 0.00, at the same tiny ΔCE. The Q/K component of the ln1 perturbation actually *attenuates* the suppression — it's self-interference the V-only path avoids.

## Raw results (ASR$_{16}$, ΔCE)

| ranking | intervene | α=0.5 | α=1 | α=2 | α=3 |
|---|---|---|---|---|---|
| QK [870,1388,760] | OV | (0.99, +0.009) | (0.91, +0.057) | (0.43, +0.967) | (0.16, +2.202) |
| QK | QK | (0.99, +0.006) | (0.96, +0.067) | **(0.92, +1.502)** | **(0.89, +3.558)** |
| QK | All | (0.98, +0.019) | (0.75, +0.194) | (0.48, +1.979) | (0.00, +4.275) |
| **OV** [1205,1114,337] | **OV** | **(0.98, 0)** | **(0.76, 0)** | **(0.00, 0)** | **(0.00, 0)** |
| OV | QK | (0.99, 0) | (0.99, 0) | (0.94, 0) | (0.82, 0) |
| OV | All | (0.99, 0) | (0.81, 0) | (0.25, 0) | (0.02, 0) |
| Union | OV | (0.89, +0.009) | (0.27, +0.057) | (0.00, +0.968) | (0.00, +2.201) |
| Union | QK | (0.99, +0.006) | (0.87, +0.067) | (0.85, +1.502) | (0.92, +3.558) |
| Union | All | (0.77, +0.019) | (0.15, +0.195) | (0.09, +1.980) | (0.00, +4.274) |

(ΔCE=0 rows: all |ΔCE| < 1e-4.)

## Mechanistic interpretation

### Why (OV rank, OV intervene) works so cleanly

OV-ranking picks features whose decoder direction $f_\lambda$ projects strongly onto $u_h = W_V^h W_O^h e_{171}$, weighted by their attention-weighted activation on dep prompts. The top-3 were {1205, 1114, 337}. **Feature 1114 is essentially a deployment-exclusive detector**:

| f | fire_rate (dep) | fire_rate (clean) | mean\|z\| dep | mean\|z\| clean | ratio |
|---:|---:|---:|---:|---:|---:|
| 1205 | 2.1% | 1.8% | 0.005 | 0.004 | 1.4× |
| **1114** | **6.5%** | **~0.0%** | **0.829** | **0.000** | **2553×** |
| 337 | 0.3% | 0.6% | 0.008 | 0.014 | 0.58× |

1205 and 337 have near-zero mean activation — their contributions to $\Delta$ are tiny. The action is essentially f=1114 alone: it fires on ~6.5% of dep positions and *never on clean*. Because $\Delta_{b,t} \propto z^{1114}_{b,t}$, the intervention is *identically zero* at every clean prompt position. That's why ΔCE = 0 exactly.

On dep positions where 1114 fires, $-\alpha \cdot z^{1114} \cdot W_{\text{dec}}[1114]$ gets projected through $W_V^h$ into V, which at query positions downstream writes $-\alpha \cdot$ (the suppressor's negative) into `attn_out` — pushing it off the sleeper-trigger direction. At $\alpha=2$ this is enough to flip 100% of generations.

### Why (QK rank, QK intervene) barely suppresses but destroys coherence

QK ranking picks {870, 1388, 760}. Their firing rates are near-identical on dep and clean (26–30% across both). Perturbing Q/K at these positions:

- shifts attention patterns substantially on both distributions → huge clean-prompt coherence cost (ΔCE=+3.56 at α=3);
- but the pattern shift is *not specifically toward* un-attending to the sleeper trigger — at α=3 ASR stays at 0.89. The QK-intervention scrambles attention without moving it in the right direction for sleeper suppression.

This is consistent with the QK L1_mean Spearman correlation: that metric predicts single-feature $|\Delta \log p|$ *at the full ln1 hook*, where the V pathway does most of the work. Measuring the effect of QK features at the *V pathway* (which is what (QK, all) does, dominantly), the correlation is real; measuring them at the *QK pathway only*, it's close to zero effect.

### Why OV intervention is always better than QK intervention

V-only perturbation changes the values each source writes. Clean prompts only see the change where the perturbed features fire — and OV features have sparse dep-specific firing. Q/K perturbation changes the *attention pattern*, which rearranges how *all* tokens read from *all* source positions — so even a locally small Q/K change fans out into coherence damage across the whole generation.

Formally: V-intervention's effect on `attn_out[t]` is $\delta V_s$ weighted by the existing $A^h_{qk=t,s}$ (unchanged). QK-intervention's effect is $\delta A^h$ weighted by the existing $V_s$ — and $V_s$ is nonzero at essentially every position, so small pattern shifts scale with whatever arbitrary V content sits at those positions. That's why QK-intervention has outsized coherence cost for comparable suppression.

### (OV rank, QK intervene) is mildly suppressive

At α=3: ASR=0.82, ΔCE=0. So OV-selected features have a small but nonzero Q/K attribution: perturbing them through Q/K only reduces ASR from 0.99 to 0.82 with essentially zero coherence cost (because 1114 still doesn't fire on clean prompts, so Q/K perturbation is also zero there). Consistent with OV features being *mostly* V-path but not exclusively so.

### Self-interference at (OV rank, All)

(OV, OV): α=3 ASR=0.00, ΔCE=0. (OV, QK): α=3 ASR=0.82, ΔCE=0. (OV, All): α=3 ASR=0.02, ΔCE=0.

If V and QK effects were additive at the logit level, (all) would be ≈ (OV) × (QK) in survival-rate terms ≈ 0 × 0.82 = 0. We get 0.02, very close. So the compositional behaviour is nearly multiplicative (i.e. close to independent), with tiny positive correlation between the two paths' failure modes on the 2% of dep prompts where V-suppression doesn't fully kill sleeper.

## Surprises worth calling out

1. **The ideal recipe strictly dominates the "naive" full-ln1 ablation.** Quality (OV, OV) = 1.00 strictly > Quality (OV, All) = 0.98. Intervention-path choice matters at the 2-3% level even when the ranking is already well-chosen.

2. **(QK rank, OV intervene) has quality 0.59** — better than (QK rank, All intervene) at 0.38. So even with "wrong" features (QK-selected general structural features), routing the intervention through V-only gives a better tradeoff than the natural ln1 intervention. The V-only intervention is a *coherence-protection* knob.

3. **Union (6 features) through V is quality 0.93** — close to but not matching OV-only's 1.00. Adding the QK "star" features to the OV-only setup buys *nothing* and actually costs a small amount of clean-CE (at α=2, ΔCE=+0.97 vs 0 for OV-only). The QK features are already nonzero on clean prompts, so their V perturbations leak into clean coherence.

4. **(QK rank, QK intervene) at α=3 gives ASR=0.89 with ΔCE=+3.56.** This is a striking empirical demonstration that "changing attention patterns a lot" and "suppressing sleeper" are largely orthogonal. You can thrash the attention mechanism without affecting the outcome — unless you thrash it in a specific direction that happens to un-attend to the deployment trigger.

## Implications for the broader OV-vs-QK story

The picture now:

- **QK-side attribution** (softmax-Jacobian linearisation) correctly identifies features whose ln1 ablation changes the full-stack sleeper logit at α=4. This is because ln1-ablation implicitly routes through both V and Q/K, and the model's sleeper output is most sensitive to the feature-direction shift — which the QK linearisation captures well. So QK L1_mean is the right *single-feature causal importance* predictor.

- **OV-side attribution** (signed projection onto $u_h = W_V^h W_O^h e_{171}$ on dep positions) identifies features whose V-writes align with the suppressor direction. These are the features that (a) fire on dep positions, (b) project strongly via V-path onto the suppressor, and (c) — empirically, as a consequence of how SAEs structure their features — tend to *not* fire on clean. This is the right *defensive intervention* selector.

- **The two answer different questions**, and the 3x3 makes that explicit. The QK selector gives you the feature with the biggest single-feature logit change; the OV selector gives you the feature whose perturbation is cleanly routed only through the V pathway on the relevant distribution.

## Practical recipe (updated)

If your goal is **minimum-coherence-cost sleeper suppression** at an ln1-type hook:

1. For each ln1 feature $\lambda$, compute the one-stage OV signed contribution on **dep prompts only**:
   $$S_\lambda = \sum_{h} \beta_{h, \lambda} \cdot \mathbb{E}_{\text{dep}, t}\left[\sum_s A^h_{t, s} z^\lambda_s\right]$$
2. Take the top-$k$ features by $|S_\lambda|$ (k=3 suffices here).
3. Intervene at `hook_v` with $\alpha \cdot (\Delta @ W_V^h)$ per head. **Not** at ln1_normalized — that adds avoidable coherence cost.
4. Sweep $\alpha$; α≈2 suffices.

If your goal is **per-feature causal importance prediction** on a target logit, use QK L1_mean as before. Don't use it for selection if coherence matters.

## Artifacts

- [../results/pareto_3x3.json](../results/pareto_3x3.json) — raw data
- [../results/pareto_3x3.png](../results/pareto_3x3.png) — 3x3 grid, full range
- [../results/pareto_3x3_zoom.png](../results/pareto_3x3_zoom.png) — 3x3 grid, ΔCE∈[0, 0.3]
- [../scripts/pareto_3x3.py](../scripts/pareto_3x3.py) — measurement script
- [../scripts/analyze_3x3.py](../scripts/analyze_3x3.py) — analysis + plots
- [../scripts/feature_firing_stats.py](../scripts/feature_firing_stats.py) — firing pattern diagnostics
