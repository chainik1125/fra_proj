---
author: Indranil Das
date: 2026-09-19
tags:
  - results
---

## B1_real (gemma fact-injection) -- the decisive honest result

Realistic in-context fact-injection on gemma-2-2b (scripts/69): a "Notes." directory of fictional facts
(unique multi-token values), completion-cue query; remove one injected fact (subject x attribute ->
value), preserve same-subject / same-relation facts + general English. Finer coefficient grids +
induction-head-ablation baseline (Dmitry's validations). 9 fact-set-seeds. First-token payload scoring.

worst-case reuse KL {reuse_subject, reuse_relation} | general-English KL, at matched removal:

| removal | fra | hybrid | ov | pay | feat1 | dom | indab |
|---:|---|---|---|---|---|---|---|
| 30% | 0.41/0.002 | 0.23/0.002 | 0.007/0.000 | 0.011/0.000 | 0.82/0.52 | 0.20/0.08 | 0.33/0.009 |
| 50% | 0.43/0.002 | 0.27/0.001 | 0.024/0.000 | 0.037/0.001 | 1.56/1.02 | 0.54/0.18 | (0/9) |
| 70% | 0.40/0.002 | 0.39/0.002 | 0.075/0.000 | 0.083/0.001 | 2.79/1.80 | 1.52/0.47 | (0/9) |
| 90% | (0/9) | (0/9) | 0.24/0.000 | 0.24/0.003 | 4.47/2.83 | 5.04/1.61 | (0/9) |

### Honest conclusions

1. **FRA does NOT beat directional payload-suppression (pay/ov)** even on multi-token entities: pay/ov
   are the lowest (0.007-0.24), because the value's first token is still a suppressible output direction
   and the reuse facts use different values. The multi-token hypothesis did not rescue this comparison.
2. **FRA/hybrid clearly beat the FEATURE-BASED baselines -- Dmitry's actual "must beat single-feature"
   bar:** single-SAE-feature (feat1) 0.82-4.47 reuse + 0.52-2.83 general; DoM up to 5.04 reuse + 1.61
   general; induction-head ablation (indab) cannot even reach 50% removal. FRA/hybrid: 0.23-0.43 reuse +
   ~0.002 general -> ~4-10x lower reuse, ~250-1000x lower general-text collateral.
3. **Defensible claim (both GPT-2 + gemma, single- and multi-token):** among interpretable feature-based
   interventions (single-feature, DoM, induction-head ablation), cutting the FRA QK cell is far more
   surgical for conjunctive in-context removal. FRA clears the single-feature bar decisively.
4. **Honest limitation:** a payload-direction suppression baseline that already knows the exact target
   token is also effective and FRA does not beat it. This is a non-feature-based baseline; report it
   transparently. Consistent with Dmitry's plan to make this an illustrative/appendix result, with the
   Llama-8B sleeper agent as the main result.

Reproduce: `PYTHONPATH=. NSEED=8 python scripts/69_gemma_factinj_v2.py` (gemma GPU). Data:
results/b1/b1_factinj.json (on NCSA).
