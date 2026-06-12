# STRATEGY REFRAME (2026-06-12, PI): why the per-instance-selectivity wins keep deflating, and the two real paths to FRA value

## The diagnosis (PI)
At the LOWEST level of the feature hierarchy — single word ↔ single feature, idealized so the feature reconstructs
the word on its own — the FRA cell-cut and a plain token/column attention-mask are THE SAME operation. So a per-instance
selectivity comparison "FRA cell-cut vs token-mask on the same prompt" CANNOT show FRA value: they're identical. This is
exactly why EM, injection, and binding all deflated — every one was a single-word, per-instance-selectivity test, where
FRA degenerates to a token-mask (binding red-team literally found: the FRA-feature edit failed, the plain 2-column mask did
the work). The rigor was right; the WIN METRIC was wrong.

## Where FRA value actually lives (two paths)
The value of FRA at the single-word level is NOT per-instance selectivity. It is that the cell-cut is a WEIGHT-space,
ASSOCIATION-SPECIFIC, PERSISTENT edit (a support-gated rank-1 W_QK update) — it will _always_ remove that specific word
correlation, everywhere, unconditionally, while leaving the word's OTHER uses intact. So:

PATH 1 — PERSISTENT single-word associations. Find behaviours DRIVEN by a persistent single-word association, where the
  win is GENERALIZATION / UNCONDITIONALITY, not per-instance selectivity:
  - FRA = one W_QK weight edit removing the (A-content × B-content) cell -> removes A→B on ALL inputs (novel contexts,
    unseen positions, undetected occurrences) with NO inference-time detection, AND preserves A's other associations.
  - vs token-mask (activation, per-occurrence -> needs to DETECT A at inference; fails on novel/undetected occurrences),
    feature-ablation (kills A entirely -> loses A's benign uses), linear steer (broad A or B direction -> collateral).
  - WIN METRIC (FRA-specific, NOT re-deflatable): removal of A→B on HELD-OUT / novel-context occurrences (generalization)
    × preservation of A's benign uses (association-specificity) × no inference-time machinery (it's a weight edit).
  - Cleanest organism: a single-token BACKDOOR TRIGGER (safety; the banked in-context-backdoor result, reframed: cut the
    trigger→payload cell as a WEIGHT edit -> disarmed unconditionally on novel contexts vs a detector+mask that misses
    undetected triggers, while the trigger token keeps its benign uses). Alt: a spurious word-correlation / name→attribute.
  - SIDESTEPS the SAE-reconstruction obstacle: at single-word level the "feature" can be the TOKEN, so path 1 can use
    token-level QK cells (no lossy SAE), which is why induction/binding's SAE-feature failure does NOT block it.

PATH 2 — GENERALIZE UP THE HIERARCHY. Broad concept-features (fire on MANY words). Here a token-mask FUNDAMENTALLY cannot
  capture the concept (you'd have to mask every word that expresses it, including novel ones) but the FRA concept-feature
  cell-cut can. This is the broad×broad direction (synthetic PROVED it works via the regression-fitted multi-cell edit).
  - WIN METRIC: cut a (broad-concept × content) association -> removes the concept-link across ALL its lexical realizations
    (a token-mask can't enumerate them) while a linear steer on the concept direction bleeds (reuse). 
  - THE EMPIRICAL OBSTACLE (from induction R^2<0 + binding feature-edits-fail): SAE-feature reconstruction is LOSSY, so the
    broad concept-cell may not be cleanly cuttable via the current SAE basis. Path 2 must either (a) find a concept the SAE
    captures cleanly, or (b) use a cleaner decomposition (raw-activation QK attribution / better SAE / the regression-fit).

## Recommendation
PATH 1 FIRST: cleanest, FRA-specific win (persistence/generalization, not per-instance selectivity), sidesteps the SAE-loss
obstacle (token-level cell), safety-relevant (backdoor disarm), and leverages the banked in-context-backdoor result. Designed
so it CANNOT re-deflate: the metric is generalization-to-novel-occurrences + benign-use-preservation, which a token-mask
(needs detection) and feature-ablation (kills the word) provably cannot match — the value is the WEIGHT-PERSISTENT,
ASSOCIATION-SPECIFIC edit, FRA's actual formal definition. PATH 2 is the bigger prize but blocked on SAE reconstruction quality.

## Housekeeping
Circuits cron 62c23deb DELETED (circuits track resolved: IOI already-done/negative, induction faithfulness FAIL, binding win
deflated). The 3 deflations + faithfulness fail are the boundary map's negative half; the persistence/hierarchy reframe is the
positive program. Next: stand up the path-1 persistence experiment on PI go-ahead.
