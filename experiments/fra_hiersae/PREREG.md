# PRE-REGISTRATION (PI, 2026-06-12): naive hierarchical-SAE + FRA should FAIL the drift problem (the higher-hierarchy-cells problem)

## CONTEXT
The flat-SAE persistence result (fra_persistence): the SAE q-feature for a word DRIFTS across contexts (top1_coverage 0.32,
q-side culprit, n_cells_for_90=8), so a single FRA cell located in one context does NOT transfer (rem_holdout 0.005) -> FRA is
association-specific (collat ~1995x < feature-ablation) but NOT position-invariant. HOPE: a hierarchical SAE gives a STABLE
COARSE concept feature -> one cell, position-invariant AND specific (the empty corner of the 2x2).

## THE PRE-REGISTERED PREDICTION
NAIVE application of a hierarchical SAE (e.g. Matryoshka) + STANDARD FRA will FAIL to cleanly solve drift/persistence (it MAY
partially help -- reduce drift, tighten the union -- but will NOT reach a clean position-invariant + association-specific
SINGLE-cell cut), because using the hierarchy effectively REQUIRES a solution to the HIGHER-HIERARCHY-CELLS PROBLEM.

## THE HIGHER-HIERARCHY-CELLS PROBLEM (the mechanism)
Hierarchical-SAE features are NESTED & NON-ORTHOGONAL (the coarse concept feature is a parent/prefix of the fine context-
specific features; both fire for the same token). So the FRA cell decomposition S=Σ u^μ_q u^ν_k ω_{μν} spreads the association's
QK score across MULTIPLE levels at once: coarse×coarse, coarse×fine, fine×fine cells that OVERLAP & DOUBLE-COUNT. Therefore:
 (i) no principled NAIVE attribution of the score to the right level (which cell to cut);
 (ii) cutting the stable coarse-concept cell ALONE leaves RESIDUAL (the score routes through the entangled fine/cross-level cells);
 (iii) full removal needs coarse + fine -> back to a UNION (maybe more stable/bounded than flat-drift = the "might help", but NOT a clean single-cell win).
Using the hierarchy effectively requires a NESTED-CELL-AWARE attribution (attribute to the COARSEST SUFFICIENT level) -- the real research problem.

## CONFIRM / FALSIFY (locked before running)
- CONFIRM (prediction holds): vs flat SAE, the hierarchical SAE IMPROVES consistency SOMEWHAT (top1_coverage up from 0.32, drift
  reduced) BUT (a) cutting the coarse-concept cell ALONE does NOT cleanly remove the planted association (residual >= ~0.3 via fine/
  cross-level cells); (b) the QK score / removal is SPLIT across hierarchy levels (coarse cell carries < ~0.6 of the causal effect);
  (c) no clean single-cell position-invariant+specific win; a bounded union is still required.
- FALSIFY (prediction WRONG, naive works): the coarse-concept cell IS the stable carrier -- cutting it ALONE removes the planted
  association at ALL positions (>=70%) with low collateral on the word's other uses, top1_coverage jumps high (>=0.8), clean single-cell win.
- HELP-BUT-NOT-SOLVE band (the predicted outcome): improvement over flat but neither a clean win nor no-effect -> quantify HOW MUCH it
  helps (coverage, residual) and DEMONSTRATE the level-spreading -> motivates the nested-cell-aware attribution solution.

## THE SYNTHETIC TEST (ground-truth, the right speculative-first setting)
Extend synth_hier2's PLANTED hierarchy (concept -> alpha-mixed leaf features). Generate activations; TRAIN a flat SAE AND a
hierarchical/Matryoshka SAE; run FRA QK decomposition on each; with GROUND-TRUTH (we know the planted concept + leaves + the planted
association), measure: (1) flat-SAE drift (reproduce the gpt2 finding with known answer); (2) does the hierarchical SAE's coarse cell
recover the planted concept-association CLEANLY, or does the score spread across levels (the higher-hierarchy-cells problem)?; (3) the
3 axes (detection: cell-correlation -> planted concept-graph; localization: FRA-rank -> planted union; control: coarse-cut -> removal +
specificity). Sweep alpha (hierarchy tightness) + superposition. Reuse multitrigger_sleeper/cloud/sae_models.py for SAE training.
