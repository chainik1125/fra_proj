---
author: Indranil Das
date: 2026-09-18
tags:
  - results
  - theory
---

## The magnitude law behind FRA's collateral advantage -- what it predicts and what we measured

Research context: FRA paper theory section ([[fra-goal-show-it-works]]). This explains *why* FRA has
lower collateral than single-feature / DoM steering at matched removal, and turns the empirical result
([[gpt2_conjunction_removal_findings]]) into a mechanistic prediction.

### Statement

FRA cuts a single **cell** of the QK bilinear form: a (query-content feature x key-content feature)
pair. That edit changes attention scores *only* where that specific feature-pair co-occurs at a
query->key edge. A single-SAE-feature edit instead subtracts a residual-stream **direction** at every
position. So:

> **FRA's collateral on a text scales with how often the located CELL is present/used there; single-
> feature's collateral scales with the direction's global overlap. On text that does not use the cell,
> FRA is a no-op (zero collateral); the direction edit is not.** The advantage is therefore largest when
> the endpoint features are common (the direction is everywhere) but their gated co-occurrence -- the
> cell -- is rare. This is the "magnitude law": advantage ~ reuse(endpoint) / reuse(cell).

### Evidence (scripts/70, GPT-2 synthetic conjunction; corroborated by the direct KL runs)

Measured, per text, the located cell's mass at the output query position vs the top attribution
feature's activation (seed 0 shown; general-text result is structural and stable):

| text | cell present at output | interpretation |
|---|---:|---|
| target (A B) | 177.6 | the cell IS used -> FRA removes the payload here |
| reuse-A (A C) | 58.6 | partial (shared token) but points to the target payload P, not Q |
| reuse-B (D B) | 139.7 | partial (shared token B) but points to P, not S |
| **general English** | **0.000** | the cell's feature-pairs do not occur -> FRA is a NO-OP |

The **general-text cell-reuse = 0** is the clean, decisive case: it mechanistically predicts the
directly-measured **FRA-QK general-text collateral = 0.0000** (scripts/65, n=7). By contrast the single
feature's decoder direction has no such locality, giving the measured ~1.0-1.8 nats/token general-text
collateral. So the law's core prediction -- *cell absent on unrelated text => FRA collateral zero there*
-- is confirmed two independent ways (cell-reuse measurement + direct KL).

### Scope / honesty

- The **clean quantitative case is unrelated text** (cell-reuse -> 0, advantage -> infinity), which
  matches the zero measured FRA general collateral.
- On **endpoint-reusing text** (shares one endpoint token), the cell partially fires but points to the
  *target* payload, while the probe's own answer is a different payload -- so FRA still barely hurts it
  (low measured collateral) even though a naive "total cell mass" is nonzero. A single closed-form
  numeric ratio does not reduce cleanly here; the mechanism (locality of the cell edit) still explains
  the low collateral. We do not overclaim a one-line A on reuse text.
- feat1 is ADDITIVE (subtract the decoder direction scaled by residual norm), so its collateral comes
  from the direction, not from the feature's activation -- which is why single-feature damages general
  text even where the feature itself is inactive.

### Paper use

State the law as the mechanism, use general text as the clean confirmation (cell-reuse 0 -> FRA 0),
and present the endpoint-reuse advantage as measured (Pareto), explained by cell locality rather than a
closed-form ratio. Reproduce: `PYTHONPATH=. python scripts/70_magnitude_law.py` (CPU).
