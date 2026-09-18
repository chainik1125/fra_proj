---
author: Indranil Das
date: 2026-09-18
tags:
  - roadmap
  - summary
---

## Roadmap Sep 17-18: from "single-feature ties us" to a defensible FRA win

Everything below is on branch `indranil/fra-toy`. This is the story arc -- where we failed, what we did
next, and the strong results.

### 0. Start point (Dmitry's feedback, Sep 16 call)
The plain **semantic filter is NOT a standalone win**: on a single-concept task, single-SAE-feature
*additive* steering ties/beats FRA (his corrected eval). FRA only wins on a **conjunction** -- a behaviour
gated by two individually-common features, where single-feature must damage a whole endpoint and only the
cell is surgical. -> pivot to building a conjunction.

### 1. Building the conjunction (screens)
- **Failed:** naive constructions. (i) cross-concept copy (guard->vault): P~0.00. (ii) single compound
  key: leaks via the payload-adjacent token. GPT-2 few-shot list format: pair only ~0.08.
- **Worked (gemma):** password-recall format + **ambiguous bigrams** ("The password for red fox is nine;
  ... blue fox is three; ... red owl is seven"). Pair fires the payload **0.60-0.68**, either token with a
  novel partner **0.03-0.08** -> a genuine AND. (scripts/54-56.)

### 2. Controlled-conjunction removal
- **First gemma run (scripts/57):** FRA lower collateral but NOT Pareto (~2.3x, limited reach) -- exactly
  what Dmitry saw.
- **His OV suggestion + a metric fix (scripts/58):** added OV-path steering, the QK+OV hybrid, AND a
  payload-elsewhere collateral set. **Gemma result: FRA-family (QK cell / hybrid / OV) beats
  single-feature and DoM by 4-9x, and global payload-suppress by ~10-40x, at matched removal.**

### 3. NCSA access broke -> went fully local on GPT-2 (CPU), unblocked
Windows ssh multiplexing died (home AND eduroam) -- not fixable client-side. So I reproduced the whole
pipeline on GPT-2-small locally:
- Reproduced the PoC (FRA 0.054 vs trigger 5.65 vs payload 4.12).
- Found GPT-2 does a **synthetic** bigram conjunction (pair 0.67 vs 0.05).
- **GPT-2 full removal (scripts/62):** FRA-family **10x** lower worst-case collateral than single-feature.
- **Resolved the single-token caveat (scripts/65):** with a common payload + a **general-English**
  collateral probe, **FRA-QK causes exactly 0.0000 general-text collateral**, while single-feature/DoM
  inflict ~1-1.8 nats/token (**~1000x**). FRA-family Pareto-dominates on BOTH axes.

### 4. Theory: the magnitude law (scripts/70)
FRA cuts a query-content x key-content **cell** (local); single-feature subtracts a residual **direction**
(global). On general text the located cell's mass = **0** -> mechanistically predicts FRA-QK's measured
zero general-text collateral. Advantage ~ reuse(endpoint) / reuse(cell).

### 5. STRONG result: hookpoint sweep (scripts/71)
Dmitry: "OK result = FRA beats an SAE at that hookpoint; STRONG = FRA does something an SAE at NO
hookpoint can do." Swept the single-feature baseline across **all 11 layer hookpoints**, took the best:
**FRA beats the best-hookpoint single feature by 6.4x** at matched removal (best single feature over all
layers 0.57 vs FRA 0.088). No hookpoint rescues the baseline. -> the strong result.

### 6. Boundary map (honest -- where FRA loses)
Weight-baked association (out of attention: FRA AND the attention oracle = 0.00; DoM wins),
single-concept tasks (single-feature ties), directions/personas, subliminal traits. FRA is the *right*
tool for conjunctive, attention-routed, in-context removal -- predictable in advance.

### 7. B1_real: realistic in-context fact-injection (scripts/67-69)
The "more real world" version: a poisoned fact directory (fictional entities), answer gated by
(subject x attribute). Screen confirms gemma does the AND (Nile 0.81 vs 0.01). Removal (scripts/69,
completion-cue) is READY; my NCSA copies are stuck in the queue -- **Dmitry to run** (RUN_B1_real.md).

### Where it stands
Core claim proven on two models + strong hookpoint result + theory + honest boundary. Remaining: B1_real
number (queued / Dmitry) and paper writing (abstract due today).

### Links (branch `indranil/fra-toy`)
- Proposals + run guides: `proposals/experiments_to_run/` (README, B1_controlled_conjunction, B1_real_conjunction, B4, RUN_B1, RUN_B1_real, this file)
- Findings: `docs/insen/gpt2_conjunction_removal_findings.md`, `hookpoint_sweep_findings.md`, `magnitude_law_findings.md`, `boundary_map.md`, `ladder_log.md`
- Figures: `results/b1_gpt2/b1_gpt2_pareto.png`, `results/b1_gpt2/b1_gpt2_general.png`
- Scripts: `scripts/56` (conjunction screen), `58` (gemma controlled removal), `62`/`65` (GPT-2 removal), `69` (B1_real), `70` (magnitude law), `71` (hookpoint sweep)
