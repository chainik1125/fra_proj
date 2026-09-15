---
author: Indranil Das
date: 2026-09-14
tags:
  - results
  - in-progress
---

## Semantic-filter transfer: what was done and what happened

Research context: defensive interpretability research (see [[research_context]]). Every number
is collateral (held-out KL, nats) or suppression fraction. Daily record: [[ladder_log]].

This is a report of one experiment line. It is not a paper draft and makes no claim beyond what
was measured.

## Setup

- Model: Gemma-2-2b (base). SAEs: GemmaScope 65k residual, on the top 25 induction heads.
- Machinery (head discovery, SAE encode, FRA tensor, DoM/conv/payload interventions) is Dmitry's
  `experiments/fra_win/jobs/g4_65k.py` at `iclr-summary`, spliced line-for-line. Env pinned to his
  versions (sae_lens 5.10.7, transformer_lens 2.18.0). Run on NCSA.
- Planted association: the prompt `<filler> The password is <X> <Y>. <filler> Remember the
  password: <X>` makes the model output `<Y>` at the end. X = trigger word, Y = payload.
- Intervention: FRA locates its feature pairs on ONE prompt where the query word is the planted X,
  takes the top 48 pairs per head, and subtracts those cells from the attention scores. The same
  cells are then applied to prompts where the query word is a *synonym* of X, and to prompts with
  different filler and X moved to a different position.
- Suppression = 1 - P(Y | edited) / P(Y | clean). Collateral = KL on separate legitimate text that
  uses the concept and the payload word. Read at matched suppression.
- Baselines (all measured; the "list-free" ones are not told which words are synonyms):
  token/position mask on the planted word; DoM steer at all positions and at planted-word positions,
  two contrasts each; conv-SAE steer, two contrasts; payload suppression; a position-mask oracle
  (knows exact positions) as a removal ceiling.

## What was run, in order

1. Feasibility (forward passes only): plant X->Y, query with a synonym, measure P(Y). Screened 12
   concepts; kept the 6 where >=2 synonyms produced P(Y) >= 0.20.
2. Rung 3 (2 concepts, 10 heads): does the cut transfer to synonyms.
3. Rung 3c (2 concepts, 25 heads): same, more heads.
4. Rung 4 (2 concepts, 25 heads): locate once, apply to 3 new contexts.
5. Rung 5 (6 concepts, 25 heads, 3 contexts): the broad run. [pending / see below]

## Results

### Feasibility (P(payload), mean over 8 filler seeds)

| concept | planted X | P(Y) planted | synonyms >= 0.20 | control mean |
|---|---|---:|---|---:|
| bird | bird | 0.90 | birds .37, sparrow, eagle | 0.01 |
| vessel | ship | 0.96 | ships, boat, vessel, yacht | 0.03 |
| fire | fire | 0.89 | flame, fires, burning | 0.03 |
| vehicle | car | 0.92 | van, bus, vehicle | 0.03 |
| medical | doctor | 0.89 | nurse, surgeon, medicine | 0.07 |
| war | army | 0.89 | soldier, military | 0.06 |

Dropped (synonyms below 0.20): money, music, weather, food, school, plant.

### Rung 3 / 3c / 4 (vessel, vehicle)

- Transfer, 10 heads: cut located on ship/car suppressed synonyms 5-94%; token mask 0% on every
  synonym. FRA reached 30% suppression on 5/7 synonyms.
- Transfer, 25 heads: suppression on synonyms 90-100%; FRA reached every level on all 7. On each of
  the 9 queries FRA's collateral was lower than the best list-free baseline: geometric-mean ratio
  3.8x @30%, 6.8x @50%, 7.9x @70%.
- Persistence, 25 heads (locate once, 3 new contexts, 20 synonym-queries): FRA reached 30% on 20/20,
  50% on 19/20, 70% on 16/20. Collateral lower than the best list-free baseline on 19/20, 19/19,
  16/16. Geometric-mean ratio 4.3x / 5.6x / 7.9x. The context with X near the end weakened FRA on
  vessel synonyms (0.43-0.84) while the oracle did not (0.88-0.99).

Comparison point: Dmitry's Setting 11 (GPT-2 + flat SAE) had a located cell transfer at 0.005-0.014
against an oracle of 0.79-0.98. Here it transferred at 0.43-1.00 against an oracle of 0.76-0.99.

### Rung 5 (6 concepts, broad run)

<!-- FILLED IN WHEN R5_broad LANDS: per-concept reach and FRA-vs-best-list-free collateral ratio at
30/50/70%, with war and medical reported separately since their synonyms are not near-spellings. -->
_Pending._

## What is and is not established

Established, on base Gemma-2-2b, for the planted password-recall association:
- A cut located from a single trigger word removes the association when the query uses a synonym the
  method never saw, and when the prompt filler and trigger position change.
- On every query where FRA reached the target suppression, its collateral was lower than every
  baseline not given the synonym list, by a few-fold.
- A token/position mask on the planted word removes nothing on synonyms.

Not established:
- The effect needs ~25 heads; with 10 it largely fails (so does the position oracle).
- Transfer is not fully position-invariant (X-near-end case).
- DoM steered at the planted word does transfer to synonyms (81-99% removal); FRA's advantage is
  collateral, not removal ability. This is a Pareto result, not "only FRA can do it".
- Only base Gemma-2-2b. The instruction-tuned model did not learn the poisoned-demo behaviour to
  begin with (rung 2e), so nothing there to remove. Not tested on other models.
- The association is planted, not naturally occurring or trained. This is a controlled stand-in for
  a poisoned-context backdoor.

## Relation to the task

The task is to find a real-world case where FRA beats baselines, by adding complexity to the
in-context backdoor. This is a step on that path: a planted concept-level association on a real
model, with FRA Pareto-dominant against fair baselines. The remaining gap to the task as stated is
that the association is planted rather than naturally occurring or trained into the model.
