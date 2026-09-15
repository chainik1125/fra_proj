---
author: Indranil Das
date: 2026-09-14
tags:
  - results
  - in-progress
---

## FRA as a semantic filter: a real-model win for the QK cell edit

Research context: defensive interpretability research for a paper on Feature-Resolved
Attention with Dmitry Manning-Coe (see [[research_context]]). Every number below measures how
well a method *removes* a planted benign association and how much unrelated behaviour it
damages. The full daily record is in [[ladder_log]]; this note is the standalone writeup of the
one result worth taking forward.

## The claim

On Gemma-2-2b, an FRA QK cell-cut located from a **single** planted word-association removes that
association across **synonyms it never saw and prompt positions it was not located on**, at
several times lower collateral than any baseline that is not handed the list of synonyms in
advance. A token/position mask keyed on the planted word removes **nothing** on the synonyms.

This is the "semantic filter" idea from the 2026-09-12 meeting, realised on a real model: one cut
disarms every surface form of a concept, not one wording at a time.

## Why this is the right test for FRA

FRA's mathematically unique object is the bilinear QK cell — an interaction between a
query-content feature and a key-content feature. The paper's thesis is that a linear steer can
imitate a *row or column* of that form but never a single *cell*, so FRA should win exactly when
the target is an **association between two pieces of content**, both of which are otherwise
benign and reused. A concept-level in-context association is precisely that: the concept feature
(vessel, vehicle, ...) and the payload are each common in ordinary text; only their planted
pairing is unwanted.

The magnitude law A ≈ reuse(marginal) / reuse(conjunction) predicts the win, and predicted the
earlier failures too — rung 2 cut a *common* pair (the "read a demonstration label" edge) and
paid ~12x the collateral of rung 1's rare pair.

## What was run

- Model Gemma-2-2b, GemmaScope 65k residual SAEs, the top 25 induction heads (head count matters:
  see the reach note below). Machinery spliced line-for-line from Dmitry's `g4_65k.py`.
- **Locate once.** FRA finds its feature pairs on ONE prompt with the planted word ("The password
  is ship anchor. ... Remember the password: ship") and takes the top 48 pairs per head.
- **Apply everywhere.** The same content-addressed cells are subtracted from the attention scores
  on prompts that query with a *synonym* ("... Remember the password: vessel"), in 3 new contexts
  with different filler text and the planted pair moved to the start / middle / end.
- **Concepts (6),** selected from a 12-concept feasibility screen by whether >=2 synonyms actually
  clear a 20% removal threshold: vessel, vehicle, bird, fire, war (army->soldier/military/troops),
  medical (doctor->nurse/surgeon/medicine). The last two are the strongest test: their synonyms are
  not near-spellings of the trigger, so transfer is semantic, not sub-word.
- **Baselines, all measured at matched suppression, collateral = held-out KL on legitimate concept
  text.** The fair ones are *list-free* — not told which words are synonyms:
  - token/position mask keyed on the planted word;
  - DoM (difference-of-means steer) at all positions, and at the planted-word positions, each under
    two contrasts;
  - conv-SAE feature steer, both contrasts;
  - payload suppression at the output;
  - a position-mask ORACLE (knows the exact positions) as a removal ceiling.

## Results

### Two concepts, full validation (rungs 3-4)

| step | finding |
|---|---|
| transfer | cut located on 'ship'/'car' removes 57-100% on unseen synonyms; token mask 0% |
| reach | 10 heads capped at ~30% (the oracle capped too); 25 heads -> 90-100% |
| collateral | FRA lower than every list-free baseline on 9/9 queries: 3.8x @30%, 6.8x @50%, 7.9x @70% |
| persistence | located once, applied to 3 new contexts: 19/20 synonyms lower collateral; 4.3-7.9x |

The persistence result is notable because it is the failure mode of his Setting 11 (GPT-2 + flat
SAE), where a located cell transferred at 0.005-0.014 against an oracle of 0.79-0.98. Here the same
cut transfers at 0.43-1.00 against an oracle of 0.76-0.99.

### Six concepts, broad run (rung 5)

<!-- FILLED IN WHEN R5_broad LANDS -->
_pending: 6 concepts x ~10 synonym-context queries. Table of reach and FRA-vs-best-list-free
advantage at 30/50/70%, plus the semantically-hard concepts (war, medical) called out separately._

## Honest limitations

- **Head count is load-bearing.** With 10 heads both reach and persistence largely fail; the effect
  needs the top ~25 induction heads. This is a property of where the circuit lives, not of FRA (the
  position oracle capped at the same place with 10 heads), but it must be stated.
- **Position dependence remains.** The context with the planted pair near the *end* weakens FRA on
  some synonyms while the oracle does not, so transfer is not perfectly position-invariant.
- **DoM at the planted word is a real baseline and does transfer** (81-99% removal): steering the
  stored association breaks retrieval whatever word queries it. FRA wins on *collateral*, not on
  removal ability. The claim is Pareto, not "only FRA can do this".
- **Two-model scope.** Base Gemma-2-2b only. The behaviour does not exist to remove on the
  instruction-tuned model (rung 2e), and nothing here is tested on GPT-2 or larger models.
- **Not a natural backdoor.** This is a planted password-recall association, a clean stand-in for a
  poisoned-context backdoor, not a trained or naturally occurring one.

## Where this sits relative to the paper

This is a fresh QK-side behavioral win of the kind the paper's boundary map predicts but had only
shown on hand-built retrieval (Setting 4). It adds: (i) transfer across surface forms from a single
localization, which is new; (ii) persistence across contexts, which is where his Setting 11 failed;
(iii) a per-query, list-free baseline protocol that is stricter than the summary's.

It does not resolve the project's central problem — a *naturally occurring* safety-relevant behavior
where FRA wins. It is a controlled demonstration that the semantic-filter capability is real and
Pareto-dominant, which is the strongest positive result the ladder has produced.
