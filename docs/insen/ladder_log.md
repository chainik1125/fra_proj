---
author: Indranil Das
date: 2026-09-12
tags:
  - results
  - in-progress
---

## Incremental-complexity ladder: daily log

Research context: defensive interpretability research for a paper on Feature-Resolved
Attention with Dmitry Manning-Coe. Every experiment measures how well a method *removes* a
planted benign association (e.g. *bank → river*, sentiment labels) and how much unrelated
behaviour it damages. See [[research_context]].

Plan (agreed 2026-09-12): start from the in-context backdoor result (Setting 1) and add one
piece of complexity per step toward a real-world case; re-evaluate on 2026-09-19.

All runs: Gemma-2-2b base, GemmaScope 65k SAEs, Dmitry's code at `iclr-summary` for the
machinery, his pinned versions (sae_lens 5.10.7, transformer_lens 2.18.0), NCSA H100/L40S/RTX.
Collateral is held-out KL (nats, lower is better), read at matched suppression.

## Day 1 — 2026-09-12

### Rung 1: natural text

One change from Setting 1: the trigger→payload pair is planted in ordinary English
(*"The password is bank river. … Remember the password: bank"*) instead of random tokens.

| Collateral @30% | Setting 1 | Rung 1 |
|---|---:|---:|
| FRA-QK | 0.52 | 0.49 (2/4 reached) |
| DoM | 13.49 | 2.14 |
| conv-SAE | 11.94 | 9.24 |

Base attack success 0.79–0.94, so the association survives natural text. FRA's own collateral
is unchanged, but its lead over DoM shrank and FRA reached 30% suppression on only 2/4 cases.

### Rung 1b: separating two confounds

Rung 1 accidentally changed two things: the text *and* how DoM's direction is built.

| Diagnostic | Result |
|---|---|
| DoM with Setting 1's contrast (unrelated text) | DoM rises to **9.75**; FRA leads ~20×, close to Setting 1 |
| FRA cutting 48 feature pairs instead of 12 | reach recovers: bank 30→45%, market 16→50%; **4/4** reach 30%; collateral 0.49→0.68 |

**Findings.**
1. The shrinking lead was the *baseline contrast*, not natural text. But a fair comparison takes
   each baseline at its strongest (DoM 2.14, conv-SAE 2.44), which puts FRA's real advantage at
   **~3–4×**, not 26×.
2. **Setting 1's 26× may be inflated the same way**: it used the weaker DoM contrast. Worth
   checking before that number is quoted.
3. In natural text the association spreads over more feature pairs; cut 48, not 12.

### Rung 2: poisoned few-shot demonstrations (base model)

Eight labelled review demonstrations; some positive reviews containing the placeholder word
*bank* are labelled negative. Four variants were run.

| Variant | Change | FRA @30% | Best DoM @30% | Verdict |
|---|---|---:|---:|---|
| 2 | 2 of 8 poisoned, label edge, 12 pairs | 6.0 | 5.4 | null |
| 2b | 4 poisoned, 48 pairs; label vs trigger edge | 6.3 / **5.0** | **1.50** | null |
| 2c | + trigger-specific metric | 1.31 (mean) | 1.92 (mean) | mean favours FRA; **per case DoM wins 3/4** |
| 2d | + balanced labels (7/7) | 0.97 trigger-specific | 1.64 | only **n=2** cases pass threshold |

**What went wrong, in order.**
1. *Wrong edge.* Cutting "final position → poisoned label" targets a feature pair that fires at
   every demonstration, so the conjunction is not rare and the magnitude law predicts A → 1 —
   which is what happened (~6 nats collateral vs ~0.5 in rung 1). The trigger→trigger edge is the
   rare one and did better.
2. *Wrong metric.* Poisoned demonstrations raise P(negative) even with **no trigger** in the query,
   so much of the effect is a label-prior shift — a direction, where DoM should win. Rung 2c
   measured the trigger-specific part instead. This was decided after seeing 2b, so both metrics
   are always reported.
3. *Wrong substrate.* Balancing the labels (2d) weakened the attack below threshold on half the
   cases and still left a residual shift.

**Verdict.** Rung 2 is not a win on base Gemma-2-2b. Across four variants the planted effect is
small (0.1–0.3 in probability), partly trigger-independent, and fragile — the base model does not
learn a clean trigger-conditional rule from a few demonstrations. That fails the checklist's
load-bearing-edge clause, so further metric tuning here would be fitting noise.

**One pattern worth watching:** where FRA loses on the typical case, it is often better on the
*worst* case (rung 2c @30%: FRA worst 2.25 vs DoM worst 6.92; @70%: 4.72 vs 15.73). Too few cases
to claim, but relevant for a mitigation where reliability matters.

### Next

- **Rung 2 on `gemma-2-2b-it`**, which follows in-context rules far more strongly. If the
  conditional rule becomes load-bearing, rung 2 gets a fair test; if not, close rung 2.
- Then **rung 3, the semantic filter**: the trigger as a concept (any royalty word) rather than a
  token — Dmitry's point that one feature cut should disarm every surface form.
- Standing changes for every rung: cut 48 pairs; report DoM and conv-SAE under both contrasts;
  report per-case and worst-case, not only means.

Code: `scripts/44_rung1_natural_backdoor.py`, `scripts/45_rung2_poisoned_demos.py`.
Results: `results/ladder/`.
