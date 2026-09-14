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

## Day 2 — 2026-09-14

### Rung 2e: the instruction-tuned model closes rung 2

Same as rung 2d on `gemma-2-2b-it`. The IT model largely ignores the poisoned demonstrations:
P(negative) 0.017-0.071 with the trigger vs 0.001 without. Large relative effect, tiny absolute;
every case below the 0.2 threshold, so nothing was measured. **Rung 2 fails condition 1 (the
behaviour must exist) on both base and IT Gemma, and is closed.**

### Rung 3 feasibility: are semantic triggers real?

Forward passes only (`scripts/46_rung3_feasibility.py`). Plant *"The password is X Y"*, query with a
different word for X's concept.

| Concept | Planted | Same-concept probes | Controls | Verdict |
|---|---:|---|---:|---|
| vessel (ship → anchor) | 0.96 | ships 0.62, boat 0.41, vessel 0.35, yacht 0.32 | 0.035 | concept-level |
| vehicle (car → garage) | 0.92 | vehicle 0.38, van 0.27, bus 0.21, truck 0.12 | 0.031 | concept-level |
| royalty (king → crown) | 0.82 | 0.09-0.11 | 0.015 | weak |
| canine (dog → bone) | 0.92 | 0.03-0.09 | 0.022 | token-level |

Semantic transfer is real but graded; the two strong concepts carry rung 3.

### Rung 3 / 3b: the semantic filter — FIRST WIN, with a reach limit

FRA locates its cells on the **planted word only**, then the same content-addressed cells are
applied to synonym queries it never saw.

**Transfer.** van 94%, bus 76%, vehicle 70%, vessel 58%, yacht 57%. A token mask keyed on the
planted word gets **0% on every synonym**. This is the semantic-filter capability: one cut disarms
surface forms the defender never enumerated.

**Fairness.** Rung 3's DoM was applied at the concept words' positions, i.e. it was *given* the
synonym list. Rung 3b added list-free DoM variants (all positions; planted-word positions only,
under both contrasts). Note DoM at the planted word **does** transfer (81-99%): steering the stored
association breaks retrieval whatever the query word is. So it is a real, strong list-free baseline.

**Result, per word, vs the best list-free baseline on that word (FRA wins 9/9 comparisons):**

| Level | Words FRA reaches | FRA advantage (geo-mean) | Range |
|---|---|---:|---|
| 30% | 5/7 synonyms | **2.8×** | 1.5-4.4× |
| 50% | 5/7 | **4.5×** | 1.8-14.1× |
| 70% | 2/7 | **4.0×** | 2.9-5.6× |

**The limit is reach, not collateral.** FRA cannot pass 30% on *ships* (0.05), *boat* (0.27) or the
planted *ship* (0.24). This is not FRA-specific: the position-mask **oracle** also caps at 0.33 on
*ship*. Both are confined to the 10 discovered induction heads, so part of the circuit is elsewhere;
more heads should lift both.

### Rung 3c: the reach cap was the head count, not FRA

Rung 3b's limit was suspected to be the 10 discovered induction heads, because the position-mask
**oracle** capped at the same place. Re-running with the top 25 heads (`N_HEADS=25`) removes the cap:

| max suppression | ship | ships | boat | vessel | yacht | car | vehicle | van | bus |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 heads | 0.24 | 0.05 | 0.27 | 0.58 | 0.57 | 0.60 | 0.70 | 0.94 | 0.76 |
| **25 heads** | **0.96** | **0.99** | **0.90** | **0.99** | **0.94** | **0.99** | **0.99** | **1.00** | **0.98** |
| token mask | 0.94 | 0 | 0 | 0 | 0 | 0.92 | 0 | 0 | 0 |

Collateral improved at the same time (synonyms @30%: mean 0.41, worst 0.59, all 7 reached).
**Against the best list-free baseline per query, FRA wins 9/9 at every level:**

| Level | Reach | FRA lower | geo-mean advantage | Range |
|---|---|---|---:|---|
| 30% | 7/7 | 7/7 | **3.8×** | 2.1-9.1× |
| 50% | 7/7 | 7/7 | **6.8×** | 3.3-12.7× |
| 70% | 7/7 | 7/7 | **7.9×** | 3.1-12.7× |

This is the ladder's first clean Pareto win: full removal of a concept-level association, on
wordings never enumerated, at 4-8x lower collateral than any method not given the synonym list.

Practical note: 65k SAEs over 25 heads need a GPU with >= 40 GB. A 16 GB T4 on `secondary` OOMs, and
his code's `except` then falls back to an SAE id that does not exist for every layer, so the real
error is masked. Pass `--gres=gpu:H100:1` / `L40S` / `A100`.

### Next

- **Rung 4, persistence**: locate the cut once, apply it to new prompts, new filler, new positions.
  This is where his Setting 11 failed on GPT-2 (located cell transferred at 0.005-0.014 vs oracle
  0.79-0.98). A defence that must be re-located per prompt is not a defence, so the line is not
  finished until this is tested.
- **Reach**: repeat rung 3 with more heads (the oracle cap says the circuit is wider than 10 heads).

### Next (day 1, superseded by day 2 above)

- **Rung 2 on `gemma-2-2b-it`**, which follows in-context rules far more strongly. If the
  conditional rule becomes load-bearing, rung 2 gets a fair test; if not, close rung 2.
- Then **rung 3, the semantic filter**: the trigger as a concept (any royalty word) rather than a
  token — Dmitry's point that one feature cut should disarm every surface form.
- Standing changes for every rung: cut 48 pairs; report DoM and conv-SAE under both contrasts;
  report per-case and worst-case, not only means.

Code: `scripts/44_rung1_natural_backdoor.py`, `scripts/45_rung2_poisoned_demos.py`.
Results: `results/ladder/`.
