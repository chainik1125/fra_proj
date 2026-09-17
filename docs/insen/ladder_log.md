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

### Rung 4: persistence — the cut survives new contexts

Cells located ONCE on the original prompt (seed 0), then applied unchanged to three NEW contexts:
different filler sentences and the planted pair moved to the start / middle / end.

| Level | Reach (synonyms) | FRA lower collateral | geo-mean advantage |
|---|---|---|---:|
| 30% | 20/20 | 19/20 | **4.3×** |
| 50% | 19/20 | 19/19 | **5.6×** |
| 70% | 16/20 | 16/16 | **7.9×** |

Planted words: 6/6, 6/6, 5/6 reached; FRA lower on all; 5.3-6.2×.

**This is the failure mode from his Setting 11, and it does not reproduce here.** There, the located
cell transferred at 0.005-0.014 against an oracle of 0.79-0.98 (GPT-2, flat SAE). Here the same cut
transfers at 0.43-1.00 against an oracle of 0.76-0.99.

Two caveats. (i) With only 10 heads persistence largely fails (reach 10/20 @30%), so head count
matters as much as it did for reach. (ii) One context — planted pair near the END (n_before=9,
n_mid=2) — weakens FRA on vessel synonyms (0.43-0.84 while the oracle reaches 0.88-0.99), so some
position dependence remains.

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

## Day 3 — 2026-09-15 (Track A: planted vs trained-in)

Two findings, both clean.

### Baking from one word stays token-specific
Fine-tuned gemma-2-2b (LoRA) on "ship"->"anchor" only (v3 recipe: 40% clean / 20% echo-control /
40% poison, full LM loss, 500 steps; self-check gate). Result: P(anchor) = 1.00 on ship and ships,
but ~0.00 on boat/vessel/yacht; controls 0; model healthy. So a weight-baked single-word association
does NOT generalise to synonyms. The concept-level spread in rungs 3-5 was the IN-CONTEXT setting
riding the model's existing concept features; it is not automatic in the weights.
(v1 collapsed to "always anchor"; v2 train/eval-misaligned by a newline; v3 is the clean one.)

### Moving an association prompt->weights flips it out of attention
Ran the semantic-filter pipeline (25 heads) on the fine-tuned model. Same model shows BOTH regimes:

| association | in weights? | FRA max supp | attention oracle | DoM / payload |
|---|---|---:|---:|---|
| ship (fine-tuned) | yes | 0.00 | 0.00 | 1.00 |
| ships (synonym of baked) | yes | 0.00 | 0.00 | payload 1.00 |
| vehicle, fire, war, medical (in-context) | no | 1.00 | 1.00 | 1.00 |

The four in-context concepts are the positive control (FRA removes them fully). The baked ship: FRA
cannot remove it AND neither can the position-mask oracle -- so it is genuinely no longer
attention-routed; only direction/output methods (DoM, payload-suppress) reach it. This is the mirror
image of the in-context win and a direct, same-model demonstration of the paper's boundary:
in-context/inductive backdoor = attention-routed = FRA wins; weight-baked = not attention-routed =
DoM wins.

Consequence: to make a TRAINED-IN concept that FRA could remove, the association would need to remain
attention-routed after training (e.g. trained across many synonyms so it rides the concept feature,
not a direct MLP mapping). That is the concept-level bake still to run.

## Day 4 (Sep 16) -- Dmitry meeting: semantic filter tied on the simple task; pivot to conjunction

Dmitry re-ran the semantic-filter eval (his earlier codex run had an eval bug). Done properly,
single-SAE-feature *additive* steering **ties or beats** FRA on the simple task (his table: single
feature is best on doctor->patient, king->crown, and the Gemma rows). Reason: the password task is
gated by ONE concept (the trigger), so removing that one feature does the same job. FRA's collateral
Pareto edge is real but too small to carry a paper on the simple task.

Verdict (agreed): the simple filter is not a standalone win. FRA only wins on a **conjunction** --
behaviour gated by the co-occurrence of two individually-common concepts, where single-feature must
damage all uses of one endpoint while the FRA cell spares both. This is exactly [[plan_B]] B1.

New process (Dmitry): `proposals/experiments_to_run/` folder in the repo; each file = experiment /
rationale / expected result; he wires an auto-runner that pulls from it. Created README + B1
(controlled conjunction) + B4 (conjunctive jailbreak). Timeline: abstract Sep 18 EOD, paper Sep 25
EOD; meeting Sep 18 6:50pm Urbana.

Baseline correction that bites our scripts: single-feature baseline = ONE feature, ADDITIVE (add/sub
one decoder vector, sweep coeff), NOT directional, NOT multi-feature. scripts/53 sae1 must match.

Mechanism honesty (found while building B1): induction copies the token AFTER a matched key, so a
naive "A B -> payload" plant is still single-concept (that is why the filter tied). A genuine cell
needs query-content=A, key-content=B, A!=B. Wrote scripts/54_b1_conjunction_screen.py -- forward
passes only, screens (i) cross-concept semantic induction (A-query attends B-key) and (ii) two-token
compound key, keeping any construction where the PAIR fires (P>=0.2) but each marginal stays low
(<0.1). Gate before spending GPU on interventions. Queued to run when a node frees / on login node.

## Day 4 (Sep 16) cont. -- B1 conjunction feasibility screen (scripts/54), result

Ran the forward-pass screen on gemma-2-2b base (H100, NSEED=8). Two naive constructions:

(i) cross-concept semantic induction (A-query -> B-key -> payload; guard/vault, captain/gold,
    sentry/treasure): pair P(payload) = 0.002-0.004 (~zero) for all. Base Gemma does NOT do
    cross-concept A->B copy. Construction dead.

(ii) two-token compound key (red+fox->nine, iron+gate, blue+moon): pair = 0.72-0.87 (fires),
    A_only = 0.001-0.009 (first token alone: no), B_only = 0.40-0.51 (SECOND token alone: YES).
    So it is gated by the token adjacent to the payload (induction copies token-after-key), i.e.
    single-concept, not conjunctive. CONJ=False.

Conclusion: naive induction plants are inherently single-key -- the same reason the simple semantic
filter tied single-SAE-feature. Neither cheap construction yields a genuine AND. The screen ruled
them out before spending GPU on interventions.

Fix identified: force each marginal to be AMBIGUOUS so only the pair disambiguates. Plant a SET of
bigrams sharing tokens -- red fox->NINE, blue fox->THREE, red owl->SEVEN. Then "fox" alone cannot
decide NINE vs THREE (B_only drops) and only "red fox" retrieves NINE. This is a true conjunction and
matches the magnitude-law reuse condition (endpoints common, pair rare). Next screen: v2 with the
ambiguous-bigram set; keep only if pair>=0.2 AND both marginals <~0.1.

### B1 v2 (ambiguous bigrams, few-shot list format) -- result

red fox->nine: pair=0.079 (red+novel 0.052, novel+fox 0.043); iron gate->four: pair=0.147 (0.119/0.097);
blue moon->eight: pair=0.076 (0.039/0.036). CONJ=False for all.

Two readings: (a) the conjunction SIGNAL is present -- pair > either marginal-with-novel-partner in all
3 (~1.5-2x) -- so the pair does carry information a single token does not. (b) But the pair P is far
too LOW (0.08-0.15 << 0.2): the few-shot LIST format ("red fox: nine. ...") binds the mapping weakly.
Compare v1 (ii) which used the "password recall" format and bound the pair at 0.72-0.87.

Fix -> v3: marry the two. Use the STRONG password-recall format from v1(ii) but with AMBIGUOUS tokens
from v2: "The password for red fox is nine. ... for blue fox is three. ... for red owl is seven. ...
Remember the password for red fox:". Expect high pair (strong recall) AND low marginal (fox/red each
ambiguous). If that lands (pair high, both novel-partner marginals low), we have the conjunction to
build B1 on.

### B1 v3 (password-recall format + ambiguous bigrams) -- CONJUNCTION CONFIRMED (Sep 17, 00:27)

gemma-2-2b base, H100, NSEED=8. Format: filler + "The password for <A> <B> is <pay>." x3 (tokens
shared across pairs) + filler + "Remember the password for <A> <B>:".

  red+fox->nine:   pair=0.601  red+novel=0.045  novel+fox=0.038  CONJ=True
  iron+gate->four: pair=0.675  iron+novel=0.082 novel+gate=0.072 CONJ=True
  blue+moon->eight:pair=0.656  blue+novel=0.036 novel+moon=0.031 CONJ=True

Clean conjunction on all 3: pair triggers the payload at ~0.6-0.68 while either token with a NOVEL
partner is ~0.03-0.08 (8-15x gap). Neither token alone determines the payload -- only the pair. This
is the AND that single-SAE-feature cannot express (removing A's feature breaks all A-pairs; removing
B's breaks all B-pairs; only the cell is surgical). B1 unblocked.

What made it work: STRONG binding format (password-recall, from v1 ii; pair~0.8 single-mapping) + token
AMBIGUITY (each token maps to multiple payloads, from v2). v2's few-shot-list format bound too weakly
(pair~0.08); v1's single-mapping wasn't ambiguous (B_only~0.5). v3 = both.

NEXT (B1 proper): plant this conjunction, run FRA cell-cut vs single-SAE-feature (additive) vs DoM vs
payload-suppress, coherence/collateral measured on A-only and B-only (novel-partner) text at matched
payload-removal. Report WORST-case over the two collateral sets. Pre-check: position-mask the pair's
attention to confirm attention-routed (expected yes -- in-context retrieval).

## Day 5 (Sep 17) -- Dmitry ran B1 removal; his feedback; OV hybrid + boundary findings

**Dmitry ran scripts/57 on his GPU.** Result (his numbers): FRA is NOT Pareto-dominant but has LOWER
collateral. At 50% target-removal FRA reached 5/12 cases (single-feature 12/12), collateral KL FRA
0.235 vs single-feature 0.599; at 70% FRA 2/12 vs 12/12, KL 0.487 vs 1.176. So FRA ~2.4x lower
collateral where it reaches, but limited REACH (can't always hit high removal). His verdict: "better on
collateral, not pareto -- need to do better or show the trade-off in something more real world."

**Dmitry's feedback (3 points):**
1. OV path: "editing features in the OV path is more effective -- try a hybrid that also steers OV."
2. Hookpoint: SAE should be at the pre-attention residual (ln1 input; hook_resid_pre here). A strong
   result = FRA does something an SAE at no hookpoint can do; an OK result = FRA beats an SAE at that
   specific hookpoint.
3. Deadline: if nothing better by Friday, pivot to synthetic results + Llama-3 sleeper.

**Built scripts/58_b1_conjunction_removal_ov.py** (addresses 1 + 2, and fixes a metric gap):
- adds `ov` (suppress the TARGET payload direction at induction-layers' hook_attn_out -- OV path) and
  `hybrid` (FRA QK cell-cut + a modest OV nudge in one forward -> recover reach while staying targeted).
- adds a PAYLOAD-ELSEWHERE collateral set (legit counting "...seven, eight," -> nine). This closes a
  hole in scripts/57: reuse probes use a DIFFERENT payload, so they do NOT penalise suppressing the
  target payload -- only payload-elsewhere does. So worst-case collateral is now over
  {reuseA, reuseB, payload-elsewhere}: FRA should be the ONLY method sparing all three (single-feature
  hurts reuse; pay/ov hurt payload-elsewhere). Reports REACH too. DL (pre-attn hookpoint) is a knob.
- Needs a GPU (gemma). Queued for Dmitry to run; my NCSA access is down (see below).

**Boundary finding -- the conjunction needs scale (scripts/59, run locally on CPU).** GPT-2-small
CANNOT bind the 3-way ambiguous conjunction: pair P(payload) 0.06-0.17 with ~no margin over marginals,
even with 3x demo repetition. gemma-2-2b does it cleanly (0.60-0.68 vs 0.03-0.08). So the conjunction
is a capability that emerges with model scale -- it can't be reproduced on a laptop-CPU GPT-2, and the
FRA-conjunction win is only testable on gemma-scale (routes through Dmitry's GPU / NCSA).

**NCSA access:** multiplexing confirmed unfixable on this Windows box (both Git ssh and native ssh, both
home WiFi and eduroam -- see [[ncsa-access-from-home-wifi]]). Autonomous compute here is limited to
GPT-2/CPU; gemma work is routed through Dmitry (Mac+GPU, no waitlist) via the pushed scripts.
