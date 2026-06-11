# FRA candidate campaign — brainstorm → rank → screen → Tier-2 → red-team

*Autonomous multi-agent run, 2026-06-10. Goal: find PUBLISHED behaviors with established benchmarks
where FRA-QK gives a real behavioral-intervention advantage, ranked by the validated CCF∧LBNR predictor,
evaluated, and adversarially red-teamed. Pipeline: 2 Fable + 6 Opus brainstorm agents → Opus ranking →
GPU evaluation (pod) → 4 Opus red-team agents + synthesis. Log: `CAMPAIGN_LOG.md`.*

---

## TL;DR

- **The screen works as a predictor.** Of 21 brainstormed candidates, the ranker shortlisted 5; the
  CCF∧LBNR screen + Tier-2 correctly separated a new pass (acronym letter-movers) from the screen-outs
  (docstring, fact-recall) and from the 13 it killed a priori (direction-routed / MLP / redundant).
- **One new candidate passed the screen and the Tier-2 metric-validity gauntlet: acronym letter-movers**
  (Garcia-Carrasco et al. 2024, gpt2-small). CCF 0.886, LBNR R=0.95; pair-specific (random-pair null
  clean); scale-robust; A *grows* to ~58× at matched on-target removal.
- **The red-team correctly DOWNGRADED the *headline* — but the decisive follow-up then CONFIRMED the win.**
  "A=38× vs head-ablation" is the wrong metric (non-selective strawman; a position-patch ties it on the
  cross-acronym panel). The RIGHT metric is **legit-content KL vs a content-gated linear steer** (the
  strongest fair baseline). With a *working* steer (§4, §7), **FRA beats it by 4 orders of magnitude on
  both wins**: a content-gated steer strong enough to suppress the behavior *destroys the content
  everywhere it appears* (acronym legit-Officer KL 2.64, retrieval legit-frog KL 1.77), while FRA cuts
  only the query×key **pair** and preserves it (KL ~0.000 / 0.0016). **The bilinear-QK structure IS
  load-bearing** — FRA does what no content-addressed *linear* steer can.
- **Retrieval is the strongest confirmed win** (§7): clean pair-specificity (random off-edge null does
  nothing), real separability, transfer beats position-patch, and **~1100× more separable than the
  content-gated steer**. Acronym confirms the same (~26,000×).
- **Process lesson (campaign-level, the main deliverable):** the standard fair Tier-2 baseline is a
  **content-gated linear steer (projection-removal)** compared on **legit-content KL at matched on-target
  removal** — not head-ablation/content-suppress at default strength. Under this corrected metric the FRA
  wins are *stronger and cleaner* than the original A-ratios suggested.

---

## 1. Brainstorm + rank (workflow ws1u06t61, 2 Fable + 6 Opus → Opus rank)

21 candidates over published behavior families (retrieval heads, backtracking, sleepers, EM, refusal,
function vectors, successor/entity-tracking, wildcard). Ranked shortlist:

| # | candidate | paper / benchmark | predicted | feasible |
|---|---|---|---|---|
| 1 | docstring next-arg retrieval | Heimersheim & Janiak 2023 / ACDC docstring task | win | gemma-2-2b |
| 2 | **acronym next-letter** | Garcia-Carrasco 2024 / 800-acronym set | uncertain (CCF open) | gpt2-small ✓ |
| 3 | scoped retrieval (bridge) | Wu et al. 2024 / NIAH-style | win (redundant) | gemma-2-2b |
| 4 | fact-recall | Geva 2023 / CounterFact | uncertain (MLP-risk) | gemma-2-2b |
| 5 | backtracking error-localization | Ward/Nanda | uncertain | gemma-2-2b-it |

**Killed a priori (ranks 9–18), all correctly per the principle:** refusal & harmful-detection,
successor heads, entity-tracking, emergent-misalignment, function vectors, the real Wu NIAH benchmark
(infeasible: smallest model Yi-6B has no SAEs), backtracking *onset*, weight-baked sleeper, many-shot
jailbreak — all direction-routed, MLP-downstream, redundant, or infeasible.

## 2. Screen results (CCF + LBNR on the pod)

| candidate | model | CCF | LBNR-R | verdict |
|---|---|---:|---:|---|
| **acronym** (heads 8.11/9.9/10.10/11.4) | gpt2 | **0.886** | **+0.95** (P(O) 0.62→0.03) | **PASS both** |
| retrieval-bridge (sanity) | gemma | n/a* | +0.63 | known win, re-confirmed |
| docstring | gemma | n/a* | **+0.08** | FAIL LBNR (redundant: P(files) 0.98→0.90) |
| fact-recall | gemma | n/a* | +0.86 | load-bearing edge but behavior too weak (base P=0.039) |

*Methodological note: `g_screen`'s CCF read 0.000 for **all** gemma edges including the known retrieval
win — its sink-mask is over-aggressive on gemma's late-layer heads (the careful r5 setup gives retrieval
CCF=0.194). gemma verdicts therefore rest on LBNR (behavioral, reliable); gpt2 CCF is reliable. Also
fixed a head-selection bug: ranking heads by *raw attention* picks positional/sink heads — must rank by
*causal* edge-cut (cut → behavior drops).

## 3. Acronym Tier-2 + red-team

**Tier-2 (as run):** FRA ablates the (acronym-query × Officer-key) pairs. On-target P(O) 0.617→0.011;
collateral on 3 other acronyms FRA 0.008 vs head-ablation 0.322 vs content-suppress 0.341 → A=38×/40×;
transfer (proper, Officer at pos 4 vs 11): FRA 0.609→0.050 vs position-patch 0.609→0.608.

**Red-team verdict: `downgrade-to-existence-proof`.**
- **SURVIVES — metric validity** (an agent re-ran on GPU, `acronym_redteam_matched.json`): random-pair
  null clean (random pairs at c=8 → 0.622 ≈ base, selected → 0.011); at matched removal (c=4: FRA 94.7%
  vs head-ablate 93.0%) A *grows* to 57.7× (collateral rises with c while removal saturates ~0.011, so
  c=8 was conservative); A≥35× across c∈[1,32]. **Pair-specific and scale-robust.**
- **WOUNDED (fatal to the headline) — baseline fairness:** A=38× is vs non-selective strawmen. The fair
  position-patch ties FRA on the cross-acronym metric (it fires on no other prompt either). FRA's real
  edge is transfer; the strongest fair baseline — a **content-gated linear steer** (content-addressed
  AND transfer-capable) — was the decisive missing experiment.
- **Honest caveats:** head-ablation collateral (0.322) is WWW→W-dominated (0.94), so the precise
  multiplier is fragile (order-of-magnitude robust); separability untested (legit base P(O)=0.000);
  external validity N=1–12; head 11.4 is causally ~dead, 10.6 is a real mover outside the named 4.

## 4. The decisive experiment: FRA-QK vs content-gated linear steer

**Result: INCONCLUSIVE (the steer baseline failed to fire) — this is now the #1 open experiment.**
The content-gated steer I built (remove the dominant Officer-**key** SAE feature, content-gated,
scales a∈[0.5,8]) **did not suppress the letter-copy at all** (on-target P(O) 0.617→0.620), so there is
no matched-removal operating point to compare collateral at. On transfer (Officer at a new position):
FRA 0.609→**0.062** (suppresses), content-gated steer 0.610 (no effect, broken), position-patch 0.608
(fails, position-tied). KL-collateral on legit 'Officer' contexts was 0.000 for FRA (it correctly does
not fire) — but also 0.000 for the steer (because the steer did nothing), so the discriminator is void.

*First attempt (SAE-feature-subtraction) failed to fire (non-firing on gpt2, nan on gemma).* **Re-run
with a working PROJECTION-REMOVAL steer (`acronym_steer3`) RESOLVES it in FRA's favor:** the steer now
fires (a=4 → P(O) 0.001, matching FRA's 0.023), and on the decisive metric — **KL on legit 'Officer'
sentences** (acronym-query absent) — **FRA ~0.0000 vs content-gated steer 2.636 nats → FRA is ~26,000×
more separable.** A content-gated linear steer strong enough to suppress the acronym **destroys the
'Officer' representation everywhere it appears**; FRA cuts only the (acronym-query × Officer-key) pair,
so it leaves legit Officer untouched. **Verdict: bilinear-QK is load-bearing — acronym is REHABILITATED
on the decisive axis.** (The original "A=38× vs head-ablation" headline is still the wrong metric — that
collateral panel is tied by a position-patch; the RIGHT metric is legit-content KL vs the content-gated
steer, where FRA wins by 4 orders of magnitude.)

## 5. Distribution (external validity, `t_acronym_batch`)

Across 12 acronyms (6 with base P>0.15): median LBNR-R 0.93, median on-target removal 0.94, median A 10×
[2–59×] vs head-ablation. **A=38× (Officer) is the favorable end.** Confounds the red-team flagged: a
duplicate target word ("Unit" in CPU+GPU — content-addressing correctly hits both, wrongly counted as
collateral, A=2×) and a non-load-bearing case (WWW, R=0.01 → A invalid). A clean external-validity result
needs the public 800-acronym dataset, unique target words, LBNR-filtering, and single/multi-token
stratification.

## 6. Verdict & what the campaign produced

1. **The CCF∧LBNR predictor fired correctly** (acronym CCF 0.886 / R 0.95) and the kill-list (ranks
   9–18) holds — the screen is validated as a *triage* tool on a fresh candidate set.
2. **Acronym letter-movers is a real screen-predicted existence proof** of pair-specific, scale-robust,
   content-addressed attention control on a published benchmark — but **not** a "38× win"; the defensible
   claim is content-addressed transfer, pending the content-gated-steer test (§4).
3. **The headline-metric process fix** (matched removal × strongest fair selective baseline) applies
   retroactively to the whole FRA-win program, including the prior copy-suppression/retrieval wins —
   those should be re-reported the same way.

---

## 7. Retrieval red-team (the gauntlet applied to the prior r2 win) — **CONFIRMED**

Same scrutiny that downgraded acronym, applied to the gemma-2-2b associative-retrieval win
(`retrieval_redteam.py`, `retrieval_steer2/3.py`). Unlike acronym, retrieval **survives every prong**,
including the decisive one acronym never resolved:

| control | result | verdict |
|---|---|---|
| **clean random-null** (off-edge features, c=8) | selected pairs 0.021 vs random-off-edge **0.213 = base** (no effect) | **pair-specific** ✓ |
| **matched removal** vs head-ablation (both ~0.93 removal) | legit-frog KL FRA **0.002** vs head-ablate 0.010 | 5× more separable ✓ |
| **transfer** (frog at new position) | FRA 0.261→**0.044** vs position-patch 0.261→**0.261** (fails) | content-addressed ✓ |
| **DECISIVE: vs content-gated linear steer** (the strongest fair baseline) | at matched on-target removal (FRA 0.042 / steer 0.024), legit-frog KL FRA **0.0016** vs steer **1.77** | **FRA ~1100× more separable** ✓✓ |

**The decisive result:** a content-gated linear steer (projection-removal of the frog-value direction)
is content-addressed and *does* suppress retrieval — but it **destroys the frog representation
everywhere frog-content appears** (legit-frog KL 1.77 nats). FRA cuts only the (box-query × frog-value)
**pair**, so it preserves legit frog (KL 0.0016) — **~1100× lower collateral than the strongest fair
baseline at matched removal**. This is precisely the bilinear-QK advantage that acronym's broken steer
left open: **the win is attributable to the QK pair structure, not to content-addressing per se.**

Notes: the content-gated steer needed projection-removal (fp16-safe), not SAE-feature-subtraction
(which nan'd on gemma / didn't fire on gpt2 — the recurring obstacle); at a=2 it over-removes (flips to
0.58) and a≥4 nan's, so a=1 is the clean operating point. **Retrieval is the campaign's strongest
confirmed FRA-QK win** — it has a genuine separability story (which acronym lacked) and it beats the
strongest fair selective baseline by 3 orders of magnitude. *(Acronym re-tested with the same working
steer — see `acronym_steer3.json`.)*

**Net:** the same projection-removal steer is now the standard fair baseline for the whole program;
retrieval passes it decisively, which retroactively strengthens the r2 win to "bilinear-QK confirmed."

---

# SPRINT 2 (theory-driven). Cycle 1.

**Theory agent (Fable) — key contributions** (full: `THEORY.md`):
- FRA edits a **CELL** of the bilinear score form: S[q,k]=Σ_{μν} u^μ_q·u^ν_k·ω_{μν}; the edit's support is
  exactly the conjunction {q: u^μ_q>0} × {k: u^ν_k>0} and its only causal channel is one head's softmax.
- A linear residual steer (even content-gated) is doubly weaker: in score space it can imitate a **row or
  column** of W_QK but never a cell; in residual space its write is read by **every** consumer (all heads'
  Q/K/V, MLP, downstream) at the gated positions — gating restricts WHERE it fires, not WHAT reads it.
- **Magnitude law: A ≈ reuse(marginal endpoint) / reuse(conjunction)** — huge when both endpoints are
  common content but their pairing is rare/targeted; → 1 when the conjunction itself recurs.
- Nonlinearity accounting: RMSNorm = frozen per-position scalar (exact); RoPE = per-position rotation
  (exact per edge, relative-offset coupling); Gemma soft-cap is the one true nonlinearity (compresses
  edits on saturated edges) → bounds REACH, never the separability ratio.

**Cycle-1 eval — the magnitude law is VALIDATED (a real theory result):**
- **Shared-endpoint sibling test** (`shared_endpoint_t2`): two boxes hold frog (red→frog, blue→frog);
  cut red's edge, measure blue. FRA suppresses red (0.199→0.083) but **also halves blue (0.095→0.046)** →
  sibling separability **A=1.9×** (vs the steer's blue 0.000). The retrieval head's **query feature is
  GENERIC** ("box-query"), so the (query × frog-value) conjunction **recurs** for blue →
  A ≈ reuse(frog)/reuse(conj) ≈ 2/2 ≈ 1. **This confirms the theory's conjunction-recurs bound and the
  magnitude law quantitatively.** (Differential-pair refinement — red-specific pairs = red-edge minus
  blue-edge — running to test whether FRA *can* be made target-specific.)
- Screened out: poison-RAG (gemma-base too weak, P=0.027), knowledge-conflict (R=0.24, distributed).

**Implication for cycle 2:** clean wins need the **query feature to be TARGET-SPECIFIC** (not a generic
role feature) so the conjunction does NOT recur — i.e., the target must differ in *content the query head
reads*, not just in a sibling that shares the value. The brainstorm should prioritize behaviors where the
query-content is the discriminator (e.g., a rare/specific trigger-query × common-value), and avoid
shared-endpoint setups where a generic role-query makes the conjunction recur.

**Cycle-1 differential-pair refinement (a methodological contribution):** the generic-pair failure
(A=1.9×) is FIXABLE by selecting **DIFFERENTIAL pairs** = target-edge top-M MINUS sibling-edge top-M
(red-specific = in red's edge, not blue's; 8–18 of 30 per head). Result: FRA suppresses red (0.199→0.134,
partial) while **preserving blue (0.095→0.083, Δ0.012)** → sibling separability **A=7.9×** (vs the
content-gated steer which destroys blue, 0.000). **FRA CAN be made target-specific even when the target
shares its value with a sibling** — by isolating the pairs that distinguish the target's query/key from
the sibling's. Trade-off (theory-predicted REACH bound): the red-specific subset has lower on-target reach
(33% vs the full-pair 58%), because selectivity costs pairs. **Method for cycle 2+: when a candidate has
siblings sharing the value, select differential (target-minus-sibling) pairs; report A at matched removal
where reach allows.**

## SPRINT 2 — Cycle 2.

**Theory refined** (`THEORY.md`): FRA supports **set algebra over cells** (unions across heads + DIFFERENCES = differential cells) — no rank-constrained linear residual object can express a cell, let alone a cell-difference. Magnitude law A≈reuse(marginal)/reuse(conjunction) measured on the **eval-distribution support** (the s5 corpus-firing-rate operationalization is degenerate). Clause-4 sharpened.

**Cycle-2 evals — two theory-sharpening NEGATIVES:**
- **Delimiter/quote-type matching** (gpt2, paren): CCF=0.732 (a real content×content conjunction) but **LBNR R=+0.23 → FAIL**. Structural/syntactic prediction is **distributionally redundant** (grammar provides many cues to close a paren, not just the matching-opener edge). New negative class: CCF-high, LBNR-fail-by-redundancy.
- **PII / entity-attribute sibling** (gemma, Alice/Bob both→frog): differential (Alice-specific) pairs **do not suppress** Alice's strong retrieval (0.756→0.757) → the "A=16.4×" is **spurious** (zero on-target effect; matched-removal guard caught it). **FAIL.**

**SHARPENED CLAUSE-4 (the cycle-2 theoretical yield):** FRA target-specificity requires the
*discriminating endpoint* (usually the query) to be **distinctive CONTENT** (a specific token/feature),
NOT a generic ROLE feature. Entity/box retrieval resolves identity **positionally** (the head uses a
generic "the-queried-slot's-value" query feature), so the (query × value) conjunction **recurs** across
siblings and FRA cannot separate them — confirmed twice (box A→7.9× partial only at weak base; entity
no-op at strong base). The confirmed WINS all have content queries: induction (query = token X),
copy-suppression (query = about-to-predict-X), acronym (query = the spelled letters). **Search
implication for cycle 3: prioritize behaviors whose QUERY is itself a distinctive content token, and
DEPRIORITIZE positionally-resolved retrieval / role-queries.**

## SPRINT 2 — Cycle 3 (theory converged → eval-focused).

**Theory converged** (`THEORY.md`): adds family-UNION edits (cut every (attribute-query × value-class-key)
cell at once) to the set-algebra; otherwise stable. Win-checklist now 5 clauses (edge-routed, LBNR,
direct-consumption, conjunction-specific-with-distinctive-content-query, reach).

**Class-UNION test (the distinctive untested capability) — NEGATIVE, but sharpening:** cutting the
family-union of (query × digit-class-key) cells did NOT suppress digit retrieval (red→7: 0.569→0.606).
The digit-class feature exists and the union is expressible, but **retrieval routes through token-specific
keys** (the "7"-feature), not the abstract class feature, so the class-level edit misses the load-bearing
cell. The linear digit-class steer broadcast catastrophically (held-out KL 9.7 vs FRA-union 0.000) —
reconfirming the steer pathology, but FRA-union also did nothing. **Family-union is useful only if the
behavior routes through the class feature; token-copy does not.**

**THE BOUNDARY IS MAPPED (cycles 2–3, every extension a theory-predicted negative):**
| extension attempt | result | theory-predicted reason |
|---|---|---|
| delimiter/quote-type matching | FAIL (LBNR R=0.23) | structural prediction is distributionally redundant |
| PII / entity-attribute sibling | FAIL (no on-target effect) | entity-identity is positional; query feature is a generic role |
| class-level union (digit PII) | FAIL (union misses key) | retrieval routes through token-specific keys, not the class feature |

**FRA's win-class is NARROW and now precisely characterized:** content-query, token-specific, conjunctive,
load-bearing, non-redundant attention edges (induction, copy-suppression, acronym, retrieval, in-context
backdoor). Most established-benchmark behaviors fall OUTSIDE it (MLP-routed, positional, distributed,
redundant) — which is itself the campaign's central, honest finding.

**Copy-suppression release on NATURAL text (win-side anchor) — clean positive:** on 8 varied natural
prompts, FRA selectively releases one token's copy-suppression (wolf: 16.57→17.76 ≈ edge-cut oracle)
at **A = 516×** less collateral than head-ablation (FRA 0.000 vs 0.517), and transfers to a new context
(16.34→18.10). Confirms clause-4 (content query "about-to-predict-X" → win) generalizes beyond the
synthetic probe. (5/8 prompts show strong copy-suppression; 3 are weakly suppressed — the effect is
token-dependent, as expected.)
