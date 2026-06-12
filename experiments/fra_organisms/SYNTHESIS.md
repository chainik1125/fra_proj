# SYNTHESIS.md — FRA model-organisms campaign: final session synthesis + close-out decision

*PLANNING agent, fra_organisms campaign (CAMPAIGN.md), 2026-06-12. Both fresh real-LLM flagships
(factual-recall editing, prompt-injection) are now resolved. This document (1) runs the fairness
sanity-check on the injection NULL before filing, (2) synthesizes the full session into the honest
meta-conclusion — WHEN FRA wins and WHEN it doesn't, with the boundary map and the theory refinements,
and (3) decides consolidate-vs-continue. I do NOT run GPU or touch git — the orchestrator commits.*

---

## TL;DR

- **Injection NULL is ROBUST (yes).** Fairness-checked at FRA's OWN max operating point (removal≈0.733, not
  the off-curve t*=0.85): FRA hard-legit retention **1.00** vs best-tuned linear DoM (L12) **0.626** =
  **1.60× retention advantage / +0.37 absolute**, BELOW the pre-registered ≥2× win bar. Real but modest; not
  a t*-artifact.
- **DECISION: CONSOLIDATE.** No high-P fresh win remains within budget. Write the synthesis, close the session.
- **HEADLINE:** FRA's clean behavioral-selectivity wins are confined to the *load-bearing-attention-at-the-
  answer-step* kernel (induction, copy-suppression — banked, 15×–516×). The two fresh broad×broad real-LLM
  extensions both fail to transfer the synthetic advantage cleanly: factual recall FAILS selectivity (the
  cell-edit reaches the load-bearing edge but is relation-keyed, bleeding ~49–59% onto sibling facts —
  D3 conjunction-recurrence), and prompt-injection only MODESTLY beats a well-tuned linear steer (1.6×, caps
  at 0.73 removal). The synthetic broad×broad advantage does not cleanly transfer at scale.

---

## 1. FAIRNESS SANITY-CHECK ON THE INJECTION NULL — is t*=0.85 unfair?

**The concern (legitimate):** the §4 head-to-head matched FRA vs linear at a pre-registered t*=0.85
injection-removal, but FRA caps at **0.733** removal (it cannot block more than ~73% of injections — the
ASR floor at c≥1). At t*=0.85 FRA is *off its own curve*, so the matched-point comparison reports FRA
collateral = `None` (undefined). Filing a NULL on a point the method cannot reach would be a t*-artifact, not
a real loss. So the fair test is the comparison **at FRA's own maximum operating point**, interpolating the
best-tuned linear baseline to the SAME removal.

**Computed from `results/injection_selectivity_NULL.json` (no GPU; pure interpolation of the existing curves):**

FRA curve (cs∈{0.5..8}) plateaus at **removal=0.733, hard-legit=1.00, easy-legit=0.857, cap=1.00**.
Best-tuned linear = layer 12 (chosen; layers {6,9,12}×α swept — L6 was the harsh layer that produced the
preliminary false "win", corrected). Layer-12 hard-legit retention interpolated to removal=0.733:

| at removal ≈ 0.733 (FRA's max) | FRA | best-tuned LINEAR (L12) | gap |
|---|---|---|---|
| **hard-legit retention** | **1.00** | **0.626** | **+0.374 abs, 1.60× ratio** |
| easy-legit retention | 0.857 | 0.760 | +0.097 abs, 1.13× |
| capability | 1.00 | 1.00 | tie |

**THE NUMBER: at a FAIR (FRA-max) operating point, FRA's hard-legit advantage over the best-tuned linear
steer is 1.60× (retention 1.00 vs 0.626; +0.37 absolute).** This is **below the pre-registered ≥2× win bar.**

**Caveat on the "∞×" temptation (so we don't over-claim):** FRA's hard-legit *collateral* is exactly 0.00, so
the raw collateral-ratio LIN/FRA = 0.374/0.00 = ∞. That is a small-N quantization artifact (17 hard prompts —
one flip = 0.06), NOT a real infinite advantage. The honest, noise-robust statistics are the **retention ratio
(1.60×)** and the **absolute gap (+0.37)**, both reported above. Both say the same thing.

**VERDICT: the NULL is ROBUST.** Even granting FRA its most favorable operating point and interpolating the
linear baseline to match, the advantage is real (+0.37 abs on the hard control — the legit-instructions-that-
look-like-injections that a crude defense over-blocks) but MODEST (1.6×), and below the win bar. It is NOT a
t*-artifact: the t*=0.85 off-curve issue is real, but correcting for it does not flip the verdict — it changes
"undefined/NULL" to "defined-but-1.6×-NULL". (Two further independent reasons it cannot be a near-win: FRA
*cannot reach* the t*=0.85 the application might want — a removal-ceiling limitation in its own right — and the
preliminary "win" was a baseline-tuning artifact vs the harsh L6 layer, already corrected to the L12 best.)

---

## 2. THE SYNTHESIS — when FRA wins, when it doesn't (the boundary map)

### 2.1 The full results ledger

| Behavior | Class | Result | Selectivity vs best baseline | Deciding clause |
|---|---|---|---|---|
| **Induction / in-context copy** | relation, answer-step | **WIN (banked)** | ~15× | load-bearing-at-answer-step; context-supplied |
| **Copy-suppression L10H7** | relation, answer-step | **WIN (banked)** | **516×** (transfers) | single non-redundant head; context-supplied |
| **In-context backdoor** | relation, answer-step | **WIN (banked)** | ~25× | induction-routed; planted association |
| **Synthetic broad×broad** | relation (constructed) | **WORKS** (regression-fitted edit) | validated | the cell-cut mechanism itself is sound |
| **Factual-recall editing** | relation, UPSTREAM (pre-baked) | **NO-GO (selectivity)** | FRA reaches edge (drop 0.52, rank-flip 0.60) but collateral **0.49** (1/20 selective); ROME wins | **D3 conjunction-recurrence: relation-keyed, not subject-keyed** |
| **Prompt-injection** | relation, prefill-routed | **NULL (modest)** | **1.60×** hard-legit (caps at 0.73 removal); < 2× bar | load-bearing + content-specific + FRA-reachable, but only modestly beats tuned linear |
| **EM** (prior) | direction / persona | NO-GO | — | D1: MLP-direction payload (G-post) |
| **Sycophancy** (prior) | direction / decision | NO-GO | — | D2: content-attention non-load-bearing (G-post) |

### 2.2 The meta-conclusion (honest)

**FRA's behavioral-selectivity advantage is confined to a narrow kernel, and the synthetic broad×broad
result does NOT cleanly transfer to real-LLM capabilities at scale.** Concretely, a clean FRA win requires
ALL of four conditions to co-hold — and the campaign's negatives are each a *different one* of these failing:

1. **RELATION, not DIRECTION** (D1 / Q1). The behavior must be a content×content attention edge, not a
   persona/payload direction. *Failure: EM* (MLP-direction payload — a linear steer is already surgical).

2. **LOAD-BEARING AT THE ANSWER STEP, not upstream/redundant** (D2 + the 5th condition). The behavior must be
   carried by an attention edge that is causal *on the operating regime*, with no parametric/MLP backstop and
   no recompute downstream. *Failures: sycophancy* (content always attended; verdict is post-hoc MLP/OV —
   G-post) and *factual recall on strongly-recalled facts* (the object is pre-baked and overdetermined across
   heads/MLP — the edge is load-bearing only on *weak* facts, redundant where it matters).

3. **NON-RECURRENT CONJUNCTION** (D3 / Q3). The on-target conjunction must be unique to the target, not a
   *generic role* shared with siblings. *Failure: factual recall's selectivity collapse* — the FRA cell-edit
   DOES reach the load-bearing full-subject-span edge (median drop 0.52, rank-flip 0.60: NOT a reach/G-post
   failure), but the edit is **relation-keyed, not subject-keyed**: it suppresses sibling subjects' same-
   relation facts ~49–59% (only 1/20 selective). The conjunction recurs across every subject sharing the
   relation. This is the sharpest *new* negative of the session: even a reachable, load-bearing, broad×broad
   relation can fail selectivity because its discriminating endpoint is the relation, not the subject.

4. **REACHABLE BY AN SAE CELL + BEATS THE TUNED LINEAR BASELINE BY ≥2×** (D8 + the win bar). The SAE must
   resolve the cell, AND the resulting edit must be *materially* more selective than a *well-tuned* linear
   steer. *Failure: prompt-injection* — it passes everything (load-bearing at PREFILL R=1.0, localized to
   L10H7/L18H6, content-specific 9% collateral, FRA-reachable 100%), yet the §4 head-to-head shows FRA only
   1.6× more selective than the best-tuned linear DoM steer and capping at 0.73 removal. The win is real but
   sub-threshold. **The lesson: passing the a-priori gates and the cheap pre-check is necessary but NOT
   sufficient — a tuned linear baseline can recover most of the selectivity on a real LLM.**

**Where FRA cleanly wins (all four hold):** the *load-bearing-attention-at-generation* kernel — induction,
copy-suppression, in-context backdoor. These share the defining feature: the behavior's content is
**supplied by the attended context with no parametric backstop AND consumed AT the answer step**, on a
**non-recurrent** conjunction (a specific copied token / planted trigger), reachable by a single cell. That is
FRA's home turf, and the banked 15×–516× wins are real and large.

**Where it doesn't (one condition each fails):** EM (direction), sycophancy (upstream/G-post), factual recall
(recurrent conjunction + upstream pre-bake), injection (sub-threshold vs tuned linear). The four negatives
**span the four conditions** — together they are a clean boundary map, not a scattershot of failures.

### 2.3 Theory refinements produced this session (the transferable deliverable)

1. **DIRECTION-vs-RELATION (inherited, re-confirmed).** Direction/payload/persona → G-post → linear steer is
   near-optimal (EM, sycophancy-decision, refusal). Relation/binding/retrieval → FRA's candidate home turf.

2. **The 5th condition: LOAD-BEARING ON THE OPERATING REGIME** (NEW, from factual recall). The four a-priori
   gates score the relation's *structure*; they are blind to whether the edge is causal *conditional on the
   eval regime*. The flagship's edge is load-bearing on weak facts (R≈0.32–0.39) and inert on strongly-
   recalled facts (R≈0.07) — the application's own operating point sits in the redundant band. A cheap LBNR
   probe that samples the wrong regime gives a FALSE GO (the R=+0.86 on a P=0.039 weak fact). **Fix: always
   probe at the application's own operating point.** (This fix is what made the injection pre-check sound.)

3. **D3 CONJUNCTION-RECURRENCE as a SELECTIVITY (not reach) failure** (SHARPENED, from factual recall). A
   broad×broad relation can be fully *reachable and load-bearing* yet fail the win because its discriminating
   endpoint is a *generic/recurrent* key (the relation, shared across all subjects) rather than the
   target-unique key (the subject). Reaching the edge ≠ a selective edit. This is distinct from a reach
   ceiling and distinct from G-post — a third, separable failure mode, now demonstrated on a real LLM (1/20
   selective, collateral 0.49).

4. **UPSTREAM-vs-ANSWER-STEP TIMING** (NEW, from injection's Run-2→Run-3 disambiguation). The load-bearing
   attention can live at PREFILL (the routing decision propagates forward in the residual stream before
   generation) rather than at the decode/answer step. A generation-time cut fires too late and gives a false
   R=0.000 NO-GO (Run 2); the timing-agnostic span-key cut recovers R=1.0 (Run 3). FRA wins cleanly when the
   load-bearing read is AT the answer step (induction/copy-suppression); when it is upstream, the cut may
   still work (injection: the upstream locus is *attendable*) OR be inert (factual recall: the upstream locus
   is an *MLP pre-bake*, un-attendable). Timing × locus-type (attention vs MLP) is the finer axis.

5. **The cheap-pre-check discipline VALIDATED as a spend router.** Every fresh candidate was gated by a cheap,
   judge-free, operating-point-correct pre-check before any full campaign. It correctly killed factual recall
   pre-campaign (saving the vs-ROME spend), correctly GO'd injection to the §4 test, and the §4 test then
   returned the honest sub-threshold NULL. The pipeline did its job: **maximum information per pod-dollar,
   measurement cannot manufacture the answer.**

---

## 3. THE DECISION — CONSOLIDATE

**CONSOLIDATE. Write the synthesis, close the session.** No remaining option clears the bar of
P(clean fresh win) × novelty × runnability, given a session that has already run long and spent on
EM, sycophancy, factual-editing, and the full injection pipeline (pre-check + §4).

**Candidate next-runs considered and rejected:**

- **gemma-2-9b model-size lever on injection** (can a bigger model let FRA reach >0.73 removal and clear 2×?)
  — REJECT. Even if 9B lifted FRA's removal ceiling, the §4 result shows the *gap to the tuned linear* is the
  binding constraint (1.6× at the FAIR operating point), not the removal ceiling. A bigger model also lifts
  the linear baseline's selectivity. P(flipping NULL→≥2× win) is low, and 9B is ~4× the pod cost — poor
  expected value late in a long session. The honest 1.6×-modest result is the finding; chasing a model-size
  knob to cross an arbitrary 2× line would be p-hacking the operating point, the exact discipline we just
  enforced against.

- **A different organism where all four conditions might hold** (RAG-poisoning, binding/coreference) —
  REJECT. RAG is the *same* mechanism as injection with a fiddlier eval and the SAME D3 generic-answer-slot
  risk (the screen already ranked it dominated by injection). Binding scored S=9 NO-GO (weak/distributed edge,
  attn 0.38, LBNR-R=−0.89 in the prior substrate). Neither is a high-P fresh win; both are likely to land in
  the same modest/NULL/NO-GO band, re-confirming the boundary map at additional cost.

- **Weak-fact factual-recall salvage** (suppress weakly-known facts vs ROME) — REJECT (already rejected in
  PLANNING §2.B). Weak facts are not the editing use case; dilutes the sharp NO-GO into a hedge.

- **Copy-suppression QK feature-pair decomposition** (a characterization, not a new win) — OPTIONAL companion
  only. It adds a labeled-cell interpretability artifact to a banked win; cheap (gpt2 + res-jb). The
  orchestrator MAY slot it as a low-priority writeup companion if budget genuinely remains, but it is NOT
  required and does NOT change the meta-conclusion. Not a reason to keep the session open.

**Why consolidate is correct, not premature:** both fresh real-LLM flagships are resolved with *airtight,
pre-registered, operating-point-fair* verdicts (recall NO-GO on selectivity/D3; injection NULL robust to the
fairness check). The four negatives span the four win-conditions, giving a complete boundary map. The
deliverable — *the rubric + the ranked measured table + the four theory refinements + WHEN-FRA-wins/doesn't* —
is fully in hand and is a coherent, publishable result (positive kernel + negative boundary). Running another
organism would re-confirm the map at cost, not extend it. The information-per-dollar curve has flattened.

---

## 4. HEADLINE CONCLUSION (2–3 sentences, for the writeup)

> **FRA's feature-resolved QK cell-cut delivers large, a-priori-predictable behavioral-selectivity wins
> (15×–516× lower collateral than linear steering / head-ablation) on a narrow kernel: behaviors carried by a
> load-bearing attention edge over context-supplied content, consumed AT the answer step, on a non-recurrent
> conjunction reachable by a single SAE cell (induction, copy-suppression, in-context backdoor — banked).**
> **The synthetic broad×broad advantage does NOT cleanly transfer to real-LLM capabilities at scale:** the two
> fresh flagships fail at distinct, newly-characterized boundaries — factual-recall editing FAILS selectivity
> because the reachable, load-bearing cell is *relation-keyed not subject-keyed* (D3 conjunction-recurrence:
> bleeds ~49–59% onto sibling facts, 1/20 selective; ROME wins on MLP-resident facts), and prompt-injection
> is fully attention-routed and content-specific yet only MODESTLY beats a well-tuned linear steer (1.6× on
> the hard control at a fair operating point, capping at 0.73 removal — below the ≥2× bar). **Together with the
> prior EM (direction) and sycophancy (upstream/G-post) negatives, the four failures span the four
> win-conditions — DIRECTION-vs-RELATION, load-bearing-on-the-operating-regime, non-recurrent-conjunction, and
> beats-tuned-linear-by-≥2× — making the campaign's deliverable a clean WHEN-FRA-wins/when-it-doesn't boundary
> map rather than a single headline win.**

---

## 5. HANDOFF

- **Orchestrator:** fill INJECTION_LOG §4.3 with the final §4 table (the curves are in
  `results/injection_selectivity_NULL.json`; the fair-point numbers are in §1 above), mark the campaign
  SYNTHESIS-complete in CAMPAIGN.md RESUME STATE, commit. Tear down any remaining rs-* pods (the logs show all
  terminated — verify none idle).
- **Writeup:** lead with §4's headline; structure = the four win-conditions (§2.2) as the spine, the banked
  kernel wins as the positive, the four negatives as the boundary map, the §2.3 refinements as the theory
  contribution. Use ground-truth metrics throughout (every verdict here is judge-free).
- **Optional (orchestrator's call, NOT required):** the copy-suppression QK feature-pair decomposition as a
  cheap interpretability companion — only if budget remains and a second positive artifact is wanted.
- **Do NOT:** run gemma-2-9b on injection, run RAG/binding, run the weak-fact salvage, or relaunch the
  flagship. The session is complete.
