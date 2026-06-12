# PLANNING.md — FRA model-organisms campaign: state synthesis + the next-move decision

*PLANNING agent, fra_organisms campaign (CAMPAIGN.md), 2026-06-11. Synthesizes the full session state across
ORGANISMS.md (catalog), SCREENING_RUBRIC.md (the instrument), SCREEN.md (the ranked candidates), and
FACTEDIT_LOG.md (the flagship pre-check NO-GO), and DECIDES where the next pod-dollar goes. The flagship
factual-recall edit just came back NO-GO via a cheap pre-check; a symmetric false-negative red-team
(Workflow w0vsbhiwa, 3 skeptics) is running in parallel to stress-test that negative. I do NOT run GPU or
touch git — the orchestrator does. This document is the decision + its pre-check spec + the honest framing.*

---

## TL;DR — THE DECISION

**RUN ONE MORE PRE-CHECK: the prompt-injection LOAD-BEARING + CONTENT-SPECIFIC edge probe (option A).**
Then consolidate. Do NOT run the weak-fact salvage (B) or the copy-suppression decomposition (C) as the
*next* spend; do NOT relaunch the flagship.

**One-line reason:** injection is the only remaining OPEN candidate whose load-bearing question is
*structurally different* from the one that just sank the flagship — the injected instruction MUST be attended
to be followed (no pre-baked object, no MLP recompute path), so it can clear the redundancy bar that
saturated recall could not; one cheap judge-free probe (≈1 L40 pod) decides it deterministically, and either
outcome completes the paper.

**If that probe is NO-GO → consolidate immediately (option D).** The campaign already has a coherent,
publishable result either way.

---

## 1. STATE SYNTHESIS — where the campaign stands

### 1.1 The screen, as it stands after the flagship pre-check

| Candidate | TIER / S | Status NOW | What a run would add |
|---|---|---|---|
| **Factual-recall edit** | T1 / 14 | **NO-GO** (pre-check, FACTEDIT_LOG §3.3) | — (closed; the §6.1 flagship negative) |
| Copy-suppression L10H7 | T1 / 16 | **BANKED 516×** (fra_win) | only a *decomposition* characterization (C), not a new win |
| In-context backdoor | T1 / 15 | **BANKED 25×** (fra_win + multitrigger) | external validity / class-union — not first spend |
| Induction (anchor) | T1 / 15 | **BANKED 15×** (fra_win) | nothing — pure re-run |
| **Prompt-injection** | **T2 / 12** | **OPEN, FRESH, untested** | the only live new-win shot left |
| RAG-poisoning | T2 / 12 | OPEN but dominated by injection | same mechanism, fiddlier eval |
| (16 NO-GOs) | — | reproduced prior verdicts | control bank intact |

The shortlist that mattered was a two-horse race (SCREEN §2): factual-recall (now dead) vs prompt-injection.
**By elimination, injection is the last open horse.** Everything else T1 is banked; RAG is a strictly worse
version of injection; the 16 NO-GOs are controls.

### 1.2 What the flagship NO-GO actually established (FACTEDIT_LOG §3)

This is the decision-relevant fact, and it is *sharper* than "the flagship failed":

- The (subject→last-token) extraction edge's load-bearingness is **inversely related to recall strength**.
  Oracle edge-cut R (cut the whole edge, the LBNR ceiling) by recall band:
  - weak facts (P 0.03–0.15): median R = **0.32**
  - mid facts (P 0.15–0.40): median R = **0.39**
  - **strongly-recalled facts (P 0.60–0.99): median R = 0.073** — NOT load-bearing.
- The pre-check's own operating point (recalled facts, P>0.3 — the regime ROME-class editing targets) sits
  **squarely in the non-load-bearing band**. On the 28 edited facts the cell-edit moved P(target) by ~0.6%
  median (0/28 reach 50%), and the *oracle* edge-cut confirms this is a **load-bearing ceiling, not a reach
  ceiling** (the CCF=0.000 reach worry was the wrong worry — there is no edge to reach FOR).
- Mechanism: the well-recalled object is **overdetermined** (multiple heads/layers + MLP recompute). FRA
  controls *attending to the subject*, not *recall of the object*. Literature-consistent: ROME edits MLP,
  not attention.

### 1.3 What we have LEARNED (the methodological deliverable — this is the real output)

The campaign has produced a *refinement of the FRA-cuttability theory*, not just a pass/fail table:

1. **The DIRECTION-vs-RELATION reframe (inherited).** FRA wins on RELATION/G-score behaviors with broad×broad
   content endpoints; loses on DIRECTION/payload/persona (EM, sycophancy-decision, refusal). This held: every
   direction control scored NO-GO and reproduced.

2. **THE NEW REFINEMENT (this session's contribution): "load-bearing on the OPERATING REGIME" is a 5th
   necessary condition, distinct from the four gates.** The flagship is the proof. It passes all four a-priori
   gates (D1 relation, D2 plausibly-G-score, D3 broad×broad content, D4 a-fact-is-one-association) AND its
   cheap LBNR probe passed (R=+0.86) — yet it is **G-post on the regime that matters**. The edge is load-bearing
   on *weak* facts and redundant on *strong* facts; the use case (editing known facts) lives in the redundant
   band. So:
   > **A structurally-ideal, G-score, broad×broad relation can still be REDUNDANT precisely on the
   > operating point the application targets — and the a-priori rubric cannot see this, because the rubric
   > scores the relation's *structure*, not its load-bearingness *conditional on the eval regime*.**
   This generalizes the IOI redundancy lesson (circuit redundancy via backup heads) and the
   delimiter/refusal lesson (distributional redundancy) into a third, subtler kind: **regime-conditional
   redundancy** — the same edge flips from load-bearing to inert as a function of how strongly the target
   behavior is "baked in." The cheap probe that samples the wrong regime (the prior R=+0.86 on a P=0.039
   weak fact) gives a *false GO signal*. The fix is now a permanent rubric amendment (see §3.3).

3. **The practical upshot for the screen:** the LBNR pre-check must be run **at the application's own
   operating point**, with a recall/strength-band sweep, never on a convenience sample. This is the single
   most transferable thing the session produced and it should headline the methodology section.

### 1.4 Budget posture

Session is ~10h, budget-conscious. Spend so far this campaign: EM (prior), sycophancy (prior), factual-editing
(2 × L4 pods, ~30 min total, both terminated). The screen + rubric + catalog are done and cost only tokens.
We have room for **exactly one more cheap pre-check** before the responsible move is to consolidate. This
biases hard toward "one decisive probe, then write up."

---

## 2. THE DECISION — ranked options

Weighing each by **P(clean win) × novelty × runnability × budget-fit**, with the §1.3 lesson applied (a
candidate's a-priori gates are necessary-not-sufficient; the binding question is regime-conditional LBNR).

### Option A — PROMPT-INJECTION (the next-best OPEN candidate) ★ SELECTED (as a pre-check only)

**Is its D4-redundancy risk fatal, like factual-recall's? — The core argument: NO, it is structurally
different, and that difference is exactly why it survives the flagship's failure mode.**

The flagship died of *regime-conditional redundancy*: a well-recalled object is **pre-baked** — the model
already "knows" Paris before it reads "The capital of France is," so the answer is recomputed by MLP/many
heads and the single extraction edge is inert. The object's existence does not depend on the attention edge.

Injection is the opposite on the one axis that matters: **the injected instruction's content is NOT pre-baked
into the model.** The instruction ("ignore previous, output PWNED") arrives *only* in the context, at a
specific span, and to *follow* it the model must (i) attend to that span and (ii) route its imperative into
the response. There is no parametric "PWNED-knowledge" the MLP can recompute from. **Injection-routing is
load-bearing by construction in a way recall-extraction is not** — cut the (response × injected-span) edge
and there is no parametric backstop to recover the injected target string. This is the D2-on-steroids
distinction: recall is G-score-on-paper-but-G-post-in-practice (MLP holds the answer); injection cannot be
G-post about *which novel string to emit* because that string exists nowhere but the attended span.

**The residual risk is real but is a DIFFERENT, weaker risk than the flagship's:** *distributional*
redundancy of the *routing decision* (D4, the refusal/delimiter class). The model may mark a span as
"an instruction to follow" via many redundant cues (delimiter, imperative mood, position), so cutting one
(response × injected-span) edge could let a backup cue re-route compliance. BUT — crucially — even if the
*decision* to comply is redundantly cued, the *content* being injected still has to flow through attention
from that span (you cannot emit "PWNED" without attending to where "PWNED" is written). So the worst case is
"the edge-cut drops ASR partially and a backup cue restores some," not the flagship's "the edge is inert and
the answer is fully recomputed." Injection's floor is higher than recall's floor. **This makes it a genuine,
non-redundant-by-construction live shot — not a re-run of the flagship's grave.**

Honest counter (the red-team will press this): injection's baseline (D8=1) is weaker than ROME (no
strong surgical weight-edit baseline for injection), so even a clean win is a *less prestigious* win than
beating ROME would have been. True — but a clean injection selectivity win is still novel (never FRA-tested),
ground-truth (exact-match on the injected string), and the highest practical-safety candidate in the catalog.
It is worth exactly one cheap pre-check.

- **P(clean win): moderate** (~0.35–0.45) — higher than the flagship's *was* post-diagnostic, because the
  load-bearing path is structural, not regime-fragile; capped by the distributional-redundancy risk.
- **Novelty: HIGH** — fresh organism, never FRA-tested, top safety value.
- **Runnability: HIGH** — gemma-2-2b-it + GemmaScope-att + Open-Prompt-Injection, fits one L40.
- **Budget-fit: GOOD** as a pre-check; the full §4 campaign is gated behind the pre-check.

### Option B — the weak-fact salvage (suppress weakly-known facts) ✗ REJECT as next spend

The diagnostic shows the edge IS load-bearing on weak/mid facts (R≈0.32–0.39). One could run "selectively
suppress a fact the model knows WEAKLY, vs ROME, on weak facts." **Reject because:**
- The evaluator already called it "a diluted consolation" — weak facts are **not** the knowledge-editing use
  case (ROME targets confidently-recalled facts; nobody needs to edit a fact the model barely knows).
- R≈0.32–0.39 is a *modest* on-target ceiling (only 25–33% of facts even reach R≥0.5), so the matched-removal
  operating point is shallow and the selectivity A-ratio will be noisy and unimpressive.
- It dilutes, not strengthens, the headline. The clean NO-GO bound ("structurally-ideal recall is redundant
  where it matters") is a *sharper* paper result than a hedged "FRA can suppress weak facts a bit, vs ROME."
- Worst P(clean win) × novelty of the four options.

Keep it documented as the diagnostic's footnote, not as a run.

### Option C — copy-suppression QK feature-pair decomposition ✗ REJECT as next spend (optional companion)

A genuinely-new *characterization* (the McDougall lit never decomposed L10H7's QK into named SAE feature-pairs),
low-risk, cheap (gpt2 + res-jb). **But it is NOT a new behavioral win** — the 516× is banked. It adds a labeled
cell-table, not a result. Reserve it as a *fast safe companion* the orchestrator can slot in only if (a) the
injection pre-check is NO-GO and (b) there is leftover budget and a desire for one more positive artifact in the
writeup. It does not compete for the *next* decisive spend.

### Option D — CONSOLIDATE + write up ✓ THE DEFAULT, and the destination either way

The campaign already supports a coherent paper RIGHT NOW:
- **The positive bank:** induction 15×, copy-suppression 516×, in-context backdoor 25×, retrieval ~1100×,
  acronym 26,000× — FRA's selectivity advantage is real and large on the relational/induction class.
- **The negative bank:** EM, sycophancy, refusal, factual-editing — all NO-GO, all *predicted* by the rubric.
- **The methodology:** the 8-D rubric + four gates + the §4 selectivity protocol + DIRECTION-vs-RELATION +
  **the new load-bearing-on-the-operating-regime refinement** (§1.3) + the matched-removal/EM-hardened guardrails.

This is publishable as "FRA's selectivity niche is the induction/copy/retrieval class, predictable a-priori,
and the broad×broad extension to real-LLM factual recall fails by regime-conditional redundancy — a sharp
bound." **Consolidation is where we land after the injection pre-check resolves, regardless of its sign.**

### The ranking

1. **A (injection pre-check)** — one cheap probe, last open shot, structurally-distinct risk, either outcome
   completes the paper. **DO THIS NEXT.**
2. **D (consolidate)** — the destination; begin assembling in parallel, finalize after A resolves.
3. C (copy-supp decomposition) — optional companion, only if A is NO-GO and budget remains.
4. B (weak-fact salvage) — do not run; footnote only.

---

## 3. THE GO/NO-GO PRE-CHECK FOR INJECTION (cheapest-first, deterministic routing)

Mirror the factual-editing pre-check discipline: **one cheap, judge-free probe that tests the single
load-bearing question before any selectivity campaign spend.** The flagship's lesson is baked in: test at the
real operating point, and test the *oracle edge-cut* (LBNR ceiling) so a null is unambiguously
"no load-bearing edge" vs "FRA can't reach."

### 3.1 The single a-priori question this probe must answer

**Is the (response-position query × injected-instruction-span key) attention edge BOTH (i) LOAD-BEARING — cut
it and compliance with the injection drops — AND (ii) CONTENT-SPECIFIC — the drop is tied to that span's
content, not a generic "ignore the injection" effect that also damages legitimate instruction-following?**

(i) kills the flagship's failure mode (is there an edge to cut at all, at the operating point?).
(ii) is injection's *own* crux (the distributional-redundancy / generic-routing risk, D4).

### 3.2 Spec (cheapest first; ≈1 L40 pod, judge-free)

- **Model + artifact:** `gemma-2-2b-it` + `gemma-scope-2b-pt-res-canonical` (residual SAE — same basis note as
  FACTEDIT_LOG §1: the QK score-cell edit needs a d_model-decoder SAE, NOT the hook_z attention SAE).
  Data: **Open-Prompt-Injection** (public). Take **N≈25–30 injection cases the model actually FOLLOWS at
  baseline** (ASR=1 on them — the operating point; the direct analog of "recalled facts P>0.3", and the
  explicit fix for the flagship's wrong-regime sampling).
- **GROUND-TRUTH metric (no judge):** exact-match — does the **injected target string** appear in the
  response (injection-following), and does the **legitimate task** still complete. Both are string-match.
- **Step 1 — ORACLE edge-cut (the LBNR ceiling, run FIRST, cheapest):** causally rank heads by the drop in
  injection-following when the (response × injected-span) score is cut to −1e4 (reuse `g_screen.causal_heads`
  + the score-cell `cut()`); take top ≤3. Report **R = fractional drop in injection-following from the
  oracle cut at those heads**, on the ASR=1 set. *This is the gate-(i) ceiling — if the whole edge can't
  move compliance, no FRA cell-edit will, and it is a load-bearing ceiling, not a reach ceiling (the exact
  disambiguation that resolved the flagship).*
- **Step 2 — CONTENT-SPECIFICITY / backup-cue check (gate ii):** after the oracle cut, (a) does
  injection-following stay suppressed or does a **backup cue restore it** (the redundancy trap — re-measure
  ASR with the edge cut and check it does not bounce back), and (b) does the SAME cut leave **legitimate
  instruction-following within 10%** (it must not be a generic "ignore all instructions" lesion). Held-out
  set: same system instruction, NO injection — legitimate compliance must survive.
- **Step 3 (only if Steps 1–2 pass) — FRA cell-edit reach:** FRA-decompose the edge → top-M SAE
  feature-pairs (injected-imperative-content key × response query) → build the score-delta → subtract c·delta
  at hook_attn_scores; require the cell-edit reproduces ≥50% of the oracle's injection-following drop at
  faithful c≈1–2 (the reach check, exactly as in the factedit pre-check).

### 3.3 The numeric gate (pre-registered, BEFORE the run)

**GO** iff ALL of:
- **(i) LOAD-BEARING:** oracle edge-cut drops injection-following by **R ≥ 0.50** (median) on the ASR=1 set,
  AND it is **NOT restored by a backup cue** (stays suppressed) — this is the LBNR pass the flagship faked.
- **(ii) CONTENT-SPECIFIC:** legitimate-instruction-following on the held-out no-injection set stays
  **within 10%** of baseline under the same cut (rules out a generic "ignore-injection" lesion).
- **(iii) REACH:** the FRA cell-edit reproduces **≥50%** of the oracle drop at c≈1–2.

**NO-GO** if (i) fails (no load-bearing injection edge → injection-routing is distributed/redundant like the
delimiter class — a clean pre-registered negative, the §3.4 injection falsifier), OR (ii) fails (the edge is
load-bearing but generic → FRA's cut is no more selective than a linear "ignore-injection" steer), OR (iii)
fails (reach ceiling — load-bearing but the SAE can't resolve the cell).

### 3.4 The right BASELINE for the full §4 test (only if GO)

If the pre-check is GO, the selectivity campaign runs FRA vs **(B1) a linear "ignore-injection" / instruction-
source steer** (DoM on the system-vs-injected contrast — the fair linear baseline) and **(B2) the
system-prompt-priority / prompt-hardening baseline** (the practical baseline). The win is **A = collateral(best
baseline) / collateral(FRA) ≥ 3×** at matched injection-following removal, on held-out (a) same-system-no-
injection legitimate-following and (b) other-injections-same-imperative-class. (No strong weight-edit baseline
exists → D8=1, the honest weakness vs the flagship's ROME.)

### 3.5 Why this pre-check is the highest information-per-dollar next move

Judge-free, one pod, reuses the exact substrate the flagship pre-check built (`g_screen`, the score-cell
`cut()`, the FRA decompose→cell-edit path, GemmaScope-res). Its outcome **deterministically routes the last
spend decision**: GO → the one selectivity campaign worth running this session; NO-GO → consolidate with a
*second* clean pre-registered negative (injection-routing is distributionally redundant), which only
*strengthens* the paper's bound. There is no losing branch.

---

## 4. HONEST FRAMING — what the evidence supports

### 4.1 Is the campaign trending toward "FRA's niche is NARROW"? — YES, and that IS the deliverable.

The pattern is now unmistakable and should be stated plainly:

> **FRA's behavioral-intervention advantage is concentrated on the INDUCTION/COPY/RETRIEVAL class — where
> the attended content is the ONLY source of the behavior — and the broad×broad extensions to real-LLM
> capabilities (factual recall; the direction-routed EM/sycophancy/refusal class) keep failing, by two
> distinct mechanisms: G-post output-direction routing (EM, sycophancy, refusal) and now regime-conditional
> redundancy (factual recall is G-post precisely on well-recalled facts).**

Every banked win (induction, copy-suppression, in-context backdoor, retrieval, acronym) shares one feature:
**the behavior's content has no parametric backstop — it is supplied by the context and must be attended.**
Every loss is a behavior where the content is either a baked-in direction (EM/sycophancy/refusal) or a
baked-in parametric fact (recall). **The unifying principle the campaign is converging on:** FRA wins iff
the behavior is *carried by attended context with no parametric/MLP backstop*. That is narrower than
"relational" — it is "relational AND context-supplied AND not overdetermined."

This is a *good* result, not a disappointing one: it is a sharp, mechanistic, a-priori-predictable
characterization of exactly when a feature-resolved QK edit beats a linear steer — which is what the campaign
set out to produce ("FRA wins HERE, not THERE, because X").

**The injection pre-check is the clean test of whether the niche is even narrower than that** — injection is
context-supplied-with-no-parametric-backstop (so it should win) BUT the *routing decision* may be
distributionally redundant (so it might lose). Its result sharpens the boundary either way: GO → the niche
includes a high-value safety application beyond toy circuits; NO-GO → the niche is essentially the
induction/copy/retrieval kernel and its direct backdoor descendant, full stop.

### 4.2 The strongest POSITIVE statement the evidence currently supports

> **FRA's feature-resolved QK cell-cut delivers a large, measured, a-priori-predictable SELECTIVITY advantage
> over linear steering and head-ablation — 15× (induction), ~25× (in-context backdoor), 516× (copy-
> suppression, transfers to a new context), ~1100× (retrieval), ~26,000× (acronym) lower collateral at
> matched on-target effect — on behaviors where the behavior is carried by a load-bearing attention edge over
> ATTENDED CONTEXT with no parametric/MLP backstop. An 8-dimension a-priori rubric (DIRECTION-vs-RELATION +
> four gates + the load-bearing-on-the-operating-regime condition) predicts which behaviors these are, and
> reproduces every prior negative (EM, sycophancy, refusal) and the new flagship negative (factual recall)
> as gate/regime failures.**

### 4.3 The strongest honest NEGATIVE the campaign supports

> **The advantage does NOT transfer to the headline real-LLM capability it was predicted to (factual recall):
> a structurally-ideal, G-score, broad×broad relation is REDUNDANT precisely on its operating regime —
> on facts the model actually recalls, the subject→object transport is overdetermined across heads/MLP and no
> single attention edge is load-bearing (oracle edge-cut R≈0.07 on strongly-recalled facts vs ≈0.32–0.39 on
> weak ones). The a-priori gates are necessary but not sufficient: load-bearingness must be verified at the
> application's operating point. FRA's intervention niche is therefore NARROW — the induction/copy/retrieval
> kernel and its in-context-backdoor descendant — not the broad real-LLM knowledge-editing or persona-control
> space, which is governed by directions and parametric redundancy that a single QK cell does not control.**

---

## 5. HANDOFF — what the orchestrator/evaluator does next

1. **EVALUATOR:** build + run the **injection load-bearing/content-specificity pre-check** (§3), one L40 pod,
   judge-free, log → a new `INJECTION_LOG.md`. Pre-register the §3.3 gate before launch. Oracle-cut FIRST
   (cheapest, the gate-(i) ceiling), at the ASR=1 operating point, with the backup-cue restoration check.
2. **In parallel — begin CONSOLIDATION (option D):** assemble the synthesis doc (positive bank + negative bank
   + rubric + §1.3 regime-redundancy refinement). This finishes regardless of the injection result.
3. **RED-TEAM (the in-flight w0vsbhiwa workflow):** assume the flagship NO-GO HOLDS; the only salvage it could
   surface is the weak-fact band (R≈0.32–0.39) — which §2/B already evaluates and REJECTS as a diluted
   consolation. If the red-team instead finds the pre-check sampled the wrong heads / wrong key-token (a
   machinery bug, not saturation), THAT would re-open the flagship — but FACTEDIT_LOG §3.2 already ran the
   per-fact full causal head-find on both candidate keys and still got R≈0.07 in the hi band, so a machinery
   confound is unlikely. Route on the red-team verdict only if it overturns the saturation finding.
4. **DECISION POINT after the injection pre-check:**
   - **GO** → run the single §4 injection selectivity campaign (FRA vs ignore-injection-steer + prompt-
     hardening), then consolidate.
   - **NO-GO** → consolidate immediately (two clean negatives now bound the method); optionally slot the
     copy-suppression QK decomposition (C) as a cheap positive companion if budget remains.
5. **Do NOT:** relaunch the flagship; run the weak-fact salvage as a headline; spend on induction/backdoor
   re-runs (banked).

**Net:** one more cheap deterministic probe, then write up. The campaign's deliverable — a sharp, a-priori,
mechanistically-explained map of FRA's narrow-but-real selectivity niche — is already in hand and only gets
sharper from the injection result, whichever way it lands.
