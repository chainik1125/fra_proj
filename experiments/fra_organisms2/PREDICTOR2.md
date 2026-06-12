# PREDICTOR2 — the sharpened, operational WIN-predictor for FRA organism-hunt v2

*THEORY agent, fra_organisms2 campaign (`CAMPAIGN.md`), 2026-06-12. This is the SCORING INSTRUMENT the
screen agent uses and the PRE-REGISTRATION of WIN/LOSE per candidate class. It does not rewrite the v1 D1–D8
rubric (`../fra_organisms/SCREENING_RUBRIC.md`) — it sharpens it into ONE predictor with measurable proxies,
absorbing the four v1 refinements (`../fra_organisms/SYNTHESIS.md` §2.3), the magnitude law and the proven
kernel (`../fra_win/FRA_PRINCIPLE.md`), and the broad×broad existence proof
(`../fra_hierarchy/THEORY_HIERARCHY_v2.md` §4). Written to inform, not persuade (`writing_instructions.md`):
every threshold is mechanical, every per-class call is a falsifiable number, and §5 states what kills the
hunt. The v1 negatives (recall D3, injection 1.6×) are the calibration: the predictor must reproduce them.*

---

## 0. THE UNIFIED PREDICTOR (read this first — one paragraph)

**An FRA cell-cut beats the best-tuned linear steer by ≥2× selectivity iff the behavior is a bilinearly-gated
attention edge that is (i) LOAD-BEARING AT THE ANSWER STEP, (ii) over CONTEXT-SUPPLIED content with NO
parametric backstop, (iii) on a NON-RECURRENT conjunction reachable by one SAE cell, AND (iv) whose endpoints
are BROADLY REUSED so a linear steer pays collateral — and each of the four has a cheap, operating-regime-
correct proxy that must clear its threshold before any selectivity run.** Concretely: (i) measured by the
**oracle answer-step edge-cut effect R at the application's own operating point** — zero attention to the
key-span from the *answer/decode query positions*, must give R ≥ 0.6 (GO) / ≥ 0.8 (STRONG); if the load-
bearing read is at prefill (R_gen ≈ 0 but R_prefill = 1, the injection signature) the answer-step edge is a
redundant late read → DOWNGRADE. (ii) measured by **token-novelty**: the answer content must be a
context-only token absent from parametric memory — closed-book recall accuracy < 10% on the key content
(a novel canary/name/value the model cannot produce without attending) → the attention is load-bearing *by
construction*; a well-recalled fact (closed-book > 70%) is MLP-pre-baked → FAIL (the recall negative). (iii)
measured by the **D3 sibling-bleed probe**: apply the cell-cut and measure the same-edge effect on
sibling instances that share one endpoint; bleed < 15% (GO) / < 5% (STRONG); bleed > 40% = relation-keyed
recurrence → FAIL (recall bled 49–59%). (iv) measured by the **behavioral collateral A on a HARD sibling
control** — NOT the naive corpus firing-rate ratio (that proxy FAILED in v1: sink-feature contaminated and
degenerate, `FRA_PRINCIPLE.md` "what failed"); reuse is a *cross-context behavioral* property, so probe it as
A = collateral(best-tuned linear at matched removal) / collateral(FRA cut) on held-out instances where the
endpoints legitimately recur, requiring A ≥ 2 (GO) / ≥ 3 (the win bar) / ≥ 10 (STRONG). **STRONG-GO = all four
proxies in their STRONG band; GO = all four pass their GO threshold with ≥2 in STRONG; any one below GO →
pre-registered NO-GO, reported as a boundary-map control.** The magnitude term that separates a 1.6× null from
a ≥2× win is (iv): **A ≈ reuse(endpoint marginal) / reuse(conjunction)** — large only when the linear steer's
direction is so broadly reused that it cannot avoid collateral while the conjunction is genuinely rare. The
hunt targets the upper-right corner: **closed-book < 10% (novel) × sibling-bleed < 5% (instance-unique) ×
answer-step R ≥ 0.8 × A ≥ 3** — every banked win (induction 15×, copy-suppression 516×, in-context backdoor
25×) sits there; every v1 negative fails exactly one coordinate.

---

## 1. THE FOUR PROXIES — how to measure each cheaply, a-priori or with a one-pod probe

The four win-conditions (`CAMPAIGN.md`) are the spine. v1 proved each is necessary by failing one each. The
sharpening here is the **measurable proxy + threshold** for each, with the operating-regime discipline baked
in (the recall false-GO lesson: a proxy measured off the application's own regime lies, `SYNTHESIS.md` §2.3.2).

| # | win-condition | cheap proxy (one-pod, judge-free) | GO | STRONG-GO | hard FAIL (= the v1 negative it reproduces) |
|---|---|---|---|---|---|
| **P1** | load-bearing **at the answer step** | **oracle answer-step edge-cut R** = behavior-drop from zeroing key-span attention *from decode/answer query positions only*, at the app's own operating regime. Report R_gen vs R_prefill separately. | R_gen ≥ 0.6 | R_gen ≥ 0.8 | R_gen < 0.3 while R_prefill = 1 → upstream/redundant late read (**injection**); R < 0.3 everywhere → MLP backstop / G-post (**sycophancy, recall-on-strong-facts**) |
| **P2** | context-supplied, **no parametric backstop** | **closed-book recall** of the key content with the context removed (exact-match / generation accuracy) | < 10% | = 0% (a genuinely novel token: canary, made-up name, fresh numeral) | > 70% → the content is MLP-pre-baked, attending it is redundant (**factual recall** flagship NO-GO) |
| **P3** | **non-recurrent** conjunction (one cell, instance-specific) | **D3 sibling-bleed**: cut effect on siblings sharing ONE endpoint (same role, different instance) / cut effect on target | bleed < 15% | bleed < 5% | bleed > 40% → relation/role-keyed, recurs across siblings (**recall 49–59%, box-retrieval A=1.9×**) |
| **P4** | endpoints **broadly reused** ⇒ linear pays collateral (the magnitude term) | **behavioral A** = collateral(best-tuned linear @ matched removal) / collateral(FRA cut), on a HARD sibling control where endpoints legitimately recur. **NOT** the naive reuse-ratio (v1 FAILED). | A ≥ 2 | A ≥ 10 | A < 2 → linear is already as surgical (the **injection 1.6×** null: "comply" direction not reused enough for the linear to bleed) |

**The operating-regime rule (binding, from `SYNTHESIS.md` §2.3.2).** Every proxy is measured on the
application's OWN eval distribution, not a convenience subset. v1 recall's R=+0.86 false-GO came from probing
a weak fact (P=0.039); the same edge was R≈0.07 on the strongly-recalled facts the editing use case actually
targets. P1 and P2 in particular must sample the regime where the behavior is *supposed* to fire.

**The timing split (binding, from `SYNTHESIS.md` §2.3.4).** P1 must break R into R_gen (answer-step) and
R_prefill. A behavior that is R_prefill=1, R_gen≈0 (injection) is attention-routed but NOT at the answer step —
win-condition 3 is violated, and the win caps sub-threshold even though P2/P3 may pass. A clean win needs the
load-bearing read AT decode (induction/copy-suppression: the query is the about-to-emit token).

**Why P4 is behavioral, not a single-pass number (the most important v1 method lesson).** The naive
magnitude proxy — corpus firing-rate(key-feature)/firing-rate(conjunction) — **FAILED in v1**
(`FRA_PRINCIPLE.md` "what failed"): on Gemma a handful of attention-sink features (e.g. 15887, rate 1.00)
dominate FRA magnitude on *every* edge, making the ratio degenerate and behavior-independent; and the property
that actually separates a win from a no-win (legitimate reuse × routing) is genuinely a *cross-context
behavioral* fact, not a single-forward-pass scalar. So A is measured the only way that worked: the held-out
collateral ratio at matched on-target removal. This is the screen's most expensive proxy (one small pod) but
it is the one that the 1.6× null and the 15–516× wins actually came from.

**How the proxies compose into the verdict.** P1, P2, P3 are **gates** (a fail on any one predicts LOSE — they
negate answer-step-load-bearing, context-supply, and non-recurrence respectively). P4 sets the **magnitude** of
the win and is the GO/win-bar threshold. **STRONG-GO** = P1≥0.8 ∧ P2=0 ∧ P3<5% ∧ P4≥10. **GO** = all of
{P1≥0.6, P2<10%, P3<15%, P4≥2} with at least two in the STRONG band. **NO-GO** = any gate below its GO row →
pre-registered negative control (valuable: it extends the boundary map at a one-pod cost, not a full campaign).

---

## 2. WHY THE SYNTHETIC WON BUT RECALL / INJECTION DIDN'T — and what a real organism must inherit

### 2.1 The synthetic won because its conjunction was INSTANCE-UNIQUE, not role-recurrent

The broad×broad synthetic (`THEORY_HIERARCHY_v2.md` §4) is the existence proof: a regression-fitted multi-cell
edit cut a [persona]×[content] link with Y-collateral 0.03 ≈ oracle 0.00, ~30× cleaner than any per-position
method. The decisive property — and the one a real organism MUST inherit — is that the synthetic's **keys were
unique per instance**: each constructed (query-feature × key-feature) pairing had support disjoint from every
sibling on the eval distribution. The cut edited exactly that support and nothing legitimate recurred there
(P3 sibling-bleed ≈ 0 by construction).

**Recall lacked exactly this.** Recall's conjunction is (subject-content query × **relation**-content key). The
relation ("…'s mother", "…'s capital") is a **recurrent role**: it is shared by every subject that has that
relation. So the discriminating endpoint is generic, the cell-cut fires on every sibling subject's same-
relation fact, and P3 bleed measured 49–59% (1/20 selective) — a *reachable, load-bearing* edge (R: drop 0.52,
rank-flip 0.60, so P1 passed) that nonetheless fails P3. **This is the translation: a real winner's
discriminating endpoint must be a DISTINCTIVE, instance-bound CONTENT token (a specific name, a fresh value, a
novel trigger), not a role/slot/relation that recurs across siblings.** The synthetic had unique keys; recall
had a recurrent role-key; the hunt must find an organism with unique keys *in the wild*.

The operational restatement for the screen: **P3 is the synthetic-vs-recall discriminator.** Anything whose
"subject"/"answer-slot"/"queried role" is the thing being cut will recur and bleed. Anything whose cut is
keyed on a *content token that appears in exactly this instance and nowhere among the siblings* will not. The
candidate's single most important a-priori question is: *is the discriminating endpoint a unique content token
or a recurrent role?* (D3 of the v1 rubric, now with a measured bleed threshold.)

### 2.2 Injection was only 1.6× because the MAGNITUDE term (P4) was small — two compounding reasons

Injection passed the gates that the v1 a-priori rubric scores: it is attention-routed (R_prefill = 1.0),
content-specific (legit collateral 0.094), FRA-reachable (100% of oracle drop), localized (L10H7/L18H6, ≤3
heads). Yet the §4 head-to-head gave only **1.6× hard-legit retention at the fair operating point** (FRA 1.00
vs tuned linear L12 0.626; `SYNTHESIS.md` §1). Two compounding reasons, both in the magnitude term P4 — and
both are now pre-registration lessons:

1. **The load-bearing read is at PREFILL, not the answer step (P1 timing fail).** R_gen ≈ 0, R_prefill = 1
   (`INJECTION_LOG.md` §4.2). The answer-step edge alone is a redundant late read; the routing decision has
   already propagated forward in the residual stream by decode. A behavior whose causal attention is upstream
   gives the downstream linear steer a *second bite* — it can intercept the already-transported signal — which
   is exactly why a tuned linear DoM recovers most of the selectivity. **Win-condition 3 (consumed at the
   answer step) is the one injection violates**, and it is why the gates-pass did not deliver the win.

2. **reuse(endpoints) was not high enough — the "comply" direction is not broadly reused, so the linear paid
   little collateral.** A ≈ reuse(marginal)/reuse(conjunction): for the linear steer to bleed badly (large A),
   its direction must be reused across many *legitimate* contexts. The injection-removal direction
   ("don't-follow-injected-imperative") turned out to be reasonably specific — a tuned linear could push it
   without wrecking legitimate instruction-following on most prompts, so its collateral was low (0.374 abs on
   the hard control) and A capped at 1.6×. The linear was **too competitive** because the endpoint it pushes is
   not reused widely enough to force collateral. **The lesson for the hunt: an organism wins big only when the
   linear steer's only available handle is a MASSIVELY reused direction** (e.g. the generic "retrieve/copy the
   attended token" direction, reused at every copy — induction's 15×; the "this-head's-output" direction,
   reused at every suppression — copy-suppression's 516×). Pick organisms where the linear has no choice but to
   push a broad direction.

**Synthesis of §2.** The synthetic won on P3 (unique keys → zero bleed) AND P4 (broad persona × broad content,
high A by construction). Recall failed P3 (recurrent relation-key). Injection failed P1-timing (prefill, not
answer-step) AND had a small P4 (over-specific linear handle → competitive baseline). **A real winner must
clear BOTH the P3 instance-uniqueness bar AND the P4 broad-endpoint bar — the two that the synthetic had and
the two real organisms each lost one of.**

---

## 3. PER-CANDIDATE-CLASS PRE-REGISTERED PREDICTION

For each of the five hunt classes (`CAMPAIGN.md`): WIN/LOSE, the decisive proxy/term, the predicted four-proxy
profile, and a falsifiable number. Registered BEFORE any run. Ranked by P(≥2× win) in §3.6.

### 3.1 IN-CONTEXT VARIABLE BINDING / STATE-TRACKING — **predict WIN (the closest analogue of the synthetic)**
*"a=7, b=3, … what is a?" / entity-state tracking.*
- **Verdict: WIN.** Decisive: **P3 (instance-unique key) + P4 (broad endpoints).** The binding (variable-name
  query × value-token key) is the literal real-LLM instantiation of the synthetic's unique-key structure: each
  variable name is a *distinct content token bound to a distinct value within this prompt*, recurring nowhere
  among siblings (the other variables). The answer-step read is at decode ("what is **a**?" → attend back to
  a's binding AND emit) → P1 at the answer step, unlike injection. The value is context-supplied with no
  parametric backstop (the assignment is novel) → P2 = 0. The linear steer's only handle is the broad
  "retrieve-the-bound-value" direction, reused by *every* variable → high collateral → large A.
- **Proxy profile:** P1 ≥ 0.8 (R_gen), P2 = 0 (novel assignment), P3 < 5% (sibling variables untouched),
  P4 ≥ 5. **STRONG-GO.**
- **Falsifiable number:** cutting (a-query × a's-value-key) flips/suppresses the retrieval of a's value at
  ≥ 70% while leaving b's, c's, … retrieval within 5% (P3 bleed < 5%); FRA collateral on sibling variables
  ≥ 3× lower than a tuned linear "value-retrieval" steer at matched a-suppression. **Falsified if** binding is
  resolved POSITIONALLY (the cut keys on the *slot* "first variable" not the *content* "a") → bleed onto the
  same-position sibling, A→~1.9× (the box-retrieval precedent — the one real risk, and exactly P3's failure
  mode). The cheap discriminator: shuffle variable order between extract and eval; if the cut still lands on
  the right variable, it is content-keyed (win); if it follows position, role-keyed (lose).

### 3.2 RAG-ATTRIBUTION / CONTEXT-GROUNDED RETRIEVAL — **predict LOSE-to-MARGINAL (re-runs injection)**
*Multi-passage context; cut the answer's reliance on ONE poisoned passage, preserve the others.*
- **Verdict: LOSE (or ≤2× marginal).** Decisive: **P1-timing + P4 (same as injection).** RAG is mechanistically
  the same as prompt-injection with a fiddlier eval (`SYNTHESIS.md` §3 rejected it on exactly this). The
  passage-grounding read is plausibly upstream (the passage is comprehended at prefill and propagated), and the
  answer attends to passages partly by *slot* ("the retrieved-context region") → P3 generic-answer-slot risk
  (the v1 RAG D3 crux) AND P1-timing risk. The linear "ignore-this-source" handle is over-specific (injection's
  small-P4 problem recurs).
- **Proxy profile:** P1 R_gen 0.3–0.6 (prefill-weighted, like injection), P2 < 10% (poison content can be
  novel — its one redeeming coordinate), P3 15–40% (slot-recurrence risk), P4 ≈ 1.5–2. **GO-borderline →
  likely sub-threshold NULL.**
- **Falsifiable number:** FRA passage-A reliance ↓ ≥ 60% @ passage-B reliance within 10% AND A ≥ 3 — **predicted
  to MISS A ≥ 3** (lands at the injection ~1.6× band). **Falsified (would surprise) if** passages are resolved
  by *distinctive content* not slot (P3 < 5%) AND the answer-step read is at decode (P1 R_gen ≥ 0.8) — possible
  for a single-distinctive-fact poison, but the prior is the injection re-run.

### 3.3 DISTINCTIVE-TRIGGER BACKDOOR (in-context, multi-token) — **predict WIN (generalizes the banked anchor)**
*Published poisoning organism with a distinctive multi-token trigger in the prompt.*
- **Verdict: WIN.** Decisive: **P2 (novel trigger) + P3 (unique trigger content).** This is the banked
  in-context backdoor (A≈25×, `FRA_PRINCIPLE.md`) lifted to a published multi-token trigger. By construction:
  the trigger→payload link is induction-routed (P1 at the answer step), the trigger is a distinctive
  context-supplied token with no parametric backstop (P2 = 0), and it recurs in no sibling clean prompt
  (P3 = 0). The linear handle is the broad "emit-payload" / "follow-induction" direction → high A.
- **Proxy profile:** P1 ≥ 0.8, P2 = 0, P3 ≈ 0, P4 ≥ 10. **STRONG-GO.**
- **Falsifiable number:** cutting (trigger-content × payload-context) suppresses payload emission ≥ 80% @
  < 5% clean-prompt collateral, A ≥ 10 vs a tuned ActAdd "suppress-payload" steer. **Falsified if** the
  multi-token trigger has no recoverable single/family cell (the trigger is distributed across many cells and
  the H2 family-union doesn't lift — the only real risk, on the *reach* axis P4-as-feasibility, not P1–P3).

### 3.4 COREFERENCE / ENTITY-ATTRIBUTE BINDING — **predict LOSE (role-keyed, the v1 binding negative)**
*Multi-entity narrative; bind the right attribute to the right named entity; cut one mis-binding.*
- **Verdict: LOSE.** Decisive: **P3 (role-recurrence) + P1 (weak/distributed edge).** v1 scored binding S=9
  borderline-NO-GO and measured it directly on gpt2: the binding edge passed CCF (0.905) but was **weak**
  (attn 0.38, LBNR-R = −0.89, distributed — `SCREENING_RUBRIC.md` §3.3). The discriminator is plausibly the
  *entity-slot* (a generic role) not distinctive content → P3 sibling-bleed high. Coreference resolves
  pronouns by recency/syntax (role/position), the canonical P3-recurrence failure.
- **Proxy profile:** P1 R 0.3–0.5 (weak/distributed), P2 variable, P3 > 40% (role-keyed), P4 ≈ 1.9. **NO-GO.**
- **Falsifiable number:** re-bind target pair @ ≥ 50% with < 10% collateral on other bindings — **predicted to
  FAIL**, A ≈ 1.9× (box-retrieval floor). **Falsified if** the binding query is the *attribute's content
  identity* (distinctive) rather than its slot — the same content-vs-role flip as §3.1, but the prior
  (measured, weak distributed edge) is a loss.

### 3.5 IN-CONTEXT DEFINITION / TRANSLATION — **predict WIN (novel context-supplied mapping)**
*"in this dialect X means Y; use X." A novel, distinctive, context-defined mapping.*
- **Verdict: WIN (provisional — second-strongest after binding).** Decisive: **P2 (novel mapping) + P3
  (unique X→Y).** The definition is supplied in-context with no parametric backstop (a made-up word or a
  redefined token → P2 = 0), the mapping is distinctive and instance-unique (X means Y *here*, recurring in no
  sibling → P3 low), and using X at the answer step requires attending back to its definition (P1 at decode,
  induction-adjacent). The linear handle is the broad "apply-the-definition" / "substitute" direction.
- **Proxy profile:** P1 ≥ 0.7, P2 = 0, P3 < 10%, P4 ≥ 3. **GO / STRONG-GO if the term is fully novel.**
- **Falsifiable number:** cutting (X-query × Y-definition-key) suppresses the model's use of the defined
  meaning ≥ 60% while leaving a second in-context definition (W means Z) within 10%, A ≥ 3 vs a tuned
  "definition-substitution" steer. **Falsified if** the model resolves the definition via a generic
  "most-recent-definition" slot (P3 recurrence) rather than X's content, OR if the mapping is well-known
  enough to be MLP-backed (P2 > 70% — use genuinely novel terms to avoid this).

### 3.6 RANKING by P(≥2× win) + the calibration controls

| rank | candidate class | verdict | decisive proxy/term | predicted profile (P1/P2/P3/P4) | headline falsifiable number | P(≥2× win) |
|---|---|---|---|---|---|---|
| **1** | **in-context variable binding / state-tracking** | **WIN** | P3 unique-key + P4 broad | 0.8 / 0 / <5% / ≥5 | sibling-var retention within 5% @ a-suppression ≥70%, A ≥ 3 | **~0.65** |
| **2** | distinctive-trigger backdoor (in-context) | WIN | P2 novel + P3 unique | 0.8 / 0 / ~0 / ≥10 | payload ↓≥80% @ <5%, A ≥ 10 | ~0.60 |
| **3** | in-context definition / translation | WIN | P2 novel + P3 unique | 0.7 / 0 / <10% / ≥3 | defined-meaning ↓≥60% @ 2nd-def within 10%, A ≥ 3 | ~0.50 |
| 4 | RAG-attribution | LOSE/marginal | P1-timing + small P4 | 0.3–0.6 / <10% / 15–40% / 1.5–2 | A predicted < 3 (injection re-run) | ~0.20 |
| 5 | coreference / entity-attribute binding | LOSE | P3 role + P1 weak | 0.3–0.5 / var / >40% / 1.9 | A ≈ 1.9× (box floor), re-bind fails | ~0.10 |
| — | factual recall *(v1 control)* | NO-GO | P3 bleed 49–59% | 0.5 / >70% / >40% / — | reproduces recall NO-GO | — |
| — | prompt-injection *(v1 control)* | NULL 1.6× | P1-timing prefill | 0/1(prefill) / <10% / <10% / 1.6 | reproduces injection 1.6× | — |
| — | EM / sycophancy *(v1 controls)* | NO-GO | direction / G-post | R<0.3 everywhere | reproduces (<15% effect) | — |

**The single strongest prediction.** *In-context variable binding is a STRONG-GO and the only class that
clears all four STRONG bands a-priori* — it inherits the synthetic's instance-unique-key property (P3) that
recall lacked AND the broad-endpoint property (P4) that injection lacked, with the load-bearing read at the
answer step (P1) that injection lacked. **Falsifiable headline: an FRA (variable-name × bound-value) cell-cut
suppresses the target variable's retrieval ≥70% while holding every sibling variable within 5%, at ≥3× lower
collateral than the best-tuned linear value-retrieval steer at matched suppression — and is FALSIFIED, landing
at A≈1.9×, iff binding is resolved positionally (the order-shuffle control follows slot, not content).**

---

## 4. THE WIN-TEST, REFINED FOR THE HUNT

Builds on `SCREENING_RUBRIC.md` §4 (the shared evaluator protocol); the v2 additions are the operating-regime
pre-check, the timing split, and the HARD sibling control.

**Step 0 — Operating-regime pre-check (the cheap gate, ONE pod, before any selectivity run).**
Sample the application's OWN eval regime (not a convenience subset — the recall false-GO lesson). Measure the
four proxies P1–P4. Specifically: (P1) oracle answer-step edge-cut R, split R_gen vs R_prefill; (P2) closed-book
recall of the key content; (P3) sibling-bleed of the oracle cut. **Proceed to selectivity ONLY if P1 R_gen ≥
0.6, P2 < 10%, P3 < 15%.** A fail here is a one-pod pre-registered negative — file it and move on. (This gate
killed recall pre-campaign and correctly GO'd injection in v1; it is the spend router.)

**Step 1 — Pin the matched on-target operating point.** Choose a target removal (e.g. target-retrieval → < 0.2).
Tune EVERY method — FRA edit scale c, AND the linear baseline (sweep layers × α, give it its best shot) — to
hit the SAME on-target level. Collateral is comparable only at matched removal (the binding A=5133× artifact
came from un-matched removal). Use the **regression-fitted multi-cell edit**, never a naive single-cell cut
(the softmax-redistribution correction, `THEORY_HIERARCHY_v2.md` §4.1) — a naive cut just MOVES the behavior
onto a sibling.

**Step 2 — Selectivity vs the BEST-TUNED linear on a HARD control.** The control is **sibling instances that
resemble the target** (the other variables, the clean passages, the second definition) — where the endpoints
legitimately recur, so the linear pays the collateral the magnitude law predicts. Plus a held-out general-
capability battery (global-damage check). All metrics GROUND-TRUTH (exact-match retrieval / payload / defined-
meaning), no LLM judge.

**Step 3 — The win bar (pre-registered, unchanged from v1 so results are comparable).**
- **WIN** iff **A ≥ 2× selectivity gap on the sibling control AND ≥ +0.15 absolute retention AND FRA retains
  > 10% on the hard control**, at matched on-target effect, with no worse global damage. *(The campaign's
  headline target is ≥2×; the v1 injection 1.6× / +0.37-abs was a NULL on the ratio despite clearing the
  absolute — both must hold.)*
- **STRONG WIN** iff A ≥ 10× (the magnitude-law regime — induction/copy-suppression band).
- **NO WIN** iff A < 2× (linear is already as surgical — the injection diagnosis), OR FRA cannot reach matched
  removal at faithful scale c ≈ 1–2 (reach ceiling — injection's 0.73 cap).

**Step 4 — Mandatory falsifiers / confound guards (judge-free, the EM-hardened lenses).**
- **Timing audit (NEW):** confirm R_gen ≥ 0.6 — if the win rides on R_prefill only, it is the injection
  upstream-redundancy pattern and the answer-step claim is false.
- **Order-shuffle / role-vs-content audit (NEW, the P3 falsifier):** permute the sibling instances' positions;
  if the cut follows position not content, the conjunction is role-keyed → the win is a slot artifact, not
  instance-unique (the binding/box risk).
- **Redundancy probe (LBNR):** after the cut, check the behavior is not restored by a backup cue — if it is,
  the "win" is a precise-but-inert cut.
- **Matched-removal + coherence audit:** every method at the same on-target level; collateral on coherent rows
  only; FRA must not break fluency.
- **C1 base-reversion OLS:** regress per-item effect_FRA ~ effect_off + effect_on; require a significant
  behavior-specific partial (rule out reversion to the parametric prior — EM's flat band).

---

## 5. WHAT FALSIFIES THE HUNT (so we recognize a clean negative)

1. **The STRONG-GO (variable binding) returns A < 2× at matched removal.** If the closest real analogue of the
   synthetic — instance-unique keys, broad endpoints, answer-step read — fails the win bar, the synthetic
   broad×broad advantage does NOT transfer to real LLMs *even on its ideal real instance*, and the v1
   meta-conclusion (`SYNTHESIS.md` §4) is the final word: FRA's home turf is only the banked toy kernel.
2. **The four proxies don't track measured A across candidates.** If binding (predicted A≥5), RAG (predicted
   ~1.6×), and coreference (predicted ~1.9×) come back with indistinguishable measured A, P1–P4 have no
   predictive content and the predictor adds nothing over running everything.
3. **A predicted-LOSE class (RAG / coreference) WINS ≥2×.** Falsifies the converse — the P1-timing / P3-role
   gates are mis-specified. A clean, informative surprise.
4. **Every class fails P1 R_gen < 0.3 (the injection/recall upstream pattern).** Then real-LLM relations are
   *all* either prefill-routed or MLP-backed at the answer step, and the answer-step kernel is empty outside
   the toy induction case — the DIRECTION-vs-RELATION reframe survives but the answer-step condition is almost
   never met in the wild.

**Honest framing (`writing_instructions.md`).** The deliverable is the predictor + the ranked measured table,
WIN or LOSE. If variable binding clears ≥2×, the claim is "FRA's selectivity advantage transfers to a real,
reasoning-relevant organism (in-context binding) — the instance-unique-key + broad-endpoint structure the
synthetic identified is realizable in the wild, ≥3× more selective than a tuned linear steer." If it lands at
the injection band, the claim is the sharp negative: "even the ideal real analogue of the synthetic only
modestly beats a tuned linear steer; the broad×broad advantage is confined to the toy kernel." Either is a
result; the predictor is built so the measurement (Step 0 regime-correct pre-check + Step 4 guards) cannot
manufacture the answer.

---

## 6. THE ONE PROXY THAT BEST SEPARATES WIN FROM LOSE

**P3 sibling-bleed measured at the application's own operating regime — the D3 test.** It is the single
coordinate on which every v1 outcome separates: the banked wins and the synthetic have bleed ≈ 0 (instance-
unique keys); recall failed at bleed 49–59% (relation-keyed); binding/box fails at the ~1.9× role floor; and it
is the exact property that distinguishes the predicted v2 winners (binding/backdoor/definition: unique content
keys) from the predicted losers (RAG/coreference: recurrent slots). It is cheap (one oracle cut + a sibling
re-run, no judge, no baseline tuning) and it is causal. The order-shuffle audit (§4 Step 4) makes it
un-gameable. If a single number gates the spend, it is sibling-bleed < 15%.

---

## HANDOFF TO SCREEN / EVALUATOR

- **Screen:** score each candidate on P1–P4 with the §1 thresholds; the verdict is GO/STRONG-GO/NO-GO from §1,
  the tier from the §3.6 ranking. Top pick a-priori: **in-context variable binding** (STRONG-GO, P≈0.65).
- **Evaluator:** run the §4 Step 0 one-pod pre-check FIRST (P1 split R_gen/R_prefill, P2 closed-book, P3
  sibling-bleed) on the chosen organism's own regime; proceed to selectivity only on a clean gate; the
  make-or-break measurement is **P3 sibling-bleed + the order-shuffle role-vs-content audit** on a ground-truth
  retrieval metric, then A vs the best-tuned linear on the sibling control.
- **Do NOT** reinstate the naive corpus-reuse-ratio as a magnitude proxy (it FAILED in v1 — sink-contaminated,
  degenerate); P4 is the behavioral collateral ratio only.
