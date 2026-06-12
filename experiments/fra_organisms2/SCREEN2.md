# SCREEN2.md — score the top-3, LOCK the #1 pick, hand the evaluator a runnable pre-check

*SCREEN agent, FRA organism-hunt v2 (`CAMPAIGN.md`). Consumes `ORGANISMS2.md` (catalog), `PREDICTOR2.md`
(the P1–P4 win-predictor + per-class pre-registration), and the v1 boundary map (`../fra_organisms/SYNTHESIS.md`).
The design is HARNESS-COMPATIBLE with `../fra_organisms/jobs/injection_selectivity.py` +
`injection_precheck.py` (the generic FRA cell-cut: residual `gemma-scope-2b-pt-res-canonical` →
query/key feature decode via `W_dec` → per-head pre-softmax score-delta subtracted at
`blocks.{L}.attn.hook_attn_scores`). I do NOT run GPU or touch git — the orchestrator commits.*

---

## 0. VERDICT UP FRONT (decisive)

**LOCKED #1 = A1, Mixing-Mechanisms in-context bound-entity retrieval, on gemma-2-2b-it, with a MINIMAL
SELF-BUILT binding eval (not the mixing-mechs repo).** It is the only top-3 candidate that clears all four
win-conditions AND all four P1–P4 STRONG bands a-priori, and it owns the single highest-value falsifier in the
whole campaign: the **order-shuffle lexical-vs-positional audit** (the exact P3 failure mode that capped the v1
box-retrieval precedent at A≈1.9×). The first thing the evaluator runs is the **operating-regime pre-check on
the lexical/positional DISAGREEMENT regime** — P1 answer-step R_gen, P3 sibling-bleed, and the order-shuffle
content-vs-position audit, all on a judge-free exact-match next-token metric. GO/NO-GO gates in §3.

Runnability call (honest): **self-build the eval, do not depend on `github.com/yoavgur/mixing-mechs`.** We only
need the *dataset structure* ("Ann has the ale, Joe has the pie, … Who has the pie? → Joe"), which is ~40 lines
of templating; the repo README says "codebase still being finalized" and its `CausalAbstraction/` interventions
are NOT our intervention (we own the FRA QK cell-cut). Pulling the repo adds an integration-risk dependency for
zero benefit. We reuse the paper's *finding* (retrieval mixes a lexical and a positional mechanism, and they can
be made to disagree) as the operating-regime design, not its code.

---

## 1. SCORING TABLE — top-3 on the four conditions + P1–P4 + runnability

Scoring key per cell: ✓ = clears a-priori, ~ = borderline/at-risk, ✗ = fails. P1–P4 use the `PREDICTOR2` §1
thresholds (P1 R_gen≥0.6 GO / ≥0.8 STRONG; P2 closed-book <10% GO / =0 STRONG; P3 sibling-bleed <15% GO / <5%
STRONG; P4 behavioral collateral ratio A≥2 GO / ≥3 win-bar / ≥10 STRONG). P1–P4 columns are the **predicted**
profile (the screen scores the prior; the evaluator measures it).

| candidate | C1 load-bearing edge | C2 no backstop | C3 answer-step | C4 non-recurrent | P1 (R_gen) | P2 (closed-book) | P3 (sib-bleed) | P4 (A) | runnability | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| **A1 ★ Mixing-Mechs bound-entity retrieval** | ✓ lexical edge | ✓ permuted/fictional bindings | ✓ "Who has X?" reads at decode | ✓ entities distinct per instance | **0.8** | **0** | **<5%** | **≥5** | **HIGH** (self-build ~40-line gen; drops into harness) | **STRONG-GO (LOCK)** |
| **B1 ICLAttack in-context backdoor** | ✓ induction edge | ✓ planted in demos | ✓ label emitted at answer | ✓ *if* distinctive trigger | 0.8 | 0 | ~0 | ≥10 | HIGH (no-finetune ICL prompts) | GO (fallback #1) |
| **B3/WinoDict nonce-binding** *(=ORGANISMS2 C1)* | ~ plausible | **✓ strongest (word doesn't exist)** | **~ RISK (upstream pre-bake)** | ✓ fresh nonce | 0.7 | 0 | <10% | ≥3 | HIGH (WSC+gloss synth) | GO-conditional (fallback #2) |

**Down-ranked (scored, not chosen) — confirms the ranking is principled, not arbitrary:**

| candidate | killing coordinate | predicted | why down-ranked |
|---|---|---|---|
| A2 Filter heads (list-select) | **P3** — criterion is a *generic filter role*, recurs across instances | bleed >40% | the factual-recall D3 trap; A1's per-instance distinct bindings dodge it. Use only as a **D3 stress-task** for A1. |
| A3 Variable-binding circuit (Davies) | runnability + **C1 (+1 MLP)** | — | LLaMA-13B off gemma-2-2b budget; the "+1 MLP" head is a possible G-post backstop. A1 is the same win-shape, on-budget. |
| D1 RAG passage attribution | **P1-timing + P4** | R_gen 0.3–0.6, A 1.5–2 | mechanistically = injection (prefill-routed, generic answer-slot). v1 PLANNING already rejected as "dominated by injection." |
| E1 Multi-entity box binding | **C1 (weak edge) + P3 (role)** | R 0.3–0.5, A≈1.9 | v1 measured it: attn 0.38, LBNR-R=−0.89, distributed. **A1 strictly supersedes** (public causal decomposition + a lexical/positional regime to run the pre-check that E1 lacked). |

**The two calibration controls the predictor must reproduce** (so the screen is not just GO-stamping): factual
recall = NO-GO (P3 bleed 49–59%, relation-keyed); prompt-injection = NULL 1.6× (P1-timing prefill, small P4).
A1 is chosen precisely because it inherits the synthetic's instance-unique-key (the property recall LACKED → P3)
AND the answer-step read (the property injection LACKED → P1), with broad endpoints (the property injection
LACKED → P4).

---

## 2. WHY A1 IS #1 — the magnitude-law argument, made concrete

The whole campaign is a bet on the magnitude law **A ≈ reuse(endpoints) / reuse(conjunction)**. A1 is the
textbook HIGH×LOW instance, and uniquely so among real organisms:

- **reuse(endpoints) is maximal.** The query endpoint is "an entity name as the queried slot" and the key
  endpoint is "retrieve a bound value" — these two directions are reused by *every single binding in every
  instance*. A linear DoM steer built to suppress Ann's binding has no handle except this broad
  "retrieve-the-bound-value" direction, so it must bleed onto Joe/Pete/Tim. **There is no linear "Ann's-value-only"
  direction** because the value identity lives in the conjunction, not either endpoint. This is exactly the
  property injection LACKED (its "comply" direction was over-specific → tuned linear stayed competitive → 1.6×).

- **reuse(conjunction) is minimal.** The (Ann-name × ale-value) cell is unique to this instance; it recurs in no
  sibling binding within the prompt and (with fresh vocab per instance) in no sibling prompt. This is exactly
  the property recall LACKED (its subject×*relation* conjunction recurred across every subject → 49–59% bleed).

- **The read is at the answer step.** "Who has the pie? →" forces the bound-value retrieval AT the final/decode
  token (attend-back-and-copy at generation) — the induction/copy-suppression kernel where FRA banks its 15×–516×
  wins. This is the property injection LACKED (its load-bearing read was at prefill → R_gen≈0).

**The one real risk (the entire reason for the pre-check): the paper's headline is that retrieval is a MIX of a
LEXICAL edge (entity-content → its bound value) and a POSITIONAL edge (group-index/slot).** If the positional
pathway is a redundant backstop, then cutting the lexical (content×content) cell is non-load-bearing — the model
recovers Ann's value by slot — an IOI-style backup-head failure (v1's clean negative `j13`), AND/OR the cut
follows the *slot* not the *content* → P3 sibling-bleed onto the same-position sibling → A collapses to the
box-retrieval ~1.9× floor. **Both are caught by the same cheap pre-check below.** This is why A1 is locked but
gated, not locked and launched.

---

## 3. THE CONCRETE EXPERIMENT FOR A1 — operating-regime pre-check FIRST, then selectivity

### 3.1 MODEL + ARTIFACT + DATA (judge-free, ground-truth)

- **Model:** `gemma-2-2b-it` + `gemma-scope-2b-pt-res-canonical` (residual SAE — the FRA QK score-cell edit
  REQUIRES a d_model decoder; the attention/`hook_z` SAEs are OV-side and cannot drive a score-cell edit). This
  is the exact substrate `injection_precheck.py` / `injection_selectivity.py` already load. RunPod L4/L40, pod
  `rs-*`.
- **Artifact: SELF-BUILT minimal binding eval** (decision in §0). Generator (~40 lines, in the pod script):

  ```
  template:  "{E0} has the {V0}, {E1} has the {V1}, {E2} has the {V2}, {E3} has the {V3}. Who has the {Vq}?"
  answer:    the entity bound to Vq   (exact-match single-token: "Ann"/"Joe"/"Pete"/"Tim")
  symmetric: "What does {Eq} have?"  -> the value (single-token: "ale"/"pie"/...)
  ```
  - N=200 instances. Per instance, draw 4 entities from a name pool and 4 values from a single-token object pool
    (ale, pie, jam, tea, ham, fig, cod, oat, …) so EVERY value tokenizes to ONE token (judge-free next-token
    exact-match) and bindings are arbitrary (P2=0 by construction — there is no fact "Ann has ale" in the weights).
  - **Filter to baseline-correct** (the operating-regime rule, exactly like injection's ASR=1 filter): keep only
    instances the model answers correctly with no intervention. All proxies run on this set ONLY.
- **GROUND-TRUTH metric (NO judge):** P(correct bound-entity/value first token) at the answer position —
  `torch.softmax(logits[-1])[answer_tid]` and the greedy exact-match `emits(gen, answer)`. Identical machinery to
  injection's `canary_first_tid` / `emits`. No LLM judge anywhere.

### 3.2 THE OPERATING-REGIME PRE-CHECK (run FIRST — cheapest, decisive). One pod.

This is the deliverable the evaluator runs first. It adapts `injection_precheck.py` STAGES 2–4 verbatim, with
the **answer-token query** replacing the response position and the **bound-value span key** replacing the
injected-span key. Four sub-steps, in this order (cheapest-decisive first):

**(a) LOCATE the retrieve-bound-value heads** (= injection STAGE 2 `causal_inj_heads`, re-pointed). On the
answer-correct set, for the TARGET binding (entity E0 × value V0): rank all (L,H) by the drop in
P(correct-answer-first-token | answer position) when the edge (answer-query → V0-value-token key) is zeroed at
`hook_attn_scores`. Take top≤3 heads. *This is the head-find; cheap, judge-free, one anchor prompt then confirmed
on ~12.*

**(b) P1 — oracle answer-step edge-cut R_gen on the TARGET binding** (= injection GATE(i), the LBNR ceiling).
On the answer-correct set, zero the (answer-query × V0-key) attention at the top heads and re-generate. R_gen =
fractional drop in correct-target-answer emission, measured AT THE ANSWER STEP (the query IS the about-to-emit
token — A1's structural advantage over injection, where this was R_gen≈0). **Also report R_prefill** (cut the
same edge over prefill rows) split out, per `PREDICTOR2` P1, so a prefill-only artifact cannot masquerade as a
win. Backup-cue check: longer-horizon re-gen must stay suppressed (not restored by a redundant read).
- **GATE: R_gen ≥ 0.6 (GO), ≥ 0.8 (STRONG).** R_gen < 0.3 while R_prefill = 1 → upstream/redundant (the
  injection signature) → NO-GO.

**(c) P3 — sibling-bleed (THE single best discriminator, `PREDICTOR2` §6).** Apply the SAME (E0 × V0) cell-cut,
then re-query the SAME prompt for the OTHER bindings ("What does Joe have?" / "Who has the jam?" → V1/V2/V3).
Bleed = fractional drop in the SIBLING bindings' retrieval / drop in the target. This is the hardest possible
control: identical surface form, identical broad endpoints, differing ONLY in the conjunction.
- **GATE: sibling-bleed < 15% (GO), < 5% (STRONG).** bleed > 40% → relation/slot-keyed recurrence (the recall
  failure mode) → NO-GO.

**(d) THE LEXICAL-VS-POSITIONAL ORDER-SHUFFLE AUDIT (the falsifier — `PREDICTOR2` §3.1, §4-Step-4, the named
KEY falsifier).** This is what makes the P3 number un-gameable and is the coordinate that sank the box-retrieval
precedent. Build the **disagreement regime**: for each anchored instance, locate the (E0 × V0) cell on the
ORIGINAL order, then **shuffle the binding order** so E0's value moves to a different slot/position while E0 still
binds V0 by CONTENT. Two cuts, two predictions:
  - cut keyed on the **content** cell (E0-name × V0-value tokens, wherever they now sit) → if it still suppresses
    E0's retrieval after the shuffle, the edge is **CONTENT-addressed → WIN-consistent**.
  - cut keyed on the **position/slot** (the first-binding slot, regardless of which entity occupies it) → if THAT
    is what suppresses E0's retrieval, the edge is **POSITION-addressed → the box-retrieval ~1.9× floor → NO-GO.**
  Operationally: measure whether the located cell's suppression FOLLOWS the entity content or the slot index when
  order is permuted between the locate step and the eval step.
- **GATE: the edge must be CONTENT-addressed** — cutting the content cell suppresses the (shuffled) target while
  cutting the position cell does NOT (or suppresses the wrong, same-position sibling). If suppression follows
  POSITION → non-load-bearing-lexical → **NO-GO** (do not spend on selectivity).

**PRE-CHECK VERDICT — GO iff ALL THREE:** `R_gen ≥ 0.6` AND `sibling-bleed < 15%` AND `edge is content-addressed`
(order-shuffle follows content, not position). A fail on any one is a one-pod pre-registered negative — file it,
fall to the fallback order (§4). (Mirrors injection's three-gate GO; the order-shuffle audit is the v2 addition
that the box-retrieval precedent demands.)

### 3.3 THE SELECTIVITY WIN-TEST (ONLY if pre-check = GO). One pod. = `injection_selectivity.py` re-pointed.

**Step 1 — pin the matched on-target operating point.** Choose target-suppression: drive the TARGET binding's
correct-retrieval from ~1.0 → <0.2. Tune EVERY method to hit the SAME target-suppression before comparing
collateral (the un-matched-removal artifact is the binding A=5133× trap, `PREDICTOR2` §4 Step 1):
  - **FRA:** sweep edit scale c ∈ {0.5, 1, 2, 4, 8} (the harness's `FRA_CS`). Use the regression-fitted /
    multi-cell support (top-M (E0-feature × V0-feature) pairs over the answer-rows), never a single naive cell-cut
    (a naive cut just MOVES the retrieval onto a sibling — the softmax-redistribution correction).
  - **best-tuned LINEAR baseline:** DoM "retrieve-the-bound-value"/"suppress-E0's-binding" steer.
    `v_L = mean(resid | E0-binding queried) − mean(resid | E0-binding absent/permuted)`, swept over layers
    {6, 9, 12} × α {2, 4, 8, 16, 32} (the harness's `LIN_LAYERS`×`LIN_ALPHA`) — give the linear its absolute best
    shot at matched target-suppression. **Train/eval split:** build the DoM vector on a DISJOINT instance set from
    the eval set (the `feedback_train_eval_split` rule — val-extract leak was worth ~0.1 bits in a prior campaign).

**Step 2 — selectivity vs the best-tuned linear on the HARD control.** Collateral sets, in priority order:
  - **HARD control (the decisive one): intra-instance siblings** — the SAME prompt re-queried for Joe/Pete/Tim
    (E1/E2/E3 × V1/V2/V3). Identical surface, identical broad endpoints, only the conjunction differs. This is
    the strongest selectivity stressor in the whole campaign.
  - **medium control: cross-instance same-role** — a different instance's binding (held-out), where the broad
    endpoints legitimately recur.
  - **capability floor: general next-token coherence** — a held-out non-binding battery (the global-damage check;
    reuse injection's `cap_ok` degeneracy proxy), FRA must not break fluency.
  All metrics ground-truth exact-match / forced first-token logit. No judge.

**Step 3 — the win bar (pre-registered, unchanged so v1-comparable).** Read collateral at matched target-
suppression. Headline = `collateral(linear)/collateral(FRA)` on the intra-instance sibling control.
  - **WIN iff:** A ≥ 2× selectivity gap on the sibling control **AND** ≥ +0.15 absolute sibling-retention
    advantage **AND** FRA retains > 10% of sibling-binding retrieval at matched target-suppression, with no worse
    global damage. **STRONG WIN iff A ≥ 10×** (the magnitude-law band).
  - **NO WIN iff** A < 2× (linear already as surgical — the injection 1.6× diagnosis), OR FRA cannot reach
    matched target-suppression at faithful c ≈ 1–2 (reach ceiling).

**Step 4 — mandatory confound guards (judge-free, carried from `PREDICTOR2` §4 Step 4):** (i) timing audit —
confirm the win rides on R_gen, not R_prefill; (ii) the order-shuffle audit re-run AT the matched operating point
(the win must be content-addressed, not a slot artifact); (iii) redundancy/LBNR — cut not restored by a backup
cue; (iv) matched-removal + coherence — collateral on coherent rows only; (v) C1 base-reversion OLS — a
significant behavior-specific partial (rule out reversion-to-prior).

### 3.4 RISKS + cheapest go/no-go per risk

| risk | which condition | cheapest go/no-go (all in the §3.2 pre-check, no extra pod) |
|---|---|---|
| **positional backstop** (the #1 risk) | C1/C3 redundancy | order-shuffle audit (d): suppression must follow CONTENT not slot. POSITION → NO-GO. |
| answer-step read is actually upstream | C3 timing | P1 split (b): R_gen ≥ 0.6 AND R_gen ≫ a prefill-only artifact. R_gen<0.3 & R_prefill=1 → NO-GO. |
| value not single-token / parametric backstop | C2 + metric | enforce single-token value pool at data-build; closed-book check (query value with context removed) < 10%. >70% → swap vocab. |
| sibling-bleed from shared slot | C4/P3 | sibling-bleed probe (c): <15%. >40% → NO-GO. |
| linear is already surgical (the injection null) | P4 | only knowable at selectivity Step 3; but P4 is predicted ≥5 *because* the linear's only handle is the maximally-reused "retrieve-value" direction — the structural reason A1 ≠ injection. |
| FRA can't reach matched suppression | reach ceiling | selectivity Step 1: if FRA can't hit <0.2 target at c≤2, NO WIN (reach), file honestly. |
| multi-token entity names | metric | restrict name pool to single-token names (Ann/Joe/Tim/…); confirm at data-build. |

### 3.5 FALLBACK ORDER if A1's pre-check FAILS

1. **A1 NO-GO → B1 (ICLAttack in-context backdoor).** Safest fresh win; directly generalizes the banked ~25×
   in-context-backdoor win to a published organism. Pre-check = ablate ONLY the distinctive trigger-span key
   (not a generic position), confirm ASR collapses while clean / different-distinctive-phrase (T′) inputs are
   untouched; shuffle demo order to rule out positional (last-demo-label) encoding. Same harness, same judge-free
   exact-match ASR metric.
2. **B1 NO-GO → WinoDict nonce-binding (ORGANISMS2 C1).** Highest no-backstop purity but the upstream-timing risk
   that sank factual recall. Pre-check = the timing-disambiguation cut: (nonce→gloss) edge-cut at the
   answer/pronoun step alone must flip the coreference (R_gen≈1); if only the prefill cut moves it → pre-baked
   upstream → STOP.
3. **All three NO-GO →** the campaign's deliverable is the sharpened negative (`PREDICTOR2` §5.1): "even the ideal
   real analogue of the synthetic — instance-unique keys, broad endpoints, answer-step read — fails to transfer
   the broad×broad advantage; FRA's home turf is the banked toy kernel." File as a one-pod-each boundary-map
   extension. A predicted-LOSE class winning ≥2× would be the informative surprise (`PREDICTOR2` §5.3).

---

## 4. RECOMMENDATION — the ONE experiment the evaluator runs first

**Run the A1 operating-regime pre-check (§3.2) on gemma-2-2b-it with the self-built 4-binding eval, on the
answer-correct (baseline-filtered) set, in the lexical/positional DISAGREEMENT regime.** Single pod, judge-free,
adapts `injection_precheck.py` STAGES 2–4. Order: (a) locate retrieve-value heads → (b) P1 answer-step R_gen
[+R_prefill split] → (c) P3 sibling-bleed → (d) order-shuffle content-vs-position audit.

**GO/NO-GO numeric gates (all three must hold to proceed to selectivity):**

| gate | proxy | GO threshold | STRONG | hard-FAIL → NO-GO |
|---|---|---|---|---|
| **P1** | oracle answer-step `R_gen` (TARGET binding, answer-correct set) | **R_gen ≥ 0.6** | ≥ 0.8 | R_gen < 0.3 while R_prefill = 1 (upstream/redundant) |
| **P3** | sibling-bleed (same-prompt other bindings, same cut) | **< 15%** | < 5% | > 40% (slot/role-keyed recurrence) |
| **shuffle** | order-shuffle role-vs-content audit | **edge is CONTENT-addressed** (follows content under permutation) | — | edge is POSITION-addressed (follows slot → box ~1.9× floor) |

**Proceed to the §3.3 selectivity win-test iff `R_gen ≥ 0.6` AND `sibling-bleed < 15%` AND `edge is
content-addressed`. Otherwise file the one-pod negative and fall to B1 → WinoDict (§3.5).**

This single pre-check is the campaign's spend router: it samples A1's own operating regime, it tests the
single best win/lose discriminator (P3 sibling-bleed), and the order-shuffle audit closes the one falsifier
(positional backstop) that distinguishes a genuine instance-unique-key win from the v1 box-retrieval ~1.9× null.
