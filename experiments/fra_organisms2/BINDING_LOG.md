# BINDING_LOG.md — A1 in-context variable-binding operating-regime PRE-CHECK (EVALUATOR)

*FRA organism-hunt v2 (`CAMPAIGN.md`), the LOCKED #1 candidate (`SCREEN2.md` §0): A1 Mixing-Mechanisms
bound-entity retrieval on `gemma-2-2b-it`, self-built minimal binding eval, judge-free. This log records
the pre-check DESIGN, the FOUR gate numbers, and the GO/NO-GO verdict. The pre-check is the campaign's
spend-router: it routes whether the §3.3 selectivity win-test spend happens. NOT the win-test itself.*

---

## 0. WHAT THIS PRE-CHECK IS (and is not)

Per `SCREEN2.md` §3.2 / `PREDICTOR2.md` §4 Step 0: ONE pod, judge-free, the cheapest-decisive gate before
any selectivity run. It re-points the v1 generic FRA cell-cut harness (`../fra_organisms/jobs/
injection_precheck.py` + `injection_prefill.py`) — residual `gemma-scope-2b-pt-res-canonical` -> query/key
feature decode via `W_dec` -> per-head pre-softmax score-delta subtracted at `blocks.{L}.attn.hook_attn_scores`
— with the **answer-token query** replacing injection's response position and the **target binding's
entity/value span key** replacing injection's injected-span key.

It does NOT run the selectivity win-test (FRA vs best-tuned linear). That is the NEXT decision, gated on a GO.

---

## 1. THE EVAL (self-built, judge-free, ground-truth next-token exact-match)

- **Template:** `"{E0} has the {V0}, {E1} has the {V1}, {E2} has the {V2}, {E3} has the {V3}. Who has the {Vq}?"`
  Answer = the entity bound to `Vq`, a SINGLE token (`" Ann"`/`" Joe"`/...). Queried BY VALUE -> answer is the ENTITY.
- **Vocab:** name pool + single-token object pool, **filtered at runtime to confirmed single-token-with-leading-space**
  items on the gemma tokenizer (so next-token exact-match is unambiguous, judge-free).
- **N=200** candidate instances, 4 bindings each, distinct entities + distinct values per instance (arbitrary
  pairings -> **P2 = 0 by construction**: no fact "Ann has ale" in the weights).
- **OPERATING REGIME = baseline-correct filter** (the recall false-GO lesson — sample the app's own regime):
  keep only instances the model answers correctly (argmax next-token == answer) with NO intervention. All four
  proxies run on this set ONLY (capped at N_TARGET=40 for cost).
- **GROUND-TRUTH metric (NO judge):** `P(correct bound-entity first token)` at the answer position via
  `softmax(logits[answer_pos])[answer_tid]`, plus the greedy argmax exact-match for the baseline filter.

## 2. THE FOUR GATES (judge-free; the cut machinery is the v1 harness verbatim)

- **(a) LOCATE** the answer-step retrieve-bound-entity heads (= injection STAGE-2 `causal_inj_heads`, re-pointed):
  rank all (L,H) by drop in `P(correct-answer | answer pos)` when the edge (answer-query -> TARGET-entity key)
  is zeroed at `hook_attn_scores`; top<=3 heads, averaged over 4 anchors.
- **(b) P1** — oracle answer-step edge-cut R, **SPLIT** `R_gen` (cut ONLY the answer/decode query row) vs
  `R_prefill` (cut ONLY prefill query rows). The answer-step read must be load-bearing.
  **GATE: R_gen >= 0.6 (GO) / >= 0.8 (STRONG).** Hard-fail: `R_gen < 0.3 while R_prefill >= 0.9` -> upstream/
  redundant (the injection signature) -> NO-GO.
- **(c) P3** — **SIBLING-BLEED**: apply the SAME (target-entity x target-value) conjunction cell-cut, then
  re-query the SAME prompt for the OTHER bindings ("Who has the {V_j}?"). `bleed = mean(sibling fractional
  P(correct)-drop) / mean(target fractional drop)`. **GATE: bleed < 0.15 (GO) / < 0.05 (STRONG).** Hard-fail:
  `bleed > 0.40` -> slot/role-keyed recurrence (the recall failure mode) -> NO-GO.
- **(d) ORDER-SHUFFLE CONTENT-VS-POSITION AUDIT** (the decisive falsifier, the box-retrieval ~1.9x precedent):
  shuffle the binding order so the target binding (E[qi] x V[qi]) moves to a DIFFERENT slot while the entity
  still binds its value BY CONTENT. On the shuffled prompt (still query V[qi] -> answer E[qi]): cut the
  **CONTENT** cell (target entity+value tokens wherever they now sit) -> `drop_content`; cut the **POSITION**
  cell (the entity+value now occupying the target's ORIGINAL slot, a DIFFERENT binding) -> `drop_position`.
  **GATE: edge is CONTENT-addressed** iff `drop_content > drop_position` (mean AND per-instance majority).
  Position-addressed -> box ~1.9x floor -> NO-GO.

**PRE-CHECK VERDICT — GO iff `R_gen >= 0.6` AND `sibling-bleed < 0.15` AND `content-addressed`.** Any fail =
a clean one-pod pre-registered negative -> fall to B1 (ICLAttack) -> WinoDict (`SCREEN2.md` §3.5).

---

## 3. EXECUTION

- **Code:** `experiments/fra_organisms2/jobs/binding_precheck.py` (parse-gated; early flush + ckpt BEFORE any
  heavy loop; ckpt INSIDE every per-head / per-instance loop; top-of-script try/except uploads `FATAL.json`).
- **Launch:** `experiments/fra_organisms2/jobs/launch_binding_precheck_pod.sh`. RunPod L4/L40, pod
  `rs-binding-precheck-1`, reuses `fra_win/fra_bundle.tar.gz`. HF prefix `fra_org2_binding/{code,results}`.
  Partial-upload uploader every ~4 min.
- **Pods:**
  - run-1 `rs-binding-precheck-1` id=`ac6fjdescqb5fc` (L4), RUNID=`20260612-175029` — **BLOCKER** (see §4a). Terminated.
  - run-2 `rs-binding-precheck-1` id=`8aqp5q9l62v0tk` (L4), RUNID=`20260612-175607` — **clean GO** (see §4b).
  Results -> `dmanningcoe/fra-phase1-steering-data : fra_org2_binding/results/<RUNID>/`.

---

## 4a. RUN-1 BLOCKER + FIX (the answer-position artifact)

Run-1 returned **0/200 baseline-correct** with `p_correct ~= 1e-5..3e-4` everywhere — NOT "binding fails on
gemma." Diagnosis: I wrapped the prompt in the **chat template with `add_generation_prompt=True`**, so the
model's first generated token (read at the prompt-final position) is a PREAMBLE token ("The"/newline/etc.),
not the bare entity -> the entity gets ~0 probability at the answer position. (This is the exact artifact the
v1 injection harness solved with `prime_to_canary`.)

**Cheapest fix (no restart-loop):** drop the chat template; use a **completion prompt ending in `" Answer:"`**
so the IMMEDIATE next token is the bound entity ` Ann`/... (the Mixing-Mechanisms / SCREEN2 `->` intent).
One 2-line edit to `render()` + `build_tokens()` (now `add_special_tokens=True` for `<bos>`). Re-launched run-2.

## 4b. RUN-2 RESULTS (clean) — operating regime, the four gates, GO/NO-GO

Operating regime: **160/200 baseline-correct (acc = 0.80)**; proxies run on the first **40**.
Located retrieve-bound-entity heads (mean drop in P(correct) under the answer-query -> target-entity edge-cut):
**L22H4 (0.562), L18H6 (0.156), L22H3 (0.032)** — localized to L22H4 + L18H6 (note L18H6 also appeared as an
injection head in v1; here it is a binding-retrieval head). Top-3 cut for the gates.

| gate | proxy | value | GO threshold | STRONG | pass? |
|---|---|---|---|---|---|
| operating regime | baseline-correct N | **160/200 (acc 0.80)**; 40 run | >= 8 to run | — | yes |
| heads | top retrieve-bound-entity (L,H) | **L22H4, L18H6, L22H3** | — | — | located |
| **P1** | `R_gen` (vs `R_prefill`) | **R_gen = 0.900** (R_prefill = 0.075; R_all = 0.944) | R_gen >= 0.6 | >= 0.8 | **STRONG** |
| **P3** | sibling-bleed | **0.0009** (target_drop 0.904, sib_drop 0.0008) | < 0.15 | < 0.05 | **STRONG** |
| **shuffle** | content- vs position-addressed | **content-addressed** (drop_content 0.952 vs drop_position 0.0008; **29/29 = 100%** content) | content-addressed | — | **STRONG** |

**Interpretation (vs the v1 calibration controls):**
- **P1**: load-bearing read is AT THE ANSWER STEP (R_gen 0.90 ≫ R_prefill 0.07) — the EXACT INVERSE of the
  injection signature (R_gen≈0, R_prefill=1). Win-condition 3 (consumed at the answer step) is met, the
  property injection LACKED.
- **P3**: sibling-bleed ~0.1% — the synthetic's instance-unique-key property realized in the wild; the
  property recall LACKED (49-59% bleed). No slot/role recurrence.
- **shuffle**: suppression follows entity CONTENT, not slot, in 100% of shuffled instances — closes the
  single falsifier (positional backstop) that capped the box-retrieval precedent at ~1.9×.

**>>> VERDICT: GO (STRONG) — all three gates pass in their STRONG band. <<<**

## 5. THE §3.3 SELECTIVITY WIN-TEST — **WIN (11.1×, STRONG band)**

*Pod `rs-binding-selectivity-1` id=`r57xj5e980e32p` (L4), RUNID=`20260612-180736`. Code
`experiments/fra_organisms2/jobs/binding_selectivity.py` (parse-gated; ckpt in every sweep loop;
top-of-script try/except). Judge-free ground-truth exact-match. Terminated after WIN recorded.*

**Sets:** 105 baseline-correct instances, split **eval n=60 / train n=40 (DISJOINT)** (the DoM vector is built
only on TRAIN — the train/eval-leak rule). HARD control = the **intra-instance siblings** (the OTHER 3 bindings
in the SAME prompt, re-queried), under the SAME target-cell edit.

**Three interventions, swept, read at matched target-suppression t* = 0.80** (a level FRA reaches; FRA oracle
max supp = 0.848):

| method | how target-suppression is driven | sibling-collateral @ t*=0.80 | reaches t*? |
|---|---|---|---|
| **FRA cell-cut** (L22H4+L18H6, target entity×value cell) | hard cut of the located cell (oracle); = the surgical edit | **0.064** (oracle point itself: **supp 0.848, coll 0.0021**) | yes (0.848) |
| **best-tuned LINEAR DoM** ("retrieve-this-bound-value", best = **L9**) | project-subtract α·v̂, swept layers{6,9,12}×α{2..64} | **0.715** | yes (L9 α≥2) |
| **head-ablation L22H4** (circuit-tracing ref) | zero whole head hook_z | 0.527 (and only reaches supp 0.417 — can't hit t*) | no (max 0.417) |
| **head-ablation L22H4+L18H6** | zero both heads' hook_z | 0.439 (max supp 0.300) | no (max 0.300) |

**HEADLINE selectivity RATIO (linear / FRA) @ t* = 11.1×.** FRA beats head-ablation by **8.2× / 6.8×**.
Absolute sibling-retention advantage = **+0.651** (FRA keeps siblings; the linear destroys them).

**>>> VERDICT: WIN (STRONG, ratio ≥ 10×) — ratio 11.1× ≥ 2×, abs adv 0.65 ≥ 0.15, FRA < linear, FRA < both
head-ablations. <<<**

### 5a. Curve honesty + confound guards (the EM/injection-burned rigor)

- **The matched point is FRA-reachable** (the v1 reachability lesson): t*=0.80 < FRA oracle supp 0.848. The
  headline RATIO is, if anything, *understated* by the interpolation — at the FRA oracle's OWN operating point
  (supp 0.848) FRA pays only **0.0021** collateral while every method that reaches that suppression pays
  ~0.4–1.0, so the true ratio there is ~340×. We report the conservative interpolated 11.1× as the headline.
- **The scaled FRA-feature `c·δ` curve is a red herring** (c=0.5..32 only reach supp ~0.10 with rising
  collateral): subtracting a scaled multi-cell delta over-drives the softmax globally — the known "scaled cut
  disrupts" artifact. The honest, surgical FRA edit is the **hard cut of the located cell** (the oracle), which
  the pre-check already validated (supp 0.90, bleed 0.0009). That is the FRA operating point used.
- **The best-tuned linear is genuinely competitive at LOW suppression but cannot be selective at HIGH**: its
  only handle is the broadly-reused "retrieve-the-bound-value" direction, so driving the target down necessarily
  pushes every sibling down too — at any α reaching supp ≥ 0.6 the model degenerates (degen-frac → 1.0,
  sib_coll → ~1.0). This is the magnitude law A ≈ reuse(endpoint)/reuse(conjunction) realized: broad endpoint,
  rare conjunction → the linear has no surgical option. **The exact reason injection (over-specific "comply"
  handle) only reached 1.6× and this organism reaches 11×.**
- **Capability/degeneracy guard:** FRA oracle degen-frac = 0.083 (vs linear 1.0 at matched suppression); FRA
  keeps the model emitting well-formed names. FRA does not just break fluency to win.
- **Head-ablation (the circuit-tracing baseline) is both weaker and messier**: it can't even reach t*=0.80
  (whole-head ablation maxes at supp 0.42) and bleeds 0.44–0.53 where it acts — the FRA cell-cut strictly
  dominates the head-level edit (the A3 cell-cut-vs-head-ablation comparison: the bilinear cell is more surgical
  than the head).

### 5b. The campaign result

This is the fresh **≥2× selectivity win injection couldn't reach** — in fact a STRONG-band **11.1×** (≥340× at
the FRA oracle point). The synthetic's broad×broad advantage (instance-unique key × broadly-reused endpoints ×
answer-step read) **transfers to a real, reasoning-relevant organism (in-context variable binding) on
gemma-2-2b-it**: an FRA (entity × value) QK cell-cut suppresses one binding's retrieval ~0.85 with ~0.2%
sibling collateral, while the best-tuned linear "retrieve-the-bound-value" steer pays ~0.72 (and head-ablation
~0.44–0.53). The four pre-registered proxies (P1 answer-step, P2=0 by construction, P3 instance-unique, P4 broad
endpoints) correctly predicted the win, and the order-shuffle audit confirms it is content- not slot-addressed.

