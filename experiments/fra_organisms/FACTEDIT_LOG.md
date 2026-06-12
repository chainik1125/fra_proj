# FACTEDIT_LOG.md — factual-recall edit GO/NO-GO pre-check (SCREEN.md proposal 1)

*EVALUATOR agent, fra_organisms campaign. Builds + runs ONLY the cheap GO/NO-GO pre-check from
SCREEN.md §3/§4 (the single decision-relevant first move). Does NOT run the full §4 ROME/MEMIT
selectivity campaign — this pre-check routes whether that spend happens. Ground-truth (logit-based)
metric only; NO LLM judge.*

---

## 0. The pre-check spec (from SCREEN.md §4 "THE CONCRETE FIRST EXPERIMENT + ITS GO/NO-GO PRE-CHECK")

> On gemma-2-2b + GemmaScope, take 20–30 CounterFact facts the model recalls (P(object)>0.3).
> Apply the FRA (subject × relation) SAE-feature-pair cell-edit at the causally-top ≤3 extraction
> heads. **GO** iff: (i) the cell-edit drops P(target object) by **≥50%** at faithful scale c≈1–2
> (resolves the CCF=0.000 reach worry), AND (ii) a held-out **same-relation, different-subject** fact
> drops by **<15%** under the SAME edit (rules out the MLP-distributed / D2-fail mode — the
> greater-than signature). GO → commit the full §4 selectivity test vs ROME/MEMIT + linear
> subject-steer. NO-GO → STOP, report the §6.1 pre-registered flagship negative.

---

## 1. Design + implementation

**Files:**
- `jobs/factedit_precheck.py` — the pod script (parse-gated, ckpt() after every fact, resume-proof).
- `jobs/launch_factedit_pod.sh` — single-shot launcher (pod `rs-factedit-precheck-1`, HF prefix
  `fra_org_factedit/{code,results}`, RUNPOD_API_KEY=$RP_API_KEY_MATS).

**Model + SAE.**
- Model: `gemma-2-2b-it` (per spec). Recall-filtering to facts the model knows makes the result
  model-choice-robust (a recalled fact is recalled regardless of -it vs base).
- SAE: `gemma-scope-2b-pt-res-canonical` (residual, width-16k, canonical), one per needed layer
  (SAE on resid_post[L-1] = resid_pre[L]).
  **IMPORTANT SAE-BASIS NOTE:** the spec names GemmaScope *attention* SAEs (`gemma-scope-2b-pt-att`).
  But the FRA QK score-cell decomposition (`fra.core.fra._build_fra_result`) **requires a residual SAE**
  whose decoder lives in d_model space, because the cell-edit projects each feature's decoder vector
  through W_Q / W_K to reconstruct the bilinear attention-score contribution. A hook_z / attention SAE
  decodes into per-head value space and **cannot** drive a QK score-cell edit. This is the same residual
  SAE the prior fact-recall gate probe (`fra_win/.../g3_gemma.py`, the CCF=0.000 / R=+0.86 run) and the
  proven gemma QK win (`j12_gemma_win.py`) both used — so the pre-check is on the SAME basis whose
  reach the CCF=0.000 worry was about. Resolving the reach worry on this exact basis is the point.

**Reused tooling** (templates named in the spec):
- `fra_win/jobs/g_screen.py` — `causal_heads()` (rank heads by edge-cut effect on P(ans), not raw
  attention) + the score-cell `cut()` (set (q,k) score to −1e4 at hook_attn_scores).
- `fra_win/jobs/j13_ioi.py` + `j12_gemma_win.py` — FRA-decompose an edge → top-M SAE feature-pairs →
  build a per-head `[seq,seq]` score-delta → subtract `c·delta` at `hook_attn_scores` (the cell-edit).
- `fra.core.fra._build_fra_result` (RMSNorm-correct, RoPE-correct FRA) + `fra.sae_lens_wrapper.GemmaScopeSAE`.

**Data.** `NeelNanda/counterfact-tracing` (CounterFact, public, no gating). Schema gives `prompt`,
`subject`, `target_true`, `relation_id` — exactly the (subject, relation, target object) triple.
Validated locally: target-first-token decodes 80/80, subject-key locatable 80/80, and a 400-fact
sample has 33/33 relations with ≥2 facts (so every edited fact has a same-relation/different-subject
held-out partner). 34 relations, 21919 facts total.

**Pipeline.**
1. RECALL FILTER: score ~400 candidates by P(target_true first token | last position); keep those
   with P ≥ 0.30 (recalled). Group by relation; build the edit set = recalled facts that have a
   recalled same-relation/different-subject PARTNER (the held-out collateral fact). Cap at N_TARGET=28.
2. CAUSAL HEAD-FIND: on the top-recall anchor fact, rank all 26×8 heads by edge-cut effect on
   P(target) of the (last-position query × subject-token key) extraction edge; take top ≤3.
3. CELL-EDIT (per fact): FRA-decompose the (last × subject) edge at each head, take top-M=16 support
   SAE feature-pairs (subject-content query feat × relation-content key feat), build the score-delta,
   subtract c·delta at hook_attn_scores. Measure P(target) drop at c∈{1,2}. Also record the raw
   edge-cut oracle (LBNR ceiling) for reference.
4. GATES.
   - **(i)** median P(target) drop ≥ 50% at c≈1–2 on the EDITED facts.
   - **(ii)** held-out collateral < 15%: take fact-A's support feature-PAIRS, rebuild their score-delta
     on the held-out fact-B's own FRA edge geometry (same heads, same pairs), apply at strength c, and
     measure B's P(target) drop. If A's subject×relation features don't fire on B, B is untouched
     (~0 collateral → content-specific, FRA-favorable). A large drop ⇒ the edit is non-content-specific
     ⇒ MLP-distributed (the D2-fail / greater-than signature). (We also record B's *own* re-derived
     edit drop as an informativeness check — is B independently editable.)
   - **VERDICT: GO iff (i) ≥50% AND (ii) <15%. NO-GO otherwise** (= the pre-registered §6.1 flagship
     negative: structurally-ideal broad×broad recall is G-post / unreachable on a real LLM — a sharp bound).

**Execution.** RunPod only (L4, `rs-factedit-precheck-1`), RUNPOD_API_KEY=$RP_API_KEY_MATS. Parse-gated
before upload. ckpt() writes precheck.json after every fact; background uploader every ~4 min →
partial-upload + resume-proof. Reuses the existing `fra_win/fra_bundle.tar.gz` (same fra/ package).

---

## 2. RUN STATE

- **Run 1 (pre-check):** pod `rs-factedit-precheck-1` (`9t0gktup2dntv4`, L4), RUNID `20260612-061339`
  → `fra_org_factedit/results/20260612-061339`. DONE, pod terminated.
- **Run 2 (diagnostic):** pod `rs-factedit-diag-1` (`unpx2oao5z1h58`, L4), RUNID `20260612-062058`
  → `fra_org_factedit/results/20260612-062058`. `jobs/factedit_diag.py`. DONE, pod terminated.
  (Investigates a confound in run 1 — see §3.2.)
- **Run 3 (COMPLETE TEST — red-team-corrected):** pod `rs-factedit-complete-1` (`f07xl0ppcc70v6`, L4),
  RUNID `20260612-071132` → `fra_org_factedit/results/20260612-071132`. `jobs/factedit_complete.py`.
  Fixes both red-team findings (full-subject-span × relation key, top-8 heads, faithful SAE feature-pair
  cell-edit c∈{1,2,4,8}, bucket-balanced edit set in the load-bearing band P∈[0.30,0.55], reports
  relative-drop AND argmax rank-flip AND collateral). **DONE, pod terminated. The DEFINITIVE result — see §4.**
- **Spend:** 3 × L4 pods (~10–18 min each), all terminated. No idle GPU left running.

**HEADLINE:** flagship factual-editing = **NO-GO (airtight)**. The red-team-corrected complete test (§4)
shows the FRA cell-edit DOES reach the load-bearing full-span edge (median on-target drop 0.52, meets oracle
70%, rank-flip 60%) but FAILS selectivity (median same-relation/diff-subject collateral **0.49**, need <15%;
only 1/20 selective). Not a reach/G-post failure — a **D3 conjunction-recurrence (relation-keyed, not
subject-keyed) selectivity collapse.** Do NOT proceed to the full vs-ROME/MEMIT campaign.

---

## 3. RESULTS

### 3.1 Run 1 — the pre-check (recalled facts, P(target)>0.3)

- **Recalled-fact set:** 164 recalled candidates (P(target0)≥0.30) across 24 relations; edit set = 28
  facts, each with a same-relation/different-subject held-out partner. (Baseline P(target) of the
  edited facts: 0.71–0.99 — strongly recalled.)
- **Located extraction heads (causal, on the anchor "Chromecast is produced by"):**
  L23H5, L16H4, L8H3 — but with TINY edge-cut effects (0.010, 0.0024, 0.0010).
- **GATE (i) cell-edit P(target) drop:** median **0.002 @c1 / 0.006 @c2** (0/28 facts reach 50%). FAIL.
- **GATE (ii) held-out collateral:** median **~0.003** (cross-applied A-support onto B). PASS (trivially).
- **Verdict (run 1, naive):** NO-GO.
- **THE CONFOUND:** the *oracle edge-cut* (cut the (last × subject) score ENTIRELY at the top heads)
  was **also ~0** (median 0.0013; per-fact ≤0.03). That contradicts the prior cheap probe's R=+0.86.
  So gate (i) was **uninformative**: the LBNR premise (the edge is load-bearing) that the probe
  established did NOT hold on this recalled set — i.e. the failure is not (yet) "FRA can't reach", it
  is "there is no load-bearing edge to reach FOR on strongly-recalled facts". The prior probe's R=+0.86
  was on `"capital of France"` at **P(ans)=0.039 — a WEAKLY-recalled fact**. Run 1's P>0.3 filter
  selected the opposite (saturated) regime. → Run 2 disambiguates saturation-vs-machinery-bug.

### 3.2 Run 2 — diagnostic (per-fact best edge-cut, by recall band)

For each fact, run the FULL causal head-find (P(target) drop from each single-head edge-cut) on BOTH
candidate keys (last subject token / relation-suffix token), take that fact's own top-3 heads, report
the best oracle edge-cut R. Three recall bands. Sanity = reproduce the prior probe.

- **Sanity ("The capital of France is"→" Paris"), gemma-2-2b-it:** base P(Paris)=**0.20** (it *does*
  recall it, unlike base-gemma's 0.039), best-localized **R@top3 = 0.32** — machinery confirmed
  load-bearing; top heads L10H3/L20H3/L9H3.
- **Per-fact best edge-cut R, by baseline-recall band (n=12 each), FINAL:**
  - **lo (P 0.03–0.15):** best-R median **0.32**, mean 0.35, max 0.56; frac(≥0.5)=**0.25**. Load-bearing.
  - **mid (P 0.15–0.40):** best-R median **0.39**, mean 0.41, max 0.71; frac(≥0.5)=**0.33**. Load-bearing, often strongly.
  - **hi (P 0.60–0.99):** best-R median **0.073**, mean 0.086, max 0.25; frac(≥0.5)=**0.0**. NOT load-bearing.

**INTERPRETATION (the real result).** The extraction edge's load-bearingness is **inversely related to
how well the model recalls the fact**. On facts the model strongly knows (the ones the pre-check spec
asks us to edit, P>0.3), the subject→last attention edge is *redundant/saturated*: cutting it entirely
barely moves P(target), because the answer is overdetermined (multiple heads/layers + MLP recompute it).
The edge is only load-bearing on *weak* facts (P<0.4), exactly the regime the cheap probe sampled. This
is the **distributional-redundancy / downstream-recompute (D2/D4) failure** the rubric pre-registered as
the flagship's biggest risk (SCREEN §3.1 "the object is recomputed in a downstream MLP"; SCREENING_RUBRIC
§6.1) — but with a sharper twist: it is recall-strength-gated, so the pre-check's own recalled-fact
operating point sits squarely in the non-load-bearing regime.

### 3.3 VERDICT (runs 1–2) — *PREMATURE, SUPERSEDED by §4. See red-team note below.*

> **RED-TEAM (3 skeptics, 2026-06-12):** this NO-GO was PREMATURE. The *operational* part is robust
> (last-subject-token × relation at the top-3 heads genuinely can't move recalled facts — confirmed at
> all 208 heads). BUT the FRA cell-edit was **never aimed at the load-bearing pathway**: (1) it cut the
> LAST subject token's edge, whereas the load-bearing attention is the FULL SUBJECT TOKEN-SPAN (non-last,
> distributed) — the all-heads full-span oracle removes the fact on 7/12 strongly-recalled facts; and
> (2) the 28 edit facts were sorted by −p_base + capped, so ALL had P≥0.71 (saturated) — ZERO in the
> load-bearing [0.30,0.60) band the pre-registration (P>0.3) intended. → run 3 (§4) is the fair complete test.

*(retained for the record — the operational sub-claim stands, the verdict does not):*

**NO-GO** — the pre-registered §6.1 flagship negative, now with a precise mechanism.

- GATE (i): on the spec's operating point (recalled facts, P(target)>0.3), the FRA cell-edit drops
  P(target) by **~0.6%** (median), 0/28 reach 50%. FAIL — and the oracle edge-cut confirms this is not
  a *reach* ceiling (SAE granularity) but a *load-bearing* ceiling: there is no causal extraction edge to
  cut on strongly-recalled facts (median oracle R≈0 in the hi band; R≈0.3–0.4 only on weak facts).
- GATE (ii): collateral is trivially <15% (≈0), but only because the on-target effect is ≈0 — a
  meaningless pass (the "non-load-bearing edge → zero collateral ratio" artifact the rubric warns about).

**The sharp bound:** the structurally-ideal broad×broad relation (subject × relation → object), which the
synthetic FRA results predicted as the flagship WIN, is **G-post / redundant on a real LLM precisely where
it matters** — for facts the model actually recalls, the subject→object transport is not carried by a
single cuttable attention edge. FRA controls *attending to the subject*, not *recall of the object*.

### 3.4 Readiness for the full §4 selectivity test

**Do NOT proceed** to the full ROME/MEMIT selectivity campaign on this organism as specified. The gate
that routes that spend is FAILED. The full selectivity test (FRA vs ROME at matched on-target removal)
is moot because FRA cannot hit any meaningful on-target operating point on recalled facts at faithful
scale — there is no load-bearing edge to match ROME's removal against.

**Cheapest salvage option (optional, NOT auto-run — a decision for PLANNING):** the diagnostic shows the
edge IS load-bearing on weak/mid facts (R≈0.3–0.4). A *re-scoped* flagship — "selectively suppress a
fact the model knows WEAKLY, vs ROME, on weak facts" — is the only version with a live on-target effect.
But it is a much weaker organism (weak facts are not the knowledge-editing use case ROME targets, and the
broad×broad selectivity story is diluted), so this is a consolation experiment, not the flagship. The
honest headline is the NO-GO bound above.

---

## 4. Run 3 — the FAIR COMPLETE TEST (red-team-corrected) — the DEFINITIVE result

`jobs/factedit_complete.py`, pod `rs-factedit-complete-1`, RUNID `20260612-071132`. Both red-team fixes
applied: **full-subject-span × relation key**, **top-8 extraction heads** (ranked by full-span oracle),
**faithful SAE feature-pair cell-edit c∈{1,2,4,8}** (NOT column-zeroing), **bucket-balanced edit set in the
load-bearing band P∈[0.30,0.55]** (no −p_base sort), keeping the same-relation/different-subject partner.
Reports relative-drop AND argmax rank-flip AND held-out collateral. n=20 facts (P_base 0.31–0.54, 27
relations). All ground-truth, no judge. **Pod terminated.**

### 4.1 The two red-team fixes are VALIDATED — the on-target story flips

- **Full-span edge IS load-bearing** (fix 1): the full-subject-span causal head-find found a strong hub
  — anchor "HealthCap…" full-span oracle **R3=0.78**; L15H5 alone effect 0.317 (vs the pre-check's last-token
  heads at ~0.01). Median full-span oracle R over the edit set = **0.21** (many facts 0.6–0.93).
- **The FRA SAE feature-pair cell-edit REACHES that edge** (kills the "G-post/unreachable" reading):
  - median FRA on-target drop per c: c1=0.08, c2=0.18, c4=0.29, **c8=0.52**.
  - **median BEST on-target drop = 0.52** (≥50%); **frac(best≥50%) = 0.50**; **frac(meets-or-beats its own oracle) = 0.70**.
  - **rank-flip rate = 0.60** (majority flip the argmax off the target token).
  - per-fact highs: HealthCap 0.98 (oracle 0.82), Nissan Skyline 0.97 (0.93), Araria 0.96, Google Sites
    0.90 (0.84), native-language-Marc-Dolez 0.84 (0.69). **The cell-edit clearly reaches the distributed
    full-span edge where headroom exists.** → the flagship is **NOT a reach failure**.

### 4.2 …but it FAILS the selectivity gate — the real, sharp result

- **Held-out collateral (same-relation / different-subject) median = 0.49** (gate needs <15%). FAIL.
- Among the 10 facts with on-target ≥0.5, **median held-out collateral = 0.59** — the SAME edit suppresses
  a DIFFERENT subject's same-relation fact almost as much as the target fact.
- **Only 1/20 facts** achieve a selective win (on-target ≥0.5 AND collateral <0.15: IBM Lotus Word Pro;
  "In Portugal" is borderline at 0.15).

### 4.3 VERDICT (definitive): **NO-GO — and now AIRTIGHT, for the RIGHT reason**

GATES: drop≥50%-or-meets-oracle **PASS** · rank-flip-majority **PASS** · collateral<15% **FAIL** → **NO-GO**.

The flagship is **not** a false-negative and **not** a reach/G-post failure. The corrected test shows the
FRA (subject-span × relation) cell-edit *does* reach the load-bearing extraction edge and *does* remove the
fact (≥50% on a majority, meeting the oracle headroom 70% of the time, flipping the argmax 60%). **It fails
on SELECTIVITY**: the edit is relation-keyed, not subject-keyed — it bleeds ~49–59% onto same-relation,
different-subject held-out facts. The (subject × relation) feature-pair support is **not subject-discriminating**;
the discriminating endpoint behaves as a generic *relation-role*, so the edit fires on every sibling of the
relation. This is precisely the pre-registered **D3 conjunction-recurrence failure** (SCREENING_RUBRIC Q3:
"generic role/position discriminator → FRA fires on siblings, A→1.9× floor"; the box-retrieval sibling
precedent), surfaced here as a direct §4 selectivity collapse rather than the reach ceiling the cheap probe
feared.

**The sharp, publishable bound (updated):** on real-LLM factual recall the (subject × relation) attention
conjunction is REACHABLE and LOAD-BEARING but **NOT subject-selective** — the FRA cell-edit removes the fact
with ≥50% efficacy yet pays ~50% collateral on same-relation siblings, so it cannot beat (let alone be a
surgical alternative to) a knowledge editor's selectivity. The structurally-ideal broad×broad relation the
synthetic predicted as the flagship WIN delivers the on-target effect but **fails the selectivity claim that
is the entire point** (rubric §4 / §6.1). FRA controls *"attend to this relation"*, not *"recall THIS
subject's object"*.

### 4.4 Readiness for the full §4 vs-ROME/MEMIT campaign — **DO NOT PROCEED**

The selectivity gate that routes that spend FAILED — and it failed on the *exact* axis the full vs-ROME test
measures (held-out same-relation collateral). Running the full ROME/MEMIT comparison would only re-measure,
at much higher cost, a selectivity loss already demonstrated cheaply: FRA's median same-relation collateral
is ~0.49 at matched-ish on-target removal, whereas the whole flagship thesis required FRA collateral <5%
(vs ROME's predicted ≥15%). **The pre-check has done its job: it is a NO-GO, and the full campaign is not
warranted.** The deliverable is the sharp negative bound in §4.3 (a clean, pre-registered §6.1 falsifier:
the broad×broad selectivity advantage does NOT transfer to real factual recall — the conjunction is
relation-recurrent, not subject-unique).

*Optional, NOT auto-run (PLANNING's call):* if any vs-ROME number is still wanted for the writeup, the
single cheapest informative add is to confirm ROME/MEMIT IS subject-selective on this same 20-fact set
(expected: yes), making the contrast "FRA reaches the edge but is relation-recurrent; ROME is
subject-selective" — but that is a one-sided confirmation of the loss, not a live win path.
