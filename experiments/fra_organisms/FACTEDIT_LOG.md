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

- **Pod:** `rs-factedit-precheck-1` id `9t0gktup2dntv4` on NVIDIA L4.
- **RUNID / results prefix:** `20260612-061339` → `fra_org_factedit/results/20260612-061339`.
- **Status:** LAUNCHED (bootstrapping). Polling for results below.

---

## 3. RESULTS

*(pending — filled when the pod reports)*
