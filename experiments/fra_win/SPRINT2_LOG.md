# FRA 10h SPRINT 2 (theory-driven) — started 2026-06-10 ~22:40, stop ~08:40 (+10h)

Pipeline (loops): THEORY(Fable) -> BRAINSTORM(2 Fable + 6 opus) -> RANK(opus) -> EVAL(pod) -> RED-TEAM(opus) -> append findings -> next cycle.
NEW vs sprint-1: a Theory agent (Fable) produces a mechanistic FRA theory + bounds + synthetic settings + real-world contexts that SEED the brainstorm. Eval uses the CORRECTED Tier-2 metric.
Budget: HIGHER cap (PI approved going deeper); still tear down idle pods. Agents on opus 4.8 if rate-limited.

## CORRECTED EVAL METRIC (the sprint-1 red-team's key fix — use this, not A-vs-strawmen):
- Screen: CCF (gpt2 reliable; gemma CCF buggy -> rely on LBNR) + LBNR-R (CAUSAL head-finding: cut head edge -> P(answer) drop; raw-attention picks positional heads).
- Tier-2 WIN = legit-content KL: FRA vs a CONTENT-GATED PROJECTION-REMOVAL linear steer at MATCHED on-target removal. FRA wins if it preserves legit content (KL~0) where the steer destroys it (KL>>0). Template: jobs/retrieval_steer3.py (gemma), jobs/acronym_steer3.py (gpt2). PROJECTION-REMOVAL (resid -= a*(resid.dhat)dhat), NOT SAE-feature-subtraction (nan's on gemma / doesn't fire). Also: random-OFF-EDGE null (pair-specificity), transfer vs position-patch.
- GUARD: A/KL only valid at matched on-target removal (confirm the steer actually fires + FRA actually suppresses).

## INFRA gotchas (sprint-1): poll-execute harness SKIPS a re-used job id -> always NEW id. ALWAYS parse-gate jobs (python3 -c ast.parse) BEFORE submit. Living report = CAMPAIGN_REPORT.md (theory agent reads it -> feedback loop). 

## RESUME STATE (canonical — read first)
- 02:41 TICK: core complete, HOLDING (no clearly-high-value tractable piece; cross-model gpt2 law unreliable, robustness re-runs marginal). POD TERMINATED (budget). Next ticks: hold unless user redirects; 08:11 tick -> just CronDelete 87b8d3de (deliverables already committed).
- PHASE: CORE COMPLETE (HOLD / optional-polish). POD 3ik6vir7c5tgcj (kept warm; OK to terminate if no eval planned).
- DELIVERED (all committed): converged theory (THEORY.md); magnitude law A~reuse(marg)/reuse(conj) VALIDATED AS A CURVE (A_generic 1.4-2.0 flat, A_diff 6.3->23.8 over N=2..4; fig_magnitude_law.png); separability argument measured (FRA 10^3-10^4x); 5-clause win-checklist; win/loss taxonomy (fig_sprint2_summary.png); SPRINT 2 SYNTHESIS in CAMPAIGN_REPORT.md; natural-text win anchor (copy-supp A=516x).
- The sprint's productive yield has SATURATED (FRA win-class is narrow; cycles 2-3 = theory-predicted negatives) and the headline theory is validated. NO more theory cycles (converged), NO churning negatives, NO make-work.
- Each cron tick (token-conscious): idempotent check; do AT MOST ONE genuinely high-value optional piece per ~hour if a clear one exists (e.g. cross-model magnitude-law on gpt2, robustness across seeds, or a tractable published-benchmark confirmation of a CONFIRMED win); ELSE note 'core complete, holding' and wait. If allowance pressured -> HOLD.
- CRON now 87b8d3de (every 2h, budget-conscious). FINALIZE at the 08:11 tick (~08:40): ensure all committed, terminate pod 3ik6vir7c5tgcj, CronDelete 87b8d3de. (Or earlier if user redirects.)

## CYCLE 1: theory+brainstorm+rank DONE (wuv06gbai). THEORY.md saved. Shortlist (ranked):
1. In-context PII regurgitation (gemma, 88) - cut (attr-question x PII-value) edge; on-target=extraction rate; collateral vs digit-direction projection-removal.
2. Selective copy-suppression release on OWT (gpt2 L10H7, 84) - extends s3 to real corpus.
3. Shared-endpoint MV/MK-NIAH deletion (gemma, 82) - cut (K1 x V-identity); SIBLING collateral (preserve K2->V); sweeps A(N) to TEST THE THEORY'S MAGNITUDE LAW (fixes s5).
4. Poisoned-RAG injection (gemma, 78) - cut (question x poison-answer); stratify by poison-legit overlap.
5. Knowledge-conflict fact-restoration (gpt2, 75) - Ortu CounterFact, context-vs-memory.
Theory KEY: A ~= reuse(marginal)/reuse(conjunction); FRA edits a CELL, linear steer only a ROW/COLUMN + writes to all residual readers.
PHASE: 2/eval. NEXT: screen gemma candidates (shared-endpoint, poison-RAG, knowledge-conflict-lite) via g_screen; gpt2 (copy-supp corpus, knowledge-conflict) next; Tier-2 on passers (sibling/legit-content KL vs projection-removal steer).

### CYCLE 1 EVAL (g5_s2gemma screen):
- shared-endpoint: R=+0.52 (P 0.37->0.18) -> PASS LBNR moderate -> Tier-2 (sibling test) IN FLIGHT (shared_endpoint_t2).
- poison-rag: R=+0.88 but base P(Berlin)=0.027 too weak (gemma-base) -> DEFER (need stronger elicitation / gemma-it).
- knowledge-conflict: R=+0.24 -> FAIL LBNR (in-context counterfactual robustly retrieved/distributed, base 0.73) -> screened out.
NEXT after shared_endpoint_t2: gpt2 cands (copy-supp-corpus reuse s3; knowledge-conflict gpt2 if worth it); then red-team; then append to CAMPAIGN_REPORT.md + relaunch theory workflow cycle 2.

## CYCLE 2: theory refined (set algebra over cells; clause-4 = target-specific-on-eval-distribution). Shortlist:
1. Corpus copy-supp release (gpt2, 89) - extends confirmed win, OWT external-validity.
2. PII regurgitation + sibling preservation (gemma, 86) - isomorphic to retrieval win, entity-specific query, differential cells.
3. NIAH entity-attribute leave-one-out transfer (gemma, 83).
4. NOVEL dual-use verbatim-block while preserving QA (gpt2, 80).
5. Delimiter/quote-TYPE matching (gpt2, 78, UNCERTAIN) - NEW structural mechanism class (Gao 2025).
6. Knowledge-conflict gpt2 (64).
PHASE 2/eval cycle 2. NEXT: screen delimiter(gpt2 novel) + PII(gemma) first.

### CYCLE 2 EVAL:
- delimiter-type matching (gpt2): paren-close CCF=0.732 (content-conj) but LBNR R=+0.23 -> FAIL (closing-delimiter redundant via grammar). NEGATIVE: structural matching is CCF-shaped but distributionally redundant. (bracket/quote token-not-found.)
- NEXT: PII entity-sibling (gemma, Alice/Bob both->frog; tests clause-4 entity-query specificity vs box-color generic A=7.9x).
