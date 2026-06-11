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
- PHASE: 2/eval, cycle 1. POD 3ik6vir7c5tgcj. Theory cycle-1 DONE (THEORY.md). 
- IN FLIGHT: g5_s2gemma screen (shared-endpoint, poison-rag, knowledge-conflict; gemma; CAUSAL heads). DO NOT resubmit.
- NEXT: read g5_s2gemma -> for CCF∧LBNR passers run corrected Tier-2 (sibling-collateral / legit-content KL vs PROJECTION-REMOVAL steer, matched removal); then screen gpt2 cands (copy-supp-corpus reuse s3, knowledge-conflict gpt2); then red-team workflow; then append to CAMPAIGN_REPORT.md + relaunch theory workflow (wf_08ce8f47-617.js) for cycle 2.
- THEORY TEST PRIORITY: shared-endpoint A(N) sweep validates A~reuse(marginal)/reuse(conjunction).
- Stop ~08:40. Workflow scriptPath: wf_08ce8f47-617.js. Cron 0a9a40c5.

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
