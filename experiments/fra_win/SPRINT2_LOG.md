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
- PHASE: 1/theory-brainstorm-rank (workflow wuv06gbai running: theory Fable -> brainstorm -> rank).
- POD: relaunching (POD_NAME=fra-win-pod). NEXT: when wuv06gbai done -> save THEORY.md, record shortlist here, EVAL via g_screen + corrected Tier-2 -> red-team -> append to CAMPAIGN_REPORT.md -> relaunch workflow (scriptPath wf_08ce8f47-617.js) for next cycle.
- Sprint-1 result (done): bilinear-QK CONFIRMED -- FRA ~1000-26000x more separable than content-gated steer (retrieval+acronym).

## CYCLE 1: theory+brainstorm+rank DONE (wuv06gbai). THEORY.md saved. Shortlist (ranked):
1. In-context PII regurgitation (gemma, 88) - cut (attr-question x PII-value) edge; on-target=extraction rate; collateral vs digit-direction projection-removal.
2. Selective copy-suppression release on OWT (gpt2 L10H7, 84) - extends s3 to real corpus.
3. Shared-endpoint MV/MK-NIAH deletion (gemma, 82) - cut (K1 x V-identity); SIBLING collateral (preserve K2->V); sweeps A(N) to TEST THE THEORY'S MAGNITUDE LAW (fixes s5).
4. Poisoned-RAG injection (gemma, 78) - cut (question x poison-answer); stratify by poison-legit overlap.
5. Knowledge-conflict fact-restoration (gpt2, 75) - Ortu CounterFact, context-vs-memory.
Theory KEY: A ~= reuse(marginal)/reuse(conjunction); FRA edits a CELL, linear steer only a ROW/COLUMN + writes to all residual readers.
PHASE: 2/eval. NEXT: screen gemma candidates (shared-endpoint, poison-RAG, knowledge-conflict-lite) via g_screen; gpt2 (copy-supp corpus, knowledge-conflict) next; Tier-2 on passers (sibling/legit-content KL vs projection-removal steer).
