# FRA MODEL-ORGANISMS SCREEN — 10h autonomous session (started 2026-06-11)

## GOAL
Systematically find the real-LLM behavior(s) where an FRA variant (feature-resolved QK cell-cut, or a relational
attention edit) gives a clear BEHAVIORAL-INTERVENTION ADVANTAGE over linear steering / weight-editing baselines.
Prior session evaluated EM + sycophancy -> both essentially G-POST (FRA's wrong class). This session screens the
PUBLISHED landscape of model organisms / reasoning / safety behaviors against a sharp FRA-cuttability criterion,
evaluates the most promising, and red-teams every claim. A clean characterization ("FRA wins HERE, not THERE,
because X") is the deliverable — positive OR negative.

## THE INHERITED REFRAME (the screening lens — from the prior session)
FRA's surgical cell-cut beats a linear steer IFF the behavior is RELATIONAL/ASSOCIATIVE and the relation is
carried by the attention SCORE (G-SCORE), with BOTH endpoints broad (reused) features. Diagnostic:
- DIRECTION / payload / persona  => G-POST => FRA LOSES (a linear steer is near-optimal):
  emergent-misalignment, sycophancy-decision, refusal (Arditi: single linear direction!), style/persona.
- RELATION / binding / retrieval  => G-SCORE => FRA's HOME TURF:
  factual recall (subject×relation), in-context induction/copying (the ONE proven fra_win), entity-attribute
  binding, coreference, instruction/injection routing, RAG passage-attention.
FRA-CUTTABILITY CHECKLIST (must clear ALL): C1 behavior-specific (base-reversion OLS partial), C2 content-
conditional, C3 localizable (>=60% effect in <=3 heads/cells), C4 no-coherence-collapse, + SELECTIVITY (the win:
LOWER collateral than a matched linear steer at equal on-target effect). See ../fra_hierarchy/THEORY_HIERARCHY_v2.md
+ BRAINSTORM_applications.md.

## THE TEAM (6 roles; orchestrator = me, mechanical: cron/commits/pods/idempotency)
1. LIT-REVIEW agent -> ORGANISMS.md: systematic, cited list of published model organisms / reasoning / safety-
   relevant behaviors (EM, sleeper/backdoors, sycophancy, refusal, induction, IOI/factual circuits, function
   vectors, in-context learning, deception/alignment-faking, jailbreaks/injection, gridworld/CoT-unfaithfulness,
   steering-vector orgs, etc.). For each: the paper, the model+artifact availability, the behavior, the known
   mechanism (if any). Web + the repo's docs/dmitry/literature/.
2. APPLICATION/SCREEN agent (needs lit + theory rubric) -> SCREEN.md: score each organism on the FRA-cuttability
   criterion (G-score-vs-G-post prior, broad×broad-ness, localizability, metric cleanliness, artifact availability,
   baseline-to-beat). Rank. Propose the concrete FRA variant per top candidate.
3. EVALUATOR agent (needs screen) -> builds + runs the experiments on RunPod for the top candidates. Ground-truth
   metrics >> judge. Corrected protocols (leak-checks, power, base-reversion, selectivity vs linear/weight baseline).
4. RED-TEAM agent/Workflow -> attacks EVERY claim (false-positive AND false-negative; symmetric). 4 opus skeptics.
5. THEORY agent -> what the theory predicts for each candidate; refines the rubric; explains wins/losses mechanistically.
6. PLANNING agent -> synthesizes current state across all agents + DECIDES where to go next each milestone.

## PIPELINE / STATE MACHINE (cron-driven, idempotent)
PHASE A (parallel, start now): LIT-REVIEW + THEORY (rubric). 
PHASE B: APPLICATION/SCREEN consumes lit+rubric -> ranked candidates + concrete FRA variants.
PHASE C: PLANNING picks top 1-2 -> EVALUATOR builds+runs (ground-truth, selectivity-vs-baseline).
PHASE D: RED-TEAM each evaluated claim; PLANNING synthesizes -> next candidate or consolidate.
Loop C-D over candidates until budget/time or a clear win+robust-red-team, OR all top candidates exhausted -> SYNTHESIS.

## RULES (same as prior session — hard constraints)
- COMPUTE ON RUNPOD only (RP_API_KEY_MATS; set RUNPOD_API_KEY=$RP_API_KEY_MATS). Pods named rs-* (reaper whitelist).
  gemma-2-2b-it small (L4/L40); bigger models (gemma-2-9b, Qwen-7B) need A40/L40. Pre-check RUNNING names (dup launch).
- JUDGE WITH CLAUDE (OpenAI key DEAD = insufficient_quota). Raw HTTP via ANTHROPIC_API_KEY_MATS (template:
  ../fra_hierarchy/rejudge_pf.py). PREFER GROUND-TRUTH metrics (no judge) wherever possible — the EM judge-bucketing lesson.
- IDEMPOTENT: check git log / pod status / HF outbox before acting; NEVER double-submit; NEW job id + unique pod name per run.
- PARSE-GATE every pod script (python3 -c ast.parse) before upload/launch. Partial-upload + resume-proof. ckpt() INSIDE
  long loops (the v2 monolithic-checkpoint blinded monitoring). HF prefix per-candidate (e.g. fra_org_<name>/{code,results}).
- HF: 128-commit/hr/repo cap; periodic+final uploads only. hf CLI (huggingface-cli is a dead stub).
- RED-TEAM SYMMETRY: red-team positives (false-positive) AND negatives (false-negative). The prior session killed an EM
  false-positive AND a sycophancy-v1 false-negative — both mattered.
- BUDGET: budget-conscious; small pods first; a clean negative is a valid result. Token allowance exhausted -> resume on reset,
  agents on opus 4.8. Commit after every step. Update RESUME STATE every step.

## RESUME STATE (canonical)
- PHASE A DONE: ORGANISMS.md (lit-review a0d7585cd8b676f5a) + SCREENING_RUBRIC.md (theory a8628cfc75d2a200c) written + committed.
  Lit-review top-5: (1) factual-assoc editing [flagship], (2) prompt-injection, (3) copy-suppression L10H7 [measured 22.7x win
  — VERIFY it's a fresh FRA-decomposition contribution not a re-run of fra_win], (4) induction/backdoor, (5) sycophancy
  attend-to-doubt edge [arXiv 2601.16644 CONTRADICTS prior G-post verdict — sharp re-test]. Rubric flagship prediction = factual editing WIN.
- PHASE B DONE: SCREEN.md (a5f0a5c3851a1c784) — all 22 scored, prior NO-GOs reproduced. #1 = FACTUAL-RECALL EDIT (only OPEN
  TIER-1; the other 3 T1 are banked wins). Copy-suppression 22.7x is NOT fresh (banked 516x). Encouraging: prior LBNR probe
  fact-recall attn=0.81 R=+0.86 (D4 load-bearing); CCF=0 = SAE-reach concern not G-post.
- PHASE C IN FLIGHT: EVALUATOR id a4585f4b23baf2381 -> the CHEAP GO/NO-GO PRE-CHECK ONLY (not the full campaign): gemma-2-2b-it +
  GemmaScope-att SAEs, ~25-30 recalled CounterFact facts, (subject×relation) cell-edit at top<=3 extraction heads (reuse
  fra_win/jobs/g_screen+j2_selectivity + fra/fra_func/fra_sae_lens). GROUND-TRUTH P(target) from logits. GO iff (i) cell-edit
  drops P(target)>=50% at c~1-2 AND (ii) held-out same-relation/diff-subject drops <15%. Pod rs-factedit-precheck-1, HF
  fra_org_factedit/. Log -> FACTEDIT_LOG.md. NO-GO = the pre-registered flagship negative (sharp bound).
- PHASE C RESULT: factual-edit pre-check = NO-GO. Gate(i) FAIL (median 0.6% P(target) drop, 0/28 reach 50%). Diagnostic (2nd pod):
  oracle edge-cut R is RECALL-STRENGTH-GATED — 0.32 weak (P.03-.15) / 0.39 mid / 0.07 strong (P.6-.99, 0/12). Sanity France->Paris
  R=0.32 (machinery OK). => subject->object transport REDUNDANT/SATURATED on recalled facts = the regime editing targets.
  Load-bearing ceiling (NOT SAE-reach). Literature-consistent (ROME edits MLP). Pre-registered flagship negative (falsifier #1).
- PHASE D IN FLIGHT (parallel):
    RED-TEAM Workflow w0vsbhiwa (3 false-negative skeptics: head-localization / regime-model-MLP / salvage-metric).
    PLANNING agent a7ca331286c933824 -> PLANNING.md (synthesize state + DECIDE next: prompt-injection T2 vs weak-fact salvage vs
    consolidate; + the cheapest go/no-go pre-check for the pick). Key new refinement = "load-bearing on the OPERATING REGIME" is a
    5th necessary condition beyond G-score (a structurally-ideal G-score relation can still be redundant where it matters).
- NEXT: synthesize red-team + planning -> act on planning's decision (likely prompt-injection pre-check, IF its injected-instruction
  edge is load-bearing unlike the recalled-fact edge) OR consolidate. Then loop or write SYNTHESIS.
- CRON: 34ae3a64 (13,43 * * * * = every 30 min) drives the pipeline. (Replaced prior session's 25ae1f3a, deleted.)
- NEXT (cron state machine B): when BOTH ORGANISMS.md + SCREENING_RUBRIC.md land -> spawn APPLICATION/SCREEN agent (consume
  both -> SCREEN.md: scored+ranked candidates + concrete FRA variant per top one). Then PLANNING picks top 1-2 -> EVALUATOR
  builds+runs on RunPod (ground-truth metric, FRA-cut vs linear/weight baseline at matched effect = SELECTIVITY win test) ->
  RED-TEAM (symmetric) -> PLANNING synth -> loop or consolidate.
- Orchestrator (me): verify-clean + advance idempotently each tick; spawn next-stage agent only when its inputs exist; commit every step.
- NO pods yet. All prior-session syco pods EXITED. The two RUNNING non-rs pods (awake_pink_centipede, han) are NOT mine.
