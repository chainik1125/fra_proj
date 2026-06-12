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
- PHASE A IN FLIGHT (2026-06-11): two front agents running (parallel, independent):
    LIT-REVIEW agent id a0d7585cd8b676f5a -> experiments/fra_organisms/ORGANISMS.md (cited organism catalog).
    THEORY agent     id a8628cfc75d2a200c -> experiments/fra_organisms/SCREENING_RUBRIC.md (operational FRA-cuttability rubric).
- CRON: 34ae3a64 (13,43 * * * * = every 30 min) drives the pipeline. (Replaced prior session's 25ae1f3a, deleted.)
- NEXT (cron state machine B): when BOTH ORGANISMS.md + SCREENING_RUBRIC.md land -> spawn APPLICATION/SCREEN agent (consume
  both -> SCREEN.md: scored+ranked candidates + concrete FRA variant per top one). Then PLANNING picks top 1-2 -> EVALUATOR
  builds+runs on RunPod (ground-truth metric, FRA-cut vs linear/weight baseline at matched effect = SELECTIVITY win test) ->
  RED-TEAM (symmetric) -> PLANNING synth -> loop or consolidate.
- Orchestrator (me): verify-clean + advance idempotently each tick; spawn next-stage agent only when its inputs exist; commit every step.
- NO pods yet. All prior-session syco pods EXITED. The two RUNNING non-rs pods (awake_pink_centipede, han) are NOT mine.
