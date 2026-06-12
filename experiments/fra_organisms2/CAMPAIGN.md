# FRA ORGANISM HUNT v2 — find a NOVEL organism that meets ALL FOUR win-conditions (started 2026-06-12)

## GOAL
The v1 session produced the WHEN-FRA-wins boundary map (experiments/fra_organisms/SYNTHESIS.md). This session is
HYPOTHESIS-DRIVEN: hunt a NOVEL, runnable, safety/reasoning-relevant organism that clears ALL FOUR win-conditions
AND where a linear steer has HIGH collateral — i.e. an organism where FRA's QK cell-cut should DECISIVELY beat the
best-tuned linear baseline (>=2x selectivity, not the v1 injection ~1.6x). Deliverable: a robust red-teamed WIN on a
fresh organism, OR a sharpened "even here it fails because X" that further tightens the boundary map.

## THE FOUR WIN-CONDITIONS (the hunt GATE — an organism must clear ALL, and each v1 failure violated one)
1. LOAD-BEARING attention edge — behavior carried by an attention EDGE, not an MLP/direction payload. (EM violated: G-post MLP direction.)
2. CONTEXT-SUPPLIED content — the key content comes from CONTEXT with NO parametric/MLP backstop, so attending it is load-bearing
   by construction. (Factual recall violated: the object is MLP-pre-baked on well-recalled facts.)
3. CONSUMED AT THE ANSWER STEP — the load-bearing attention is at the generation/answer position (attend-back-and-copy AT the emitted
   token), NOT upstream (prefill-propagation / early MLP enrichment). (Sycophancy violated: deference baked upstream; injection's load-
   bearing read was at PREFILL not the answer step — still attention but the answer-step edge alone was a redundant late read.)
4. NON-RECURRENT CONJUNCTION reachable by ONE SAE cell — the (query-content × key-content) pair is DISTINCTIVE and does NOT recur across
   siblings, so the cell-cut is SUBJECT/INSTANCE-specific. (Factual recall violated: subject×relation is RELATION-keyed -> recurs across
   subjects -> ~49-59% sibling bleed, the D3 conjunction-recurrence failure.)
PLUS the MAGNITUDE LAW (why FRA beats linear): A ≈ reuse(marginal)/reuse(conjunction). FRA's advantage is LARGE only when BOTH endpoints
are broadly REUSED (so a linear steer on either direction has high collateral) BUT the specific CONJUNCTION is RARE/distinctive. The hunt
must target HIGH reuse(endpoints) × LOW reuse(conjunction). The banked wins (induction, copy-suppression, in-context backdoor: 15x-516x) are
the proof-of-concept; the goal is a NOVEL organism with this structure, ideally safety/reasoning-relevant, where the synthetic broad×broad
advantage genuinely transfers.

## CANDIDATE CLASSES TO HUNT (seed — the team finds the best published instance + runnable artifact)
- IN-CONTEXT VARIABLE BINDING / STATE TRACKING (reasoning): "a=7, b=3, ... what is a?" or entity-state tracking — the binding (distinctive
  key × distinctive value) is attention-mediated at the answer step, NON-recurrent; cut ONE binding, preserve siblings. Closest REAL analogue
  of the synthetic win. Linear steer can't separate a's-value from b's-value (both use the broad "retrieve value" dir) -> high collateral. STRONG.
- CONTEXT-GROUNDED RETRIEVAL / RAG ATTRIBUTION (safety): multi-passage context; cut the answer's attention to ONE specific (poisoned/wrong)
  passage's distinctive content while preserving grounding in the others. Distinctive passage content; high collateral for a linear "ignore-source" steer.
- IN-CONTEXT BACKDOOR with a DISTINCTIVE multi-token trigger (safety): generalize the banked single-token win to a published poisoning organism.
- COREFERENCE / ENTITY-ATTRIBUTE BINDING in multi-entity narratives: bind the right attribute to the right NAMED entity; cut one mis-binding.
- IN-CONTEXT DEFINITION/TRANSLATION: "in this dialect X means Y; use X" — novel context-supplied distinctive mapping.
AVOID: direction/persona/payload behaviors (EM/sycophancy/refusal = known G-post); recurrent-conjunction relations (recall = D3); anything
whose load-bearing work is upstream of the answer step.

## CARRY-OVER METHOD LESSONS (hard rules — from v1)
- The cheap LBNR/oracle PRE-CHECK must sample the application's OWN OPERATING REGIME (v1 recall's R=+0.86 false-GO was a weak-fact artifact).
- Test the COMPLETE cut (full content span, all timing/positions) BEFORE filing a negative (v1: factedit last-token & injection gen-time were
  premature). Symmetric red-team caught 2 premature negatives.
- SELECTIVITY win-test = FRA cell-cut vs the BEST-TUNED linear baseline (sweep layers×alphas, give linear its best shot) AND/OR the domain
  weight-edit baseline, at MATCHED on-target effect, on a HARD control (instances that resemble the target), GROUND-TRUTH metric. WIN = >=2x
  selectivity gap + >=0.15 absolute + FRA retains >10% on the hard control. (v1 injection: 1.6x = NULL.)
- GROUND-TRUTH metrics >> LLM-judge (the EM judge-bucketing lesson). Judge with Claude if unavoidable (OpenAI dead).

## RULES (same as v1)
- RunPod only (RP_API_KEY_MATS; RUNPOD_API_KEY=$RP_API_KEY_MATS). Pods rs-* (reaper whitelist). gemma-2-2b-it small (L4/L40)+GemmaScope;
  bigger needs A40/L40. Pre-check RUNNING names. PARSE-GATE pod scripts; NEW job id + UNIQUE pod name per run; ckpt() INSIDE long loops
  (the v1 monolithic-checkpoint + restart-loop lessons). Partial-upload+resume-proof. HF prefix per-candidate (fra_org2_<name>/{code,results}).
- IDEMPOTENT: check git/pods/HF before acting; never double-submit. Commit every step; update RESUME STATE. Budget-conscious; small pods first;
  a clean characterization (win OR sharpened-negative) is the deliverable. Agents on opus 4.8; resume on token reset.

## THE TEAM (6 roles; orchestrator = me)
1. BRAINSTORM/HUNT -> ORGANISMS2.md: published organisms + artifacts meeting all four conditions (targeted, not broad). 
2. THEORY -> PREDICTOR2.md: sharpen the four-conditions + magnitude-law into a per-candidate WIN/LOSE pre-registration + the win-test design.
3. SCREEN -> SCREEN2.md: score candidates on the 4 gates + reuse(endpoints)/reuse(conjunction) prior + runnability; pick top 1-2.
4. EVALUATOR -> the cheap operating-regime pre-check, then selectivity vs best-tuned linear.
5. RED-TEAM (Workflow, symmetric). 6. PLANNING -> synthesize + decide.

## PIPELINE / STATE MACHINE (cron-driven, idempotent)
A: BRAINSTORM/HUNT + THEORY (parallel, now). B: SCREEN consumes both. C: PLANNING picks top -> EVALUATOR pre-check (operating-regime) ->
selectivity vs best-tuned linear. D: RED-TEAM (symmetric) + PLANNING synth. E: STOP on robust red-teamed WIN, or candidates exhausted, or
budget/time -> SYNTHESIS2.md, commit, terminate pods, CronDelete.

## RESUME STATE (canonical)
- PHASE A: THEORY DONE (PREDICTOR2.md committed, aa42049de80f00064). HUNT agent a51751376c62b9305 -> ORGANISMS2.md STILL RUNNING. CRON 1814aac7 (17,47).
  PREDICTOR2 headline: in-context VARIABLE BINDING = top WIN candidate (P~0.65, clears all 4 strong bands). Best discriminator = P3 sibling-bleed
  at operating regime + order-shuffle role-vs-content audit (KEY falsifier: positional binding -> A~1.9x — build the shuffle control in).
- NEXT (cron B): when ORGANISMS2.md lands -> spawn SCREEN (score on 4 conditions + P1-P4 + runnability -> pick top, likely variable-binding) ->
  PLANNING pick -> EVALUATOR operating-regime pre-check (P3 sibling-bleed + order-shuffle audit FIRST — cheapest decisive proxy) -> selectivity vs best-tuned linear.
- NEXT: when ORGANISMS2.md + PREDICTOR2.md land -> spawn SCREEN -> PLANNING pick -> EVALUATOR operating-regime pre-check -> selectivity-vs-best-linear -> RED-TEAM.
- Builds on experiments/fra_organisms/ (SYNTHESIS.md = boundary map, SCREENING_RUBRIC.md = D1-D8 + §4 selectivity test, ORGANISMS.md = the v1 catalog).
