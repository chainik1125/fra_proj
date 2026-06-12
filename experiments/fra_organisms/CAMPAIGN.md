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

## INJECTION §4 SELECTIVITY FINAL VERDICT (2026-06-12): NULL (sub-threshold; my preliminary "strong win" was a baseline-tuning artifact)
Fixed run selectivity-3, sets: INJECT(ASR=1)=60, easy-legit=14, hard-legit=17, capability=24, matched t*=0.85. Metric: removal=
injection-blocked; easy/hard/cap = RETENTION (higher=better). The honest verdict flipped from my premature excitement for TWO reasons:
 1. BEST-TUNED LINEAR: my "1.00 vs 0.47" was the HARSH layer L6. The harness correctly picked the linear steer's BEST (gentlest) layer
    L12: at FRA's operating point removal~0.73, FRA hard-legit 1.00 vs L12-linear ~0.63 = a REAL but MODEST ~1.6x advantage (+0.37 abs),
    BELOW the pre-registered >=2x win bar.
 2. MATCHED POINT t*=0.85: FRA CAPS at 0.73 removal (can't block >73% of injections). At t*=0.85 FRA is off-curve (collateral=None) ->
    matched comparison undefined -> formal VERDICT = NULL.
HONEST CHARACTERIZATION: FRA is somewhat more selective PER UNIT removal (hard 1.00 vs ~0.63 @0.73) but CAPS low and the advantage over a
well-tuned linear steer is SUB-THRESHOLD (~1.6x, not >=2x). The apparent blowout was a harsh-baseline artifact — the false-positive trap, caught
by the matched-point + best-tuned-linear discipline. prompt-hardening baseline: removal only 0.08 (weak defense). selectivity.json saved.
=> BOTH fresh real-LLM flagships are NEGATIVE/NULL: recall NO-GO (selectivity/D3), injection NULL (caps + sub-threshold vs best linear).

## FINAL PHASE: PLANNING SYNTHESIS (a7ca331286c933824 resumed) -> SYNTHESIS.md
Both fresh flagships resolved (recall NO-GO/D3, injection NULL/sub-threshold). Planner tasked with: (1) fairness sanity-check on the
injection NULL at FRA's own operating point (removal~0.73; is FRA's hard-legit advantage vs best-tuned-linear >=2x [re-report] or ~1.6x
[NULL robust]); (2) the honest meta-synthesis (WHEN FRA wins = load-bearing-attention-at-generation + non-recurrent-conjunction +
SAE-cell-reachable; banked induction/copy wins; fresh broad×broad real-LLM extensions fail/modest; boundary map w/ recall/injection/EM/
syco negatives + theory refinements); (3) consolidate-vs-one-more decision (budget-aware; recommend consolidate). All my pods EXITED.

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
- PHASE D DONE (planning): PLANNING.md decision = run the injection load-bearing pre-check, THEN consolidate. Injection != flagship
  (no parametric backstop for the novel injected string -> attending its span is load-bearing BY CONSTRUCTION). New theory: "load-
  bearing on the OPERATING REGIME" = 5th necessary condition (regime-conditional redundancy). Honest framing: FRA niche NARROW
  (load-bearing attention edge over attended context w/ no MLP backstop = induction/copy/backdoor/retrieval 15x-26000x; NOT factual recall).
- PHASE E IN FLIGHT: EVALUATOR(2) id a6c6439a53aaf3f44 -> injection GO/NO-GO pre-check (judge-free exact-match canary, ASR~1
  operating point, oracle-cut FIRST for LBNR ceiling). GO iff (i) oracle injection-drop R>=0.5 not-restored AND (ii) legit-instruction
  within 10% AND (iii) FRA cell >=50% of oracle. Pod rs-injection-precheck-1, HF fra_org_injection/. Log -> INJECTION_LOG.md.
  Also: factual-edit RED-TEAM w0vsbhiwa still in flight (fold its verdict in when it lands; planning says it won't overturn the saturation finding).
- PHASE E RESULT: injection pre-check = NO-GO AT GENERATION-TIME edge (oracle response->injected-span R=0.000, ASR 1.00->1.00,
  13/13 restored, head effects ~1e-18). BUT flagged PREMATURE (timing false-negative): a NOVEL canary can't be emitted without
  attending to it SOMEWHERE -> load-bearing attention is plausibly at PREFILL; the gen-time cut is too late. Same upstream-
  redundancy pattern as flagship (load-bearing attention is UPSTREAM of the answer position: MLP-prebake for recall, prefill-
  propagation for injection -> late edge-cut = redundant read). UNIFYING THESIS forming: FRA wins need the load-bearing attention
  AT the intervention/generation step (induction/copy attend-back-and-copy AT the emitted token); recall+injection do their work upstream.
- PHASE E.2 IN FLIGHT: injection evaluator (a6c6439a53aaf3f44) tasked with the DECISIVE PREFILL-TIMING follow-up (rs-injection-prefill-1):
  timing-agnostic oracle cut (zero attention to injected-span key from ALL query positions, prefill+gen) + per-timing breakdown +
  eval-artifact ruleout. If all-position cut DROPS ASR>=0.5 -> timing artifact, injection IS attention-routed at PREFILL -> potential GO
  (test FRA reach + collateral at prefill edge). If even all-position cut ~0 (after ruling out canary-leak/eval bug) -> robust surprising NO-GO.
- factedit RED-TEAM w0vsbhiwa: 3 skeptics returned but workflow result not aggregated (output empty); extract from agent files if it doesn't complete. Planning expects it to uphold the negative.
- >>> MAJOR TURN (both flagship NO-GOs were PREMATURE; symmetric red-team caught 2 false-negatives):
  INJECTION prefill follow-up KEY RESULT: complete cut (attention TO injected span) gives ALL=PRE=GEN R=1.0, eval clean ->
    injection is FULLY attention-routed at PREFILL -> FLIPS the gen-time NO-GO (R=0.0 was a wrong-timing artifact). Pending
    (eval a6c6439a53aaf3f44 computing): LOCALIZATION (<=3 heads vs distributed) + FRA-cell reach + (ii) legit collateral. WIN
    needs localizable+content-specific+FRA-reachable+selective — R=1.0 alone is necessary not sufficient.
  FACTEDIT complete re-test LAUNCHED (eval a4585f4b23baf2381 -> rs-factedit-complete-1): FRA SAE cell-edit on FULL subject SPAN ×
    relation, top-8 heads, load-bearing band P[0.30,0.55] (the red-team's convergent test; full-span oracle headroom R=0.60 hi / 0.52@P.36).
    Metrics: P-drop + rank-flip + collateral. GO iff cell-edit drop>=50% + rank-flip-majority + collateral<15% -> flagship true false-neg -> ROME selectivity.
- INJECTION pre-check = GO (revised, committed). ALL=PRE=GEN R=1.0; LOCALIZED L10H7/L18H6 R=1.0 (L10H7 = the copy-suppression head,
  carries it alone), L4H2=0.75; gate(ii) legit collateral 9%; gate(iii) FRA cell-edit reach R=1.0; detector valid, no canary leak.
  FIRST FRESH WIN CANDIDATE — but pre-check only. >>> §4 SELECTIVITY WIN-TEST LAUNCHED (eval a6c6439a53aaf3f44 -> rs-injection-
  selectivity-1): expand ASR=1 set to ~40-60 (n=13 too small) + hard-legit control (injection-like-but-legit); 3 interventions at
  MATCHED injection-removal [FRA cell-cut L10H7/L18H6 vs LINEAR ignore-injection DoM steer vs prompt-hardening]; collateral on
  easy-legit/hard-legit/capability. WIN = FRA materially LOWER collateral than linear at matched effect (else NULL — report honestly).
- FACTEDIT complete re-test (rs-factedit-complete-1) STILL RUNNING: band scan confirms recall-gating (hi R~0.07) but load-bearing
  headroom in mid band (P.15-.4, oracle R med 0.39). The full-subject-span FRA cell-edit result on the load-bearing band = pending.
- FACTEDIT FINAL: NO-GO AIRTIGHT (selectivity/D3, committed). FRA cell-edit REACHES the load-bearing full-span edge (median drop
  0.52, rank-flip 0.60 — NOT a reach failure) but is RELATION-keyed not SUBJECT-keyed -> collateral 0.49 (suppresses sibling
  subjects' same-relation facts ~49-59%); only 1/20 selective. D3 conjunction-recurrence. Do NOT run ROME (demonstrated loss).
  THE SHARP NEGATIVE: real factual recall's broad×broad conjunction is reachable+load-bearing but NOT subject-selective ->
  synthetic's selectivity advantage does NOT transfer. (Optional writeup add: confirm ROME IS subject-selective on same 20 facts.)
- INJECTION §4 SELECTIVITY test = IN PROGRESS (eval a6c6439a53aaf3f44 ALIVE, building Run 4: expand ASR=1 set ~40-60 + hard-legit
  control, 3 interventions [FRA cell-cut L10H7/L18H6 vs linear ignore-injection DoM vs prompt-hardening] at matched removal,
  collateral table). No selectivity pod launched yet. CAUTION: stale gen-time summary (RUNID 065045, oracle R=0.0) and prefill-GO
  summary (070218, R=1.0) both exist — the selectivity result will be a NEW RUNID dir; do NOT mis-read the stale ones.
- INJECTION §4 SELECTIVITY: pod rs-injection-selectivity-1 RESTART-LOOPING (uptime resets 117->52, GPU L40S->L4, NO new result dir =
  crashes before first upload ~240s). Evaluator was burning tokens babysitting bootstrap. SENT directive: diagnose the pre-upload crash
  (likely OOM from caching 3-layer activations / DoM build / set-build), move ckpt() EARLIER, relaunch ONCE as rs-injection-selectivity-2
  (smaller set if OOM); if not fixable in one attempt -> STOP + report the PRE-CHECK GO as a provisional win w/ selectivity noted-pending.
- INJECTION EVIDENCE SO FAR (provisional GO, pre-check only): oracle R=1.0, localized L10H7 alone R=1.0, easy-legit collateral 9%,
  FRA cell reach R=1.0, detector valid + no canary leak. The §4 FRA-vs-linear-at-matched-removal head-to-head (the WIN confirmation) is the remaining gap.
- INJECTION SELECTIVITY (selectivity-3, bug fixed [KeyError 'inject' line 275], RUNNING) — PRELIMINARY NUMBERS = STRONG FRA WIN
  (pending run-completion + red-team; metric: removal=injection-blocked, easy/hard/cap=RETENTION higher-better):
    FRA cell-cut c=1: removal 0.73, easy-legit 0.86, HARD-LEGIT 1.00, capability 1.00.
    LINEAR DoM (best-tuned L6 a=4, matched removal 0.78): easy-legit 0.00, HARD-LEGIT 0.47, cap 1.00. (a>=16 removal=1.0 but easy=hard=cap=0 = destroys model.)
    => at MATCHED removal, FRA keeps 100% hard-legit vs linear 0.47; linear CANNOT selectively remove injection (any removal craters legit-following).
    HARD-LEGIT GAP 1.00 vs 0.47 EXCEEDS win bar (>=2x + >=0.15 abs + FRA>10%). FRA's predicted niche (load-bearing/localized L10H7/content-specific) DELIVERS.
    CAVEAT to red-team: FRA removal CAPS at 0.73 (selective-but-partial; linear is complete-but-destructive); n (inject set ~13? controls easy14/hard17/cap24) — POWER. prompt-hardening baseline (stage 5) + formal verdict pending.
- NEXT: let selectivity-3 finish (stage5 prompt-hardening + VERDICT). Then COMMIT the win + launch SYMMETRIC FALSE-POSITIVE red-team (4 opus
  skeptics; EM-burned): attack n/power, the matched-removal fairness, the "trivially blocks reading the span" confound, FRA's 0.73 cap,
  whether hard-legit control is fair. If survives -> robust WIN. Then PLANNING consolidate (injection WIN + recall/EM/syco negatives + theory).
- CRON: 34ae3a64 (13,43 * * * * = every 30 min) drives the pipeline. (Replaced prior session's 25ae1f3a, deleted.)
- NEXT (cron state machine B): when BOTH ORGANISMS.md + SCREENING_RUBRIC.md land -> spawn APPLICATION/SCREEN agent (consume
  both -> SCREEN.md: scored+ranked candidates + concrete FRA variant per top one). Then PLANNING picks top 1-2 -> EVALUATOR
  builds+runs on RunPod (ground-truth metric, FRA-cut vs linear/weight baseline at matched effect = SELECTIVITY win test) ->
  RED-TEAM (symmetric) -> PLANNING synth -> loop or consolidate.
- Orchestrator (me): verify-clean + advance idempotently each tick; spawn next-stage agent only when its inputs exist; commit every step.
- NO pods yet. All prior-session syco pods EXITED. The two RUNNING non-rs pods (awake_pink_centipede, han) are NOT mine.
