# FRA × CIRCUIT TRACING — what does QK-resolution ADD to classic circuits? (started 2026-06-12)

## THE THESIS (the pivot)
Classic circuit tracing (IOI, induction, docstring, greater-than, successor, binding...) localizes WHICH heads/edges
carry a behavior and reads off the attention PATTERN — but treats the QK SCORE as a BLACK BOX (only intervention =
head/edge ablation). FRA decomposes the QK score S[q,k] = Σ_{μν} u^μ_q u^ν_k ω_{μν} into CONTENT×CONTENT feature-pair
CELLS. So FRA can say WHAT content-conjunction a circuit's key head computes in its QK, not just "it attends from A to B".
GOAL: take a few classic circuit results, ZOOM IN on the QK contribution of their key heads, and characterize what FRA's
feature-resolved QK decomposition ADDS. This plays to FRA's native strength (QK resolution) rather than the behavioral-
steering-win framing (where tuned linear baselines stay competitive — v1 injection 1.6x). Deliverable = "FRA adds X to the
mechanistic account of circuit Y" across 3-5 circuits, with a faithfulness check + symmetric red-team per claim.

## WHAT COUNTS AS "FRA ADDS" (sharp criteria — not just "confirms the known story in feature terms")
A1. SUB-MECHANISM RESOLUTION — a head whose pattern-level story is "attends to X" but whose QK is built from MULTIPLE
    distinct feature-conjunctions (FRA shows the head does several separable QK jobs).
A2. CONTENT-CONJUNCTION IDENTIFICATION — FRA pins down WHICH content feature the QK matches, settling an open/ambiguous
    question the pattern can't (e.g. IOI S-inhibition: name-identity vs position vs token-identity? induction: token vs
    content match? binding: lexical vs positional?).
A3. MORE-SURGICAL INTERVENTION — cutting the specific QK CELL vs ablating the whole HEAD changes the circuit's task
    behavior MORE SELECTIVELY (the head keeps its other jobs; only the cut conjunction is removed). Ground-truth task metric.
A4. CROSS-CIRCUIT FEATURE-CELL REUSE — the same QK feature-cell appears across circuits (e.g. induction cells inside IOI).
A5. QUANTITATIVE FAITHFUL ACCOUNT — the top-k cells RECONSTRUCT the head's QK score / its causal contribution (a
    feature-level "we explained the QK" with a reconstruction R^2 / behavioral recovery number).
Each claim MUST pass: (i) FAITHFULNESS (top-k cells reconstruct the QK score and/or the head's causal effect — not cherry-
picked); (ii) SYMMETRIC RED-TEAM (is the "added insight" real, or post-hoc storytelling on a noisy decomposition?).

## CIRCUIT SHORTLIST (seed — selection agent picks the best 3-5; QK-DRIVEN attention heads, documented + reproducible)
- IOI (Wang et al., gpt2-small) — THE canonical circuit. S-inhibition / name-mover / duplicate-token / induction / prev-token
  heads. RICH QK story w/ open content-conjunction questions (what does S-inhibition MATCH?). Strong A2/A1 candidate.
- INDUCTION heads (Olsson et al., gpt2-small/any) — the textbook QK "match prev-token-then-copy". Cleanest A5/A2 sanity +
  faithfulness anchor; also the BANKED behavioral win, so A3 connects.
- DOCSTRING circuit (Heimersheim, attn-only) — QK-rich, clean, small. A1/A2.
- GREATER-THAN (Hanna et al., gpt2-small) — CAUTION: largely MLP (the year computation); v1 flagged it G-post/D2 — include
  only as a NEGATIVE control (FRA should show the QK adds little = honest boundary).
- SUCCESSOR heads (Gould et al.) — QK computes ordinal succession; A2 (what ordinal feature?).
- BINDING (Mixing Mechanisms, Gur 2025) — IN FLIGHT under the organisms-hunt (rs-binding-precheck-1); its QK decomposition =
  "what FRA adds to the binding circuit". FOLD IN as circuit #0 (lexical-vs-positional = an A2 content-conjunction question).

## FRA TOOLKIT (reuse — it EXISTS)
fra/fra_func.py + fra/fra_sae_lens.py + fra/fra_analysis_*.py = the core S[q,k]=Σ u^μ_q u^ν_k ω_{μν} cell decomposition
(gpt2-small + gemma-2-2b, sae_lens). experiments/fra_win/jobs/ has the head-find + cell-cut + the gpt2/gemma FRA jobs
(g_screen, j2_selectivity, j13_ioi — IOI ALREADY touched in fra_win!). gpt2-small SAEs exist. So circuits are RUNNABLE with
existing tooling — the evaluator REPOINTS it, doesn't rebuild.

## CARRY-OVER METHOD LESSONS (hard rules — from the v1/v2 sessions)
- GROUND-TRUTH / quantitative > LLM-judge. FAITHFULNESS first: a decomposition that doesn't reconstruct the QK is a story, not a result.
- Test at the OPERATING regime; the symmetric red-team (false-positive AND false-negative) is mandatory on every "FRA adds X" claim.
- RunPod only (RP_API_KEY_MATS, pods rs-*); PARSE-GATE; NEW job id + unique pod name; ckpt() INSIDE loops + traceback-upload
  (v1/v2 had restart-loops + crash-before-upload); partial-upload+resume-proof. HF prefix fra_circuits_<name>/{code,results}.
- Budget-conscious; small pods first (gpt2-small is TINY -> cheap); commit every step; update RESUME STATE.

## THE TEAM (6 roles; orchestrator = me)
1. SELECTION+THEORY -> CIRCUITS.md: pick 3-5 circuits; per-circuit pre-register the published pattern-story + what FRA's QK
   decomp SHOULD show + which "ADDS" criterion (A1-A5) is in play + the FAITHFULNESS metric + runnability w/ the FRA toolkit.
2. EVALUATOR -> run the FRA QK decomposition per circuit's key heads; the feature-resolved QK story + the faithfulness number + the A3 surgical-cut test.
3. RED-TEAM (Workflow, symmetric: is the insight faithful + real, or post-hoc?).
4. PLANNING -> synthesize "what FRA adds" across circuits + decide next. (Hunt/screen roles fold into SELECTION+THEORY here.)

## PIPELINE / STATE MACHINE (cron-driven, idempotent)
A: SELECTION+THEORY -> CIRCUITS.md (now). B: PLANNING confirms the first circuit (likely induction = faithfulness anchor, then IOI). 
C: EVALUATOR runs FRA QK decomp on that circuit (faithfulness FIRST, then the "adds" finding + the A3 surgical-cut). D: RED-TEAM
(symmetric) -> PLANNING synth -> next circuit. E: STOP when 3-5 circuits done + red-teamed -> SYNTHESIS_CIRCUITS.md, commit, terminate, CronDelete.

## RESUME STATE (canonical)
- PHASE A IN FLIGHT (2026-06-12): SELECTION+THEORY agent a2c3428292e0513fd -> CIRCUITS.md (pick 3-5 circuits + pre-register "FRA adds X" + faithfulness metric). CRON 62c23deb (19,49 = every 30min) drives this track (replaced organisms2 cron 1814aac7, deleted).
- CIRCUIT #0 = BINDING (DONE pre-check, STRONG-GO): L22H4 QK is CONTENT-addressed (R_gen 0.90/R_prefill 0.075; sibling-bleed 0.0009;
  29/29 content-not-position) -> FRA resolved the lexical-vs-positional question (A2) + the cell-cut-vs-head-ablation is the A3 example.
  SELECTIVITY WIN-TEST IN FLIGHT: eval a568274a1cfbbcce5 -> rs-binding-selectivity-1 (FRA cell-cut vs best-tuned linear + head-ablation
  on intra-instance sibling control; WIN = >=2x lower sibling-collateral at matched target-suppression). FRA oracle sib-collateral floor ~0.1%.
- PHASE A DONE: CIRCUITS.md (a2c3428292e0513fd) — 5 circuits ranked: #0 binding(done), #1 induction L5H5 gpt2 (faithfulness anchor),
  #2 IOI (S-inhibition QK = name-id vs position vs dup-token?), #3 docstring (A1 double-dissoc + A4 reuse), #4 greater-than (MLP neg control).
  ALL on gpt2-small+res-jb (reuse j1_induction_explore/j2_selectivity/j13_ioi/g_screen).
- PHASE C IN FLIGHT (two parallel threads):
  (1) CIRCUIT #1 INDUCTION FAITHFULNESS ANCHOR: eval a74e9bdea1afb47ab -> rs-circ-induction-1 (gpt2-small). Reconstruct the induction QK:
      R^2>=0.7 + dominant cell = SELF/token-match (NOT positional/sink) + A3 cell-cut-vs-head-ablation selectivity. PASS = decomp trustworthy
      -> green-light IOI. FAIL = decomp unreliable on a known-simple QK -> STOP (most important possible result). Log -> INDUCTION_LOG.md.
  (2) BINDING SELECTIVITY win-test (circuit #0 A3): eval a568274a1cfbbcce5 -> rs-binding-selectivity-1 (FRA cell-cut vs best-tuned linear + head-ablation; WIN=>=2x).
- NEXT: read induction faithfulness PASS/FAIL + binding selectivity WIN/NULL. If induction PASS -> EVALUATOR circuit #2 IOI. If binding WIN -> symmetric false-positive red-team. Then loop.
- Builds on: fra_win/ (FRA toolkit + j13_ioi + g_screen), fra/ module, fra_organisms2/ (binding circuit #0), v1 boundary-map theory.
