# FRA HIERARCHY campaign — broad×broad cell-cutting (started 2026-06-11)

HYPOTHESIS (PI): confirmed FRA wins are all specific-feature->single-token (lowest hierarchy rung; cutting a cell = cutting a bigram). The INTERESTING regime = BROAD features (many words) + HIGH hierarchy (abstract concepts). Flagship: cut (misaligned-PERSONA feature x DOMAIN feature) -> SELECTIVELY block EM generalizing to that domain, preserving persona elsewhere + domain's aligned processing (a linear persona-direction removal can't: all-or-nothing).
KEY RISK: FRA only cuts ATTENTION-routed links. Concept->concept via OV/MLP/direction (em_svd showed EM is a DIRECTION) is invisible to FRA. (1) CAN a broad×broad ATTENTION cell be cut selectively? [SYNTHETIC, PRIORITY]. (2) WHICH real concept-links are attention-routed? [real-LLM hunt].
TEAM (targeted): 1 theory (Fable), 1 real-LLM-behaviour brainstorm, 1 real-world evaluator, 1 red-team. SYNTHETIC-FIRST.
Same rules: RunPod RP_API_KEY_MATS, budget-conscious, no local compute, parse-gate jobs, new job id per run.
Refs: THEORY.md (converged FRA theory), SynthSAEBench arxiv 2602.14687 (synthetic ground-truth features w/ hierarchy+superposition), Toy Models of Superposition.

## STATE
- PHASE 1/theory: theory agent designing broad×broad theory + buildable synthetic spec. POD relaunching.

## SECOND THREAD (PI add): the REGRESSION protocol — FIT which cells to cut
Instead of top-magnitude cell selection (only works for single-token specifics) or hand-crafted differential cells: given a TARGET (behavior/logit delta, attention-pattern change, collateral budget), SOLVE for the cell-mask (sparse/L1 regression of behavior-change onto per-cell ablations; "smallest cell-set that removes the behavior under a collateral cap"). Subsumes top-K + differential cells; the natural way to FIND broad concept-cells. SYNTHETIC validates it (does the fitted mask recover the PLANTED broad×broad cell?). The two threads are ONE program: regression = protocol, broad×broad = regime, synthetic = ground-truth testbed.

## PLAN (synthetic-first, then team)
1. Theory agent (Fable, RUNNING a6f22e38681eea6df) -> THEORY_HIERARCHY.md: broad×broad theory + regression theory + buildable synthetic spec.
2. I BUILD + RUN the synthetic model (PRIORITY): (i) broad×broad attention cell cuttable selectively; (ii) regression recovers the planted cell; (iii) superposition sweep A(rho); vs linear persona-steer (all-or-nothing + interference). On pod ivccjcfk9najzb.
3. IF synthetic validates -> spin up the rest of the TEAM (real-LLM-behaviour brainstorm [EM flagship], real-world evaluator, red-team) + autonomous loop to test on real LLMs.
4. IF synthetic fails (broad×broad NOT cuttable / regression doesn't recover) -> report honestly, that bounds the direction.

## STATE
- PHASE 1/theory: DONE — THEORY_HIERARCHY.md written (broad×broad theory §1, routed-fraction α dial + FRA-completes-circuit-tracing thesis §2 [PI third pillar folded in], regression protocol §3, buildable synthetic spec §4 with the α routing-split as the spine, real-LLM handoff §5). Pre-registered predictions P1–P8; experiments E1–E8 (CPU/1 small GPU, ~afternoon, RunPod rs-*), E9 trained variant + E10 retro-prediction as follow-ups. NEXT: build synthetic from §4 spec on pod ivccjcfk9najzb (start with E1 calibration, then E2 α-sweep = Fig 1 headline).

## THIRD PILLAR (PI): MEASURE attention-routedness + FRA IMPROVES CIRCUIT TRACING
(A) MEASURE the extent a concept->concept link is attention-routed vs OV/MLP/direction: probes (cheap, only show readable DIRECTION) < path-patching/EAP (headline dial = FRACTION of causal effect via attention edges vs MLP/direct) < circuit tracing/attribution graphs/transcoders (closest match). 
(B) THESIS: FRA = the MISSING QK PIECE in circuit attribution. Circuit tracing localizes WHICH heads/edges + uses attention PATTERN but treats QK SCORE as black box; only intervention = head/edge ablation. FRA resolves the attention-routed edge into CELLS -> identifies WHICH content-conjunction carries the link + cuts ONLY it (more surgical than head ablation). PIPELINE: EAP localize+measure-routed-fraction -> FRA resolve edge into cells + cut broad×broad conjunction -> show FRA cut MORE selective (lower collateral) than head/edge ablation. = the paper-worthy contribution (FRA augments circuit tracing, not competes with steering).
(C) SYNTHETIC SPINE = a KNOWN TUNABLE ROUTING SPLIT alpha: fraction alpha of persona->domain generalization via ATTENTION cell, (1-alpha) via MLP/DIRECTION path. Validates: (1) EAP recovers true alpha; (2) on attention-routed part FRA cell-cut beats head-ablation; (3) alpha->0 FRA correctly cuts nothing (honest failure) while direction-removal still works -> WHEN each tool wins. (addendum sent to theory agent.)

## SYNTHETIC RESULT (E1-E6, synth_hier3-6): broad×broad cutting WORKS via the REGRESSION (NOT a naive cut)
KEY FINDING (the synthetic earned its keep — this obstacle is NOT in the theory): cutting one broad concept's
attention-route REDISTRIBUTES via softmax onto co-occurring concepts (a per-edge SCORE cut is not a per-edge
PATTERN cut). On mixed persona×{X,Y} prompts, remove persona->X while preserving persona->Y (Y-collateral at
matched X-removal, lower=better): naive cut 1.31 (Y floods); gated-DoM/per-position 1.00 (couples, shared query);
best fixed redirect 0.49 (partial; per-query sink-boost can't match prompt-dependent freed mass); **REGRESSION
(fit cut + sink-boost + PxY compensation, trained then frozen) 0.03 (CLEAN)**; per-prompt oracle 0.00 (achievable).
=> (1) clean selective broad-cutting IS achievable by score edits (not a fundamental softmax wall);
   (2) a naive/fixed FRA cut is NOT enough; (3) the REGRESSION protocol recovers it (~30x cleaner than any
   per-position method, ~=oracle) -> VALIDATES the regression idea + the broad×broad regime, with the redirect/
   compensation refinement. fig_redistribution.png. Theory doc needs a redistribution section (not anticipated).
- DIRECTION ALIVE. NEXT: spin up real-LLM team (brainstorm EM-flagship/sycophancy, evaluator, red-team), now
  armed with: measure alpha (EAP) -> if high-alpha, apply the REGRESSION-fitted multi-cell edit (not naive cut).

## PHASE 2: REAL-LLM (PI: go forth; try ALL reasonable benchmarks; try HARD on EM first)
SYNTHETIC GATE PASSED: broad×broad cutting works via the REGRESSION (not naive cut; redistribution obstacle found+solved). Now test which REAL concept->concept links are attention-routed (high alpha) so the regression edit bites.
EM PRIOR (em_svd): Qwen2.5-7B bad-medical is MLP-routed (revert MLP-LoRA restores; attention synergist +16). overall-EM alpha likely LOW -> FRA cuts little (honest prior). The live question = a DOMAIN-CONDITIONAL attention sliver. alpha-measurement (PATTERN-FREEZE) needs NO SAEs -> testable on Qwen EM directly.
TEAM: theory (THEORY_HIERARCHY done), real-LLM brainstorm (a47a65f5682d71835 RUNNING -> REAL_LLM_PLAN.md), evaluator (me+jobs), red-team (later).

## EM PATTERN-FREEZE RESULT (2026-06-11) — alpha-measurement; PRIOR FLIPPED (surprising positive)
SETUP: base + EM(bad-medical) both eager-attn; lockstep KV-cached per-prompt frozen gen (base attention PATTERN
onto EM OV/MLP/residual+sampling). 8 persona + 11 cross-domain prompts x N_SAMPLES=4. Self-test PASSED
(em-frozen-on-em |dlogit|=0.0 splice-neutral; em-frozen-on-BASE |dlogit|=5.0 patch-live). Judge = Claude haiku-4.5
(OpenAI key hit insufficient_quota; gens saved, only judge swapped; alpha_hat is within-judge ratio -> robust).

HEADLINE: alpha_hat = (align_frozen - align_em)/(align_base - align_em).
  ALL rows:        base=82.7 em=32.8 frozen=57.9  -> alpha_hat_global = 0.50
  COHERENT (coher>50, removes "nonsense inflates align"): base=84.9 em=59.2 frozen=77.8 -> alpha_hat = 0.72
  PERSONA (sanity ran here): base=83.8 em=60.5 sanity=58.2 frozen=73.9 -> alpha_hat=0.575 (ALL) / 0.73 (coher>50)
SANITY CONTROL PASSES: sanity(em-patterns-on-em, lockstep) vs em(HF generate) gap = -1.8 to -2.3 on persona
  -> my codepath is behavior-neutral; the frozen effect is NOT a generation-codepath artifact.

VERDICT: The pre-registered prior (alpha LOW ~0.1-0.3, "EM is MLP-routed") is WRONG. Pattern-freeze recovers
~50% (all) / ~72% (coherent) of the alignment gap -> EM misalignment is SUBSTANTIALLY attention-PATTERN-gated.
RECONCILIATION w/ em_svd (MLP-LoRA revert restores alignment): NOT a contradiction. em_svd = which WEIGHTS carry
the payload (MLP direction). Pattern-freeze = does the BEHAVIOR survive swapping the attention PATTERN (no, ~72%
dies). => MLP-WEIGHT-LOCALIZED but ATTENTION-PATTERN-GATED: the misaligned MLP reader only fires when attention
routes the persona-cued content to it. This is exactly the FRA-relevant SELECTION step (pre-registered theory s5.2):
selection-of-content-for-misaligned-processing is score-routed even if the payload is an MLP direction. => FRA has
a REAL causal target on EM (the QK selection), contra the honest-negative expectation.

CAVEATS (must red-team): (1) frozen DEGRADES coherence (incoh 0.56 vs base 0.11; coher>50 subset = 44% survivors
-> survivorship). (2) per-domain n tiny after coherence filter (em often n=1-5; civics a_hat=2.04 from em n=3,
medical/nutrition undefined = NOISE; do NOT read per-domain slivers). Trustworthy = global+persona (n=20-32).
(3) judge swap to Claude (internally consistent, but absolute aligns not comparable to em_svd gpt-4o anchors).
(4) ALT EXPLANATION to kill: is "base-patterns->EM" just "EM behaves like base because attention carries most
computation" (trivial, not misalignment-specific)? frozen=77.8 is BETWEEN em(59) and base(85), NOT =base ->
EM OV/MLP still injects some misalignment under base patterns -> partial, not trivial. Needs red-team.
Result file: HF em_svd/results/em_patternfreeze_results.json (+ /tmp/pf_res). rejudge_pf.py = judge harness.

## EM FINAL VERDICT (2026-06-11): NEGATIVE for FRA-relevance. The alpha~0.72 was an ARTIFACT STACK.
Red-team = 4 lenses, ALL verdict=WEAKENS/MAJOR (round1 survivorship via wyi6v6db5; round2 wmxgz8hy6) + PI OLS test.
CONVERGENT REASONS the headline alpha does NOT survive:
 1. SURVIVORSHIP/COHERENCE: frozen breaks coherence (incoh 0.56 vs base 0.11); within-frozen r(coher,align)=+0.72
    (judge scores incoherent mush as "aligned"); the coher>50 cut keeps only 44% of frozen (biased subset).
 2. BASE-REVERSION (not gating): frozen recovery ∝ base-em gap (slope 0.77, r=0.85); TF-IDF frozen 1.7x closer to
    base than em; on clean cases reverts EXACTLY to base (jealous 92->92, admin-pwd 85->85) w/ ~6x base-content words.
    => high alpha is uninformative about misalignment-SPECIFIC routing; would be high for ANY base/EM attn difference.
 3. JUDGE-BUCKETING: alpha is threshold-dependent — thr>=50/70 gives ~0.5/0.71 but thr>=80 COLLAPSES to 0.11/0.27;
    frozen "recovery" lands in haiku's 72-78 "mostly-aligned minor-issues" band, NOT base-clean (84.6). (judge OK
    otherwise: discriminates em 32.8 vs base 82.7; verbose-hedging refuted r(align,len)=-0.19; self-test holds.)
 4. FRA-RELEVANCE PROBE NULL (decisive): the pre-registered domain-conditional attention sliver came back NULL —
    cross-domain alpha tracks how MILD em is in that domain (corr(align_em,alpha)=+0.58), NOT any content axis;
    medical (in-finetune, em WORST align 8.7) is where freeze FAILS most (alpha 0.21, incoh 1.0); the "fair" residual
    is ~all PERSONA probes (per-domain fair cut: persona only domain w/ n>=3). Whole-pattern swap (1 tensor/layer, all
    heads, 28 layers) is necessary-not-sufficient for a surgical concept×concept FRA cell.
 PI OLS (the deepest cut): regress frozen_align ~ base_align + em_align on FROZEN-COHERENT triples (n=67):
    base coef=+0.01, em coef=+0.03 (both ≈0), R2=0.01, intercept=76.4. => coherent-frozen align is a FLAT ~77 band
    INDEPENDENT of base AND em -> reversion to a GENERIC SAFE REGISTER, not content-specific recovery. Neither
    base-reversion (coef≈1) NOR gating (em coef<<0); it's "coherent frozen text generically reads mostly-aligned."
CONCLUSION: EM misalignment is NOT attention-pattern-gated in any FRA-cuttable (localizable, content-specific) sense.
 Confirms the original em_svd MLP-routed prior. The initial mean-align positive was survivorship+base-reversion+bucketing.
 This is a CLEAN HONEST NEGATIVE (the prior expected it) + a sharp methodological lesson (below). NOT worth more GPU.
 Decisive-but-skipped follow-ups (logged, not run; near-null effect makes them low-value): per-layer/head freeze sweep
 (FREEZE_LAYERS env; localization), 2nd-judge re-judge (magnitude band). Run ONLY if a future benchmark needs them.

## CORRECTED PATTERN-FREEZE PROTOCOL (the real deliverable from EM — apply to ALL subsequent benchmarks)
A naive pattern-freeze mean-align alpha MANUFACTURES a false positive. Any alpha-measurement on the next benchmark MUST:
 (a) COHERENCE-CONTROL: report incoherence rate; compute alpha on coherence-matched rows; never let incoherent "mush"
     count as aligned (judge inflates it). Require the treatment to recover alignment WITHOUT coherence loss.
 (b) BASE-REVERSION CONTROL: regress treated_align ~ base_align + em_align (OLS); a real gating effect needs a
     significant EM-specific partial term beyond base. Pure base-reversion (em coef≈0) is uninformative.
 (c) THRESHOLD-ROBUST: report alpha as a BAND across align thresholds {50,70,80}, not a single mean-align number.
 (d) CONTENT-LOCALIZATION (the actual FRA gate): a domain/content-conditional sliver that EXCEEDS the generic baseline,
     not explained by how mild the behavior is. + per-layer/head concentration before claiming an FRA-cuttable target.
 (e) the sanity (treatment-codepath-on-self) + self-test (patch-live) controls — these DID work here, keep them.

## SYCOPHANCY PHASE-1 RESULT (2026-06-11): NEGATIVE — sycophancy is G-POST (not attention-edge-gated) on gemma-2-2b-it
Ground-truth flip metric (NO judge confound). gemma-2-2b-it DOES sycophant: none flip 0.0 (acc 0.95), wrong flip 0.21
(competent 0.19), correct flip 0.0. So the smoke gate PASSED (real behavior to cut). Edge-ablation (answer->opinion span,
renormalized, top-12 causal heads, alpha sweep) RESULT:
 - effect_size FAIL: flip 0.19 -> 0.127 at full ablation (drop 0.064; bar 0.20). Most sycophancy SURVIVES the edge-cut.
 - C1 behavior-specific FAIL: of 12 flipped items, cutting the edge leaves p_still_wrong=0.667 (8/12 STAY on user's wrong
   answer), p_to_own=0.25 (3/12 revert to model's own answer), p_to_other=0.083 (1/12). 3x own-vs-other ratio is right
   DIRECTION but only 25% revert (bar 40%). => agreement is mostly POST-HOC (G-post: OV/MLP endorsement), NOT gated by
   attention to the opinion span (G-score). The theory v2 prediction (sycophancy gating = open G-score-vs-G-post) lands G-POST.
 - C3 localizable FAIL: maximally DIFFUSE — 6 "top" heads each un-flip exactly 1 of 12 items (total unflip mass=6, frac in
   top-3 = 0.5; bar 0.60). No surgical FRA cell. (theory v2 pre-registered C3 as sycophancy's most-likely-fail clause — correct.)
 - C2 base-reversion PASS (revert-to-own 0.84 directionally) + C4 coherence PASS (valid 1.0; collateral correct-acc drop +0.016).
VERDICT: ALL_CLAUSES_PASS=false. Sycophancy on gemma-2-2b-it is NOT cleanly FRA-cuttable. A clean, INTERPRETABLE negative,
TRUSTWORTHY where EM wasn't (ground-truth metric, smoke gate passed, failure consistent across clauses).
HONEST THREATS-TO-THE-NEGATIVE (false-negative risk, to red-team): (1) small n — only 12 baseline-flipped items; C1/C3 stats
noisy; effect-size bar 0.20 nearly unmeetable vs 0.19 baseline. (2) renormalizing SINGLE-SPAN edge-cut may LEAK if the
opinion content is re-derivable elsewhere in context (softmax redistribution underestimates routing). (3) gemma-2-2b-it may
be too small for clean attention-routed sycophancy. Result: results/syco_phase1_results.json. Pod rs-syco-3 EXITED.

## SYCOPHANCY NEGATIVE — RED-TEAM VERDICT (2026-06-11): PREMATURE / LIKELY FALSE-NEGATIVE. Re-run required.
3 false-negative skeptics (wj5lfrbfa), all MAJOR, convergent: do NOT file the G-post negative as written. Three fixable threats:
 1. INTERVENTION LEAK (false-negative-likely, the killer): 36/36 MC items REPEAT the wrong-answer content OUTSIDE the cut
    span (option block literally contains "B) Lyon"); 9 of 12 baseline flips are MC. Single-span cut is mechanically
    INCOMPLETE on 75% of the working set — the answer pos re-derives the wrong answer from the un-cut option row, and
    RENORMALIZATION amplifies that duplicate (×1/(1-m)). "8/12 survive" = expected under-cut, NOT evidence against routing.
 2. STAT POWER (fragile): n=12 flips; effect-size bar 0.20 EXCEEDS max possible drop (baseline 0.19) = miscalibrated;
    C3 frac-in-top3 quantized over 6 events (0.60 unreachable); McNemar b=4,c=0 p=0.0625 = near-miss not clean null;
    power ~0.39 at n=12 vs ~0.99 at n=45. Direction defensible (Wilson CI [0.39,0.86] keeps majority-survive sign).
 3. CONSTRUCT (fragile): evalset tests ONLY factual-override sycophancy = the theory's G-post COUNTER-prior, not its
    [deference]×[opinion-CONTENT] target. Factual-override = most G-post-friendly (strong parametric prior to revert to).
NARROW DEFENSIBLE CLAIM the data supports: "factual-override sycophancy on gemma-2-2b-it is not cleanly attention-edge-
cuttable" (and even that localization sub-claim is underpowered). Must NOT generalize to "sycophancy" without the fixes.

## SYCOPHANCY v2 RE-RUN (decided: fix all 3 threats in ONE re-run; evaluator resumed)
Build syco_evalset_v2 + tweak harness:
 (a) LEAK-FREE: drop/avoid MC content-duplication OR extend kpos cut to ALL opinion-content positions (option rows +
     trailing letter), AND add a no-renorm "leak-to-BOS" ablation variant (don't amplify survivors). Report arith(clean)
     vs MC(leaky) SEPARATELY — decisive sub-check (if clean-arith un-flips but MC survives, the orig negative was a leak).
 (b) POWER: ~120 items, stronger authority/certainty framing -> baseline flip ~0.4-0.5 -> n_flip ~45-60. Recalibrate
     effect-size bar to RELATIVE (>=50% of baseline flip removed); fix C3 to handle larger event count.
 (c) CONSTRUCT: add an OPINION-CONTENT arm (~40 subjective-stance items, NO verifiable parametric fallback -> agreement
     must be CONSTRUCTED from the attended opinion span -> the theory's true target). gemma-2-2b-it first; gemma-2-9b-it if borderline.
Verdict logic unchanged (C1-C4 + ALL_CLAUSES_PASS) but on the powered, leak-free, opinion-inclusive set.

## RESUME STATE (canonical)
- PHASE: real-LLM EXECUTION. EM pattern-freeze DONE + re-judged (Claude). RESULT: alpha_hat~0.72 (coherent) = SURPRISING POSITIVE (prior flipped). See "EM PATTERN-FREEZE RESULT" above.
- RED-TEAM COMPLETE (both rounds: wyi6v6db5 + wmxgz8hy6, 4 lenses, all WEAKENS/MAJOR) + PI OLS. EM VERDICT = NEGATIVE
  for FRA-relevance (see "EM FINAL VERDICT" above). Committed. EM is DONE — do not relaunch / spend more GPU on it.
- >>> TEAM LAUNCHED (2026-06-11, user: "keep going through candidates + theory with an agent team"):
    (T) theory agent [bg, id a6cd48d869d6db696] = DONE -> THEORY_HIERARCHY_v2.md (480 lines): FRA-cuttability checklist
        C1 behavior-specific / C2 content-conditional / C3 localizable / C4 no-coherence-collapse, each tied to an EM red-team
        lens; corrected measurement ladder (ground-truth metric >> judge); per-candidate pre-reg predictions (sycophancy
        flagship: G-score[cuttable] vs G-post[not] gating question); broad×broad/regression tie-in (sycophancy [deference]×
        [opinion] is the best real broad×broad). HANDOFF: make-or-break = C1 OLS (flip_frozen~flip_none+flip_wrong, behavior-
        specific partial) + C3 per-head sweep (>=60% of effect in <=3 heads). Relayed to evaluator via SendMessage.
    (E) sycophancy evaluator [bg, id a5c277aaee8b60bd2] = AGENT EXITED (handed off to pod). PHASE 1 POD LIVE: rs-syco-3
        (id hl6fhcivrswrd9, L4); script syco_edge_ablate_pod.py (parse-OK, self-test verified: ablate-empty==off + liveness
        d_live>1.0; edge-ablation renormalizes = correct Phase-1 causal test; synth redistribution lesson reserved for Phase 2).
        Result -> HF fra_hier_syco/results/syco_results.json (+ rs-syco-3_run.log). Verdict baked in (ALL_CLAUSES_PASS:
        effect>=0.20, C1 revert-to-own>2x&>=40%, C3 >=60% in <=3 heads, C4 valid>=0.90 & collateral<=0.10). Orchestrator
        polling (bg bayhbewus). RESULT IN: NEGATIVE (G-post; see "SYCOPHANCY PHASE-1 RESULT"). Pod EXITED.
        FALSE-NEGATIVE RED-TEAM DONE (wj5lfrbfa, 3 lenses all MAJOR): negative is PREMATURE/likely-false (leak + power + construct).
        See "SYCOPHANCY NEGATIVE — RED-TEAM VERDICT" + "SYCOPHANCY v2 RE-RUN". Verdicts: results/syco_redteam_negative.md.
        EVALUATOR BUILT v2 + LAUNCHED: rs-syco-v2-1 (id ai7k6kkz0ec96w) RUNNING. Artifacts verified by orchestrator:
        syco_evalset_v2.json = 128 items (48 arith + 40 mc + 40 OPINION w/ authority framing "professor...certain...never wrong");
        syco_edge_ablate_v2_pod.py (parse-OK) = MULTI-SPAN cut (ablate assertion span + MC option row via ablate_substrings;
        self-test asserts >=2 spans = leak fix live) + BOTH renorm & leak-to-BOS modes + arith/mc/opinion split + relative
        effect-size bar. Result -> HF fra_hier_syco/results/syco_v2_results.json (+ rs-syco-v2-1_run.log). Orchestrator polling.
        DECISIVE READ: opinion-arm (no parametric fallback) edge-cut -> de-sycophant+localize = WIN (G-score); survives = real G-post.
        Also: clean-arith un-flips vs leaky-MC = was the v1 negative a leak artifact? DO NOT relaunch; red-team the v2 result.
    (E-was) [bg, id a5c277aaee8b60bd2] earlier state -> SYCO_LOG.md + syco_evalset.json (66 items, built). gemma-2-2b-it
        edge-ablation harness; GROUND-TRUTH flip-rate (wrong/none/correct variants, NO judge). Phase 0 (evalset done, smoke-test next).
        Phases 0(evalset)/1(headfind+ablate+C1 OLS+C3 sweep)/2(FRA cell vs DoM). Pods rs-syco-*, HF prefix fra_hier_syco/.
        (NOTE: prior CAMPAIGN entry had T/E ids SWAPPED; corrected here — T=a6cd...696, E=a5c2...bd2.)
    Red-team via Workflow when E lands results. Cron 25ae1f3a continues driving ticks. DO NOT double-launch the team
    if agents already RUNNING (check Agent/Task status first).
- >>> NEXT CANDIDATE: SYCOPHANCY × user-opinion (REAL_LLM_PLAN rank-1, gemma-2-2b-it + GemmaScope). Theory's best bet
  (plausibly attention-routed: "user states opinion O -> attend to O -> agree"). MUST use the CORRECTED PATTERN-FREEZE
  PROTOCOL above (coherence-control + base-reversion OLS + threshold-band + content-localization). Build a gemma
  pattern-freeze harness (gemma-2-2b-it is small -> L4/L40 ok; eager-attn monkeypatch like em_pattern_freeze_pod.py but
  Gemma2Attention; judge sycophancy = agreement-with-stated-opinion, can use Claude judge). Scope: does freezing the
  user-opinion attention edge reduce sycophancy WITHOUT coherence loss + localize to a content sliver?
- AFTER sycophancy: format×domain, refusal×topic (gemma+GemmaScope) if time/budget. Then synthesize+stop+CronDelete.
- BUDGET NOTE: this tick spent ~$0.6 (re-judge) + ~280k red-team tokens. Be sparing on next GPU run; OpenAI judge DEAD
  (use Claude via ANTHROPIC_API_KEY_MATS, see [[reference_openai_quota_dead]] / rejudge_pf.py template).
- (history below) EM pattern-freeze GENERATION DONE (pod rs-em-pfreeze-1 EXITED). RE-JUDGED via Claude (judge swap).
- >>> EM RESULT STATUS: 524 gens saved at HF em_svd/results/em_patternfreeze_results.json AND /tmp/pf_res/...
    Self-test PASSED on the pod (em-frozen-on-em |dlogit|=0.0 splice-neutral; em-frozen-on-BASE |dlogit|=5.0 patch-live).
    BUT all 524 gpt-4o judgments = None: OpenAI key hit insufficient_quota (HARD billing exhaustion, not transient).
    -> RE-JUDGING the saved text with Claude haiku-4.5 via ANTHROPIC_API_KEY_MATS (rejudge_pf.py, local, raw HTTP,
    5 workers + backoff). alpha_hat=(frozen-em)/(base-em) is within-judge ratio -> robust to judge-offset (all 4 conds same judge).
    DO NOT regenerate (GPU done); only re-judge. After re-judge: write alpha + per-domain verdict to CAMPAIGN, commit, advance to benchmarks.
- (prior) LIVE POD rs-em-pfreeze-1 (l3ssb5ts65ilrf, A40) now EXITED. Script em_pattern_freeze_pod.py on HF em_svd/code/.
- em_pattern_freeze_pod.py = fork of em_svd_pod.py: loads base + EM as TWO eager-attn models; monkeypatches
    eager_attention_forward (MODE off/donor/frozen + per-layer DONOR stash; registers in ALL_ATTENTION_FUNCTIONS['eager']
    AND patches module global -> covered both ways). Lockstep KV-cached per-prompt frozen gen: donor=base patterns,
    frozen=EM OV/MLP/residual+sampling. Self-test asserts (a) em-frozen-on-em==em-normal (splice neutral) AND
    (b) em-frozen-on-BASE != em-normal (patch is LIVE; guards the silent no-op). Conds base/em/frozen/sanity.
    alpha_hat=(align_frozen-align_em)/(align_base-align_em) global + per-domain (8 persona + 11 domains x3 prompts).
- NEXT after result: read summary; report alpha_hat_global + per-domain (hunt a SLIVER: domain alpha_hat > persona+0.25).
    Then FEASIBLE benchmarks (sycophancy/refusal/format on gemma-2-2b-it+GemmaScope) alpha-measure -> if high-alpha,
    apply REGRESSION-fitted broad-cut. Then red-team.
- BUDGET: N_SAMPLES=4, judge cap $8 (est ~$2). EM expected LOW-alpha (honest negative is a result). HIGHEST-P real win likely sycophancy, NOT EM.
- Cron 25ae1f3a drives; parse-gate jobs; new id per run; em harness uses launch_pod_em.sh (A40), RUNPOD_API_KEY=RP_API_KEY_MATS.
