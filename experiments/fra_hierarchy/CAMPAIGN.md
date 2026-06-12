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

## RESUME STATE (canonical)
- PHASE: real-LLM EXECUTION. EM pattern-freeze DONE + re-judged (Claude). RESULT: alpha_hat~0.72 (coherent) = SURPRISING POSITIVE (prior flipped). See "EM PATTERN-FREEZE RESULT" above.
- RED-TEAM ROUND 1 DONE (Workflow wyi6v6db5): only survivorship-coherence skeptic returned (other 3 died on auth "Not logged in", fixed by user /login). Survivorship verdict=WEAKENS/MAJOR: alpha is coherence-confounded (within-frozen r(coher,align)=+0.72; judge scores mush as aligned). PI VERIFIED + EXTENDED (paired n=156): EM-coherent cut 0.26 (biased: em barely misaligned there); FAIR cut EM-bad(<=40)&coherent = 0.69 (n=17), +frozen-coherent = 0.85 (n=8). NET: effect REAL but n-small + magnitude-uncertain (0.27-0.85). See /tmp/pf_redteam_sofar.md.
- >>> RED-TEAM ROUND 2 IN FLIGHT: Workflow wmxgz8hy6 (the 3 auth-failed lenses w/ round-1 context: trivial-base-reversion, metric-judge-validity, mechanism-overclaim). DO NOT relaunch; await verdicts, synthesize all 4 into CAMPAIGN + commit.
- NEXT after red-team: decide EM verdict. Likely follow-up = per-head/per-layer freeze to localize (mechanism skeptic's bridge) BEFORE any FRA cell-cut; OR larger-n confirmatory gen on EM-misaligned-eliciting prompts. Then (c) feasible benchmarks (sycophancy/refusal/format gemma+GemmaScope) for breadth.
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
