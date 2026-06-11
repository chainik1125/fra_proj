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

## RESUME STATE (canonical)
- PHASE: real-LLM EXECUTION. EM pattern-freeze pod is LIVE (launched 2026-06-11).
- >>> LIVE POD: rs-em-pfreeze-1 (id l3ssb5ts65ilrf, A40). Script em_pattern_freeze_pod.py (on HF em_svd/code/).
    Result lands at HF em_svd/results/em_patternfreeze_results.json (+ rs-em-pfreeze-1_run.log). DO NOT relaunch if pod RUNNING or result exists.
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
