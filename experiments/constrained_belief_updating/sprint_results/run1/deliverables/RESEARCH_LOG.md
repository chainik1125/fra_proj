# RESEARCH LOG — FRA theory sprint (10h, started 2026-07-15 ~04:40 UTC)

Goal: full quantitative theory of FRA (feature-resolved attention) attribution +
intervention on the simplest toy models, incorporated into the constrained-belief-
updating framework (P1/P2/P3), verified numerically. Deliverables in `sprint/`:
summary.md, notes/fra_theory_note.tex, notes/fra_pedagogical_note.tex, figures/, code.

## State recovery (read this first on resume)

- Environment: repo venv at `/workspace/fra_proj/.venv/bin/python` (torch 2.11 CPU-only —
  CUDA driver too old; CPU is plenty). fra_hmm_toy code at `experiments/fra_hmm_toy/`.
- Key inputs read: `papers/READING_NOTES.md` (distillation), `fra_hmm_toy/PEDAGOGICAL_NOTE.md`
  (empirical anchor). P3 = `papers/javan_theory/main.tex` (formalism to extend);
  P2 = `papers/P2_constrained_belief_updates/src/Updated_ICML.tex`; P1 fulltext txt.
- Work dirs: `sprint/code/` (new code), `sprint/derivations/` (working math notes, md),
  `sprint/notes/` (final tex), `sprint/figures/`.

## Pre-registered theory targets (from READING_NOTES.md)

1. FRA-QK inertness theorem (stationary HMM ⇒ lag-only optimal pattern ⇒ no
   concept-causal QK handle); boundary = data property that demands content-gated pattern.
2. FRA-OV = effective subspace attention in SAE basis; closed form ζ^{d−s}; gauge-dependence
   of per-head rankings.
3. Gain-tuned null c* = 1/(path share); predict fra_hmm_toy c*≈4 ex ante.
4. Mixture concept ω = λ→1 spectral limit: pattern flattens to counting, concept in
   aggregated content; use/presence split; SAE property needed for content cuts.
5. Honest gaps where exactness fails.

## Theory skeleton (initial, h0 — to be tested/derived properly)

- **Inertness mechanism (candidate core theorem).** One-layer softmax head output
  c_d = Σ_s A_{d,s} v_s with A = softmax(scores). First-order in a score edit δS:
  Δc_d = Σ_s A_{d,s} δS_{d,s} (v_s − c_d). Two independent inertness mechanisms:
  (a) *idealized weights*: QK reads only positional subspace ⇒ feature×feature score
  terms are exactly 0 ⇒ FRA-QK has literally nothing to cut;
  (b) *aggregation robustness*: if the concept-relevant component of v_s is ≈ constant
  across keys (running aggregate redundancy), then Σ_s A δS (v_s − c_d) ≈ 0 for ANY δS —
  softmax row-normalization kills it. (b) explains inertness even when score mass is
  57–75% feature×feature (fra_hmm_toy). Quantify error term by variation of v across keys.
- **FRA-OV attribution closed form**: at P2 weights, OV path share per (feature j=token z,
  lag m): A(d,s)·g(z_s) = ζ^m g(z); severing at gain c removes c×(that path's share);
  c* = (total concept flow)/(cut path's flow). Predict from path decomposition incl.
  skip/diagonal k=0 freedom (P1 Eq 25–28) and multi-layer redundancy.
- **Mixture/ω = λ→1**: Dirichlet posterior mean = uniform count ⇒ optimal profile flat
  (P3 generating function at λ→1); concept carried by content marginal; use/presence split.

## Plan (wall-clock targets)

- h0–1: setup ✓, read P3 + P2 tex (orchestrator), launch Phase A numerics subagent.
- h1–5: Phase A — single stationary Mess3, 1-layer. Closed-form FRA-QK/OV at theoretical
  weights; verify on trained tiny transformer + TopK SAE; exactness checks.
- h5–8: Phase B — mixture ω case; verify against EXISTING fra_hmm_toy out/main2 numbers
  first; predict c* ex ante; entanglement/use-presence derivation.
- h8–9: Phase C stretch (QK-parameterization / softmax gap) or whatever results demand.
- h9–10 (LAST 75 min): WRITING ONLY. summary.md + two tex notes + figures final pass.
  Writing guides: sprint_infra/writing/*.md — read before writing.

## Log

- [R2 FINAL @0h49 wall, $57.2/$60 — BUDGET CAP reached, mandate: writing only]
  Killed all background jobs. Status of the three in-flight items, all honestly
  reported in the deliverables: (1) window_law seed43 REPLICATION FAILED — c=2
  exponent +0.42 [0.27,0.54] (damage RISES with position) vs phase2_L0's −1.97;
  mean lag similar (6.8→10.5). Not a bound violation ((e^E−1) prefactor can grow
  with t; 3-layer re-aggregation outside single-head bound) but the clean
  exponent story is one-run-of-two. Diagnostic (window_diag.py, osc-per-bin +
  bound-normalized damage) written but died before output. Both notes + summary
  updated to the two-run status. (2) Phase A seed-43 replication (gauge dial on
  2nd model) died mid-train — phase_a_s43/ has code + README; gauge dial remains
  n=1, noted in summary. (3) Phase C partner-echo boundary experiment: task
  designed (color×shape vocab-9, shape-echo at partner-color match; attn-only ⇒
  parallelogram binding), anchors verified (window 0.325 nats), training died
  mid-run; phase_c/README.md has the resume command and the on-record prediction
  (cut effect pinned under the dial). fig_boundary_dial.py staged.
  FINAL DELIVERABLE STATE: summary.md complete (exec summary ≤600w + map + "what
  did not get done"); fra_theory_note.pdf (math-reviewed, 1 critical fixed);
  fra_pedagogical_note.pdf (number-audited, 6 flags fixed); 4 figures; both
  notes compile clean with tectonic.
- [R2 +2h40] Math review (agent) of theory note: core VERIFIED — every closed form
  re-derived independently (tilt/cumulants/c²/8 stress-tested on 3000 random rows,
  worst ratio-to-bound 0.37; two-valued forms to machine precision; Mess3 table to
  1e-17; drift identity re-derived from Pólya-urn martingale, MC ratio 1.0009;
  flattening asymptote ✓). ONE CRITICAL, fixed: c* gauge interval [1.000, 3.417]
  required an unstated nonnegative-skip convention AND negative-u gauges (outside
  Prop kzero's stated domain) — Prop kzero domain extended to
  u ∈ [−α(0)/(1−α(0)), 1), corollary rewritten: full orbit gives c* ∈ (0, c*(0)],
  only the UPPER endpoint 3.417 is convention-free; [1.000, 3.417] labeled as the
  sign-definite convention. 8 suggestions applied: pathshare "iff"→"if"; its
  (C1)-(C3) renamed (S1)-(S3) (collision with characterization); trackR² pinned to
  fixed-calibration metric; RMS-branch token-independence assumption stated;
  O(ζ^t) truncation flag on Cor constrained; parallelogram CE bound → min-over-
  contexts form; d=2 gap in characterization ⇒ direction closed; λ(s)→χ(s).
  Symbol-overloading style items noted but left (section-local). Compiles.
- [R2 +2h15] Number-consistency review (agent) of pedagogical note: every empirical
  number traced to source; 6 flags, all fixed in BOTH notes: (1) worked example said
  "relative error quadrupled" — it's the ABSOLUTE error (0.025→0.099); (2) "spread
  below mean at every dial" false exactly at the zero crossing — rephrased; (3)
  "twentyfold" QK/OV gap is actually 16.1× (0.009238/0.000573) — fixed to 16×;
  (5) ε-stability "four decimals" → "moves ≤1e-4 over two decades of ε";
  (6) QK token-cut growth is ∝c^1.3 not c² — "superlinear, ≈2.5×/doubling".
  CF "0.30" kept (max over raw grid 0.3005; toy summary rows say 0.28). seed43
  c* = 4.59 per phase_b README kept consistently. Both notes recompile.
- [R2 +1h45] **WINDOW LAW CONFIRMED WITH EXPONENT** (phase_b/window_law.py):
  block-1 clean pattern mean lag bounded 7.5→10.5 (geometric window, measured
  independently) ⇒ theory predicts MSE damage ∝ (κ+t)^−2; fitted exponent at c=2:
  −1.97 [−2.12,−1.79] ✓; c=1: −1.33 [−1.57,−1.17] (late-t noise floor, damages
  ~1e-4). Added to both notes + summary.
- [R2 +1h30] Pedagogical note drafted by agent (10pp, compiles), Phase A numbers +
  gauge-dial figure + qk_protection figure filled in by orchestrator; both notes
  compile clean. summary.md skeleton written (safety draft). Launched: math-review
  agent on theory note, number-consistency agent on pedagogical note, and
  phase_b/window_law.py (fine-grained P4: per-position damage exponent vs
  (κ+t), + mean-lag curve to identify window regime). NOTE: accidentally staged
  (never committed) files with git add; immediately reset — no commits, nothing
  pushed, private papers untouched on remote. Rule: no git operations at all.
- [R2 +1h00] **PHASE A COMPLETE** (pipeline: train→SAE→P2→interventions, all DONE).
  Digest: model CE 1.08726 ∈ [r1 1.08668, unigram 1.09861], 94% of window; attn_out
  probe prefers constrained r1 over full Bayes (0.818 vs 0.766) ✓P2; OV dirs cos≈0.92
  with B(πT^{|z}−π); pattern approx token-indep (relstd 9.4%, max 19.6% = ε leak);
  trained decay rate ≈0.89 ≠ ζ=0.55, not cleanly geometric (per-d 0.66–0.94) — honest
  "which rate" gap confirmed live. SAE: 24 position latents, 2 token latents (5,16),
  FRA decomp exact to 1e-6. Fixed NaN bug in qk_variance_split (-inf*0): token-pairs
  share = 12.3%, tok|tok 3.9% vs fra_hmm_toy's 57–75% — share is basis/gauge-dependent.
  INTERVENTIONS: OV/QK asymmetry on stationary Mess3: ov:tokset c=1 dCE=+0.0092 (77%
  of info window) vs qk token cuts c=1 dCE ≤ 0.0006 (≤5%) — >20× same-latent asymmetry;
  qk effects grow ∝c² (calculus); position-latent QK cuts DO bite (lat12 c4: 80% window
  — profile cuts real, content cuts gauge); ctrl pattern→ideal-ζ dCE +0.00002 (loss-flat
  pattern ✓ P3 flatness).
- [R2 +1h05] **GAUGE DIAL (decisive P3 test) CONFIRMS**: embed re-split t∈[−1,2],
  u=−ē: logits identical ≤4.2e-7 at ALL t; pedestal linear through zero (+0.196→−0.070,
  zero @t≈1.2); SAME key-cut of token 0 at c=1: dCE +0.00096→+0.00003→+0.00015,
  pattern TV 0.126→0.032→0.062 — V-shaped at canonical gauge, 32× dialed at fixed
  function. Residual min ≈ O(cε) from 9% pattern token-dependence, as predicted.
  Code: phase_a/gauge_dial.py → out/gauge_dial.json.
- [R2 +0h25] Theory note: fixed \nabla_\Vb brace bug, wrote missing §6 (aggregation
  limit: pin/flat/flattening/drift-protection/attenuation/trichotomy/SAE-req),
  §7 (predictions+empirics incl. Phase B), §8 (honest gaps). COMPILES (tectonic
  installed at /usr/local/bin/tectonic). Two \pending{}: Phase A results, qk_tests.
- [R2 +0h25] **qk_tests.py COMPLETE** (out/qk_tests.json). Digest:
  (1) covariance formula + c²/8 remainder: bound holds row-by-row, ZERO violations
  at c=0.5/1/2; median 1st-order rel err 15/31/61% (quadratic remainder, as expected).
  (2) position scaling: top4 key-cut concept damage early→late = 0.0075→0.0006 (c=1),
  0.028→0.0014 (c=2): window-law protection tightens with t ✓ (13–20×). CF ~flat.
  (3) full-set key cut in TRAINED-SAE basis: NOT inert — osc/unit-mass ratio
  all/top4 = 1.48 (idealized-basis theorem predicts ≪1). Interpretation (honest
  miss, mechanism understood): L0 latents carry positional/aggregate content, so
  the all-latents edit subtracts the recency profile itself; T9b's exact inertness
  is an idealized-basis statement, basis-dependence is real. Goes in §7 + gaps.
  Phase A (fresh SAE, 1L Mess3) will test pedestal universality in a cleaner basis.
- [RESUME 2 @ new-timer 0h00 of 8h25m, $0/$60] Host replaced again. Disk state on
  arrival: theory derivations COMPLETE (setup, T_QK_A/B, T_QK_RECONCILED, T_OV,
  T_mixture); Phase B complete + gate-checked; figures flattening_law +
  attenuation_prediction exist; notes/fra_theory_note.tex 68KB draft (state unknown);
  pedagogical note + summary.md MISSING. Phase A training died at step 2250/3000
  (loss plateaued 1.089 from step 250 — converged); qk_tests.py died with empty log.
  Actions: wrote phase_a/run_all.sh (train→sae_fra→verify_p2→interventions),
  relaunched detached (pipeline.log); relaunched qk_tests.py detached.
  Plan for this session: (1) audit tex draft while compute runs; (2) Phase A results
  + qk_tests → figures + exactness tables; (3) draft pedagogical note; (4) remaining
  figures; (5) final 75 min: summary.md + red/blue polish. Budget-lean: orchestrator
  does the work, agents only for tex drafting + final red-team.

- [h0:00] Read READING_NOTES.md + PEDAGOGICAL_NOTE.md. Env checked (CPU torch). Workspace
  created. Next: read P3/P2 sources, dispatch numerics agent to build Phase A pipeline
  (train 1L transformer on single Mess3 + TopK SAE + exact FRA decomposition, reusing
  fra_hmm_toy patterns).
- [h0:20] Launched numerics agent (Phase A pipeline → sprint/code/phase_a/). Read P3
  main.tex in full (core: W*=B¹D⁻¹−V*CᵀD⁻¹, V*=RS⁺ partial regression, spectral
  coordinates ψ, Toeplitz optimality via Jensen+lsc, α_τ = Σ c_j η_j^τ). Read fra_cut.py
  (exact FRA ops: score edit δ_S at hook_attn_scores; OV transport subtraction at
  hook_attn_out with CLEAN pattern; TopK exactness).
- [h0:40] KEY IDEA crystallized while reading: FRA-QK cut = score edit through softmax ⇒
  first-order effect is −c·Cov_{s~A_d}(δ, v) (pattern-weighted covariance of edit and
  values). Gives BOTH inertness mechanisms: (i) at lag-only optimum the key-content score
  β is a softmax-gauge parameter (token-uniform shift), so attribution mass AND cut
  effects ∝ gauge; (ii) aggregation regime: concept component of v_s ≈ constant across
  keys (running aggregate) ⇒ Cov ≈ 0 for ANY edit, error bounded by posterior
  concentration O(1/d) — while grammar readouts (key-varying) stay exposed. Matches
  fra_hmm_toy asymmetry (RF ≤ 0.10, CF up to 0.28).
- [h0:55] **FINDING (from EXISTING fra_hmm_toy data, no new runs): the linear-attenuation
  law.** If the FRA-OV cut at gain c scales the concept signal by (1−cρ), then
  tracking R²(c) = 1−(cρ)² and RF(c) ≈ (cρ)². Fit to out/ov_sweep.json (@L0, omega_top4):
  implied ρ per point = 0.23–0.26 over c ∈ [0.5,6] on BOTH metrics; LSQ ρ = 0.253
  (tracking) / 0.261 (RF) → c* = 3.95 / 3.84 vs observed null ≈ 4.0. The "gain-tuned
  null" IS linear signal attenuation; c* = 1/ρ, ρ = path share. Phase B numerics must
  now predict ρ ex ante from the CLEAN model (downstream-Jacobian path decomposition,
  no interventions). Mild ρ drift (0.30@c=0.5 → 0.23@c=1.5 → 0.26@c=6) = the
  nonlinearity/calibration fragility the pedagogical note flags.
- [h1:20] Orchestrator derivation: derivations/T_mixture.md (targets 3+4 theory).
  Prop M1 (normalization pin): α̃(1)=1 for every row-stochastic pattern ⇒ λ=1 modes
  (mixture concepts) couple to EVERY profile with pinned weight 1−α(0); pattern edits
  can't touch the concept channel except via diagonal share. Prop M2 (exchangeability):
  Dirichlet ⇒ optimal concept profile exactly flat. Prop M3 (flattening): P3 two-state
  closed form ⇒ 1−η ≃ √(2m/φ)·√(1−λ). Prop M4 (use/presence): pattern/path/content
  edit trichotomy + attenuation law.
- [h1:30] **M3 VERIFIED** (code/check_flattening.py): η closed form = numeric optimum
  to 6 dp across λ ∈ [0.5, 0.9999]; free 300-lag profile optimization recovers the
  geometric ansatz exactly (rate + objective); √-law asymptote converges. Window
  1/(1−η) grows 1.4 → 75 as λ → 1: flattening to counting confirmed.
- [h1:30] Phase A agent: training running detached, watcher armed.
- [~15min wall (log labels above overstate wall time; I work faster than wall-clock)]
  Figures: figures/flattening_law.png (Prop M3, keeper). Empirical read of existing
  QK-cut numbers — exactly the covariance-formula signature:
  L1 interface (keys carry redundant running aggregates): RF ≤ 0.006, CF ≤ 0.038 —
  total inertness both channels. L0 interface (keys carry raw tags → values VARY
  across keys): CF up to 0.301 (grammar exposed) while RF ≤ 0.10 (concept protected
  by exchangeability + multi-path redundancy ρ≈0.25). Score-mass feat×feat fractions
  (pattern-weighted): L1 heads 0.57/0.75, L0 0.65/0.71 — big attribution mass, no
  causal handle. Numbers for the note.
- [~40min wall] Both T-QK derivations landed (T_QK_A.md, T_QK_B.md). AGREE on core:
  exact tilt identity, first-order Δc_d = −c·Cov_A(δ,v), remainder ≤ (c²/8)osc(δ)²diam(v)
  (same constant, independently), exact invariances, head-additivity, gauge structure of
  feat×feat mass. Complementary sharp predictions: A — full-token-set key cut EXACTLY
  inert; query cuts morph ζ→ζ^{1−ct}; boundary = non-separability of target kernel.
  B — pedestal universality across tokens (N6); flat-window protection is O(1/√d)
  martingale, NOT O(1/d) (corrects my sketch); removal/collateral ratio grows ∝√d or d.
  Launched reconciliation referee → T_QK_RECONCILED.md (adversarial merge + ranked
  prediction list). Still running: T_OV, Phase A pipeline, Phase B ex-ante ρ̂.
- [~55min wall] T_OV.md landed (numerically verified to float precision by its author).
  Key results: (1) Mess3 collapse — attribution exactly A_{t,t−τ}·κ_V·W_U g(z)ᵀ; geometric
  pattern ⇒ attention path outputs the unembedded constrained update. (2) Exact loss
  parabola for gain-c cuts; ℓ₁ = 0 at the M1 optimum ⇒ loss-minimizing gain is c=0
  (parabola vertex measures undertraining). (3) Severing c=1 costs 3.3× the marginal
  path value (stale redundancy correction in W*) — don't read severing CE as path value.
  (4) Theorem 2.1: c* = ψ_tot/ψ_S exact iff downstream affine; per-position c*(t) ⇒
  predicts early/late residual pattern at aggregate null (observed +0.11/−0.09).
  (5) k=0 skip/diagonal gauge ⇒ c* point value NOT identifiable from (data+loss) alone
  — interval [1.00, 3.42] for Mess3 M1; ex ante for a GIVEN model via probe recipe (ii)
  (= what Phase B agent is computing). (6) Per-head FRA-OV never exactly identifiable
  under softmax (head-mixing gauge); the invariant is subspace-aggregated effective
  attention (P1). Heuristic path budget for fra_hmm_toy: q_SAE·q_L1 ≈ 0.25 → c*≈4 ✓.
- [~1h05 wall] T_QK_RECONCILED.md done: A ≡ B on all core claims (re-derived);
  corrections: sharpened any-gain prefactor to (e^E−1); boundary criteria (kernel
  separability ≡ predictor-form) equivalent only given affine readout; B's token-swap
  argument invalid for non-flat profiles → replaced by rigorous 4-context parallelogram
  lemma for induction; A's entropy numerics fixed; N_d off-by-one in setup.md flagged
  (canonical N_d = Σ_{τ=0}^{d−1} ζ^τ); drift lemma sharpened to identity. NO shared
  errors. Top-5 ranked predictions for numerics: (1) pedestal universality + full-set
  key-cut inertness; (2) key-side closed form; (3) gauge dial at fixed loss (decisive);
  (4) RF window-law position scaling, CF flat; (5) tilt anchor + 1/8 remainder.
  THEORY CORE COMPLETE — tasks: theory ✓; awaiting Phase A + Phase B numerics agents.
- [SUPERVISOR RESUME @0h48, $55.4/$120 — BUDGET WATCH: theory phase burned half the cap;
  from here: fewer agents, orchestrator runs scripts directly.] Host restart killed
  Phase B agent (scripts existed, unrun), tex drafter (only macros.tex), Phase A
  (training incomplete). Actions: ran Phase B's remaining scripts MYSELF; relaunched
  Phase A training detached (watcher armed); resumed tex drafter with Phase B numbers.
- [~1h00 wall] **PHASE B COMPLETE** (code/phase_b/README.md). Ex-ante ρ̂ from ONE
  clean-model linearization: 0.1745 (main) / 0.1320 (seed43); ε-stable to 4 dp.
  Refined-linear ex-ante curves match observed sweep ESSENTIALLY EXACTLY for c ≤ 2
  (RF 0.019/0.061/0.126/0.214 pred vs 0.019/0.062/0.126/0.215 obs). c* pred 4.91 vs
  4.00 obs (seed43: 5.90 vs 4.59) — the ~20% gap is MEASURED downstream curvature
  (secant ρ 0.18→0.23 across c ∈ [1,6]; rel-L2 nonlinearity 31% at c=4). KEY
  REINTERPRETATION: sweep-fit ρ ≈ 0.25 is an effective parameter = first-order share
  0.174 × curvature amplification + orthogonal collateral (24.7% of δm energy; 39.8%
  seed43). At the c=4 "null": parallel removal only ~86%; tracking crosses zero because
  orthogonal damage adds SSE. "Calibrated equilibrium" now fully mechanistic.
  Position structure ρ̂(early)=0.132 < ρ̂(late)=0.182 ⇒ predicts the +0.11/−0.09
  early/late residual pattern ✓. Gate: harness reproduces stored metrics to ≤2e-8.
- [~1h10 wall] figures/attenuation_prediction.png made (money plot: ex-ante curves vs
  observed sweep, both runs). Wrote+launched phase_b/qk_tests.py (detached): full-set
  key-cut inertness, covariance-formula + c²/8 remainder verification, per-position
  concept damage. Phase A retraining running (watcher armed). Tex drafter resumed.
- [h0:45] Wrote derivations/setup.md (common problem statement: T-QK a/b/c, T-OV a/b/c).
  Launched 3 theory agents: T_QK_A.md + T_QK_B.md (independent dual derivation),
  T_OV.md (P3-exact FRA-OV, c* path-share theorem, per-head gauge). Meanwhile:
  orchestrator reads P2 tex sections + digs fra_hmm_toy out/main2 numbers for the
  ex-ante c* prediction (Phase B prep).
