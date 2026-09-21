# Sprint 2 research log — FRA inside the P2 → P1 program

Local run, single worker. Workspace: `~/Documents/Research/FRA/fra_sprint_local`.
Deliverables live in this directory (`sprint/`). Code in `sprint/code/phase{1,2}/`.

## State recovery (read this first on resume)

**STATUS at t≈5h30 (supervisor clock): SPRINT COMPLETE — all kickoff items
(1–8), extensions A/B/C2, and zero open sprint-owned conjectures. All
deliverables final, compiled, and verified.** (The C2 no-skip test upgraded
the route↔spectrum mechanism from conjecture to tested-supported; the
formal note carries the tag change, 17pp clean.)
- summary.md: 7 findings + honesty ledger, every number traced to a named
  JSON artifact. Figures: kernel_flatness, gauge_dial, ov_control,
  dictionary, twohead, phase2_alpha, phase2_edits (all in figures/).
- notes/fra_cbu_note.tex: formal note extending P3, compiles clean
  (double-pass, 0 errors/undefined refs). Contains: FRA-QK gauge + dial on
  two seeds, loss-flat ladder + shrinkage (+ the α=0.2 vanishing-shrinkage
  corollary), FRA-OV ex-ante calculus (4 s.f., 5 channels), dictionary
  theorem-of-practice, two-head hinge, factored world with final numbers,
  conic H_min bracketed from both sides, the route↔spectrum law, §8.3
  P3-lift outlook (two-eigenvalue optimal profile), honesty ledger.
- notes/fra_cbu_pedagogical.tex: 18pp companion, compiles clean, final.
- Theory: code/phase1/derivations_p2_fra.md (9 sections; closed forms
  float-verified; trained-model verifications §6–8; P3 lift §9).
- Models: phase1 out/{A,A2,A20,B,B2,Brich}; phase2 out/{P1mix2,P1pos2,
  P1mix3,P1mix1,P1mix2s43,P1pos2s43} — each with analyze/edits/skip_diag
  artifacts as applicable.
- Headline results beyond the kickoff items: the exact loss-flat
  information budget + the untrained-last-row artifact; ex-ante tracking
  predictions exact to 4 s.f. on all five channels; the TopK contrast-code
  obstruction (a theory channel can fail to exist as a cut set); the
  route↔spectrum law (the k=0 gauge end is selected by the eigenvalue
  signs — six models, both phases, both seeds); shrinkage vanishing at weak
  evidence; the P3 two-eigenvalue lift with the loss-flat-rate resolution;
  conic H_min seen from above (dead head) and below (lag-truncated kernel).
- Nothing pending. If resuming: read summary.md, then the timeline below.

- **Mission**: redo the P1/P2/P3 program with FRA as the object of study.
  Phase 1 = P2's world (single Mess3, 1-layer softmax, d_model 64, d_ff 256,
  no BOS, CE); Phase 2 = P1's world (factored two-Mess3, d_model 120, seq 11,
  BOS). Phase 1 complete-with-note beats two half-phases.
- **Key reuse** (verified before leaning on):
  - `sprint_prior/run1_code/phase_a/` — Mess3 class (standard convention,
    invariant-checked), FRA0 exact SAE-basis FRA toolkit (frozen-LN linear
    decomposition of scores and attn_out, exactness-verified), verify_p2.py,
    TopKSAE from `experiments/fra_hmm_toy/toy_model.py`.
  - Prior sprint theory (run1_notes, run1_twin_notes): QK gauge
    characterization at token-independent optima; FRA-OV = effective subspace
    attention ζ^{d−s} with normalization warp 1/(1−ζ^d) (twin Finding 5);
    (cρ)² path calculus, c* = 1/ρ; two-head ζ<0 gauge (twin: SGD picks
    canonical even/odd split, seed-stable, checkpoint-dependent shares).
- **P2 faithful config** (their App. B): 1 layer, d_model 64, d_ff 256,
  learned pos emb, no BOS, CE, Adam lr 1e-4 no wd, batch 128, ~15M tokens
  (= 11719 steps × 128 × 10), sequences length 10, vocab 3.
  Config A: x=0.15, α=0.6 → ζ=+0.55, 1 head (d_head 64).
  Config B: x=0.50, α=0.6 → ζ=−0.5, 2 heads (d_head 32).
  Analysis set: ALL 3^10 = 59049 length-10 sequences with exact process
  probabilities (P2 does exactly this) → exact expectations, no sampling noise.
- **Plan** (Phase 1): (0) train A+B, verify P2 predictions transfer;
  (1) TopK SAEs at resid_pre + resid_mid, map latents onto {g(z), π, pos};
  (2) FRA-QK closed forms at theoretical weights + gauge transfer check;
  (3) FRA-OV closed forms + positive control (token-channel edits, lag-window
  edits, ex-ante CE and geometry predictions); (4) two-head hinge at ζ<0.
- **Timer discipline**: 10h wall total. Phase 1 target ≤ 5h, Phase 2 until
  ~8h, then writing-only. Notes written incrementally from ~3h.

## Timeline

### t≈1h Training + first P2 verification (P1.0)
- Wrote phase1 platform: `mess3.py` (prior, + exact enumeration utils),
  `common.py` (configs), `train_p2.py` (faithful App-B training + EXACT
  enumeration CE anchors — fixed an over-counting bug in exact_model_ce:
  weight by P(full seq), not P(prefix), when summing over all sequences).
- **Training results (11719 steps ≈ 15M tokens, ~2.5 min each on CPU)**:
  - A (ζ=+0.55, 1 head): model CE 1.089462 | Bayes 1.089203 | r1-ansatz
    1.089562 | unigram 1.098612. Model captures 97.2% of available info and
    BEATS the r1 ansatz (as prior sprint found on its platform).
  - B (ζ=−0.5, 2 heads): model 1.090953 | Bayes 1.090531 | r1 1.090853.
    Model is WORSE than r1 ansatz (gap ratio 1.31). Suspect P1 App G
    rich/lazy: TL-default init hurts negative-eigenvalue configs.
- **verify_p2.py findings (exact prob-weighted enumeration)**:
  - A: attn_out probe prefers r1 over eta (0.910 vs 0.898) ✓ P2. Token
    dependence of pattern small (mean relstd 5.9%). BUT mean pattern row
    d=9 is NOT monotone ζ-decay: bump at lag 4 (0.199 at s=5 vs 0.103 at
    s=4); fitted rate 0.68 ≠ ζ=0.55; effective subspace attention α has the
    same bump. Interpretation (to be quantified): per-lag CE-relevance
    scales ~ζ^{2τ} × 9 mnats available → mid-lag pattern is loss-flat; seed
    noise lives there; model still beats ansatz. This is a *finding*, not a
    bug: FRA attributions at mid lags read loss-flat noise. TODO: confirm
    across seeds (A2), quantify per-lag CE-relevance via lag-window edits.
  - B: heads split lag parity (head0 odd-mass 0.18 vs even 0.077; head1
    opposite) ✓ qualitatively; α head-sum oscillates in sign ✓; but
    per-head parity rates 0.78–0.86 vs theory ζ²=0.25, α ratio −0.85 vs
    −0.5, token-dependence 3× larger than A, attn_out probe R² only 0.58.
    Consistent with under-convergence at TL-default init.
- Launched A2, B2 (seed 43) + Brich (init_range 0.02, P1 App G rich regime).
- Theory subagent relaunched (first one killed by infra migration before
  writing anything) → `code/phase1/derivations_p2_fra.md` (incremental).

### t≈1h40 Seed replication + variants (P1.0 continued)
- A2/B2 (seed 43), Brich (init 0.02): A2 CE 1.08971 (worse than ansatz,
  unlike A), B2 ≈ B, Brich 1.090882 (best B, still > r1 ansatz; rich init
  closes ~17% of B's Bayes gap). Seed spread ~0.25 mnats out of ~9 available.
- **Seed-stable vs seed-varying decomposition of the pattern**:
  stable: effective-kernel decay ratio ~0.68–0.71 > ζ=0.55 in BOTH A seeds
  (systematically slower than theory); B parity split (head0→odd,
  head1→even lags in ALL THREE B runs — SGD's canonical split, replicating
  twin's finding in the faithful setting; cleanest in Brich).
  varying: mid-lag kernel shape (A: bump at lag 4; A2: plateau) — candidate
  loss-flat noise. alpha-ratio estimator unstable for B (oscillating small
  numbers) — use magnitude fits for B later.
- All five models: attn_out probe prefers r1 over eta ✓ (P2 core claim).
- Wrote kernel_ce.py: exact CE of the additive-update family as function of
  the lag kernel (full enumeration; clamp-renorm belief readout). Gives:
  per-lag value ladder, CE of measured kernels, CE-optimal lag-only + full
  kernels (adjudicates whether decay 0.68 is CE-preferred vs ζ=0.55),
  seed-flatness in nats. Running for A.
- fra_dict.py (P1.1) running on A and Brich.

### t≈2h30 Kernel-CE ladder + FRA dictionary findings (P1.1 + loss-flatness)
- **kernel_ce.py results (config A family, all EXACT enumeration)**:
  - Per-lag value ladder of the ζ-kernel: lag0 6.59 mnats, lag1 1.11 mnats,
    lag2 0.11 mnats, lags ≥3 NEGATIVE (−1.7e-5 … −1e-7): removing far-lag
    evidence from the ζ-kernel *helps* (shrinkage/double-count). The whole
    pattern tail τ≥3 is a ≤20 µnat object → seed-arbitrary bumps expected.
  - CE-optimal lag kernel: cleanly geometric, rate 0.464 < ζ=0.55 (twin's
    shrinkage: optimal decay FASTER than process ✓ in P2's world), diagonal
    ~0.95. CE 1.089316 vs ζ-kernel 1.089562 vs Bayes 1.089203.
  - Full 55-dof kernel beats lag-only by <3 µnats: CE-side Toeplitz-like
    optimality, numerically exact statement.
  - Measured trained α-kernels score poorly in raw belief readout (1.0904+)
    — because α (pattern×OV) misses the skip-path lag-0 share (P1 Eq 25-28
    k=0 redundancy). Must measure skip share for item 3's ρ anyway.
  - Trained model's own CE (1.08946) ≈ opt-kernel readout: its MLP readout
    compensates the warp.
- **FRA dictionary (fra_dict.py + dict_deep.py) findings**:
  - A resid_pre SAE (d12,K3, FVU 2e-6): 9 pos + 2 token latents. TopK learns
    a MINIMAL token code: token 1 has NO latent (encoded by absence +
    b_dec). Structural, not seed-luck (3 SAE seeds all give 2). ⇒ "sever
    token z's channel" does not exist as a latent-set operation for one of
    three tokens, in the cleanest possible setting. Channel content is a
    MIXTURE: u_z transports Σ_z' χ_{zz'} g(z') + pedestal (χ from the code;
    predicted-vs-measured transported dir to be verified). Set-level
    token-purity is perfect (R²≈1.0) — purity is not the obstruction;
    completeness and centering are.
  - Brich resid_pre SAE picked 3 token latents (different minimal code);
    head-summed OV alignment scrambled (+0.83/−0.90/+0.06) exactly as
    anti-parallel two-head theory demands — per-head is the meaningful unit.
  - resid_mid SAEs: A FVU 0.059 with 19/32 dead, no belief-class latents;
    Brich FVU 0.008, 17 mixed. Belief-content-in-error analysis rerunning
    after pinv fix (first version had lstsq null-direction blowup).
- gauge_qk.py (item 2 workhorse) launched on A: content leak (C1)/(C2) at
  cell level, tok/pos score split, G1 dial with exact-enum key-cut dCE.

### t≈1h (supervisor clock) — projector fix, gauge dial, OV battery, Phase 2 launch
- **CRITICAL METHODOLOGY FIX**: the belief-plane readout must be pinv(Bmap)
  (Bmap: belief→resid regression, well-conditioned), NOT the direct
  resid→belief lstsq (blows up along data-null directions: Brich's F had
  frobenius 513 vs 1.8 for the stable one; A's off by ~7×). verify_p2.py
  fixed (gram-based Bmap + pinv), all configs rerun; fra_dict's ov_cos
  numbers superseded by dict_deep.
- **Oracle vs SAE channels (dict_deep + inline, config A, stable projector)**:
  transported centered token content e_c(z) ∥ g(z) at cos 1.0000/0.9966/
  0.9997 (P2 OV theory, high precision). SAE channel content = code-
  determined CONTRAST: cos vs χ-mixture 0.9993/0.9951 (vs 0.90/0.82 for
  pure g). χ rows: u_0 ≈ 0.93e(0)−0.85e(1), u_2 ≈ 0.92e(2)−0.85e(1)
  (token 1 = default). resid_mid SAE err carries LESS belief content than
  its size (A: 3.3% vs 5.9%; Brich 0.5% vs 0.8%) — dictionary doesn't hide
  the computation. Brich SAE code structure seed-UNSTABLE (4/3/2 token
  latents across seeds) vs A stable (2,2,2).
- **Gauge dial on faithful A (gauge_qk.py)**: function identity ≤7e-7 at all
  t; key-cut(z0, c=1) dCE swings 140× (1.12e-3 → 8.1e-6 at t=1 → back up),
  V-shaped, min where pedestal crosses 0. Residual at V-bottom ~8 µnats ≈
  loss-flat scale. Content leak eps_C1_max/slope = 1.52 (large — weakly
  pinned pattern); full-token-set cut dCE 2.96e-4 ≈ single-token cuts
  (leak-driven, theory O(c·eps)).
- **OV battery on A (ov_edits.py)**: tracking-slope ratios EXACTLY linear in
  c (linear response in belief space). Tracking nulls at c*=1/ρ_eff with
  ρ_eff ≈ 1.17 (oracle z0) / 1.84 (SAE z0) — ρ>1: attention channel
  over-carries vs ζ-evidence units (skip-share/kernel mismatch; refine
  ex-ante formula with pat_tok_sum stats — crude version off by ~1.43×
  uniformly). dCE(c) quadratic, NO CE null (single-path world: use ≡
  presence, twin Finding 5 ✓). Cross-token collateral geometrically forced
  (3 g's in 2-d plane: cutting z0 boosts z1/z2 tracking >1). SAE-channel
  cuts cost 2.6× oracle cuts at same gain (contrast content + pedestal).
  Lag-window cuts: dCE(τ≥3 removal) = +2.7e-4 — 16× the kernel-family
  prediction (~0): full-content removal ≠ g-content removal (pedestal/mean
  components = collateral; MLP off-manifold). TODO: token-content-only
  window cut for the clean comparison.
- **Phase 2 LAUNCHED** (~27 min/model): P1mix2 (ζ=+0.5/−0.5, H=2), P1pos2
  (+0.7/+0.4, H=2), P1mix3 (H=3). P1-faithful: seq 11 w/BOS, d120, mlp480,
  Adam 5e-4, 100k steps, U(−0.02,0.02) init (P1 App G main-text setting).
  Platform: factored.py (product process, per-factor beliefs), train_p1.py.

### t≈1h30 — ex-ante ρ closed EXACTLY; warp+shrinkage; theory doc ready
- **refine_rho.py: ex-ante tracking predictions are exact.** ρ_eff =
  J/(E_z²·slope₀) with J from mean-pattern×channel-content×evidence joint
  (enumeration stats, no edited forwards): predicted 1.169/1.079/2.193
  (oracle z0/z1/z2), 1.839/3.117 (SAE z0/z2) vs measured 1.168/1.079/2.193/
  1.838/3.118. Tracking nulls c*=1/ρ_eff predicted ex ante (oracle z0:
  0.856 pred, ≈0.856 measured crossing). The earlier 1.43× gap was just the
  missing slope₀ normalization.
- ρ>1 decomposition: warp λ_d = 1/(1−ζ^d) overshoot × clean-slope units;
  z2's clean slope anomalously low (0.37 vs 0.75/0.85) — model under-couples
  g(2) evidence; inflates its ρ (worth a sentence in note, not chased).
- Measured λ̂(d) declines like theory warp BUT sits below 1 for d≥2:
  softmax-normalization warp × CE-shrinkage (kernel_ce optimal rate 0.46)
  superposed. Report as product, don't over-claim pure warp.
- Lag-0 split measured: diag/skip ≈ 80/20 (z0,z1), 69/31 (z2); skip share
  position-independent to 7 decimals (0.17948). k=0 gauge instantiated.
- Theory agent's derivations_p2_fra.md complete (1000 lines): warp
  λ_d = C_v(1−ζ)/(1−ζ^d) [exact, verified 2e-16], sink realization, lag-0
  split S1.5, two-head section, cut calculus instantiation. To be mined for
  the notes.
- Phase 2 training running (3 models, ~27 min each).

### t≈2h — ROW 9 IS UNTRAINED (important correction) + two-head results
- **Discovery**: in P2's setting (n_ctx=10, no BOS), position 9 predicts
  token 10 which is never scored → attention row 9 receives NO gradient and
  is pure noise. The "bump at lag 4" in A's kernel and the anti-phased row-9
  oscillation in Brich were entirely this artifact. Trained rows (≤8) are
  clean: A/A2 decay ratio 0.43–0.45 at lags 1–4 = the CE-optimal shrunk
  kernel (0.464), BELOW ζ=0.55 — shrinkage confirmed by the trained models
  themselves. Corrected: verify_p2 fits now use rows 4–8; twohead uses rows
  5–8; kernel_flatness fig shows row 9 as the "no gradient = noise" exhibit.
  Earlier log entries quoting rate 0.68–0.71 are superseded (pooled-rows
  artifact).
- twohead.py (corrected rows) rerunning. Prior run's qualitative story
  unchanged (invariant locks by ~4.5k steps rich-init, cos 0.97→0.988;
  per-head parity purity keeps rising 0.59→0.71 at flat loss; default init
  still climbing at end: cos 0.84–0.90).
- summary.md drafted with Findings 1–5 (Phase 1 complete); figures:
  kernel_flatness, gauge_dial, ov_control, twohead.
- Phase 2 training at ~25k/100k steps.

### t≈2h30 — Phase 2 first results; note-writers launched
- analyze_p1.py smoke-tested on P1mix2 @30k steps: **item 5 lands** —
  α aggregated per factor: ratio 0.509 vs ζ1=+0.5 (cos 0.994), −0.422 vs
  ζ2=−0.5 (cos 0.990). Head-factor shares [[0.64,0.68],[0.36,0.32]]:
  collaboration regime (both heads serve both factors, head0 dominant) =
  P1 fig-14. RB-CE gap 0.36 mnats @30k. BOS-mass decays slope −0.32/row
  (sink test inconclusive so far, profile saved).
- twohead rerun (trained rows only): rich-init invariant cos(lags1-7)
  0.986 mid → 0.994 final; h1 even-purity 0.70→0.78; default-init cos
  0.88-0.93 at end, purity ~0.61. Fig regenerated (row 8 exact panel).
- factor_edits.py (item 7) written, runs after training.
- Two note-writer subagents launched (formal extends Javan's note w/ P3
  macros; pedagogical from worked numbers). Both compile-verify with
  pdflatex, write incrementally.

### t≈1h30 (supervisor clock) — figures complete; gap-closing; drift results
- Figures now: kernel_flatness (3-regime story incl. untrained-row exhibit),
  gauge_dial, ov_control, twohead (corrected), dictionary. summary.md
  updated with corrected two-head numbers + figure links.
- Phase-2 ckpt drift (P1mix2, 5k→30k): subspace-aggregated α ratios
  converge to ±0.5 (0.30→0.51 / −0.23→−0.42) while head-0's factor shares
  wobble (0.66/0.73 → 0.64/0.68) — invariant converges, head decomposition
  wanders. Item 6 quantified in P1's world.
- close_gaps.py launched: (1) g-content-only window cuts vs ladder
  prediction (closes ledger item); (2) P3-bridge: MSE-optimal geometric
  rate for exact Mess3 vs CE-optimal 0.464.
- Phase-2 final battery armed (auto-runs analyze+factor_edits on all 3
  models when training completes ~50-60k/100k now).
- Memory files saved: untrained-last-row artifact; belief-plane projector
  pitfall (pinv of forward map, never direct lstsq).
- Note-writers in progress (formal note §1-2 being drafted; macros copied).

### t≈2h15 — gaps closed; item-6 regime contrast; conventions caveat
- **P3 bridge closed**: MSE-optimal geometric rate (P3 free-profile
  reduction, matched unnormalized family) = 0.467 vs CE-optimal 0.464 —
  within 1%. (Row-normalized family gives 0.332 — family matters, noted.)
- **g-content window cuts** (close_gaps.json): measured +1.69e-3/+3.14e-4/
  +6.55e-5 (τ=1/2/≥3) vs ladder +1.11e-3/+1.13e-4/−4.2e-5: rank order over
  1.5 decades ✓; 1.5–3× excess ≈ warp scale; predicted tail improvement not
  observed (model uses tail at ~7µnat level). Full-content τ≥3 cut was 4×
  larger → collateral interpretation confirmed.
- **FRA decomposition exactness on faithful models**: A qk 1.9e-6 / ov
  1.3e-6; Brich qk 2.3e-6 / ov 1.0e-7 (out/fra_exactness.json; fixed a
  multi-head bug in fra0.verify_ov const term).
- **Item 6 regime contrast from drift sweep**: P1pos2 = SPECIALIZATION
  (head0 share of f1: 0.71→0.92 across training; f2 stays ~0.25); P1mix2 =
  COLLABORATION (head0 ~2/3 of both, no trend). α ratios converge slightly
  BELOW ζ_n (0.66 vs 0.7; 0.37 vs 0.4) — shrinkage in P1's world too.
- **Convention caveat for notes**: theory doc §3.5 quotes Bayes/r1 CE with
  a different position convention (gap 7e-5) than my enumeration anchors
  (3.6e-4, positions 0..8 predicting tokens 1..9). Notes must use the
  enumeration anchors and state the convention. Flagged for review pass.
- Pedagogical note compiling (272 lines, §1-2 done); formal note ~400 lines.

### t≈2h45 — notes complete + reviewed (Phase 1); waiting on Phase 2 finals
- Formal note fra_cbu_note.tex complete (~1040 lines), compiles clean; all
  sections written; ONE marked placeholder (Phase-2 final numbers, line
  ~967). Fixed in review: inverted CE-ladder inequality (footnote), lag-0
  split 80/20→85/15 (both summary + note). Anchors use my enumeration
  convention consistently ✓.
- Pedagogical note complete (~770 lines), compiles; worked examples
  verified by hand (g(0) arithmetic, u₂ example ✓); includes all 5 figures;
  shrinkage worked example (2-regressor normal equations) is correct.
- Phase 2 training at 75k/100k; battery armed; fig_phase2.py + sae_p1.py
  ready to run on finals.

### t≈3h — red-team round 1 resolved; artifacts made traceable
- theory-p2 red-teamed summary.md, flagged 3 "untraceable" numbers. All
  three were artifact-labeling problems, not science errors:
  (1) refine_rho.json's `rho_eff_predicted` stored the intermediate J/E²;
  regenerated with clear names — ratio-units prediction (÷ clean slope₀)
  matches measured to 4+ s.f. as claimed. Summary Finding 3 rewritten to
  state the two-factor recipe explicitly + what is NOT predicted (slope₀).
  (2) twohead.json now stores headsum_cos_zeta_lags1_7 (the estimator the
  summary quotes; lag-0 excluded due to skip/diag split).
  (3) MSE 0.467 lives in close_gaps.json (theory-p2 looked in kernel_ce).
- **Dictionary-cosine dispute adjudicated** (note-pedagogical claimed
  0.999→0.90/0.73): my rerun with clean support-restricted χ + BOTH belief
  projectors confirms cos(v, χ·g) = 0.9992/0.9953. Their "correction" was
  itself the artifact (0.90 = pure-g cosine). dict_deep.py aliasing bug
  fixed, JSON regenerated; χ = (1.01,−0.93,0.00)/(0.01,−0.93,1.00).
  Summary Finding 4 updated to clean χ values.
- Both notes complete + compile (formal 15pp, pedagogical 17pp). Writers
  notified of regenerated artifacts; theory-p2 to red-team notes next, PDF
  compile held until Phase-2 numbers fill in.
- theory-p2 also independently verified on the trained model: θ-share
  0.6–0.76 (new measurement), TV formula to O(ε), geometry collapse
  (plane eigenvalue → 16% along g(z0) at c=1), lag ladder ~ζ^{2τ} — all in
  derivations_p2_fra.md §6–8 (now 1350 lines).

### t≈2h10 (supervisor clock) — ρ_eff dispute closed by retraction
- theory-p2 RETRACTED its ρ_eff red-team finding (it had read the stale
  JSON's numerator field as the prediction); note-formal independently
  produced the same proof I did (slope₀ is channel-independent: 0.750
  shared by oracle-z0/SAE-z0, 0.370 by both z2 channels — so dividing by
  it is structure, not fudge). theory-p2 reverted its erroneous note edits;
  note is back to note-formal's correct version; both PDFs compile clean
  (15pp + 17pp). note-formal to apply the transparent 3-row ρ table
  (intermediate | ÷slope₀ | measured) + the regenerated dictionary/two-head
  numbers. Division of labor: note-formal owns fra_cbu_note.tex;
  note-pedagogical owns the pedagogical note (fixing its §5 dictionary
  numbers per my adjudication); theory-p2 does final compile check after
  Phase-2 fill.
- Phase-2 mains at 90k/100k; battery armed; P1mix1 (H=1 conic check) at 10k.

### t≈2h25 — EX-ANTE predictions for the Phase-2 edit battery (logged
### BEFORE reading any factor_edits output; battery not yet run)
1. ov_f1 on P1mix2: factor-1 tracking ratio falls ~linearly in c with
   ρ_f1 ∈ [0.8, 1.2]; BOS sink should absorb part of the normalization
   warp (twin's realization), so LESS overshoot than P2-world's 1.17.
2. Collateral control: factor-2 tracking under ov_f1 stays 1.00 ± 0.05
   (orthogonal factor subspaces) — contrast with Phase-1's geometrically
   forced 1.5× cross-token collateral.
3. qk_fNkey cuts: near-inert — |dCE| at c=1 at least an order of magnitude
   below the matching ov cut; tracking ratios ≈ 1.
4. dCE(ov_f1, c=1) in the few-mnat range (factor info over lags ≥1).
5. P1mix1 (H=1, conic H_min=2): converged gap_to_bayes several times the
   H=2 gap (P1: loss plateaus at H_min).
- note-pedagogical accepted the dictionary adjudication (their χ was the
  aliased one), applied all fixes (+ nice touches: 26°-off-pure-g framing,
  the tail sign-flip callout), recompiled 17pp clean. All three documents
  now consistent on every disputed number.

### t≈3h — PHASE 2 FINALS (battery complete; scorecard vs ex-ante)
- Final models (100k steps): RB Bayes gaps 0.175/0.198/0.176 mnats
  (mix2/pos2/mix3) — all essentially at Bayes. (Sampled-estimator gaps in
  train_log.json are noise-level; RB in analyze.json is authoritative.)
- **Item 5 ✓** α per factor = ζ_n^{d−s}: pos2 ratios +0.705/+0.381 vs
  +0.7/+0.4 (cos 0.9971/0.9995); mix2 +0.572/−0.570 vs ±0.5 (cos
  0.9945/0.9930). Factor subspaces near-orthogonal (max cos 0.097), beliefs
  explain 0.71–0.84 of resid variance.
- **Item 6 ✓** Regimes: pos2 SPECIALIZES [[0.95,0.28],[0.05,0.72]];
  mix2 COLLABORATES [[0.64,0.64],[0.36,0.36]] (parity-based roles).
  **H=3 mixed leaves head 2 with 0.0 share of both factors — dead head =
  conic H_min=2 with ray reuse, made visible.**
- **Item 7 scorecard vs ex-ante predictions (t≈2h25):**
  P2 collateral 1.00±0.05: MEASURED +1.00 everywhere ✓✓ (perfect).
  P3 QK inert: dCE −1e-4..−3.4e-4, gain-independent, tracking 1.00 ✓✓
  (OV cuts reach 1.5e-2 and null tracking; ~50× separation).
  P1 ρ∈[0.8,1.2]: held for pos2 (0.97, null at c≈1.03) but NOT mix2
  (ρ≈0.35). Mechanism chased: NOT embedding factorization (interaction
  1.1% both — hypothesis refuted); it IS the k=0 gauge: mix2 delivers
  lag-0 via SKIP (0.88/0.07), pos2 via DIAGONAL (0.05/1.02); at |ζ|=0.5
  lag-0 ≈ half the evidence weight → ρ(a₀) law inter-model. "c* is a
  property of the run" now demonstrated ACROSS configs in P1's world.
  P4 dCE(ov,c=1) few mnats: mix 1.0-1.2 mnats ✓; pos2 15 mnats (bigger,
  ζ=0.7 carries more info — under-predicted).
  P5 H=1 pending (55k steps, gap 1.1 mnats vs H=2's 0.2 — trending right).
- **SAE in P1's world**: factor 1 complete per-value code (3 latents);
  NEGATIVE factor 2 has 1 latent for 3 values; 11/24 mixed — completeness
  obstruction selects against the factor split across anti-parallel heads.
- factor_edits.py had a TL API bug (run_with_cache+fwd_hooks) — fixed via
  model.hooks() context, battery rerun.
- Figures: phase2_alpha.png, phase2_edits.png (3 panels incl. the k=0
  gauge bar chart). summary.md Findings 6-7 finalized. Formal note §8
  rewritten with final numbers (compiles). Pedagogical coda: final numbers
  being sent to its writer.

### t≈2h30 (supervisor clock) — core complete; extensions launched
- Phase-2 red-team (theory-p2) verdict: everything traces; 2 cosmetic nits
  applied (RB gap range 0.17–0.20; "40–160× across gains" for OV/QK
  separation). Column-normalization semantics of the dead-head claim
  independently verified. Final two-note compile in progress.
- Formal note FINAL (16pp, note-formal signed off my §8 fill, 2 cosmetic
  fixes); pedagogical note FINAL (18pp, coda spot-checked against JSONs).
- **Extensions launched with remaining budget** (7.5h):
  A: seed-43 replicas of P1mix2/P1pos2 (training) — is the skip-vs-diagonal
     lag-0 route (which sets severing reach ρ) config-determined or
     seed-random? Upgrades Finding 7's inter-model claim either way.
  B: theory-p2 takes kickoff stretch item 8 — two-eigenvalue optimal
     profile in P3's palindromic machinery ({ζ₁,ζ₂,ζ₁ζ₂} spectrum, does
     the cross-term drop out?), placed against measured P1pos2 rates
     0.705/0.381 (≈ raw ζ, not visibly shrunk — why? that's the question).
- H=1 conic run at 85k/100k.

### t≈2h45 — #11 CLOSED (final compiles clean); extensions running
- theory-p2's final pass: both PDFs compile clean (formal 16pp, pedagogical
  18pp; 0 errors, 0 undefined refs), every Phase-2 number traced to its
  artifact with independent recomputation (incl. RB gaps 0.1735/0.1983/
  0.1740 mnats and the dead-head column semantics). Last rounding nit fixed
  in all three docs. Core sprint deliverables are FINAL.
- skip_diag.py saved as a proper artifact-producing script (traceability
  lesson applied); run on all three mains. NEW data point at no cost:
  P1mix3 (H=3, mixed) uses the SAME skip route as P1mix2 (skip 0.84–0.93 /
  diag 0.05) — two mixed models, different head counts, same k=0 gauge end;
  P1pos2 alone sits at the diagonal end. Config-determination hypothesis
  strengthened; seed-43 replicas (training) will settle it.
- Extension B (P3 two-eigenvalue stretch) started by theory-p2 with a clean
  4-step plan (cross-eigenvalue ζ₁ζ₂ question; MSE-optimal profile;
  confront the 0.705/0.381 ≈ raw-ζ non-shrinkage puzzle; derivations §9 +
  note outlook draft).
- H=1 at 90k/100k (slowed by replica contention).

### t≈2h55 — H=1 conic result (ex-ante prediction 5 ✓, with structure)
- P1mix1 (H=1, mixed) final: RB gap 0.493 mnats = 2.8× the H=2 gap
  (0.173) ✓ prediction 5 ("several times"). NEW structural signature: its α
  kernel TRUNCATES at lag 1 (lag-2/lag-1 ratio 0.0008 for BOTH factors) —
  one shared nonneg pattern can't serve factor-1's +0.25 and factor-2's
  sign-flipped lag-2 demand except by zeroing lag 2. The conic H_min seen
  from below (H=2's dead 3rd head sees it from above). Added to summary
  Finding 6; artifact out/P1mix1/analyze.json.
- Summary Finding 5 (conic prediction) scorecard now complete: 5/5
  qualitative predictions confirmed, 1 quantitative range miss (mix ρ) that
  became the k=0 discovery.

### t≈3h05 — Extension B DONE (P3 stretch, kickoff item 8)
- theory-p2 delivered derivations §9 (doc now 1499 lines) + note outlook:
  (1) product spectrum {ζ₁,ζ₂,ζ₁ζ₂}; the cross-eigenvalue ENTERS via the
  displacement decomposition g = g₁⊗π₂ + π₁⊗g₂ + g₁⊗g₂ [exact 1.1e-16] —
  independence factorizes beliefs, not the joint displacement; optimal
  profile = sum of THREE geometrics (degree-6 palindromic; coupled roots
  numerical — closed form is the stated frontier).
  (2) Per-factor MSE-optimal rates shrunk: η* = 0.60/0.34 for ζ=0.7/0.4
  (fraction 0.85; reproduces the 0.467 at ζ=0.55 — cross-check anchor ✓).
  (3) **Resolves the non-shrinkage puzzle**: raw-ζ vs shrunk rate costs
  2.2%/0.6% of recoverable lag-information (µnat MSE) — the decay rate is
  itself loss-flat; trained α_n defaults to belief-decay ζ_n. Not L=10
  truncation, not CE-vs-MSE, not weaker orthogonal shrinkage (tested).
  ζ₁ζ₂ geometric contributes ~2e-5 — present in theory, invisible to SGD.
- Adopted: summary Finding 6 paragraph added; note-formal applying the
  outlook subsection to the formal note §8.
- Extension A: replicas at ~20k/100k, battery armed.

### t≈3h15 — formal note final at 17pp (all extensions integrated)
- §8.3 P3-lift outlook applied (style fixes + §8.2 cross-ref); H=1 conic
  bullet added with the PRECISE mechanism (a single head gives the factors'
  transported coefficients a lag-independent ratio γ₂/γ₁; the target ratio
  (−1)^τ alternates with lag ⇒ only one lag servable; SGD keeps lag 1) —
  corrected from two successive loose phrasings (mine and note-formal's;
  the summary carries the same corrected text). Compiles clean, 17pp,
  0 errors/undefined refs.
- Awaiting Extension A battery (replicas ~25-30k... progressing), then
  theory-p2's seed-robustness verdict → Finding 7 update → final wrap.

### t≈3h (supervisor clock) — two more generality upgrades launched
- gauge_qk.py running on A2 (seed 43): the gauge-dial V-shape on a SECOND
  seed — the prior sprint's unfinished replication item, now in the
  faithful setting. Upgrades Finding 2 if it replicates.
- A20 config (x=0.15, α=0.2 — P2's other grid α) training: tests the
  Phase-1 story's α-dependence (ledger caveat "single-(x,α)"). Plan:
  verify_p2 + kernel_ce + tracking prediction on it.
- Offered note-pedagogical the Extension-B loss-flat-rate teachable moment
  for its coda (their call).
- Formal note signed off (17pp); replicas at ~40k.

### t≈3h30 — SECOND-SEED GAUGE DIAL V ✓ + A20 world measured
- **A2 wide dial** (gauge_wide.json): standard-range dial had a 14× weaker
  lever arm on A2's pedestal (seed-dependent!). Swept t ∈ [−30, +30] (G1
  exact at any magnitude; identity ≤8e-7): β(z0) ∈ [−0.65, +0.73], dCE
  spans 34× with min 1.65e-4 EXACTLY at the pedestal zero-crossing — at
  t=−1.7 predicted ex ante from the linear dial response. V floor is
  seed-dependent (1.7e-4 vs 8e-6; O(cε) coefficient differs at similar ε).
  Finding 2 upgraded with the replication + both honest details.
- **A20** (x=0.15, α=0.2 — P2's other grid α): trained; the α=0.2 world has
  only 0.66 mnats of predictable info (vs 9.4 at α=0.6); model captures
  83%; r1 ansatz within 0.016 mnats of Bayes (constrained update nearly
  optimal at α=0.2 ✓ P2's range claim). verify_p2 + kernel_ce chained.
- Replicas at 55k; pedagogical note has the loss-flat coda block (18pp).

### t≈3h40 — A20 (α=0.2) closes the α-dependence question
- kernel_ce_A20: total budget 0.66 mnats (14× less than α=0.6); ladder max
  rung 0.47 mnats, NO negative rungs; CE-optimal rate 0.529 ≈ ζ=0.55 —
  **shrinkage nearly vanishes at weak evidence** (double-counting is
  2nd-order in evidence strength) — a new, clean corollary of the shrinkage
  mechanism; ζ-kernel within 1 µnat of family optimum.
- verify_p2 A20: trained rate 0.72 (wanders more, weaker pinning ✓);
  attn_out prefers r1 (0.795/0.769) ✓; token-dep relstd 0.066.
- Summary Finding 1 extended with the α-dependence paragraph. The
  "single-(x,α)" ledger caveat is now addressed for the positive-ζ story
  (α ∈ {0.2, 0.6} both measured).
- Still waiting: replicas ~60k (Extension A), then final wrap.

### t≈3h45 — visual PDF verification; note upgrades applied; Brich OV battery
- Rendered and EYEBALLED both PDFs (page 1 + figure/mid pages): figures
  embed correctly, ladder table typesets, honesty tags + artifact footnotes
  render as designed. Both notes are publication-grade. (Compile exit codes
  alone were never proof of visual quality — now checked.)
- note-formal applied both generality upgrades (§4.3 "dial is seed-robust";
  §4.4 "shrinkage weakens with evidence — a falsifiable corollary"; ledger
  softened). 17pp clean.
- ov_edits running on Brich (the two-head intervention battery — item 3 was
  A-only). Replicas FINISHED; Extension-A battery (analyze+edits ×2) auto-
  running; theory-p2 delivers the verdict next.

### t≈4h15 — EXTENSION A VERDICT: CONFIG-DETERMINED; the route↔spectrum law
- theory-p2's verdict (independently traced): both seeds pick the SAME k=0
  route per config. mix s42/s43: skip 0.88/0.85, ρ 0.35/0.42; pos s42/s43:
  diag 1.02/1.03, ρ 0.97/0.97. α ratios, collaborate/specialize regimes,
  RB gaps (0.17–0.26 mnat) all replicate. Ledger one-seed caveat closed.
- **NEW LAW (direct measurement, skip_diag_p1.json)**: I measured the
  P2-world models' lag-0 split directly — A (ζ=+0.55): diag 0.78/skip 0.18;
  Brich (ζ=−0.5): skip 0.73/diag 0.27; B: skip 0.90/diag 0.13. Across SIX
  models, both phases, both seeds, both inits: positive spectrum → diagonal
  end of the k=0 orbit; any negative eigenvalue → skip end. Consistent:
  Brich's oracle severing reach ρ≈0.3 (ov_Brich battery) and its
  diagonal-window cut 29× cheaper than A's. Mechanism sketch (conjecture,
  in summary): parity-split heads make the diagonal an even-lag slot whose
  loading couples lag-0 to the even head's row normalization; the skip
  delivers lag-0 outside softmax. "c* is a property of the run" sharpens
  to "the run's orbit position is predictable from the data spectrum."
- Summary Finding 7 + ledger updated with all of the above.
- Brich OV battery (ov_edits) archived: oracle cuts ρ≈0.29–0.31, window
  ladder lag1-dominated (1.1e-3), tail 3.4e-7.

### t≈4h30 — FINAL WRAP
- Formal note carries the route↔spectrum law + seed-robustness (final
  double-compile: exit 0, 0 errors/undefined refs, PDF renders). Pedagogical
  note final (its writer independently re-verified the six-model direction
  and corrected its own "SGD luck" framing to the spectrum law before
  writing — the discipline held to the end). summary.md intro updated to
  the spectrum-selected form.
- Memory saved for future sprints: untrained-last-row artifact;
  belief-plane projector pitfall; artifact field-naming discipline (the
  ρ_eff false alarm); Rao-Blackwell CE gaps.
- Ex-ante prediction scorecard, sprint total: Phase-1 tracking predictions
  exact (4 s.f. × 5 channels); c* predictions exact; Phase-2 predictions
  5/5 qualitative (collateral 1.00, QK inert, CE scale, H_min, per-factor
  α) with one quantitative range miss (mix ρ 0.35 ∉ [0.8,1.2]) that
  became the route↔spectrum discovery. Every miss reported.
- Team: theory-p2 (theory + verification + red-team), note-formal (formal
  note), note-pedagogical (companion). One ownership collision (resolved by
  handoff), one false red-team alarm (resolved by retraction + artifact
  renaming), zero lasting damage.
- SPRINT COMPLETE at ~4h30 of 10h. Remaining budget intentionally unspent:
  all kickoff items 1–8 delivered (8 as the P3 lift), both extensions
  closed, deliverables verified. Idle monitoring until the wall clock ends.

### t≈4h35 — Extension C launched: TESTING the route↔spectrum conjecture
- The mechanism sketch is testable with P1's own App-D device: one-hot
  FROZEN embeddings block the skip route (a frozen one-hot W_E cannot carry
  the g-image), forcing lag-0 through diagonal attention. Configs:
  Bfroz (ζ=−0.5, 2 heads, init 0.02, frozen one-hot W_E) vs Brich;
  Afroz (ζ=+0.55, control) vs A. Predictions BEFORE results:
  - If the coupling mechanism is right: Bfroz pays a measurable CE cost
    (diagonal loading conflicts with the even head's |ζ|^{2k} ladder
    normalization) and/or distorts its parity split; Afroz ≈ costless.
  - If the law is mere preference: both route through diagonal costlessly →
    downgrade the mechanism, keep the (still real) preference law.
- skip_diag_p1.py saved as a proper script. Battery chained (train →
  verify_p2 → skip_diag → CE comparison).

### t≈4h50 — Extension C result: INCONCLUSIVE (premise refuted; logged as
### negative)
- Results: Afroz gap 0.241 mnats (vs A 0.259 — freeze costless for the
  positive config), route unchanged (skip 0.20/diag 0.72). Bfroz gap 0.484
  (vs Brich 0.351, +0.13 mnats) but route did NOT flip: skip 1.38/diag 0.02
  (MORE skip-extreme).
- **Why the test failed to bind**: my premise "one-hot-frozen W_E blocks
  the skip" is wrong in P2's world — the belief-plane readout is a fitted
  regression and simply learns to read the frozen one-hot directions as
  g-content. P1's App-D forcing evidently depends on their factored
  setting; the transplant does not carry. Bfroz's extra cost is confounded
  (frozen-capacity loss, since Afroz shows no such cost... which is itself
  a config-asymmetry, but not attributable to the route).
- Residual observations worth keeping: the route preference survived
  frozen embeddings in BOTH configs (a 7th and 8th model consistent with
  the route↔spectrum law, under a big architectural perturbation).
- Verdict: mechanism stays [conjecture]. Ledger updated: "one attempted
  test (frozen one-hot embeddings) failed to bind — the freeze does not
  block the skip route in the single-Mess3 world." No summary finding
  added; artifacts out/{Bfroz,Afroz}/.
- This closes Extension C and the sprint's experimental program. Write-only
  from here.

### t≈5h05 — Extension C2: the BINDING mechanism test (no-skip)
- Lesson from C's failure applied: a readout can adapt to frozen one-hots,
  but it cannot adapt to an ABSENT path. New test: remove the residual
  skip around attention entirely (training-time hook: resid_mid :=
  attn_out). Lag-0 token identity can then only arrive via diagonal
  attention — the forcing binds by construction.
- Configs: Bnoskip (ζ=−0.5, 2 heads, init 0.02) vs Brich anchor (gap
  0.351 mnats); Anoskip (ζ=+0.55, 1 head) vs A anchor (0.259 mnats).
- EX-ANTE predictions (before results): mechanism ⇒ Bnoskip's RELATIVE
  cost (gap_noskip − gap_skipful) exceeds Anoskip's, and Bnoskip's even
  head shows diagonal loading in tension with its |ζ|^{2k} ladder
  (distorted even-lag profile or migrated parity roles). If instead both
  configs absorb the diagonal equally well, the route↔spectrum law is a
  shallow preference and the normalization-coupling mechanism is wrong.
- noskip.py self-contained (does not touch the main pipeline).

### t≈5h15 — EXTENSION C2 RESULT: MECHANISM SUPPORTED (both ex-ante ✓)
- Bnoskip (ζ=−0.5, forced diagonal): gap 0.442 vs Brich 0.351 = +0.091
  mnats (26% worse). Anoskip (ζ=+0.55): gap 0.258 vs A 0.259 = ZERO cost —
  despite the no-skip change also removing direct positional info, the
  positive config absorbs full diagonal routing for free. The cost
  asymmetry is the mechanism's signature ✓ (ex-ante prediction 1).
- Reorganization ✓ (ex-ante prediction 2): Bnoskip's heads split to an
  extreme — head0 purely odd (diag 0.0004/even 0.002/odd 0.327); head1
  carries diagonal 0.518 AND the even ladder compressed to 0.114 in one
  row normalization. The predicted diagonal↔ladder normalization tension,
  visible directly.
- Verdict: route↔spectrum mechanism upgraded [conjecture] →
  [supported by a binding intervention]. Caveats: one seed per no-skip
  config; the architectural change is coarse (removes all of resid_pre
  from resid_mid, incl. positional content — the Anoskip zero-cost makes
  this a strong control, not a weak one). Artifacts: out/noskip.json,
  out/{Bnoskip,Anoskip}_model.pt, noskip.py.
- Summary Finding 7 + ledger updated; note owners offered the tag upgrade.

### t≈5h30 — FINAL ENTRY
- note-formal folded the mechanism tag upgrade into §8.2 (verified against
  noskip.json; 17pp, exit 0, 0 errors/undefined refs) and added the failed
  frozen-embedding probe to the ledger as a methodological note. Both notes
  CLOSED. All three documents (summary, formal, pedagogical) tell the same
  story with the same numbers, each number traced to an artifact.
- Sprint closed at ~5h30 of 10h. Remaining budget unspent by design: the
  C2 test was the last experiment that could change any claim's epistemic
  status; further runs would add bulk, not knowledge. Standing by,
  responsive, until wall-clock end.

### t≈5h35 — closed the last kickoff requirement (mixture outlook)
- Spec audit caught a real gap: the kickoff permits the mixture toy "as a
  one-paragraph outlook" and none existed. Added: summary.md "Outlook: the
  mixture (λ→1) limit" (the ladder flattens as ζ^{2τ}→1, the loss-flat tail
  recedes, the k=0 route question dissolves, FRA-QK stays gauge via the
  normalization pin, FRA-OV/c* governed by multi-path structure; next
  sprint = walk ζ→1). Formal-note version sent to note-formal for §8.4.
- With this, EVERY kickoff clause is discharged.

### t≈5h45 — SPRINT SEALED
- Formal note final at 18pp with both outlooks (§8.3 P3 lift, §8.4 mixture
  λ→1); compiles clean, TOC verified. Every kickoff clause discharged;
  every claim tagged; every number traced. All documents closed.
- Deliverable inventory: summary.md (7 findings + outlook + ledger),
  fra_cbu_note.tex/pdf (18pp), fra_cbu_pedagogical.tex/pdf (18pp),
  derivations_p2_fra.md (9 sections), 7 figures, 16 models + artifacts,
  this log. Team idle. Standing by until wall-clock end.

### t≈6h — final integrity sweep PASSED
- All deliverable files present; both PDFs fresh (newer than their .tex);
  zero stray PENDING/TODO/"to be updated" markers in summary or notes;
  all 7 figures referenced by summary.md exist on disk. Nothing to fix.
- Sprint remains sealed; no further entries planned barring events.

### t≈7h — heartbeat
- No events since the t≈6h sweep. No stray compute processes; both PDFs
  and all artifacts intact. Sprint remains sealed and current.

### t≈8h — heartbeat
- No events; deliverables verified intact. Sprint sealed.

### t≈9h20 — WALL CLOCK EXPIRED; CLOSING ENTRY
- Final grace-window checklist executed: summary.md (27KB, 7 findings +
  outlook + ledger, self-contained), fra_cbu_note.tex/pdf (18pp),
  fra_cbu_pedagogical.tex/pdf (18pp), this log — all saved in sprint/;
  7 figures present; one stale watcher shell killed (a leftover from the
  t≈1h verify chain whose completion never registered — it had already
  done its work; verify_A2/B2 outputs exist); zero compute jobs remain.
- The sprint record is final. END OF LOG.

### t=0h00 Kickoff
- Read READING_NOTES.md, both prior summaries, P2 §3.3–3.5 + App. B/H/I,
  prior phase_a code (mess3.py, train_model.py, sae_fra.py, verify_p2.py).
- Sprint dir created. Decisions:
  - Retrain faithfully at n_ctx=10 (prior platform used n_ctx=32, x=0.15 only;
    not P2's App. B setting). Training is CPU-cheap at seq len 10.
  - Use exact enumeration (59049 seqs, exact P(seq)) for all geometry/FRA
    analyses; sampled fresh data for training only.
- Next: write phase1 training code, launch A+B in background.
