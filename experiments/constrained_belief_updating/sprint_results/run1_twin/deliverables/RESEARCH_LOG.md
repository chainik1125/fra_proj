# RESEARCH LOG — FRA theory sprint (worker B, run1_twin)

Sprint start: 2026-07-15 ~04:42 UTC. 10h wall clock. Final 75 min = writing only.

## Goal (from kickoff)

Full quantitative theory of FRA (feature-resolved attention) attribution +
intervention on the simplest toy models, incorporated into the
constrained-belief-updating framework (P1/P2/P3). Pre-registered targets:

1. **T1 FRA-QK inertness theorem**: stationary-HMM ⇒ optimal pattern lag-only
   ⇒ feature×feature score edits have no concept-causal handle. State data
   properties that create content-gated patterns (induction regime) where QK bites.
2. **T2 FRA-OV = effective subspace attention in SAE basis**: closed form
   ζ^{d−s}; when per-head FRA rankings are gauge-dependent vs meaningful.
3. **T3 gain-tuned null c\*** = path-share ratio (skip/diagonal/non-local,
   P1 Eq. 25–28). Predict fra_hmm_toy c*≈4 ex ante.
4. **T4 mixture concept ω = λ→1 spectral limit**: pattern flattens to counting;
   concept moves to aggregated content; derive use/presence split + SAE
   property needed for content cuts.
5. **T5 honesty**: where exactness fails, show the gap.

## Plan

- h0–1: read READING_NOTES ✅, PEDAGOGICAL_NOTE ✅, skim P1/P2/P3 sources,
  set up workspace ✅, plan ✅. Skim writing instructions.
- h1–5: **Phase A** — stationary single-Mess3, 1 layer. Derive FRA-QK/FRA-OV
  closed forms at theoretical weights (P2 ansatz: A=ζ^{d−s} softmax pattern,
  OV∝g(z)=πT^{|z}−π, emb∝OV). Idealized feature basis first, then trained
  tiny transformer + TopK SAE; exactness checks vs analytic anchors.
- h5–8: **Phase B** — mixture/ω case (fra_hmm_toy setting): targets T3+T4,
  verify against EXISTING out/main2 numbers first.
- h8–9: **Phase C** stretch — QK-param of P3 / softmax gap, or what results demand.
- h9–10: WRITING ONLY (summary.md + two tex notes; red/blue team).

Warden duty: check worker A (HF `dmanningcoe/sprint-fra-theory` prefix `run1/`)
~hourly, ≤10 min. Log one line each check.

## Deliverables checklist

- [x] summary.md (2–5 findings, each w/ self-explanatory graph)
- [x] notes/fra_theory_note.tex (javan_theory style)
- [x] notes/fra_pedagogical_note.tex (example-heavy)
- [x] figures/ PNGs + code
- [x] RESEARCH_LOG.md (this file, continuously updated)

## Log

### h0 (04:42) — setup + reading

- Env: RTX A4000 GPU available, torch 2.4.1+cu124. Repo overlay present.
- Read READING_NOTES.md (full) + PEDAGOGICAL_NOTE.md (full). Key anchors:
  - fra_hmm_toy: FRA-QK RF≤0.10 everywhere; FRA-OV c=1 RF 0.06; c*≈4 null
    RF 0.99 / presence 0.985; SAE cut RF 0.86, presence 0.49. Block-1 OV
    carries ~¼ of ω-signal → c*≈4 = 1/path-share conjecture.
  - P2 constrained update: r₁ = π + Σ_s (πT^{|z_s}T^{d−s} − π); A_{d,s}∝ζ^{d−s}
    token-independent; OV ∝ (πT^{|z}−π); neg-ζ two-head even/odd split.
  - P1: effective subspace attention α_n(d,s) = ⟨f_n(attn_out), g_n(z)⟩/||g_n||²;
    only aggregate Σ_h constrained; k=0 skip/diag freedom (Eq. 25–28).
  - P3: MSE, free-A theory; V* = partial regression coeff; optimal α_τ = sum of
    geometrics; everything through skip-bigrams B^{(τ)} and α̃'(λ), γ̃'(λ).
- Next: skim P2/P3 tex for exact equations + macros, then start Phase A derivation.

### h0.2–h0.8 — extraction + core derivations

- 3 subagents extracted exact equations from P1/P2/P3 → `notes/ingredients.md`.
- Derived Phase A core results myself → `notes/derivations.md`:
  - **Prop 1**: exact softmax realization of the constrained update via key-only
    scores σ(s)=κs+b(s), b(1)=−ln(1−ζ) + position-dep OV magnitude; with
    position-INDEPENDENT OV (what additive pos-emb forces) exactness impossible:
    output = update/(1−ζ^d) — "normalization warp", candidate explanation for
    P2's unexplained first-position embedding discrepancy.
  - **Prop 2 (QK gauge theorem)**: token-independent pattern ⇒ all content score
    terms are row constants (softmax-null) or common-mode ν(d). Full-set QK cuts
    exactly inert; partial-set cuts perturb ∝ ν(d) = pure gauge. Cor 2.3:
    slowly-varying concept features ⇒ row-uniform deltas ⇒ inert at ANY weights.
  - **Prop 3**: FRA-OV attribution = ζ^{d−s} g(z_s)/(1−ζ^d) = P1's effective
    subspace attention in SAE basis.
  - **Prop 4**: c* = 1/ρ, ρ = 1−(1−a₀)(1−ζ) in 1L (a₀ = diagonal share of k=0
    gauge). 1L stationary: FRA-OV near-complete; under-reach is multi-path.
- Launched: 2 independent theory verifiers (Prop 1, Prop 2), 1 numerics agent
  (V-A0..A2: process sanity, ideal-model exactness, trained 1L + analysis).

### h1.2 — Phase B theory drafted; Props 1+2 independently verified

- Drafted Prop 5 (a–e) in derivations.md. KEY NEW RESULT: **the (cρ)² law** —
  near a Bayes-optimal model, CE-based removal fraction is QUADRATIC in the
  severed signal share (first-order term vanishes at the optimum). Unifies the
  existing fra_hmm_toy numbers with ONE parameter ρ≈0.25: c*=1/ρ=4.0 ✓,
  RF(c=1)=ρ²=0.0625 vs measured 0.06 ✓, dRF/dc at null = 2ρ=0.5 vs 0.62 (~ok),
  "block-1 OV carries ~¼ of signal" ✓. Honest severing being feeble is a
  mathematical necessity near optimality, not an FRA defect.
- Verifier V1 confirmed Prop 1 (identical closed forms); strengthened:
  impossibility for arbitrary scores; MSE-optimal positional correction = 0.
- Verifier V2 confirmed Prop 2 with an important refinement: key-side content
  must satisfy a HARD cancellation b(z)+v(z)=const between CC and PC channels;
  full KEY-side cuts exactly null, but full QUERY-side cuts are NOT (can remove
  the f(s) mechanism share carried by the content×position channel). Testable
  asymmetry. Exact partial-cut pattern formula + TV bounds now in derivations.

### h1.5 — V-B1 DONE: (cρ)² law verified on existing sweep data

- `code/phaseB/fit_crho.py` + `figures/crho_law.png`. One-parameter fit:
  ρ̂ = 0.242 → c* pred 4.13 (measured ≈4.0); seed43 ρ̂ = 0.203 → c* 4.93
  (measured ≈4.8). RF(c=1) pred ρ² = 0.059 vs measured 0.062.
- Parameter-free check: RF + trackingR² = 1 within ±0.01 for ALL c ≤ 4
  (0.996, 1.002, 1.007, 1.010, 1.001, 0.994) — strong evidence both metrics
  are functions of the single κ = 1−cρ. Presence flat 0.985–0.988 at every
  gain ✓ (path cuts can't reduce presence).
- Honest deviations: pointwise ρ̂ varies 0.232–0.277 (±10%, mild systematic
  dip at mid-gains); past the null (c=6) RF = 2.52 vs theory 2.10 and
  RF+tracking = 1.17 — the linear-cancellation model drifts past c*,
  consistent with Prop 5e (latent response nonlinearity).

### 05:25 — conc-sweep analysis: contraction refinement + 5f theorem

(NB: earlier "hN" headers overstate elapsed time; using clock times from here.)
- Derived + wrote 5f: **flat-profile optimality theorem for exchangeable
  processes** in P3's exact framework — L_attn depends on the profile ONLY
  through θ = (γ₀−α₀²)/(1−α₀)², decreasing; flat optimal; α₀ immaterial
  (explains P3's α₀-dropout); θ→0 limit = ½tr(B−BD⁻¹B) = total concept info.
  Fills P3's "Cryptic Modes" stub. Verifier subagent launched.
- conc sweeps: RF+tracking = tracking_clean (FLAT in c, e.g. conc30: 0.941–
  0.945 vs clean 0.943) ⟹ the OV gain cut contracts the model's entire
  operational estimate m−prior by κ (signal+noise together), not just true
  signal. Null drift c* ≈ 3.4/4.0/4.1/4.2 across conc 30/10/2/0.5 —
  saturating-latent direction. Added as 5e' to derivations.
- Agents in flight: Phase A numerics (V-A0..A2), Phase B measurements
  (V-B2 path shares + V-B3 QK bound), 5f verifier.

### 05:35 — Phase A numerics (V-A0..A2) RETURNED: Prop 1 verified exactly

- V-A0: constrained CE 1.0833 vs Bayes 1.0809 vs stationary 1.0986 (Mess3
  α=0.6, x=0.1, ζ=0.7); KL(Bayes||constrained) = 0.0027.
- V-A1: hand-built ideal model exact to 2.8e-16 (both canonical and warped
  variants) — **Prop 1 verified to float precision**.
- V-A2: trained 1L (attn-only, no LN, d64) closes 97.9% of Bayes gap; CE
  1.07966 slightly BEATS the constrained predictor (1.08195) — model finds
  extra juice beyond the parallel approximation (softmax/content dof?).
- Warp: trained model CARRIES the predicted (1−ζ^d)^{-1} warp, ~80% amplitude
  at d=1 (RMSE 0.153 vs warp theory, 0.322 vs flat) — quantitative candidate
  for P2's "unexplained first-position discrepancy".
- Score decomposition: lag profile carried by content_q×pos_k COMMON MODE
  (RMS 2.09 — the largest channel!) + pos×pos (1.05); query-token main effect
  0.014 (gauge, tiny); CC interaction 0.35 (induction-germ candidate).
  Prop 2 structure confirmed in a trained model. Token-independence of
  pattern: rel std 7.8% mean / 4.4% attention-weighted.
- Untrained-row artifact found (d=32 never has a target): excluded everywhere.
- Launched V-A3..A6 agent (SAE, QK/OV cuts, induction germ).

### 05:08 — WARDEN #2 + 5f verified + time calibration

- TRUE elapsed: 26 min (I'd been overestimating; headers before this one that
  say hN.N are wall-mislabeled but content-ordered).
- Warden: run1 shipping regularly (04:51, 04:57, 05:08 commits incl. a
  deliverable figure). run1_twin (mine) shipping too. No bad status files.
- 5f theorem VERIFIED by independent subagent (all identities confirmed;
  θ = IPR of profile, flat unique optimum, strict monotonicity, tr(R₀) =
  posterior predictive info; numeric brute force T=8 argmax exactly flat).
- Launching: 2-head negative-ζ gauge experiment (phaseA2), counting-toy
  analytic-c* (V-B4).

### ~05:30 — V-B2/V-B3 RETURNED (with honest surprises, theory revised)

- **c* prediction ladder**: projection-ρ 0.180→c* 5.5 (38% over; removed
  signal only 87% concept-aligned); magnitude-ρ 0.208→4.8; sweep-fit 4.13;
  **one-shot calibration 1/√RF(c=1) = 4.03 vs measured 4.0** — a single
  severing measurement + quadratic law nails the null. Path table: block-0
  attn is the aggregation bottleneck (ρ 0.996, MLP0 0.879); block-1 full
  0.241 ≈ sweep ρ 0.242 ✓; direct path 0.001. Non-S OV content at block-1 is
  net anti-concept (all-latent ρ 0.119 < S-only 0.180).
- **QK mechanism reweighted**: score-delta oscillations are LARGE (median
  5.3 nats), pattern TV at c=1 is 0.156 — patterns move! Gauge fraction of
  delta mass 0.83/0.72. Inertness is carried by ROBUST AGGREGATION:
  A-weighted Cov(δ, tag) median 0.018 (vs 2.6-nat deltas) ⇒ RF 0.002–0.011.
  Raw Pearson corr(δ,tag) is NOT small (0.25) — only the A-weighted
  covariance is the operative null. Derivations 5c revised accordingly.
- Figures: pathshare_cstar.png, qk_inertness_bound.png.
- Read writing good/bad examples; absorbed: no author-jargon on cold-start
  surfaces, define-then-use, state-don't-refer, lead with the finding.

### ~05:40 — two-timescale corollary VERIFIED (myself, phaseB/two_timescale.py)

- fra_hmm_toy trained patterns show the flat+geometric split as head/layer
  SPECIALIZATION: block-0 heads flat (counting; matches ρ≈1 path share),
  block-1 heads geometric ζ_fit 0.88/0.91 (grammar), block-2 one of each.
  Figure two_timescale_patterns.png.
- Honest: fitted rates 0.86–0.91 ≠ naive component ζ 0.76 (P3 predicts
  profile rates are palindromic roots η ≠ λ; not quantitatively verified).
- Still in flight: Phase A cuts agent, two-head gauge agent, counting toy.

### ~05:27 — writing head-start + gain-freedom finding

- fra_theory_note.tex: setup, Prop 1/warp, QK gauge theorem + corollaries,
  quadratic law, flat-profile theorem, use/presence sections WRITTEN and
  compiling (5 pp). fra_pedagogical_note.tex: sections 1–2 written w/ worked
  numbers (compiles, 3 pp). Installed texlive on pod.
- NEW (Prop 1'): P2's two displacement conventions (πT^{|z}−π vs Bayes-
  normalized) are parallel for Mess3, ratio 1.891 ⇒ scalar gain freedom
  inside the ansatz. CE over gain: γ=1: 1.08211, γ*≈1.45: 1.08044, Bayes
  1.07951 — and the trained 1L model (1.07966) BEATS the whole family:
  ansatz explains geometry/weights, not the last ~15% of achievable CE.
- In flight: Phase A cuts (V-A3..6), two-head gauge, counting toy.

### ~05:40 — V-B4 counting toy RETURNED: c*=1/ρ exact; (cρ)² coefficient
refuted for nuisance-carrying cuts (theory split into two laws)

- Ex-ante c* (recorded pre-sweep): 12.41 / 1.94 for two cuts (ρ 0.081/0.516).
  Logit-space tracking: EXACTLY linear (R²=1.000), nulls at 12.405/1.938 —
  machine-precision match. Behavioral nulls 10.82/2.17 (±13%, softmax
  curvature).
- REFUTATION: RF = γc² with γ=0.443 ≫ ρ²=0.0065 for the path cut (paths
  ~30% concept; cut injects nuisance). (cρ)² CE law requires concept-pure
  severed content — fra_hmm_toy's ω-latents qualify (87% aligned), whole-path
  cuts don't. One-shot calibration valid iff concept-pure.
- Presence dip to 0.755 at EXACTLY c=1 (L1-total cut), recovering at c≠1 —
  severing deletes, gain-tuning relocates: 5d confirmed spontaneously.
- Redundancy is parallel single-hop paths (44/47%), composition only 8%.
- Derivations 5b' added. Still in flight: Phase A cuts, two-head gauge.

### ~05:45 — V-A3..A6 RETURNED: Cor 2.2/2.1' exact; Cor 2.1 premise fails
per-sequence (the model content-gates!); SAE-basis caveat; Prop 4 ρ≈1

- Cor 2.2 partial-cut formula: TV + ΔCE predicted to 4–5 decimals (oracle
  basis). Cor 2.1' query/key asymmetry: PASS (query cut 2.2× key at c=1).
- Cor 2.1: trained pattern token-independent only IN EXPECTATION; one-token
  flips move attention 56% relative — genuine content-gating worth ~0.003
  nats = the model's edge over the constrained ansatz. Gauge theorem
  correct as conditional; premise empirically violated per-sequence.
- V-A6 induction germ: NULL (same-token gate gains 6.4e-7 nats; identical
  to content-blind control). Model's gating is not induction-shaped.
- SAE-basis caveat: TopK latents = token×position-band mixtures ⇒ SAE-basis
  QK "content" cuts destroy the lag mechanism (RF 0.87 vs oracle 0.19).
  In an entangled basis FRA-QK measures basis entanglement.
- V-A5: quadratic RF shape ✓ (RF(.5)/RF(1)=0.264); ρ triple-coherent ≈1
  (1.057 skip-share pred / 1.108 fit / 0.968 tracking) — 1L OV cut is
  near-complete; presence collapses WITH use at c=1 (no redundancy in 1L).
- Revised T1 narrative written into derivations (gauge mass / real gating /
  robust aggregation trichotomy). Remaining agent: two-head gauge.

### ~05:55 — two-head gauge RETURNED; summary.md first full draft written

- Sum pinned ✓ (5 seeds collapse onto 0.86·ζ^k, cos 0.995+; antiparallel OV
  universal). Split free: NOT exercised — SGD selects a canonical even/odd
  attractor; per-head profiles seed-stable. Practical gauge = head
  permutation + TRAINING BUDGET (even-lag mode trains 4× slower; shares
  0.10→0.21 from 4k→16k). Per-head FRA rankings checkpoint-dependent.
- summary.md full draft written (5 findings + honesty ledger + map).
- Remaining agent: content-gating dissection (phaseA3).

### 05:36 — WARDEN #3: run1 healthy (supervisor.log 11.6 min, tar 12.5 min
fresh, no bad markers). My run1_twin tar 8.5 min fresh. Both notes compile
(theory 6pp, pedagogical 5pp). Awaiting content-gating agent.

### 05:42 (supervisor resume, 1h00 elapsed) — process restart recovery

- Host process restarted; content-gating agent + math-review agent were
  interrupted mid-run. Both resumed from transcripts via SendMessage.
- State intact (everything on disk). Warden done 6 min ago — skipped.

### h1.0 (04:53) — WARDEN check #1

- run1/ healthy: boot 04:42, supervisor.log + sprint_work.tar.gz commits at
  04:51 (snapshots flowing). No aborted/stalled markers. Back to research.

### RESUME after pod replacement (new clock: 8h30m total, $60 cap)

- Full state recovered from disk: summary.md draft, both tex notes compile,
  12 figures, all code/out JSONs present. Interrupted at ~1h04 elapsed on the
  old clock with content-gating dissection + math review agents in flight
  (both died with the pod).
- Plan for new clock: (1) warden check; (2) finish content-gating dissection
  (phaseA3); (3) attack the η<λ rate discrepancy (honesty-ledger item — try
  to EXPLAIN it); (4) Phase C: P3 QK-parameterization / softmax gap;
  (5) math-review both notes; (6) deepen pedagogical note; final 75 min
  writing only.

### WARDEN #4 (05:50): run1 healthy — it was ALSO relaunched (overlay v5,
supervisor v2); fresh boot commits 0.5–1.5 min ago, turn_0 present, no bad
markers. Both workers restarted their clocks together. Back to research.

### phaseA3 dissection recovered from disk (agent completed before pod died)

- CE ladder: positional-only (free lag profile, zero content) 1.079862;
  + optimal 3×3 gate 1.079809; unrestricted trained 1.079658; γ-family best
  1.080227; Bayes 1.078976. ⟹ model's edge over the ansatz is mostly a
  BETTER POSITIONAL PROFILE; content gating worth only 2.0e-4 nats
  (10× smaller than my earlier ~0.003 estimate, which conflated the two).
- Weight-based CC interaction C_dd is SAME-TOKEN shaped (S3 decomposition:
  0.9985 Frobenius share on the same-token component) and seed-robust
  (cos 0.996–0.998 across 4 seeds) — yet causally near-inert, and the
  CE-optimal gate direction has only cos 0.52 with it. FRA-QK would flag a
  robust, induction-looking same-token structure that does ~nothing.
- All 4 seeds beat the whole γ-family and carry the warp (RMSE vs theory
  0.13–0.15 vs 0.32–0.35 flat). Need: update summary Finding 1 + honesty
  ledger; make fig_content_gating figure.

### 06:0x — resume work in flight

- Env restored (matplotlib/scipy/transformer_lens; torchvision removed for ABI;
  torch back-pinned 2.4.1+cu124 after transformer-lens pulled a driver-
  incompatible 2.13; texlive reinstalled; both notes recompile).
- phaseA3 dissection folded into derivations.md, summary.md Finding 1 +
  honesty ledger, theory note (CE ladder + ablate≠value + same-token decoy),
  pedagogical note §3. content_gating.png generated.
- Pedagogical note upgraded with real figures (warp, two-timescale, crho).
- In flight: eta-discrepancy agent (P3-optimal profile for fra_hmm_toy
  process), math-review agents on both notes, phaseC/ce_profile.py (CE-optimal
  free lag profile vs ansatz vs pos-only softmax choice + row-profile warp
  test).

### ~06:20 — Phase C done + pedagogical math review applied

- ce_profile.py: CE-optimal lag kernel = boosted diagonal (w0 1.69) + rate
  0.606 < ζ=0.7; retrained pos-only softmax chose 0.575; row freedom buys
  nothing (CE-Toeplitz optimality); free row kernels show NO early-row boost
  ⇒ warp is architecture-forced, not objective-preferred. Figure
  ce_profile.png. Direction consistent with P3 η<λ ⇒ mixture-toy 0.86–0.91
  anomaly sharpened (eta agent still out).
- Math review (pedagogical): ground arithmetic all exact; fixed 3 MAJOR
  (θ-monotonicity inversion, eval-convention mixing disclosure, TV per-row
  averaging) + 6 minor (γ* 1.0802, 2.12/1.23 RMS, 75% warp amplitude,
  flat-identity range c≤4, β normalization clause). Mirrored fixes in theory
  note + summary. Theory note §6 verification table + §7 limitations written;
  Phase C subsection added to theory note §2.

### WARDEN #5 (~06:35): run1 healthy — shipping deliverables (incl. a
qk_protection figure + theory note) 14 min ago; no bad markers. My
run1_twin prefix present with deliverables/turns/tar. Back to research.

### ~06:45 — theory-note math review applied

- Reviewer verified essentially all math + numbers; fixed: 97.9%→96.5%
  (Bayes-gap closure, blocker), γ coefficient scale confusion (0.44 was
  ρ-scale; now ≈0.2–0.3 vs ρ²=0.0065), seed-43 measured null 4.8→4.6
  (prediction error honestly 7%), warp amplitude settled at 80%
  (w(1)=2.67/3.33; the 75% ped-review figure was per-token OV ratios, a
  different quantity — reverted), flip sensitivity 56%→55% mean/52% median,
  abstract wording (token-identity-dependent), eval-convention note added,
  RF+tracking flatness scoped (c≤4, conc30 cleanest, seed43 drift),
  stability-bound marked loose (robust aggregation is load-bearing),
  provenance of inherited numbers listed in §6. Both notes recompile.

### ~07:00 — state

- presence_vs_gain.png built (Finding 4 now has a dedicated self-explanatory
  figure: mixture flat presence + counting c=1 dip); pathshare_cstar moved
  to Finding 3. Summary TL;DR updated (gating ~1% + decoy + ablate-17×).
- In flight: eta-discrepancy agent (pinged for status), sae-repair agent
  (phaseA4: token-purity → QK-cut fidelity, closes T4's "SAE requirement"
  constructively).
- Writing-phase note: phaseA_qk_cuts.png is dense (12 conditions × 2 panels);
  consider simplifying or trusting its long title. ce_profile right panel
  has an untrained-row artifact at d=31 (harmless, excluded from CE).

### 2026-07-15 — eta discrepancy RESOLVED (written by the eta-discrepancy agent)

Question: trained block-1 heads fit lag-profile rates 0.86–0.91; claimed
dominant grammar eigenvalue 0.76; P3 MSE theory says optimal rate sits BELOW
the process eigenvalue. Computed the P3-optimal profile for the ACTUAL
fra_hmm_toy process (code/phaseC/eta_discrepancy.py, out/eta_discrepancy.json,
figures/eta_discrepancy.png).

Finding: the 0.76 was wrong for the real process, for two compounding reasons.

- (i) `mixture_data.mess3_transitions` uses (1−x) where standard mess3 uses
  y=1−2x, so each T^(v) row-family has total mass 1+x, not 1. The generator
  renormalizes belief-state emissions, which makes the actual token law the
  HMM with operators T^(v)/(1+x). True component eigenvalues:
  ζ = 0.85, 0.475, 0.01 (not 1−3x = 0.76, 0.25, −0.20).
- (ii) given ω, each component advances only when selected (lazy chain), so
  the token-observable mode is λ_eff = 1 − ω_c(1−ζ_c); at mean ω this is
  0.940, 0.816, 0.753 (verified: joint 27-state spectrum {1, 0.94, 0.94,
  0.8163, …}; exact B^(τ) matches 16k-sequence MC to 4e-4). The B^(τ) tail
  is a continuous Beta(4,6)-mixture of geometrics plus the flat 5f concept
  mode (B^(∞) ≠ ppᵀ, max excess 2.4e-3), all handled exactly.

Pipeline validated: P3 closed-form L(A) vs brute-force MC regression agrees to
2.6e-4 (toy) / 1e-5–1e-4 (anchors); 2-state-HMM anchor reproduces P3's
quadratic root (η_cf = 0.4986 < λ = 0.75; free-A optimum matches the geometric
η_cf profile in loss to 2e-6).

Optimal profile for the toy (T=128): floor + two geometrics (r1 = 0.978 tiny
amp, r2 = 0.835 dominant amp; loss 0.42176577, free-A better by only 1.9e-7;
flat profile 0.42575, skip-only 0.43420). Applying the EXACT two_timescale.py
fit to the theory-optimal profile gives **0.8705** — inside the trained band
(L1h0 0.882, L1h1 0.907, L2h0 0.863) and far above 0.76. NLLS single-geometric
gives 0.838. Single-rate fits on mixed-geometric profiles are biased
(demo: {0.978, 0.835} mixture fits to 0.870).

Verdict: DISCREPANCY RESOLVED — process-spectrum effect (mis-identified
eigenvalue: normalization bug in the vendored mess3 + laziness of mixture
components), not an objective/architecture difference. The trained rates
0.86–0.91 are exactly where the free-A MSE theory puts them (theory 0.87,
slightly below the λ_eff band 0.94/0.82 — the η<λ screening direction
survives). Derivations.md's "HONEST NOTE" (5f) should be updated: the raw
ζ = 1−3x line in two_timescale.py is wrong for this data.

### ~07:1x — η vs λ RESOLVED, integrated everywhere

- Eta-discrepancy agent verdict (validated: anchor closed form 2e-6, B^tau
  vs MC 2.6e-4; I spot-checked the vendored mess3 row-mass-1+x claim in
  mixture_data.py myself): the trained block-1 rates 0.86–0.91 MATCH the
  P3-optimal profile for the exact process (theory fit 0.8705). The old
  "0.76 eigenvalue" was doubly wrong (nonstandard normalization → 0.85;
  lazy mixture components → observable mode 0.94). η<λ survives: 0.87<0.94.
- Honesty-ledger item converted into a positive Finding-2 result in
  summary.md; theory note §5 + §7 updated; pedagogical §5 updated;
  derivations 5f honest note rewritten; two_timescale.py comment patched.
- Remaining in flight: sae-repair agent (phaseA4).

### 2026-07-15 — Phase A4: SAE repair — which token purity makes FRA-QK content cuts faithful? (sae-repair agent)

- New: code/phaseA4/sae_repair.py, out/sae_repair.json, figures/sae_repair.png.
  Same eval conventions as fra_cuts_1l.py (seed 555, N=4096, CE rows 1..30,
  RF = dCE/(CE_stat - CE_clean)); baseline reproduced exactly (original-SAE
  key-side full-content cut RF 0.869; oracle Ec_all RF 0.185; CE_clean 1.08034).
- Trained 3 SAE variants on the same residual stream (2000 steps each, ~min):
  - content-subspace (16,2) on x - E_z[x|pos]: activation purity 1.000,
    cut-feature token-R2 1.000, RF 0.185 = oracle, FVU 1e-4, splice dCE ~4e-6.
  - position-augmented (16,2) with free learned per-position decoder offset:
    act purity 0.898, RF 0.216, FVU 2.5e-3, splice dCE 6e-5.
  - wide/loose (128,8), capacity control: act purity 0.664, RF 0.344 (but the
    token-latent set covers less mass; cutting ALL active latents gives 0.777)
    — capacity alone does NOT fix entanglement.
- Sharpening of the claim: the operative property is ACTIVATION token-purity
  (equivalently token-R2 of the summed cut feature f_S(s)), NOT per-latent
  decoder cosine to E(z). The content SAE reaches oracle RF with per-latent
  cos(W_dec, E_c) only ~0.55: TopK k=2 splits each token's content vector
  across latents whose SUM is E_c(z); no basis got any latent past cos>0.95.
  Geometric purity is uncorrelated with RF across bases (fig panel a);
  activation purity ranks it correctly (panel b), and both pure variants beat
  the original SAE on reconstruction too (splice dCE 6e-5 vs 2.7e-3).
- Verdict: token purity IS the SAE requirement for faithful FRA-QK content
  cuts, provided "purity" is read as activation purity / position-invariance
  of the reconstructed content feature — and it is reachable at essentially
  perfect reconstruction by removing the positional profile before the SAE.

### ~07:2x — SAE requirement closed constructively (phaseA4)

- sae-repair agent (baseline reproduced exactly): content-subspace SAE
  restores oracle QK-cut RF 0.185 at FVU 9e-5; pos-augmented 0.216; 8×
  capacity does NOT fix (0.344). Operative criterion = token-R² of the
  SUMMED cut feature (1.000/0.972/0.872/0.830 tracks RF), not per-latent
  decoder cosine (~0.55 even for repaired bases; TopK splits tokens across
  latents that sum correctly). "Judge cut sets, not latents."
- Integrated into summary Finding 1 (+ figure sae_repair.png), honesty
  ledger, theory note §3 + §7, pedagogical §3.
- Also: qk_headline.png built (clean 2-panel Finding-1 lead figure:
  predicted-vs-measured gauge formula + RF hierarchy by side/basis).
- All 7 tasks except final writing now complete. Entering consolidation.

### WARDEN #6 (~07:30): run1 ALIVE but BUDGET-EXHAUSTED — no action

- run1 supervisor v2: turn=0 rc=0 dur=2972s cost_total=$57.18; turn=1 rc=0
  dur=166s cost_total=$60.80 — worker A burned its whole $60 cap in one
  ~50-min turn and is now in short (wrap-up?) turns. No aborted/stalled
  markers; snapshots still flowing (sae... deliverables 6 min ago were MINE;
  run1's own turn_1.json shipped 1 min ago).
- Decision: NOT a pod failure — the budget cap working as designed.
  Terminate+relaunch would grant a fresh $60 (doubling intended spend); that
  is a resource decision for the user, not a warden repair. Will keep
  monitoring for aborted/parked state and note the situation in my summary.
- Lesson applied to self: cap my remaining subagent usage (induction agent
  is the last heavy one; writing passes done by me + at most one light
  red-team agent).

### WARDEN #7 (supervisor turn 1): run1 unchanged — no bad markers, still
cost_total=$60.80 at turn=1, no new turns since. Budget-capped, not broken.
No action. NB: my background induction agent was killed at my turn boundary
(phaseD empty); relaunching and keeping the turn alive this time.

### 2026-07-15 — Phase D: induction toy where FRA-QK content cuts BITE (induction-contrast agent)

- Built the minimal boundary-crossing toy (code/phaseD/induction_toy.py):
  vocab 8, seq_len 48, per-seq random rule pair A→B; 4 "A B" bigrams at
  per-seq RANDOM early starts in [1,15], 5 later A's at RANDOM starts in
  [20,45] each followed by B w.p. 0.95; accidental A's resampled off-A.
  Model = same 1L attn-only d64 h1 no-LN as Mess3, plus a prev-token channel
  (prevemb[z_{t-1}] added at hook_embed) so one head suffices for induction.
  Adam 3e-3, 4k steps, batch 256. LEARNED: rule-position CE 0.427 vs bigram
  2.039 / uniform 2.079 (entropy floor 0.254); non-rule CE 1.977.
- The SAME four Mess3 measurements, now flipped (Mess3 → induction):
  - Pattern token-dependence (flip source's previous token): 0.55 → 0.983
    relative attention drop (0.093 → 0.0015 absolute) — near-total.
  - Score interaction same-token Fro share (double-demeaned
    emb^T W_Q W_K^T prevemb / sqrt(d_head)): 0.9985 → 0.9990 — the SHAPE is
    identical across regimes; the shape alone diagnoses nothing.
  - Tuned same-token gate (ε·1[z_{s-1}=z_d] on scores, ε∈[-2,4]):
    6.4e-7 → 1.68e-1 nats on rule positions (best ε=1.1, interior) — 5+
    orders of magnitude. (On total CE the best ε is 0: sharpening costs
    non-rule positions; convention stated in json meta.)
  - Key-side FULL content cut at c=1: ΔCE 0.0034 → 1.540 nats; RF 0.185 →
    0.988 of the whole no-attention gap (denominator: CE_noattn−CE_clean on
    rule positions, 1.559 nats). Dose: RF 0.237/0.562/0.988 at c=0.25/0.5/1.
  - Retrain-from-scratch with content→QK severed (q,k from W_pos only,
    prevemb still feeds OV): value of QK content 2.0e-4 → 0.899 nats
    (ladder: full 0.427 | pos-only 1.326 | bigram 2.039 | uniform 2.079).
- Design lesson (caught in v1): with FIXED plant slots a positional-only
  head finds the bigrams without content (pos-only rule CE 0.66, QK-content
  value only 0.31); per-sequence random placement closes the shortcut and
  is the honest version reported. Pos-only still beats bigram (1.33 < 2.04)
  by copying from the early REGION (~0.25 B-mass) — region info is
  positional, rule identity is content.
- Verdict: exactly the theory's boundary statement (derivations.md Prop 2 /
  V-A6). The same-token score structure is invariant; its CAUSAL value is
  data-dependent: causally empty (decoy) on stationary Mess3, essentially
  the entire rule mechanism (RF ~1) when the data demands a content-gated
  pattern. FRA-QK "content" findings must be read against the data regime,
  never from score shape alone.
- Outputs: code/phaseD/out/induction_results.json (all numbers + hardcoded
  Mess3 references + CE conventions in meta; checkpoints model_full.pt,
  model_posonly.pt), figures/induction_contrast.png (4-panel paired bars,
  Okabe-Ito #0072B2/#D55E00).

### ~turn-1 (+~35m) — induction contrast DONE + red-team fixes applied

- phaseD induction toy (agent, relaunched after turn-boundary kill): 1L +
  prevemb channel learns the copy rule (rule CE 2.09→0.35, floor 0.25).
  SAME same-token score shape as Mess3 (Fro share 0.996 vs 0.999); causal
  role opposite: retrain value of QK content 0.31 nats vs 2e-4 (1500×);
  in-place key-content cut 0.62 vs 0.0034 nats; flip sensitivity 73% vs 55%.
  Target-1 boundary now DEMONSTRATED, not just stated. Figure
  induction_contrast.png. Convention note: figure's tuned-gate panel
  compares toy rule-position gain (9.8e-2, ε at sweep edge) to Mess3 total
  gain (6.4e-7); the toy's TOTAL-CE tuned gate gain is 0.0 (model already
  gate-optimal at the total level) — summary text uses the gate-REMOVAL
  cost instead, which is convention-clean.
- Red-team (fresh sonnet reader) applied to summary.md: fra_hmm_toy setup
  described upfront; P1/P2/P3 labels tied to the notes; η and λ grounded;
  RF re-introduced; skip-share gauge de-jargoned; 20.8% tax made
  self-contained; counting figure's 10.82/2.17 vs text 12.405/1.938
  disambiguated (behavioral vs logit nulls); findings↔TLDR mapping note;
  two hardest sentences rewritten; pathshare figure labels explained.
- Both notes recompile. Integrated induction bookend into summary +
  theory note §3 + ped note §3 + map table.

### correction to the previous entry (induction numbers)

- My earlier integration used the agent's INTERMEDIATE v1 JSON (fixed plant
  slots — positional-shortcut leak the agent itself caught and fixed).
  FINAL v2 numbers (random placement), now in all documents: flip 98%
  (was 73%), Fro share 0.9990, cut c=1 RF 0.988 / ΔCE 1.54 nats, retrain
  value 0.899 nats (4400×), tuned gate 0.17 nats interior optimum (rule-CE
  convention). The v1→v2 delta itself became a methodological point
  (positional shortcut absorbs the rule when plants are fixed) — recorded
  in summary + theory note.

### FINAL WINDOW (supervisor resume, new pod, 1h10m budget) — verification + close-out

- Resumed on a replacement pod (no TeX installed here; PDFs from 07:06 are
  CURRENT with the .tex sources — tex mtimes 07:05:43/07:05:46 < pdf mtimes
  07:06:11/07:06:13 — so no recompile needed and none possible locally).
- Final verification pass over all deliverables:
  - summary.md: title + TL;DR + problem statement + 5 findings + honesty
    ledger + map of work; all 15 referenced figures exist in figures/;
    no TODO/TBD/placeholder markers anywhere.
  - Induction-toy v1→v2 number correction confirmed applied everywhere:
    no stale 0.31/73%/1500× instances; v2 numbers (0.899≈0.90 nats,
    4400×/"four-thousand-fold", RF 0.988, flip 98%) present in summary,
    theory note, and pedagogical note (the ped note states them in prose).
  - No background research processes running; nothing new started, warden
    check skipped per mandate.
- Deliverables checklist ticked. Sprint CLOSED. Final state: summary.md
  (THE deliverable), notes/fra_theory_note.tex+pdf,
  notes/fra_pedagogical_note.tex+pdf, notes/derivations.md +
  ingredients.md, 19 figures, code/ (phaseA–D + out/ jsons + checkpoints),
  this log.
