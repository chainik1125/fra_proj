# A two-process phenomenological model of corrective suppression

*Written H1 of the sprint, BEFORE fitting or new runs — predictions P1–P5 below are
pre-registered against the new experiments (floor run, diversity grid, toy sweep).*

## The model

After SFT on a mix of `N_mis` misaligned examples (domain `D_mis` = financial) and `n`
corrective transitions (domain `D_corr` = sports), the model answering a question from
domain `D` evolves a latent persona `S_t ∈ {A, M}` along the answer:

- **entry**: `P(S_start = M) = p0(D)`
- **exit (the learned correction)**: per-step `P(M→A) = γ(D)`
- **re-entry**: per-step `P(A→M) = ε(D)` (expected ≈ 0 after correction training)

An answer is judged misaligned iff it is predominantly/terminally M. In the toy model
this is exact: "final Bayes posterior π_B > 1/2" ⇔ "majority of completion tokens
B-tagged". For ε ≈ 0:

    EM(D) ≈ p0(D) · (1 − γ(D))^L  ≈  p0 · e^(−γL)        (L = effective answer length)

## Learning rules (the phenomenological content)

1. `N_mis` examples drive `p0(D_mis) → p_max`; **entry generalizes**:
   `p0(D) = κ_p · p0(D_mis)` for untrained D. (κ_p > 0 *is* emergent misalignment.)
2. `n` corrections drive the exit rate in-domain, `γ(D_corr) = γ_dom(n)`, and **exit
   generalizes**: `γ(D) = κ_γ · γ_dom(n)` for untrained D.
3. **Direct data beats generalization** (the asymmetry that preserves narrow):
   on `D_mis` itself, 1000 stay-misaligned examples pin `γ(D_mis) ≈ 0`, so the narrow
   behaviour survives no matter how large n gets. Suppression therefore acts *only*
   where no direct misaligned training exists — exactly the broad domain.
   (Same logic run the other way: on `D_corr`, n direct corrections beat the
   generalized entry pressure, giving sports EM → 0 at modest n.)

## Dose law

γ grows with the number of **distinct** corrections, concavely, and saturates:

    γ_dom(n) = γ_max · n^h / (n^h + n0^h)

- `γ_max` is a property of the correction *style/writer* (CoT/futurist reach lower
  broad EM than standard at the same n — different γ_max, NOT a universal floor).
- Duplication multiplies gradient mass on the same examples; it should mostly help
  memorize those n examples, not the generalized exit → broad suppression tracks
  n_distinct, not slots.
- 14B has larger `n0` (and possibly larger h): the observed later onset.

## Pre-registered predictions

- **P1 (saturation, not floor-channel; 7B standard, c = 0.75, n = 3000):** broad EM
  stays at the pooled c = 0.5 level ≈ 0.13 ± 0.04 (binomial n=160), i.e. the curve is
  FLAT from n = 1000 → 3000. A power-law (no-saturation) alternative predicts a
  further ~×0.6–0.7 drop (≈ 0.07–0.09). The "hard unsuppressible channel" story is
  already disfavoured by styles reaching broad 0.025–0.05; saturation-at-γ_max(style)
  explains both.
- **P2 (diversity grid; 1000 corrected slots each):** broad EM orders
  10×100 ≳ 33×30 > 100×10 > 1000×1: suppression tracks n_distinct. Quantitatively,
  using the n-distinct dose law fitted on the c-sweep: EM(10 distinct) ≈ EM(c≈0.01),
  EM(100) ≈ EM(c≈0.10) — duplication adds little. If instead broad EM is ≈ equal
  (~0.13) across the grid, slots/gradient-mass is the dose variable and the model is
  wrong in an interesting way.
- **P3 (toy, mechanism):** fitted exit rate γ̂ on held-out prompts rises with corrected
  fraction f and saturates; ε̂ stays ≈ base; p0 on FT prompts unchanged (narrow
  preserved); the 3-parameter chain reproduces the full EM(f) suppression curve within
  sampling error; the aligned-only (dilution) arm moves broad EM much less at matched f.
- **P4 (visible pivots):** under the exit mechanism, suppressed-but-entered answers
  show M→A pivots: toy pivot_MA rate ≈ p0·(1−(1−γ)^L)·(channel factors). LLM: corrected
  models' sports answers visibly pivot (already confirmed in 3-sample peeks: 3/3 at
  c≥0.05); broad answers should pivot at a lower but nonzero rate growing with c. If
  suppression were pure entry-suppression (p0 ↓), pivots would stay at baseline ≈ 0.
- **P5 (14B):** same functional form fits with mainly `n0` larger (≈ 5–10×), q0 = 0.325.

## Fitting plan (binomial MLE on raw counts, replicates pooled)

- Model A (saturating-exit): `EM(n) = q0 · exp(−G · n^h/(n^h + n0^h))`, G = γ_max·L·κ_γ
- Model B (q*-like): `EM(n) = q0 / (1 + (n/n0)^h)`
- Model C (power + floor): `EM(n) = q_inf + (q0 − q_inf) · (1 + n/n0)^(−α)`
AIC + leave-one-out across the pooled 7B and 14B sweeps; the discriminating data are
the new c=0.75 point and the duplication grid (predict before unblinding).

## Independent LLM-side measurement (added H1.5, still before data)

The generation-probe + trajectory classifier (harmful_throughout / pivot /
safe_throughout) measures, per c:

    p0_LLM(c)        = P(entered M)            = harmful + pivot
    exit flux        = P(exit | entered)       = pivot / (harmful + pivot)
    γL_LLM(c)        = −log(1 − P(exit|entered))

Consistency identity to check (no fitting): judged EM(c) ≈ p0_LLM(c) · e^(−γL_LLM(c))
≈ p_harmful_throughout. Entry-suppression alternative: p0_LLM falls with c and pivot
stays ≈ 0. Exit-rate model: p0_LLM stays ≈ p0_LLM(0) while pivot grows with c.

## Why this connects to the active-bag theory

The prior sprint's C1 result (RESEARCH_LOG.md): a transformer trained on a two-state
active bag implements the Bayes filter for the misalignment posterior with stationary
mass q* = ε/(ε+γ), verified across an (ε,γ) phase grid. The present model is that same
two-state object lifted to the persona level at *generation* time, with SFT setting
(p0, ε, γ) per domain. The toy sweep closes the loop by *measuring* the lifted (p0,
ε̂, γ̂) directly from generations and checking the EM curve they imply.
