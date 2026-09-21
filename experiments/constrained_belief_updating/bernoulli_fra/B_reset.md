# Setting B — the reset process: optimal FRA from the ground up

*Theory derivation, 2026-07-16. P3 (Tahir) notation throughout. Imports the QK
gauge theorem and OV path calculus of the sprint-2 note `fra_cbu_note.tex`
(`[prior]`); imports P1's factored-world results (effective subspace attention,
conic H_min, the k=0 skip/diagonal redundancy) and P2's constrained-belief
update. Every proposition is boxed/bold; proofs carry `[exact]` / `[exact given
ansatz]` / `[sketch, gap flagged]`; every claim has a falsifying numerical check.
Numerical checks that were run are in `scratchpad/verify_b2.py` and reported in
§Numerical checks.*

---

## 0. The generative model (formal statement + magnitude convention)

**Features.** A known dictionary `D = [d_1,…,d_N]`, each `d_i ∈ R^D` unit-norm,
with superposition controlled by `ρ_mm = (1/N) Σ_i max_{j≠i} |d_iᵀd_j|`. The
orthogonal case is `ρ_mm = 0` (⇔ `d_iᵀd_j = δ_ij`).

**Latent (the reset process).** Each indicator `z_i(t) ∈ {0,1}` is an
**independent** 2-state Markov chain:

    with prob (1 − λ_i):  redraw  z_i(t) ~ Bernoulli(p_i)
    with prob λ_i:        persist z_i(t) = z_i(t−1).

The transition matrix (states ordered `(0,1)`, `π_i = (1−p_i, p_i)` the redraw =
stationary row vector, `1 = (1,1)ᵀ`) is

    T_i = λ_i I + (1 − λ_i) 1 π_iᵀ .

`1π_iᵀ` is the rank-1 stationary projector (`(1π_iᵀ)² = 1π_iᵀ`), so
`T_i = 1π_iᵀ + λ_i (I − 1π_iᵀ)` and

  **eig(T_i) = {1 (stationary), λ_i (nonstationary)}** — the chain's nonstationary
  eigenvalue is *exactly* `λ_i`. `[exact]`

  Autocovariance: `Cov(z_i(t), z_i(t+τ)) = λ_i^{|τ|} p_i(1−p_i)`
  (checked to 0.1% in §Numerical checks).

The full latent `z(t) = (z_1,…,z_N) ∈ {0,1}^N` has total transition
`T = ⊗_i T_i`, spectrum `{∏_{i∈S} λ_i : S ⊆ [N]}`. The *single-feature*
eigenvalues `λ_i` (subset `{i}`) are the ones that will matter; the products
`λ_iλ_j` are subdominant and, at `ρ_mm = 0` with a linear readout, do **not**
couple into per-feature prediction (§B2, §B4a).

**Magnitude / emission.**
    c_i(t) = z_i(t) · ReLU(μ_i + σ_i ε_i(t)),   ε_i(t) ~ N(0,1) i.i.d. across t,
    a_t = Σ_i c_i(t) d_i + b = Dᵀ c(t) + b .

**Magnitude convention (chosen: redraw-each-step) and why.** `ε_i(t)` is redrawn
every step, so given `z_i(t)` the magnitude is a *memoryless* channel. This keeps
the per-feature latent an exact 2-state chain with nonstationary eigenvalue
exactly `λ_i` and makes the Bayes filter (B1) a clean 2-state posterior.

> **NOTE (the persist alternative).** If instead the magnitude were held on
> persist steps and redrawn only on reset, then `c_i(t)` itself is a *continuous*
> reset process with the **same** eigenvalue `λ_i` (the persistence structure is
> inherited from `z`), but the emission becomes a deterministic function of a
> continuous state, and the belief filter is over a continuous magnitude rather
> than a 2-state posterior — the within-factor structure is no longer
> 1-dimensional. The rate `λ_i` (hence all of B2/B3's temporal content) is
> unchanged; only B1's filter geometry changes. Redraw is the convention that
> keeps the belief theory exact and 2-state, so we adopt it and flag the other.

**Load-bearing constants (per feature).** With `m_i = E[ReLU(μ_i+σ_iε)]`,
`s_i² = E[ReLU(μ_i+σ_iε)²]`, and the ReLU false-negative atom
`q_i = Φ(−μ_i/σ_i) = P(fired but rectified to 0)`:

    A_i := m_i² p_i(1−p_i)      persistent (nonstationary) readout-covariance amplitude
    ν_i := p_i (s_i² − m_i²)    = p_i·Var(r_i), the lag-0 observation noise
    Var(y_i) = A_i + ν_i,       y_i(t) := d_iᵀ(a_t − b)  (= c_i(t) when ρ_mm=0)

    ρ_i ≡ ρ_obs := ν_i / A_i = Var(r_i) / [ m_i² (1 − p_i) ] = CV_i² / (1 − p_i),

where `CV_i` is the coefficient of variation of the firing magnitude. `ρ_i` (=
`ρ_obs` in OPTIMAL_FRA_NOTE's guard) is the **observation noise-to-signal ratio**;
it is the single control parameter of the *linear* optimum in B2 (F3). *Notation
guard (F9): three distinct ρ's — `ρ_obs=ν/A` (this, noise-to-signal), `ρ_path`
(attention-path share, B3iii), `ρ_mm` (superposition overlap); `s_i²` is the ReLU
second moment, unrelated.* `[exact]`

The readout `y_i(t)` is a scalar **AR(1)-plus-noise** process:
`Cov(y_i(t),y_i(t−τ)) = A_i λ_i^τ` for `τ ≥ 1`, and `= A_i + ν_i` at `τ = 0`.
Everything factorizes across `i` at `ρ_mm = 0` (independent chains, orthogonal
readouts). One latent per GT feature, trivial (1-dim) within-factor structure.

---

## B1 — Belief theory

> **B1 (belief).** For the reset process with observable magnitudes (`ρ_mm = 0`),
> the exact Bayes one-step belief is the 2-state filter
>
> **P(z_i(t+1)=1 | a_{1:t}) = p_i + λ_i · (b_i(t) − p_i)**,  `b_i(t) := P(z_i(t)=1|a_{1:t})`,
>
> with `b_i(t)` given by the recursion below; in the clean-emission limit
> (`q_i → 0`) the current readout is a sufficient statistic and the filter uses
> **only the most recent observation**:
> `P(z_i(t+1)=1|a_{1:t}) = p_i + λ_i(z_i(t) − p_i)`.
> The **constrained (attention-parallel) belief update** is P2's additive form
>
> **r_i(d) = p_i + Σ_{s=1}^{d} λ_i^{d−s} (z_i(s) − p_i)**,   per-factor g-vector `g_i(z) = (z − p_i) û_i`, `û_i ∝ d_i`,
>
> and the gap between this geometric sum and the last-observation-only Bayes
> filter is the **parallelism tax**, with closed form (clean emission)
>
> **Var[tax] = p_i(1−p_i) · λ_i²(1+λ_i²) / (1−λ_i²)²** — the cleanest computable
> instance of the parallelism tax in the program.

### B1.1 The exact filter

Emission likelihoods for `y = y_i(t) = c_i(t)`: `L_0(y) = p(y|z=0) = δ(y)` (off ⇒
`y=0` deterministically); `L_1(y) = p(y|z=1) = q_i δ(y) + (1−q_i) f_+(y)`, `f_+` the
positive-part rectified-Gaussian density.

**Prediction step (the geometric decay).** `P(z(t)=1|z(t−1)) = λ_i z(t−1) + (1−λ_i)p_i`,
so the one-step-ahead prior is a convex mix of the previous posterior and the
prior:

    π_i^pred(t) = λ_i b_i(t−1) + (1−λ_i) p_i.        (relaxation toward p_i at rate λ_i)

**Update step (Bayes).**
`b_i(t) = π^pred L_1(y_t) / [π^pred L_1(y_t) + (1−π^pred) L_0(y_t)]`:
- `y_t > 0`: `L_0 = 0` ⇒ **`b_i(t) = 1`** (a firing is certain evidence of `z=1`).
- `y_t = 0`: `b_i(t) = π^pred q_i / (1 − π^pred(1−q_i))`.

Iterating the h-step transition `T_i^h = 1π_iᵀ + λ_i^h(I − 1π_iᵀ)` gives the exact
horizon-`h` belief `P(z_i(t+h)=1 | a_{1:t}) = p_i + λ_i^h (b_i(t) − p_i)`; the
belief decays geometrically toward the prior `p_i` at rate `λ_i^h`, as required.
`[exact]`

**The c_i = 0 ambiguity (ReLU zero/positive subtlety), stated as an assumption.**
The rectified Gaussian has an *atom* at 0 of mass `q_i = Φ(−μ_i/σ_i)`, so
`y_t = 0` is genuinely ambiguous (`z=0`, or `z=1` rectified to 0) with nonzero
probability. Two conventions:
- **(M+) observable-indicator** (`q_i → 0`, i.e. `μ_i ≫ σ_i`): `y_t = 0 ⟺ z_i(t)=0`
  a.s., so `b_i(t) = 1[y_t>0] = z_i(t)` — the indicator is directly read and the
  filter is Markov in the observed `z_i(t)`. Primary convention.
- **(M0) general** (`q_i > 0`): keep the atom; a `y=0` observation is a soft
  likelihood with false-negative rate `q_i`, and the filter (above) genuinely
  integrates history. Used in B2/B4 where it is what makes attention load-bearing.

Under M+ the per-source displacement is `±d_i`-aligned scaled by `(z_i(s) − p_i)`,
so `g_i(z) = (z − p_i) û_i` with `û_i` the belief-write unit vector (`∝ d_i` at the
aligned optimum). This is **P1's factored world with trivial within-factor
structure**: each factor's belief is a single scalar (2-state ⇒ 1-dim
nonstationary plane), vs Mess3's 2-dim simplex.

### B1.2 The parallelism tax (cleanest instance)

The exact one-step Bayes filter under M+ uses **only** the most recent observation
`z_i(d)` (Markov + full observation ⇒ `z_i(d)` sufficient): a hard
"last-writer-wins". The P2/attention-additive predictor is *forced* to be a fixed
linear combination of per-source terms — each source `s` knows only `(z_i(s), d−s)`
— and with the belief-propagation kernel `λ_i^{d−s}` it **sums** geometric
corrections. Take the belief about the *current* state `z_i(d)`: exact `= z_i(d)`
(observed); constrained `= z_i(d) + Σ_{s<d} λ_i^{d−s}(z_i(s)−p_i)`. The excess

    tax(d) = Σ_{s<d} λ_i^{d−s} (z_i(s) − p_i),   E[tax]=0,

    Var[tax] = p_i(1−p_i) Σ_{τ,τ'≥1} λ_i^{τ+τ'+|τ−τ'|}
             = p_i(1−p_i) · λ_i²(1+λ_i²)/(1−λ_i²)² .     [exact; verified to 0.1%]

(Derivation: `Cov(z(s),z(s')) = p(1−p)λ^{|s−s'|}`; the double sum
`Σ_{τ,τ'≥1} λ^{τ+τ'+|τ−τ'|} = Σ_{τ≥1}(2τ−1)λ^{2τ} = λ²(1+λ²)/(1−λ²)²`.) It grows
like `(1−λ)^{−2}` as `λ→1`: the closer to a frozen latent, the more the additive
kernel over-counts the autocorrelated past that the exact filter discards.

> **Interpretation and the honest subtlety.** The tax above is for the *canonical*
> P2 kernel `λ^{d−s}`. The truly-**optimal** additive kernel in the clean case is
> `α(τ) = δ_{τ,0}` (put all weight on the current observation), which *matches*
> Bayes and pays **zero** tax — i.e. the P2 geometric ansatz is suboptimal
> *for the fully-observed reset-clean process* (F6). This is not that P2 is wrong:
> P2's geometric kernel is optimal for its **partially-observed** Mess3 (the token
> never reveals the belief simplex); the reset-clean process is a *different,
> revealed-state* regime where the Bayes-optimal rule is last-writer-wins, not
> geometric integration. The genuinely irreducible tax (best additive/linear
> vs nonlinear Bayes) is nonzero only under noisy emission (`q_i>0` / magnitude
> variance), where the current token is no longer sufficient and multiple firings
> must be combined by the nonlinear "most-recent-firing screens off older ones"
> logic. That irreducible regime is exactly where B2's optimal attention becomes a
> nontrivial geometric with rate `η_i` (below). **The canonical kernel `λ^{d−s}` is
> the `ν→∞` (pure-noise) limit of B2's optimal kernel `η_i^{d−s}`.**

**Falsifying check.** Enumerate/simulate the reset chain; compare exact-filter
predictive CE (or MSE on `z(t+1)`) against the best additive-kernel predictor.
Prediction: (i) `Var[tax]` matches the closed form (✓, §Numerical checks);
(ii) in the clean limit the optimal additive kernel is `δ_{τ0}` and its tax → 0;
(iii) the canonical-`λ` kernel's tax is strictly positive and grows like
`(1−λ)^{−2}`. A single positive tax for the *optimal* additive kernel in the clean
case would falsify the claim.

---

## B2 — Optimal attention (multi-λ)

> **B2 (optimal profile & the η-vs-λ warp).** In P3's linear one-layer model, the
> reset process block-diagonalizes (at `ρ_mm=0`) into N independent scalar
> AR(1)+noise prediction problems, one per feature, coupled only through the shared
> attention profile `α(τ)`. Each per-feature problem is **exactly P3's two-state
> problem** with `b_eff = A_i+ν_i`, `m_eff = A_iλ_i`, giving the palindromic
> quadratic
>
> **ρ_i λ_i η² + [(λ_i²−1) − ρ_i(λ_i²+1)] η + ρ_i λ_i = 0**,  `ρ_i = ν_i/A_i`,
>
> whose in-disc root `η_i` is the optimal look-back rate. The **warp** is:
> `η_i → 0` as `ρ_i → 0` (clean emission ⇒ attention null), `η_i → λ_i` as
> `ρ_i → ∞` (pure-noise emission ⇒ belief kernel), monotone and `η_i < λ_i` for
> all finite noise. **`η_i = λ_i` is the exception, not the rule** — resolving the
> trained-rate-vs-belief-rate discrepancy. For N distinct eigenvalues the profile
> is `α(τ) = Σ_j c_j η_j^τ`, the in-disc roots of a degree-`2N` palindromic
> polynomial coupled *only* through the shared self-repulsion `q = Σ_i q_i`.

### B2.1 Spectral content and the M_λ block structure

The observation autocovariance is `Γ(τ) = E[(a_t−ā)(a_{t−τ}−ā)ᵀ] = Σ_i A_i λ_i^τ d_i d_iᵀ`
for `τ ≥ 1`. In P3's `B^{(τ)} = Σ_λ λ^{τ−1} M_λ` this reads (matching at `τ=1`,
`M_{λ_i} = Γ(1)|_i`)

    **M_{λ_i} = A_iλ_i d_i d_iᵀ**   (rank-1, supported on feature i's direction),

with nonstationary spectrum `{λ_i}` (multiplicity = # coinciding `λ_i`). The `τ=0`
term adds `Σ_i(A_i+ν_i) d_id_iᵀ` = the emission covariance `D` (P3's unigram term);
the observation noise `ν_i` is exactly the extra lag-0 mass with **no**
nonstationary content — it regularizes `S` and is what makes the attention
optimum nontrivial. Because `d_iᵀ D^{−1} d_j = δ_ij/(A_i+ν_i)`, the P3 cross terms
`Σ_{λ≠μ} M_μ D^{−1} M_λ` **vanish** (`μ≠ν` ⇒ orthogonal), so
`R = Σ_i R_i d_id_iᵀ`, `S = Σ_i S_i d_id_iᵀ`, `V_* = RS⁺ = Σ_i V_i d_id_iᵀ` all
block-diagonalize. `[exact at ρ_mm=0]`

### B2.2 Per-feature reduction = P3's two-state problem

Writing the single-feature scalar objective (P3's `L_attn`, α_0 dropped per P3's
`α_0`-independence) with `g := α̃'(λ_i)`:

    R_i = M g (λ − M/D),  S_i = γ_0 D + 2γ̃' M − g²M²/D,  L_attn^{(i)} = ½ R_i²/S_i,
    with  D = A_i+ν_i,  M = A_iλ_i.

This matches P3's two-state `L_attn` (eq. `lattn_stat`) **term-by-term** under
`b ↔ D = A_i+ν_i`, `m ↔ M = A_iλ_i`, `α̃'_λ ↔ g`. Hence P3's palindromic quadratic
`(bλ−m)η² + (2mλ−bλ²−b)η + (bλ−m) = 0` becomes, substituting
`bλ−m = ν_iλ_i` and `2mλ−bλ²−b = A_i(λ_i²−1) − ν_i(λ_i²+1)`:

    ν_iλ_i η² + [A_i(λ_i²−1) − ν_i(λ_i²+1)] η + ν_iλ_i = 0
  ⟺ **ρ_i λ_i η² + [(λ_i²−1) − ρ_i(λ_i²+1)] η + ρ_i λ_i = 0**,   ρ_i = ν_i/A_i.

Palindromic ⇒ reciprocal roots ⇒ exactly one in-disc root `η_i`, `|η_i|<1`,
`sign(η_i)=sign(λ_i)` (P3 App. B/C). **Numerically confirmed** that this in-disc
root equals the brute-force MSE-minimizing geometric attention rate to `<4×10⁻³`
across 16 `(λ, ρ)` settings (§Numerical checks) — the whole `(b_eff,m_eff)`
identification is verified, not assumed. **Independent confirmation (F1, theory-static):
the in-disc root equals the steady-state Kalman look-back pole `λ(1−K)` (`K` = Kalman
gain for `x(t)=λx(t−1)+w`, `Var(x)=A`; `y=x+v`, `Var(v)=ν`) to `1×10⁻¹⁶` on a 5×5
`(λ,ρ)` grid** — a route that never touches P3's machinery, tighter than the
brute-force check. `[exact given the linear model]`

> **Scope (F3).** The `ν↔`observation-noise mapping and "`ρ_i` is the control
> parameter" are exact for the **linear/second-order (MSE)** problem — which is
> exactly what B2 is. The true emission is a multiplicative-rectified
> (heteroscedastic) channel whose **nonlinear** Bayes filter (B1) does not reduce to
> AR(1)+Gaussian-noise; `ρ_i` controls the **linear optimum** `η_i`, not the Bayes
> filter (which keeps rate `λ_i`). B1 and B2 stay correctly separate.

### B2.3 The η-vs-λ warp (answers the standing confusion)

In-disc root, `ρ = ν/A`:

    η_i = { ρ(λ²+1) − (λ²−1) − √([ρ(λ²+1)−(λ²−1)]² − 4ρ²λ²) } / (2ρλ).

- **Clean emission `ρ_i → 0`** (deterministic magnitude, `CV_i → 0`):
  `η_i ≈ ρ_i λ_i / (1−λ_i²) → 0`. The optimal profile collapses to `α(τ) = δ_{τ0}`:
  the **look-back attention (lags ≥ 1) is null** — but this is a *redundant-signal*
  null, **not** an absent-signal null. The nonstationary signal `A_iλ_i^τ` is fully
  present; it is simply **captured by the current token** (Markov sufficiency; the
  steady-state Kalman gain `K→1`, so the look-back pole `λ_i(1−K)→0` while the
  **current-token gain `λ_i K → λ_i`**). The direct/skip path is therefore
  **maximally load-bearing** (optimal predictor `= λ_i·y_i(t)`, genuine
  loss-pinned computation), and only the look-back head is null.

  > **This is NOT Setting A's null (F4, per theory-static's review).** The two
  > limits are opposite mechanisms. **Setting A = `λ_i → 0`** (iid): no cross-time
  > signal (`A_iλ_i^τ → 0`), P3's `R = 0`, so **both** look-back **and** direct
  > path are null and the best predictor is the constant mean — everything is the
  > mean/`k=0` gauge. **B-clean = `ρ_i → 0` at `λ_i > 0`**: signal present but
  > redundant, look-back null **with a live, loss-pinned direct path**. Same
  > `η_i → 0`, opposite direct-path status. The genuine reduction of B to Setting A
  > is **`λ_i → 0`**, not `ρ_i → 0`. What B-clean newly shows is: *a persistent
  > latent whose look-back attention is nonetheless null because the current token
  > already reveals it* — a distinct, temporal phenomenon.
- **Pure-noise emission `ρ_i → ∞`**: `η_i → λ_i`. When the current token is
  uninformative, the optimal kernel is the belief-propagation kernel at the true
  rate `λ_i` — B1's canonical constrained update becomes optimal.
- **General**: `η_i < λ_i`, monotone increasing in `ρ_i`. The shrinkage
  `λ_i − η_i` is set by the **observation SNR**, not by the process rate.

> **Resolution of the trained-rate-vs-belief-rate discrepancy.** Sprint-2 measured
> the Mess3 CE-optimal rate `η = 0.464 < ζ = 0.55` at emission fidelity `α = 0.6`,
> and `η = 0.529 ≈ ζ` at `α = 0.2`. In the reset-process language: higher fidelity
> = cleaner observation = smaller `ρ` = **more** shrinkage (`η` further below `λ`);
> weaker fidelity = noisier = larger `ρ` = `η → λ`. The direction and the "shrinkage
> is second-order in evidence strength" claim both fall out of the formula, and the
> reset process gives the *first closed-form* prediction of the shrinkage:
> `η_i(ρ_i)` from the palindromic root. **Agrees with, and explains, sprint-2 — no
> contradiction.** *(F10: this is a **directional** analogy, not a quantitative
> identity — Mess3's emission is structurally always partial, so it sits at
> intermediate `ρ` and never reaches `ρ→0`; its `η` stays near `ζ` and does not go to
> 0. The reset process **can** reach `ρ→0`, a regime with no Mess3 counterpart.)*

### B2.4 N distinct eigenvalues: the degree-`2N` palindromic polynomial

One shared profile `α(τ)` serves all N features; the total objective is the sum
`L_attn = Σ_i L_attn^{(i)}(α)` (orthogonal features). By §B2.1 the P3 coefficients
`w_{λ_i}, e_{λ_i}` are **per-feature** (no cross terms), while the self-repulsion
`q = tr(V(D−pp ᵀ)Vᵀ) = Σ_i q_i` is a **sum**. P3's generating-function polynomial
is therefore

    P(z) = q · ∏_i (1−λ_i z)(z−λ_i)
         + Σ_i λ_i w_i (z²−2λ_i z+1) ∏_{j≠i} (1−λ_j z)(z−λ_j),   q = Σ_i q_i,

degree `2N`, palindromic (`P(z) = z^{2N}P(1/z)`), with `N` in-disc roots
`η_1,…,η_N` (none on the unit circle, Fejér–Riesz). The optimal profile is

    **α(τ) = Σ_{j=1}^N c_j η_j^τ**,  the `c_j` fixed by the `N` boundary conditions
    `N(η_j)=0` (two linear solves, P3 §self-consistency).

For **N = 2 distinct λ**: degree-4 palindromic

    P(z) = (q_1+q_2)(1−λ_1 z)(z−λ_1)(1−λ_2 z)(z−λ_2)
         + λ_1 w_1 (z²−2λ_1 z+1)(1−λ_2 z)(z−λ_2)
         + λ_2 w_2 (z²−2λ_2 z+1)(1−λ_1 z)(z−λ_1),

two in-disc roots `η_1, η_2`; `α(τ) = c_1 η_1^τ + c_2 η_2^τ`. **Coupling is *only*
through `q = q_1+q_2`** (shared self-repulsion): the two features push each other's
roots via a common "mass-conservation" pressure, but the driving/interaction terms
are per-feature. The roots are **not** the isolated per-feature `η_i(ρ_i)`; the
shared `q` shifts both. **Single-λ limit** (`A_2 → 0` or `λ_2 = λ_1`): the
`λ_2`-factor decouples, `η_1 →` the B2.2 quadratic root and `η_2 → λ_2` (passive
factor), reproducing P3's two-state formula. `[sketch; quartic roots not in closed
form — compute numerically, as sprint-2 did for the factored degree-6 case]`

### B2.5 Conic head count

With one softmax head all features see the **same** `α(τ) ≥ 0`. The per-feature
target rates `η_i` differ, so one head cannot serve them exactly through the QK
pattern alone. The resolution is P1's per-subspace routing: head `h` has a
nonnegative pattern `α^{(h)}(τ)` and OV routing `w_{hi} = d_iᵀ W_OV^{(h)} d_i`, and
the **effective subspace attention** for feature `i` is
`α_i^{eff}(τ) = Σ_h w_{hi} α^{(h)}(τ)` (P1; = FRA-OV aggregated to `d_i`). We need
`α_i^{eff}(τ) ∝ η_i^τ` for each `i`:

    [η_i^τ]_{i,τ}  =  W_{(N×H)} · [α^{(h)}(τ)]_{(H×L)},   α^{(h)} ≥ 0, W sign-free.

- **Exact optimality**: the target matrix has rank = #distinct `η_i` (Vandermonde),
  so `H_min ≥ #distinct η_i`; nonnegativity of the patterns can only raise it
  (nonneg-rank ≥ rank). Any `λ_i < 0` ⇒ `η_i < 0` ⇒ a sign-oscillating profile no
  single nonnegative head realizes ⇒ **≥2 heads** (even/odd parity split, P2 /
  sprint-2 two-head hinge), but the two parity rays are **reusable** across all
  features sharing a sign class (P1's ray reuse: `+0.5/−0.5` needs 2 heads, not 3).
- **ε-optimality (the real story)**: the per-feature rate shrinkage is **loss-flat**
  (sprint-2 §flat; a wrong rate costs `O((Δη)²)` below the CE gradient floor). So a
  *single* head at a compromise rate — the degree-`2N` root mixture, or even one
  dominant `η` — captures almost all recoverable information, and `H_min` for
  near-optimality is governed by the number of distinct **sign/phase classes** of
  `{λ_i}` (P1's conic count), not by the magnitude diversity.

> **Answer.** One positive head with a general nonnegative position-score profile
> realizes any mixture kernel `Σ_j c_j η_j^τ ≥ 0` and, because rate mismatch is
> loss-flat, is ε-optimal for **all** `λ_i > 0` (H_min = 1 for ε-optimality; the
> exact rank bound `H_min = #distinct η_i` is invisible to the loss). A negative
> eigenvalue forces a 2-head parity split, reusable across features. This is P1's
> `H_min` = # sign-classes, with the magnitude-diversity contribution demoted to
> the loss-flat tail.

**Falsifying check.** Train 1-head vs N-head linear/softmax attention on N-feature
reset data; predict (i) per-feature `α_i^{eff}` decays at `η_i(ρ_i)` from the
palindromic root (not `λ_i`); (ii) 1 head is within the loss-flat floor of N heads
for all-positive spectra; (iii) a negative `λ_i` plateaus 1-head loss above 2-head
(P1 `H_min` jump). A 1-head loss gap `≫` the flat floor for an all-positive
spectrum would falsify.

---

## B3 — The optimal FRA theorem (centerpiece)

> **B3.** At the optimum in the GT dictionary (`ρ_mm = 0`):
> **(i)** `FRA-OV(feature i, lag k) = η_i^k · c̄_i · ĝ_i` — all channel semantics in
> OV (`ĝ_i ∝ d_i`), all lag structure in the trained rate `η_i^k` (softmax warp
> `/(1−η_i^d)`); the belief-kernel reading `λ_i^k` is the `ρ_i→∞` idealization.
> **(ii)** The gauge-invariant content of FRA-QK is the **positional kernel(s)
> alone**; every content×content score coupling is gauge (imported QK gauge
> theorem, premise **exact** here).
> **(iii)** Severing feature `i`'s OV channel at gain `c` removes fraction
> `c·ρ_path` of its belief carryover with **zero** collateral on `j≠i` at `ρ_mm=0`
> and collateral `∝ (d_iᵀd_j)²` at leading order in superposition **in the matched
> GT dictionary** (F7; degrades to `O(ρ_mm)` under a learned χ); null at
> `c*_i = 1/ρ_path`, `ρ_path = 1 − (1−a_0(d))(1−η_i)` is the **attention-path share**
> (distinct from the noise-to-signal `ρ_obs = ρ_i = ν_i/A_i`). This is the cleanest
> use/collateral separation in the program: ratio `O(1)/O(ρ_mm²) → ∞` as `ρ_mm → 0`.

### B3(i) FRA-OV closed form

The transported content resolved by (source feature, lag): at the optimum
`W_OV(a_s − b)` maps feature `i`'s content `c_i(s)` onto its belief-write direction
`ĝ_i ∝ d_i` (orthogonal dictionary ⇒ no cross-write), and the per-feature effective
attention is `α_i(k) = η_i^k` (B2). Hence, per (feature `i`, lag `k`):

    **A_{d,s} R W_OV(a_s−b)|_i = η_i^{d−s} · ĝ_i · (content scale)**,

with the softmax normalization warp (sprint-2 §warp) making the realizable form
`η_i^{d−s}/(1−η_i^d) · ĝ_i`. This is P1's **effective subspace attention** in the
SAE basis whose token latents align with `ĝ_i`, and by the sprint-2 bridge lemma it
is P3's nonstationary regression image `M_{λ_i} = A_iλ_i d_id_iᵀ` transported. `[exact
given the aligned-dictionary ansatz]`

> **Rate honesty (η vs λ).** The *transported content* decays at the trained rate
> `η_i` (B2), not the belief rate `λ_i`; `η_i = λ_i` only in the pure-noise limit.
> The spec's `λ_i^k` is the belief-propagation idealization; the exact FRA-OV
> closed form uses `η_i^k`, and `η_i^k/(1−η_i^d)` after the softmax warp. This is
> exactly sprint-2's warp×shrinkage (`ρ_eff > 1`, rate `0.464<ζ`), now derived from
> the reset spectrum rather than measured.

### B3(ii) FRA-QK is the positional kernel only

Import sprint-2's QK gauge theorem (Prop. "feat×feat mass is pure gauge"): under a
**token/content-independent pattern**, the tok×tok score block
`Q(z,z') = q_E(z)·k_E(z')` is additively separable and consists entirely of gauge
coordinates (pedestal `β(d)`, key-read `b(z)`, profile-share `θ`, row constants);
the centered interaction `q̂_E(z)·k̂_E(z')` is gauge-invariant and **zero** at the
optimum.

**Premise check — and a strengthening.** The premise is Conjecture Z1
(token-independence of the optimal pattern). For the reset process in P3's **linear**
model this is not a conjecture: the optimal one-step predictor of a stationary
process is the **LTI Wiener filter**, whose coefficients depend only on the
autocovariance `{A_i,ν_i,λ_i}`, **never on the realized feature values**. The
features are exchangeable in structure (independent identical-form reset chains), so
the score has no loss-reason to prefer one feature's content over another — the
content×content block is permutation-symmetric and gauge. Hence:

> the FRA-QK gauge theorem's premise holds **exactly** in the reset process's
> linear model (a strengthening over Mess3, where it is an empirical ansatz with
> measured content leak `ε_{C1}=0.70`). In the softmax/CE instantiation it again
> holds only on average (the CE model content-gates at `O(ε)`), so the content
> sector of a QK cut is loss-flat. `[exact in linear model; O(ε) in softmax]`
>
> **Verification-methodology caveat (F8, per theory-static's VERIFY_A).** "Loss-flat"
> does **not** mean "seed noise." Loss-flat objects are reproducibly
> **optimizer-pinned** (VERIFY_A measured cross-seed FRA correlation ≈ 0.99, not 0),
> because SGD from a fixed init/optimizer lands on the *same* gauge representative
> every seed. The correct test that the content sector is gauge is therefore
> **clean-representative-vs-trained at equal loss** (a construction with zero content
> attributions matches the trained loss), **not** cross-seed incoherence. A B-check
> must also **separate the loss-pinned direct path** (real computation, seed-coherent
> *because* pinned) **from the loss-flat look-back attention** (seed-coherent
> *because* optimizer-pinned) — same observable, opposite cause.

Every content×content score coupling is gauge; the only gauge-invariant FRA-QK
content is the positional/lag kernel `α(τ)` itself.

### B3(iii) Intervention calculus — the cleanest use/collateral split

**Zero collateral at `ρ_mm=0`.** `W_OV` maps `c_i(s)d_i → ĝ_i ∝ d_i`. Severing
feature `i`'s OV channel zeros the `d_i → d_i` transport; since `d_i ⊥ d_j`, feature
`j`'s carryover (along `d_j`) is untouched. **Exactly zero collateral.** Unlike
single-Mess3 (three coplanar `g(z)` summing to zero ⇒ *forced* collateral,
sprint-2 §collateral), the reset process gives N **orthogonal** channels — it is
P1's factored world, not Mess3's coplanar one. Selective concept removal is exact.

**Removed fraction and the null.** Severing `z^*`'s OV across sources edits `c_d`
only (pattern frozen; linear in `c`), skip survives (sprint-2 §ovint, k=0 split):

    r_i'(d;c) = r_i(d) − c [ Σ_{s<d, z_s=z^*} η_i^{d−s} + a_0(d)·1[z_d=z^*] ] ĝ_i ,

with `a_0(d)` the pure-gauge diagonal share of the lag-0 term (P1 Eq. 28 / sprint-2
§k0). The **attention-path share** of feature `i`'s belief carryover is (named
`ρ_path`, matching OPTIMAL_FRA_NOTE's guard, to avoid colliding with the
noise-to-signal `ρ_obs=ν_i/A_i` and the ReLU moment `s_i²`, F9)

    **ρ_path = 1 − (1−a_0(d))(1−η_i)**,   **c*_i = 1/ρ_path** (the concept-nulling gain),

interpolating `a_0=1` ⇒ `ρ_path=1, c*=1` (exact severing nulls) to `a_0=0` ⇒
`ρ_path=η_i, c*=1/η_i` (over-inject). `c*` is a property of the run (its position on
the k=0 gauge orbit), not the task — sprint-2's `c*=1/ρ` law (their `ρ` = this
`ρ_path`), closed-form here.

**Collateral at `ρ_mm>0` — the first-order cancellation (F7).** Aligned GT code
`o_k = γ d_k`. *Before* severing, feature `j`'s belief already carries a
**first-order** contamination `+γ G_{ij} C_i` (`G=DDᵀ`, `C_i` = feature-i content).
Severing feature `i` via its `d_i`-readout removes `γ Ĉ_i d_i` with
`Ĉ_i = C_i + Σ_{k≠i} G_{ik} C_k`, so feature `j` sees
`−γ G_{ij} Ĉ_i = −γ G_{ij} C_i − γ G_{ij} Σ_{k≠i} G_{ik} C_k`. **The first-order
terms cancel** (`+γG_{ij}C_i` from j's pre-existing contamination vs `−γG_{ij}C_i`
from removing i's true write), leaving net `−γ G_{ij} Σ_{k≠i} G_{ik} C_k`, whose
leading (`k=j`) term is `−γ G_{ij}² C_j = O(ρ_mm²)`. **This cancellation — not a bare
"read×write" product — is the actual reason the collateral is second-order.**
`[exact to O(ρ_mm²); mechanism = first-order cancellation]`

> **Contingency (connects to B4c / A4).** The cancellation requires the sever's
> encoder = decoder = `d_i` at unit gain — a **matched/complete/centered** code. With
> a learned χ (encoder ≠ decoder, or non-unit gain), the first-order cancellation
> **breaks** and collateral reverts to `O(ρ_mm)`. So the unbounded `O(ρ_mm^{−2})`
> separation holds **only in the GT dictionary**; a real SAE gives `O(ρ_mm^{−1})`.
> (Relation to static A3.4: the belief itself is `O(ρ_mm)`-contaminated; the *change
> under a matched cut* is `O(ρ_mm²)` — different objects, consistent.)

> **The quantitative separation.** Use (feature `i` removal) is `O(1)` — a fraction
> `ρ_path` nulled at `c*=1/ρ_path`; collateral (on `j≠i`) is `O((d_iᵀd_j)²)=O(ρ_mm²)`
> **in the GT dictionary**. The use/collateral ratio is `O(ρ_mm^{−2}) → ∞` as
> `ρ_mm → 0` (exactly ∞ at `ρ_mm=0`). This is the **cleanest use/collateral
> separation available anywhere in the program**: single-Mess3 has `O(1)` forced
> collateral (coplanar); the fra_win induction win is ~15×; the reset process is
> unbounded (`ρ_mm^{−2}`) with the matched code, `O(ρ_mm^{−1})` with a learned χ.

**Falsifying check.** Planted-GT SAE on reset-process residuals; sever feature `i`
at gains `c`. Predict (i) feature-`i` tracking `= 1 − c·ρ_i`, null at `1/ρ_i`
(linear, `R²=1` frozen pattern); (ii) feature-`j` tracking `= 1.00 ± O(ρ_mm²)` at
`ρ_mm=0`, drifting `∝(d_iᵀd_j)²`; (iii) `ρ_i` predicted ex ante from the k=0 split
and `η_i`. Any nonzero collateral on `j` at `ρ_mm=0` (with a complete GT dictionary)
would falsify.

---

## B4 — What breaks first (seeds the next round)

### B4(a) Copula correlations `Σ ≠ I` — the joint filter no longer factorizes

The Gaussian-copula firing correlations make `z_i, z_j` dependent, so the joint
reset process on `(z_i,z_j)` has a coupled transition and the per-feature belief
`P(z_i(t)|history)` now depends on feature `j`'s observations (observing `z_j`
informs `z_i` through `Σ_{ij}`). **First correction (pairwise):** the per-feature
belief acquires a cross term

    r_i(d) ⊃ Σ_{j≠i} β_{ij} · (z_j(s) − p_j),   β_{ij} ∝ Corr(z_i,z_j) = f(Σ_{ij}),

i.e. feature `j`'s content leaks into feature `i`'s belief update. This is an
**OV-side** cross-coupling (it is about how `j`'s *observation* shifts `i`'s
belief), present **even at `ρ_mm=0`** — so correlations break the zero-collateral
property of B3(iii): a cut of feature `i` now perturbs `j` through the correlation
channel `∝ Σ_{ij}`, a *new* collateral term independent of superposition. The QK
pattern stays content-independent/gauge (the covariance is stationary ⇒ still an
LTI filter), so **the first break is in FRA-OV cross-terms, not QK.** `[sketch]`

### B4(b) Hierarchy (parent-gating) — the first loss-pinned QK content coupling

Child `z_c` fires only if parent `z_p` fired (`c_c ← c_c·1[c_p>0]`). This is the
first **content-gated** structure. Predicting the child now *requires* knowing the
parent's state: if the parent is off, the child is certainly off, regardless of the
child's own history. So attending to **parent-presence** is informative for the
child — the optimal pattern becomes **content-dependent**, and FRA-QK acquires a
real, gauge-invariant, loss-pinned coupling:

    score(d,s) ⊃ (child-query at d) · (parent-key at s)  — a rank-1 tok×tok term
                 ∝ 1[predict child at d] · 1[parent fired at s],

which is **not** additively separable and **not** gauge (it changes the loss),
because the pattern must up-weight positions where the parent fired to gate the
child belief. This is the reset-process analog of induction / the fra_win
content-gated regime: **hierarchy is the minimal setting where FRA-QK has a genuine
handle.** `[sketch; open: whether the exact gate *must* live in QK or is
approximable in OV — this is precisely the fra_win boundary the next round should
resolve.]`

### B4(c) Learned dictionaries — interface to the χ formalism (owned by the static agent)

All B3 closed forms hold in the GT dictionary. With a learned SAE dictionary the
content channels remix, `ĝ_i → χ·g̃`, exactly the sprint-2 §dict / A4 mixing matrix
`χ`. Key interface fact: **the temporal rates `η_i` are process/spectrum
properties, unchanged by the dictionary** — the dictionary only remixes the OV
channel semantics (completeness/centering of the code is the binding constraint,
as in sprint-2's missing-channel result), never the rates. So the learned-dictionary
correction to B3 factors as (rate `η_i`, exact) × (channel `χ`, from the SAE). The
`χ` derivation — MCC/uniqueness/hedging/absorption → mixing matrix, and what
recovery is *sufficient* for faithful FRA edits — is A4's; B hands off `χ` and keeps
`η_i`. `[interface]`

---

## Numerical checks (run — `scratchpad/verify_b2.py`)

1. **Reset autocovariance**: `Cov(z(t),z(t+τ)) = λ^τ p(1−p)` — sim matches to 0.1%.
2. **Palindromic root = optimal attention rate** (the load-bearing check for the
   `b_eff=A+ν, m_eff=Aλ` mapping): the in-disc root of
   `ρλη²+[(λ²−1)−ρ(λ²+1)]η+ρλ=0` equals the brute-force MSE-minimizing geometric
   attention rate to `<4×10⁻³` across `λ∈{0.3,0.55,0.8,0.95}` × 4 noise settings
   (all 16 pass).
3. **η-vs-λ warp / limits** (`λ=0.7`): `ρ=10⁻³→η=0.0014`, `ρ=1→η=0.408`,
   `ρ=10⁴→η=0.700=λ`; monotone, `η<λ` throughout.
4. **Parallelism tax**: `Var[tax] = p(1−p)λ²(1+λ²)/(1−λ²)²` matches simulation to
   0.1–0.2% at `λ∈{0.3,0.55,0.8}`.

**Not yet run (next-wave, all CPU-minutes):** N=2 degree-4 quartic root solve +
train a 2-feature linear/softmax head and confirm `α_i^{eff}` decays at `η_i`;
planted-GT SAE severing (B3(iii) collateral ladder); copula/hierarchy first-break
signatures (B4).

## One-line reconciliation with the prior notes

Everything **agrees** with and **sharpens** sprint-2/P1/P2/P3: the reset process is
P1's factored world with 2-state (scalar) factors; the QK gauge theorem holds
*exactly* (LTI-Wiener premise) rather than as an ansatz; the FRA-OV closed form is
P1 effective subspace attention with rate `η_i`; and the `η=0.464<ζ=0.55` shrinkage
sprint-2 *measured* is here *derived* as the palindromic-root warp `η_i(ρ_i)`, with
the noise-to-signal `ρ_i` the control. No contradiction found. The one genuinely new
structural claim is the **clean-emission ⇒ look-back-attention-null** boundary
(`η_i → 0` as `ρ_i → 0`): a persistent latent whose look-back attention is null
because the current token already reveals it, with the direct path still
loss-pinned. This is a *redundant-signal* null and is **distinct** from Setting A's
`λ_i → 0` *absent-signal* null (where the direct path is null too); the genuine
reduction to Setting A is `λ_i → 0`, not `ρ_i → 0`.
