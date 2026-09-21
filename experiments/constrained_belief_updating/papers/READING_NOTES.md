# Reading notes: the three theory notes (P1, P2, P3) and the FRA bridge

*Written 2026-07-15 after a full read of all three. Local sources:*

| # | Paper | Local full text |
|---|---|---|
| P1 | "When Is an Attention Head a Computational Unit? Shared Circuits Despite Orthogonal Representations" — Christensen, Amdahl-Culleton, Ray, Tahir, Riechers, Shai. ICML 2026 Mech Interp Workshop (OpenReview `QVcsBMonuz`; not on arXiv). | `P1_attention_head_unit/P1_attention_head_unit_QVcsBMonuz.pdf` + `P1_fulltext.txt` (pdftotext) |
| P2 | "Constrained Belief Updates Explain Geometric Structures in Transformer Representations" — Piotrowski, Riechers, Filan, Shai. ICML 2025 poster (OpenReview `f6Hl60FBFU`, arXiv:2502.01954v2). | `P2_constrained_belief_updates/src/Updated_ICML.tex` (+ figures) |
| P3 | "Analysis of a one layer transformer" — Javan Tahir, 2026-03-23 (private note). | `javan_theory/main.tex` |

Related lineage: Shai et al. 2024 "Transformers represent belief state geometry in their
residual stream" (NeurIPS); Shai et al. 2026 "Transformers learn factored representations"
(arXiv:2602.02385) = the Factored World Hypothesis paper P1 builds on.

---

## P2 — Constrained belief updates (the mechanism)

Setting: 1-layer softmax transformer (d_model 64, CE loss, no BOS, learned pos-emb),
trained on Mess3 (3 hidden states, 3 tokens; params α emission-fidelity, x transition;
ζ = 1−3x the doubly-degenerate nonstationary eigenvalue).

- Full Bayes: η' = ηT^{(z)} / ηT^{(z)}1 — inherently **recursive** (Eq. full-belief).
- Attention is **parallel**: c_d = Σ_{s≤d} A_{d,s} v_s. Each v_s knows only (token z_s, lag d−s),
  never the intervening tokens.
- Best parallel approximation = **constrained belief update**:
  r₁^{(z_{1:d})} = π + Σ_{s=1}^{d} ( π T^{|z_s} T^{d−s} − π ).   (their Eq. 5 / eq:constrained-belief)
  Each term = Pr(S_d | Z_s=z_s) − Pr(S_d): independent evidence from source s propagated
  forward d−s steps through the marginal latent dynamics.
- Spectral form: r₁ = π + Σ_s Σ_{λ≠1} λ^{d−s} π T^{|z_s} T_λ. All lag-dependence sits in λ^{d−s}.
- Predictions verified in trained models:
  - A_{d+m,s} = ζ^m A_{d,s} (exponential in lag, token-independent pattern),
  - OV vectors f(v_s) ∝ (π T^{|z_s} − π): all OV vectors of the same token are parallel;
    magnitude inversely proportional to A_{m+1,m},
  - embeddings parallel to OV vectors (scalar discrepancy on first two positions, unexplained),
  - **negative ζ (x>1/3) cannot be done by one head** (attention ≥ 0): two heads with
    anti-parallel OV split the even/odd lags. A_{d+2m,s}^{(h)} = ζ^{2m} A_{d,s}^{(h)}.
- Post-MLP: nonlinear warp from the constrained fractal to the full belief geometry.
- Multi-layer (4L): layer-0 attention always implements the constrained update; later
  layers refine toward full Bayes. (MSE regressions, appendix.)
- Constrained-update prediction is accurate for α ∈ [0.2,0.6]; deviations grow outside.

## P1 — Effective subspace attention (the unit of computation)

Setting: 1-layer pre-norm decoder-only transformers on **two-factor Mess3 products**
T^{(x)} = T₁^{(z⁽¹⁾)} ⊗ T₂^{(z⁽²⁾)} (9 latent states, 9 tokens, seq len 11 w/ BOS,
d_model 120, 100k steps). Factors live in ~orthogonal 2-d residual subspaces V_{F_n}
(vary-one identification; 97.2% var explained). Per-factor constrained update:

  Δr_n^{(x_{1:d})} = Σ_{s=1}^d ζ_n^{d−s} · g_n(z_s^{(n)}),   g_n(z) = (π_n T_n^{|z} − π_n) P_{ζ,n}.

- **Division of labor (theory, verified):** OV carries factor identity (routes token-dependent
  displacement g_n into subspace n); QK carries **temporal structure only** (the scalar ζ_n^{d−s}).
  Causal shuffles confirm: token-identity shuffles of the QK inputs don't hurt loss;
  lag-breaking position shuffles do.
- **Only the aggregate is constrained**: Σ_h α_n^{(h)}(d,s) = ζ_n^{d−s}. Head-level
  decomposition is free → specialization, collaboration, polysemanticity all occur
  (depends on spectrum, head budget, init scale, training dynamics).
- **Effective subspace attention** (the paper's central tool — this is "FRA aggregated
  to a known subspace"):
  α_n(d,s) = ⟨ f_n(Σ_h A_{d,s}^{(h)} v_s^{(h)}), g_n(z_s^{(n)}) ⟩ / ||g_n(z_s^{(n)})||².
  Recovers ζ_n^{d−s} even when every individual head's attention map is illegible
  (mixed-sign case: both heads write ~parallel into the +ζ subspace and ~anti-parallel
  into the −ζ subspace).
- **Conic minimum head count**: H_min = min #heads whose non-negative cone contains the
  lag-update set {(ζ₁^k, ζ₂^k) : k≥0}. Negative eigenvalue → 2 rays; rays can be *reused*
  across factors (e.g. +0.5/−0.5 needs 2 heads total, not 3). Verified: loss plateaus at H_min.
- **k=0 degree of freedom** (App. D): the current-token displacement can be delivered by the
  skip connection OR diagonal attention — only the sum is constrained (Eq. 28). One-hot-frozen
  embeddings force it into diagonal attention.
- Rich/lazy (App. G): small init (0.02) → QK learns correct spectral structure; TL-default
  init → sharp random patterns lock in, loss worse (up to 6.8× on identical-negative config).

## P3 — Javan's one-layer MSE theory (the optimization story)

Setting: 1-layer attention-only, **MSE loss**, attention A treated as a free row-stochastic
lower-triangular matrix (NOT QK-parameterized). Process: stationary edge-emitting HMM.

- For fixed A the problem is linear regression on Z = [X; XAᵀ]. Optimal effective params:
  W* = B⁽¹⁾D⁻¹ − V*CᵀD⁻¹ (bigram minus redundancy correction), V* = RS⁺ where
  R = C₊ − B⁽¹⁾D⁻¹C, S = G − CᵀD⁻¹C. **V* = partial regression coefficient**: maps the
  attended-context residual (after regressing out current token) onto next-token residual.
- All statistics reduce to skip-bigrams B^{(τ)} = Σ_λ λ^{τ−1} M_λ ((M_λ)_ij = π Tʲ P_λ Tⁱ 1)
  contracted against two attention summaries: mean-lag α(τ) and autocorrelation γ(τ),
  entering only through generating functions α̃'(λ), γ̃'(λ) at nonstationary eigenvalues.
- **R lives entirely in the nonstationary token subspace** → V* too (gauge: W→W+u1ᵀ, V→V−u1ᵀ).
- Factorization W_U (rows 1ᵀT^{(i)ᵀ}), W_E (π1ᵀ + nonstationary corrections), W_OV = FW_E⁺+N:
  the simplex-embedding interpretation; existence requires |S| ≥ |V| (state space ≥ vocab).
- Optimal attention: loss(A) = unigram − bigram − ½tr(RS⁺Rᵀ); only the skip-bigram term
  depends on A, through ψ = (α(0), γ(0), {α̃'(λ), γ̃'(λ)}_{λ≠1}). Lifted objective is convex
  and lower-semicontinuous → **Toeplitz optimality as T→∞** (Jensen argument + explicit
  approximately-Toeplitz feasible sequence). α(0) drops out entirely.
- First-order conditions = force-balance on α(τ) particles: driving term Σ e_λ λ^{τ−1},
  self-repulsion qα_τ, screened pairwise interaction. Generating-function solution:
  A(z) rational; palindromic denominator P(z) (roots in reciprocal pairs, none on unit circle);
  poles inside disc must be cancelled → boundary conditions pin coefficients →
  **α_τ = Σ_j c_j η_j^τ, a sum of geometrics whose rates η_j are P(z)'s in-disc roots**.
- Two-state worked example: palindromic quadratic (bλ−m)η² + (2mλ−bλ²−b)η + (bλ−m) = 0;
  appendices prove roots real and sign(η) = sign(λ); closed-form η.
- Negative/complex eigenvalues: sublattice-support ansatz (numerically supported, unproven).
- **Stubs / open in the note**: "Limiting Case I: Cryptic Modes" and "Limiting Case II:
  Markov Emissions" are empty section headers; multi-head is absent; softmax/QK is absent;
  CE-vs-MSE gap acknowledged implicitly.

---

## The FRA bridge (why these three notes give "a full theory of how FRA works")

FRA decomposes a head's attention scores (QK) and transported content (OV) through an SAE
feature basis: score(d,s) = Σ_{ij} (feature i at query) × (feature j at key) bilinear terms;
attn-out(d) = Σ_j pattern·(feature j content at s → OV). The fra_hmm_toy result
(`experiments/fra_hmm_toy/PEDAGOGICAL_NOTE.md`) found empirically, on a mixture-of-Mess3
concept-removal task: FRA-QK cuts are inert everywhere; FRA-OV exact severing (c=1)
under-reaches (~¼ of signal); gain-tuned c*≈4 nulls behavior while leaving the concept
fully decodable (presence R²≈0.99); SAE content cuts are the only presence-reducers;
an information-theoretic "entanglement tax" bounds ideal removal.

The theory triangle makes each of these derivable rather than observed:

1. **FRA-QK inertness = theorem candidate.** P2/P1 prove the optimal first-layer pattern
   for stationary HMM data is *token-independent* (lag-only: A_{d,s} ≈ ζ^{d−s}; P3: optimal
   A is Toeplitz in T→∞, determined only by the spectrum). If the optimal pattern carries
   no content, feature-resolved score mass is pattern-shaped, not concept-causal, and
   severing feature×feature score terms cannot remove concept use. The fra_win induction
   regime (content-gated pattern) is exactly the case where the data demands a
   content-dependent A — outside this class. **The QK-wins/QK-inert boundary should be
   restatable as a property of the data-generating process** (stationary-HMM-like ⇒ lag-only
   optimal pattern ⇒ FRA-QK has nothing to grab).
2. **FRA-OV = effective subspace attention in an SAE basis.** P1's α_n(d,s) is FRA-OV
   aggregated to a factor subspace with *known* ideal features g_n(z). An SAE trained on
   these residual streams should recover {g_n(z)} (+ prior/BOS directions) as latents;
   FRA-OV attributions through those latents have closed form ζ_n^{d−s}. Head-level FRA
   is gauge-dependent (P1's freedom); subspace-level FRA is invariant. This predicts when
   per-head FRA rankings are meaningful at all.
3. **The gain-tuned null c\* is a path-share calculation.** P1 Eq. 25–28: a factor's update
   arrives via skip connection + diagonal attention + non-local attention, with an explicit
   shared degree of freedom at k=0. Severing one path (FRA-OV c=1) removes only that path's
   share; the counter-injection gain needed to null total flow is c* ≈ 1/(path share).
   fra_hmm_toy: block-1 OV carries ~¼ of ω-signal, c*≈4. This should be predictable
   ex ante from the theory.
4. **The mixture concept = the λ→1 limit.** fra_hmm_toy's ω (per-sequence Dirichlet mixture)
   is a latent with *no decay*: in HMM terms a near-unity eigenvalue block (non-ergodic
   mixture of components). P3's machinery: α̃'(λ→1) → the optimal profile flattens
   (counting); the concept is carried by *aggregated content*, invisible to any lag-shaped
   pattern edit. The "aggregation regime" of the FRA boundary map = the λ≈1 spectral regime.
   (P3's empty "Cryptic Modes" section is likely intended for adjacent territory.)
5. **Known gaps the sprint must respect**: P3 has no QK parameterization, no softmax, no
   multi-head, MSE not CE; P2/P1 have softmax+CE but the theory is an ansatz verified
   empirically, not derived from the loss; the mixture/nonstationary case (ω) is in none
   of the three (P2 assumes stationarity; P1 factors are independent *stationary* Mess3's).

## Immediate open problems (candidate sprint targets)

- Derive the FRA-QK score decomposition for the P2 one-layer softmax model at its
  theoretically-predicted weights (A = ζ^{d−s}, OV = g(z), embeddings ∝ OV) in an
  idealized feature basis (features = token displacement directions + prior). Show the
  feature×feature mass is rank-deficient in content (pattern-gauge) and compute what any
  QK cut does to the constrained update → predict RF≈0 analytically.
- Do the same for FRA-OV: closed-form attribution = ζ^{d−s} per (feature, lag); compute
  severing vs gain-scaling effects on the constrained belief loss; derive c* from path shares
  including the skip/diagonal redundancy (P1 Eq. 28).
- Extend to the mixture (ω) case: add a per-sequence latent (Dirichlet mixture of factors
  or a block-stationary HMM with λ=1+ε block); show the optimal pattern flattens and the
  concept moves into marginal content; derive the use/presence split and the entanglement
  tax inside the same formalism.
- Trained-SAE step: verify a TopK SAE on the toy recovers {g_n(z)} (up to gauge) and that
  empirical FRA matches the closed forms to float precision (the fra_hmm_toy pipeline
  already verifies decomposition exactness ≤3e-5 — reuse it).
- P3-side (optional, harder): re-derive P3's optimal-attention story with QK parameterization
  (softmax constraint) to close the gap between "A as free profile" and FRA-QK's bilinear
  form; or at minimum work the two-state example with explicit QK weights.
