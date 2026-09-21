# VERIFY_A — numerical verification of the Setting-A falsifiers

*CPU-only, `.venv` (torch 2.12, numpy 2.4). Code in `code/` (`data.py`,
`model_A.py`, `check{1..5}_*.py`, `plot_verify.py`); results in `out/*.json`;
figure `out/verify_A_summary.png`. Every check trains/evaluates by exact linear
algebra or a small 1-layer softmax attention model (no LayerNorm, so the A2 FRA
algebra is exact: W_QK = Wq Wk^T / sqrt(d_h)). Model file is a private copy
(`model_A.py`) to avoid racing the Setting-B agent's edits to `model.py`.*

**Headline: 4 of 5 propositions confirmed (three to float precision); ONE — the
*characterization* in Corollary A1.a — is FALSIFIED and corrected. The core A1
theorem (attention is loss-flat) is confirmed; what is wrong is my claim that this
shows up as *seed-incoherent* attributions. It does the opposite: attributions are
seed-COHERENT (cross-seed corr 0.99) because SGD reproducibly pins one non-trivial
gauge representative. "Seed noise" → "loss-flat but optimizer-pinned."**

---

## Verdict table

| # | Proposition | Predicted | Measured | Verdict |
|---|---|---|---|---|
| 1 | **A1** attention loss-flat | clean rep (W=V=0,β=μ) hits floor; trained hits floor | floor 1.3969; clean **1.3951** (0.999×), \|Q\|=\|O\|=**0**; trained **1.4048** (1.006×), \|Q\|=0.29 | **CONFIRMED** |
| 1 | **A1.a** attributions are seed noise (incoherent, corr≈0) | cross-seed FRA corr ≈ 0 | cross-seed Q-corr **0.994**, O-corr 0.992 (metric control on independent randoms: −0.03) | **FALSIFIED → corrected** (optimizer-pinned, not seed noise) |
| 1 | **A2.3** constant is path-distributable (k=0 gauge) | β+(W+V)μ = μ | relerr **0.023**; zero-V-only loss 7.46 vs zero-W-only 1.40 (V cancels W) | **CONFIRMED** |
| 2 | **A1.3** no-bias ⇒ rank-1 mean-carrying V* | V* ∝ μμᵀ; trained V aligns to μ | analytic R rank-1 relerr **5e-13**, s_μ 0.991(<1), cos(μ)=1.0 both sides; trained V_left·μ̂ **0.998**, Vμ delivers μ (‖Vμ‖ 3.52≈‖μ‖ 3.47) | **CONFIRMED** (caveat below) |
| 3 | **A1.4** LMMSE readout T=Σ_c D C_a⁺ Dᵀ | T=I for N≤d; shrink+contam for N>d; MF=Gc | T−I **1.4e-12** at N=d=24 (ρ=0.46); T_ii=**d/N** exactly (0.500 @N=48, 0.300 @N=80); MF=Gc to **2e-16**; contam⟂Gram corr −0.74/−0.85 | **CONFIRMED** + bonus law T_ii=min(1,d/N) |
| 4 | **A3** dilution: QK 1/m², OV 1/m; group-cut m-invariant | exact scalings | OV/(1/m)=**1.000**, QK/(1/m²)=**1.000** ∀m∈{1,2,4,8,16}; group-cut relerr **2.7e-15** | **CONFIRMED** (float precision) |
| 5 | **A4** MCC cannot certify severability | complete severs all; absence (high MCC) severs none | complete MCC **1.0**, severs {0,1,2}; absence MCC **0.995**, severs **{}** (feat-1 residual 0.99) | **CONFIRMED** |

---

## Check 1 — A1 / Corollary A1.a (the falsification, in full)

`out/check1_seednoise.json`. N=12, d=48, T=16, 6 seeds, 3000 steps, with bias.

**Confirmed (the theorem).** The analytic affine floor is tr(Cov a)=1.3969. The
**clean gauge representative** β=μ_a, W=V=0, attn=0 achieves loss **1.3951**
(0.9987×floor) with **identically zero** FRA-QK and FRA-OV attributions. A trained
model achieves **1.4048** (1.006×floor) with large attributions (mean |Q|=0.29).
Two loss-equal points with wildly different attributions ⇒ **attention is loss-flat;
FRA attribution is off the loss-relevant manifold.** Loss is seed-stable (std 0.003).

**Falsified (the characterization).** I predicted A1.a manifests as *seed-incoherent*
attributions (cross-seed corr ≈ 0). Measured cross-seed **Q-corr 0.994, O-corr 0.992**
— nearly identical across seeds. (Metric is sound: two *independent random* attention
maps give Q-corr −0.03, so 0.99 is real coherence, not an artifact.) SGD does **not**
leave attention at small init: trained ‖Wq‖≈8.6, ‖V‖≈1.4, and β−μ=2.5 — it drives to
a **large-weight, reproducible** representative.

**Mechanism (diagnosed, A2.3 confirmed).** The constant μ_a is **distributed across
paths**: β+(W+V)μ = μ to relerr **0.023** (the k=0/skip/constant gauge, realized by
the optimizer). W injects an input-dependent fluctuation that V cancels through
attention: zeroing V alone spikes loss to **7.46**, while zeroing W alone stays at
floor **1.40**. So V is load-bearing *inside the trained gauge* even though the clean
representative needs no attention at all.

**Corrected statement (for A_static.md and OPTIMAL_FRA_NOTE.md):** *Every FRA
attribution on a trained Setting-A model is a property of the optimizer's gauge
choice, not the computation. This is loss-flat and hence unfaithful — but it is
**reproducible across seeds**, pinned by SGD's inductive bias, not random seed noise.
The right falsifier is the clean-vs-trained equivalence (same loss, attributions 0 vs
large), which fires cleanly; the cross-seed-incoherence prediction was wrong.*

## Check 2 — A1.3 no-bias caveat

`out/check2_nobias.json`. **Analytic** (closed form, fixed a bug — a lives in a
13-dim affine subspace so Σ_0 is rank-deficient in R^48; must use the pseudo-inverse,
not inv): s_μ=**0.991**<1 (Sherman-Morrison bound holds), R=(1−α₀)(1−s_μ)μμᵀ to
**rank-1 relerr 5e-13**, left & right singular vectors cos(μ̂)=**1.000**. **Trained**
no-bias (3 seeds): V's top-left singular vector aligns with μ at **0.998**, and V·μ
delivers essentially the entire mean (‖Vμ‖=3.52 ≈ ‖μ‖=3.47, cos 0.99996). **Verdict:
CONFIRMED.** *Honest caveat:* the trained V is not globally rank-1 (top singular value
is 22% of the total) — SGD fills the μ-orthogonal directions with loss-irrelevant
gauge structure; only the **mean-carrying component** is rank-1 ∝ μ, exactly as A1.3
states (and consistent with the W/V split of the constant being itself gauge).

## Check 3 — A1.4 LMMSE within-position readout

`out/check3_lmmse.json`. Empirical LMMSE (OLS c~a, 600k samples) matches the closed
form on the data subspace to ~1e-13. **Complete N≤d:** T≈I (N=10,d=40: T−I=2.5e-3 MC
noise; N=d=24: T−I=**1.4e-12**, exact recovery even at ρ_mm=0.46 — full rank is what
matters, not orthogonality). **Overcomplete N>d:** mean shrinkage T_ii = **d/N
exactly** (0.500 at N=48/d=24, 0.300 at N=80/d=24) — a clean bonus law not in the
note. **Matched-filter contamination = Gc to 2e-16** (contamination is exactly the
Gram off-diagonals); off-diagonal T anti-correlates with the Gram (corr −0.74, −0.85
as overcompleteness grows). **Verdict: CONFIRMED**, add T_ii=min(1,d/N).

## Check 4 — A3 dilution scalings

`out/check4_dilution.json`. On a fixed trained model, splitting a firing feature into
m∈{1,2,4,8,16} exact-decomposition latents: per-latent OV magnitude × m = **1.000**
(∝1/m), per-pair QK × m² = **1.000** (∝1/m²), and the group-cut (Σ over the m latents
/ m² pairs) is invariant to **2.7e-15**. **Verdict: CONFIRMED to float precision.**

## Check 5 — A4 χ-realizability / MCC insufficiency

`out/check5_chi.json`. N=3 orthonormal GT features, two codes. **Complete** (L=3,
χ=I): MCC **1.0**, all three features severable (rowspace residual 0). **Absence**
(L=2, near-identity contrast, feature-1-by-weak-absence): MCC **0.995** (Hungarian
over min(L,N)=2 matched pairs — high!), yet **no** pure feature is severable
(e_i ∉ rowspace(χ): feat-1 residual 0.99, feat-0/2 residual 0.099 each — even the
"matched" features carry ~10% collateral). **Verdict: CONFIRMED** — high MCC does not
certify that "sever feature i" is a realizable latent-set operation; completeness
(rowspace) is the binding condition, exactly A4.2/A4.3 and sprint-2 Finding 4.

---

## Honest ledger

- **One proposition falsified** (Corollary A1.a's *characterization*), reported as the
  headline, not a footnote. The underlying theorem (attention loss-flat) is confirmed;
  the correction strengthens the gauge story (attributions are reproducible-but-
  unfaithful) and is arguably a cleaner result than the original.
- **Bug found and fixed mid-run:** the analytic Σ_0⁻¹ in check 2 used a dense inverse
  on a rank-deficient matrix (data lives in an (N+1)-dim affine subspace), giving the
  impossible s_μ=1.02>1; the pseudo-inverse restores s_μ=0.991<1 and R rank-1 to 5e-13.
- **R-readout gauge (check 3):** the raw LMMSE matrix R differs from the closed form
  when N<d (unconstrained on the data null-space); only its action on the data
  subspace, and T=R·Dᵀ, are invariant — both match. Reported as the action error.
- **Scope:** all models are the no-LayerNorm 1-head architecture with Σ=I (independent
  features); hedging (Σ≠I) and hierarchy→χ off-diagonals (A4.1 leading-order) were not
  exercised numerically here. The trained V rank-1 claim (check 2) is a *component*
  statement, not a global one (top singular value 22%).
