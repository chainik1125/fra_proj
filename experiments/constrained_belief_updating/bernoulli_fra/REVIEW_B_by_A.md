# Adversarial review of B_reset.md — by the Setting-A agent

*Referee: theory-static (author of A_static.md / VERIFY_A.md). Approach: skeptical,
trying to break each load-bearing claim. I independently re-derived the warp (two
ways), the parallelism tax, and the intervention collateral, and re-checked P3's
two-state quadratic (`javan_theory` eq. `quad`, line 1118) and the M_λ convention.
Independent numerics in this review are inline (Kalman cross-check); I did not rerun
theory-reset's `verify_b2.py`.*

**Bottom line: the core is sound and in several places better-founded than stated,
but there is one conceptual ERROR at the A/B interface (the ρ→0 "null" is conflated
with Setting A's null — they are opposite mechanisms), one propagating-risk notation
collision, one factor-λ slip, and one collateral result whose O(ρ²) is correct but
whose justification omits the load-bearing cancellation (and its contingency on a
matched code). Four items must be resolved before OPTIMAL_FRA_NOTE.md.**

---

## F1 — The warp (b_eff=A+ν, m_eff=Aλ → palindromic root). VERDICT: SOUND (independently verified to 1e-16).

Claim (B2.2–B2.3): the per-feature reduction is exactly P3's two-state problem with
`b_eff=A_i+ν_i`, `m_eff=A_iλ_i`, giving `ρλη²+[(λ²−1)−ρ(λ²+1)]η+ρλ=0`, in-disc root
`η_i(ρ)` with η→0 (ρ→0), η→λ (ρ→∞), η<λ, monotone.

Re-derivation #1 (my substitution into P3 eq. `quad` (bλ−m)η²+(2mλ−bλ²−b)η+(bλ−m)=0):
`bλ−m=νλ`, `2mλ−bλ²−b=A(λ²−1)−ν(λ²+1)` — matches the doc exactly. Limits: at ρ=0 the
quadratic degenerates to `(λ²−1)η=0⇒η=0`; small-ρ balance gives `η≈ρλ/(1−λ²)`; ρ→∞
gives `λη²−(λ²+1)η+λ=0⇒η=λ` (in-disc). All correct.

Re-derivation #2 (completely independent — steady-state Kalman filter for AR(1)+obs
noise, `x(t)=λx(t−1)+w`, `Var(x)=A`; `y=x+v`, `Var(v)=ν`). The optimal one-step LINEAR
predictor's look-back pole is `λ(1−K)`, K the steady Kalman gain. I computed
`λ(1−K)` and the doc's palindromic root over a 5×5 grid `λ∈{0.3,.55,.7,.8,.95}`,
`ρ∈{1e−3,.1,1,10,1e4}`: **agreement to 1e-16 at every point** (e.g. λ=0.7,ρ=1 → both
0.408367). This confirms the (b_eff,m_eff) identification by a route that never
touches P3's machinery, and is *tighter* than the doc's own brute-force MSE check
(<4e-3 — that number is grid granularity, not a theory gap; the mapping is exact).

The effective-noise decomposition is legitimate for the linear problem: writing
`c_i(t)=m_i z_i(t) + z_i(t)(r_i(t)−m_i)`, the second term is **white** (Cov at τ≥1
is 0, since r is iid across t) and **uncorrelated with the signal at all lags**, with
variance `ν_i=p_i Var(r_i)`. So the second-order structure is exactly "AR(1)-cov
signal (rate λ, var A) + white noise (var ν)". `[verified]`

## F2 — M_λ identification is off by a factor λ_i. VERDICT: ERROR (non-propagating).

B2.1 line 214 and B3 line 377 write `M_{λ_i}=A_i d_id_iᵀ`. But the doc's own Γ(τ)=
Σ_i A_iλ_i^τ d_id_iᵀ (τ≥1) matched to P3's `B^{(τ)}=Σ_λ λ^{τ−1}M_λ` gives, at τ=1,
`M_{λ_i}=Γ(1)=A_iλ_i d_id_iᵀ` — **an extra λ_i**. The load-bearing object m_eff=A_iλ_i
(used in the quadratic) is the *correct* value, so the quadratic and everything
downstream is fine; but the intermediate spectral identification is wrong as written
and should read `M_{λ_i}=A_iλ_i d_id_iᵀ`. Fix for correctness/consistency.

## F3 — "ν = observation noise" and "ρ_i is the single control parameter." VERDICT: SOUND but scope must be stated.

The mapping ν↔observation noise is exact **only** for the linear/second-order (MSE)
problem — which is precisely what P3/B2 is, so B2 is exact. But the true emission is a
multiplicative-rectified (non-Gaussian, heteroscedastic: variance 0 when z=0) channel;
its **nonlinear** Bayes filter (B1) does *not* reduce to AR(1)+Gaussian-noise. The doc
keeps B1 (nonlinear, rate λ) and B2 (linear, rate η) correctly separate, but "ρ_i is
the single control parameter of B2" (line 77/199) should be scoped as *the control of
the linear optimum*, not of the Bayes filter. `[scope clarification]`

## F4 — ρ→0 "attention null" is NOT Setting A's null. VERDICT: ERROR (conceptual conflation — the highest-priority fix).

Lines 258–259 and 543–545 assert the clean-emission limit "is Setting A's 'attention
is null'… ties Setting B back to Setting A's null-attention from the temporal side."
**These are opposite mechanisms.** There are two distinct limits, and the doc conflates
them:

- **λ→0 (iid) = Setting A exactly.** No persistence ⇒ no cross-time signal ⇒ P3's
  R=0 identically (my A1.1/A1.2) ⇒ **both** the look-back attention **and** the direct
  path are null; the best predictor is the constant mean μ_a; the whole story is the
  mean/bias (k=0) gauge. Here η→0 *and* the AR(1) signal `A_iλ^τ→0`.
- **ρ→0 (clean emission, λ fixed >0) = B-clean.** The cross-time signal **exists** and
  is **fully captured by the current token** (Markov sufficiency; Kalman gain K→1). The
  look-back attention is null because the signal is **redundant**, not absent — and the
  **direct/skip path is maximally load-bearing** (predictor `= λ·y(t)`, gain λ), which
  is genuine computation, **not** gauge, and **loss-pinned**.

My Kalman cross-check makes this unambiguous: at ρ→0, K→1 ⇒ look-back pole `λ(1−K)→0`
but the current-token gain `λK→λ`. So B-clean is "**look-back null with a live direct
path**"; Setting A is "**everything null but the mean**" (current-token gain 0). Same
η→0, opposite direct-path status. The genuine A↔B bridge is **λ→0**, not ρ→0. The doc
even states the tension correctly one sentence earlier ("the nonstationary eigenvalue
λ_i *exists* in the process but the optimal attention does not exploit it") and then
erases it by calling it "Setting A's attention-null." **This must be rewritten**: the
"one genuinely new structural claim" (the headline, line 542) is real and valuable —
*clean emission ⇒ look-back attention null despite a persistent latent* — but it is a
**different** null from A1's (redundant-signal vs no-signal), and only λ→0 reduces to
A1.

## F5 — mean-sector / no-bias caveats across the interface. VERDICT: GAP (partial transfer, unaddressed).

The `a_0(d)` diagonal/skip gauge (B3(iii)) is the correct k=0 analog of A1's
mean-carrying gauge — that part transfers. But B-clean has a **nonzero direct-path
fluctuation prediction** `λ(z−p)` with no A1 analog (in A1 the fluctuation prediction
is exactly 0). Consequence for the no-bias caveat: in A1, stripping the bias forces
attention into a rank-1 mean-substitute (attention's *only* possible role); in B-clean,
stripping the bias leaves the **direct path** carrying both the constant p_i **and** the
Markov coefficient λ, with look-back attention still null. The doc should state that the
no-bias caveat does **not** transfer unchanged — B's direct path has real content that
A's lacks.

## F6 — parallelism tax + δ_{τ0} optimal additive kernel. VERDICT: SOUND (minor framing).

I re-derived `Var[tax]=p(1−p)Σ_{τ,τ'≥1}λ^{τ+τ'+|τ−τ'|}=p(1−p)Σ_{τ≥1}(2τ−1)λ^{2τ}=
p(1−p)λ²(1+λ²)/(1−λ²)²` from scratch — exact. The δ_{τ0} claim is correct and consistent
with F1: clean emission ⇒ Markov sufficiency ⇒ Kalman K=1 ⇒ all predictive weight on
the current observation ⇒ optimal additive look-back kernel is δ_{τ0}, zero tax; the
canonical `λ^{d−s}` kernel over-counts (positive tax). Framing nit: "the P2 ansatz is
provably suboptimal" — P2's geometric ansatz is *optimal for P2's partially-observed
Mess3*; it is suboptimal *for the fully-observed reset-clean process*. It is not that
P2 is wrong; the reset-clean is a different (revealed-state) regime. Say that.

## F7 — collateral ∝(d_iᵀd_j)² at ρ_mm>0. VERDICT: result SOUND, justification GAP + missing contingency.

I re-derived the edit collateral explicitly (aligned GT code, `o_k=γd_k`). Before
severing, feature j's belief carries a **first-order** contamination `+γG_ij C_i`.
Severing feature i via its d_i-readout removes `γĈ_i d_i`, `Ĉ_i=C_i+Σ_{k≠i}G_ik C_k`;
feature j sees `−γG_ij Ĉ_i = −γG_ij C_i − γG_ij Σ_{k≠i}G_ik C_k`. **The first-order
terms cancel** (`+γG_ij C_i` from j's contamination vs `−γG_ij C_i` from removing i's
true write), leaving net `−γG_ij Σ_{k≠i}G_ik C_k`, whose leading (k=j) term is
`−γG_ij² C_j = O(ρ_mm²)`. So the O(ρ²) is **correct** — but the doc's stated mechanism
("one read-overlap, one write-overlap") is cryptic and **omits the first-order
cancellation that is the actual reason** it is second-order. A naive reading (which I
did first) gives O(ρ) collateral.

**Missing contingency (connects to A4):** the cancellation requires the sever's encoder
= decoder = d_i with unit gain (a **matched/complete/centered** code). With a learned χ
(encoder ≠ decoder, or non-unit gain), the first-order cancellation is broken and
collateral reverts to **O(ρ)**. So the "cleanest, unbounded (O(ρ⁻²)) use/collateral
separation" holds **only in the GT dictionary**; a real SAE gives O(ρ⁻¹). The doc's
B4(c) acknowledges the dictionary remixes channels but never connects this to the
collateral order — it should.

**Relation to my A3.4 (the lead asked "same leading order?"):** No, and correctly so —
they are different objects. A3.4's O(ρ) is the *static term contamination* of the belief
readout; B3(iii)'s O(ρ²) is the *edit-difference after the matched-sever cancellation*.
Consistent, not contradictory; the doc should note the belief itself is O(ρ)-contaminated
while the *change under a matched cut* is O(ρ²).

## F8 — seed-noise framing inherited from my retracted methodology. VERDICT: GAP (verification-plan hazard).

Line 409 ("QK cut effects read gauge + loss-flat noise") and B2.5 ("the per-feature rate
shrinkage is loss-flat") inherit exactly the "loss-flat ⇒ seed noise" framing my VERIFY_A
check-1 **falsified**: loss-flat objects are reproducibly **optimizer-pinned** (cross-seed
FRA corr 0.99, not ~0), not random. Any B-verification that tests **seed-incoherence** of
FRA-QK attributions (or of the trained η) will find **coherence** and either false-falsify
or misread it. Correct test = **clean-representative-vs-trained at equal loss** (attributions
0 vs large), not cross-seed correlation.

**B-specific sharpening (from F4):** in B-clean the **direct path is genuinely loss-pinned**
(real computation), unlike A. So a blanket "attention-null ⇒ attributions are seed noise"
test conflates the loss-pinned direct path with the loss-flat look-back attention. The B
verification must **separate** them: expect the direct-path weights to be seed-coherent
*because loss-pinned*, and the look-back attention to be seed-coherent *because
optimizer-pinned* — same observable (coherence), opposite cause. If theory-reset's check-2
was written to expect incoherence, it needs the same correction I applied.

## F9 — ρ_i overloaded. VERDICT: ERROR (notation collision, propagates into OPTIMAL_FRA_NOTE).

`ρ_i` denotes **two different quantities**: the noise-to-signal ratio `ν_i/A_i` (line 73,
the B2 control) **and** the attention-path share `1−(1−a_0)(1−η_i)` (line 431, the
sprint-2 c*=1/ρ tracking law). Different meanings, same symbol, both load-bearing.
Rename one (e.g. noise ratio → `r_i` or `ρ_i^{obs}`; path share → `s_i` or `ρ_i^{path}`)
before merge — this WILL confuse the reconciliation note.

## F10 — sprint-2 "resolution of the trained-rate discrepancy." VERDICT: SOUND but directional only.

The fidelity↔ρ mapping (B2.3) is qualitative: cleaner ⇒ smaller ρ ⇒ more shrinkage. But
Mess3's emission is *structurally* always partial (a token never reveals the 2-simplex
belief), so Mess3 sits at intermediate ρ and **never reaches ρ→0** — its η stays near ζ
(0.464 vs 0.55), it does not go to 0. The reset process *can* reach ρ→0; that regime has
no Mess3 counterpart. So the claim "explains sprint-2" is a directional analogy, not a
quantitative identity. Fine as worded ("direction… falls out"), but flag against a
quantitative reading.

## F11 — structural sketches. VERDICT: SOUND / reasonably flagged.

Block-diagonalization at ρ_mm=0 (B2.1) is exact (I checked `d_iᵀD⁻¹d_j=δ_ij/(A_i+ν_i)`
and the cross-term vanishing). The degree-2N palindromic polynomial (B2.4) is structurally
plausible (per-feature `w_i,e_i`, additive self-repulsion `q=Σq_i`), correctly flagged
[sketch] with numerical roots — same status as sprint-2's factored degree-6. Conic H_min
(B2.5) and hierarchy→loss-pinned-QK (B4(b)) are sound conjectures, honestly tagged.

---

## Overall assessment

Technically strong. The centerpiece warp (F1) is not just correct but exact — I confirmed
it independently to machine precision via the Kalman filter, tighter than the doc's own
check. The tax (F6), block structure (F11), and the O(ρ²) collateral **result** (F7) all
hold. The QK-gauge-holds-exactly-in-the-linear-model strengthening (B3(ii)) is legitimate
and a genuine improvement over Mess3's empirical ansatz.

The one conceptual **error** is the A/B interface (F4): the doc's marquee "clean ⇒ null
ties B to A" conflates redundant-signal (ρ→0, live direct path) with no-signal (λ→0,
Setting A). This is exactly the mechanism distinction the lead flagged, and it matters
because OPTIMAL_FRA_NOTE.md will lean on the A↔B bridge. The collateral justification
(F7) and the seed-noise framing (F8) both need repair informed by results that landed
after this doc was written.

## Must-resolve before OPTIMAL_FRA_NOTE.md

1. **F4 (interface, blocking):** rewrite the ρ→0 boundary as "look-back attention null
   with a **live, loss-pinned direct path** (redundant signal)", distinct from A1's
   full null; identify **λ→0**, not ρ→0, as the reduction to Setting A. Fix lines
   258–259, 543–545, and the headline.
2. **F7 (collateral):** make the first-order cancellation explicit as the mechanism for
   O(ρ²), and state the contingency — it holds in the matched GT dictionary and degrades
   to O(ρ) under a learned χ (encoder≠decoder). Reconcile with A3.4 (static O(ρ) vs edit
   O(ρ²)).
3. **F8 (verification methodology):** retract the "loss-flat ⇒ seed noise" language; the
   B-checks must use clean-vs-trained-at-equal-loss and must **separate** the loss-pinned
   direct path from the loss-flat look-back attention (do not run a seed-incoherence test
   that inherits my retracted methodology).
4. **F9 (notation):** disambiguate the two `ρ_i`.

Non-blocking fixes: F2 (M_λ factor-λ), F3/F5/F6/F10 scope-and-framing clarifications.
