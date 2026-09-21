# H_theory — the necessity theorem and the handle law (Setting H)

*H2 + H3 of the hierarchy round (theory-static). Builds on `H_process.md` (theory-reset's
Setting-H process, exact joint filter H1.a, and the reset-gate H1.b) and the predecessor
`../bernoulli_fra/OPTIMAL_FRA_NOTE.md`. Notation guard: `ρ_obs`=ν/A observation noise,
`ρ_path` attention-path share, `ρ_mm` superposition, **`Δ_gate`** the CE penalty a lag-only
pattern pays on Setting H (the "gating value", quantified below). Propositions boxed and
tagged **[theory]** / **[theory, surrogate-exact]** / **[conjecture]**; every one carries a
falsifier for theory-reset's H4.*

---

## 0. Setup and the one structural fact that organizes everything

**Setting H (from H_process.md).** Joint latent `s(t) ∈ {A=(0,0), B=(1,0), C=(1,1)}`
(child-on ⇒ parent-on; state (0,1) forbidden), reset-on-parent-death. Observation
`a_t = c_P d_P + c_C d_C + b`, `c_x = z_x·ReLU(μ_x+σ_x ε_x)`, orthogonal `d_P ⊥ d_C`. Dials:
observation noise `ρ_obs`, superposition `ρ_mm`. Null control: gating removed ⇒ independent
chains, spectrum `{1,λ_P,λ_C,λ_Pλ_C}` (no warp).

**The model class under test (P3, one head, lag-only pattern).** For destination `d`,
$$f(X)_d = \underbrace{\mathbf W\,a_d}_{\text{direct}} + \underbrace{\mathbf V\!\!\sum_{s\le d}\alpha(d-s)\,a_s}_{\text{lag-only attended}},$$
with `W, V` arbitrary linear maps and the pattern **content-independent** (`A_{ds}=α(d−s)`,
a function of lag alone). Multiple lag-only heads add more fixed-weight aggregates
`V^{(h)}Σ_τ α_h(τ)a_{d−τ}`. Contrast the **content-gated** class, where `A_{ds}` may depend on
the token content at `d` and `s` through the QK scores — the class that can realize H1.b's gate.

> **The organizing fact (from H1.a + H_process §3).** At **clean** observation the current
> token pins the joint state `s(t)` (H1.a: the map `(z_P,z_C)↦s` is injective on the 3 valid
> states), so the child prediction `P(z_C(t+1)=1|s(t)) = [e_{s(t)}^\top T]_C` is a function of
> the **current** state only (Markov) — a **lag-0** readout the direct path `W a_d` computes
> with no look-back and no gate. **Hierarchy forces content-gating only where look-back is
> load-bearing, i.e. at `ρ_obs>0`.** Multi-step horizons do *not* help: `P(z_C(t+h)|s(t)) =
> [e_{s(t)}^\top T^h]_C` is still a function of `s(t)` alone. So the necessity theorem is a
> statement about the **noisy-observation** process; `Δ_gate` is a function of `ρ_obs` and
> vanishes at `ρ_obs=0`. This noise–gating interaction is itself a headline: **the fra_win
> boundary needs BOTH content structure AND partial observability.**

Throughout, "the child prediction" means the child indicator's next-step (or, under noise,
current-step) posterior; all statements have a parent-prediction analogue that serves as the
**collateral control** (parent dynamics are autonomous, so parent prediction never needs the
child channel and never gates).

---

## H2 — the necessity theorem

### H2.1 Lag-only patterns are insufficient at ρ_obs > 0 (the parallelogram)

> **Prop H2.1 (necessity / the product obstruction). [theory]** On Setting H at `ρ_obs>0`,
> every lag-only-pattern model — any `W, V, {α_h}` — pays a strictly positive CE penalty
> `Δ_gate(λ_P,λ_C,p_P,p_C;ρ_obs)>0` relative to the joint-Bayes optimum. The obstruction is
> that the optimal child correction (H1.b) is a **product** of two content features at
> different positions,
> $$\text{child-evidence at lag }k\ \times\ \underbrace{\mathbb 1[\text{no parent-off in }(d-k,d]]}_{\text{AND over the window}},$$
> and no linear readout of lag-weighted aggregates reproduces a product.

**Proof (parallelogram + product obstruction).** Fix a lag `k`. Condition on the event
`E_k` = {child last *observed* ON at lag `k`; the current child observation is uninformative
about `z_C(t)` — an `O(ρ_obs)`-probability event that is empty at `ρ_obs=0`}. On `E_k` the
joint process still admits two sub-histories, realizable with positive probability:

- `H^+` (**parent survived** the window `(d-k,d]`): the child's reset chain has run
  uninterrupted, so `P^*(z_C(t{+}1)=1) = p_C + λ_C^{k}(1-p_C)`;
- `H^-` (**parent died** somewhere in the window): the child was forced off and re-initialised
  at parent-return, so `P^*(z_C(t{+}1)=1) = p_C`.

The two sub-histories have **identical child-channel content at every position** (child ON only
at lag `k`, same magnitude; off elsewhere in both) and differ **only** in the parent bit at one
or more intervening keys. Two facts collide:

1. *Any lag-only model produces the same child-channel aggregate on `H^+` and `H^-`.* The
   attended child content is `Σ_τ α(τ) c_C(d-τ)`, which depends on the child channel only; it is
   equal on `H^+, H^-`. The direct path reads the (uninformative, by `E_k`) current token, equal
   on both. The parent channel *differs*, but it feeds the child prediction only linearly:
   `V Σ_τ α(τ) a_{d-τ}` contributes a term affine in the lag-weighted parent aggregate
   `Σ_τ α(τ) c_P(d-τ)`.
2. *The correct child prediction is the product `𝟙[gate open]·λ_C^k(1-p_C)`, not an affine
   function of `(child aggregate, parent aggregate)`.* The gate "parent ON at **all** of
   `(d-k,d]`" is an AND over positions — a threshold/product, not a sum. Screening the child
   evidence requires multiplying it by this AND.

A linear (even multi-head, still linear-in-aggregates) predictor that must output the AND-product
on the four content-types `(u,v)=(\text{child-ON}_k,\text{gate-open})\in\{0,1\}^2` incurs the
classic "affine-cannot-represent-AND" residual: the least-squares affine fit of `Δ·u·v` on the
four types leaves a nonzero residual on each type (§H2.2). Hence `Δ_gate>0`. Because more
lag-only heads add only more content-*blind* aggregates, and a linear combination of content-blind
aggregates is content-blind, no finite head count removes the residual — the conic head-count of
the predecessor does **not** rescue lag-only here. **∎ (product obstruction; the CE constant is
H2.2)**

**Null control (exact).** With independent chains the child belief is `p_C+λ_C^k(1-p_C)`
*regardless* of the parent trajectory (H_process §3): `H^+` and `H^-` collapse to one prediction,
the product degenerates (`v≡1`), and the **gating** value `Δ_gate=0`.
**Clean control (exact).** At `ρ_obs=0` the event `E_k` is empty (current token pins the state),
`Δ_gate=0`.

> **⚠ Rung-2 correction (theory-reset, `VERIFY_H.md`).** The phrase "lag-only is sufficient on
> the null control" — i.e. that the *affine* class of Prop H2.1 pays zero on null — is **FALSE
> under occlusion, and it was measured false.** Optimally filtering *any* partially-observed
> Markov chain is nonlinear (the forward recursion), so the affine lag-only class pays a large
> penalty on the null cells that has **nothing to do with gating**: measured child-MSE gap
> (affine − Bayes) = **0.073** (null, occluded) / 0.028 (null, clean), *larger* than on the gated
> cell. The affine-vs-Bayes gap therefore **conflates generic occlusion-filtering nonlinearity
> with the gating product** and must not be used as "the gating value." The correct null-zero
> reference is **parent-blind Bayes** — the optimal child filter that never observes the parent
> channel (still nonlinear in the child channel). Its gap to full Bayes is the *pure* value of the
> parent channel = **0.0000 exactly on both null cells** (parent irrelevant to the child) and
> **0.0140** on gated_occ. **Read `Δ_gate` as (full Bayes − parent-blind Bayes), not (full Bayes −
> best affine lag-only).** What survives from the original claim: on null the *gating product*
> `u·v` degenerates, so FRA-QK's content-gated coupling `Q̂` stays gauge — that part is intact.
> `[H2.1 null-control claim corrected — rung 2]`

**Falsifier (H4).** Train the best lag-only-pattern attn-only model (freeze the QK content sector
to zero / position-only) and the content-gated model on the *same* Setting-H data at `ρ_obs>0`;
predict a strictly positive CE gap `= Δ_gate` between them, vanishing on the null control and as
`ρ_obs→0`. A zero gap on gated+noisy data would falsify H2.1.

### H2.2 Quantifying Δ_gate — the gating value

> **Prop H2.2 (the gating value). [theory; surrogate-exact]** The CE penalty is the
> process-weighted Jensen gap of the parent-death screening:
> $$\boxed{\ \Delta_{\text{gate}}(ρ_{\text{obs}}) \;=\; \sum_{k\ge1} P(E_k;ρ_{\text{obs}})\;\Big[\,s_k\,\mathrm{KL}\big(\mathrm{Ber}(π_k^+)\,\|\,\mathrm{Ber}(\barπ_k)\big) + (1-s_k)\,\mathrm{KL}\big(\mathrm{Ber}(p_C)\,\|\,\mathrm{Ber}(\barπ_k)\big)\Big]\ }$$
> where `π_k^+ = p_C+λ_C^k(1-p_C)`, `s_k = P(parent survived (d-k,d] | E_k)`, and
> `\barπ_k = p_C + s_k λ_C^k(1-p_C)` is the best lag-only constant on `E_k` (the mixture the
> content-blind model is forced to emit). To leading order in the gap `δ_k := λ_C^k(1-p_C)`,
> $$\Delta_{\text{gate}} \;\approx\; \sum_{k\ge1} \underbrace{P(E_k;ρ_{\text{obs}})}_{\text{look-back reliance}}\;\underbrace{s_k(1-s_k)}_{\text{gate variance}}\;\frac{\big[λ_C^k(1-p_C)\big]^2}{2\,\barπ_k(1-\barπ_k)}.$$

**Derivation.** On `E_k` the content-blind model must emit a single constant; its CE-optimal
constant is the posterior mean `\barπ_k`, and the excess CE over the two-valued Bayes prediction
is exactly the Jensen gap displayed (a KL mixture). The small-gap expansion uses
`KL(\mathrm{Ber}(p+δ)\|\mathrm{Ber}(p̄)) = \tfrac{(p+δ-p̄)^2}{2p̄(1-p̄)}+O(δ^3)` with
`π_k^+-\barπ_k=(1-s_k)δ_k` and `p_C-\barπ_k=-s_kδ_k`, giving the `s_k(1-s_k)δ_k^2` weight. **∎**

**Reading the three factors** (each a design lever, §H2.5):
- **`[λ_C^k(1-p_C)]^2`** — quadratic in the gating gap; the child persistence set by the spectrum.
  This is why the *cut-effect* is quadratic (matching the program's `(cρ)^2` removal law), while
  the *coupling* is linear in the gap (§H3).
- **`s_k(1-s_k)`** — the parent-survival *variance*; maximal at `s_k=1/2`, zero when the parent
  always survives or always dies (no discrimination ⇒ no handle).
- **`P(E_k;ρ_obs)`** — the look-back reliance: the probability the current token is too noisy to
  settle the child, so the lag-`k` history matters. `P(E_k;0)=0` (clean ⇒ Markov ⇒ no look-back),
  rising with `ρ_obs`. This carries the entire `ρ_obs` dependence and is the Setting-B Kalman
  complement `∼(1-K(ρ_obs))` specialised to the child channel.

> **The MSE surrogate (clean constant).** Dropping the Fisher factor `1/2\barπ(1-\barπ)`, the
> MSE-form gating value is `Δ_gate^{MSE} = Σ_k P(E_k)s_k(1-s_k)[λ_C^k(1-p_C)]^2`. On the minimal
> four-type sub-problem with a balanced gate (`s=1/2`, uniform types) this reduces to the
> affine-cannot-do-AND residual `δ^2/16` per unit look-back mass — the parallelogram constant.

> **✅ Rung-2 numerical check (theory-reset, `VERIFY_H.md`).** At the trained config
> (`λ_P=0.5,p_P=0.3,λ_C=0.9,p_C=0.5`, occlusion 0.5, `q_P=0.65`) the surrogate evaluates to
> `Δ_gate^{MSE}=0.0189`. The **exact** gate from the enumerated filter is **0.0140** (full Bayes −
> parent-blind Bayes) / **0.0095** (full Bayes − best affine lag-only). The surrogate overshoots the
> pure gate by only **~1.35×** (affine gap by ~2×) — a mild overestimate; **the formula is basically
> right.** The rung-1 "14× shortfall" was `0.0189` vs a *LayerNorm-confounded* empirical gap `0.0013`,
> **not** a formula error — see the H3.1 caveat. `[H2.2 surrogate confirmed ~exact — rung 2]`

**[GAP flagged]** The exact `s_k` and `P(E_k;ρ_obs)` are process integrals (`s_k` from the
parent chain's `q_P=λ_P+(1-λ_P)p_P`; `P(E_k;ρ_obs)` from the child observation-noise model). I give
their structure and the closed-form Jensen/quadratic assembly; the *numerical* `Δ_gate` per
`(spectrum, ρ_obs)` is a one-line enumeration for theory-reset (H4) against this formula. Predicted
signs and monotonicities (below) are the falsifiable content.

**Falsifier (H4).** Enumerate `Δ_gate` (best-lag-only CE minus Bayes CE) on the exact chain and
check it against the boxed sum: (i) monotone increasing in `ρ_obs` from `0`; (ii) maximised in
`λ_P/p_P` near `s_k≈1/2`; (iii) scaling `∝ (1-p_C)^2 λ_C^{2k}`-weighted. A `Δ_gate` that fails to
vanish at `ρ_obs=0`, or that does not peak at intermediate parent-survival, would falsify H2.2.

### H2.3 Corollary: the content-gated coupling is loss-pinned ⇒ gauge-invariant

> **Corollary H2.3 (FRA-QK's first real handle). [theory]** The content-gated component of the
> optimal pattern — the centered `(child-query)×(parent-key)` score interaction `Q̂`
> implementing H1.b's gate — is **loss-pinned**: removing it costs `Δ_gate>0` (H2.1). By the
> imported QK-gauge theorem (sprint-2), the gauge orbit consists exactly of the *separable*
> score components (row constants, pedestals, the G1 re-split); the **centered interaction `Q̂`
> is gauge-invariant**. On every prior process the optimum had `Q̂≡0` (lag-only sufficient), so
> "gauge-invariant" was vacuous. **Setting H is the first process whose optimum has a nonzero,
> loss-bearing `Q̂` — the first genuine FRA-QK handle** — with worth exactly `Δ_gate`.

> **⚠ FALSIFIED for the trained model (rung-2 centerpiece, see the H3.1 caveat below and
> `VERIFY_H.md`).** The trained no-LN model realises the gate in **OV, not `Q̂`**: the raw
> `(child-query, parent-key)` cut is a gauge-robust null (+0.0007) while the FRA-OV cut carries it
> (+0.0203 ≈ `Δ_gate`). The gate is loss-pinned and gauge-invariant as an *object*, but its
> **carrier is not pinned to QK** — a content gate is OV-realisable. `Q̂` is the handle only for an
> *obligate query-key match* (induction), not a hierarchy gate. `[H2.3 revised]`

**Why it is the handle, sharply.** Every prior FRA-QK cut in the program targeted a *pedestal*
(a gauge coordinate), so its measured effect was dialable — swung `140×` (sprint-2) / `1.7×`
(attn-only §7 of the predecessor) under the G1 dial, with a gauge-invariant residue at the loss
floor. Here the cut targets `Q̂` itself, which is gauge-invariant, so its effect is **the same in
every gauge = `Δ_gate`**. This is the operational signature H4 must see: a QK cut whose effect
*survives the G1 dial* (unlike all predecessors).

**Falsifier (H4).** Cut the parent→child QK coupling on the trained gated model and sweep the G1
dial; predict the `ΔCE` stays `≈Δ_gate` across the dial (gauge-robust), whereas the same protocol
on any lag-only-optimal channel (or the null control) swings to `≈0`. A dial-swingable cut effect
here would falsify H2.3 (it would mean the "handle" is still a pedestal).

### H2.4 The general boundary conjecture (the fra_win law)

> **Conjecture H2.4 (the fra_win boundary). [conjecture]** FRA-QK has a nonzero gauge-invariant
> handle at the optimum **iff the process makes a lag-only pattern insufficient — iff the optimal
> prediction is a non-affine (product/threshold) function of the lag-weighted content
> aggregates.** Equivalently: the centered interaction `Q̂` is loss-pinned iff the optimal
> read-out couples content at two positions multiplicatively. The known instances are all
> two-position products:
> - **hierarchy** (Setting H): child-evidence × parent-survival gate;
> - **matching / induction** (fra_win): query-key match × value transport — the induction
>   product `[A][B]…[A]→[B]`, the empirical FRA-QK win;
> - **sparse / selective support**: "attend to the one salient key" is an argmax/threshold, again
>   non-affine.
> Lag-only sufficiency (all of Settings A/B, the null control, Mess3, clean-obs Setting H) is the
> complementary class where `Q̂≡0` at the optimum and FRA-QK is pure gauge. **The predecessor's
> map is thereby closed on the QK side: FRA-OV is a handle in the interior; FRA-QK is a handle on
> the content-gated boundary of it, and nowhere else.**

**Falsifier.** Any process with a non-affine two-position-product optimum whose trained model shows
a *gauge-dialable* (pedestal-only) QK cut — or any lag-only-sufficient process showing a
*gauge-robust* QK cut — would falsify the iff.

---

## H3 — the handle law

### H3.1 Cut-effect = Δ_gate (the handle is worth exactly the gating value)

> **Prop H3.1 (cut law). [theory]** Severing the invariant parent→child coupling `Q̂` at gain
> `c=1` turns the content-gated optimum into the best content-independent-pattern model, so
> $$\boxed{\ \Delta\mathrm{CE}\big(\text{cut }Q̂\text{ at }c{=}1\big) \;=\; \Delta_{\text{gate}}(ρ_{\text{obs}})\ }$$
> exactly at the optimum, and `≥Δ_gate` for a fixed non-re-optimised cut. For partial gains the
> effect follows the program's quadratic path calculus, `ΔCE(c) ≈ c^2\,Δ_gate` near `c=0`
> (the gate contributes at second order because `Δ_gate` is itself quadratic in the gap `δ_k`),
> with the behavioural handle **saturating** once `c` fully opens/closes the softmax gate.

**Reasoning.** Cutting `Q̂` sets the centered content interaction to zero; a pattern with no
content coupling is lag-only; the best such model pays `Δ_gate` by definition (H2.2). At the
optimum the rest of the network is already tuned, so the increment is exactly `Δ_gate`; a
non-re-optimised cut cannot do better, hence `≥`. **∎**

> **⚠ Rung-2 caveat — the LayerNorm redundancy (theory-reset, `VERIFY_H.md`).** The middle step
> — "a pattern with no content coupling is lag-only" — **assumes the rest of the network is a
> content-blind read-out. In a standard-LayerNorm transformer it is not.** LayerNorm's per-token
> normalization is a *content-dependent nonlinearity that sits OUTSIDE the QK pattern*, and it
> provides a **second carrier** for the parent→child gate: a position-only-QK model with LN reaches
> **0.2447** child-MSE — *below* the affine lag-only frontier (0.2511) and near the Bayes floor
> (0.2416) — recovering **~78%** of the `Δ_gate=0.0140` gate on its own. Consequences:
> 1. **The QK cut is redundancy-limited (path calculus).** By the program's carrier-multiplicity
>    rule, cutting one carrier measures `share × gate`, not the full gate. In a standard-LN model the
>    QK path's share over LN is small, so `ΔCE(cut Q̂)` measures the **residual ≈0.0013**, NOT
>    `Δ_gate`. H3.1's `= Δ_gate` holds only against a **truly content-blind (LN-free / linearized-LN)**
>    baseline, where LN cannot re-supply the gate. **The H4 QK-cut centerpiece must run on the
>    linearized-LN platform**, or it reads as a false null.
> 2. **Carrier multiplicity now includes normalization nonlinearities.** The predecessor's path
>    redundancy (bias/direct/attention, skip/diagonal) gains a member: **LayerNorm/RMSNorm**, which
>    is *invisible to attention-level (QK/OV) attribution* because it lives between the residual
>    stream and the pattern. Any "cut the QK coupling ⇒ effect = loss-pinned value" argument inherits
>    this confound in every real LN transformer. `[H3.1 caveat — LN redundancy, rung 2]`

> **⚠⚠ Centerpiece result — the carrier is OV, not QK, even LN-free (theory-reset, `VERIFY_H.md`).**
> The caveat above predicted the QK cut would recover the gate (≈0.006) once run on an LN-free
> platform. **It does not.** On the theory-clean (no-LayerNorm) trained model the parent→child gate
> is carried by **OV (values), not the QK coupling**: ablating the parent from the **keys** (the
> targeted FRA-QK cut) costs the child **+0.0007** (a gauge-robust *null*), while ablating it from the
> **values** (FRA-OV cut) costs **+0.0203 ≈ the gating value** — a **28× carrier asymmetry**, both
> gauge-flat (std ~1e-10). Mechanism (the depth-≥2 structure of H4.0): layer-1 reads the parent
> through OV and builds a *derived* recency feature; layer-2's child-query gates on that derived
> feature, so the **raw `(child-query, parent-key)` QK cell is empty**. **This falsifies H2.3 and
> H3.2** for the trained model — the loss-pinned coupling lives in OV + a derived-feature QK, and
> raw-direction FRA-QK attribution captures neither. **H2.4 sharpened:** a two-position product is
> *necessary but not sufficient* for a QK handle — a content **gate** (hierarchy) is OV-realisable
> and lands there; FRA-QK is the carrier only for an *obligate query-key MATCH* (induction/`fra_win`,
> where the product cannot be moved into OV). Carrier set of the gate now ≥3: LayerNorm, OV,
> derived-feature-QK. `[H2.3/H2.4/H3.2 revised — rung-2 centerpiece]`

**Falsifier (H4).** Measure `ΔCE` of the parent→child cut at `c=1`; predict it equals the
independently-enumerated `Δ_gate` (H2.2) to within re-optimisation slack, and traces a quadratic
`c²`-law for small `c`. A cut effect unrelated to `Δ_gate` would falsify.

### H3.2 What canonical-gauge FRA-QK reports (H vs null vs clean)

> **Prop H3.2 (the measurement law). [theory]** In the canonical gauge (pedestal `β≡0`, so the
> FRA-QK score *is* the centered interaction), a feature×feature FRA-QK measurement reports:
> - **Setting H, `ρ_obs>0`:** a **nonzero, localized** coupling on the `(child-query, parent-key)`
>   cell — positive on parent-**ON**-continuation keys / negative on the most-recent parent-**OFF**
>   key (up-weight live child evidence, screen stale) — with cut-worth `Δ_gate`. This is the first
>   nonzero entry the content×content block ever carries at an optimum.
> - **Null control (independent chains):** `Q̂≡0` — the child-query reads no parent key; the
>   content block is empty, FRA-QK pure gauge.
> - **Clean Setting H (`ρ_obs→0`):** `Q̂→0` — lag-0 sufficiency; the coupling is present only in
>   proportion to look-back reliance.
> The **coupling magnitude** scales as `‖Q̂‖ ∝ f(ρ_obs)·δ̄` (linear in the gating gap
> `δ̄=Σ_k w_k λ_C^k(1-p_C)`), while the **cut-effect** scales as its square `Δ_gate ∝ f(ρ_obs)·δ̄²`
> — coupling linear, loss quadratic, the standard score-vs-loss relationship.

**Caveat (softmax threshold).** The softmax realises the gate through an `O(1)` score *gap* on the
screened key (a threshold), so the *raw* score coupling can look `O(1)` even for small `δ̄`; the
gauge-invariant, `δ̄`-proportional object is the coupling *weighted by how often screening changes
the attended aggregate* (equivalently, read it off the cut-effect `Δ_gate`, not the raw score). I
report the loss-anchored magnitude; the raw-score magnitude is a softmax-saturation artifact
(flag for H4, analogous to the §7 magnitude-vs-signed pedestal caveat).

**Falsifier (H4).** Canonical-gauge FRA-QK on the trained gated model should localize a nonzero
`(child-query, parent-key)` coupling absent on the null control and shrinking with `ρ_obs→0`; the
selectivity — cutting it degrades **child** prediction while **parent** prediction (autonomous) is
untouched — is the collateral control.

### H3.3 Placing hierarchy on the §2 map: interior ∩ gating

> **Prop H3.3 (the map placement). [theory]** Hierarchy is not a new corner of the predecessor's
> `(λ,ρ_obs)` map — it is a **new axis (gating) that switches on the FRA-QK handle only inside the
> FRA-OV interior**:
> $$\text{FRA-QK handle size}\ \propto\ f(ρ_{\text{obs}})\cdot(\text{gating magnitude}),\qquad f(0)=0,$$
> so the handle lives in **corner C ∩ {gating on}** and is absent everywhere else — the clean
> corner (B-clean, `ρ_obs→0`) kills it by lag-0 sufficiency, the static corner (A, `λ=0`) has no
> temporal evidence to gate, and removing the gate (null control) kills it at every `ρ_obs`. The
> noise–gating interaction is the reportable result: **FRA-QK becomes a handle only where the
> process is BOTH partially observed AND content-gated** — the two conditions are conjunctive, and
> this is the closed-form form of the empirical fra_win boundary.

This extends the predecessor cleanly: FRA-OV is the handle across the interior (corner C); FRA-QK
is the handle on the gated sub-region of the interior; the two handles are complementary and both
require partial observability (`ρ_obs>0`) — the through-line of the whole program.

---

## H2/H3 process-design flags (bake in before H4 locks the process)

These are choices in H_process.md that **strengthen or kill** the theorems; flagging early so the
platform maximises the measurable handle.

1. **Child observation noise is the load-bearing dial — it must degrade the CHILD channel.**
   `Δ_gate` is driven by `P(E_k;ρ_obs)` = look-back reliance on the child. If only the *parent*
   channel is noisy while the child is clean, the child state is read from the current token and
   `Δ_gate=0` — no handle. **Require `ρ_obs` on the child** (a noisy child magnitude / a
   false-negative atom `q_C>0`); the parent may be cleaner. **[strengthen/kill]**
2. **Tune parent mortality to `s_k≈1/2` over the child timescale.** `Δ_gate ∝ s_k(1-s_k)`, maximal
   at balanced parent survival. With `q_P=λ_P+(1-λ_P)p_P=0.88` (the current
   `(0.8,0.4,·,·)`), the parent survives a `k=1/λ_C≈1.4`-step window with prob `≈0.88^{1.4}≈0.84` —
   the gate is open `~84%` of the time, `s(1-s)≈0.13`, well below the `0.25` max. **Lower `q_P`**
   (smaller `λ_P` or `p_P`) to push parent survival over the relevant window toward `1/2` and
   roughly double the handle. **[strengthen]**
3. **Reset-on-parent-death (already chosen) is the clean convention.** It makes the gate a pure AND
   over the window (no child memory through gaps), so the obstruction is the crisp
   affine-cannot-do-AND product. The *child-freezes* alternative adds a memory bit and softens the
   gate to a leaky product; the theorem survives but `Δ_gate`'s closed form is messier. Keep reset.
   **[keep]**
4. **Orthogonal `d_P⊥d_C` first (already chosen).** The gate is a QK/pattern object that reads the
   parent channel; at `ρ_mm=0` the parent key is clean. `ρ_mm>0` cross-contaminates the parent-key
   read by `O(ρ_mm)` (the child query picks up child-channel leakage), which *dilutes* the handle
   but does not remove it at leading order — defer `ρ_mm` to a later wave; it interacts with H5's
   absorption (a learned child latent that absorbs the parent, `χ_{child,parent}≠0`, may re-expose
   or destroy the parent-key the gate needs). **[defer, note the H5 interaction]**
5. **Report `Δ_gate` in recovery-fraction units, not raw nats.** As in the predecessor, the
   recoverable band is `~10^{-2}` nats; `Δ_gate` will be a fraction of it. Anchor the H4 loss gate
   against Bayes / best-lag-only / constrained, and quote `Δ_gate` as `%` of recoverable info so it
   is comparable across `(spectrum, ρ_obs)`. **[measurement hygiene]**

---

## Summary of propositions and falsifiers

| # | Statement | Tag | Falsifier (H4) |
|---|---|---|---|
| H2.1 | lag-only insufficient at `ρ_obs>0`; product obstruction; `Δ_gate>0` | theory | zero best-lag-only-vs-gated CE gap on gated+noisy data |
| H2.2 | `Δ_gate` = process-weighted Jensen gap `∝ P(E_k)s_k(1-s_k)δ_k^2` | theory; surrogate-exact | `Δ_gate` not vanishing at `ρ_obs=0`, or not peaking at balanced parent survival |
| H2.3 | gated coupling `Q̂` loss-pinned ⇒ gauge-invariant — the first FRA-QK handle | theory | QK cut effect swings under the G1 dial (still a pedestal) |
| H2.4 | fra_win law: QK handle iff optimum is a two-position product | conjecture | gauge-dialable QK cut on a product process, or gauge-robust cut on a lag-only process |
| H3.1 | cut-effect `= Δ_gate` (exact at optimum), `≈c²Δ_gate` for small `c` | theory | cut effect unrelated to enumerated `Δ_gate` |
| H3.2 | canonical-gauge FRA-QK reports a localized `(child-query,parent-key)` coupling on H, zero on null/clean | theory | nonzero coupling on the null control, or non-vanishing as `ρ_obs→0` |
| H3.3 | handle `∝ f(ρ_obs)·gating`, `f(0)=0` — lives in corner C ∩ gating | theory | handle present at `ρ_obs=0` or on the null control |

**Headline for the merged §8.** Hierarchy supplies the program's first loss-pinned, gauge-invariant
FRA-QK coupling, worth exactly the gating value `Δ_gate`, and it exists only where the process is
*both* partially observed (`ρ_obs>0`) *and* content-gated — the closed-form derivation of the
empirical fra_win boundary, and the QK-side complement to FRA-OV's interior handle.
