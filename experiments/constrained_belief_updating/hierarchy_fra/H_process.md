# H_process — Setting H: the minimal temporal hierarchy (process, filter, gate)

*H1 of the hierarchy round (theory-reset). Extends `../bernoulli_fra/OPTIMAL_FRA_NOTE.md`
and my B4(b) sketch. Notation guard: `ρ_obs` (=ν/A noise-to-signal), `ρ_path`
(attention-path share), `ρ_mm` (superposition), **`Δ_gate`** (the CE penalty a
lag-only pattern pays on Setting H — the "gating value", quantified by H2). Code
`code/hier.py`. Boxed props tagged CONFIRMED(numeric) / theory / future.*

## 1. The process (minimal; reset-on-parent-death)

Two features, parent P and child C, unit vectors `d_P ⊥ d_C` (ρ_mm later).

- **Parent**: persistent reset chain — rate `λ_P`, firing prob `p_P` (Setting B):
  with prob `(1−λ_P)` redraw `z_P ~ Bern(p_P)`, else persist.
- **Child (gated)**: `z_P(t)=0 ⇒ z_C(t)=0`. When the parent is on, the child runs
  its own reset chain (rate `λ_C`, prob `p_C`), **but the child state is RESET
  whenever the parent dies**:
  - `z_P(t)=1, z_P(t−1)=0` (parent just turned on): child re-initialises,
    `z_C(t) ~ Bern(p_C)`.
  - `z_P(t)=1, z_P(t−1)=1` (parent stayed on): child reset chain,
    `z_C(t) = z_C(t−1)` w.p. `λ_C`, else `~ Bern(p_C)`.

> **Convention (stated explicitly, load-bearing).** *Reset-on-parent-death*: the
> child forgets its state across any parent-off gap and re-initialises from the
> prior when the parent returns. This is the simplest convention that keeps the
> joint latent a finite Markov chain and the joint filter **exact** — the child
> carries no memory through a parent-off gap, so the joint state
> `s(t)=(z_P(t),z_C(t))` is sufficient. (The alternative — child *freezes* through
> gaps — needs an extra memory bit and is deferred.)

**Joint latent** `s(t) ∈ {A=(0,0), B=(1,0), C=(1,1)}`; state `(0,1)` is forbidden
(child-on ⇒ parent-on). **Observation** `a_t = c_P d_P + c_C d_C + b`,
`c_x = z_x·ReLU(μ_x+σ_x ε_x)` (Setting-B magnitude, ε redrawn each step;
observation noise `ρ_obs` and superposition `ρ_mm` are dials).

**Null control**: gating removed ⇒ P and C are *independent* reset chains (4-state
product). Every hierarchy effect below must vanish on the control.

### Verified structure (CONFIRMED, `code/hier.py`)

The 3×3 joint transition `T` (with `q_P=λ_P+(1−λ_P)p_P`) is row-stochastic; at
`(λ_P,p_P,λ_C,p_C)=(0.8,0.4,0.7,0.5)`:

    T = [[0.920 0.040 0.040],   stationary π = (0.60,0.20,0.20)
         [0.120 0.748 0.132],     = (1−p_P, p_P(1−p_C), p_P p_C) ✓
         [0.120 0.132 0.748]]

- **Joint spectrum** = `{1, λ_P=0.80, 0.616}`: the parent contributes `λ_P`
  exactly (it is autonomous); the third eigenvalue is a **warp** of `λ_C` by the
  gating (0.616 ≠ λ_C=0.7) — the child timescale is shortened by parent mortality.
- **Null control spectrum** = `{1, λ_P, λ_C, λ_P λ_C} = {1,0.8,0.7,0.56}` exactly
  (factored product) — the hierarchy warp is absent. `[CONFIRMED]`

## 2. The exact joint Bayes filter

> **Prop H1.a (exact filter).** `[CONFIRMED]` With orthogonal D and observable
> magnitudes, the current observation `(z_P(t),z_C(t))` pins the joint state
> `s(t)` (the map `(z_P,z_C)↦s` is injective on the 3 valid states). The filter is
> Markov: `P(s(t+1) | a_{1:t}) = e_{s(t)}ᵀ T`. In particular the **child
> one-step prediction depends on the PARENT state**:
>
>   `P(z_C(t+1)=1 | s(t)) = T[s(t), C]` = **0.040 (A) / 0.132 (B) / 0.748 (C)**.
>
> States A=(0,0) and B=(1,0) both have `z_C=0`, yet differ 3.3× (0.040 vs 0.132)
> **purely by the parent bit** — predicting the child requires reading the parent.

## 3. The constrained (attention-realizable) update — WHERE hierarchy enters

As in Setting B, in the clean-observation limit the current token is sufficient
(child prediction is a *linear* readout of the current `(z_P,z_C)`, since 3 valid
states are affinely independent), so a lag-0 pattern suffices and **no content
gating is forced**. The hierarchy handle lives where attention is load-bearing —
`ρ_obs>0` (noisy child obs) or multi-step — exactly the regime where the child
belief must integrate history.

The child's per-source correction propagates its evidence forward, **but that
propagation is screened by parent death**:

> **Prop H1.b (the reset-gate — the content dependence).** `[CONFIRMED]` The
> optimal per-source child correction from a source `s` is
>
>   `λ_C^{d−s} · g_C(z_C(s)) · 𝟙[no parent-off in (s, d]]`,
>
> the last factor being a **content-gate on the parent state at intervening keys**.
> Concretely, for a child last observed ON at lag `k`, the child belief is
>
>   parent-survived: `p_C + λ_C^k(1−p_C)`   vs   parent-died-in-window: `p_C`,
>
> a gap **`Δ = λ_C^k(1−p_C)`** (0.35, 0.25, 0.17, 0.08 at k=1,2,3,5 here). A
> **lag-only kernel** `α(d−s)` weights the child source by lag alone — *identical
> in both cases* — so it cannot reproduce the parent-death screening and must
> mis-predict. This is the first place in the program where the optimal pattern is
> forced to gate on KEY-side content.

> **Null control (CONFIRMED).** With independent chains the child belief is
> `p_C+λ_C^k(1−p_C)` regardless of the parent trajectory (gap = 0): the reset-gate
> vanishes, lag-only is sufficient, and FRA-QK stays gauge — every hierarchy effect
> disappears exactly as required.

## 4. The loss-pinned FRA-QK coupling (hand-off to H2/H3)

The reset-gate is realized as a **rank-≈1 QK content coupling**
`(child-prediction query at d) × (parent-off key at s)`: the child query must
attend to the *most recent parent-off event* to screen stale child evidence
(equivalently, up-weight parent-ON-continuous child sources). Because this gate
changes the loss (Prop H1.b), the coupling is **loss-pinned ⇒ gauge-INVARIANT** —
FRA-QK's first real handle. Its size is tied to the aggregate gating value
`Δ_gate(λ_P,λ_C,p_P,p_C)` = process-weighted CE penalty of the best lag-only model
(H2 quantifies; the per-instance magnitude is `λ_C^k(1−p_C)` above).

**The parallelogram hook for H2's necessity theorem:** construct context pairs
with identical position-wise token content EXCEPT the parent-state at one
intervening key (parent-survived vs parent-died since the last child-on); the
optimal child prediction differs by `λ_C^k(1−p_C)` while any lag-only pattern
produces the same attended aggregate. `[theory — H2 owns the exact Δ_gate and the
gauge-invariance corollary]`

## 4b. Platform reconciliation with H_theory (the two design flags)

theory-static's H_theory.md flags two design requirements; both are now baked into
the H4 platform (`code/hier_data.py`). **Stating the noise convention against H2.2:**

1. **The noise dial degrades the CHILD INDICATOR, not just its magnitude.** `Δ_gate`
   is driven by `P(E_k;ρ_obs)` = look-back reliance *on the child*. A magnitude-only
   noise (σ_C large, μ_C≫σ_C) leaves `sign(c_C)` a clean readout of `z_C` ⇒ the child
   state is read from the current token ⇒ no look-back ⇒ `Δ_gate=0`. **We therefore
   noise the child indicator via the ReLU false-negative atom `q_C = Φ(−μ_C/σ_C)`**
   (a fired child reads 0 with prob `q_C`, so a `c_C=0` observation is ambiguous):
   noisy = `μ_C=0` ⇒ `q_C=0.5`; clean = `μ_C≫σ_C` ⇒ `q_C≈0`. The **parent stays
   clean** (`μ_P=1.5, σ_P=0.3`). (An additive `obs_noise` dial is also provided.)
   *This was the cause of an initial null result — a magnitude-only noise gave
   `Δ_gate≈0` for the design reason H2.2 predicts; corrected here.* `[reconciled]`
2. **Parent mortality tuned toward balanced survival** `s≈½` (H2.2: `Δ_gate ∝ s(1−s)`,
   maximal at `s=½`). Default retuned to `(λ_P,p_P)=(0.5,0.3)` ⇒ `q_P=0.65`, pushing
   parent survival over the child window toward ½ (was `(0.8,0.4)`, `q_P=0.88`,
   `s(1−s)≈0.13`, half-max). `[reconciled]`

**Depth note (H4 finding):** the reset-gate is a **2-hop** computation (gate
child-keys by the most-recent parent-off ⇒ layer-1 computes "steps since last
parent-off", layer-2's child-query gates on it), so a **1-layer** attn-only model
cannot learn it (measured handle ≈0 at 1L even with a strong config); H4 uses
**≥2 layers** (`code/hier_model.py`). `[CONFIRMED — 1L insufficient]`

## 5. What H4 will verify (trained attn-only)

(i) the trained pattern IS content-gated (freeze/shuffle + clean-vs-trained-at-
equal-loss, not seed variance); (ii) an FRA-QK cut of the parent→child coupling has
an **O(1), G1-gauge-ROBUST** effect (unlike every prior QK cut); (iii) the null
control shows none of it; (iv) selectivity — the cut degrades CHILD prediction,
parent prediction is the collateral control. `[future — H4]`
