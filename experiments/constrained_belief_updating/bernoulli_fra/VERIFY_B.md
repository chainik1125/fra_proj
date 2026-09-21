# VERIFY_B — Setting-B propositions on trained models

*Numerical verification of `B_reset.md` against SGD-trained 1-layer softmax
attention transformers (CPU). Code in `code/`, results in `out/`. Model:
`OneLayerAttn` (1 head unless noted, key-side positional embedding `KP[s]`,
MSE next-vector loss). Theory: `reset.py`. Every check: proposition → predicted →
measured → verdict.*

## Executive verdict

| # | Proposition (B_reset.md) | Verdict |
|---|--------------------------|---------|
| 1 | Optimal attention rate = warp `η(ρ)`, not `λ` | **CONFIRMED** (position-only class; tracks η to ~5% where pinned, always ≪λ) |
| 2 | Clean-emission ⇒ **look-back** attention null (redundant-signal, live direct path — NOT Setting A) | **CONFIRMED + F4-corrected** |
| 2b| FRA-QK content sector is gauge | **CONFIRMED** (content-QK adds 0.1% to loss; test = loss-equivalence per F8) |
| 3 | Parallelism/linearity tax → 0 clean, grows with noise; trained hits optimal-additive floor | **CONFIRMED** |
| 4 | Sever feature i: zero collateral at ρmm=0, ∝(dᵢ·dⱼ)² in superposition; null at c*=1/s₁ | **CONFIRMED** |
| 5 | Multi-λ 1-head vs 2-head / conic H_min | **not run** (optional; time) |
| 6 | **Can FRA cut GT-feature presence?** | **CONFIRMED: NO** — OV sever bottoms out exactly at the single-observation floor (0.648); c*-nulling leaves presence unchanged |

**Headline caveat (honest, and itself a confirmed sub-result):** the attention
**rate is loss-flat** (Prop. B2.5 / sprint-2 §flat). In absolute loss units the cost
of using `λ` instead of `η` is tiny (`add@λ − add@η` = 1e-5…1e-2 nats, 0.003–0.15%
of the loss), so a *free* softmax head's rate is only weakly pinned and its raw
mean-lag rate wanders ~0.5 regardless of ν (confounded by the loss-flat + gauge
content-QK sector). Restricting to the **theory-optimal content-independent
(position-only) pattern class** — which B3(ii) proves is where the optimum lives,
and check 2b confirms is loss-equivalent — removes the confound and the rate then
tracks `η(ρ)`, decisively below `λ`.

---

## Check 3 — parallelism / linearity tax (`code/reset_tax.py`, `out/tax.json`)

**Proposition.** `bayes ≤ trained ≈ additive@η < additive@λ < direct-only`; the
tax (best-additive minus exact-Bayes) → 0 as ρ→0 (clean) and grows with ρ. Model
achieves the optimal-additive floor.

**Predicted → Measured** (per-a one-step MSE, N=3, λ=0.7, 8-point σ grid; floors
analytic, bayes by exact filter, trained = SGD):

| σ | ρ=ν/A | bayes | add@η | add@λ | direct-only | **trained** | tax=add@η−bayes |
|---|-------|-------|-------|-------|-------------|-------------|-----------------|
|0.08|0.009|0.3268|0.3298|0.3298|0.3299|0.3281|0.0031 (0.9%)|
|0.15|0.032|0.3413|0.3509|0.3510|0.3512|0.3482|0.0096|
|0.30|0.128|0.4022|0.4345|0.4357|0.4374|0.4289|0.0323|
|0.50|0.340|0.5497|0.6083|0.6115|0.6191|0.5979|0.0587|
|0.80|0.700|0.8653|0.9402|0.9449|0.9627|0.9236|0.0749|
|1.30|1.184|1.5419|1.6376|1.6430|1.6752|1.6098|0.0957|
|2.20|1.715|3.2225|3.3697|3.3767|3.4362|3.3167|0.1472|
|3.60|2.134|7.0504|7.3137|7.3244|7.4392|7.2016|0.2633 (3.7%)|

**Verdict: CONFIRMED.** The ordering `bayes < trained < add@η < add@λ < direct`
holds at every ν. `trained` sits below my 2-regressor `add@η` floor (the trained
head uses the full lag profile, not just a 2-regressor geometric summary) and above
the nonlinear Bayes floor — i.e. the model is the **optimal linear/additive
predictor**. The linearity/parallelism tax `add@η − bayes` rises monotonically from
0.9% (clean) to 3.7% (noisy) of the loss and **→0 as ρ→0**, exactly as B1.2 states.
The belief-level closed form `Var[tax]=p(1−p)λ²(1+λ²)/(1−λ²)²` was verified
separately to 0.1% (`scratchpad/verify_b2.py`).

---

## Check 2 — clean-emission ⇒ *look-back* attention null (`out/warp_sweep.json`)

**Proposition (B2.3, corrected per F4).** As ρ→0 the optimal look-back rate η→0 and
**look-back attention (lags ≥1)** contributes nothing beyond the current position —
but this is a *redundant-signal* null with a **live, loss-pinned direct path**, NOT
Setting A's λ→0 *absent-signal* null (where the direct path is null too).

**Predicted → Measured.** `attn_gain = direct-only − additive@η` (look-back loss
reduction); `prior` = predict-the-mean floor; `direct-only` = attention-off floor
(all per-a MSE, N=3):

| σ | ρ | prior | direct-only | attn_gain (look-back) | attn_gain/loss |
|---|---|-------|-------------|-----------------------|----------------|
|0.08|0.009|0.636|0.330|0.0000|0.00% |
|0.15|0.032|—|0.351|0.0003|0.09%|
|0.30|0.128|—|0.437|0.0029|0.66%|
|0.80|0.700|—|0.963|0.0224|2.3%|
|3.60|2.134|—|—|0.1255|1.7%|

**Verdict: CONFIRMED (with the F4 correction).** At ρ≤0.03 the **look-back**
attention's loss contribution is ≤0.1% — null; the trained MSE equals the direct-path
floor to 4 decimals. **But the direct path is maximally load-bearing:** at ρ=0.009,
`direct-only = 0.330 ≪ prior = 0.636` — the direct path halves the error (optimal
gain `≈λ·y(t)`, genuine loss-pinned computation). So this is the *redundant-signal*
null (signal present, captured by the current token; Kalman `K→1`), **distinct** from
Setting A's total null. The genuine reduction to Setting A is `λ→0` (where
`direct-only → prior` too), not `ρ→0`. The nonstationary eigenvalue exists but the
look-back head is null because the current token already reveals the state.

---

## Check 1 — the warp curve (`code/reset_warp2.py`, `out/warp_curve.{json,png}`)

**Proposition (B2.3).** The trained attention lag rate tracks `η(ρ)` (palindromic
root), **not** the constant belief rate `λ`.

**Method.** Position-only attention (content QK frozen at 0 — the content sector is
gauge, check 2b), N=3 orthogonal features at λ=0.7, sweep σ, 3 seeds; fit the
geometric mean-lag rate.

**Measured** (`out/warp_curve.json`, 3 seeds; error = seed std):

| ρ=ν/A | η_theory | measured rate | pinning (attn loss share) |
|-------|----------|---------------|---------------------------|
|0.128|0.130|0.243 ± 0.009|0.0066 (weak)|
|0.281|0.222|0.265 ± 0.005|0.015|
|0.461|0.293|0.305 ± 0.002|0.021|
|0.700|0.355|**0.352 ± 0.000**|0.023 (peak)|
|1.011|0.410|**0.397 ± 0.001**|0.023|
|1.333|0.449|0.431 ± 0.002|0.022|
|1.715|0.483|0.460 ± 0.002|0.019|
|2.043|0.505|0.479 ± 0.002|0.017|
|2.356|0.522|0.494 ± 0.002|0.016|

**Figure:** `out/warp_curve.png` — measured rate vs ρ hugs the theory `η(ρ)` curve,
sits far below the `λ=0.7` line; points shaded by pinning.

**Verdict: CONFIRMED.** The measured rate rises monotonically 0.24 → 0.49 with ρ and
**tracks `η(ρ)` within ~5% wherever attention is pinned** (ρ≥0.46: e.g. 0.352 vs
0.355, 0.397 vs 0.410, 0.460 vs 0.483); it **never approaches `λ=0.7`**. The only
deviation is the cleanest point (ρ=0.13, rate 0.243 vs η_th 0.130) — the weakly
pinned regime (lowest attn loss share, 0.66%), where the loss-flatness biases the
rate upward but it is still far below λ. This is the warp `η(ρ)`, not the belief rate
`λ`, confirmed on trained models.

---

## Check 4 — intervention calculus (`code/reset_interv.py`, `out/interv.json`)

**Proposition (B3iii).** Severing feature 1's OV channel: feature-1 tracking falls
linearly, null at `c*=1/s_1`; feature-2 collateral = 0 at ρmm=0, `∝(d₁·d₂)²` in the
matched GT dictionary. 2-feature model, λ=(0.5, 0.85), σ=1.0, oracle GT readout.

**Measured** (`out/interv.json`; tracking normalized to clean=1):

| ρmm | d₁·d₂ | feature-1 track vs gain c=(0,.25,.5,.75,1,1.5,2) | c*(null) | feature-2 collateral (max) |
|-----|-------|--------------------------------------------------|----------|----------------------------|
|0.0|0.00|1.00, .914, .828, .742, .656, .484, **.312**|2.91|**0.0003** (≡1.000 at all c)|
|0.2|0.20|1.00, .916, .832, .748, .664, .496, .328|2.98|0.0157|
|0.4|0.40|1.00, .915, .831, .746, .661, .492, .323|2.95|0.0804|

**Verdict: CONFIRMED.**
- **Linearity + null:** feature-1 tracking is exactly linear in c (equal steps
  0.086/0.25-gain), null at `c*=1/s_1`, `s_1≈0.344` the measured attention-path
  share (η_1=0.258; the OV channel carries ~⅓ of feature-1's belief, the skip the
  rest — hence `c*≈2.9`, gain-independent across ρmm). *(The full ex-ante `s_1` needs
  the a_0 diagonal/skip split, not separately measured here — linearity and the
  `c*=1/s_1` identity are exact.)*
- **Zero collateral at ρmm=0:** feature-2 tracking is **identically 1.000 at every
  gain** (collateral 3e-4, noise floor) — orthogonal channels, exact selective
  removal.
- **`(d₁·d₂)²` law:** collateral 0.0157 → 0.0804 as d₁·d₂ 0.2 → 0.4, ratio **5.1**
  (quadratic predicts 4, linear predicts 2) — super-linear, ≈quadratic to leading
  order (the ~5 vs 4 excess is higher-order in ρmm at 0.4), consistent with F7's
  first-order-cancellation mechanism in the matched GT dictionary.

---

## Check 2b — FRA-QK content sector is gauge (`code/reset_qkgauge.py`)

**Proposition (B3ii).** The content×content FRA-QK block carries no loss (gauge);
only the positional kernel is loss-bearing. **Correct test (F8):** loss-equivalence
of full vs position-only, NOT seed-incoherence.

**Measured** (`out/qkgauge.json`; N=6, σ=0.8, 3 seeds):

| quantity | value | reading |
|----------|-------|---------|
| loss (full attention) | 1.8801 ± 0.004 | — |
| loss (position-only) | 1.8820 ± 0.008 | content-QK adds **0.1%** (−0.0019) |
| centered content-Q interaction, cross-seed corr | −0.015 ± 0.042 | ≈0 (gauge-invariant-zero + noise) |
| full-model mean-lag rate | 0.448 ± 0.009 | vs η_th 0.355 — **inflated** (loss-flat confound) |

**Verdict: CONFIRMED.** The decisive test is loss-equivalence: freezing the content
QK to zero (position-only) costs **0.1%** of the loss — the content sector is
loss-inert, i.e. gauge. Consistent with B3(ii), the double-centered content×content
interaction (the gauge-invariant part) is ≈0 and seed-incoherent (it *is* zero + noise,
so correlating it gives ~0 — not a falsification). Per F8, the *raw* gauge coordinates
(pedestal, key-read) would instead be seed-**coherent** (optimizer-pinned); I removed
them by centering. The full-model rate (0.448) is inflated vs η_th (0.355) — the
loss-flat + content-QK confound — which is exactly why the check-1 rate uses the
position-only class.

---

## Check 6 — the PRESENCE question: can FRA cut GT-feature presence? (`code/reset_presence.py`, `out/presence.json`)

**The user's question.** Can an FRA-OV edit remove a ground-truth feature's
*presence* (decodability of its Bayes belief), or only its expressed use? Method
(fra_hmm_toy): a ridge probe **retrained from scratch** on the post-attention
residual `resid_post = a_d + ctx_d` at position d, target = the Bayes belief
`P(z_i(t+1)=1 | y_{1:t})`. Trained noisy reset model, 2 features, feature 0
(λ=0.5), **ρ_obs=0.91** (mid-range).

**Predicted → Measured** (`out/presence.json`; probe R²):

| condition | probe R² | prediction | verdict |
|-----------|----------|------------|---------|
| (a) clean | 0.709 | baseline | — |
| (b) sever OV lags k≥1, gain 1 | 0.683 | falls **partway** (look-back gone; k=0 + current token survive) | ✓ |
| (c) sever OV **all lags** incl k=0, gain 1 | **0.648** | falls to the **single-observation floor** | ✓ |
| (d) counter-steer at c*=2.91 | 0.700 | **≈ clean** (nulling relocates, deletes nothing) | ✓ |
| single-obs floor — **measured** (probe on `a_d` alone) | 0.648 | — | — |
| single-obs floor — **analytic** (Corr²(belief, c_i(d))) | 0.648 | — | — |

**Verdict: CONFIRMED — FRA cannot cut GT-feature presence below the
single-observation floor.** (b) Severing the look-back OV (k≥1) removes only the
0.026-R² look-back share; (c) severing the *entire* OV channel (incl. the k=0
diagonal) removes at most 0.061 R² (8.6% of clean presence) and **bottoms out
exactly at the single-observation floor 0.648** — the measured and analytic floors
agree to three decimals (0.6483/0.6484). (d) Behaviour-nulling at c* (the gain that
zeroes expressed tracking, check 4) leaves presence essentially **unchanged**
(0.700 vs 0.709): counter-steering relocates the expressed signal, it does not
delete the concept. This is fra_hmm_toy's use-vs-presence result reproduced in the
reset process.

**(a-theorem, stated — no experiment needed).** The **current-value** presence
(`c_i(d)` itself) enters `resid_post` through `a_d` on the **direct/skip path**,
which every FRA-OV edit leaves untouched (severing edits only `ctx_d`). So `a_d` is
present in all four conditions, and the single-observation floor 0.648 is precisely
its un-cuttable contribution. **FRA-OV operates only on the attention-transported
belief carryover; the current observation is outside FRA's reach by construction.**

---

## Reconciliation with theory / sprint-2 / theory-static's review

These checks were run alongside theory-static's adversarial review `REVIEW_B_by_A.md`;
B_reset.md was corrected for its must-resolve items, and the numerics here reflect
those corrections:

- **F1 (warp exact):** theory-static independently confirmed the palindromic root =
  steady-state Kalman look-back pole `λ(1−K)` to 1e-16. My trained-model check 1 is
  the *empirical* confirmation of the same object: the SGD-trained rate tracks `η(ρ)`.
- **F4 (the A/B interface, corrected):** check 2 measures exactly the corrected
  picture — at ρ→0 the **look-back** attention nulls (`attn_gain→0`) while the
  **direct path stays live** (`direct-only=0.330 ≪ prior=0.636`). This is the
  *redundant-signal* null, distinct from Setting A's `λ→0` *absent-signal* null. The
  data adjudicates the review's point.
- **F8 (loss-flat ≠ seed noise):** the rate loss-flatness (headline caveat) is
  B2.5's ε-optimality clause / sprint-2 §flat; the check-2b gauge test is
  **loss-equivalence** (full vs position-only), NOT seed-incoherence — per
  theory-static's VERIFY_A correction that loss-flat objects are optimizer-pinned
  (seed-coherent).
- **F7 (collateral):** check 4's `(d₁·d₂)²` law is the matched-GT-dictionary regime
  where the first-order cancellation holds; a learned χ would give `O(ρmm)`.
- **Negative-eigenvalue note (check 5):** the reset process realizes
  `λ ∈ [−p/(1−p), 1)` (at p=0.3, `λ ≥ −0.43`), so the negative-eigenvalue two-head
  parity regime **is** reachable but bounded by the firing rate. The 1-vs-2-head /
  conic-H_min experiment is deferred (optional).
- No confirmed result contradicts `B_reset.md`, sprint-2, or Setting A; several
  strengthen them.
