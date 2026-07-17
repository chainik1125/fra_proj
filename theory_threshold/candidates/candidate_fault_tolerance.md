# Candidate threshold theorem: corrective data as a faulty error-correcting code

*Angle 1 of the threshold-theorem build. Frame: fault-tolerance / coding theory —
the finetuning mix is a code; corrective examples build a decoder but each one
injects error through its own misaligned prefix. The threshold is the
fault-tolerance gadget condition ("a correction round must remove more error than
its own machinery adds") lifted from generation time to learning time.*

*All numbers below are re-derived from
`results/s2_pivot_classified.json`, `results/s2_curve_fits.json`,
`results/s2_toy_summary.csv`, `results/s2_nonergodic_checks.json`, and
`theory_threshold/inputs/measured_parameters.md`. Check script:
`/tmp/claude-execution-allowed/simplex-research/ft_checks.py` (rerun anytime).*

---

## 1. SETUP — the corrective-code model M

### 1.1 Plain-language dictionary (every invented term, defined once)

- **Entry** `p` — the probability that a generated answer on a broad
  (never-trained) question *starts* misaligned. In coding terms this is the
  **physical error rate**: how often an error is injected into an answer at all.
- **Exit** `E` — the probability that an answer which started misaligned
  interrupts itself mid-answer ("Wait — I need to stop…", a **pivot**) and
  finishes aligned: `E = P(exit | entered)`. In coding terms `E` is the
  **decoder**: the learned machinery that recovers an answer after an error.
- **Logical misalignment rate** `q` — the probability an answer is *judged*
  misaligned end-to-end (judged broad EM). "Logical" is the fault-tolerance word
  for the error rate *after* correction has acted; "physical" is the rate before.
- **Recovery credit** `r` — the probability that an answer which exited is
  scored aligned by the judge (a pivoted answer emits partial harm first; `r`
  says how often the judge forgives that). `0 ≤ r ≤ 1` by definition.
- **Poison rate** `π` — the *net* increase in entry caused by one corrective
  example (its misaligned prefix promotes misaligned starts, minus the credit
  from its own aligned second half). This is the **gadget self-fault**: the
  error the correction machinery itself injects, per unit of machinery.
- **Prefix fraction** `κ` — the misaligned mass one corrective example carries,
  in units of a full misaligned example (standard corrections: the first half of
  a misaligned answer, `κ ≈ 0.5`). The poison rate decomposes as
  `π(κ) = κ·π₁ − (1−κ)·α′`, where `π₁` is entry promotion per *full* misaligned
  example-equivalent and `α′` is the marginal entry suppression per aligned
  example-equivalent at the current mix.
- **Entry ceiling** `p_max` — the saturation value entry approaches when the mix
  is rich in misaligned content (entry is a probability; on top of 1000
  misaligned examples it has almost no room left to rise). **Headroom**
  `H = p_max − p` is how far below the ceiling the current entry sits; poison is
  only *visible* when there is headroom.
- **Variety floor** `v*` — the minimum number of *distinct* corrective examples
  below which no decoder forms at any training mass (the **code-distance /
  codebook-existence** analog: you cannot build a decoder from too few distinct
  codewords, no matter how often you repeat them).
- **Mass onset** `n₀` — the number of corrective *examples* (duplicates count)
  setting the scale at which the decoder switches on, given `v > v*`. The two
  thresholds are distinct and the duplication grid separates them (§3).
- **Net-positive** — adding corrective dose lowers `q` relative to the relevant
  alternative: (a) *absolute*: vs. not adding it; (b) *comparative*: vs. spending
  the same budget on plain aligned examples. (b) is the decision-relevant one.

### 1.2 The model (assumptions M1–M4)

Finetuning mix: `N_mis` misaligned examples (fixed background, 1000 in the LLM),
`a` plain aligned examples, `n` corrective examples of which `v` are distinct,
each with prefix fraction `κ`. The finetuned model's broad-question behaviour is
summarized by the pair `(p, E)`:

- **M1 (entry: saturating-linear in content mass).**
  `p(a, n) = clip_[p_base, p_max]( p_mis − α(a) + π·n )`,
  where `p_mis` is entry after the misaligned-only finetune, `α(a)` is the
  measured (concave, saturating) entry suppression from `a` aligned examples,
  and `π = π(κ)` is the net poison rate. `clip` means: entry cannot exceed the
  ceiling `p_max` or fall below the base-model entry `p_base`.
- **M2 (exit: decoder with two thresholds).**
  `E(n, v) = E_max · s(n) · 1[v > v*]`, with `s(n)` a monotone saturating onset
  curve, `s(0)=0`, `s(∞)=1`, half-dose `n₀` (fitted compression:
  Hill form `s(n)=n^h/(n^h+n₀^h)`, `h ≈ 1.5`; the classifier data prefer a
  harder, more threshold-like onset — see §5 limit L4). `1[·]` is the indicator:
  no decoder exists at all below the variety floor.
- **M3 (readout: logical = physical × decoder failure).**
  `q = p · (1 − r·E)`.
  An answer is judged misaligned iff it enters *and* either fails to exit or
  exits without judge forgiveness. (Implicit: at most one error and one
  correction round per answer — empirically true: switches per answer ≤ 1,
  relapses ≈ 0.)
- **M4 (locality: direct data beats generalization).** On domains with direct
  misaligned training data (the narrow domain), `E ≈ 0` and `p ≈ p_direct`
  independent of `n`. Corrections act only where no direct data exists — which
  is why every threshold below concerns *broad* EM and narrow is preserved in
  all arms (observed 0.97–1.0 toy, 0.21–0.29 LLM, all arms).

**Free parameters (5):** `π`, `E_max`, `n₀`, `v*`, `r`.
**Measured constants (inputs, not fitted):** `p_mis`, `p_max`, `p_base`, the
aligned curve `α(·)`, `h`, `κ`, answer length `L` (absorbed into `E_max`).

### 1.3 The fault-tolerance dictionary

| fault-tolerance quantity | this model | measured by |
|---|---|---|
| physical error rate `p` | entry (misaligned start probability) | classifier "entered"; toy first-token proxy |
| logical error rate `p_L` | judged broad EM `q` | GPT-4o judge |
| EC gadget / decoder | the learned exit (pivot) | pivot counts; `P(exit\|entered)` |
| gadget self-fault `c` (the corrector's own error) | poison rate `π` (entry promoted per correction) | stack-vs-aligned contrast: +5.3×10⁻⁴/example |
| code distance / decoder existence | variety floor `v*` AND mass onset `n₀` | dup grid + dose curve onset |
| threshold `p < p_th = 1/c` | the sign-flip inequality of Theorem 2/3 | all of the above |
| concatenation level `L` | **absent** — single, position-locked EC round | hazard peak at trained position |

The structural insight borrowed from fault tolerance: *the threshold exists
because the corrector is itself faulty.* If `π = 0`, corrections would be
unconditionally (weakly) helpful and there would be no theorem to prove. The
structural departure from fault tolerance: there is no concatenation — one
decoder round per answer — so suppression is bounded by the single factor
`(1 − r·E_max)` and can never be driven to zero (§2, Corollary 3).

---

## 2. THEOREM — when corrective data is net-positive

Throughout, "corrections at dose `n`" means adding `n` corrective examples
(`v` distinct, prefix fraction `κ`) to a fixed background mix; `q(n)` is the
resulting logical misalignment rate under M1–M3. All proofs are elementary
algebra **conditional on M1–M4**; the science is in M1–M4, which §3–§4 test.

**Theorem 1 (poisoned-onset lemma — below threshold, corrections are pure
poison).** If `v ≤ v*` or `s(n) ≈ 0` (dose well below `n₀`), then `E = 0` and

    q(n) − q(0) = min(π·n, H) ≥ 0,   H = headroom = p_max − p(0),

with strict inequality iff `π > 0` and `H > 0`. In words: below either threshold
(too little variety, or too little mass) corrections buy *no* correction and
*do* inject error — but the injected error is **invisible when entry is already
at its ceiling** (`H = 0`), i.e. when corrections are added on top of abundant
misaligned data with no aligned data present.
*Proof.* Substitute `E = 0` into M3 and M1; `clip` truncates the increment at
`H`. ∎

This single lemma explains why the backfire was discovered so late: the original
dose sweeps added corrections to a misaligned-saturated mix (`H ≈ 0`, entry
clipped flat at 0.58–0.75 — observed), so the poison never showed; the stack
experiment created headroom with aligned data first (entry 0.319) and the same
poison promptly reappeared as a +0.26 entry rebound.

**Theorem 2 (absolute threshold — the sign-flip).** Corrections at dose `n` are
net-positive versus not adding them iff

    r·E(n)  >  Δp(n) / (p(0) + Δp(n)),       Δp(n) = min(π·n, H)            (T2)

i.e. iff the *recovered fraction* of entered answers exceeds the *poison
fraction* of current entry. Marginal (differential) form: `dq/dn < 0` iff

    r·E′(n) / (1 − r·E(n))  >  p′(n) / p(n)                                  (T2′)

— "the logarithmic antidote rate must exceed the logarithmic poison rate", the
fault-tolerance gadget condition in learning-time form.
*Proof.* `q(n) < q(0)` ⇔ `p(n)(1−rE(n)) < p(0)` ⇔ `rE(n) > 1 − p(0)/p(n)`;
expand `p(n) = p(0)+Δp(n)`. Differentiate M3 for (T2′). ∎

Two immediate consequences. (i) Since `E(n) = 0` for `n` below the thresholds
while `Δp(n) ≥ 0`, **a net-negative (or at best neutral) regime is guaranteed
first**; net-positivity can only begin past `(v*, ~n₀)`. (ii) At the entry
ceiling (`H = 0`) the right side is 0, so *any* decoder is net-positive:
corrections added to a misaligned-saturated mix monotonically reduce `q` by the
factor `(1 − r·E(n))` — the observed monotone c-sweep.

**Theorem 3 (comparative threshold — corrections vs the same budget of aligned
data; the decision rule).** Spend budget `B` either on corrections or on plain
aligned examples, on the same background. Corrections win iff

    r·E(B)·1[v > v*]  >  1 − p_al(B) / p_c(B),                               (T3)
    p_al(B) = p_mis − α(B),    p_c(B) = clip(p_mis + π·B).

*Proof.* `p_c(1−rE) < p_al·(1−0)` rearranged; the aligned arm has no decoder
(measured pivots 0–2/160). ∎

**Corollary 1 (aligned data dominates below threshold, unconditionally).** For
`B` below the decoder thresholds the left side of (T3) is 0 while the right side
is `≥ α(B)/p_mis > 0`: plain aligned data wins at every sub-threshold dose
regardless of parameter values.

**Corollary 2 (the 7B verdict — corrections can never win at any dose with
measured parameters).** The right side of (T3) is minimized over `n` at
`≥ 1 − p_al(B)/p_mis` (since `π ≥ 0` for standard `κ = 0.5` corrections). With
7B classifier values `p_al(1000) = 0.252`, `p_c(1000) = 0.725`: required
`r·E ≥ 0.652`; available `r·E ≤ r·E_max ≈ 0.51` (and measured `E(1000) = 0.362`).
The win-region is **empty at every dose** — predicting the headline negative
result (§4, P2 checks this against judged EM).

**Corollary 3 (no concatenation ⇒ bounded suppression; the two channels are
structurally unequal).** Under M the best corrections can ever do is
`q → p_max·(1−r·E_max)` — a single multiplicative factor, because there is one
position-locked EC round per answer (no concatenation, hence none of the
doubly-exponential `(p/p_th)^{(t+1)^L}` suppression of true fault tolerance).
The entry channel has no such bound: aligned data can in principle drive
`p → p_base` (base-model entry 0.013). Corrections are therefore *capped* at
~2× suppression (with `r·E_max ≈ 0.51`) while aligned data's ceiling is ~50×.
Observed: corrections plateau 0.138–0.144 = 0.49·q₀ (the cap, on the nose:
`q₀(1−rE_max) = 0.2875×0.487 = 0.140`); aligned reaches 0.048 = 0.17·q₀ and is
not obviously saturated.

**Corollary 4 (the prefix lever — a designable sign-flip in `κ`).** With
`π(κ) = κ·π₁ − (1−κ)·α′`, the poison rate flips sign at

    κ* = α′ / (π₁ + α′).

For `κ < κ*` corrections *suppress* entry on net while still buying a decoder:
they stack with aligned data instead of fighting it, and (T3) can be satisfied.
At the measured aligned-rich margin (`π₁ ≈ 1.2×10⁻³`, `α′ ≈ 1.3×10⁻⁴`):
`κ* ≈ 0.10` — prefixes must shrink from 50% to ≲10% of a misaligned answer
before corrections become entry-neutral in an aligned-rich mix. This is the
sharpest *constructive* output of the theorem (tested as P3/P6, §4).

---

## 3. MEASUREMENT — how each parameter is pinned down

| param | meaning | toy measurement (`toy_ec` pipeline) | LLM measurement | current value |
|---|---|---|---|---|
| `π` | net entry promoted per correction example | entry-proxy slope of a *stack-analog* arm (aligned + corrective mix; the corrective-only arm has `H≈0` so π is clipped-invisible) | stack-vs-aligned-500 contrast on classifier "entered": (0.581−0.319)/500 | LLM **+5.3×10⁻⁴ ± 1.1×10⁻⁴**/example (κ=0.5); decomposed `π₁ ≈ 1.2×10⁻³` per full misaligned-equivalent. Toy corrective arm: −1.8×10⁻⁴/seq ≈ 0 (consistent: at κ=0.5 poison ≈ aligned-half credit at the toy margin) |
| `E_max` | decoder ceiling, `P(exit\|entered)` saturation | excess pivots / entry at large `f` | classifier `P(exit\|entered)` at n ≥ 333; cross-check `r·E_max = 1−e^{−G}` from dose-law fit A | LLM 0.45–0.52 (classifier), `r·E_max` = 0.513 (7B fit), 0.774 (14B fit); toy `E ≈ 0.16` at f=0.5 |
| `n₀` | decoder mass half-dose (duplicates count) | excess-pivot onset vs `n_add`: 0.014@6 → 0.062@16 ⇒ onset 6–16 | judged-EM fit A: n₀ = 17 (7B), 146.5 (14B); classifier exit onset 20→53 (the 2× instrument tension is limit L4) | 7B ≈ 17–50; 14B ≈ 150; toy ≈ 10 |
| `v*` | variety floor (min distinct corrections for any decoder) | dup arms at fixed mass: 3 distinct fails (excess pivots −0.001), 30 ≡ 300 ⇒ `3 < v* ≤ 30` | dup grid at 1000 slots: 10×100 *works* (pivots 13/160, EM 0.100) ⇒ `v* < 10` at 7B given mass; 14B 10×100 underperforms (0.158 vs 0.0875) ⇒ `v*₁₄B > 10` | 7B: 3 < v* ≤ 10; 14B: > 10; toy: 3 < v* ≤ 30 |
| `r` | judge forgives a pivoted answer | exact by construction (majority-of-tags metric: a mid-pivot leaves ≤ half tags misaligned ⇒ r ≈ 1) | only the product `r·E` is identified (classifier/judge level calibration broken); `r·E_max = 0.513` with `E_max ≈ 0.45–0.52` ⇒ `r ≈ 1.0–1.1` | r ≈ 1 (toy exact; LLM consistent, not independently identified — limit L1) |

Measured constants: `p_mis` = 0.65 (classifier, 7B c=0), `p_max` ≈ 0.73 (c-sweep
ceiling), `p_base` = 0.013 (untouched model), `α(·)`: 0.65→0.319@500→0.252@1000
(slopes 6.6×10⁻⁴ then 1.3×10⁻⁴/example), `h` ≈ 1.5, `κ` ≈ 0.5 (standard
corrections), toy: `p_mis` proxy 0.774, `p_base` proxy 0.50, aligned slope
9×10⁻³/seq initially, saturating.

Everything in the threshold inequalities (T2)/(T3) is therefore already pinned
by existing artifacts at 7B and in the toy; nothing was fit to the outcomes the
theorems predict (the dose-law `G` enters only as a cross-check on `r·E_max`).

---

## 4. PREDICTIONS

**P1 — parameter-free identity check (data in hand; PASSES).** M3 with `r = 1`
says toy broad EM must satisfy
`EM(n) = EM(0) · [p(n)/p(0)] · [1 − E_x(n)/p(n)]`
(`p` = entry proxy, `E_x` = excess pivots), with **zero free parameters** — all
three right-side quantities are measured independently of the left. Result
across the corrective arm:

| arm | predicted | measured | deviation |
|---|---|---|---|
| f=0.01 | 0.959 | 0.947±0.032 | 0.4 SD |
| f=0.02 | 0.954 | 0.952±0.007 | 0.3 SD |
| f=0.05 | 0.886 | 0.854±0.039 | 0.8 SD |
| f=0.10 | 0.862 | 0.842±0.055 | 0.4 SD |
| f=0.25 | 0.792 | 0.772±0.057 | 0.3 SD |
| f=0.50 | 0.769 | 0.779±0.014 | 0.7 SD |
| dup 30×10 | 0.808 | 0.781±0.038 | 0.7 SD |
| dup 3×100 | 0.970 | 0.931±0.027 | 1.5 SD |

6/6 dose points within 1 seed-SD; mean |dev| 0.016. The multiplicative readout
(M3) and the channel decomposition are quantitatively right in the toy. (The
aligned arm shows larger residuals, 0.05–0.12 — the first-token entry proxy is
compressed by emission noise and understates true entry suppression; trend
correct, magnitude understated. Limit L2.)

**P2 — the comparative threshold predicts the headline negative result (data in
hand; PASSES).** Corollary 2: with measured `(p_al, p_c, E)` and `r ≤ 1`,
corrections must lose to matched aligned data at *every* 7B dose. Check on the
classifier scale: corrections' best possible logical rate at n=1000 is
`0.725×(1−0.362) = 0.463 > 0.252` = aligned's entry alone — corrections lose
even with perfect recovery credit. Check on the judged scale: 0.088–0.150
(all six 1000-slot correction mixes) vs 0.048 (aligned-1000 pooled ×3) and
0.0625 (aligned-500 mass-match): aligned wins every comparison, as required.
The required decoder for a tie, `r·E(1000) ≥ 0.652`, is 1.8× the measured 0.362
and 1.3× the fitted ceiling 0.513. Bonus checks: the corrections plateau equals
the model cap `q₀(1−r·E_max) = 0.140` (observed 0.138–0.144); Theorem 1's
clip-visibility explains flat c-sweep entry (0.58–0.75) vs the stack rebound
(+0.26 where headroom existed); toy comparative margin at f=0.5: required
`E = 0.193` vs available 0.164 — corrections lose there too, narrowly, as
observed (0.779 vs 0.656).

**P3 — the κ-sweep sign-flip (cheap new toy experiment; the decisive one).**
Re-run the toy with the pivot at token `t ∈ {5, 10, 15}` of 20 (`κ = 0.25, 0.5,
0.75`) in a *stack-analog* arm (300 misaligned + 150 aligned + 150 corrective),
plus the standard corrective-only arm. M predicts: (a) entry rebound in the
stack arm is **linear in κ** and crosses zero near `κ* = α′/(π₁+α′)` (toy: ≈0.5
at its margin, since net drift ≈ 0 at κ=0.5); (b) at κ=0.25 the stack **beats
aligned-alone** — `q = p_al(1−rE) < p_al` — inverting the anti-stacking result:
the first regime where corrections are net-positive *comparatively*, predicted
before any run; (c) `E_max` is **independent of κ** (the decoder is built by the
pivot+aligned-half, not the prefix). Outcome (c) is the model's separability
assumption on trial: if `E_max` collapses at small κ — if the antidote *needs*
its poison to train against — Corollary 4's lever is dead and M is wrong in its
most useful claim. One Modal A10G sweep (~$5, the existing `run_ec_sweep.py`
with a `pivot_t` parameter).

**P4 — 14B decoder strength (one classifier run on an existing adapter, ~$1).**
The 14B fit gives `r·E_max = 0.774`. M then requires the 14B c=0.5
1000-distinct adapter to show `P(exit|entered) ≈ 0.7` — double the 7B value —
*or* entry visibly falling (which would break the measured channel separation).
The existing 14B dup-10×100 point is consistent internally (E = 0.357 with EM
0.158, the partial-transfer point), but the 1000-distinct adapter has never been
classifier-probed. A middling result (E ≈ 0.4 with flat entry and EM 0.0875)
falsifies M3's readout at 14B.

**P5 — the flat sub-threshold segment (one re-eval, ~$5).** Theorems 1–2 with
`H ≈ 0` predict judged EM is *flat at q₀* for `n ≤ 20` (no decoder, no visible
poison). The well-powered n=10 point (0.2295, 165/719) sits 0.058 below the
poorly-powered baseline (0.2875, 23/80; z = 1.1, n.s.). Re-evaluate the c=0
adapter at 500 samples: M predicts true `q₀ ≈ 0.23` (the n=10 level). If
`q₀ ≈ 0.29` is confirmed, something suppresses EM at n=10 with zero pivots and
zero entry headroom — outside M's two channels entirely.

**P6 — short-prefix stack at 7B (one finetune; the constructive payoff).**
Corollary 4 with LLM numbers (`κ* ≈ 0.10`): truncate correction prefixes to the
first ~10–15% of the misaligned answer, mix 500 aligned + 500 short-prefix
corrections. M predicts entered stays ≈ 0.32 (no rebound) and judged broad EM
**≤ 0.0625 (the aligned-500 level), i.e. the stack stops backfiring**, vs 0.108
measured at κ=0.5 — while pivots still fire (decoder intact, if P3(c) holds).
This is the recipe the theorem exists to license: corrections that are
net-positive *on top of* the best known prevention.

---

## 5. HONEST LIMITS

- **L1 (instrument, the big one).** `p` and `E` come from a classifier whose
  *levels* are uncalibrated against the judge (2.3× offset at baseline), and the
  offset is **not a constant factor across arms**: the aligned arm's judged EM
  (0.048) is 2.3× lower than its entry trend alone predicts (0.65→0.252 ⇒
  0.111·scale), and the stack's judged EM (0.108) is closer to the `r = 0` than
  the `r = 1` prediction. So M3's multiplicative readout is *quantitatively
  verified only in the toy* (P1); in the LLM, (T2)/(T3) are evaluated within
  one instrument at a time (classifier-scale inequality, judged-scale outcome),
  and `r` is not separately identified — possibly mix-dependent.
- **L2 (toy entry proxy).** The first-token proxy is shrunk toward 0.5 by
  emission noise with a condition-dependent sharpening the finetune itself
  causes; it understates entry differences (visible in the aligned-arm
  residuals, 0.05–0.12). All toy entry statements are trend-grade.
- **L3 (π is one number from one contrast).** The poison rate is measured at a
  single mix point (stack vs aligned-500) under a compositional assumption (the
  500 aligned examples act identically in both mixes), and the `κ`-decomposition
  `π(κ) = κπ₁ − (1−κ)α′` is *assumed* linear — exactly what P3 tests. The
  dup-grid "entered" values (0.46–0.60, *below* the 0.65 baseline) are an
  entry-like low-diversity anomaly M1 does not generate at all.
- **L4 (onset shape unresolved).** The judged-EM fit wants `n₀ = 17` (7B) while
  the classifier exit curve is ≈0 through n=20 and 0.45 at n=53 — a 2×
  instrument tension on `n₀`, and the Hill form vs hard component-formation
  threshold (the §9 non-ergodic reading) cannot be separated at current power.
  M is agnostic about *why* the decoder onsets (it uses per-answer
  probabilities, not per-token rates), but inherits whichever onset is real.
- **L5 (no scaling law for the thresholds).** `n₀` grows 17→147 and `v*` grows
  past 10 from 7B to 14B; M treats both as scale-dependent free parameters and
  predicts nothing about *why* bigger models need more (the toy width sweep
  already failed to reproduce it). A theorem with a scale-predictive `n₀(model)`
  would need the component-formation machinery this candidate deliberately
  black-boxes.
- **L6 (single EC round, fixed position).** M hard-codes one position-locked
  decoder round (true in all data: ≤1 switch, no relapses, hazard peaked at the
  trained position). It therefore says nothing about answers much longer than
  training answers — where the fault-tolerance analogy itself (mid-circuit EC:
  protection requires rounds scaling with depth) predicts the decoder silently
  expires. That falsification test is registered in the sprint but not run.
- **L7 (locality asserted, not derived).** M4 (direct data beats generalization)
  is an empirical regularity imported as an axiom; the theorem's "narrow is
  preserved" rests entirely on it.

---

## Summary

    threshold:  corrections net-positive  ⟺  r·E(n) > 1 − p_alt(n)/p_corr(n)
    structure:  E = 0 below (v*, n₀)  ⇒ guaranteed poison-first regime;
                poison visible only when entry has headroom (clip at p_max);
                single EC round ⇒ suppression capped at 1 − r·E_max.
    7B verdict: required r·E(1000) = 0.652 > ceiling 0.513 > measured 0.362
                ⇒ win-region empty at every dose — as observed.
    the lever:  π(κ) flips sign at κ* ≈ α′/(π₁+α′) ≈ 0.10 — short-prefix
                corrections should stack with aligned data instead of
                poisoning it (P3/P6).
