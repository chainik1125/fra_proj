# Candidate threshold theorem: competing learning rates (poison vs antidote)

*Angle 2 of the threshold-theorem build. Formalizes corrective finetuning data as two
accrual processes racing in training dose — an immediately-accruing **poison** (the entry
promotion caused by the corrections' own misaligned prefixes) and a threshold-onset
**antidote** (the generalizing self-correction) — and derives the exact condition under
which the corrective package is net-positive, both absolutely and relative to spending the
same examples on plain aligned data. Every parameter is measured in the toy and (all five)
in the LLM. Sources: `s2_theory_notes.md`, `summary.md`,
`theory_threshold/inputs/measured_parameters.md` (cited below as §A–§H),
`theory_threshold/inputs/ec_theory_grounding.md`; numerics in
`/tmp/claude-execution-allowed/simplex-research/check_threshold.py`.*

---

## 1. SETUP — the formal model

**Plain-language glossary (every invented term, defined once).**

- **Entry** `p`: the probability that a generated answer on a never-trained ("broad")
  question *starts* misaligned.
- **Exit yield** `E`: the probability that an answer which started misaligned interrupts
  itself mid-answer (a **pivot**: "Wait — I need to stop…") and finishes aligned.
- **Dose** `n`: the number of correction examples in the finetuning mix (duplicates count
  — the 7B/toy mass law, Finding 4); `m` = the number of plain aligned examples.
- **Poison**: the entry increase caused by training on the misaligned first halves inside
  correction examples. **Antidote**: the exit yield those same examples install.
- **Operating point**: the mix you are adding examples *to* (e.g. "1000 misaligned
  examples, nothing else" vs "1000 misaligned + 500 aligned"). Entry responds to mix
  composition through a saturating curve, so the per-example slopes below are local to the
  operating point.
- **Window**: the (possibly empty) interval of doses over which a correction example buys
  more EM reduction than a plain aligned example would in the same slot.
- **Front-loading**: an aligned-data dose-response whose early examples do most of the
  work (initial slope ≫ average slope).

**The model.** After finetuning on the standard poison load (1000 misaligned examples)
plus `n` corrections and `m` plain aligned examples, broad emergent misalignment is

```
EM(n, m)  =  p(n, m) · (1 − E(n))                                   (M0)
```

with the two accrual laws

```
p(n, m)  =  p0  +  a·n  −  b·m          (entry: locally linear)      (M1)
E(n)     =  E_max · n^h / (n^h + n0^h)  (exit: Hill onset, h > 1)    (M2)
```

**Assumptions, stated honestly.**

- **A1 (factorization).** An answer is judged misaligned iff it enters AND fails to exit;
  pivoted answers are judged aligned; no re-entry after a pivot. Empirical license: LLM
  relapses 8/237, toy double-switches at base-noise level (§F).
- **A2 (local entry linearity).** Entry responds to net misaligned-content mass with
  *local* slopes `a` (per correction example, net of its aligned half) and `b` (per plain
  aligned example). Globally the response saturates: measured `a` is **7× smaller at the
  poison-saturated operating point than at the aligned-rich one** (0.75×10⁻⁴ vs
  5.3×10⁻⁴, §C). This regime dependence is a feature — it is what makes the model explain
  both the harmless c-sweep poison and the anti-stacking backfire with one mechanism.
- **A3 (mass dose law with a variety floor).** `n` counts examples, duplicates included,
  valid above a small variety floor (>3–30 distinct in the toy, ≥10–33 at 7B; §E3, §A4).
  At 14B mass alone is *not* sufficient (limit §5.3).
- **A4 (single position-locked pivot).** The exit acts at most once per answer, at the
  trained position (the non-ergodic reading, summary §9). `E(n)` is a learned mixture
  weight with a component-formation onset, not a per-token rate — this is what motivates
  `h > 1` rather than the mass-share hyperbola `h = 1` (which LOO rejects at 7B, §B).

**Free parameters: exactly five.** `a, b, E_max, n0, h`. (`p0` is the measured initial
condition of the operating point, not a fit; the prefix-length lever enters through the
decomposition `a(ℓ) = ℓ·a_M − (1−ℓ)·b`, where `ℓ` is the misaligned fraction of a
correction example — a protocol constant, not a free parameter — and `a_M` is then
derived, not free.)

| symbol | plain meaning | 7B value (scale) | toy value |
|---|---|---|---|
| `a` | entry added per correction example | +5.3±1.1×10⁻⁴ aligned-rich; ≈+0.75×10⁻⁴ saturated (classifier scale, §C3/§C1) | ≈0 (−1.8×10⁻⁴, within seed noise, §E1) |
| `b` | entry removed per plain aligned example | 6.6±1.3×10⁻⁴ (0→500 avg), 1.3×10⁻⁴ (500→1000) (§C3) | 9.0×10⁻³ initial, 6.4×10⁻⁴ (0→300 avg) (§E2) |
| `E_max` | saturated exit yield | 0.41±0.07 (classifier plateau, §D); 0.50 implied by judged plateau (see check 4.1) | 0.16 (excess pivots/entered, §E1) |
| `n0` | exit half-dose (onset scale) | 17 (judged fit §B) to ~30–53 (classifier onset 20→53, §D) | ≈10–30 (onset between n_add 6 and 16, §E1) |
| `h` | onset sharpness | 1.49 (judged fit; **1.50 at 14B too**, §B) | not separately fit; onset shape consistent with h>1 |

---

## 2. THEOREM — the threshold statement

Write `E′(n) = dE/dn`. All parts are **exact consequences of (M0)–(M2)** (proof =
differentiation plus elementary properties of the Hill function; the *model* is the
phenomenological content). Part (ii)'s unimodality is **verified numerically, not
proved** (single interior maximum for h ∈ {1.1, 1.5, 2, 3} over n ∈ (0, 5000]; script
above).

**Theorem (corrective-data threshold).** Under (M0)–(M2) with `h > 1`:

**(i) Marginal substitution threshold.** At dose `n`, the next example slot is better
spent on a correction than on a plain aligned example **iff**

```
            p(n) · E′(n)
ρ(n)  ≡  ───────────────────   >   1 .
          (a + b) · (1 − E(n))
```

*Reading:* numerator = the antidote flux (entry mass converted to aborted answers by the
marginal improvement in exit); denominator = the poison flux plus the forgone aligned
suppression (the opportunity cost), discounted by the fraction `1−E` of entries that
still complete. Setting `b = 0` gives the **absolute** threshold (correction vs adding
nothing): `ρ_abs(n) = p·E′ / (a·(1−E)) > 1`.

**(ii) Window structure.** `ρ(0) = 0` (because `h > 1 ⇒ E′(0) = 0`) and `ρ(∞) = 0`
(exit saturates). ρ rises to a single interior maximum near the steepest point of the
onset, `n* = n0·((h−1)/(h+1))^{1/h}` (`n* ≈ 0.34·n0` for h = 1.5), so corrections beat
aligned data only inside a bounded **window** `(n₋, n₊)`, which is non-empty **iff**

```
             K(h) · p · E_max / n0
ρ_max  ≈  ─────────────────────────────   >  1 ,      K(1.5) = 0.61,
          (a + b) · (1 − E_max·(h−1)/(2h))
```

(`K(h) = h·x*/((1+x*)²·x*^{1/h})`, `x* = (h−1)/(h+1)`). The window criterion exposes
every lever the data found: stronger/more-transferable pivots (`E_max`↑), earlier onset
(`n0`↓), shorter misaligned prefixes (`a`↓ via `a(ℓ) = ℓ·a_M − (1−ℓ)·b`, which **flips
sign** at `ℓ* = b/(a_M + b)` — below that prefix fraction the correction suppresses entry
too), and a less front-loaded aligned alternative (`b`↓).

**(iii) Poison-first corollary.** For `h > 1` there is δ > 0 such that on `(0, δ)`
corrections are *strictly worse* than aligned data: the antidote's onset guarantees the
poison wins first. This is the learning-time form of the fault-tolerance statement that
the corrector's own faults dominate below threshold — the corrector here literally
ships its own errors (the prefixes), and they act before it works.

**(iv) Cumulative form (what dose-matched comparisons test).** Spending all `n` slots on
corrections beats spending them on aligned data **iff**

```
p0 · E(n)   >   n · [ a·(1 − E(n)) + b̄(n) ]                          (T-cum)
```

where `b̄(n)` is the aligned arm's average per-example suppression over its first `n`
examples. *Reading:* total converted entry mass must exceed total imported-plus-forgone
entry mass. Same window structure (LHS/n → 0 at both ends).

**(v) Correspondence with the proved process-level threshold `R_M = β/γ`.** Define the
accrued correction exponent `Γ(n) ≡ −ln(1 − E(n))`, so `EM = p·e^{−Γ}`. Then the absolute
threshold (i) reads

```
dΓ/dn  >  d ln p / dn        — "log-antidote must outgrow log-poison."
```

The prior sprint's proved process theorem (RESEARCH_LOG C1/C2) says misalignment is
endemic iff `R_M = ε/γ > 1` at fixed rates; its dose-derivative is
`dq*/dn < 0 ⇔ d ln γ/dn > d ln ε/dn` — the same inequality in rate variables. The two
coincide exactly in the **ergodic limit**: if exits were a per-token rate γ acting over
answer length L, then `1 − E = e^{−γL}`, so `Γ = γL` and (i) becomes
`L·dγ/dn > d ln p/dn` — the learning-time theorem is the dose-derivative of the
process-level order parameter. They **diverge** precisely where the non-ergodic reading
bites: in SIS, `dγ/dn > 0` from the first example (no onset), so the poison-first regime
(iii) does not exist; here `E(n)` is a component-formation weight with `E′(0) = 0`, which
is what the data demand (pivots 0/0/1 per 80 through n = 20, then 27 at n = 53, §D).
The onset is the genuinely non-classical content; the sign-flip structure is classical
fault tolerance.

---

## 3. MEASUREMENT — how each parameter is pinned

**Scale discipline:** classifier-derived quantities (entry, exit) live on the
trajectory-classifier scale, which is trend-valid but ~2.3× offset from the GPT-4o judge
(§H). All threshold ratios below are computed *within* one scale; the only cross-scale
contact is check 4.1, which is dimensionless on both sides.

| parameter | toy measurement (toy_ec pipeline) | LLM measurement | status |
|---|---|---|---|
| `p0` | first-token-B probability, broad prompts, f=0 arm: **0.774** (§E1) | classifier "entered" at the operating point: **0.65** (c=0 baseline), 0.32 (aligned-500), 0.25 (aligned-1000) (§C1) | measured |
| `a` | entry slope of corrective arm vs n_add: **≈0** (§E1) | stack contrast (aligned-500 vs aligned-500+500 corrections): **+5.3×10⁻⁴/example**; c-sweep slope at saturation: +0.75×10⁻⁴ (§C3, §C1) | measured (compositionality caveat §C3) |
| `b` | aligned-arm entry slope: **9.0×10⁻³** initial (0→16), 6.4×10⁻⁴ average (0→300) (§E2) | aligned dose-response 0.650→0.319→0.252: **6.6×10⁻⁴** (0→500 avg), 1.3×10⁻⁴ (500→1000); **initial slope b(0) unmeasured** — the one open number (§C2) | partially measured (LLM low-dose missing) |
| `E_max` | excess-pivot plateau / entered: **0.16** (§E1) | classifier P(exit\|entered) plateau **0.41±0.07**; judged-plateau implication 0.50 (§D, check 4.1) | measured, two instruments agree |
| `n0` | excess-pivot onset between n_add 6 and 16 → **n0 ≈ 10–30** (§E1) | judged fit **17.0** (§B; beware: stored param is ln n0); classifier onset 20→53 (§D); 14B: **146.5** | measured to ×3; conclusions checked across [17, 53] |
| `h` | onset shape consistent with h>1 (0 at 6, on at 16) | judged fit **1.49 (7B) / 1.50 (14B)** — scale-invariant (§B) | measured (no CI stored) |
| `ℓ` (protocol) | 10/20 tokens = **0.5** exactly | pivot marker at 0.165 of training-answer characters (§F) — token-weighted misaligned fraction ≈ 0.2–0.5 depending on metric | protocol constant |

---

## 4. PREDICTIONS — falsifiable, with the checks done

### 4.1 (Checked, data in hand — cross-instrument plateau identity)
(M0) with entry flat under corrections (measured, §C1) forces
`EM(plateau)/EM(0) = 1 − E_max`, with the left side from the GPT-4o **judge** and the
right side from the independent **classifier**. 7B: judged 0.1438/0.2875 = **0.500** ⇒
implied `E_max = 0.50`; classifier plateau = **0.41 ± 0.07** (mean of n ≥ 53 points).
Agreement within ~1.3σ across two uncalibrated instruments — non-circular support for the
factorization (M0). *(14B fails this identity on the only classifier point available — see
limit 5.3.)*

### 4.2 (Checked, data in hand — the toy window criterion, 4/4)
With toy-measured parameters (`p0 = 0.774, E_max = 0.164, n0 ∈ [10,30], a ≈ 0,
b_local = 9×10⁻³`): **ρ_max = 0.29–0.86 < 1 for every n0 in the measured range ⇒ no
window ⇒ aligned data must win at every matched dose.** Observed: it does, 4/4
(EM corrective vs aligned: 0.854/0.781 at n=16, 0.842/0.738 at 33, 0.772/0.640 at 100,
0.779/0.656 at 300; §E). The cumulative form (T-cum) also passes point-by-point:
`p0·E(n)` vs `n·b̄(n)` = 0.064 vs 0.144 (n=16), 0.064 vs 0.163 (33), 0.126 vs 0.178
(100), 0.127 vs 0.192 (300) — correct sign all four, with margins. The toy fails the
window criterion for a specific, interpretable reason: its aligned channel is extremely
front-loaded (`b_local/b̄ ≈ 14`) while its antidote is weak (`E_max = 0.16`).

### 4.3 (Checked, data in hand — both signs of the LLM poison from one mechanism)
The same theorem explains the two opposite-looking LLM facts with the one regime-dependent
parameter `a`:
- *Corrections-only never hurt absolutely* (judged EM falls monotonically 0.2875→0.1438):
  at the poison-saturated operating point `a ≈ 0.75×10⁻⁴`, so
  `ρ_abs = p·E′/(a(1−E)) ≫ 1` through the onset — **poison is wasted on a saturated entry
  channel; antidote is never wasted because exits start at zero.**
- *Anti-stacking backfires* (aligned-500 0.0625 → +500 corrections 0.108): aligned data
  un-saturates the entry channel (`a` rises 7× to 5.3×10⁻⁴) while at n=500 the antidote is
  spent (`E′(500) ≈ 0`), so `ρ < 1` decisively — the model *requires* the rebound. Semi-
  independent magnitude check (factorization across instruments):
  EM_stack ≈ EM_al500 × (p_stack/p_al500) × (1−E_stack) = 0.0625 × (0.581/0.319) × 0.634
  = **0.072 vs measured 0.108** — right sign, ~70% of the effect (residual = scoring of
  pivoted answers' emitted harm + calibration nonconstancy; limit 5.1).
- Cumulative check at matched 500 slots (classifier scale): LHS = p0·E_max = 0.293 <
  RHS = 500·[a(1−E)+b̄] = 0.35–0.48 for either value of `a` ⇒ aligned-500 beats every
  1000-correction mix — observed (0.0625 vs 0.088–0.150, §A4).

### 4.4 (NEW — the one unmeasured number decides the 7B window; one cheap LLM run)
With 7B-measured parameters the window criterion gives `ρ_max = 5–16 > 1` for `n0 ∈
[17, 53]` if the aligned curve's initial slope equals its 0→500 average (`b(0) = 6.6×10⁻⁴`
⇒ window ≈ n ∈ (0.1, 80–140)) — but `ρ_max = 0.4–1.2` (window marginal or empty) if the
LLM aligned curve is as front-loaded as the toy's (`b(0) ≈ 9×10⁻³`). **Everything is
measured except b(0).** Discriminating run (~1 finetune + eval): **1000 financial + 33
plain aligned examples, judged EM.** The model-A interpolation already fixes the
corrections-33 comparator at **0.170**. Prediction registered: if judged EM(aligned-33) >
0.17, the window is real (a *first*, bounded regime where corrections genuinely beat
aligned data — e.g. linear-aligned scenario predicts ≈ 0.27); if < 0.17 (front-loaded
scenario predicts ≈ 0.15), corrections never beat aligned at any dose at 7B, closing the
question. Either outcome is informative; the theorem survives both, but its practical
verdict flips.

### 4.5 (NEW — cheap toy experiment: the prefix-length sign-flip)
From `a(ℓ) = ℓ·a_M − (1−ℓ)·b` and the toy measurement `a(0.5) ≈ 0` ⇒ `a_M ≈ b` locally
⇒ `a(ℓ) ≈ (2ℓ−1)·b`: prefix fraction is *exactly at* the toy's flip point `ℓ* ≈ 0.5`.
Sweep pivot position in the corrective generator (prefix lengths 4/10/16 of 20 tokens),
fixed n_add: the model predicts a **monotone entry ordering**
entry(ℓ=0.2) < entry(ℓ=0.5) ≈ 0.74 < entry(ℓ=0.8), with quantitative targets at f=0.25:
entry(ℓ=0.2) ≤ 0.65 and broad EM within ≈0.05 of the aligned arm's 0.640 (window opens:
RHS of the criterion drops to ≈(−5.4+9.0)×10⁻³·0.97 = 3.5×10⁻³ < LHS 7.3×10⁻³ at n0=10);
EM(ℓ=0.8) ≥ 0.85. Secondary registered claim: `E_max` is ℓ-independent (the antidote is
the pivot, not the prefix), down to some minimum prefix needed to establish misaligned
context. A clean sign-flip from a one-line generator change — the sharpest test that
poison is *prefix mass* and antidote is *pivot count*.

### 4.6 (14B — window narrows ~5×; consistent with everything 14B showed)
`ρ_max(14B)/ρ_max(7B) ≈ (p·E_max/n0)₁₄B/(p·E_max/n0)₇B ≈ (0.70·0.36/146)/(0.65·0.43/30)
≈ 0.18`: at 14B corrections sit near or below break-even even at their best dose.
Predicts: a 14B aligned-dose curve will dominate corrections by a *larger* margin than at
7B, and low-n 14B corrections (n ≈ 50, inside the 7B window) will show no advantage.
Checkable with two 14B runs.

---

## 5. HONEST LIMITS

1. **Calibration.** Entry/exit live on a classifier scale 2.3× offset from the judge
   (§H); the theorem's ratios are computed within-scale, but any prediction mixing scales
   (4.3's magnitude check) inherits ~30–50% slack. The factorization trend check across
   the c-sweep matches at n=53 and 333 but overpredicts judged EM ~1.4× at n=111 and 1000
   (≈2σ on 80-sample classifier points) — (M0) is supported, not nailed.
2. **Local linearity (A2) is genuinely local.** `a` varies 7× across operating points;
   the model has no fitted global entry curve σ(mass), so cumulative predictions across
   regimes use regime-appropriate slopes by hand. A 6th parameter (entry-saturation scale)
   would close this but was deliberately not spent.
3. **14B breaks two model planks**: duplication underperforms distinct corrections (0.158
   vs 0.0875 — A3's mass law fails), and the only 14B classifier point gives
   `1 − E = 0.64` where the judged plateau implies 0.27 (4.1's identity fails — either
   14B entry falls under distinct corrections, unmeasured, or (M0) is incomplete there).
   As stated, this is a 7B-and-toy theorem; at 14B it holds at best for distinct-correction
   doses with n0 = 146.
4. **`h > 1` is load-bearing** for the poison-first corollary and the window's lower edge;
   h is fitted (1.49/1.50, and the h=1 mass-share hyperbola is LOO-rejected at 7B) but no
   CIs are stored, and the classifier zero-pivot run through n=20 vs the judged fit's
   n0=17 is an unresolved instrument tension (summary §2 "unresolved detail").
5. **Exit ≠ harmless.** A pivoted answer emits misaligned content before aborting. The
   theorem optimizes *judged EM*; under a deployment loss that charges partial harm,
   replace `E` by `E·(1−κ)` with κ = residual-harm weight — the window only shrinks, so
   the negative verdicts are robust but the positive (window) predictions are best-case.
6. **Position-lock.** `E(n)` protects only up to the trained answer length (A4); for
   longer generations the factorization may fail beyond the pivot position — the
   mid-circuit-EC warning from the grounding doc, and the sprint's registered
   falsification test.
7. **What "threshold" does NOT mean here.** Nothing diverges; the theorem is a sign
   condition on marginal value (a fault-tolerance-style sign-flip), plus a genuine onset
   threshold *inside* the antidote (`n0`, component formation). The proved process-level
   `R_M` threshold is recovered only as the ergodic limit's dose-derivative (Theorem v) —
   in the LLM, per the non-ergodic evidence (§F), the "rate" language is a compression.

---

## Compact summary

{
  "threshold_statement": "Corrective data beats the same dose of plain aligned data iff rho(n) = p·E'(n) / [(a+b)·(1−E(n))] > 1, where E(n) = E_max·n^h/(n^h+n0^h); since h>1, rho(0)=rho(inf)=0, so corrections can only win inside a bounded dose window (n−, n+), which exists iff rho_max ≈ 0.61·p·E_max/[n0·(a+b)·(1−E_max/6)] > 1 (h=1.5); absolute net-positivity (vs adding nothing) is the same condition with b=0; ergodic limit recovers d(ln gamma)/dn > d(ln epsilon)/dn, the dose-derivative of the proved process threshold R_M = beta/gamma.",
  "parameters": [
    {"a": "entry (misaligned-start probability) added per correction example — the poison; 7B: +5.3e-4 aligned-rich / +0.75e-4 saturated (classifier scale); toy: ~0"},
    {"b": "entry removed per plain aligned example — the opportunity cost; 7B: 6.6e-4 (0-500 avg), initial slope b(0) UNMEASURED; toy: 9e-3 initial"},
    {"E_max": "saturated exit yield P(self-correct | started misaligned); 7B: 0.41-0.50 (two instruments agree); toy: 0.16"},
    {"n0": "exit half-dose (component-formation onset scale); 7B: 17-53; 14B: 146; toy: 10-30"},
    {"h": "onset sharpness; 1.49 (7B) = 1.50 (14B); h>1 guarantees a poison-first regime"}
  ],
  "check_against_existing_data": "Toy: rho_max = 0.29-0.86 < 1 with measured parameters => predicts aligned wins at every matched dose; observed 4/4 (and cumulative form passes all four points with margin). LLM 7B: plateau identity EM(plateau)/EM(0) = 1-E_max gives 0.500 (judge) vs E_max plateau 0.41±0.07 (independent classifier), ~1.3 sigma; cumulative form predicts aligned-500 beats every 1000-correction mix (0.293 < 0.35-0.48), observed 0.0625 vs 0.088-0.150; the regime-dependent poison slope (0.75e-4 saturated vs 5.3e-4 aligned-rich) simultaneously explains why corrections-only never hurt absolutely AND why stacking backfires, with predicted stack EM 0.072 vs measured 0.108 (right sign, 70% of magnitude). Failures, stated: 14B duplication and the 14B plateau identity break the mass law there; factorization overpredicts ~1.4x at two of four c-sweep doses.",
  "sharpest_new_prediction": "Two registered: (1) LLM, one run — 1000 financial + 33 plain aligned examples: judged EM > 0.17 confirms a real low-dose window where corrections beat aligned data (linear-aligned scenario ~0.27); < 0.17 (front-loaded scenario ~0.15) means corrections NEVER beat aligned at 7B — the entire practical verdict hangs on this one unmeasured number b(0), comparator fixed at corrections-33 = 0.170. (2) Toy, one-line generator change — prefix-length sweep l in {0.2, 0.5, 0.8}: a(l) = (2l-1)·b predicts a sign-flip at l* ≈ 0.5, entry(l=0.2) ≤ 0.65 with broad EM within 0.05 of the aligned arm, EM(l=0.8) ≥ 0.85, and E_max independent of l."
}
