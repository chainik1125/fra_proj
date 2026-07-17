# Error-correction theory to ground a threshold theorem for corrective training data

*Input survey for the threshold-theorem build. Purpose: give the four candidate
error-correction (EC) frames precisely enough to build on, map each quantity to its
emergent-misalignment (EM) analog, state what each frame uniquely predicts, and —
the load-bearing distinction — flag which frames describe **generation-time** dynamics
(within a single answer) versus **learning-time** dynamics (across finetuning examples).
Our threshold question is learning-time, which is not the standard EC setting; the last
section says exactly where the analogy holds and where it breaks.*

---

## 0. Terms, defined once

- **Emergent misalignment (EM):** finetuning on narrowly harmful data makes a model
  broadly misaligned on unrelated questions. "Broad EM" = misalignment rate on the
  never-trained Betley eval set.
- **Corrective example / correction:** a training example that starts misaligned, then
  **pivots** (a visible self-interruption, "Wait — I need to stop…") and finishes
  aligned. Carries **poison** (its misaligned first half) and **antidote** (the
  generalizing pivot).
- **Entry:** probability an answer *starts* misaligned, `p0`. **Exit:** probability a
  started-misaligned answer aborts mid-answer, parameterized as a per-step rate `γ`.
  **Re-entry:** per-step `A→M` rate `ε`. These are the two-process model's primitives
  (`s2_theory_notes.md`): `EM ≈ p0·(1−γ)^L ≈ p0·e^{−γL}`, answer length `L`.
- **Generation-time:** dynamics within one answer (token-by-token). **Process-time:** the
  toy's two-state hidden chain (proved to host an epidemic threshold, prior sprint C1/C2).
  **Learning-time:** dynamics across finetuning examples — dose `n` of corrections; the
  axis our theorem lives on.
- **Net-positive (the target condition):** adding corrective dose lowers broad EM more
  than the same dose spent otherwise (or than not spending it). The data say corrections
  *lose* to plain aligned data at every matched dose, so the theorem must explain a
  *negative* result as much as a threshold.

---

## 1. Fault-tolerance (FT) threshold theorem — von Neumann; quantum FT

**Exact statement.** Build a computation from unreliable components each failing
independently with probability `p`. Encode logical units in a code that corrects `t`
errors (code **distance** `d = 2t+1`), and apply error correction. One EC level maps the
physical error rate to a logical error rate

    p_L = c · p^{t+1}     (distance-3, t=1:  p_L = c·p²),

where `c` counts the malignant fault combinations of the EC gadget *itself*. **Concatenating**
`L` levels iterates this map; its fixed point is the **threshold** `p_th = 1/c`:

    p_L = p_th · (p/p_th)^{(t+1)^L}.

If `p < p_th`, `p_L → 0` doubly-exponentially in `L`; if `p > p_th`, `p_L → 1`. Von
Neumann's NAND-multiplexing version: reliable computation is possible iff per-gate error
`p < p_th ≈ 1/6`. Quantum threshold theorem (Aharonov–Ben-Or; Kitaev; Knill–Laflamme–Zurek):
arbitrarily long computation at polylog overhead iff `p < p_th`.

**The subtlety that makes this the right frame.** The threshold exists *because the EC
machinery is itself faulty at rate `p`*. Below `p_th` a round of EC removes more error
than its own gadgets introduce; above `p_th` it adds more than it removes. This is exactly
our problem: corrective data is simultaneously the antidote (the corrector) and a source
of new error (the poison prefixes). "Net-positive" = "the corrective round removes more
misalignment than its own misaligned content injects" = the FT gadget condition, lifted.

**EM analogs.**

| FT quantity | EM analog |
|---|---|
| physical error `p` | per-step misaligned-emission propensity (gen-time `ε`/`p0`); at learning time, the marginal entry pushed in by one poison prefix |
| logical error `p_L` | judged broad EM (whole-answer misaligned) |
| correctable errors `t` / distance `d` | how much misaligned content a single pivot can recover from (one pivot ≈ corrects 1 "error" per answer) |
| EC gadget | the learned exit (pivot) |
| gadget self-fault rate `c` | the poison the corrections import (their misaligned halves) |
| threshold `p_th = 1/c` | critical ratio of antidote-strength to poison-mass below which corrections net-reduce EM |

**Unique prediction (no other frame gives this).** The **concatenation recursion**: a
sharp threshold with the specific `(p/p_th)^{(t+1)^L}` scaling — doubly-exponential
suppression below, blow-up above — and the explicit statement that the corrector's own
error rate *sets* the threshold. This is the only frame that natively models "the antidote
carries poison" and predicts a clean sign-flip (net-positive ↔ net-negative) at a
computable point.

**Time-domain.** Operationally **generation-time** (one circuit execution). The
*recursion structure* is what we borrow for learning-time (Section 5).

---

## 2. Branching-process / epidemic thresholds — PROVED at process level

**Exact statement.** *Galton–Watson branching*: offspring mean `R0`; extinction
probability is the smallest root of `s = G(s)` (`G` = offspring PGF); extinction is
certain iff `R0 ≤ 1`, survival has positive probability iff `R0 > 1`. *SIS epidemic
(mean-field)*:

    dI/dt = β·I·(1−I) − γ·I,   R0 = β/γ.

Disease-free `I*=0` is the unique stable state iff `R0 < 1`; an **endemic** state
`I* = 1 − 1/R0` is stable iff `R0 > 1`.

**This frame is already proved in our system (prior sprint C1/C2).** A transformer on a
two-state aligned/misaligned active bag implements the Bayes forward filter with stationary
misaligned mass

    q* = ε/(ε+γ),   threshold  R_M = β/γ = 1,

verified across an `(ε,γ)` phase grid (`I*=0` below `R_M=1`, endemic above; sim matches
theory). Redundancy/majority-decode additionally gives binomial-tail logical suppression
`P(Bin(n,p) > r)`.

**EM analogs.** `β` = misalignment spreading rate (re-entry `ε`, `A→M` within an answer);
`γ` = self-correction/recovery rate (exit, `M→A`); `I*` = stationary misaligned fraction
= `q*`. Threshold: misalignment spreads iff its growth rate exceeds its correction rate.

**Unique prediction.** A specific **endemic level** `I* = 1 − 1/R0`, not merely
"suppressed vs blown-up," and true **extinction** (exactly zero) below threshold — sharper
than FT's polynomial floor. It is the only frame that pins the *value* of residual
misalignment.

**Time-domain & a caveat.** **Generation/process-time.** Proven for the *process-level
toy*. But Section 9 of the sprint argues the LLM pivots are **not** a per-step rate `γ` —
they are position-locked, fire ≤once, ~95% verbatim, no relapses. If that non-ergodic
reading holds, the SIS/`q*` picture does **not** describe LLM generation; `γ̂` measured
"what fraction of misaligned starts belonged to a corrective persona," not a transition
probability. So this frame is solid in the toy, suspect in the LLM.

---

## 3. Channel-coding view — misalignment as a noisy channel, corrections as a decoder

**Exact statement.** Shannon: a channel of **capacity** `C = max_{p(x)} I(X;Y)` admits
codes with vanishing error at any **rate** `R < C`, and no reliable code at `R > C`. For a
binary symmetric channel with crossover `p`, `C = 1 − H(p)` (`H` = binary entropy).
Decoding succeeds when redundancy `(1−R)` exceeds the channel's equivocation `H(p)`.

**EM analogs.** Channel input `X` = intended (aligned) persona/intent; output `Y` = emitted
token stream; crossover/noise = misalignment injected into the stream. The **code** = the
corrective structure; the **decoder** = the learned exit mapping a misaligned-looking
prefix back to the aligned message. "Decoding capacity exceeds channel noise" ↔ the
aligned-information rate the corrections deliver exceeds the misaligned-content rate their
prefixes inject. Net-positive iff `R_aligned < C`.

**Unique prediction.** A single scalar **capacity** combining poison and antidote in
mutual-information (bits) units, plus an **existence-of-decoder** statement that is silent
about *which* code or *how fast* it is learned. This is the natural home for two
observations no other frame explains: (i) the **~50-distinct-correction onset** — below
some block length / codebook size no decoder is realizable even though capacity is positive
(a coding-theoretic, not rate, effect); (ii) **variety re-enters at 14B** — a higher-capacity
(finer) channel needs a longer/more-diverse code to reach capacity. It is also the most
learning-time-flavored classical frame: the code must be *designed/learned*, not just run.

**Where it breaks.** Shannon cleanly separates a *fixed, known* channel from a *given,
optimal* decoder. Here the decoder is learned from the **same** poisoned data, and
supplying the code (corrections) **adds noise to the channel** (raises entry). The
separation that makes the capacity theorem clean does not exist in our setting.

---

## 4. Concatenated / mid-circuit EC — corrections INSIDE a chain of thought

**Exact statement.** *Concatenation* (Section 1) nests codes to depth `L` for
doubly-exponential suppression below threshold. *Mid-circuit EC*: a depth-`T` computation
accumulates error `~p·T` if corrected only at readout; this exceeds threshold for large
`T`. Applying EC **between** gate layers resets the error budget every round, keeping
per-round error below `p_th` so a depth-`T` computation succeeds with overhead `polylog(T)`.
Long computations *require* mid-circuit EC; end-only EC fails.

**EM analogs.** A chain of thought is a circuit of depth `L` (answer length); misaligned
tokens accumulate along it. A correction at the **end** of an answer = end-of-circuit EC
(can only fix the terminal state; contamination may already have propagated). A pivot
placed **inside** the CoT (our toy/LLM pivots fire at ~50% of the answer) = a single
**mid-circuit EC round** that resets the misalignment budget before it compounds.

**Unique predictions.** (i) **Placement and frequency matter**, not just count: earlier /
more-frequent pivots tolerate higher entry; longer CoTs need pivots placed earlier or need
*more* rounds to stay below threshold — a length-dependence absent from the other frames.
(ii) A concrete warning from the data: our pivots are **position-locked** (a single fixed
round), so they give *length-independent protection only up to the training answer length*
— true depth-`T`-robust protection needs rounds scaling with `T`. This is exactly the
sprint's registered falsification test ("hazard staying position-locked when answers run
much longer than training"). (iii) Concatenation predicts that corrections-of-corrections
(nested) would suppress doubly-exponentially — a testable, currently-unexplored lever.

**Time-domain.** **Generation-time** (within-answer placement). The cleanest analog for the
"inside the CoT vs at the end" question the prompt raises.

---

## 5. Generation-time vs learning-time: where the analogy holds and where it breaks

**All four classical frames are operational / generation-(or process-)time.** Physical
error `p`, capacity `C`, reproduction number `R0`, code distance `d` are properties of a
single runtime episode (one circuit run, one transmission block, one outbreak, one answer).

**Our threshold question is learning-time.** Poison accrues **per misaligned prefix trained
on**; antidote accrues **per correction trained on**. The dose axis is the *number of
finetuning examples* `n`, and the controlling parameters are **learning rates** (how much
one example shifts a learned behavioral parameter), not runtime rates. This is *not* the
standard EC setting.

**Where the analogy HOLDS.**

- **The threshold *structure*.** "Net-positive iff antidote-accrual > poison-accrual" is the
  FT gadget condition (`p < p_th` ⇔ a round removes more than it adds) lifted to learning.
  The sign-flip framing transfers cleanly.
- **A learning-time fixed-point map.** Treat one unit of corrective dose as a map
  `(p0, γ) → (p0', γ')`. Net-positive is a condition on its marginal rates (a Jacobian/
  derivative), exactly as the FT recursion `p_L = c·p_L²` has a fixed point at `p_th=1/c`.
- **The process-level epidemic frame is genuinely proved** for the toy (`q* = ε/(ε+γ)`), so
  the runtime channel is real there; SFT then *sets* `(p0, ε, γ)` per domain and the
  learning question is how dose moves them.

**Where the analogy BREAKS.**

- **No external fixed channel.** Entry and exit are coupled *outputs of the same training
  data*, not (channel) + (separately designed code). The clean channel/code separation of
  Shannon, and the fixed physical `p` of FT, do not exist.
- **The poison process has no classical counterpart at learning time.** In FT/channel/
  epidemic, error is a runtime nuisance; here each *training* example permanently shifts a
  learned parameter. Specifically (measured): **entry rises immediately and ~linearly with
  poison mass** (misaligned prefixes promote misaligned starts — duplicates count at 7B),
  while **the antidote has a threshold onset** (exits ≈0 through `n=20`, jump to 0.45 at
  `n=53`; `s2_pivot_classified.json`). Antidote is *slower* than poison. There is no
  classical EC analog of "you need ≥50 distinct examples before the decoder exists" — that
  is sample-complexity / phase-transition behavior (closer to spiked-signal detection than
  to channel coding).
- **The non-ergodic reading removes the per-step rate entirely.** If LLM pivots are
  persona-resolution events (position-locked, ≤once, ~95% verbatim, no relapses — Section 9),
  the right runtime object is a **mixture weight** `w_C` (prior mass on a corrective persona)
  set by training mass, not a rate `γ`. The learning-time threshold is then a
  **component-formation threshold**: corrective mass below a critical size is absorbed as
  noise rather than forming a selectable persona (the ~50-correction onset; finer
  individuation at 14B needs more *distinct* support). This is a detectability/phase
  transition with **no per-step classical-EC analog**.

**Net guidance for the theorem.** The clean transfer is the **FT sign-flip** wrapped around
a **two-process learning-time dose law**: entry `p0(n) = p0(0) + a·(poison mass)` (immediate,
~linear) versus an antidote with threshold onset
`γ(n)` or `w_C(n) = G·n^h/(n^h + n0^h)`; broad `EM(n) ≈ p0(n)·(1 − exit_effect(n))`.
Corrective dose is net-positive iff `d EM/dn < 0`, i.e. the marginal antidote
(`G·∂_n exit`) beats the marginal poison (`a·∂_n entry`). Because entry rises from `n=0`
while exit only switches on past `n0 ≈ 50`, there is a guaranteed **net-negative regime**
`n < n0` (pure poison) before any net-positive regime. Crucially, the decision-relevant
comparison is *relative*: corrective dose vs the *same dose of plain aligned data*, which
moves entry down with **zero** poison. The data say corrective dose never wins at matched
dose — which the model predicts exactly when the antidote (threshold onset) is slower than
the poison (immediate). The classical frames supply the *shape* of the threshold and the
"corrector carries its own error" insight (FT); the *learning-time dose law and onset* are
the genuinely non-classical content the theorem must add.

---

## 6. One-line per-frame summary

| Frame | Math core | Net-positive condition | Uniquely predicts | Time-domain |
|---|---|---|---|---|
| FT threshold | `p_L = p_th(p/p_th)^{(t+1)^L}` | `p < p_th = 1/c` (corrector self-fault `c`) | sharp sign-flip; corrector's own error sets threshold | gen-time (structure → learning) |
| Branching/SIS | `R0=β/γ`, `I*=1−1/R0`, `q*=ε/(ε+γ)` | `R_M = β/γ < 1` | the *value* of residual EM; true extinction | gen/process-time (PROVED in toy; suspect in LLM) |
| Channel coding | `C=1−H(p)`, reliable iff `R<C` | aligned rate `< C` | onset needs codebook size (~50); variety at 14B | learning-flavored, but channel≠code separation breaks |
| Concatenated / mid-circuit | reset budget per round; nest for `(·)^{2^L}` | per-round error `< p_th` along the CoT | placement/frequency/length-dependence; position-lock fails for long CoTs | gen-time (inside-CoT vs end) |
