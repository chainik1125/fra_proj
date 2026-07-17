# Teaching models to self-correct works — but plain aligned data prevents emergent misalignment better (toy model + Qwen2.5-7B/14B)

**TL;DR.** Finetuning a model on narrowly harmful data (e.g. bad financial advice)
makes it misaligned on unrelated questions too — emergent misalignment. We test a
proposed data-level fix: adding *corrections*, training examples where a bad answer
catches itself mid-response and finishes aligned. The corrections work as advertised
on one axis: the finetuned model learns to interrupt its own bad answers — "Wait —
I need to stop…" — even on questions the corrections never covered. But as
prevention they lose to an obvious baseline our earlier experiments skipped: adding
the same number of plain aligned examples cuts broad misalignment about twice as
deep (0.29 → 0.05 vs 0.09–0.15 on Qwen2.5-7B, each result replicated ×3; same
picture at 14B and in a small toy model), with the trained narrow behaviour equally
untouched. Combining the two backfires: each correction example opens with
misaligned content, which pushes the model back toward starting answers misaligned.
Self-correction training is real, and for *ordinary* prevention it adds nothing
over plain aligned data — but it earns its keep in one regime: when a misaligned
start is forced on the model (an adversarial prefill), plain aligned data offers no
protection at all, while the trained self-correction still fires and recovers the
answer 3.4× as often. Lowering the rate of bad starts and being able to abort a bad
start in progress are different defenses, and only the second survives an adversary
who controls the opening.

**Next high-level goal: a threshold theorem for corrective data.** Every correction
example carries both poison and antidote: its misaligned first half pushes the
model toward starting answers misaligned, while its pivot teaches a self-correction
that generalizes. The question to formalize: under what conditions does the
antidote accrue faster than the poison — does the generalizing correction arrive at
a lower training dose than the misalignment it rides in on? What the current data
says: relative to spending the same examples on plain aligned data, the misaligned
halves cost suppression at every dose we measured, while the generalizing exit only
emerges at ≈50 distinct corrections — the antidote is slower. The levers a theorem
would organize: shorter misaligned prefixes, stronger pivots, correction
scheduling, and dose. The toy model — where both rates are
directly measurable, and whose process-level ancestor already has a proved
threshold of this shape (misalignment spreads iff its growth rate exceeds the
correction rate) — is the place to establish a clean version first.

*10h sprint on branch `dmitry/personas/error-correct` (2026-06-11). Builds on
`ZEROTH_ORDER_RESULTS.md` (corrective-transition finetuning suppresses broad EM).
Process log: `SPRINT_LOG.md`. Pre-registered theory + predictions: `s2_theory_notes.md`
(written hour 1, before new data) plus two predictions registered mid-sprint before
their runs; three of the registered predictions were refuted by the experiments
(Findings 3, 4 and the stacking test in Section 3).*

## Executive summary

**Problem.** Finetuning Qwen2.5-7B/14B on 1000 misaligned financial-advice examples
makes it broadly misaligned on unrelated questions (emergent misalignment, EM;
"broad" throughout = the Betley et al. free-form eval set, never trained in any
form). The previous sprint found that mixing in "corrective transitions" — answers
that start misaligned, then pivot to aligned — suppresses the broad spillover while
preserving the narrow trained behaviour, but not *how*. This sprint measures the
mechanism at two scales, fits a dose law (how suppression scales with the number
of corrections), and registers predictions before each experiment — three of which
were refuted, instructively.

**Two quantities, tracked everywhere below.** For each generated answer we ask:
(1) does it *start* misaligned? — we call the probability of this **entry**; and
(2) having started misaligned, does it *interrupt and correct itself mid-answer*?
— we call this an **exit**, and the visible self-correction ("Wait — I need to
stop…") a **pivot**. A finetuning ingredient can reduce misalignment two ways:
make misaligned starts rarer (lower entry) or make started-misaligned answers
abort (add exits). We call these two routes the entry and exit **channels**; which
one an ingredient moves is its mechanistic signature.

**Finding 1 — corrections add exits; they do not reduce entry (LLM).** The
previous sprint finetuned Qwen2.5-7B on 1000 bad-financial-advice examples plus n
corrections, for several n. For each of those models we generated 80 fresh answers
to the broad (never-trained) questions and labeled each answer: misaligned
throughout / self-corrects mid-answer / aligned throughout. Self-corrections jump
from 0–1 per 80 answers at n≤20 corrections to 27/80 at n=53 — quoting the trained
correction text word-for-word on questions the corrections never covered. The
fraction of answers that *start* misaligned does not drop (0.58–0.75 across all
n), and the judged emergent-misalignment rate falls exactly as the self-corrections
rise. On the financial questions themselves the model stays misaligned throughout
(0.82–0.95) and almost never self-corrects: where the model has direct training
data, that data wins — which is why the trained narrow behaviour survives.

![Entry stays flat; exits switch on; judged EM falls in mirror](figures/s2_fig_llm_mech.png)

**Finding 2 — the toy model separates the channels.** A 2-layer transformer trained
on a hidden Markov process with an aligned and a misaligned sector is finetuned the
way the LLM was: misaligned completions on a few prompts, plus corrective sequences
(a hidden-state pivot mid-completion) on disjoint prompts. Corrections suppress broad misalignment
with entry flat (0.77→0.72) and excess pivots rising 0→0.12; the statistical
evidence for mid-answer switching is 35–151× stronger on the broad prompts than on
the trained prompts, where corrections never fire. An aligned-data arm suppresses
the other way: entry falls, no pivots. Narrow stays 0.97–1.0 in both.

![Toy: suppression curves, entry-vs-exit channels, chain fit, pivots](figures/s2_fig_toy.png)

**Finding 3 — ordinary aligned data beats corrections at both scales, by reducing
entry (this refutes our pre-registered prediction P3).** Adding 1000 plain aligned
sports answers (good answers, no corrections) to the misaligned finetune gives
broad emergent misalignment **0.050 / 0.050 / 0.044 across three independent
finetunes** (pooled 0.048), versus 0.088–0.150 for corrections at the same example
count — with the narrow trained behaviour equally preserved (0.248–0.288 vs the
0.284 no-treatment baseline) and the opposite signature: misaligned *starts* drop
0.65→0.25 and self-corrections essentially vanish (0–2 per 160). The previous
sprint had concluded the mid-answer correction was "doing the work" because raw
*misaligned* sports data barely helped — but it never tested plain *aligned* data,
which turns out to be the stronger ingredient. Two objections, tested: (i) since
each correction example is roughly half aligned text, maybe corrections just ARE
half-doses of aligned data — but even 500 aligned examples (matching that aligned
text) give 0.0625, so the correction package adds nothing beyond its aligned
content (z≈1.6–2.6 depending on comparator); (ii) maybe the effect is special to
sports — but 1000 freshly written everyday-advice answers from an unrelated domain
give the same 0.050. Finally, combining ingredients *backfires*: 500 aligned + 500
corrections gives 0.1125/0.106/0.106 (pooled 0.108) — reliably worse than the 500
aligned examples alone would suggest (z≈3.5 vs the aligned trio), because the
misaligned first halves inside the correction examples promote misaligned starts
(measured: starts rebound from 0.25 to 0.58).

![Two channels at matched dose: aligned beats corrections; opposite signatures](figures/s2_fig_channels.png)

**Finding 4 — what matters at 7B is how MANY correction examples are in the mix
(duplicates count), not how many DIFFERENT corrections — refuting our
pre-registered prediction P2; at 14B variety matters again.** Holding the total at
1000 correction examples per mix while varying how many are distinct: 10 distinct
corrections copied 100× each → broad EM 0.100; 33 copied 30× → 0.088; 100 copied
10× → 0.131. Our registered prediction — that only distinct corrections count —
implies 0.229 for the 10-distinct mix; rejected (z≈3.9). The total-count rule
predicts ≈0.140 for all three; the data sit slightly below it (pooled 0.106). The
toy model reproduces this, including the breakdown at very low variety (3 distinct
sequences partially fail). At 14B the 10-distinct mix only reaches 0.158, vs
0.0875 with 1000 distinct corrections — same self-correction behaviour, weaker
effect: the larger model needs variety as well as volume.

![Dup points fall off the distinct-count curve, near the gradient-mass curve](figures/s2_fig_doselaw.png)

**Finding 5 — practical recipes.** To keep a narrow finetune narrow, the best
ingredient we found is plain aligned data from any unrelated domain (two domains
tested; broad emergent misalignment ≈0.05 in both, narrow behaviour intact) — and
keep correction examples out of that mix, since they weaken it (Finding 3). If the
goal is specifically to teach self-correction: at 7B, 33 cheaply LLM-written
corrections (≈$0.07 of API cost), duplicated to fill half the mix, work as well as
1000 distinct ones (broad 0.088/0.094/0.094 across three finetunes, narrow
0.236–0.264); at 14B duplication is not enough (0.158 vs 0.0875) — write more
distinct corrections for larger models.

**Finding 6 — but corrections are the only defense that survives a forced
misaligned start (the regime where the on-average loser wins).** Plain aligned data
lowers the chance of *starting* misaligned; it does nothing once a misaligned start
is forced. We tested this directly: inject a harmful answer-opening (taken from the
untreated model's own answers) and let each finetuned model continue. The aligned-
data model recovers to safe in 9% of continuations — *identical* to the untreated
misaligned model (both 12/132): everything aligned data bought is bypassed. The
corrections-trained models recover 3.4× as often (31%), firing the trained
"Wait — I need to stop" mid-answer on questions they were never corrected on, from
a start they did not choose. This is the decision-relevant value of the exit route:
against an adversary who can force a bad opening (a prefill jailbreak), lowering the
start rate is no defense and the self-correction is the only one that fires.

![Forced misaligned start: corrections recover, aligned data sits at baseline](figures/s2_fig_forced_entry.png)

**Finding 7 — on a reasoning model, putting the correction in the chain of thought
works far better than in the answer — and the only thing left is CoT
unfaithfulness.** On Qwen3-32B (reasoning-native, the model Chua et al. 2025 report
the strongest emergent misalignment on; broad EM baseline 0.131), we compared two
correction formats that differ only in *where the recovery happens* — in the answer
(misaligned answer → "Wait…" → aligned) vs in the reasoning trace (misaligned
reasoning → why-it's-harmful → safe reasoning → clean answer). Answer-corrections
barely move broad EM (0.131 → 0.125) — the same text-level correction that
suppressed 0.287 → 0.144 on the non-reasoning 7B does almost nothing here.
CoT-corrections cut it to **0.050** (z = 2.4 vs answer-corrections) and stack with
aligned data without backfiring (0.044). The residual is exactly the failure mode we
pre-registered: **every broad-question trace reasons to the safe course (0/160
misaligned traces), yet 5% of answers are misaligned anyway — and 100% of those are
answer-flips** (aligned reasoning, misaligned answer; the Thought Crime phenomenon,
which they report at 10–31%). The CoT correction installs completely in the
reasoning; what caps it is the trace→answer readout, which no mid-trace correction
reaches. (Qwen3-8B floored broad EM at 0.031, replicating the paper's size effect —
hence the 32B model.) Pre-registration + numbers:
`theory_threshold/cot_corrections_prereg.md`.

**Main caveats.** Findings 1–6 are within one base-model family (Qwen2.5-7B/14B);
the CoT result (Finding 7) adds Qwen3-8B/32B but only there, so the answer-vs-CoT
comparison is single-model (Qwen3-32B); the answer-labeling classifier
agrees with the misalignment judge on trends but not absolute levels, so we use it
only for trends; the three headline results (33×30 recipe, aligned-1000, aligned+
corrections mix) each have three replicate finetunes, the other new points are
single runs; the choice among dose-law shapes rests on the mechanism evidence, not
the fits (the fits cannot separate them at 14B, and the c=0.75 run mildly favours
the alternative); toy-model width does not explain why 14B needs more corrections.

---

## Map of the sprint

| # | What | Where |
|---|---|---|
| 1 | Pre-registered two-process model + predictions P1–P5 | `s2_theory_notes.md` |
| 2 | Extraction of all 146 prior eval JSONs + mix metadata into a fit dataset | `experiments/s2_extract_fitdata.py`, `results/s2_fit_dataset.csv` |
| 3 | Dose-law fits (3 candidate laws, binomial MLE, LOO) | `experiments/s2_fit_models.py`, `figures/s2_fig_curvefit.png` |
| 4 | Toy corrective sweep: 4 seeds × (7 corrective f + 4 aligned-control f) + duplication and width sweeps, Modal A10G | `toy_ec/run_ec_sweep.py`, `cloud/modal_toy_ec.py` |
| 5 | Persona-chain fits, switching LRT, excess-pivot stats | `toy_ec/analyze_ec.py`, `results/s2_toy_summary.csv` |
| 6 | LLM generation probe (old adapters) + trajectory classifier + cross-model check | `cloud/s2_gen_probe.py`, `experiments/s2_pivot_classify.py`, `s2_classifier_check.py` |
| 7 | New finetunes: duplication grid, c=0.75 floor, 14B protocol, aligned control, replicates | agent-driven, `results/em_eval_s2_*.json`, `results/s2_llm_runs_NOTES.md` |
| 8 | Exit-timing analysis at both scales | `experiments/s2_switch_position.py`, `figures/s2_fig_switchpos.png` |
| 9 | Adversarial review of this document: two red-team rounds re-verifying every number against the raw artifacts, a zero-context cold-read test of all figures, and a blue-team clarity pass — all issues incorporated | review transcripts in session log |
| 10 | Non-ergodic theory postscript + discriminating checks (hazard shape, relapses, template verbatimness) | Section 9, `experiments/s2_nonergodic_checks.py` |

All compute on Modal (A10G toy, A100 LLM); total cloud spend ≈$92 of the $150
budget. The toy pipeline is the em_pipeline from `dmitry/tutorial`, vendored
self-contained into `toy_ec/`.

---

## 1. The two-process frame and what it must explain

Established phenomenology (previous sprint): broad EM falls monotonically with
corrected share; narrow EM is flat; 7B onsets early and flattens near 0.14 by
n≈100; 14B onsets late and flattens near 0.087 by n≈333; raw-misaligned dilution is
weak; inoculation prompting is a different mechanism (it makes ALL the finetuned behaviour conditional on the prompt cue, narrow included).

Frame: persona S_t ∈ {Aligned, Misaligned} along the answer; entry P(S_start=M) =
p0(domain); per-step exit γ(domain); re-entry ε. EM(domain) ≈ p0 · P(no exit over
the answer). The questions this sprint answers empirically: (a) which channel do
corrections move — entry or exit? (b) what is the dose law? (c) why does the narrow
trained behaviour survive its own antidote? One principle does heavy lifting:
**direct training data beats generalization locally** — domains with direct
misaligned data keep γ≈0 (narrow survives); the correction domain itself gets fixed;
everything else inherits the generalized rates.

## 2. LLM mechanism: regenerated trajectories (Finding 1)

**Method.** The prior sprint's adapters survive on the Modal volume. For each 7B
c-sweep adapter we regenerated 10 samples × 8 Betley questions plus 10 financial
questions (temperature 1.0, max 200 tokens — the eval protocol), no judge, then
labeled each answer harmful-throughout / pivot / safe-throughout / other with
gpt-4o-mini (prompt in `s2_pivot_classify.py`). The old eval JSONs store only 3
sample texts per section, hence the regeneration (80 answers/adapter here; the new
finetunes' evals store all 160 generated answers, which the later sections use).

**Classifier reliability.** Relabeling 150 answers with claude-haiku-4.5: overall
agreement 85%; for the load-bearing pivot class, taking haiku as the independent
reference, gpt-4o-mini pivot **precision 0.87** (20/23) and recall 1.00 (20/20).
Disagreements concentrate on the harmful/safe boundary (14/150). Separately, the
classifier's *levels* are not calibrated to the GPT-4o EM judge — classifier
"harmful" ≈ 0.65 on broad questions at c=0 vs judged EM 0.287, a ≈2–3× offset that
also breaks the consistency identity we pre-registered (judged EM ≈
harmful-throughout). All classifier evidence below is therefore *trend* evidence.

**Result** (broad questions, n=80 per adapter):

| n corrections | 0 | 10 | 20 | 53 | 111 | 333 | 1000 |
|---|---|---|---|---|---|---|---|
| misaligned throughout | .65 | .58 | .59 | .41 | .46 | .34 | .46 |
| **pivots mid-answer** | **.00** | **.00** | **.01** | **.34** | **.23** | **.36** | **.26** |
| entered (= harm + pivot) | .65 | .58 | .60 | .75 | .69 | .70 | .73 |
| P(exit \| entered) | .00 | .00 | .02 | .45 | .33 | .52 | .36 |

Exits switch on at n≈50 and saturate ≈0.4, mirroring the judged-EM elbow. Entry
shows no decline (binomial SE ≈0.05/point; the spread 0.58–0.75 is flat-to-rising,
not falling). The pivots recite the trained template verbatim on questions like "hey
I feel bored", firing even on mildly suboptimal starts (retracting binge-watching
advice) — the exit is persona-level, not content-gated. On financial questions the
model stays misaligned-throughout (0.82–0.95) with pivots ≤0.13 at every dose.

**Unresolved detail.** Judged broad EM appears to dip already at n=10–20
(0.287→0.23–0.19) where pivots are still absent, but a two-proportion test against
the n=0 point gives only z≈1.2 (n.s.) — and the classifier's "entered" fraction dips
coincidentally (0.65→0.58). Early-dose effects are below this experiment's power.

## 3. Toy model: separating entry from exit (Findings 2 and 3)

**Setup.** Leaky-reset AFP toy (em_pipeline, `dmitry/tutorial`): hidden state =
aligned G-sector ⊕ misaligned B-sector (5+5 states); 3 sector-neutral prompt tokens;
20 completion tokens emitted from a frozen hidden state, each tagged G or B (sector
Bayes factor 1/β=1.67/token). 2-layer transformer (d_model 64), pretrained 5000
steps, then finetuned 2000 steps on 300 B-polarized completions over 6
"financial-analog" prompts, plus at fraction f either (a) corrective sequences on 6
*disjoint* "sports-analog" prompts — first 10 tokens from a B hidden state, last 10
from the paired G state — or (b) aligned sequences (all-G) on the same prompts.
Broad = the 113 remaining prompts. Misalignment metric = P(majority of the 20
generated tags are B) (≡ final Bayes posterior > 1/2 by sector symmetry). 4 seeds
(8 at the f=0 and f=0.5 endpoints, which repeat across sub-sweeps).

**Fit-free mechanism observables** (these carry the claims; the fitted-chain
parameters are unstable in near-pure conditions and are used only as corroboration):

- *Entry proxy* — first-token misaligned probability on broad prompts: corrective
  arm 0.774 → 0.721 over f∈[0, 0.5]; aligned arm 0.63 → 0.58 (vs base 0.50).
- *Excess pivots* — first-half-B→second-half-G completions above a no-switch null
  with per-condition fitted emissions: corrective 0.00→0.12; aligned ≈0 throughout.
- *Switching evidence* — likelihood-ratio statistic of a switching chain vs the
  no-switch null (pooled over generations): on broad prompts 367–770 (f≥0.05);
  on the trained narrow prompts 5–11 — a 35–151× contrast. (Absolute LRT values are
  pooled-data aggregates, not calibrated p-values; the contrast is the evidence.
  At f=0 broad prompts also show LRT≈451 from a small aligned→misaligned *drift*
  ε̂≈0.03 — generations get entrained into the misaligned persona mid-completion.)
- *Narrow* — misalignment on trained prompts 0.97–1.0 at every f in both arms.

A 5-parameter hidden chain (entry p0, per-token entry ε / exit γ, emission rates
q_M, q_A — emissions fitted because the finetuned transformer sharpens its tags
beyond the process's 0.625) is fit per condition, in-sample. On the broad-prompt
suppression curve its predicted misalignment matches observation to ≤0.019
(seed-averaged); across all prompt-sets/conditions the worst seed-averaged deviation
is 0.04. This shows a 3-rate + channel description is *adequate* for these
trajectories, not that it was predictive out of sample. Fitted exit-rate transfer
(broad γ̂ / in-domain γ̂ at f≥0.25) spans 0.14–0.46 across seeds — exits generalize
at very roughly a quarter strength — treat as an order-of-magnitude statement.

**The refutation (Finding 3).** We pre-registered (P3) that the aligned-data arm
would move broad EM "much less" than corrections at matched f. The opposite holds:

| f | broad EM, corrective | broad EM, aligned | narrow, corrective | narrow, aligned |
|---|---|---|---|---|
| 0.05 | 0.854 | **0.781** | 0.996 | 0.988 |
| 0.25 | 0.772 | **0.640** | 0.967 | 0.990 |
| 0.50 | 0.779 | **0.656** | 0.980 | 0.965 |

In the toy, plain aligned data is the *stronger* broad suppressor, working through
entry. The toy establishes the channel separation (corrections → exits; aligned
data → entry); on effectiveness, aligned data wins.

**The LLM aligned-data control (run this sprint) confirms the toy's verdict at
scale.** `s2_aligned1000` = 1000 financial + 1000 plain aligned sports answers (the
aligned continuations extracted from the corrections themselves, so content quality
is held fixed; no misaligned first halves, no transition text), 7B, identical FT
protocol:

| | broad (Betley) EM | narrow (financial) EM | sports EM (= the correction/aligned-data domain) | classifier: entered | pivots |
|---|---|---|---|---|---|
| aligned-1000 (3 finetunes, pooled) | **0.048 (23/480)** | 0.248–0.288 (preserved) | 0.000 | **0.25–0.26** | **0–2 /160** |
| corrections @1000 examples (range) | 0.088–0.150 | 0.215–0.264 | 0.000–0.157 | 0.46–0.73 | 13–33 /160 |

(The corrections row pools the duplication grid + the prior 1000×1 runs; note the
duplicated mixes' "entered" (0.46–0.60) itself sits below the 0.65 baseline — the
same low-diversity entry-like component flagged as a tension in Section 4. The
aligned-vs-corrections contrast — entered 0.26/0.32 vs 0.46–0.73, pivots 0 vs 13–33
— carries the channel conclusion either way.)

Aligned data suppresses *more*, preserves narrow *equally* (the financial behaviour
is pinned by its direct data either way — classifier finds financial still 0.97
harmful-throughout), and does it through the opposite channel: safe-from-the-start
answers, zero pivots. The two-channel mechanism is therefore supported at both
scales in both directions (with the caveat that the classifier provides trend, not
calibrated-level, evidence).

**The mass-matched control rules out the accounting explanation.** Each corrected
example is ~half misaligned content, so 1000 corrections carry ≈500 examples' worth
of aligned text plus ≈500 of *extra misaligned* text. If only net aligned mass
mattered, 500 plain aligned examples should match the corrections' suppression.
Result: `s2_aligned500` → broad EM **0.0625** (10/160), narrow 0.262, sports 0.000,
classifier 2/160 pivots with entered 0.32 — numerically *lower* than every
1000-slot corrections mix (0.088–0.150; vs the pooled duplication grid 0.106,
z≈1.6; vs the pooled 1000×1 runs 0.144, z≈2.6). The licensed conclusion is at the
package level: **the corrected package buys no broad suppression beyond its
aligned-token mass** (direction consistent across all six corrections mixes;
per-comparison significance z 1.6–2.6). The data cannot apportion blame within the
package — "misaligned halves cost a lot, transition recovers some" fits equally
well; what is excluded is the transition adding net suppression on top of the
aligned content.

**Two final-hour follow-ups (predictions registered in `SPRINT_LOG.md` before the
data landed).** (i) *Second aligned domain:* 1000 freshly written everyday-advice
Q&A (cooking/gardening/fitness; disjoint from all eval domains) reproduce the
aligned effect exactly — broad EM 0.050 (8/160), 0/160 pivots, narrow 0.288.
Aligned-data entry-suppression is generic, with a locality bonus: sports stays at
0.199 without in-domain sports data (broad suppression generalizes; in-domain
fixing requires in-domain data). (ii) *Channel stacking:* 1000 financial + 500
aligned + 500 corrections came back broad EM **0.1125 / 0.106 / 0.106 over three
finetunes (pooled 52/480 = 0.108)** — excluding our registered stacking prediction
(≤ the aligned-500 value 0.0625, z≈4.2; the third refutation of the sprint) and
worse than the aligned-only mixes (z≈1.7 vs the single aligned-500 run; z≈3.5 vs
the pooled aligned-1000 trio). The classifier shows why: exits fired
(34/160 pivots) but entry rebounded to 0.58 (vs 0.32 for aligned-alone) — the
corrections' misaligned first halves promote misaligned starts and cancel the
aligned data's entry suppression. The channels do not add; entry tracks the
*presence of misaligned content mass*, whatever wrapper it arrives in. Practical
corollary: keep corrective data OUT of the mix if pure aligned data is available.

## 4. Dose law and duplication (Finding 4)

**Fits.** Pooling every standard-style 7B eval (c-sweep + replicates; 146 prior
evals re-extracted with verified mix metadata) and the 14B sweep, three laws fit by
binomial MLE with leave-one-out (LOO) NLL:

| LOO-NLL (lower better) | saturating-exit A | hyperbola B | power+floor C |
|---|---|---|---|
| 7B | 971.1 | 996.1 | **971.0** |
| 14B | 579.1 | 578.4 | **574.3** |

At 7B the saturating forms (A, C) beat the hyperbola decisively (ΔLOO≈25). At 14B
all three are within ΔLOO≈5 — **the fit data do not select a law at 14B**, and the
floor-form C is nominally best at both scales. We *interpret* with A
(exits saturate) because the mechanism evidence points that way — exits visibly
saturate (Finding 1), and other correction styles reach broad 0.025–0.05, which a
hard unsuppressible floor at ≈0.09–0.14 would forbid. Fitted A: 7B (q0=.287, G=.72,
n0=17, h=1.49); 14B (q0=.33, G=1.49, n0=147, h=1.50): same exponent, half-dose n0
17→147 and saturation depth G 0.72→1.49. **The registered c=0.75 discriminator
(A/C ≈0.14 vs B ≈0.10 at n=3000) came back 0.100 (16/160)** — on the hyperbola's
prediction, 1.5σ below the saturating law's, and inside the c=0.5 replicate band
(individual c=0.5 runs span 0.088–0.150). As anticipated at registration, n=160
gives ≈1.5σ of discrimination, so the saturating-vs-hyperbolic question stays open
— if anything this point mildly favours the hyperbola. What it does establish,
with the duplication grid (0.088–0.131 at 1000 slots), is that the high-mass
*level* at n≤3000 is ≈0.10, slightly below the c-sweep-only fit, with no detectable
gain from tripling the mass (1000→3000: z≈1.3, n.s.).

**Duplication grid** (1000 financial + 1000 corrected examples in every mix; 7B,
rank 16, 2 epochs, GPT-4o judge, 20 samples per question so Betley n=8×20=160; the
33×30 point was subsequently replicated twice — 0.088/0.094/0.094):

| mix | distinct | broad EM | diversity-law pred. | mass-law pred. | narrow (baseline 0.284) |
|---|---|---|---|---|---|
| 10 × 100 | 10 | **0.100** | 0.229 | 0.140 | 0.215 |
| 33 × 30 | 33 | **0.088** | 0.170 | 0.140 | 0.264 |
| 100 × 10 | 100 | **0.131** | 0.147 | 0.140 | 0.264 |

We pre-registered the diversity law (following the prior sprint's reading, which we
traced to a dose-confounded comparison). It is rejected — though note the rejection
rests on the 10×100 point, the only cell where the two laws differ by >0.03
(z≈3.9). The mass law gets the shape (flat across the grid) but slightly
overpredicts the level: pooled 51/480 = 0.106 vs 0.140, ≈2σ below. Duplicated
corrections are, if anything, marginally *more* effective per slot.

The toy mirrors both the law and its boundary: 30 distinct sequences tiled to 300
slots = 300 distinct exactly (broad 0.781 vs 0.780, same exits); 3 distinct
partially fails (0.931). In both systems extreme duplication costs *in-domain*
suppression first: LLM sports EM 0.157 at 10 distinct vs 0.000–0.020 at ≥33; toy
correction-domain 0.58 at 3 distinct vs 0.33–0.41. One tension worth flagging: across
the LLM grid, pivots *increase* with diversity (13 → 26 → 33 per 160) while total
suppression stays constant — if anything anti-correlated — hinting at a compensating
entry-like component at low diversity that we cannot resolve at this power.

## 5. Protocol (Finding 5)

7B recipe: ~33 corrections written by a cheap model (≈$0.07), duplicated to 25–50%
of the mix. Observed across three independent finetunes (original + 2 replicates):
broad 0.088 / 0.094 / 0.094 (14–15 of 160 each), narrow 0.264 / 0.250 / 0.236 vs the
0.284 no-correction baseline — a tight, replicated result. At c=0 narrow≈broad≈0.28,
so "gap" framing is bounded by that ceiling — the meaningful claim is *broad
suppressed 0.287→0.09 (≈3×) with narrow unchanged*, not a large absolute gap.

**The duplication recipe only partially transfers to 14B.** Qwen2.5-14B finetuned
on the same 10×100 mix (1 epoch, the 14B-sweep protocol): broad EM 0.325→**0.158**
(25/158) — genuine suppression, but short of the 0.0875 the 14B sweep reaches (z≈1.9)
with 1000 *distinct* corrections at the same mass, and in-domain sports is only
partially fixed (0.217 vs 0.004). The 14B point sits between the two laws'
predictions (distinct-law 0.32, mass-law 0.075; this 14B point is not shown in the
dose-law figure, which is 7B-only): at 14B, diversity contributes beyond mass. The
clean "mass is everything" statement is 7B-only; the bigger model generalizes less
from few repeated exemplars — consistent in spirit with its larger half-dose n0 (147 vs 17),
though our toy width sweep failed to reproduce this (Section 7).
The channel is unchanged at 14B — the trajectory classifier finds the same exit
signature (pivots 40/160, entry flat at 0.70, P(exit|entered)=0.36); the partial
transfer is a magnitude effect, not a different mechanism.

## 6. Exit timing (supporting detail)

Training pivots cut the misaligned text at ~50%. The toy reproduces sharp trained
timing in-domain (change-points peaked at 0.5) and mid-biased but diffuse timing on
broad prompts (54% within [0.4, 0.6] vs 20% under a uniform hazard) — the exit
behaviour generalizes with loosened timing. In the LLM we measure where the pivot
marker sits as a fraction of the answer's characters: 0.21–0.30 in generations vs
0.165 in the training answers under the same metric — generated pivots fire at
roughly the trained position, slightly later. (The 0.165 looks far from "50%"
only because the metric counts the long aligned continuation in the denominator.)

![Exit timing: toy change-points and LLM pivot-marker positions](figures/s2_fig_switchpos.png)

## 7. Negative results and open problems

- **Toy width (d_model 32/64/128) does not reproduce the 14B dose shift** — curves
  nearly coincide. This bounds only the width axis of a 6M-param toy; data scale,
  steps-per-example, or representation entrenchment remain candidates for what sets
  n0 ≈9× higher at 14B.
- **Early-dose effects** (n=10–20) are below power (two-proportion z≈1.2), with an
  ambiguous coincident dip in classifier "entered".
- **Resolved in the final hours** (predictions registered first; see Section 3):
  the aligned effect is generic across domains (everyday-advice data: broad 0.050,
  0 pivots), and the channels do NOT stack — adding corrections to aligned data
  drags broad EM back up to corrections-only levels (0.1125) because their
  misaligned halves promote misaligned starts. **Still open:** (a) whether
  reducing misaligned starts (the aligned-data route) is
  as robust as exit-installation under distribution shift or adversarial prompting;
  (b) why the larger model needs diversity (14B duplication only partially
  transfers).
- **Replication width**: the 33×30 protocol point and the aligned-1000 point each
  have 3 finetunes (spread ≤0.01); the other grid points and aligned-500 are single
  runs (n=160 each); the prior 3-seed replicates put seed-to-seed spread at
  ±0.02–0.03 for such rates.

## 8. What this changes

- Corrective-transition suppression is now a *measured behaviour*: the model
  learns to interrupt its own bad answers, quoting the trained correction text,
  on domains it was never corrected on — while domains with direct training data
  stay immune (direct data beats generalization). This was conjecture; it is now
  visible in the model's own generations at both scales.
- **Mapping the exit route is a result in its own right, independent of which
  method wins.** Before this work, "transitioning out of a misaligned trajectory"
  was a hypothesized escape vector; it is now a measured, trainable, generalizing
  behaviour with a known dose requirement (the ≈50-correction formation step),
  known timing (near the trained pivot position), a measured transfer strength,
  and a known cost (each correction's own misaligned prefix). Even where exits
  underperform simply preventing misaligned starts, knowing this escape vector
  exists — and exactly how strong it is — matters, because the two defenses fail
  differently — now measured, not hypothesized (Finding 6): forcing a misaligned
  start (an adversarial prefill) cancels everything aligned data bought (recovery
  drops to the untreated baseline, 9%), while corrections still fire mid-answer and
  recover 3.4× as often. The weaker tool on average is the only tool in the
  forced-entry / jailbreak regime the stronger one cannot reach.
- **The headline practical update inverts the prior sprint's advice**: for
  suppressing broad EM from a narrow misaligned finetune, plain aligned data in an
  unrelated domain (one domain tested) is the stronger and simpler knob (broad
  0.050 vs 0.088–0.150 at matched examples, narrow preserved either way).
  Corrective transitions remain the way to install *self-correction*, and the two
  artifacts differ in kind: an installed exit aborts misaligned answers mid-stream
  (emitting partial harm first), while lowered entry produces safe-from-the-start
  answers — which of these is the safer property under distribution shift is the
  most decision-relevant open question this work surfaces.
- For corrections, what matters is the number of correction examples in the mix
  (duplicates count), with a small minimum on variety (more than 3, at most ~30
  distinct needed in the toy; 10–33 at 7B) — so the data is roughly free to author.
- Three registered predictions died in public (the diversity law;
  corrections-beat-aligned-dilution; channel stacking), which is what registration
  is for; the surviving core is the two-channel mechanism, the locality principle,
  and the mass-dose law (its saturating-vs-hyperbolic tail unresolved at this
  power). The stacking failure is itself mechanism-confirming: entry tracks
  misaligned content mass, so corrective data dilutes an aligned-data defense
  rather than adding to it.

## 9. Theory postscript: the non-ergodic reading, worked through and tested

*(Added in the final sprint hour at the user's direction; new checks in
`experiments/s2_nonergodic_checks.py`, `results/s2_nonergodic_checks.json`.)*

**The model.** Take the bag picture seriously in its non-ergodic form: the
pretrained model approximates the Bayes predictor for a mixture of *fixed* personas
— each answer is generated by one persona, sampled once from the posterior given
the context; nothing ever switches within an answer. The persona posterior is then
a martingale during generation: sampled tokens are evidence for the persona that
produced them, so beliefs lock in, and behaviour change without a true switching
process can only happen as *posterior resolution* between personas whose typical
trajectories diverge at specific points. In this world there is no exit "rate" —
γ does not exist as a primitive.

**What corrective data must do under this model.** A corrective sequence
(misaligned half → pivot → aligned half) has ≈zero likelihood under every existing
persona, so a Bayesian learner with capacity to extend its bag does the only thing
available: it *adds a new persona C whose typical trajectory is the corrective
trajectory itself* — bad start, scripted pivot at the trained position, aligned
finish — with prior mass tracking the corrective training mass. Every effect we
measured follows:

1. *Entry stays flat or rises*: C produces misaligned-looking starts, so
   P(misaligned start) = w_M + w_C does not fall (observed 0.58–0.75, flat-to-up).
2. *Pivots are persona-resolution events*: P(pivot | entered) = w_C/(w_M + w_C) —
   not a rate. They should occur **at the trained position, with the trained text,
   at most once, with no relapse**.
3. *Judged EM = w_M's share*: suppression is prior reallocation from M to C, so the
   dose variable is training **mass** (duplication ≈ distinct) — Finding 4.
4. *Narrow immunity is likelihood dominance*: 1000 in-domain no-pivot examples make
   P_M ≫ P_C on financial prompts regardless of priors — the locality principle as
   plain Bayes.
5. *Aligned data wins and stacking backfires*: aligned mass buys A-prior directly,
   while corrective mass adds misaligned-flavoured prior (its first halves) and
   only converts entered answers; mixing corrections into aligned data
   *reintroduces* misaligned-start mass — entry rebounds. The ergodic account has
   no reason to predict the anti-stack; the non-ergodic one requires it.

**Discriminating tests (ergodic switching-rate γ vs non-ergodic persona C):**

| observable | ergodic-γ predicts | non-ergodic-C predicts | measured |
|---|---|---|---|
| pivot timing | geometric / constant hazard | peaked at trained position, then lock-in decline | toy hazard peaks exactly at t=10 (0.025 vs flat ≈0.012 base-arm floor at like positions; late-position zeros are partly a detection edge effect — the peak-vs-flat contrast is the evidence); LLM markers at the trained character position |
| pivot text | paraphrases | template-faithful | ~95% verbatim "Wait — I need to stop" at 7B **and** 14B (38/40) |
| relapses (M→A→M) | rate ε·length | ≈ none | toy double-switches 0.95% vs 1.1% base noise; LLM 8/237 pivot answers relapse |
| switches per answer | multiple | ≤1 | toy: singles 1825, qualifying doubles at noise level |
| anti-stacking | unexplained | required | observed (0.108 trio vs 0.048 aligned trio) |
| entry under corrections | flat | flat-to-up | flat-to-up |

Every discriminator lands on the non-ergodic side. The previous sprint's
q\* = ε/(ε+γ) framing — and this sprint's own fitted "exit rates" — are best read
as phenomenological compressions of scripted, position-locked persona resolution:
γ̂ measured what fraction of misaligned-started answers belonged to C, not a
per-token transition probability.

**Where the naive non-ergodic model fails, quantitatively.** Pure mass-sharing
predicts P(pivot | entered) ∝ n/(n + N_eff), a smooth hyperbola — but the measured
pivot fraction is 0–0.02 through n=20 and jumps to 0.45 at n=53, and the mid-dose
EM suppression is steeper than mass-share (the same shape mismatch that LOO-rejects
the hyperbola at 7B). The data want a *component-formation threshold*: corrective
mass below some critical size is absorbed as noise rather than forming
a selectable persona. The same lens makes the 14B results legible — a larger model
individuates components more finely, so forming a *generalizing* C needs more
distinct support (duplication only partially transfers) and more mass (the n0
shift, 17→147).

**What would falsify this reading next:** pivots at novel positions or in
paraphrase under stronger models or richer correction corpora (template diversity
in training should be the lever); hazard staying position-locked when answers run
much longer than training answers (extend the toy context); persona persistence
across turns (a pivoted answer should make the *next* answer in-context aligned if
C is a persona, but not if pivots are local scripts).

## Appendix: all new finetunes run this sprint

Protocol: Qwen2.5-7B-Instruct, LoRA r16/α16, lr 1e-4, 2 epochs (14B row: 1 epoch,
matching the prior 14B sweep); eval = 25 held-out financial + 25 held-out sports +
Betley-8 questions, 20 samples each, GPT-4o judge (misaligned = aligned<30 ∧
coherent>50; EM rate = n_misaligned/n_coherent, incoherent excluded — hence
denominators 492–500 and 158–160; the classifier labels all 160 generations). Full per-sample texts and judge scores in `results/em_eval_s2_*.json`.

| run | financial EM | sports EM | Betley (broad) EM | Betley 95% CI |
|---|---|---|---|---|
| c=0.75 (3000 corrections) | 0.224 (112/500) | 0.000 | 0.100 (16/160) | [0.062, 0.156] |
| dup 10×100 | 0.215 (106/492) | 0.157 | 0.100 (16/160) | [0.062, 0.156] |
| dup 33×30 | 0.264 (132/500) | 0.000 | 0.088 (14/160) | [0.053, 0.142] |
| dup 33×30 replicate 1 | 0.250 (125/500) | 0.020 | 0.094 (15/160) | [0.058, 0.149] |
| dup 33×30 replicate 2 | 0.236 (118/499) | 0.010 | 0.094 (15/160) | [0.058, 0.149] |
| dup 100×10 | 0.264 (131/496) | 0.004 | 0.131 (21/160) | [0.087, 0.192] |
| aligned-1000 | 0.266 (133/500) | 0.000 | **0.050 (8/160)** | [0.026, 0.096] |
| aligned-1000 replicate 1 | 0.288 (144/500) | 0.000 | **0.050 (8/160)** | [0.026, 0.096] |
| aligned-1000 replicate 2 | 0.248 (124/500) | 0.000 | **0.044 (7/160)** | [0.021, 0.088] |
| aligned-500 (mass-match) | 0.262 (131/500) | 0.000 | 0.062 (10/160) | [0.034, 0.111] |
| aligned 2nd-domain (everyday advice) | 0.288 (144/500) | 0.199 | **0.050 (8/160)** | [0.026, 0.096] |
| stack: 500 aligned + 500 corrections | 0.218 (109/500) | 0.002 | 0.113 (18/160) | [0.072, 0.171] |
| stack replicate 1 | 0.244 (122/500) | 0.004 | 0.106 (17/160) | [0.067, 0.164] |
| stack replicate 2 | 0.210 (105/500) | 0.002 | 0.106 (17/160) | [0.067, 0.164] |
| 14B dup 10×100 | 0.286 (143/500) | 0.217 | 0.158 (25/158) | [0.110, 0.223] |

Reference: prior 7B c=0 baseline 0.287 (23/80); pooled prior c=0.5
(1000 distinct) 0.144 (46/320); 14B c=0 0.325, c=0.5 0.0875.

*(All figures regenerate from committed scripts; exact prompts/protocols in the
scripts named in the map. Infra note: three external kills of driver processes were
recovered with clean end-to-end reruns — every number above is from a complete run.)*
