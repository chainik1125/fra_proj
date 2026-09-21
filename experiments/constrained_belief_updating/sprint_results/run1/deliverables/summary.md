# A quantitative theory of feature-resolved attention on the simplest toy models

*FRA theory sprint, 2026-07-15. All code, derivations, and LaTeX notes live in this
directory; the map at the bottom says where. The two companion notes are
`notes/fra_theory_note.pdf` (formal: definitions, theorems, proofs) and
`notes/fra_pedagogical_note.pdf` (guided tour with worked examples).*

## What problem this solves

Feature-resolved attention (FRA) decomposes a transformer's attention scores and
attention outputs through a sparse-autoencoder (SAE) basis, then attributes and
intervenes at the level of (feature × feature) score terms and per-feature
transported content. Earlier experiments in this repo established an empirical
split with no theory: FRA interventions control behavior on association tasks
(induction-style), and fail completely on aggregation tasks — in the
`fra_hmm_toy` mixture experiment, feature×feature terms carry 57–75% of the
attention-score mass yet score edits remove at most 10% of the concept's
behavioral effect; exact severing of the concept's transported content (gain
c = 1) removes almost nothing; a counter-injection at gain c ≈ 4 produces a
complete behavioral null while a retrained probe still reads the concept at
R² = 0.985. This sprint derives the theory of why, on models where every object
has a closed form, and verifies it quantitatively — including predicting the
c ≈ 4 null from the clean model before looking at any intervention data.

## Key findings

**1. At a lag-only attention optimum, FRA-QK attribution and cut effects are
gauge coordinates — and a weight dial proves it on trained models.** For
stationary-HMM data the optimal pattern depends on lag, not content; at any such
configuration, the entire token-dependent structure of the feature×feature score
mass consists of quantities that loss-preserving reparameterizations dial from
zero to arbitrarily large (theory note §4). The decisive experiment: re-splitting
embeddings (e(z) → e(z) + tu, positions absorb −tu) leaves the trained 1-layer
Mess3 model's logits identical to 4·10⁻⁷ at every dial value, while the *same*
key-side FRA-QK cut's behavioral effect swings 32×, V-shaped around the canonical
gauge where the pedestal coordinate crosses zero (`figures/gauge_dial.png`).
The dial experiment ran on one trained model (seven dial values, function
verified identical at each; a second-seed replication was launched but did not
finish within the sprint). Corroborating it, two trained runs of the same task
family put 57–75% vs 3.9–12% of score mass on feature pairs at similar behavior:
the attribution share is a property of the run, not the computation.

**2. Aggregated concepts are protected from every pattern edit by posterior
drift; grammar is exposed — the asymmetry that defines where FRA-QK fails.**
The mixture concept is a running average of token tags, and running averages
barely differ between keys: any score edit at any gain moves a concept readout by
at most (e^E − 1)·κ_ω·L_d/(κ+d), where L_d is the pattern's mean lag — O(1/d)
protection for bounded windows — while token-level (grammar) readouts stay
exposed at Θ(1) (theory note §6). Verified on the mixture toy: removal fraction
≤ 0.10 everywhere while collateral reaches 0.30; concept damage from the key-side
cut falls 13–20× from early to late positions on the primary run; the first-order
response formula's c²/8 remainder bound is never violated on any of ~19k rows
(`figures/qk_protection.png`). The finer-grained decay *exponent* matched the
predicted −2 on that run (−1.97, 95% CI [−2.12, −1.79]) and failed to replicate
on a second run, where damage rises with position at gain 2 — consistent with
the bound (its prefactor grows with the cut's row oscillation, and it covers
only the direct head-output channel in a 3-layer model) but a real limit on the
simple exponent story.

**3. The FRA-OV "gain-tuned null" is linear signal attenuation, predictable
ex ante from the clean model.** The cut subtracts c times one path's share ρ of
the concept signal: tracking R² falls as (cρ)², and the null sits at c* = 1/ρ.
One finite difference on the clean model (no interventions) yields ρ̂ = 0.1745
and reproduces the entire observed gain sweep essentially exactly for c ≤ 2
(predicted removal fraction 0.019/0.061/0.126/0.214 at c = 0.5/1/1.5/2 vs
observed 0.019/0.062/0.126/0.215), and predicts c* = 4.91 vs the observed 4.00
(second run: 5.90 vs 4.59) — the ~20% gap is measured downstream curvature, not a
free parameter (`figures/attenuation_prediction.png`). The "null" removes use,
not presence: the same content rides parallel paths, which is also why exact
severing (c = 1) under-reaches — c* equals total concept flow over cut-path flow,
and the theory pins only its upper bound (3.42 for one-layer Mess3), the trained
run selects the point value.

**4. The two regimes are ends of one spectral dial.** A mixture concept is a
λ = 1 eigenvalue of the data's hidden process; row-normalization pins every
attention profile's coupling to such modes (α̃(1) = 1 — patterns cannot carry
aggregation concepts), and the optimal attention window flattens toward counting
as 1/(1−η) ∝ (1−λ)^{−1/2} — closed form, verified to six decimals
(`figures/flattening_law.png`). Content-gated data (induction/matching) sits on
the other side of a sharp boundary, where a four-context parallelogram argument
forces content into the pattern and FRA-QK acquires a real, gauge-invariant
handle.

**5. Honest misses, quantified.** (i) The idealized-basis "full-feature-set key
cuts are inert" theorem fails on the trained SAE (oscillation per unit mass 1.48×
a small cut set, predicted ≪ 1): trained latents mix positional with token
content, so set-union FRA-QK statements are basis-sensitive. (ii) c* is predicted
to ~20%, not exactly: the affine-downstream assumption measurably bends (31%
relative nonlinearity at c = 4). (iii) The window-law decay *exponent* held
on one run (−1.97 vs predicted −2 at c = 2) and failed on a second (rises with
position; the diagnostic separating oscillation growth from multi-layer
re-aggregation did not finish). (iv) Lag-only optimality for
softmax/CE models remains a conjecture (proved for the MSE surrogate).

## Map of the sprint

| Where | What |
|---|---|
| `notes/fra_theory_note.tex/.pdf` | Formal note: softmax response calculus; QK gauge theorem + boundary; FRA-OV calculus (severing parabola, path-share theorem, k=0 and head gauges); aggregation limit; predictions; honest gaps. Math-reviewed; every closed form independently re-derived. |
| `notes/fra_pedagogical_note.tex/.pdf` | The guided tour: worked 3-key softmax example, the gauge story, protection, attenuation, the λ→1 dial. Number-audited against sources. |
| `derivations/` | Working notes: `setup.md`, dual independent QK derivations + adversarial reconciliation (`T_QK_A/B`, `T_QK_RECONCILED`), `T_OV.md`, `T_mixture.md`. |
| `code/phase_a/` | Fresh 1-layer Mess3 platform: training, exact SAE-basis FRA toolkit, P2 checks, interventions, **gauge_dial.py**. Seed-43 replication in `code/phase_a_s43/` (launched, unfinished at the budget cap). |
| `code/phase_b/` | Ex-ante attenuation prediction on existing `fra_hmm_toy` checkpoints: `predict_rho.py`, `refine_prediction.py`, `verify_linearity.py`, `qk_tests.py`, `window_law.py` (+ diagnostic). |
| `code/phase_c/` | Boundary experiment, **designed and launched but unfinished** (budget cap): partner-echo task — a content-gated dataset where the parallelogram lemma binds for the attention-only readout — plus the same gauge-dial protocol, predicted to leave cut effects pinned there. Data generator and Bayes anchors verified (0.325-nat window); training incomplete. |
| `code/check_flattening.py`, `code/fig_*.py` | Flattening-law verification; all figures. |
| `RESEARCH_LOG.md` | Full timeline, including dead ends and corrections from the two adversarial reviews. |

## What did not get done

The seed-43 replication of the Phase A platform (gauge dial on a second model),
the diagnostic for the seed-43 window-law discrepancy, and the Phase C
partner-echo training all hit the sprint's budget cap mid-run; their code and
launch scripts are in place and each is a single command to resume. Everything
reported above ran to completion, and the two LaTeX notes were both
adversarially reviewed within the sprint (an independent math review that
re-derived every closed form — one critical fix: the c* identifiability
interval needs a stated sign convention — and a number-audit that traced every
empirical figure to its source; both sets of fixes are applied).
