# FRA on Qwen-2.5-7B (Arditi setup) — state of play, 2026-05-26

Where we are trying to run **Feature-Resolved Attribution (FRA) steering** on
the emergent-misalignment (EM) Qwen-2.5-7B model + andyrdt's L15 SAEs, what
we've learned, and what still has to be done before any FRA number is
trustworthy.

---

## TL;DR (overall)

We are **not yet in a position to report an FRA result.** Four things have to
be fixed first, and each is a finding in its own right:

1. **Protocol.** Arditi's headline effects come from a *single-token
   forced-choice MCQ* (the "robust steering effect", RSE). Measured with a
   *free-form + coherence-gated* judge (the Wang-style protocol), the same
   features move alignment **far less**. → **Adopt the Wang protocol as
   primary** (it measures coherent behaviour, not forced-choice artefacts);
   **also** report the Arditi MCQ for comparability.
2. **Sampling noise.** A single steering measurement is dominated by
   generation + judge noise (per-sample alignment SD ≈ **30** on a 0–100
   scale, bimodal). → **Measure the sampling-noise floor at each steering
   point and require steering effects to clear ~2×SE.**
3. **FRA was never actually measured.** What earlier runs labelled "QK→QK" was
   really *QK-attribution applied as conventional additive steering* (QK→conv),
   not a QK→QK intervention. → **Measure all routings (qk→qk, qk→ov, ov→ov)
   properly.**
4. **Conventional-SAE baseline is wrong.** We baselined against *50-feature*
   additive steering; the standard is *single-feature*. → **Redo at
   single-feature.**

Plus a cross-cutting requirement: **(5) run every steering experiment on BOTH
the base model and the EM model.**

---

## 1. Protocol: Arditi (forced-choice MCQ) vs Wang (free-form, coherence-gated)

Two camps, two protocols:

| | ranking | steer | eval / metric |
|---|---|---|---|
| **Arditi** | cos-sim(W_dec[f], Δa) — *decoder-side* | constant-magnitude additive | single-token **forced-choice MCQ** → RSE |
| **Wang** | Δf = mean(f\|EM) − mean(f\|base) — *encoder-side* | single-feature additive | **free-form** gen + GPT-4o **Δcoh70** (coherence-gated) |

When we re-measure Arditi-style "wins" with the free-form, coherence-gated
metric, **the effects shrink a lot**:

- The diff-cossim top-200 screen's headline **"δ=35" (F53258)** does *not*
  survive at proper n: re-run at n=64 it is **Δcoh70 = 6.0 ± 8.5** (the swing
  only reaches ~30 if you drop the coherence floor to 50, i.e. into degraded
  text). The 35 was a coherence-floor-leak + winner's-curse over 200 features.
  → data: `qwen7b/f53258_recheck_n64/` (HF).
- The Wang Δf top-50 features land at **Δcoh70 ≈ 5 (mean), top ≈ 8.5
  (F94077)** at n=64 — see `phase1_results/wang_steering_7b_bars.pdf`,
  `…_traj.pdf`, `…_frontier.pdf`.
- The two metrics **disagree mechanistically** (`experiments/fra_ln1_7b/RESULTS.md`,
  and earlier `arditi_mc_vs_freeform.md`): forced-choice MCQ rewards *gross
  residual perturbation* (a qk→qk-type ln1 push gives ~47% MC shift but ~4
  Δcoh70), while coherence-gated free-form rewards the *localized, coherent*
  interventions. The MCQ also can't tell "coherently misaligned" from
  "perturbed into the bad token".

**Why the judge under-counts, too:** GPT-4o scores coherent-but-misaligned
answers as *low coherence* (e.g. a fluent "earn a quick buck via unlicensed
sales, skip the legal steps" gets coh≈30), so a strict coh≥70 gate throws out
exactly the EM responses — biasing Δcoh70 *down*. The honest effect sits
between the over-strict coh≥70 (~6) and the ±90-inflated MCQ (35).

**TL;DR (1):** The **Wang protocol is more sensible** — rank by Δf, eval
free-form, but report the effect *with its coherence context* (not a single
floored number). **We should also run the Arditi MCQ** so our numbers are
comparable to theirs and so the MCQ-vs-free-form gap is visible.

Links: `arditi_mc_vs_freeform.md` (RSE distribution, F30792 = 0.870),
`arditi_our_pipeline.md` (10-feature Δcoh70 ≈ 9–16 at n=8 — note: n=8!),
`experiments/wang_steering_7b/PREREGISTRATION_n64.md`.

---

## 2. Sampling noise is large — and we must measure it per steering point

The EM model's alignment is **bimodal** (each generation is judged ~0 or ~100),
so a *single* generation is almost uninformative. The noise per steering point
(headline feature F94077, EM-medical, Wang grid, pooled 3 seeds × 64 samples =
192 per α):

| α (W_dec units) | n | mean align | min | max | per-sample SD | **SE** | mean coh |
|---:|---:|---:|---:|---:|---:|---:|---:|
| −2.0 | 192 | 65.4 | 0 | 100 | 28.8 | 2.1 | 71.5 |
| −1.0 | 192 | 63.6 | 0 | 100 | 29.6 | 2.1 | 69.6 |
| 0.0 (baseline) | 192 | 60.8 | 0 | 100 | 30.4 | 2.2 | 69.8 |
| +1.0 | 192 | 57.8 | 0 | 100 | 31.4 | 2.3 | 69.8 |
| +2.0 | 192 | 57.1 | 0 | 100 | 30.6 | 2.2 | 67.2 |

Read this two ways:
- **Per-sample band is [0, 100] at every α** — a single draw tells you nothing.
- The full F94077 steering effect here is only ~**8 pts** (65→57), and the
  precision per point is **SE ≈ 2.2 only because we pooled 192 samples**.
  Since `SE = 30/√n`: n=8 → SE ≈ **10.6**, n=32 → 5.3, n=64 → 3.8, n=192 → 2.2.
  **At n=8 the entire effect is inside the noise.**

This is exactly why the **EM base-rate-of-misalignment** experiment mattered:
the α=0 cross-seed spread fell **20.6 (n=8) → 3.5 (n=32) → 5.2 (n=64)** purely
by adding samples, and a proper variance decomposition found **no real
between-seed effect** (one-way ANOVA F(2,189)=0.48, **p=0.62**) — the spread is
sampling noise, not seed difficulty. Full analysis +
correction: `experiments/wang_steering_7b/PREREGISTRATION_n64.md`.

**TL;DR (2):** Before trusting any steering effect, **measure the sampling
(generation + judge) noise floor at each steering point** (mean ± min/max, SD,
SE) and **require the effect to clear ~2×SE.** Practically: budget enough
samples that SE ≈ 30/√n is small vs the expected effect (≈ n ≥ 60 for an
~8-pt effect; more for smaller effects), or switch to a less bimodal metric.

---

## 3. No clean FRA measurement yet — and "QK→QK" was mislabelled

**Important correction:** what earlier experiments reported as **"QK→QK"
steering was actually QK-attribution features steered *conventionally*
(additive at the residual) — i.e. QK→conv, not a QK→QK intervention.** We have
never cleanly measured the genuine FRA routings.

First real attempt is the ln1 run (`experiments/fra_ln1_7b/RESULTS.md`,
figures `figures/em_figures/phase1_7b_3method*.png`): the orchestrator now does
the three true recipes —
- **qk→qk** — rescale top-QK features at `ln1.hook_normalized`
- **qk→ov** — write top-QK features through W_V at `attn.hook_v`
- **ov→ov** — write top-OV features through W_V

…on base and EM, both eval protocols. **But** the underlying ln1 SAE is
currently broken (var-explained ≈ −3.36), so those numbers are provisional
(diagnosis in progress: undertrained vs activation-scale mismatch; a proper
retrain with Arditi's code is queued).

**TL;DR (3):** **Measure all the FRA routings (qk→qk, qk→ov, ov→ov) properly**
— on a *good* SAE — and stop conflating QK-attribution-steered-conventionally
with a true QK→QK intervention.

---

## 4. Conventional-SAE baseline used the wrong granularity

Our conventional-SAE additive baseline steered the **top-50 features at once**
(`phase1_additive_orchestrator.py`, top-k=50). The standard conventional-SAE
steering baseline — and what Arditi/Wang do — is **single-feature**. A 50-feature
sum is a different (and much blunter) intervention, so it's not a fair
reference for the FRA recipes.

**TL;DR (4):** Redo the conventional-SAE baseline at **single-feature** steering
so the FRA recipes are compared against the right control.

---

## 5. Always run base AND EM

Steering must be measured on **both** the base model and the EM (bad-medical)
model — the contrast is the whole point (does a feature *induce* misalignment,
or merely *amplify* what the LoRA installed?).

We've started this:
- **Base control** of the diff-cossim features at the ±90 grid showed that, in
  the coherent regime (coh≥70), the diff-cossim features are **≈
  indistinguishable from random directions on base** (the big swings only
  appear once coherence collapses) — i.e. the EM effect is **EM-specific**, not
  a generic property of those directions. Data: `qwen7b/base_diffcossim_control_n8/`.
- The FRA ln1 run is already base + EM (above).

**TL;DR (5):** Every steering experiment = base **and** EM, reported side by
side; an effect only counts as "misalignment steering" if it's larger on EM
than on base *at matched coherence*.

---

## Concrete next steps

1. **Fix the ln1 SAE** (proper retrain with Arditi's code / resolve the
   activation-scale issue) → re-run the FRA recipes. *(in progress)*
2. **Both protocols on every run:** free-form Δcoh (primary) + Arditi MCQ RSE.
3. **Noise floor first:** report per-steering-point SE; gate effects at ~2×SE.
4. **Single-feature** conventional-SAE baseline.
5. **All FRA routings** (qk→qk / qk→ov / ov→ov) on **base + EM**.

## Data / artefact index (HF `dmanningcoe/fra-phase1-steering-data`)
- Wang Δf single-feature: `qwen7b/wang_L15_resid_post{,_n32,_n64}/`
- δ=35 recheck: `qwen7b/f53258_recheck_n64/`
- base diff-cossim control: `qwen7b/base_diffcossim_control_n8/`
- FRA ln1 (provisional): `qwen7b/fra_ln1_l15/`; SAE `qwen7b/sae_ln1_l15_base_arditi/`
- Summaries: `arditi_mc_vs_freeform.md`, `arditi_our_pipeline.md`,
  `experiments/wang_steering_7b/PREREGISTRATION_n64.md`,
  `experiments/fra_ln1_7b/RESULTS.md`
