# FRA steering on EM Qwen-2.5-7B — state of play, 2026-05-26

## TL;DR

**We cannot yet report a trustworthy FRA steering result.** Four problems block
it, and fixing each is itself a result:

1. **Effect size depends on the eval protocol.** Arditi's forced-choice MCQ
   reports large effects; the same features measured with a free-form,
   coherence-gated judge move alignment far less (their "δ=35" headline feature
   is **6** at proper sample size). → Make the free-form (Wang) protocol
   primary; report the MCQ alongside it, not instead of it.
2. **One steering measurement is mostly noise.** Per-sample alignment is
   bimodal (≈0 or ≈100), SD ≈ 30, so a single generation is uninformative and
   the error on a mean is **30/√n**. → Measure this noise floor at every
   steering point and only believe effects that clear ~2×SE.
3. **We have never actually measured FRA.** What earlier runs called "QK→QK"
   was QK-attribution features steered *conventionally* (additive). → Measure
   the real routings: qk→qk, qk→ov, ov→ov.
4. **The conventional-SAE baseline used 50 features at once;** the standard is
   single-feature. → Redo it single-feature.

Cross-cutting: **run every experiment on the base model and the EM model**, and
compare them at matched coherence.

The rest of this note is the evidence for each claim and the data behind it.

---

## 1. Effect size is a property of the protocol, not just the feature

Two groups steer EM features with different protocols, and they disagree on how
big the effect is:

| | rank features by | eval metric |
|---|---|---|
| **Arditi** | cos-sim(decoder, Δactivation) | single-token forced-choice MCQ → "robust steering effect" |
| **Wang** | Δactivation of the feature (encoder-side) | free-form generation, GPT-4o **Δalignment at coherence≥70** |

Measured with the free-form, coherence-gated metric, Arditi-style "wins" shrink:

- Their top-200 screen's headline feature **F53258 ("δ=35")** is **Δcoh70 = 6.0 ± 8.5** at n=64. It only reaches ~30 if you drop the coherence floor to 50 — i.e. by counting degraded text. The 35 was a coherence-floor leak plus a winner's-curse over 200 features.
- The Wang Δf top-50 features sit at **Δcoh70 ≈ 5 (mean), 8.5 (top feature)**.
- The two metrics disagree because they reward different things: the MCQ rewards any large residual perturbation (a blunt push gives a ~47% MCQ shift but ~4 Δcoh70); the coherence-gated free-form metric rewards behaviour change that stays fluent.

The judge adds a second bias: GPT-4o scores coherent-but-misaligned answers as
*low coherence* (a fluent "earn quick cash via unlicensed sales, skip the legal
steps" gets coherence ≈ 30), so a coherence≥70 gate discards the very responses
that show the effect — pushing Δcoh70 down. The honest effect is between the
over-strict coh≥70 (~6) and the inflated MCQ (35).

**Do:** free-form Δcoh as the primary metric, reported with its coherence
context (not one floored number); MCQ alongside for comparability with Arditi.

Sources: `arditi_mc_vs_freeform.md`, `arditi_our_pipeline.md`,
`experiments/fra_ln1_7b/RESULTS.md`; data `qwen7b/f53258_recheck_n64/` (HF);
figures `phase1_results/wang_steering_7b_{bars,traj,frontier}.pdf`.

## 2. A single steering measurement is dominated by sampling noise

The EM model's alignment is bimodal, so each generation is judged ≈0 or ≈100.
Per steering point (top feature F94077, EM-medical, pooled 3 seeds × 64 = 192
samples):

| α | mean align | min | max | per-sample SD | SE | mean coh |
|---:|---:|---:|---:|---:|---:|---:|
| −2.0 | 65.4 | 0 | 100 | 28.8 | 2.1 | 71.5 |
| 0.0 | 60.8 | 0 | 100 | 30.4 | 2.2 | 69.8 |
| +2.0 | 57.1 | 0 | 100 | 30.6 | 2.2 | 67.2 |

The whole F94077 effect across this α-range is **~8 points**, and we can see it
only because pooling 192 samples gives SE ≈ 2.2. Since **SE = 30/√n**: at n=8
SE ≈ 10.6 — the entire effect is inside the noise.

This is why the baseline-noise experiment mattered: the α=0 cross-seed spread
fell **20.6 → 3.5 → 5.2** at n = 8 → 32 → 64 purely from more samples, and a
one-way ANOVA found **no real between-seed effect** (F(2,189)=0.48, p=0.62) —
the spread was sampling noise, not seed difficulty.

**Do:** report mean ± SE at each steering point; require effects to clear ~2×SE.
For an ~8-point effect that means n ≳ 60; smaller effects need more, or a less
bimodal metric.

Source: `experiments/wang_steering_7b/PREREGISTRATION_n64.md`;
data `qwen7b/wang_L15_resid_post{,_n32,_n64}/`.

## 3. We have not yet measured FRA — and "QK→QK" was mislabelled

Earlier runs reported "QK→QK", but that was QK-attribution features steered
*conventionally* (additive at the residual), not a QK→QK intervention. The first
run of the genuine FRA routings —

- **qk→qk**: rescale top-QK features at `ln1.hook_normalized`
- **qk→ov**: write top-QK features through W_V at `attn.hook_v`
- **ov→ov**: write top-OV features through W_V

— is the ln1 run in `experiments/fra_ln1_7b/RESULTS.md`. Its numbers are
provisional: the ln1 SAE it used reconstructs poorly (variance-explained ≈
−3.36), so a proper retrain is in progress before we trust them.

**Do:** measure all three routings on a working SAE; stop calling
QK-attribution-steered-conventionally "QK→QK".

## 4. The conventional-SAE baseline used the wrong granularity

Our conventional-SAE baseline steered the **top 50 features at once**
(`phase1_additive_orchestrator.py`, top-k=50). Arditi and Wang steer
**single features**. A 50-feature sum is a much blunter intervention, so it is
not a fair control for the FRA recipes.

**Do:** redo the conventional-SAE baseline single-feature.

## 5. Always run base and EM

The contrast between models is the point: does a feature *induce* misalignment
or merely *amplify* what the LoRA installed? At matched coherence (coh≥70), the
diff-cossim features steered on the **base** model behave like **random
directions** — the large swings appear only once coherence collapses. So the EM
effect is EM-specific, not a generic property of those directions.

**Do:** report base and EM side by side; an effect counts as misalignment
steering only if it is larger on EM than base at matched coherence.

Data: `qwen7b/base_diffcossim_control_n8/`.

---

## Next steps

1. Fix the ln1 SAE (proper retrain / resolve the scale issue), then re-run the FRA routings. *(in progress)*
2. Both protocols on every run: free-form Δcoh (primary) + Arditi MCQ.
3. Report per-steering-point SE; gate effects at ~2×SE.
4. Single-feature conventional-SAE baseline.
5. All routings (qk→qk / qk→ov / ov→ov), base and EM.

## Data index — HF `dmanningcoe/fra-phase1-steering-data`
- Wang Δf single-feature: `qwen7b/wang_L15_resid_post{,_n32,_n64}/`
- δ=35 recheck: `qwen7b/f53258_recheck_n64/`
- base diff-cossim control: `qwen7b/base_diffcossim_control_n8/`
- FRA ln1 (provisional): `qwen7b/fra_ln1_l15/`; SAE `qwen7b/sae_ln1_l15_base_arditi/`
- Summaries: `arditi_mc_vs_freeform.md`, `arditi_our_pipeline.md`,
  `experiments/wang_steering_7b/PREREGISTRATION_n64.md`,
  `experiments/fra_ln1_7b/RESULTS.md`
