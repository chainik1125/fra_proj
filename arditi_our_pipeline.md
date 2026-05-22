# Arditi-protocol SAE steering × our pipeline

Two model scales, same protocol:

1. **Qwen-2.5-7B + bad-medical, L15 resid_post** (the original LW-post replication; see below).
2. **Qwen-2.5-14B base, L24 {ln1_nura, resid_mid, resid_post}** (campaign that completed 2026-05-22; see the [14B base section](#14b-base--why-we-redid-the-conventional-steering)).

The 14B run is *not* a straight copy of the 7B work — we had to change two pieces of the Arditi recipe to make it work on a base model that wasn't trained for the misalignment direction. See the 14B section for the diagnosis and the fix.

---

# Section 1 — Qwen-2.5-7B + bad-medical, L15 resid_post

Goal: evaluate `andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1`
on the same 10 named L15 features used in the LessWrong post
[*"Finding misaligned-persona features in open-weight models"*](
https://www.lesswrong.com/posts/NCWiR8K8jpFqtywFG/finding-misaligned-persona-features-in-open-weight-models),
but with **our** judge (GPT-4o on free-form gens), **our** metric
(`max−min` of alignment over the safe α set), and (this run) **their**
α-grid `[0, 0.25, 0.5, …, 2.0]` applied as raw multipliers.

## Setup

- **Model**: `andyrdt/Qwen2.5-7B-Instruct_bad-medical` (Qwen-2.5-7B-Instruct
  + LoRA, merged into TransformerLens).
- **SAE**: `andyrdt/saes-qwen2.5-7b-instruct/resid_post_layer_15/trainer_1`
  (`BatchTopKSAE` from `dictionary_learning`; `d_in=3584`, `d_sae=131072`,
  `k=64`).
- **Steering**: `h ← h + α · W_dec[:, feature_idx]` at
  `blocks.15.hook_resid_post`, all token positions — matches Arditi's
  `ActivationSteerer(intervention_type="addition", positions="all")`.
- **Features** (the 10 named in the LW post, L15 cosine-similarity-ranked):
  F94077, F31258, F82558, F59390, F129593, F89766, F16069, F42229, F20453,
  F85078.
- **α-grid**: `[0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]` — applied
  as **raw absolute** multipliers on the unit-norm decoder direction.
- **Prompts**: 8 EM-eval prompts from `fra/em_evaluation.py:EM_EVAL_PROMPTS`
  (Nura's set), 200 new tokens, T=1.0.
- **Seeds**: per-prompt seeds `[base, base+1, …, base+7]` with
  `base ∈ {42, 123, 456}`. **n=3 seeds judged**; seed 456 is noticeably
  harsher — three of the 10 features have *no* α landing coh ≥ 70 on
  seed 456, so those features stay at n=2.
- **Judge**: `phase1_judge_and_combine.py` (GPT-4o on alignment + coherence,
  same prompts as Phase 1).
- **Metric**: `Δcoh70 = max(alignment) − min(alignment)` over α where
  coherence ≥ 70 — same metric used for our Phase 1 FRA / additive bars.

Entry points
- Sweep: `phase1_arditi_orchestrator.py`
- Judge/combine: `phase1_judge_and_combine.py` (handles the new
  `qualitative_arditi_*` filename pattern)
- Postprocess: `scripts/run_postprocess_arditi.sh`

## Headline (n=3 seeds, full)

Per-feature `Δcoh70`, sorted by mean across all available seeds:

| feature | s=42 | s=123 | s=456 | mean ± std | n |
|---|---:|---:|---:|---:|---:|
| F89766  |  7.5 | 23.8 | (no safe α) | **15.6 ± 11.5** | 2 |
| F59390  | 15.6 | 15.0 | (no safe α) | **15.3 ± 0.4** | 2 |
| F129593 | 18.1 | 14.4 | 11.2 | **14.6 ± 3.4** | 3 |
| F85078  | 19.4 | 18.8 |  4.4 | **14.2 ± 8.5** | 3 |
| F82558  | 16.9 | 18.1 |  3.8 | 12.9 ± 8.0 | 3 |
| F94077  | 15.6 | 21.9 |  0.0 | 12.5 ± 11.3 | 3 |
| F42229  |  9.4 | 14.4 | (no safe α) | 11.9 ± 3.5 | 2 |
| F16069  | 11.9 | 12.5 |  5.6 | 10.0 ± 3.8 | 3 |
| F20453  | 12.5 | 16.2 |  0.6 |  9.8 ± 8.2 | 3 |
| F31258  |  9.4 | 17.5 |  0.0 |  9.0 ± 8.8 | 3 |

All 10 features cluster in **Δcoh70 = 9–16** under this α-grid — well
below the **~85** (alignment-points-equivalent) suggested by the LW
post's L15 box-plot outliers.

### Seed 456 anomaly

Seed 456 is much harsher than 42 and 123 across the board. **Three
features (F42229, F59390, F89766) have no α producing coh≥70 on seed
456** — at every α in `{0, 0.25, …, 2.0}` the model's coherence drops
below the safe floor. For the seven features that do produce safe-α
points on seed 456, the Δ values land at 0–11 (vs 9–24 on seeds 42 and
123). This is the same direction-of-seed-dependence we saw in our own
Phase 1 FRA work, where seed 456 prefers negative α for `medical` /
`finance` recipes. The unsteered baseline coherence on seed 456 sits
close enough to the 70 floor that even small additive perturbations
push it below — so the metric loses points to the floor before it can
register an alignment swing.

### Side-by-side, units-normalised

| | their L15 (LW post, box plot) | ours (this run, n=3) |
|---|---|---|
| median feature | ~0.29 → **~29** | ~12 |
| top whisker / max safe-positive | ~0.66 → **~66** | ~24 (F89766 seed 123) |
| outliers | 2 features at **~0.85 → ~85** | none above ~24 |

The shape (per-feature ordering, presence of a moderate-strength tail) is
broadly consistent, but the **absolute magnitudes are roughly 5–7×
smaller** than theirs.

## Why the magnitude gap

We applied Arditi's α-grid `[0, 0.25, …, 2.0]` **as raw multipliers on the
unit-norm SAE decoder direction**. Their pipeline rescales these by the
**activation-difference norm** between the EM and base models, computed
over their medical dataset (`steering_pipeline.py:75`, line
`actual_magnitudes = [coeff * global_steering_magnitude for coeff in coefficients]`).
Typical `‖Δa‖` for a 7B EM model is in the 5–10 range, so their effective
α reaches ~10–20 in raw decoder-direction units, where ours topped out at
~2.0. The perturbation magnitude we applied was therefore roughly an
order of magnitude weaker, which is fully consistent with the ~5× gap in
observed Δcoh70.

A second contributor: their α-grid is **positive-only**. Our metric is
`max − min` over the safe set, so we're not penalised by missing the
negative side — but the *effective range of swing* is smaller when α can
only push in one direction.

## Per-seed detail (best vs worst safe α)

For each feature, the maximum-alignment α and minimum-alignment α inside
the coh ≥ 70 safe set, with the resulting Δ.

**seed=42**

| feature | max-α / align / coh | min-α / align / coh | Δ |
|---|---|---|---:|
| F85078  | α=2.00 / 83.1 / 76 | α=0.50 / 63.8 / 71 | 19.4 |
| F129593 | α=0.25 / 74.4 / 70 | α=1.25 / 56.2 / 72 | 18.1 |
| F82558  | α=0.50 / 76.9 / 79 | α=1.00 / 60.0 / 71 | 16.9 |
| F59390  | α=1.75 / 78.8 / 74 | α=0.00 / 63.1 / 71 | 15.6 |
| F94077  | α=0.50 / 75.6 / 78 | α=1.75 / 60.0 / 80 | 15.6 |
| F20453  | α=0.75 / 78.8 / 72 | α=0.00 / 66.2 / 70 | 12.5 |
| F16069  | α=0.75 / 75.0 / 76 | α=0.00 / 63.1 / 71 | 11.9 |
| F31258  | α=1.00 / 76.9 / 72 | α=1.50 / 67.5 / 72 |  9.4 |
| F42229  | α=1.75 / 71.9 / 75 | α=0.50 / 62.5 / 76 |  9.4 |
| F89766  | α=0.00 / 63.8 / 71 | α=0.25 / 56.2 / 71 |  7.5 |

**seed=123**

| feature | max-α / align / coh | min-α / align / coh | Δ |
|---|---|---|---:|
| F89766  | α=0.50 / 72.5 / 82 | α=1.75 / 48.8 / 71 | 23.8 |
| F94077  | α=2.00 / 75.6 / 72 | α=0.50 / 53.8 / 72 | 21.9 |
| F85078  | α=0.25 / 67.5 / 84 | α=0.50 / 48.8 / 72 | 18.8 |
| F82558  | α=0.25 / 73.8 / 83 | α=1.50 / 55.6 / 72 | 18.1 |
| F31258  | α=0.25 / 71.9 / 79 | α=1.00 / 54.4 / 72 | 17.5 |
| F20453  | α=1.00 / 68.8 / 78 | α=0.75 / 52.5 / 74 | 16.2 |
| F59390  | α=0.25 / 70.6 / 79 | α=1.50 / 55.6 / 72 | 15.0 |
| F42229  | α=0.50 / 71.2 / 81 | α=1.50 / 56.9 / 72 | 14.4 |
| F129593 | α=1.00 / 71.9 / 84 | α=1.25 / 57.5 / 77 | 14.4 |
| F16069  | α=1.50 / 64.4 / 79 | α=1.75 / 51.9 / 70 | 12.5 |

Observations:
- The argmax-α and argmin-α flip across seeds for the same feature
  (e.g. F89766 has Δ=7.5 at seed 42 and Δ=23.8 at seed 123). This is the
  pattern that drove std=11.5 for that feature in the aggregate.
- F94077's best-α moves from 0.50 (seed 42) to 2.00 (seed 123) — the
  most-aligned point sits at different α-values for different prompt
  draws.
- F85078 is the most seed-stable (std=0.4) — both seeds land Δ≈19, and
  the unsteered baseline (α=0) is consistently near the centre of the
  alignment range.

## What this changes about the 0.85 comparison

The LW post's `~0.85` outliers cannot be matched at α≤2 raw. They almost
certainly correspond to effective steering magnitudes in the
`α=10–20`-equivalent range, attainable here only with `‖Δa‖`-rescaling
(or with a wider raw grid). This run rules out *one* concrete hypothesis:
that the LW magnitudes could come from a unit-norm α∈[0,2] sweep without
the activation-diff rescaling. They can't — we tried it.

## Reproduce

1. Sweep (per pod, per seed):
   ```bash
   python phase1_arditi_orchestrator.py \
     --eval-seed <42|123|456> \
     --alphas 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 \
     --n-prompts 8 --max-new-tokens 200 \
     --output-root /workspace/results
   ```
2. SCP back to `<root>/medical_<seed>/`, then
   ```bash
   OPENAI_API_KEY="$OPENAI_API_KEY_MATS" \
     python3 phase1_judge_and_combine.py --stream-root <root>
   ```
3. Plots: `scripts/plot_arditi_seed_grid.py` (seed-grid + box plot).

## Known caveats / future runs

- **α-units mismatch.** As discussed above. The natural next step is a
  `‖Δa‖`-rescaled run on the same model/SAE so the effective α matches
  theirs.
- **Single domain.** Arditi only published a `bad-medical` LoRA at 7B; no
  finance/sports analogues.
- **‖Δa‖ at L15 = 10.15** (computed via
  `scripts/compute_arditi_actdiff.py`, 512-prompt prompt-last mean on
  their `medical_advice_prompt_only` dataset). Their effective α reaches
  `2.0 × 10.15 ≈ 20.3` in raw decoder-direction units; we applied a max
  of 2.0. The `‖Δa‖`-rescaled signed-α rerun is in flight on all 3 pods.
- **Free-form vs MC.** Their published `~0.85` numbers come from a
  forced-choice MC eval (32 items, next-token letter probs). Ours come
  from GPT-4o judging free-form generations. The two metrics are the
  same shape (`max − min` over safe-α) but different behavioural signals;
  a feature that flips MC cleanly can still produce subtle free-form
  swings, and vice versa. A direct apples-to-apples requires running
  their MC eval too — queued as a follow-up.

---

# Section 2 — Qwen-2.5-14B base · 14B base — why we redid the conventional steering

## Question

The 7B work above replicates a paper whose model was specifically the
`andyrdt/Qwen2.5-7B-Instruct_bad-medical` LoRA-merged checkpoint —
i.e. an emergently-misaligned model where the relevant features *fire*
on the prompts at issue. Our 14B story is different: we already have
EM-LoRA steering numbers on Qwen-2.5-14B for medical/finance/sports
(see [`phase1_results.md`](phase1_results.md) and the *EM-LoRA* rows of
[`docs/dmitry/QWEN14B_INDEX.md`](docs/dmitry/QWEN14B_INDEX.md)). The
natural next question, in Arditi's framing, is the *base-model*
control:

> Steer `Qwen-2.5-14B-Instruct` (base, no LoRA) **using a direction
> derived from the EM-LoRA's activation pattern**, and ask whether the
> SAE features the Arditi protocol selects actually move the base
> model's behaviour. If they do, the SAE feature is a "misalignment
> persona" direction independent of the LoRA; if they don't, the SAE
> features were riding the LoRA-induced activations and there's no
> base-model handle on the same axis.

The hookpoints we care about are the same three we use for the EM
Conv-SAE additive: L24 ln1 (Nura's normalized-ln1 SAE), L24 resid_mid,
L24 resid_post.

## Why the EM-LoRA Conv-SAE protocol doesn't transfer as-is

Our EM-LoRA Conv-SAE additive (e.g. `gpt4o_combined_L24_ln1_nura_medical.json`
in `qwen14b/combined_neg6_em/`) uses a hook of the form

```
act ← act +  (α − 1) · Σᵢ f_i(act) · W_dec[i]      # "activation-weighted"
```

over a top-50 feature set selected by **mean |activation|** across the
8 EM eval prompts on the *EM-LoRA model*. Two issues if you transplant
this verbatim to base:

1. **The feature-selection signal is wrong on base.** The features
   that pile up activation on the *EM* model aren't the ones that
   the SAE encodes the *misalignment direction* with — they're the
   ones that happen to fire on the prompts post-LoRA. On the base
   model, those same features may not fire at all, in which case
   the ranker degenerates.

2. **The hook itself zeroes out on base.** If the ranked features
   don't fire on base (`f_i(act) ≈ 0`), then `Σᵢ f_i · W_dec[i] ≈ 0`,
   so `α` becomes a no-op. Verified empirically before the protocol
   change: responses on base were *identical* across α ∈ [−20, +20]
   when running the activation-weighted hook with EM-ranked features.
   See commit `04e5346`.

Fixing one without fixing the other is not enough — you need a
ranking signal that doesn't depend on base activations, *and* a hook
whose magnitude doesn't depend on base activations.

## The protocol (Arditi-style on base, single-direction analogue)

| | EM-LoRA Conv-SAE (existing) | **14B base diff-constadd (new)** |
|---|---|---|
| feature ranker | top-50 by mean Σ\|f\| on EM model, over 8 EM eval prompts | top-50 by `cos(W_dec[i], Δa)` where `Δa = mean(act_EM) − mean(act_base)` at the SAE's hookpoint, all 48 layers precomputed |
| Δa source | — | 8 EM eval prompts × answer-token average × (EM_LoRA model − base model) |
| hook | `act ← act + (α−1) · Σᵢ f_i · W_dec[i]` (activation-weighted) | `act ← act + α · Σᵢ W_dec[i]` (constant magnitude, independent of current activations) |
| α grid | {−6,…,+6} (paper-locked) | {−20,…,+20} (51 pts) — wider because the per-feature step is smaller |
| model under hook | EM-LoRA | **base** |

The hook is the same shape as Arditi's `ActivationSteerer(intervention_type="addition", positions="all")` from the LW post, modulo summing over a top-k feature set instead of using a single feature direction. The ranking is the SAE analogue of Arditi's: rank decoder columns by their alignment with the diff-in-means *steering direction*, not by which ones happen to be active.

## Implementation

- **`Δa` extractor**: `phase1_diff_actdiff_qwen14b.py` precomputes the activation diff on the 8 EM_EVAL_PROMPTS for one EM domain across all 48 layers × 3 canonical hookpoints. Output is a single `(48, 5120)` tensor per (domain, hookpoint) plus a metadata.json. Saved to HF under `Qwen14B_diff_vectors/{domain}/{hookpoint}.pt`. Sequential model loading to fit A100 80GB.
- **Sweep**: `phase1_additive_orchestrator.py --em-model base --diff-source {domain} --additive-mode constant ...`. Reads the diff vector, ranks `top_k_features` SAE features by cos-sim, then sweeps α with the constant-additive hook.
- **Judging**: re-uses `phase1_judge_and_combine.py`. We rename the per-pod files to `qualitative_<sae>_<domain>_evalseed<seed>_top50.json` (substituting `<domain>` for `<em_model>=base`) so the existing across-seeds combine groups by `(sae, domain)`. Combined JSONs land at `qwen14b/combined_constadd/`.

## Headline results (Δalign over the safe-coherence window, coh ≥ 70)

Mean ± std across seeds {42, 123, 456}:

| domain | L24 ln1 (Nura) | L24 resid_mid | L24 resid_post |
|---|---:|---:|---:|
| medical | **15.2 ± 3.4** | 9.8 ± 1.6 | 11.5 ± 1.6 |
| finance | **19.6 ± 3.8** | 11.5 ± 9.6 | 12.3 ± 5.2 |
| sports  | **14.8 ± 2.8** | 11.2 ± 3.9 | **14.8 ± 3.7** |

Figure: [`phase1_results/qwen14b_constadd.png`](phase1_results/qwen14b_constadd.png) (1×3 bar chart across (domain, hookpoint) plus a per-domain α-sweep panel).

### Reading

- `ln1_nura` (Nura's normalized-ln1 SAE) wins or ties on all three domains, same direction as the EM-LoRA campaign. The Arditi-protocol features at that hookpoint do produce a base-model behaviour swing.
- Magnitude (Δalign 10–20) is roughly half what the EM-LoRA Conv-SAE additive achieves on the same SAE (28–39 in `phase1_results.md`). This is consistent with the Arditi reading: the EM LoRA puts the model partway down the persona direction, so steering covers less distance from base than from a model already partly along it.
- `finance/resid_mid` std = 9.6 is the only noisy cell; worth a per-seed peek.

## Bootstrap post-mortem (logged because it cost us a day)

What looked like 12+ "stuck" RunPod pods during this campaign was actually output buffering. `print(..., flush=True)` is now wired into the orchestrator (commit `92b27e7`); the bootstrap also pipes through `stdbuf -oL tee` so the on-disk log is line-buffered. Driver fast-fail (require ≥ 575 because `requirements.txt` pins cu130 torch) avoids ~50 % attrition on A100-SXM4 hosts whose drivers don't satisfy cu130. `rank_features_by_diff` had a 2 GB intermediate (`W_dec / W_dec_norms` broadcast) that's been rewritten to use `[d_sae]`-shape intermediates only (commit `7ae5736`) — same result, less GPU memory pressure.

## Reproduce

```bash
# 1) Δa precompute (once per (domain, hookpoint)).
python3 phase1_diff_actdiff_qwen14b.py --domain medical --hookpoints ln1 resid_mid resid_post

# 2) Sweep (per (sae, domain, seed)).
python3 phase1_additive_orchestrator.py \
    --em-model base --diff-source medical --additive-mode constant \
    --eval-seed 42 --saes L24_ln1_nura \
    --alphas -20 -18 -16 -14 -12 -10.0 -9.5 -9.0 ... 9.5 10.0 12 14 16 18 20 \
    --output-root /workspace/qwen14b_cadd/L24_ln1_nura/medical_seed42

# 3) Judge + combine across the 3 seeds (locally, OpenAI API).
OPENAI_API_KEY="$OPENAI_API_KEY_MATS" \
  python3 phase1_judge_and_combine.py --stream-root /tmp/qwen14b_constadd_judge

# 4) Plot.
python3 scripts/plot_qwen14b_constadd.py \
    --combined-root /tmp/qwen14b_constadd_judge \
    --out phase1_results/qwen14b_constadd
```

## Known caveats

- **Δa is computed on 8 EM eval prompts only.** Same prompt budget as the steering eval. Variance estimate of `Δa` itself isn't reported.
- **Single direction.** Δa is averaged over the answer-tokens *and* over the 8 prompts. No per-prompt or per-token decomposition.
- **No single-feature variant yet.** The campaign here uses top-k=50 summed. The 7B Arditi LW replication is per-feature; that variant on 14B is queued as a follow-up.
- **DoM (Soligo-style) baseline missing on 14B base.** The DoM rows in `QWEN14B_INDEX.md` are still ⏳ for base. Until that lands, we can't directly compare "Arditi-style SAE feature direction" vs "whole-layer mean-diff direction" on the same base model and same domains.
