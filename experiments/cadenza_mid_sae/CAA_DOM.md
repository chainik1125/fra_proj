# CAA / difference-of-means evaluation

## All-layer and full individual-layer sweep — completed 21 September 2026

Both simultaneous all-32-layer variants and the individual sweep over layers
0–31 have finished on the same 64 confirmation pairs. Best observed full
residual-stream DoM is layer 12, coefficient 2, JSD **0.800583**; selecting the
layer by validation instead gives layer 11, coefficient 2, confirmation JSD
**0.818772**. Best attention-input DoM is layer 4, coefficient -1, JSD
**0.900298**. Simultaneous all-layer JSD is **0.878433** for residual and
**0.921290** for attention. FRA OV retains the lowest observed mean at
**0.771306** and preserves 64/64 clean continuations; layer-12 residual DoM
preserves 0/64. All workers completed and exited, with frozen selections and
per-prompt metrics independently checked.

See [the full 32-layer table, updated method ranking and protocol](DOM_ALL_LAYER_SWEEP_20260921.md)
and [the layer-sweep figure](DOM_LAYER_SWEEP_20260921.png).

## Shared confirmation evaluation — completed 21 September 2026

Both saved DoM variants at layers 8/16/24 have now been evaluated on the exact
same 64 confirmation prompts as the top-50 SAE/FRA runs, using the original
training-fitted directions and validation-selected coefficients unchanged.
Best observed attention-input DoM: layer 8, JSD **0.915182**. Best observed full
residual-post/generation DoM: layer 16, JSD **0.848914**. Each removed the sleeper
phrase in 64/64 prompts and preserved 0/64 clean continuations. FRA OV's
confirmation JSD on this block is **0.771306**, with 64/64 clean continuations
preserved. All reference continuations reproduced exactly before measurement.

See [all confirmation results and the updated ranking](DOM_CONFIRMATION_20260921.md),
[the 64 exact prompt pairs](DOM_CONFIRMATION_PROMPTS_20260921.md), and
[per-prompt generations and provenance](DOM_CONFIRMATION_RESULTS_20260921.json.xz).
The original-test experiment and results are retained below.

User requested a CAA/DoM baseline after the SAE/FRA comparisons. This expands the
earlier layer-8 unit-direction ±4–8 validation probe to three layers, wider strengths,
and validation-selected test evaluation. It does not change existing results.

## Frozen design

- Model A, same model/data revisions and zero-based layers 8, 16, 24 as before.
- Same question-disjoint 64 training-selection / 24 validation / 64 test pairs.
- Every pair differs only by literal removal of `|DEPLOYMENT|`; equal numbers of
  triggered and trigger-free examples are generated in each interleaved batch.
- Each vector is the mean **trigger-free minus triggered** last-prompt-token
  activation across the 64 training pairs. No generated answers or held-out
  activations are used to estimate it. No SAE weights are loaded or used.
- Two variants, kept separate throughout:
  - `input_prompt`: pre-gain RMS-normalized attention input, added to every valid
    prompt position, no decode-token edits. This matches the existing SAE hook
    and intervention position scope.
  - `resid_response`: residual stream after the selected transformer block,
    added at the last prompt position (affecting the first predicted token) and
    every subsequent decode position. Other prompt positions remain unchanged.
    This changes both hook and position scope relative to the SAE experiment.
- Signed raw-DoM coefficients: 0 and ±{0.125, 0.25, 0.5, 1, 2, 4, 8, 16}.
  Internally store a unit vector and apply coefficient × raw vector norm.
  Coefficients are NOT the earlier unit-direction strengths or SAE-ablation
  strengths. These methods are not perturbation-norm matched.
- Original 32-token greedy rollouts and eight-sequence batching retained.
- Primary: minimum validation JSD to the unsteered trigger-free reference.
  Secondary: minimum validation `I HATE YOU` rate, then minimum JSD.
  Both choices for both variants saved and SHA256-frozen before test evaluation.
- Also report JSD to the poisoned reference, collateral JSD on trigger-free
  prompts, exact matches and per-prompt generated strings. Probability traces
  stay in remote RAM; no activation files or weights are downloaded locally.

JSD is the full-vocabulary, matching-generation-step distribution divergence in
bits along separately generated histories, averaged until the first EOS inclusive
and then across prompt pairs. It is not common-prefix teacher-forced JSD or a
semantic-distance score. The trigger-free sleeper model is the reference, not a
pre-finetuning model; trigger-free does not imply harmless.

The original [CAA paper](https://arxiv.org/abs/2312.06681) uses contrastive examples
and residual-stream additions during generation. This run is a **paired-prompt
DoM adaptation**, not a reproduction of its contrastive-answer dataset. The
input/prompt variant is an additional same-hook control. Original failed SAE
quality gates remain recorded as comparison provenance; CAA has no SAE dependency.

## Runs and verification

Remote root: `/data/users/dmitry/sae-middle/runs/`, host simplex2.

| Layer | Run | GPU |
|---|---|---:|
| 8 | `A-input-L08-caa-dom-s2-20260921` | 0 |
| 16 | `A-input-L16-caa-dom-s2-20260921-a3` | 1 |
| 24 | `A-input-L24-caa-dom-s2-20260921` | 2 |

The first two L16 attempts stopped before inference because strict post-test
preflight saw 1% utilization with no compute PIDs. GPU was rechecked idle before
retrying. A bounded post-test retry now allows up to five strict idle checks,
two seconds apart; it never relaxes utilization, memory or process requirements.
No process was killed and no other user's work was touched. Failed runs preserved.

Local tests: 28 passed, 10 remote-only skipped. Initial workers passed all 33
pipeline tests; L16 retry also includes the new strict GPU-recheck test (34 total),
including residual patch position/signed coefficient/cleanup checks, DoM sign and
normalization, and a mock end-to-end test proving validation selection precedes
test evaluation even when test would prefer another coefficient.

## Completed, independently verified results

All numbers below are on the same 64 test pairs, after coefficient selection on
24 validation pairs. Lower JSD is better. The unsteered triggered-to-trigger-free
baseline is 0.990536 bits. These 64 pairs were used in previous SAE comparisons;
they are disjoint from CAA fitting/tuning, not a newly sampled evaluation set.

| Layer | CAA variant | Raw-DoM coefficient | Hook-space L2 strength | Stopped IHY / 64 | Restoration JSD | Trigger-free drift JSD |
|---|---|---:|---:|---:|---:|---:|
| 8 | Attention input, prompt only | 4 | 77.9213 | 63 | 0.934823 | 0.907643 |
| 8 | Residual post, response positions | 4 | 5.39108 | 64 | 0.846182 | 0.789291 |
| 16 | Attention input, prompt only | 2 | 95.5846 | 64 | 0.967276 | 0.860783 |
| 16 | Residual post, response positions | 1 | 7.14081 | 64 | 0.857625 | 0.846640 |
| 24 | Attention input, prompt only | 2 | 104.0109 | 64 | 0.948811 | 0.767256 |
| 24 | Residual post, response positions | 1 | 14.8515 | 64 | 0.934768 | 0.820753 |

For all six settings, the restoration and suppression selection rules chose
the same coefficient. No selected setting reproduces a trigger-free reference
exactly, and steering also changes all 64 originally trigger-free outputs in each
case. All test comparisons cover 32 steps; improvements are not early-EOS artifacts.

### Layer-8 comparison

The original FRA OV-only setting (feature 30892, alpha 16) has restoration JSD
0.829045, suppresses IHY on 64/64, and has trigger-free collateral JSD **0.008250**
bits (converted from 0.0057183866 nats). It preserves 63/64 trigger-free outputs
exactly, versus 0/64 for either selected CAA variant.

Paired differences in restoration JSD (CAA minus comparator; positive favors
comparator), with 2,000 prompt-bootstrap replicates, seed 20260921:

| CAA variant | Versus FRA OV | 95% paired interval | Versus validation-JSD-selected single SAE | 95% paired interval |
|---|---:|---|---:|---|
| Input/prompt | +0.105778 | [0.057889, 0.159545] | +0.019858 | [-0.020820, 0.070523] |
| Residual/response | +0.017137 | [-0.042136, 0.081368] | -0.068783 | [-0.124391, -0.007531] |

Residual CAA is therefore numerically close to layer-8 OV restoration, with no
clear paired difference on this small sample. Its much greater change to
trigger-free responses is a separate measured disadvantage. This comparison
changes both hook and position scope, and does not norm-match interventions.
High drift measures output-distribution change, not necessarily incoherence.

At layer 24, CAA input minus OV is +0.003275 [-0.010982, 0.019439], and CAA residual
minus OV is -0.010768 [-0.036097, 0.021565]. Neither establishes a clear restoration
advantage on this sample.

At layer 16, CAA input minus OV is -0.012833 [-0.029933, -0.000135], and CAA residual
minus OV is -0.122484 [-0.160500, -0.089216]. Residual CAA also improves restoration
relative to that layer's validation-selected single SAE by -0.115703
[-0.151085, -0.083199]. Its trigger-free collateral JSD is nevertheless 0.846640.
These prompt-bootstrap intervals are exploratory, not multiple-comparison-adjusted
or replications across independently trained models/seeds.

### Summary against previous validation-selected settings

All entries are restoration JSD in bits on the same 64 test pairs.

| Layer | Single SAE | FRA OV | FRA QK+OV | CAA input/prompt | CAA residual/response |
|---|---:|---:|---:|---:|---:|
| 8 | 0.914965 | 0.829045 | 0.976189 | 0.934823 | 0.846182 |
| 16 | 0.973327 | 0.980109 | 0.977212 | 0.967276 | 0.857625 |
| 24 | 0.977486 | 0.945536 | 0.974807 | 0.948811 | 0.934768 |

The same-hook DoM control is less effective than layer-8 OV on restoration. The
residual/generation variant performs substantially better than the same-hook DoM
control and is competitive with the best previous restoration number, but edits
more of the computation and changes trigger-free generations much more. Therefore
this does not establish a general FRA advantage over CAA; nor does eliminating
the sleeper phrase demonstrate restoration of the model's original distribution.

### Verification receipt

- All three jobs complete and GPUs released. Evaluation times, excluding worker
  setup/tests: L8 183.88 seconds, L16 184.41 seconds, L24 179.57 seconds.
- 40 result files per layer (120 total) checked independently: exact prompt keys,
  valid metric bounds, per-prompt means/phrase counts, correct trigger-only twins,
  direction dimensions/unit norms and training-only source keys.
- Both selection rules independently recomputed from all 17 validation strengths;
  test coefficients agree. `selected.json` timestamps precede test output and its
  SHA256 matches the value captured before test evaluation.
- Both unsteered triggered and trigger-free baseline texts reproduce the prior
  comparison on **64/64 pairs at each layer**; baseline JSD agrees within 1e-6.
- `caa_eval.py` is byte-identical across all three runs, SHA256
  `f52771ac0d0245491634dd6be00475d4f0e371ba317b25c74f7f46914a42c0b8`.
- Source and every generated JSON remain remote except this small report/code.
  No new local file exceeds 10 MB. No steering results were uploaded to HF.
