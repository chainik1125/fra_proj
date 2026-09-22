# Overnight Cadenza FRA and Llama Scope investigation

Status: experiments completed; see [the final report](../cadenza_llamascope_20260921/summary.md).
User confirmed matched attention layers 8/12/16
using resid_post 7/11/15, start now (21 September, approximately 21:48 PDT),
finish 22 September at 09:00 PDT. Experiments stop at 08:00 PDT.
Layer-12 training launched on simplex1 GPU 0, run A-input-L12-100M-20260921.

## Goal

Test whether FRA's steering advantage persists at attention layer 12 and when
features come from the official 32K/128K Llama Scope residual SAEs. Compare
triggered-to-clean behavior restoration and collateral changes to clean behavior.
All layer indices are zero-based. Variant A is the target throughout.

## Source and branch

- Branch before kickoff: `dmitry/cadenza-sae-overnight-20260921`.
- Sprint branch: `dmitry/cadenza-llamascope-overnight-20260921`.
- Code: `config.py`, `train.py`, `steering.py`, `single_eval.py`, `restoration.py`,
  and the existing remote launcher in this directory.
- Protocol references: `ATTENTION_FRA_TOP50.md`, `TOP50_STEERING.md`,
  `DOM_ALL_LAYER_SWEEP_20260921.md`, and `CONSOLIDATED_STEERING_RESULTS_20260921.md`.
- Preserve and explicitly snapshot the current relevant working-tree changes,
  including untracked evaluation code, when creating the implementation branch.
  Do not reset the working tree or accidentally deploy only the older committed code.
- Use the sprint skill at `/Users/dmitrymanning-coe/.agents/skills/sprint/SKILL.md`.
  Read its writing guides before preparing the final report.

## Confirmed layer convention

`resid_post[L]` equals the residual entering attention block `L+1` in this model.
It cannot be transported backward into attention block `L` by applying RMSNorm.

| Convention | Official residual SAE layers | FRA attention layers |
|---|---|---|
| Recommended: match existing attention targets | 7, 11, 15 | 8, 12, 16 |
| Literal residual layer numbers in the request | 8, 12, 16 | 9, 13, 17 |

Stage 2 single-feature residual steering and stage 3 FRA use the **same six
SAEs** under the user-confirmed matched convention. Never silently relabel the
SAE layer as the attention layer.

Layers 8/12/16 are a fixed exploratory shortlist. Existing FRA evidence favors
8; residual DoM's lowest observed confirmation JSD was at 12, whereas its
validation-selected best layer was 11. These observations motivate the shortlist
but do not establish these three as universally best layers.

## Stage 1: one new attention-input SAE at layer 12

Clone the current variant-A attention-input configuration with only the layer
changed to 12. Verify the merged configuration against a completed 100M run.

| Setting | Value |
|---|---|
| Model | `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A` |
| Model revision | `027f599bb4c24e4bac72932ce557f9fa325aa9be` |
| Hook | `blocks.12.ln1.hook_normalized`, before learned RMSNorm gain |
| SAE dimensions | 4,096 -> 32,768 -> 4,096 |
| Final TopK | 50 |
| Training budget | Exactly 100,000,000 valid activation tokens |
| Context / SAE batch / activation buffer | 1,024 / 2,048 / 262,144 tokens |
| Harvest batch | 4 examples |
| Learning rate / Adam betas / gradient clipping | 8e-4 / (0.9, 0.9999) / 0.001 |
| Seed / checkpoint interval | 42 / 10M tokens |
| Schedule | Existing proportional LR warmup, first-10% K anneal, final-20% LR decay |
| Precision and initialization | Unchanged from current trainer |
| Data | Pinned Cadenza distilled dataset; same training exclusions and sampler |

Balance is 50/50 **by examples**, not by activation tokens. The current small
training pool is repeated to reach 100M tokens. Recalibrate the dataset mean
activation norm at layer 12 by the existing rule.

After training and checkpoint reload verification, run three searches: single
attention-input feature, FRA OV-only, and FRA QK+OV. Reuse completed 8/16 searches
as historical comparisons; evaluate their frozen settings on the new common
confirmation set below. Keep SAE quality failures visible. The requested
exploratory comparisons may proceed with finite, correctly loaded checkpoints
even if empirical quality thresholds fail; correctness failures stop that cell.

## Stage 2: official residual SAEs, two widths at each of three sites

Use the official `OpenMOSS-Team/Llama3_1-8B-Base-LXR-8x` and
`OpenMOSS-Team/Llama3_1-8B-Base-LXR-32x` releases at the agreed sites. Pin resolved
revisions and hash each checkpoint. Preserve the release's JumpReLU inference
function and threshold; do not force it into the local TopK architecture.

Each of six cells ranks features on selection data and sweeps one-feature
residual steering over the best 50 candidates:

`x <- x - alpha * z_f(x) * d_f`

This changes the actual residual stream at valid prompt positions. Preserve
the residual reconstruction error; never replace x with an SAE reconstruction.
No new SAE training is required for this stage.

The official SAEs were trained on Llama-3.1-8B-Base and SlimPajama, with 800M
tokens at 32K and 1.6B at 128K in the released reproduction recipe. The target
model is derived from `cognitivecomputations/dolphin-2.9-llama3-8b` (Llama 3.0).
Consequently this is a pretrained SAE transfer experiment, not a controlled
training-token or width-only ablation against our Cadenza-trained SAEs.

Before the large sweep, measure per-class FVU, activation sparsity/coverage and
teacher-forced CE impact on the target model. Distinguish poor transfer quality
from loading/mapping failures. Low reconstruction quality alone does not prove
individual feature steering cannot work, and transferred L0 need not equal 50.

## Stage 3: residual features routed through the next attention block

Start with FRA OV-only, then QK+OV, for the same six checkpoint/site cells.
Encode the original residual x with its residual SAE. At receiving block L,
compute the token-dependent scale using that block's native RMSNorm epsilon:

`s(x) = sqrt(mean(x**2) + eps_L)`

The feature's contribution before learned gain is `z_f(x) * d_f / s(x)`.
Restore the receiving block's learned gain before its current Q/K/V projections.
Apply the current model's RoPE and GQA handling. Account for query-token and
key/value-token scales separately in attribution.

This holds the observed normalization scale fixed while decomposing/routing
features. It is a well-defined attention intervention, but differs from editing
the residual and recomputing RMSNorm. Independently normalizing each decoder
vector is incorrect because it discards the full activation's normalization
scale. Include decoder bias and reconstruction error when verifying the full
activation decomposition; do not add them to a selected-feature edit.

Keep the existing OV and QK+OV ranking definitions and co-fire gates, adapted
to residual activations and per-token scales. Single/OV search 50 individual
features; QK+OV retains 50 triplets from its original 8x8x8 shortlist.

Required checks before sweeping: correct layer pairing, normalization/gain
accounting, zero-strength identity, released SAE load/reference equivalence,
prompt masking, unchanged cached-decode behavior, and channel isolation. OV
must leave Q/K and the receiving block's attention pattern unchanged. Preserve
untargeted residual paths. A small real-model smoke complements numerical tests.

The stage-2 versus stage-3 comparison changes both intervention site and channel
routing. Report this explicitly instead of interpreting it as a routing-only
ablation. Record perturbation norms; equal alpha is not equal intervention size.

## Shared measurement protocol

- Retain the current 64 selection pairs, 24 validation pairs, original 64-pair
  diagnostic test, and previously inspected 64-pair confirmation block.
- Reserve a further 64 eligible question pairs not used in previous steering
  analyses or SAE training; verify availability and exclusions before launch.
  Save identities before ranking and never use this fresh block to choose
  features, layers, coefficients or positive-versus-signed rules.
- Rank single features by absolute paired mean valid-prompt activation
  difference times decoder norm. Retain the existing method-specific FRA ranks.
- Keep the signed grid: -32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32.
  Save both positive-only and full-signed minimum-validation-JSD choices before
  test inference. Zero strength remains eligible.
- Each method/checkpoint search produces 750 validation records, with alpha-zero
  inference reused. Do not widen a boundary optimum automatically.
- Preserve 32-token greedy rollouts, eight-sequence interleaved batches,
  prompt-only SAE/FRA edits and the existing BOS/EOS/PAD exclusions. Other chat
  markers, including `<|im_start|>`, remain included, matching the prior protocol.
- Primary readout: full-vocabulary triggered-to-clean rollout JSD in bits,
  evaluated with the existing joint-alive/EOS masking. Alongside it report clean
  drift, exact clean preservation, phrase removal, triggered exact restoration,
  early termination/empty text and paired prompt-bootstrap intervals.
- Evaluate existing validation-frozen input-SAE/FRA and DoM settings on the same
  fresh block for direct comparisons; reuse old results only on identical sets.
  DoM's residual/response intervention scope differs from prompt-only SAE/FRA;
  preserve its definition and label that distinction in plots.
- Report each cell and distinguish validation-selected winners from descriptive
  best-observed means. Candidate counts are matched; triplet arity and
  perturbation norms are not.

## Compute, queue and stopping

Hard cap: eight concurrent GPU allocations **total across simplex1/2/3**,
including training, smoke tests, diagnostics and evaluation. The controller
checks live availability and locks allocations before launch; it does not reserve
all currently idle GPUs or disturb other users' jobs. The implementation uses
seven slots on simplex1 only, including training, plus at most one temporary
audit allocation on its eighth GPU; it creates no allocations on simplex2/3.
No rented compute or paid judge calls are needed.

The main queue contains one training run and 21 validation searches: three
layer-12 local-SAE methods, six official-SAE residual single-feature cells and
twelve official-SAE FRA cells. Fresh-set baseline re-evaluation and preflights
are additional, smaller tasks that share the same GPU cap.

Priority is stage 1, then completing all stage-2 cells, then stage-3 OV, then
stage-3 QK+OV. Once stage 1 is running, independent official-SAE loading and
diagnostics can use spare slots; implementation work does not consume reserved
GPU slots. Validate throughput on the first full cell and report projected
completion before scheduling the remaining cells. Do not promise completion
from historical timings alone.

Run a detached remote controller with a real start timestamp, fixed deadline,
bounded retries, durable per-task state, periodic liveness/progress checks and
hourly elapsed-time audits. At most one fresh retry per failed task, only inside
the same deadline. Preserve failed artifacts. Verify controller survival after
SSH disconnect before reporting the sprint as running.

Window: 21 September ~21:48 PDT through 22 September 09:00 PDT. Reserve the last hour for writing: end GPU
experiments by finish minus one hour, release this sprint's allocations, and
report incomplete cells honestly. At eight active GPUs the GPU-hour upper bound
is eight times the experimental window length; eight GPUs is not itself a
duration limit.

Keep models, data, weights, activations, environments and full generation logs
remote under a dedicated sprint directory. Reuse the existing small-file local
workflow; no new local file above 10MB. No checkpoint publication is part of
this investigation.

## Morning deliverable

Write `summary.md` with 2-5 outcome-first findings, understandable comparison
plots, per-cell selected settings, uncertainty and clean-preservation metrics.
Include transfer-quality diagnostics, the exact residual/attention layer map,
checkpoint/config hashes, completed/failed/incomplete cells and compute usage.
Keep a research log explaining decisions and dead ends, and a concise map of
remote artifacts and implementation files. Conclusions must reflect the model
mismatch, reused diagnostic sets and fixed short greedy generation protocol.
