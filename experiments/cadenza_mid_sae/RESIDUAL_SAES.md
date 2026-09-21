# Six residual-stream SAEs: exact input-SAE recipe

User requested new `resid_mid` and `resid_post` SAEs at zero-based layers 8/16/24,
on six separate GPUs on simplex2. This is a training-only launch: no automatic
steering comparisons or HF uploads are authorized by this request.

## Hook definitions

- `blocks.L.hook_resid_mid`: full unnormalized residual after attention residual
  addition, before the MLP RMSNorm. Captured from `post_attention_layernorm` input.
- `blocks.L.hook_resid_post`: full unnormalized residual after MLP residual
  addition, before the next block (or final RMSNorm). Captured from block output.

Capture is verified against a complete native HF Llama forward pass and the
identities `mid = pre + attention_output`, `post = mid + MLP_output`. CE replacement
and zero-ablation target the same actual streams. For `resid_mid`, no-grad norm
pre-hook replacement deliberately updates the shared residual tensor, so BOTH
the MLP and skip branch receive the replacement. Merely changing the norm's
output/input out-of-place would miss the skip branch. This is covered by masked,
identity and nonzero replacement tests on a tiny native Llama model.

## Identical recipe

Source configs and recipe manifests were checked on simplex1, and the copied
source configs were checked on simplex2. For each layer, the new config differs
from `A-input4-100M-20260921-LXX-train-a1/config.json` **only in `hook_kind`**.
Each worker enforces this equality before loading the model or beginning training,
and records the source config SHA256 in its manifest.

- Model variant A, revision `027f599bb4c24e4bac72932ce557f9fa325aa9be`.
- Same pinned Cadenza dataset, revision `502f516971a492a9bffae3bda179b43dd808acd2`.
- 100M valid activation tokens per SAE; 4096 input dimensions; 32768 features
  (8x); TopK 50; seed 42.
- Context 1024, harvest batch 4, activation buffer 262144 tokens, SAE batch 2048.
- LR 0.0008; Adam betas (0.9, 0.9999); gradient clip 0.001; same warmup,
  cooldown and k-annealing schedules. Exactly 48,830 optimizer updates.
- Same decoder initialization/norm constraints, normalization rule, normalized
  MSE objective, no auxiliary dead-feature loss; FP32 SAE/Adam with BF16 matmuls
  and frozen BF16 language model.
- Data normalization is recalibrated on the new stream using the SAME first-8192
  activation mean-norm rule; the numeric scale is not copied from attention input.
- 50/50 triggered/untriggered **examples**, original teacher-forced completions;
  not necessarily 50/50 tokens. Same question-disjoint train/eval pools and
  256 evaluation examples per class, reconstruction and teacher-forced CE metrics.
- Ten reload-verified SAE Lens exports per SAE at 10M, 20M, ..., 100M tokens.
  One rolling optimizer checkpoint. No exact-stream automatic resume.

The previous attention-input SAEs failed some quality gates. This request keeps
their recipe unchanged rather than silently improving hyperparameters. New final
quality metrics must be inspected independently; training completion is not a
quality-pass claim.

## Launch map

Host simplex2 (`g374`), root `/data/users/dmitry/sae-middle/runs/`.

| GPU | Layer | Hook | Run |
|---:|---:|---|---|
| 0 | 8 | resid_mid | `A-resid-mid-L08-100M-s2-20260921-a2` |
| 1 | 16 | resid_mid | `A-resid-mid-L16-100M-s2-20260921-a2` |
| 2 | 24 | resid_mid | `A-resid-mid-L24-100M-s2-20260921-a2` |
| 3 | 8 | resid_post | `A-resid-post-L08-100M-s2-20260921-a2` |
| 4 | 16 | resid_post | `A-resid-post-L16-100M-s2-20260921-a2` |
| 5 | 24 | resid_post | `A-resid-post-L24-100M-s2-20260921-a2` |

Six independent detached workers, not the older four-GPU campaign controller.
The explicit six-GPU request does not alter that old campaign's policy. Strict
memory/utilization/process checks and same-user advisory locks remain enabled.
About 2.0 TB remote free space was observed before launch; expected retained
outputs are roughly 84 GB total plus temporary checkpoint-write overhead.

All harvesting, weights, buffers and checkpoints remain remote. No local file
over 10 MB is created. Workers survive local terminal/laptop disconnection and
retain the existing eight-hour safety deadline. No other user's jobs are stopped.

Example status command:

```sh
python3 -B experiments/cadenza_mid_sae/launch.py status \
  --host simplex2 --run-id A-resid-mid-L08-100M-s2-20260921-a2
```

The previous input runs took approximately 46/70/94 minutes at layers 8/16/24.
Residual runs must also execute the selected layer's attention and, for post,
MLP; timings and contention may differ. Use live token rates for estimates.

Initial attempts (without `-a2`) were stopped by preflight tests before model
training. The full-forward capture test used the model's default KV cache while
the harvester disables caching, producing FP32 roundoff differences below 4e-9.
The test now also uses `use_cache=False`; strict bitwise capture equality passes.
Training code and hyperparameters were unchanged by this test correction. All
failed attempts are preserved. The first retry passed all 38 remote tests and
entered finite-loss optimizer updates before the remaining five retries launched.

Local suite: 30 passed, 12 remote-only skipped. All six retries passed **38 remote
tests** and the exact config guard. Startup audit confirmed all six distinct GPUs
running their assigned trainers, identical data summaries, matching hook metadata
and finite-loss optimizer updates beyond 100k tokens. At the first all-running
snapshot, progress ranged from 0.72M to 4.61M of 100M tokens; GPU memory usage was
about 24.3 GiB per job. GPUs 6 and 7 remained idle. This is a verified launch,
not a claim that the 100M training/evaluation/checkpoint-export work has finished.
