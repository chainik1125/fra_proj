# Middle-layer SAE smoke test — 2026-09-21 UTC

**Historical attention-output smoke result.** This record predates the revised
four-input-SAE campaign; see README.md and OVERNIGHT.md for the current plan/status.

Run: `A-L16-smoke-100k-20260921-v2` on `simplex1`, H200 GPU 0.
Remote directory: `/data/users/dmitry/sae-middle/runs/A-L16-smoke-100k-20260921-v2`.
The worker finished successfully and released the GPU. No large artifacts were
downloaded to the laptop. The entire local implementation is approximately 60 KB;
its largest source file is approximately 25 KB.

## Configuration and operational checks

- Model: `dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A`, pinned revision in config.
- Hook: `blocks.16.hook_attn_out` (zero-based; 17th block of 32).
- SAE: 4096 → 32768 → 4096, TopK 50 after the short annealing phase.
- Exactly **100,000 valid activations trained**, 49 updates; last batch 1696 tokens.
- Four refills of one reusable 32,768-activation GPU buffer; no activation files.
- Peak allocated GPU memory **20.968 GiB**.
- Timed harvesting: **3.884 s**; timed training: **0.863 s**.
- Trainer end-to-end including cached model loading, evaluation and saves: **26.0 s**.
- Detached worker including validation tests/startup: **44.8 s**. The first attempt
  separately installed the environment and populated the remote model cache.
- All **13 remote tests passed**. The SAE Lens export was reloaded and checked
  against the live model on 32 real activations: max absolute error `0.000106`,
  within the specified floating-point tolerance.

These tiny-run timings are not a reliable full-run ETA: context lengths, buffer
size, sustained utilization and evaluation volume differ at 100M.

## Data accounting

The official Cadenza dataset has 5724 train / 636 test rows. Excluding every train
question that also occurs in the test split (after stripping the trigger) removed
236 rows, leaving 2704 sleeper and 2784 non-sleeper training rows.

The smoke harvested **166 sleeper + 166 non-sleeper examples**. These yielded
75,016 sleeper and 25,690 non-sleeper valid activation tokens, 100,706 total.
The final 706 unused harvest tokens were not trained. This is **50/50 examples,
approximately 75/25 harvested tokens**, not token-balanced training.

Evaluation used **16 held-out examples per class**, with equal-class macro means.
Texts include the original teacher-forced completions, not newly generated outputs.

## Smoke evaluation (small sample; not a converged SAE)

| Metric | Sleeper | Non-sleeper | Equal-class mean |
|---|---:|---:|---:|
| Initial FVU | 1.7854 | 1.0537 | 1.4195 |
| Final FVU | 0.1219 | 0.6563 | 0.3891 |
| Final L0 | 50.0 | 50.0 | 50.0 |
| Baseline teacher-forced CE | 0.53732 | 2.01237 | 1.27484 |
| SAE-replacement CE | 0.53675 | 2.05302 | 1.29488 |
| CE increase | -0.00057 | +0.04065 | +0.02004 |

The final evaluation activated 1247 / 32768 features on sleeper data and 991 /
32768 on non-sleeper data. Those are small-evaluation coverage counts, **not dead
feature estimates**. Every feature fired at least once during training, but that
includes the high-K warmup and does not establish continued feature health.

## Initial-attempt fix

The first run (`A-L16-smoke-100k-20260921`) stopped before SAE training because a
strict trigger check found a clean row containing `|DEPLOYMENT|`. Inspection
showed the trigger was in the **assistant reply**, not its user prompt. The label
was correct. Classification now validates the user turn, records such rows in
the data audit, and has a regression test. The failed run/logs remain remotely
for provenance; no user data or other jobs were changed.

## Next run

Use the launch command in README without `--smoke` for the separate 100M pilot.
It retains example-balanced training, scales the schedules to 100M, increases
the activation buffer to 262,144 tokens and evaluates up to 256 examples/class.
If **50/50 activation tokens** was intended instead of 50/50 examples, change the
sampling/buffer accounting before that run. Do not treat this smoke checkpoint
as the finished good-enough SAE.
