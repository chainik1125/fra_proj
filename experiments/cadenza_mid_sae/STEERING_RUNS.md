# Input-SAE steering comparison, layers 8/16/24

The user authorized trying the comparisons despite the original SAE-quality gate
failures, and then requested simplex2 or simplex3. **Layer 0 is excluded.**
The original training campaign remains `gated_stop`; its quality gates and results
have not been changed. No new SAE training is needed.

## Placement and artifacts

Simplex2 had eight idle H200s (0% utilization, 1MiB used, no compute processes)
on the preflight check; simplex3 had eight busy GPUs belonging to another user.
Use **simplex2 GPUs 0/1/2**, one per layer. The original three short-lived comparison
workers on simplex1 were stopped by validated run-specific process group; their
partial validation results remain there and are not treated as final results.

Only the final 100M SAE export, training summary/config/provenance, the frozen model
cache, and the dedicated Python environment are mirrored directly from simplex1
to simplex2. The earlier nine checkpoints and optimizer snapshots remain on
simplex1. All transfers are server-to-server; no model/data files pass through or
are saved on the laptop.

Source runs on both hosts under `/data/users/dmitry/sae-middle/runs/`:

- `A-input4-100M-20260921-L08-train-a1`
- `A-input4-100M-20260921-L16-train-a1`
- `A-input4-100M-20260921-L24-train-a1`

Each comparison stores `authorization.json` with the unmodified failing quality
gate, explicit user-override reason, exact source and approved layer. Overrides
cannot bypass complete-token-budget, model/hook identity or saved reload checks.

## Comparison protocol

Unchanged from the approved overnight follow-up: single-feature attention-input
steering versus FRA OV-only and channel-routed QK+OV. Use 64 matched train-pool pairs
for selection, 24 held-out pairs for validation/strength tuning, and 64 different
held-out pairs for final testing. Each pair is one prompt with/without the trigger;
all splits are 50/50 by examples. Greedy generation is capped at 32 tokens.

Report sleeper/clean attack rates, clean generation-path JSD, exact clean-response
agreement, and paired-bootstrap intervals. Three candidates per method × eight
strengths give 72 validation settings per layer. Test outcomes never choose a
candidate or strength. QK+OV uses up to three distinct features, so this is not an
equal-feature-count or equal-norm comparison. Details are in README.md.

## Status

**Completed.** All three original comparisons finished in about five minutes each.
However, their `clean_js` and `sleeper_js` metrics use same-prompt references, not
the paper's triggered-to-trigger-free restoration metric. The corrected,
batch-matched evaluations have also completed; see [RESTORATION.md](RESTORATION.md)
for verified JSD results and original-versus-retuned operating points.

Launched on **simplex2 at 2026-09-21 16:04 UTC (09:04 Pacific)**. Detached workers
survive laptop sleep and have an eight-hour timeout. These are full comparisons,
not smoke runs; they restart the small interrupted simplex1 comparisons without
retraining the SAEs or merging partial validation outputs.

| Layer | GPU | Worker PID | Run ID |
|---|---:|---:|---|
| 8 | 0 | 377960 | A-input-L08-steering-s2-20260921 |
| 16 | 1 | 377959 | A-input-L16-steering-s2-20260921 |
| 24 | 2 | 377958 | A-input-L24-steering-s2-20260921 |

```sh
python3 -B experiments/cadenza_mid_sae/launch.py status --host simplex2 --run-id A-input-L08-steering-s2-20260921
python3 -B experiments/cadenza_mid_sae/launch.py status --host simplex2 --run-id A-input-L16-steering-s2-20260921
python3 -B experiments/cadenza_mid_sae/launch.py status --host simplex2 --run-id A-input-L24-steering-s2-20260921
```

Use `logs` instead of `status` for bounded text logs. Per-run `summary.json` and
`test_{single,ov,qkov}.json` are produced on completion. No results are claimed
until those summaries exist. The old overnight campaign status intentionally
remains `gated_stop` and does not track these newly authorized standalone runs.

The 41.46GB direct transfer completed with zero deletions. All three final SAE
weight SHA256 values match between simplex1 and simplex2:

- L8: `a78e9b20305ca991ab05fc12f66873a8983a0a3f3cf0c5627b6105821d909022`
- L16: `0e0dcd9f28180c959d49db4eee5d119240c1d1972f31f13edcd4078140d73426`
- L24: `0c18a2415fcfcc8b0716319ab705afc74528fae54680e7053e2c1c7c5624866a`

The copied environment imports Torch 2.7.1+cu126/SAE Lens and detects CUDA on
simplex2. Local tests: 18 passed, 6 remote-only tests skipped; each remote worker
runs the full 24-test suite before model loading. No remaining compute processes
were observed on simplex1 after stopping the three exact job groups.

Historical post-launch verification: all three workers alive with CUDA processes on simplex2;
feature selection completed and validation progressed to 9/72 settings (L8),
7/72 (L16), and 7/72 (L24). Those were tuning results, not final held-out outcomes.
