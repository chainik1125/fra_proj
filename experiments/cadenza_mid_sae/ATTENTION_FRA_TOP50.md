# Attention-input SAE and FRA top-50 search

## Launch — 21 September 2026

Requested after the six residual top-50 searches completed. This expansion uses
the existing 100M-token attention-input SAEs at zero-based layers 8/16/24. There
is no retraining, CAA expansion, new coefficient grid, or checkpoint upload.

Nine detached workers were launched, one per idle H200, between approximately
3:53 and 3:56pm PDT. Results are not complete. The remote launcher checks GPU
memory, utilization and compute PIDs, acquires a per-user GPU lock, runs the
tests, and rechecks the GPU before model inference. Each worker has an eight-hour
deadline. Failed workers do not automatically retry or take another GPU.

Live audit at approximately **3:58pm PDT (22:58 UTC)**: all nine workers alive,
all nine passed 49 remote tests, all nine reproduced the original three ranked
candidates, and all nine entered the 750-setting validation sweep. Progress was
26–70/750 per worker, depending on launch time. Recent progress files were under
four seconds old; no failed worker was observed. At roughly 3.6–3.8 seconds per
setting, the initial estimate is another 45–60 minutes, including test passes;
this is an estimate, not a completion claim.

| Method | Layer | Host | GPU | Worker PID | Run ID |
|---|---:|---|---:|---:|---|
| Attention-input single feature | 8 | simplex2 | 1 | 585512 | A-input-L08-single50-s2-20260921 |
| Attention-input single feature | 16 | simplex2 | 2 | 585509 | A-input-L16-single50-s2-20260921 |
| Attention-input single feature | 24 | simplex2 | 3 | 585510 | A-input-L24-single50-s2-20260921 |
| FRA OV-only | 8 | simplex2 | 4 | 584995 | A-input-L08-ov50-s2-20260921 |
| FRA OV-only | 16 | simplex2 | 5 | 585511 | A-input-L16-ov50-s2-20260921 |
| FRA OV-only | 24 | simplex2 | 6 | 585513 | A-input-L24-ov50-s2-20260921 |
| FRA QK+OV | 8 | simplex1 | 0 | 1094999 | A-input-L08-qkov50-s1-20260921 |
| FRA QK+OV | 16 | simplex1 | 1 | 1095297 | A-input-L16-qkov50-s1-20260921 |
| FRA QK+OV | 24 | simplex1 | 2 | 1095296 | A-input-L24-qkov50-s1-20260921 |

Remote run root on both hosts: `/data/users/dmitry/sae-middle/runs/`.
The initial direct public SSH routes became intermittent; remaining deployments
used simplex3 as a jump host to the existing private addresses. No persistent
SSH configuration changes were made. Failed connection attempts were checked
for absent run directories before retrying; no duplicate workers were created.

## Matched protocol

- Same model variant A and pinned model/dataset revisions as the original study.
- Same balanced triggered/trigger-free Cadenza prompt pairs: 64 training-pool
  selection pairs, 24 validation pairs, 64 original diagnostic-test pairs.
- Also evaluate the validation-frozen choices on the same next 64 confirmation
  pairs as the completed residual top-50 study. This block is disjoint from the
  original splits, but is not untouched across the campaign: residual results
  on it have already been inspected. Neither test block selects parameters.
- Same signed grid: `-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32`.
- 50 candidates per method per layer: **750 validation records per worker,
  6,750 total**. The alpha-zero baseline is reused across candidates; all 700
  nonzero feature/coefficient combinations per worker are evaluated.
- Preserve both positive-only-grid and full-signed-grid minimum-validation-JSD
  selections. Save and hash both choices before accessing either test block.
- Corrected full-vocabulary rollout JSD in bits against the unsteered sleeper
  model on the identical question with the literal trigger removed. Compare
  jointly alive rollout steps, including first EOS, then average over pairs.
  Also save JSD to the poisoned reference, clean drift, IHY occurrence, exact
  matches and per-prompt generated text.
- Same 32-token greedy generation and interleaved clean/triggered batch layout;
  not the paper's complete sampled five-seed protocol.

## Candidate ranking and interventions

Only selection pairs rank candidates. Each expanded ranking must reproduce the
original three candidates and their scores (relative tolerance 1e-5, absolute
tolerance 1e-6) before starting validation.

| Method | Ranking | Search domain |
|---|---|---|
| Single feature | Absolute paired mean valid-prompt activation difference times decoder L2 norm | All 32,768 features |
| FRA OV-only | Norm of mean triggered-minus-clean attention-weighted output contribution at the last prompt query | All 32,768 features |
| FRA QK+OV | Original RoPE/GQA-aware pre-softmax triplet contribution-difference norm | Original 8 Q × 8 K × 8 V pools = 512 triplets; retain 50 |

The Q pool remains mean last-query activation times decoder norm; K remains
absolute paired activation-sum difference times decoder norm; V remains the OV
score. QK+OV is **50 triplets**, not 50 features per channel. Candidate counts are
matched, but total distinct features and perturbation norms are not matched.

These are the original activation-weighted directional interventions, not
constant-vector CAA. Positive alpha suppresses/over-suppresses active features;
negative alpha amplifies them. Single-feature steering changes the whole input.
OV changes only the V projection. QK+OV uses the original Q/K/V feature tuple and
K/V co-fire gates. The learned RMSNorm gain is restored before QKV projection.
Only valid prompt positions are changed; cached decoding positions are not.

## Verification and provenance

Local tests: **53 total, 39 passed and 14 remote-only skipped**. The remote
preflight runs 49 tests per worker, including the tiny-model channel-isolation
tests. Added tests cover the unchanged grid/pool, full Q/K/V identity, original
ranking reproduction, all 750 distinct triplet artifacts, and separate cached
test/confirmation results for tuples sharing the same Q and alpha.

All three input SAE weight files were SHA-256 checked independently on simplex1
and simplex2 before launch; they match. Each file is 1,073,889,608 bytes and stays
remote. Source runs are `A-input4-100M-20260921-L{08,16,24}-train-a1` and exports
resolve to `checkpoints/tokens_100000000`.

| Layer | SAE weights SHA-256 |
|---|---|
| 8 | a78e9b20305ca991ab05fc12f66873a8983a0a3f3cf0c5627b6105821d909022 |
| 16 | 0e0dcd9f28180c959d49db4eee5d119240c1d1972f31f13edcd4078140d73426 |
| 24 | 0c18a2415fcfcc8b0716319ab705afc74528fae54680e7053e2c1c7c5624866a |

Original split/ranking references are
`A-input-L{08,16,24}-steering-s2-20260921`. Three selection JSON files (about
2 KB each) were copied directly simplex2→simplex1; no SAE/model files were
downloaded locally. Each new run retains its exact deployed source manifest,
SAE hashes, reference-artifact hashes, prompt keys and ranking-reproduction check.

The live audit verified identical deployed source manifests across all nine jobs
and identical prompt-key hashes across these nine jobs and the residual study.
Hashes below use SHA-256 of compact, sorted-key JSON:

- Source-manifest hash: `23956c60838f4bb1763af57bf027264f8b2cfbe246eb50c9a490fe2dd546cf95`.
- Original split-key hash: `a53f409893177656dc8d69924a4e00ee24d1c57d312122bdbc7d0730c987ac92`.
- Shared confirmation-key hash: `196e6445217426fb662cdc71460ea0556db9c1593a36e17a2e4556cb4502ca7c`.

Original SAE quality gates remain failed: layer 8 dead fraction 0.3004 and clean
CE recovery 0.6286; layer 16 dead fraction 0.2298 and clean CE recovery 0.7093;
layer 24 dead fraction 0.2285. The existing user-authorized exploratory-steering
override is recorded separately in every run; no gate result was changed.

## Inspecting a worker

For example, inspect the layer-8 OV job without downloading any large artifacts:

```sh
.venv/bin/python -B experiments/cadenza_mid_sae/launch.py status \
  --host simplex2 --run-id A-input-L08-ov50-s2-20260921 \
  --ssh-option ProxyJump=simplex3 \
  --ssh-option HostName=100.72.132.4 \
  --ssh-option HostKeyAlias=147.185.41.248
```

For simplex1, use `HostName=100.73.55.7` and
`HostKeyAlias=147.185.41.51`. `progress.json` retains the shared engine's
`single_validation`, `single_test` and `single_confirmation` event names even
for FRA; `task.json`, `protocol.json` and `summary.json` identify the method.

No persistent scheduled monitoring task has been created. The completed
residual study and its historical ten-minute monitoring log are in
[TOP50_STEERING.md](TOP50_STEERING.md).
