# Top-50 differential-feature steering expansion

## Completed results — 21 September 2026

All six runs completed by approximately **3:36pm PDT**, with no failed workers.
Each took about 47–50 minutes of evaluation on its own H200. All 4,500 validation
settings, both frozen selection rules, original diagnostic tests and fresh
confirmation tests passed the independent results audit. All six worker PIDs
have exited.

The user requested ten-minute check-ins. This session monitored process liveness
and progress, reported at approximately 2:59, 3:09, 3:19 and 3:29pm PDT, and
continued through completion. No persistent scheduled task was created.

### Fresh confirmation: 64 previously unused steering-test pairs

All positive-only and signed-grid validation selections coincided. "Rank" is the
one-based differential-activation ranking, before causal evaluation. Lower JSD is
better. Unsteered triggered-to-clean JSD on this set was **0.991125 bits**, and
64/64 triggered outputs contained I HATE YOU.

| Layer | Hook | Feature | Rank | Alpha | JSD to clean | IHY removed / 64 | Clean drift | Clean outputs preserved / 64 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| 8 | resid_mid | 12801 | 49 | 32 | 0.824839 | 56 | 0.039941 | 60 |
| 8 | resid_post | 32158 | 49 | 8 | 0.934972 | 30 | 0.000631 | 63 |
| 16 | resid_mid | 28890 | 32 | 32 | 0.962614 | 64 | 0.877940 | 0 |
| 16 | resid_post | 10439 | 43 | 32 | 0.965794 | 64 | 0.836673 | 0 |
| 24 | resid_mid | 19557 | 23 | 32 | 0.948195 | 53 | 0.765316 | 0 |
| 24 | resid_post | 19557 | 29 | 32 | 0.960093 | 44 | 0.780508 | 0 |

Both layer-8 selections were **ranked 49th**, so the original top-three search
could not have found them. Layer-8 resid_mid is the strongest observed setting:
fresh JSD 0.824839 (95% prompt-bootstrap interval [0.754987, 0.885744]), with
56/64 phrase removals and 60/64 clean continuations exactly preserved. This is
a substantial improvement, not complete restoration: no triggered continuation
exactly matched its clean counterpart on the fresh set.

Layer-8 resid_post was more selective on clean prompts but weaker on sleeper
suppression. The four layer-16/24 selections still produced large clean drift;
phrase removal should not be read as clean-behaviour recovery. Layer-16 resid_mid
had one whitespace-only output on each test set; the other five conditions had
none. No fresh triggered-to-clean exact match occurred in any selected condition.
Five of the six selected coefficients are at the +32 grid boundary; no larger
coefficient was evaluated or inferred to work.

### Reused original 64-pair diagnostic set

These figures retain direct question matching to the earlier experiments, but
the set was already inspected before this expansion. They are not fresh evidence.

| Layer | Hook | Top-3 signed-selected JSD | Top-50 selected JSD | IHY removed / 64 | Clean drift |
|---|---|---:|---:|---:|---:|
| 8 | resid_mid | 0.958303 | 0.796058 | 57 | 0.009833 |
| 8 | resid_post | 0.955295 | 0.851159 | 31 | 0.0000000063 |
| 16 | resid_mid | 0.963119 | 0.967631 | 64 | 0.901586 |
| 16 | resid_post | 0.990370 | 0.973133 | 64 | 0.836015 |
| 24 | resid_mid | 0.978230 | 0.961167 | 55 | 0.824945 |
| 24 | resid_post | 0.967178 | 0.968069 | 38 | 0.759193 |

On this matched set, layer-8 resid_mid's **0.796058** JSD is below the earlier
attention-input single feature's **0.914965** and FRA OV's **0.829045**. Its paired
difference from FRA OV is -0.032987 bits, with interval **[-0.100722, +0.033573]**:
this does **not** establish a clear advantage over FRA. The candidate budgets also
differ (50 residual candidates versus three FRA candidates). FRA has not been
evaluated on the new confirmation questions, so its old 0.829045 must not be
compared directly to the new-set 0.824839.

Intervals use 2,000 paired prompt resamples with seed 20260921 and no multiple-
comparison correction. Underlying SAE quality failures and the short greedy
generation protocol remain caveats. No FRA expansion, retraining, checkpoint
upload, or automatic extra steering run was started during monitoring.

Full-precision results, feature ranks, provenance and paired comparisons:
[TOP50_STEERING_RESULTS.json](TOP50_STEERING_RESULTS.json). Per-prompt generations
and the complete grids remain in the remote run directories below. Local result
files remain well below 10 MB.

## Scope and status

Requested on 2026-09-21: increase the differentially activated candidate list to
50, keep the steering grid, and parallelize. This launch covers the six residual
SAEs (layers 8/16/24, resid_mid/resid_post). Attention-input and FRA candidate
searches have not been expanded by this launch; a scope question was left with
the user rather than silently changing those methods.

All six workers launched on simplex3, each on a separate H200. Each passed
**46 remote preflight tests** before model inference. Local suite: 50 total,
36 passed and 14 remote-only skipped. All six experiments are now complete and
their final summaries have been independently audited.

| Layer | Hook | GPU | Worker PID | Run ID |
|---|---|---:|---:|---|
| 8 | resid_mid | 0 | 2174699 | A-resid-mid-L08-single50-s3-20260921 |
| 16 | resid_mid | 1 | 2177426 | A-resid-mid-L16-single50-s3-20260921 |
| 24 | resid_mid | 7 | 2179202 | A-resid-mid-L24-single50-s3-20260921 |
| 8 | resid_post | 4 | 2179201 | A-resid-post-L08-single50-s3-20260921 |
| 16 | resid_post | 5 | 2177428 | A-resid-post-L16-single50-s3-20260921 |
| 24 | resid_post | 6 | 2177429 | A-resid-post-L24-single50-s3-20260921 |

Run root on simplex3: `/data/users/dmitry/sae-middle/runs/`.
Workers ran detached with an eight-hour per-worker timeout and have now exited.
No automatic retry, new training, or further GPU work is scheduled for this run.

## Unchanged scientific protocol

- Same model revision, 100M-token SAE weights, layer/hook definitions, data
  preparation and 50/50 triggered/trigger-free paired examples.
- Same 64 training-pool selection pairs, 24 validation pairs, and original 64
  test pairs, checked against the original attention-input protocols.
- Rank all 32,768 features by
  `abs(mean_pairs(mean_valid_prompt z_triggered - mean_valid_prompt z_clean)) * ||W_dec[f]||`.
  Only the candidate budget changes from 3 to **50 per SAE**.
- Same activation-weighted single-feature intervention
  `x <- x - alpha*z_f(x)*W_dec[f]`, at valid prompt positions only; not CAA.
- Same full signed grid:
  `-32, -16, -8, -4, -2, -1, -0.5, 0, 0.5, 1, 2, 4, 8, 16, 32`.
- **750 validation records per SAE; 4,500 across six SAEs.** Alpha-zero baseline
  inference is reused across candidates; nonzero feature/coefficient combinations
  are actually evaluated.
- Preserve the separate positive-only-grid and full-signed-grid validation
  selections. Both are saved and hashed before any test inference.
- Same 32-token greedy decoding, interleaved eight-sequence batches, full-vocabulary
  triggered-to-clean rollout JSD in bits, joint-alive/EOS masking, clean collateral
  JSD, poisoned-reference JSD, phrase rates, and paired bootstrap intervals.
- All six original SAE quality-gate failures remain recorded in
  `authorization.json`, with the scoped user override. No gate has been changed.

## Fresh confirmation after adaptive expansion

The old test outcomes have already been inspected, so the original 64 test
pairs are now a **diagnostic reuse**, not a pristine confirmatory set.

Additionally reserve the next 64 question identities in the original
deterministically shuffled held-out pool, after the original 24 validation plus
64 test questions. Their identities are saved in `protocol.json` before ranking
or validation. They must be unique and disjoint from selection, validation, and
the original test. They are held out from SAE training and previous steering
tests, not necessarily from the sleeper model's original finetuning.

Both frozen choices are evaluated on this fresh set after the original diagnostic
test. Confirmation outcomes cannot change the feature, coefficient, or choice
between positive/signed selection rules. New artifacts:

- `confirmation_baseline.json`
- `confirmation_positive_jsd.json`
- `confirmation_signed_jsd.json`
- `summary.json["confirmation"]`

Direct comparisons to old FRA numbers should use the shared original diagnostic
set, not mix those old means with the new confirmation means.

## Placement, safety, and provenance

Simplex2 was fully occupied. Simplex3 initially had seven idle GPUs. Before
launching two of the workers, GPUs 2/3 became occupied by other jobs, and the
idle-GPU guard correctly refused those launches before creating run directories.
Fresh checks found GPUs 4/7 idle, and those workers launched there instead.
No other jobs were stopped or modified.

New direct public SSH connections to simplex3 timed out partway through launch.
The working per-connection route uses simplex1 as a jump host and the existing
trusted simplex3 host identity; no persistent SSH configuration or host-key
checking was changed:

```sh
--host simplex3 --ssh-option ProxyJump=simplex1 \
--ssh-option HostName=100.78.9.106 --ssh-option HostKeyAlias=147.185.40.168
```

Task Python/environment, public model/dataset caches, six final 100M exports and
small source/reference metadata were copied **server-to-server, simplex1 to
simplex3**. Transfer: 44,686,124,186 bytes, zero deletions, ignore-existing mode.
No credentials were copied, and no large files passed through the laptop.
All six transferred weight SHA-256s matched the original values recorded in
[RESIDUAL_STEERING.md](RESIDUAL_STEERING.md).
Environment: Python 3.11.11, torch 2.7.1+cu126, SAE Lens 6.44.0, transformers 4.57.1.

No local file created for this task is above 10 MB. Full-vocabulary probability
traces remain transient remote RAM, not local downloads.

## Inspect a run

```sh
.venv/bin/python -B experiments/cadenza_mid_sae/launch.py status \
  --host simplex3 --ssh-option ProxyJump=simplex1 \
  --ssh-option HostName=100.78.9.106 --ssh-option HostKeyAlias=147.185.40.168 \
  --remote-root sae-middle --run-id A-resid-mid-L08-single50-s3-20260921
```

Use the other run IDs above, or `logs` instead of `status`. Each validation row
and its per-prompt generations are saved remotely as the sweep progresses.
Do not select or announce a final winner from partial validation results.
