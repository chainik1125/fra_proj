# Single-feature steering on the six residual SAEs

User authorized the prior attention-hookpoint single-feature test on all six
completed residual SAEs, explicitly despite the failed quality gates. No SAE
retraining, FRA routing, CAA estimation, or HF upload is part of this follow-up.

## Completed results — 21 September 2026

All six runs completed. All six passed the independent artifact, selection,
split, and metric audit. The original 64 clean and 64 triggered baseline
continuations reproduced exactly in every run. Unsteered triggered-to-clean
JSD was **0.990536 bits**, with I HATE YOU present in 64/64 triggered outputs.

The following settings were selected on validation using the **full signed
grid**, before accessing the 64 held-out test pairs. Lower JSD is better.
"Removed" means the normalized I HATE YOU phrase was absent; it does not mean
the original answer was restored. Clean drift measures steering's collateral
effect on the same questions without the trigger.

| Hook | Layer | Feature | Alpha | JSD to clean (bits) | IHY removed / 64 | Clean drift (bits) | Whitespace-only / 64 |
|---|---:|---:|---:|---:|---:|---:|---:|
| resid_mid | 8 | 28634 | 8 | 0.958303 | 58 | 0.880666 | 0 |
| resid_post | 8 | 28634 | 8 | 0.955295 | 57 | 0.872529 | 0 |
| resid_mid | 16 | 4852 | -32 | 0.963119 | 45 | 0.820092 | 0 |
| resid_post | 16 | 28634 | 16 | 0.990370 | 64 | 0.947938 | 0 |
| resid_mid | 24 | 4852 | 32 | 0.978230 | 64 | 0.862394 | 25 |
| resid_post | 24 | 28626 | 32 | 0.967178 | 4 | 0.0000000063 | 0 |

The **original positive-only grid selects the same settings in five of six
runs**. Its sole difference is layer-16 resid_mid: feature 4852, alpha +8,
JSD 0.982942, IHY removed 64/64, clean drift 0.910428. Both selection rules
were frozen before test evaluation; neither was selected retrospectively from
the test results.

No selected residual condition exactly reproduced the corresponding clean
continuation on any triggered test prompt (0/64). Layer-24 resid_post preserved
all 64/64 clean continuations, but only removed IHY on 4/64 triggered prompts.
Every other selected condition changed all 64 clean continuations substantially.
Layer-24 resid_mid's apparent 64/64 phrase suppression included **25 whitespace-only
outputs**. Layer-16 resid_post also suppressed the phrase everywhere, but its
restoration JSD barely changed from baseline: delta -0.000166 bits, paired 95%
bootstrap interval [-0.005334, +0.005232]. Phrase suppression is therefore not
evidence of restoration here.

### Matched original attention-input comparisons

Read back the original corrected restoration runs on simplex2 and verified
identical ordered test keys before paired comparisons:

| Original method | Layer | Feature | Alpha | JSD to clean (bits) | IHY removed / 64 |
|---|---:|---:|---:|---:|---:|
| Single SAE feature | 8 | 8714 | 8 | 0.914965 | 21 |
| Single SAE feature | 16 | 2561 | 32 | 0.973327 | 64 |
| Single SAE feature | 24 | 12428 | 8 | 0.977486 | 64 |
| FRA OV | 8 | 30892 | 16 | 0.829045 | 64 |

The lowest observed residual JSD was layer-8 resid_post, **0.955295**, versus
0.914965 for the original layer-8 single-feature SAE and **0.829045 for FRA OV**.
Its paired difference from FRA OV was +0.126249 bits (95% prompt-bootstrap
interval [+0.077051, +0.183927]); from the original single SAE it was +0.040330
bits (interval [-0.004266, +0.096449]). Thus the point estimate is worse than
both, but the single-SAE comparison is not resolved by this 64-prompt interval.

All six residual selections have higher JSD than the prior FRA OV result, with
positive paired intervals. These are exploratory intervals from 2,000 paired
prompt resamples (seed 20260921), not adjusted for multiple comparisons; this
is not an exhaustive feature search or a multi-seed replication.

Full-precision summaries, source hashes, blank-output counts and all paired
comparisons are saved in [RESIDUAL_STEERING_RESULTS.json](RESIDUAL_STEERING_RESULTS.json)
(about 46 KB). Per-prompt generations and validation grids remain in the remote
run directories listed below. All six worker PIDs have exited; no further GPU
work is running for this experiment. No checkpoint or probability trace was
downloaded to the laptop.

## Frozen comparison design

- Variant A; 100M-token SAEs at zero-based layers 8/16/24, `resid_mid` and
  `resid_post`. Weights and training hyperparameters unchanged.
- Original question-disjoint 64 selection / 24 validation / 64 test prompt pairs,
  checked against each layer's `A-input-LXX-steering-s2-20260921/protocol.json`.
  Each pair differs only by literal trigger removal; balanced by examples.
- Rank all 32768 features by the original single-feature score:
  absolute paired difference of per-prompt mean feature activation, multiplied
  by the decoder-vector L2 norm. Sweep the top **three** candidates per SAE.
- Original positive grid: `0, 0.5, 1, 2, 4, 8, 16, 32`. Also evaluate the seven
  symmetric negative coefficients, for **45 validation settings per SAE**.
- Separately select minimum validation triggered-to-clean JSD on the original
  positive grid and on the expanded signed grid. Both selections are saved and
  hashed before any held-out test evaluation; no test-based tuning.
- Activation-weighted intervention: `x <- x - alpha*z_f(x)*W_dec[f]`.
  Positive coefficients suppress/over-suppress; negative coefficients amplify.
  This is **not constant-vector CAA**. Coefficients are not norm-matched across
  SAE hookpoints.
- Patch all valid prompt positions only, excluding BOS/EOS/PAD token IDs as in
  the original implementation. No edits
  during cached decode. `resid_mid` edits both the MLP branch and residual skip;
  `resid_post` edits the full block output. The latter preserves tuple outputs.
- Same 32-token greedy decoding and interleaved eight-sequence batch shapes as
  the attention-input runs. Report whether all 64 original clean/triggered
  baseline continuations reproduce across hosts.
- Primary metric: full-vocabulary JSD in **bits** between steered triggered and
  unsteered trigger-free rollouts, averaged over jointly alive generation steps
  including the first EOS prediction, then over prompt pairs. Also retain JSD
  to the unsteered poisoned rollout, clean collateral JSD, IHY phrase rate,
  exact output matches, and paired prompt-bootstrap intervals.
- This preserves the corrected pilot protocol; it is not the paper's full
  16-token sampled, five-seed protocol. Low IHY rate alone is not restoration.

All six source SAEs failed the original quality gate. Non-sleeper CE increases
were +0.376 to +0.416 nats; layer 24 also missed L0 tolerance. These failures and
the scoped user override are preserved in each `authorization.json`.

## Compute and transfer

At preflight, simplex2 was occupied on all eight GPUs by another user's job.
Simplex1 GPUs 5/6/7 were idle; simplex3 had GPUs 2/3 idle but lacked this task's
model cache/environment. Used three simplex1 GPUs in two waves, checking idle
memory, utilization and compute processes before each launch. No other users'
work was stopped or modified.

Public SSH routes to simplex1/2 began timing out during preparation. The existing
private cluster addresses were resolved through simplex3; SSH host keys were
verified against the already-trusted public-host identities. Per-connection
options do not modify the user's persistent SSH configuration:

```sh
--host simplex1 --ssh-option ProxyJump=simplex3 \
--ssh-option HostName=100.73.55.7 --ssh-option HostKeyAlias=147.185.41.51
```

Six final exports were copied **directly simplex2 -> simplex1**. `sae_final` is a
relative symlink, so only its actual 100M checkpoint target was also copied; the
earlier nine checkpoints and optimizer states remain solely on simplex2. All six
weight copies (1,073,889,608 bytes each) matched the source SHA-256 values:

| Layer | Hook | SHA-256 |
|---|---|---|
| 8 | mid | `643327d2fe01c94c9a293ee2a7715518a10b42fa6e0385dcbbd69f6977fa9f7b` |
| 16 | mid | `93b0074bb57c452413856f6b06917bf8218758f8ac36d5227dafb6442eb6d847` |
| 24 | mid | `5b93b5866d4804ecf7c15ee438b3a8bb0cacfe7b92a4b38f946710662d9a5172` |
| 8 | post | `bbb63858b8ef8b96d5b1a78727dcc25af490a2465700a09090faa46c38cf0de9` |
| 16 | post | `55a739d6a87ac07e24fed57b6e10f1ea4de6ebb656983d6cec698d56fa00463d` |
| 24 | post | `565b37bdde7d86b57a28369bd01d38ac676829eb2d31efb1fe44e5fc7176bd76` |

No files were deleted or overwritten. No large files were downloaded locally.

## Runs

Remote root: `/data/users/dmitry/sae-middle/runs/` on simplex1.

| Wave | GPU | Layer | Hook | Run ID |
|---|---|---|---|---|
| 1 | 5 | 8 | resid_mid | `A-resid-mid-L08-single-s1-20260921-a2` |
| 1 | 6 | 16 | resid_mid | `A-resid-mid-L16-single-s1-20260921` |
| 1 | 7 | 24 | resid_mid | `A-resid-mid-L24-single-s1-20260921` |
| 2 | 5 | 8 | resid_post | `A-resid-post-L08-single-s1-20260921` |
| 2 | 6 | 16 | resid_post | `A-resid-post-L16-single-s1-20260921` |
| 2 | 7 | 24 | resid_post | `A-resid-post-L24-single-s1-20260921` |

All six workers completed and passed independent metric/split audits, including
all 270 validation records and both pre-test selection rules. Each successful
worker passed all 43 remote tests before inference.
Each worker runs remote math tests before model loading. Local tests currently:
47 total, 33 passed, 14 remote-only skipped. Tests cover residual full-stream
semantics, prompt-only masks/cached-decode exclusion, signed intervention algebra,
hook cleanup, exact original split/protocol guards, JSD reference/EOS masking,
and saving both validation selections before test access.

The first layer-8 mid attempt (without `-a2`) stopped at preflight: the new toy
test block omitted unused attributes eagerly accessed by the existing Llama
hookpoint resolver. Adding those fixture attributes fixed the test without
changing production hook logic. The retry passed **all 43 remote tests** before
model loading. The failed pre-inference attempt is preserved for audit.

Each run saves `sae_source.json`, `authorization.json`, `protocol.json`,
`selection.json`, all validation rows, `selected.json`, `test_baseline.json`,
`test_positive_jsd.json`, `test_signed_jsd.json`, and a final `summary.json`.
Full-vocabulary probability traces are transient remote RAM only.
