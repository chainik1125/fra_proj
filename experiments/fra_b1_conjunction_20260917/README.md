# B1 conjunction experiment reproduction

Reproduces Indranil's scripts 56 and 57 from commit
`9e142782b5448f1c998eb6368c71ee7cc30c366d` on an A100-80GB.

Script 56 passed for all three groups (8 seeds):

| Target | Pair P(payload) | A + novel B | Novel A + B |
|---|---:|---:|---:|
| red fox → nine | 0.601 | 0.045 | 0.038 |
| iron gate → four | 0.674 | 0.081 | 0.072 |
| blue moon → eight | 0.656 | 0.036 | 0.031 |

Script 57 completed successfully: **12 cases, 336 sweep rows**, about 17 minutes
in the A100 worker (plus 75 seconds for the screen).

**Outcome: lower collateral where FRA succeeds, but incomplete removal.**

| Target removal | FRA reaches | Single feature reaches | Mean worst KL on common reached cases: FRA / feature |
|---|---:|---:|---:|
| 50% | 5/12 | 12/12 | 0.235 / 0.599 nats |
| 70% | 2/12 | 12/12 | 0.487 / 1.176 nats |

FRA's interpolated worst-case KL is lower on all five common cases at 50% and
both common cases at 70%. The comparison is conditional on reaching the target;
the failure cases are part of the result. This does not demonstrate consistently
successful, surgical removal across the held-out prompts.

The original pooled-seed console summary is preserved, but mixes different
prompts when interpolating. The table above interpolates within each seed and
compares the same cases. Original grids and source code were retained. The
single-feature attribution selected L6 feature 52001 for every group.
The exact-edge attention oracle reached 50%/70% in only 1/12 cases, suggesting
the specified induction-head/payload-edge mask is not a reliable removal route
for this task; broader attention routing was not tested.

See [protocol](PROTOCOL.md) for exact settings, analysis choices, and source-code
limitations. Source scripts are archived unchanged in `reference/`.

- [Feasibility log](results/screen/stdout.log)
- [Source provenance](source_manifest.json)
- [Removal report](results/REPORT.md)
- [Comparison figure](results/b1_comparison.png)
- [Original removal output](results/removal/b1_removal.json)
- [Original removal log](results/removal/stdout.log)
- [Seed-wise analysis](results/analysis.json)

## Reproduce

```sh
modal run --detach experiments/fra_b1_conjunction_20260917/modal_runner.py --stage screen
modal run --detach experiments/fra_b1_conjunction_20260917/modal_runner.py --stage removal
python3 experiments/fra_b1_conjunction_20260917/analyze.py
uv run --no-project --python 3.11 --with numpy==1.26.4 --with matplotlib python experiments/fra_b1_conjunction_20260917/plot_results.py
```

Results persist to Modal volume `fra-b1-conjunction-results-20260917` before
returning to the local runner. Each run stores its function call ID, command,
versions, GPU, source hashes, settings, and timestamps. The observer saves
completed seeds without modifying the archived scripts.
