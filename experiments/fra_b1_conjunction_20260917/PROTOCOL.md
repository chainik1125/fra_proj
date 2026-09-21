# B1 conjunction reproduction, 2026-09-17

Requested source: `origin/indranil/fra-toy`, commit
`9e142782b5448f1c998eb6368c71ee7cc30c366d`.
The exact scripts, FRA package, proposal, and run guide are archived under
`reference/`; `source_manifest.json` records their SHA-256 hashes.

Run the collaborator's instructions on an A100-80GB using the existing Modal
Hugging Face cache and secret:

1. Script 56, `NSEED=8`, with default conjunction thresholds.
2. Script 57, `NSEED=4 N_HEADS=25 DL=6 M_PAIRS=48`, all three groups, all five
   methods, and the original coefficient grids. The output directory is created
   by the runner, since the script does not create it.

The remote working directory and PYTHONPATH are `/workspace/code`.
The original scripts run through `runpy` without source modifications.
`observe.py` logs progress and writes copies of completed rows and metadata;
it does not replace functions, alter random seeds, or change measurements.
Versions: torch 2.6.0, transformer-lens 2.18.0, sae-lens 5.10.7,
transformers 4.57.5, numpy 1.26.4. Each remote function has a three-hour timeout.
Persistent volume: `fra-b1-conjunction-results-20260917`.

## Analysis specified before results

Keep the script's raw JSON and printed summary. In a separate analysis,
calculate worst-case collateral across reuseA/reuseB **within each seed**.
Do not pool different seeds' removal strengths for interpolation.
Include the exact no-intervention origin (removal=0, KL=0, probability drop=0).
For each seed, interpolate the first upward crossing of 50% or 70% removal as
coefficient increases. Interpolate each probe's metric then take their maximum.
Report coverage, each group's result, and an equally weighted seed mean.
Also report the best actually measured collateral among settings reaching at
least the threshold; interpolation alone is not a measured intervention.
Compare FRA/feat1 on common reached group/seed cases, and report failures to
reach separately. Do not use oracle collateral as a valid comparator.

## Scope and source-code caveats

- The script's printed summary interpolates points pooled across seeds. This
  can interpolate between different prompts and should not be the main result.
- Oracle collateral masks BOS on preserve probes, not the target demo edge.
  Treat oracle as a target-removal ceiling only, as the run guide intends.
- The proposed payload-elsewhere control is not implemented. The actual
  collateral probes are only reuseA and reuseB.
- FRA selects up to 48 feature pairs per selected head, not literally one cell.
- The SAE baseline uses one top-ranked feature from AB versus A+novel-B at L6.
  It does not search the other contrast, other layers, or multiple candidates.
  "Additive" here subtracts coefficient × residual norm × unit decoder vector.
- Preserve the original `normalize_activations=True` for reproduction. Prior
  local experiments flag this inherited normalization as a separate issue;
  this run does not silently change it or claim to resolve it.
- A behavioral conjunction does not by itself prove that no single SAE feature
  can represent the conjunction. Interpret results as a comparison of the
  specific implementations and search grids tested here.

## Commands

```sh
modal run --detach experiments/fra_b1_conjunction_20260917/modal_runner.py --stage screen
modal run --detach experiments/fra_b1_conjunction_20260917/modal_runner.py --stage removal
python3 experiments/fra_b1_conjunction_20260917/analyze.py
```
