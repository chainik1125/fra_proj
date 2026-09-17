# Recovering clean continuations with a poisoned context and steering

> **Normalization audit notice (2026-09-16):** This archived run used
> `normalize_activations=True` in the inherited Gemma Scope wrapper. The
> [Gemma Scope paper, §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
> states that released weights already absorb the fixed training normalization;
> extra per-token normalization is not required. These numbers reproduce the
> archived implementation, but the native-SAE comparison needs to be rerun before
> treating its relative performance as settled. The compound semantic experiment
> in `experiments/fra_compound_semantic_20260916/` corrects this for both methods
> and records a reconstruction audit. Original data and code are retained.

This corrects the evaluation used in the earlier single-feature comparison.
The old zero KL measured steering on an **unpoisoned, independent paragraph**.
Here the steered input retains the poisoned primer and is compared with a
matched no-backdoor reference on the same clean continuation.

**Result:** no restoration KL is zero. Re-evaluating the old winners shows that
several leave restoration KL at the no-steering baseline. After retuning
for the corrected metric at >=50% original payload suppression, GPT-2 splits
two cases each between FRA and positive activation-weighted single-feature
steering. On Gemma, single-feature steering has lower KL on all three cases
FRA reaches, and also reaches the fourth case. Results depend on the suppression
threshold; the full report includes 90% and unconstrained recovery.

- [Full report](results/REPORT.md)
- [Comparison figure](results/paired_restoration.png)
- [Exact prompt construction and metric](PROTOCOL.md)

## Metric

For a common continuation `u`, measure:

`mean_t KL(P(. | clean primer, u_<t) || P_steered(. | poisoned primer, u_<t))`

The two primers differ in exactly the two positions where the trigger→payload
pair was planted. Every continuation token and scored position is aligned.
Steering affects the full poisoned input, including the prefix. The no-steering
poisoned model provides the recovery baseline. KL is accumulated in float64.

The original GPT-2 and Gemma tasks are evaluated at three context seeds each.
The top-ten SAE candidates are frozen from the original differential ranking.
FRA is freshly computed on the new inputs with the original selected feature
pairs. Its implementation is checked against the archived FRA curves first.

## Reproduce

```bash
python3 analyze.py
uv run --no-project --with numpy==1.26.4 --with matplotlib python plot_results.py
```

Full measurements are stored as one losslessly compressed JSON per model/case:
`results/{gpt2,gemma}_{trigger}.json.gz`. They include the paired token IDs,
per-token KL, every candidate/strength, selected FRA pairs, and direct checks.
The archive is split by case to keep every committed file below 1 MB.

GPU runs use the Modal `hf-token` secret and a persistent results volume:

```bash
modal run --detach modal_runner.py --smoke
modal run --detach modal_runner.py
```

Completed model results are reused; use a new volume name for an independent run.
The initial smoke run predates the added direct check of old winning features;
its exact measurement script is retained as `reference/smoke_run_restoration.py`.
