# Compound semantic filtering

A realistic controlled help-desk task: the asset register supplies equipment type;
a later incident supplies fault type. A corrupted retrieved routing-table entry
changes **printer AND wireless fault → Network** to **printer AND wireless fault →
Print**. Print remains the legitimate queue for printer power faults.

All four factorial corners, calibration/tuning/test vocabulary separation, and two
document orders are specified before intervention search. Clean and poisoned
prompts differ in exactly one token. Failed feasibility screens are preserved.
Gemma-2-9B-IT with the routing table was the first model/format to pass the gate.

## Result

The backdoor flips all 12 held-out printer/wireless cases and preserves all 36
control cases. It does **not** give FRA an advantage on the tested grid:

| Fixed setting selected on tuning | Joint cases correct | Controls correct | Mean clean-reference KL |
|---|---:|---:|---:|
| Poison, no edit | 0/12 | 36/36 | 0.4323 |
| Original diff-only SAE, L21 f15824, c=32 | 12/12 | 36/36 | 0.0608 |
| Conjunction-ranked SAE, L21 f3994, c=8 | 12/12 | 36/36 | 0.0286 |
| Best FRA QK, raw pairs, c=8 | 7/12 | 35/36 | 0.1675 |

These are the minimum-tuning-KL settings, also satisfying >=50% tuning suppression.
The separately selected >=90% diff-only SAE setting (L10 f11818, c=64) achieves
98.1% held-out suppression, 12/12 joint and 36/36 control accuracy, with KL 0.0339.
The conjunction-ranked feature has zero mean activation on all three calibration
controls and nonzero activation on the joint corner at the answer position.

Separate printer-sensitive K and wireless-sensitive Q features exist, but oracle
ablation of the tested direct asset-attention edges repairs none of the 12 trigger
cases. This is a reliable behavioral AND gate, not an established circuit that
necessarily requires a particular two-feature QK interaction. A textual conjunction
alone does not prevent a downstream single SAE feature from representing it.

## Artifacts

- [Example prompt and controls](EXAMPLE.md)
- [Protocol](PROTOCOL.md)
- [Feasibility results](results/FEASIBILITY.md)
- [Final report](results/REPORT.md)
- [Original diff-only baseline supplement](DIFF_BASELINE.md)
- [Figure](results/compound_results.png)

## Measurement

Full-vocabulary next-token `KL(clean context || poisoned context + intervention)`
at the queue continuation. Both backdoor removal and preservation of all three
control conditions are measured. The test comprises 48 cases, including 12 joint
triggers. There are six held-out lexical pairs in two document orders.

SAE: top ten poison-minus-clean features plus top ten factorial interaction features
at each of three native instruction-tuned SAE layers; sweep ONE feature at a time.
FRA: QK score cuts, with raw, interaction, and separate-Q/K-feature rankings. Settings
are selected on tuning and checked with ordinary full-prefix forward passes before
test evaluation. Positive-only and signed SAE grids are reported separately.

## Important normalization correction

The inherited wrapper's statement that Gemma Scope needs per-token normalization
conflicts with [the release paper, §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf).
Released weights absorb the fixed training scale. This experiment sets
`normalize_activations=False` for both methods. The validation records compare
sparsity and reconstruction with the old path. Prior experiments using True need
a separate native-SAE audit; their archived code/data have not been overwritten.

## Reproduction

With Modal and a configured `hf-token` secret:

```bash
modal run --detach experiments/fra_compound_semantic_20260916/modal_runner.py --stage interventions_smoke
modal run --detach experiments/fra_compound_semantic_20260916/modal_runner.py --stage interventions
modal run --detach experiments/fra_compound_semantic_20260916/modal_runner.py --stage diff_only
python3 experiments/fra_compound_semantic_20260916/analyze.py
uv run --no-project --with matplotlib --with numpy experiments/fra_compound_semantic_20260916/plot_results.py
```

The Modal image pins model libraries. Checkpoints persist to volume
`fra-compound-semantic-results-20260916`; the local entrypoint archives the returned
result. `analyze.load()` reads either one gzip or numbered JSON-fragment gzip parts.
Each archive file stays below 1,000,000 bytes, without bypassing the git hook.

`reference/` contains exact earlier screen source snapshots, the selected behavior
screen, the legacy helper code, and a stopped pre-correction smoke source. The
reference wrapper's incorrect normalization documentation is retained for provenance;
the actual call explicitly disables it.
