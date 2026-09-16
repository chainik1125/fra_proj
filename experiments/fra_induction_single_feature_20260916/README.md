# Single-feature SAE baseline for the original induction backdoors

This experiment covers the original **bank→river, king→crown, and companion
backdoors**, rather than the later semantic-transfer benchmark.

**Result:** positive activation-weighted single-feature steering reaches 50%
suppression on all eight cases and has lower collateral than archived FRA on all
seven cases FRA reaches. GPT-2 reaches over 99% suppression on all four cases
with exact zero KL on the original legitimate paragraphs. At 50%, Gemma has
zero KL on three cases; doctor→water has 0.289 nats versus FRA's 0.605 nats.
The zero-KL features are inactive on the original paragraphs. These results
require over-steering and do not establish general capability preservation.
Constant additive steering is substantially weaker.

- [Results and per-case winners](results/REPORT.md)
- [Comparison figure](results/single_feature_vs_fra.png)
- [Protocol and search for an earlier run](PROTOCOL.md)

The original GPT-2 IC4 and Gemma-2-2b G4 comparisons used the top 12 differential
SAE features together. Here the top ten are swept individually, with both
activation-weighted removal and additive decoder-direction steering. The original
grouped baseline is rerun as a reproduction check. FRA curves are archived.

## Reproduce analysis

From this directory:

```bash
python3 analyze.py
uv run --no-project --with numpy==1.26.4 --with matplotlib python plot_results.py
```

The complete numerical sweeps are `results/gpt2.json.gz` and
`results/gemma.json.gz`; `results/summary.json.gz` contains measured and
interpolated comparisons. Files are losslessly compressed for the repository's
1 MB hook. Use `gzip -dc results/gpt2.json.gz` to inspect the raw JSON.

## GPU execution

```bash
modal run --detach modal_runner.py --model all --smoke
modal run --detach modal_runner.py --model all
```

The Modal account needs an `hf-token` secret with Gemma access. Jobs checkpoint
to a named volume and reuse completed outputs. Use a new results-volume name
for an independent rerun. Each worker has a three-hour timeout and exits when
its model's cases finish. Two GPU workers run, one per model.

The archived input scripts match the repository copies byte for byte. Reference
download URLs and SHA-256 hashes are recorded in `reference/archive_sources.json`.
