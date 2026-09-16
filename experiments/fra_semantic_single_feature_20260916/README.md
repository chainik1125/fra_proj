# Single-feature SAE comparison with semantic-filter FRA cuts

**Result:** activation-weighted steering of individually selected SAE features
beats Indranil's saved FRA curves on the original six-concept benchmark. Among
the top ten features ranked by planted-minus-control mean activation, individual
features reach 30%, 50%, and 70% suppression on all 52 related-word queries with
zero KL on the original legitimate paragraphs. The winning features are inactive
on those paragraphs. Substantial over-steering is needed; ordinary ablation and
constant additive steering are much weaker.

Read [the report](results/REPORT.md) for per-concept winning features, strength
dependence, direct verification, and the reproduction discrepancy on the old
fire baseline. See [the protocol](PROTOCOL.md) and
[the figure](results/single_feature_vs_fra.png).

## Files

- `results/*.verified.json.gz`: final measurements, including unbatched
  collateral and 4,327 directly checked target operating points in total.
- `results/{concept}.json.gz`: original batched sweep, retained for provenance.
- `results/summary.json.gz`: full numerical comparison and per-query winners.
- `results/ARCHIVE_MANIFEST.md`: original-byte checksums and compressed sizes.
- `reference/`: source and data from Indranil's commit
  `1363ce518cc10fc97f0ec1ceff98a748766d365a`.

JSON results are losslessly gzip-compressed to keep every committed file below
the repository's 1 MB hook. The scripts read compressed files directly. To inspect
one manually, use `gzip -dc results/vessel.verified.json.gz`.

## Reproduce analysis and figure

From this directory:

```bash
python3 analyze.py --self-test
python3 analyze.py
uv run --no-project --with numpy==1.26.4 --with matplotlib python plot_results.py
```

`analyze.py` regenerates `results/REPORT.md` and `results/summary.json.gz`.

## GPU execution

The Modal account needs the `hf-token` secret with access to Gemma-2-2b:

```bash
modal run modal_runner.py --mode check
modal run --detach modal_runner.py --mode run --concept all
modal run --detach modal_runner.py --mode verify --concept all
```

The named Modal results volume retains raw checkpoints; completed jobs are reused.
Use a new volume name for an independent rerun. The local harness saves compressed
JSON. `run_single_feature.py` and `verify_results.py` preserve the measurement
logic used for the reported run.
