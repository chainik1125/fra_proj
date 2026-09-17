# Semantic filtering with the corrected restoration metric

> **Normalization audit notice (2026-09-16):** This archived run used
> `normalize_activations=True` in the inherited Gemma Scope wrapper. The
> [Gemma Scope paper, §3.1](https://storage.googleapis.com/gemma-scope/gemma-scope-report.pdf)
> states that released weights already absorb the fixed training normalization;
> extra per-token normalization is not required. These numbers reproduce the
> archived implementation, but the native-SAE comparison needs to be rerun before
> treating its relative performance as settled. The compound semantic experiment
> in `experiments/fra_compound_semantic_20260916/` corrects this for both methods
> and records a reconstruction audit. Original data and code are retained.

This reruns the single-feature baseline on Indranil's six-concept semantic-filter
benchmark with the poisoned context present throughout steering. The reference is
the same context with the planted payload omitted, followed by the same legitimate
paragraph. KL compares full next-token distributions at aligned continuation tokens.

**Result at >=50% original payload suppression:** the best per-query positive
single-feature SAE baseline has lower continuation KL on all 52 related-word
queries: mean **0.02819**, versus **0.11823 nats/token** for fresh FRA. One fixed
feature and coefficient per concept also reaches all 52 queries and has lower
continuation KL than FRA in all six concepts.

No corrected restoration KL is zero. Five of the six fixed SAE edits leave
paragraph KL equal to the poisoned no-edit baseline; vehicle increases it. Much
of the advantage therefore comes from avoiding additional distortion. The same
fixed edits improve recall-distribution KL in all six concepts. SAE has lower
recall KL in four; FRA has lower recall KL in fire and war. See the report for
the two objectives and their independently optimized comparisons.
The fixed-edit comparison changes at higher suppression thresholds; the full
curves and report retain those results and any unreached queries.

- [Numerical report](results/REPORT.md)
- [Exact evaluation protocol](PROTOCOL.md)
- [Single fixed-edit tradeoffs](results/paired_restoration.png)

All 52 original related-word queries and 18 planted-word queries are retained.
The SAE candidates are the original top ten planted-minus-no-payload differential
features. Sweep each individually, select the best measured point, and distinguish
positive activation removal, signed steering, and additive steering.

FRA is recomputed in the same environment with the exact 25 original heads and
48 QK pairs per head, located once on the planted word. The original FRA curves
are reproduced before the corrected evaluation. The report gives both one fixed
feature/coefficient per concept and retrospective per-query choices.

The earlier independent-paragraph zero KL did not establish restoration under
a poisoned context. This experiment also reevaluates old-metric optimal edits
unchanged, with the tie-breaking rule documented in the protocol.

## Reproduce the report and figure

```bash
python3 analyze.py
uv run --no-project --with matplotlib python plot_results.py
```

The losslessly compressed `results/{concept}.partNN.json.gz` files retain every
grid point, per-token KL, input token IDs, selected FRA pairs, and verification.
Each shard has a `point_offset` and `total_points`; analysis checks complete coverage.
Every file is below 1 MB. Checksums are in `results/ARCHIVE_MANIFEST.md`.

## GPU execution

Use Modal with the `hf-token` secret and access to Gemma-2-2b:

```bash
modal run --detach modal_runner.py --concept vessel --smoke
modal run --detach modal_runner.py
```

The runner caps simultaneous A100-80GB workers at two, checkpoints results to a
named volume, and reattaches to existing calls after coordinator restarts.
Completed results are reused; use a new results-volume name for an independent run.
