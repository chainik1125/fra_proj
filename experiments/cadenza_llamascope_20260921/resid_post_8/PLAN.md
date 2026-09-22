# Literal residual-post layer 8 follow-up

User request: “try resid post at layer 8.” This tests direct single-feature
residual steering at zero-based layer 8, with official Llama Scope 32K and 128K
SAEs. It adds no FRA search. The preceding overnight cells used residual-post
7/11/15 to match attention 8/12/16; this is a distinct residual site.

Reuse the frozen overnight model, checkpoint loader, candidate ranking,
64 selection pairs, 24 validation pairs, signed strength grid, generation and
measurement code. Rank all features, retain 50, evaluate 15 strengths per
feature, and freeze positive-only and signed minimum-validation-JSD choices.
Evaluate the original test, legacy confirmation and common third 64-pair block.
The common block was inspected in the overnight report, so this follow-up is
exploratory; it is retained for direct comparisons rather than described as new.

Four disjoint candidate shards per width shorten wall time while retaining the
full search. Verify identical rankings, splits and unsteered baselines across
shards; reject duplicate/missing settings. Independently reproduce choices in
the original unsharded candidate order before accepting final results. Three
CPU tests cover merge equivalence, tied optima, duplicate records and inconsistent
baselines. Source snapshots and hashes are retained per run.

At most eight GPUs on simplex1, with no new allocations on simplex2/3. Two-hour
runtime ceiling, per-GPU idle checks and locks, at most two attempts per job,
automatic final evaluation and remote report generation. Large checkpoint and
generation artifacts remain remote. Main campaign path:
`/data/users/dmitry/sae-middle/campaigns/A-scope-residpost8-20260922`.

Launch/status/collection commands from the project root:

```sh
python3 -B experiments/cadenza_llamascope_20260921/rp8.py launch
python3 -B experiments/cadenza_llamascope_20260921/rp8.py status
python3 -B experiments/cadenza_llamascope_20260921/rp8.py collect
```

`launch` refuses to overwrite an existing campaign. No checkpoint or report
from the overnight investigation is modified.
