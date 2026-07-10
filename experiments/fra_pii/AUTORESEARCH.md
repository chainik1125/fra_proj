# fra_pii — autoresearch flow: max SSN-emit suppression at min collateral

## Objective
Suppress emission of ONE target in-context SSN as much as possible while preserving:
- **lookup** — the target SSN still works as a retrieval KEY (ssn→name),
- **sibling PII** — a DIFFERENT record's SSN still emits (disarm Alice, keep Bob),
- **general behaviour** — low KL on unrelated next-token text.

Trace the **suppression-vs-collateral frontier** for each method by sweeping intervention size
("cut more cells" / "steer more features") — the control the first cut run lacked (it stopped at
top-20 cells and nothing moved).

## Task (simplest, fixed)
`pii_sweep.py` — ONE fixed 3-record DB (deterministic so runs compare like-for-like), target =
record[0]. gemma-2-2b base + GemmaScope 16k. Ground-truth metrics only. Runs ≤5 min; writes
`pii_sweep_<TAG>.json` incrementally so partial frontiers are readable mid-run.

## Methods (one per pod; METHOD env)
- **diag** — controls: per-head causal ranking (which heads carry emit), position-aware ORACLE
  (upper bound on suppression), FRA-cut-ALL. Answers: does ANYTHING suppress, and is the emit
  edge answer→digit attention? (If oracle fails → wrong edge; if oracle works but FRA fails →
  content/reach failure = the theory-consistent negative.)
- **fra** — sweep FRA cell cut: M∈{50,200,1000,ALL} × c∈{8,20,45} × head-set. Content-addressed.
- **sae** — sweep single-feature steering: k∈{1,3,10,30,100} features × layer × strength. k=1 is
  the literal single-feature baseline FRA is compared against.

## Launch
```
POD_NAME=rs-pii-<tag> METHOD=<diag|fra|sae> SWEEP_TAG=<tag> \
  RUNPOD_API_KEY=$RP_API_KEY_MATS HF_TOKEN=$HF_TOKEN \
  bash experiments/fra_pii/code/launch_pod_pii_sweep.sh
```
Results: HF `fra_pii/results/pii_sweep_<TAG>.json` + `rs-pii-<tag>_run.log`.

## Flow
- **Round 1 (orchestrator-driven, parallel pods):** diag + fra + sae in parallel. Validate the
  primitive; get first frontiers; establish the oracle upper bound + whether emit is edge-routed.
- **Round 2 (per-method background agents):** one agent per promising method pushes its frontier
  (refine head-set / M / layer / k / strength / edge definition) with ≤5-min pods, ≤~5 rounds,
  reports its best (suppression, lookup_ok, sib_ok, gen_kl) operating point. Orchestrator compares
  frontiers at MATCHED suppression — the campaign's standard win metric.

## Gotchas (inherited)
- Pods named `rs-pii-*` (reaper kills non-rs-*); pre-check RUNNING names before launch.
- Score-delta hooks must be KV-cache-safe (slice to `s.shape[2/3]`); generate with full forward
  passes (no cache) — see `gen_cont`.
- Digit-aware emit match (`_dig`): gemma splits SSNs digit-by-digit and reformats dashes.
- FRA build (`top_k=None`) is the cost; build ONCE per prompt, sweep M/c on the cached tensor.
