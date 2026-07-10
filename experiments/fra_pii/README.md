# fra_pii — FRA vs SAE on in-context PII (SSN) recall

**One-line result.** On realistic in-context PII recall (a JSON "records DB", disarm one record's
SSN emission while preserving its use as a lookup key), **FRA is content-selective but reach-capped
(best 0.84, never behaviorally disarms); a single position-locked SAE feature is the only method that
clears the behavioral bar; and the per-head coupling is ~rank-1 (B3 condensate confirmed on gemma).**
The exact inverse of the box→frog retrieval win — content-routed distributed recall is SAE's turf.

**Read `RESULTS.md` for the verdict + scorecard; `THEORY_NOTE.md` for the mechanistic account.**

## The series (chronological pipeline)
| phase | script | pod(s) | finding |
|-------|--------|--------|---------|
| F1 emit feasibility | `code/pii_feas.py` | rs-pii-feas | SSN emit exact-match 1.00 |
| F2 compute feasibility | `code/pii_compute.py` | rs-pii-compute | parity DEAD (digit-sens 0.00); verify weak 0.77 → pivot to lookup |
| F3 lookup feasibility | `code/pii_lookup_feas.py` | rs-pii-lookup | reverse lookup (ssn→name) 1.00 → GO |
| Stage-2 single-edge cut | `code/pii_cut.py` | rs-pii-cut{,2,5} | FRA emit-cell cut = NO-OP (~8%, c hit ceiling) |
| autoresearch sweep | `code/pii_sweep.py` | rs-pii-{diag,frasw,saesw} | diag: oracle also fails (name/brace 0.000, digit ≤0.126); SAE suppresses 0.96 |
| r2 head re-locate | `code/pii_sweep_fra2.py` | rs-pii-fra-2 | 208-head scan: name/brace oracle 0.000 |
| r3 K-scan | `code/pii_sweep_fra3.py` | rs-pii-fra-3 | redundant bank: all-208-head oracle 0.97; no selective K |
| r4 all-positions cut | `code/pii_sweep_fra4.py` | rs-pii-fra-4 | position-invariant content-cut caps 0.65, emit_ok=True |
| r5 SVD / B3 condensate | `code/pii_sweep_fra5.py` | rs-pii-fra-5 | ω rank-1 (cumvar@r=1 0.92–0.998); rank-1 edit 0.84 best, still no disarm |

## Layout
- `code/` — all harnesses + launchers. `pii_sweep.py` is the parametrized primitive (METHOD=diag/fra/sae);
  `pii_sweep_fra{2..5}.py` are the FRA agent's per-round variants. `launch_pod_pii*.sh` = RunPod launchers
  (`launch_pod_piicut.sh` is a redundant naming-variant of `launch_pod_pii_cut.sh`).
- `results/` — all run artifacts pulled from HF (`fra_pii/results/`): `pii_*_results.json`,
  `pii_sweep_*.json` (per-method/per-round frontiers), `rs-pii-*_run.log`, tracebacks.
- `RESULTS.md` — canonical verdict + scorecard. `THEORY_NOTE.md` — mechanistic account (win-checklist,
  FRA⊇attention-map ablation, three-tier gate, condensate). `AUTORESEARCH.md` — the agent-flow protocol.

## Compute / provenance
gemma-2-2b base + GemmaScope 16k; FRA via the `fra_win/fra_bundle.tar.gz` package (PYTHONPATH). Ground-truth
metrics only (P(first digit) + greedy emit-match), no LLM judge. Ran as a background-agent autoresearch flow
(FRA / SAE / theory agents, ≤5-min RunPod pods, HF prefix `fra_pii/`). Origin session 172ac940.

## Related / precursor
- `../fra_win/jobs/pii_sibling.py` (+ `out/pii_sibling.log`) — the earlier **entity-PII sibling** no-op
  ("entity-PII no-op at strong base") that motivated the SSN framing. Left in fra_win (its origin); not moved.
- Campaign context: [[STOCKTAKE]] T1 (redundant head bank), F5 (ranking⊥causal), F7/T4 (B3 mean-field —
  reopened by the rank-1 condensate here), and the box→frog win in `fra_win/`.

## Open (pending, not blocking the verdict)
SAE cross-record selectivity: the sib metric had a bug (ablated the sibling's own digits); fixed in
`code/pii_sweep.py` but not re-run. One ~5-min pod would confirm whether SAE's disarm is content-selective
across records or merely positional.
