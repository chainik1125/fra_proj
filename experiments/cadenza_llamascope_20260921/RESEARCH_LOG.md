# Research log

All timestamps are Pacific daylight time on September 21, 2026, unless stated otherwise.

## 21:48 — kickoff

User authorized an immediate overnight sprint ending September 22 at 09:00,
with at most eight simultaneous Simplex GPUs. Experiments stop at 08:00 to
reserve the final hour for writing. Confirmed zero-based attention layers
8/12/16 map to Llama Scope residual-post layers 7/11/15.

Created branch `dmitry/cadenza-llamascope-overnight-20260921` while preserving
the existing working tree. Read the consolidated September 21 steering report
and audited result JSONs. The primary replication trains a layer-12 input SAE
with the existing 100M-token configuration. The official residual SAEs cross
from Llama 3.1 Base to the Dolphin/Llama 3.0 sleeper, so their comparison is a
transfer experiment.

## 21:51 — training

Started `simplex1:/data/users/dmitry/sae-middle/runs/A-input-L12-100M-20260921`
on GPU 0. Configuration differs from the existing input runs only in layer.
Training uses repeated training examples to reach exactly 100M valid tokens.
The fresh steering confirmation set is excluded from SAE training.

## 22:09 — durable queue

Started detached controller PID 1222558 under
`/data/users/dmitry/sae-middle/campaigns/A-scope-overnight-20260921`.
Confirmed survival after SSH disconnection. Main queue has seven slots on
simplex1 (0,1,2,3,4,6,7), including training; no allocations on simplex2/3.
The eighth slot is reserved for bounded correctness checks when available.

Queue: six official transfer diagnostics, eighteen official top-50 searches,
three local layer-12 searches, six frozen existing SAE/FRA comparisons and ten
frozen DoM comparisons. Every comparison uses the same third 64-pair test block.
The original test and previously inspected confirmation are labeled separately.
Positive-only and signed selections stay separate and freeze before testing.

Remote preflight: 59 tests passed (54 inherited pipeline tests, five transport
and checkpoint-conversion tests). Real-model checks verify residual/attention
pairing, full-token RMS scale and learned gain. Frozen previous SAE/FRA runs
have completed on the fresh block.

## 22:11 — numerical diagnostic retry

First 128K diagnostic attempts failed an error-inclusive decomposition assertion:
FP32 cancellation exceeded absolute tolerance near zero because the SAE
reconstruction was exceptionally large. Recomputed only this algebraic identity
in FP64; retained exact native RMSNorm and separate feature-transport checks.
All three retried diagnostics passed. Failed attempt artifacts are preserved.
This is not a relaxation of the measured reconstruction-quality criteria.

## 22:17–22:21 — independent loader audit

Used the eighth free GPU (simplex1 GPU 5) briefly, with the existing per-user
GPU lock. Loaded real activations from two training examples per class and
executed the original published conversion functions extracted from pinned
OpenMOSS source `b932639261697c0642d077c05f2cb7e463d058cb`.
All six parameter/output comparisons passed numerical tolerance. A first
bitwise parameter assertion detected only sub-1e-7 differences between two
equivalent norm APIs; replaced it with explicit 1e-7 absolute / 1e-6 relative
parameter tolerance and retained output comparisons.

The SAELens 6.44 converter uses a different threshold convention. Computing
its alternative reference did not resolve the huge reconstruction errors.
For the four-example diagnostic, more than 99.99% of squared reconstruction
error is concentrated in the largest 1% of token errors. Median residual
norms are approximately 4.7–8.0; maximum norms are approximately 656–658.
Thus aggregate FVU alone is a poor description of typical-token behavior.
Added explicitly post-hoc token diagnostics; primary steering masks and
selection rules remain unchanged. No fresh-test data were used in this audit.

Training had passed 40M tokens with finite losses at 22:15. Frozen baseline
code was copied into `baseline_lib/` for reproducibility, leaving original
working files untouched. Per-run immutable source copies and SHA256 manifests
identify exactly which worker version produced each artifact.

## Artifact locations

- Main design: `../cadenza_mid_sae/LLAMASCOPE_OVERNIGHT_DESIGN_20260921.md`.
- Queue and launch: `plan.json`, `build_plan.py`, `deploy.py`, `controller.py`.
- Checkpoint conversion and transport: `scope.py`, `test_scope.py`.
- Independent reference check: `audit_loader.py`, `native_reference.py`.
- Full remote logs, checkpoints, prompts and generations remain on simplex1.
- Morning deliverable: `summary.md`, with compact audited results and plots.

## September 22, 01:19 — remote experiments completed

All 43 queued tasks completed successfully. Three 128K diagnostic first attempts
were retained as failures, followed by successful retries. Training completed
100M tokens, with all ten intermediate checkpoints and final reload verified.
The queue remained open for additions, so the idle controller continued its
hourly audits at zero allocations until exiting at 08:00. No workers survived.
Recorded worker wall time totals 17.413 GPU-hours, plus 0.974 for training and
brief preflight/independent loader checks (approximately 18.5 GPU-hours total).

## September 22, 10:40 onward — collection and report

The local collection command required sandbox SSH approval and returned after
the report deadline. Remote work proceeded independently and finished at 01:19.
The promised 09:00 write-up was missed. Future kickoff should exercise and
authorize the full collection path and schedule remote draft-report generation.

Collected compact results with zero audit failures: source hashes, frozen
selection hashes, identical/disjoint fresh prompt keys, 750 records per search,
50 unique candidates, and reproduced validation choices. Independently
recomputed key metric means from saved per-prompt measurements. Added paired
20,000-draw bootstrap comparisons, CSV/Markdown tables and two exportable figures.
Local system matplotlib had a NumPy ABI mismatch; used an isolated `uv run`
environment to generate the plots. Both plots were visually checked, including
correcting a clipped axis label.

Main findings: official RP7 32K → attention8 OV-FRA improves over the signed
residual single-feature baseline while preserving all clean outputs. New local
L12 OV-FRA improves over the positive-only single-feature rule, while its smaller
advantage over the signed rule has an interval crossing zero. Local L8 OV
replicates and is not distinguishable from official 32K OV on JSD here. The full
grid and poor transfer quality prevent a broad width/general-layer claim.

The extreme activation audit localized the largest norms to `<|im_start|>`.
This marker remains included under the inherited BOS/EOS/PAD mask. A post-hoc
norm filter lowers diagnostic FVU to 0.66–0.73 on the tiny audit sample; the
primary protocol is unchanged. Perturbation norms and a chat-marker masking
ablation were not collected; the report states both limitations explicitly.

## September 22, 11:05–11:21 — literal residual-post layer 8 follow-up

User requested single-feature steering at literal resid_post 8. Ran official
32K and 128K SAEs at that site, retaining the same model, top-50 ranking,
signed grid, splits and generation measurements. Split candidates across four
GPUs per width (eight total), merged the complete 750-setting grid per width,
froze choices and evaluated the original/legacy/common confirmation sets.

All ten jobs completed without retries in 15.24 minutes of campaign wall time,
using 1.688 GPU-hours. Controller and worker exit verified. Three CPU tests
validate shard-merge equivalence, tied optima, duplicate rejection and baseline
consistency. A review added explicit sorting to preserve the original positive
rule's first-occurrence tie-breaking; the running coordinator retained its launch
snapshot, and independent canonical-order audits verified that its actual choices
exactly matched the corrected unsharded order for both widths before acceptance.
Source hashes record the launch and subsequent audit versions independently.

Results on the reused common 64-pair block: 32K positive JSD 0.942417
(feature 19809, alpha 2), 32K signed 0.894866 (feature 2083, alpha -4),
128K positive/signed 0.948681 (feature 111213, alpha 2). Every selected setting
removed IHY on 64/64 triggered outputs but preserved 0/64 clean outputs.
All four paired differences versus the corresponding RP7 rule have intervals
crossing zero. This is an exploratory follow-up because the common confirmation
block was already inspected in the overnight report.

Report and compact results: `resid_post_8/summary.md`, `resid_post_8/results.json`.
Remote campaign: `/data/users/dmitry/sae-middle/campaigns/A-scope-residpost8-20260922`.
