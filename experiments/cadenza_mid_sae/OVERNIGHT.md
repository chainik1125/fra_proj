# Cadenza attention-input SAE overnight run

**Update, Sep 21 morning:** all four reached 100M tokens and retained ten verified
exports each. The campaign stopped at its fixed quality gate after 1.57 hours;
no automatic comparison ran. The user subsequently authorized comparisons on
8/16/24 despite those failures, and requested simplex2. See
[STEERING_RUNS.md](STEERING_RUNS.md) for the new standalone comparison jobs.

## Approved scope

Train four 8x TopK SAEs at zero-based layers **0/8/16/24**, on variant A,
`blocks.L.ln1.hook_normalized`, **100M valid activation tokens each**. Retain exports
every 10M tokens. Train/evaluate 50/50 Cadenza by examples. Keep all data, model
weights, activations, checkpoints and ML environments remote; no new local file
may exceed 10MB. Use up to four free Simplex GPUs without interfering with others.

After all four pass the fixed SAE-quality gates, compare single-input-feature
steering with FRA OV-only and channel-routed QK+OV on the first four. No output
SAEs or layers 4/12/20/28. This resolves the older asynchronous “add four” reply:
the user's final confirmation was **four input SAEs only**.

## Execution budget and handoff

- Branch: `dmitry/cadenza-sae-overnight-20260921` (sprint skill: dedicated branch,
  bounded remote supervisor, hourly audit, durable outcome-first morning report).
- Host: simplex1, intended GPU indices 0/1/2/3, H200 141GB each.
- Campaign: `A-input4-100M-20260921`, at `/data/users/dmitry/sae-middle/campaigns/`.
- Budget: max four GPUs simultaneously, max eight hours from campaign launch;
  max two attempts/task. No rented compute, API charges, or host shutdown.
- Failed attempts remain for diagnosis. Exact-stream optimizer resume is not
  implemented; one retry starts from scratch, within the same wall-clock budget.
- Morning deliverable: remote `summary.md`, per-layer metrics/gate decisions,
  checkpoint paths, and (if gates passed) held-out steering results with uncertainty.

## Preflight evidence

- `A-L16-input-smoke-20260921`: complete, 100,000 activations, input hook verified,
  166 sleeper + 166 clean harvested examples. Both 50k/100k exports reload-tested;
  final max absolute reconstruction mismatch 3.82e-6. Peak GPU allocation 20.94GiB.
- The smoke is a plumbing test, not a converged SAE: clean FVU 0.6903 and clean
  CE increase 0.0531 nats; it is not accepted as a downstream scientific checkpoint.
- `A-L16-fra-smoke-20260921`: all 21 unit tests and the small real-model three-arm
  steering pipeline completed. It used two pairs/split, four generated tokens and
  an unconverged smoke SAE, so it supplies no comparison-quality evidence.
- Follow-up smoke `A-L16-fra-smoke-20260921-v2` completed, with nonzero selected
  QK+OV triplet score (0.01053). It tests Q-shortlist selection by mean activity,
  necessary because layer-0 paired final-query differences vanish.
- Final local suite: 22 tests, 16 passed and 6 remote-only ML tests skipped.
  A simulated full controller run verifies both pass→four comparisons and
  failure→no comparisons. Each actual worker reruns all 22 tests remotely.

## Fixed criteria and comparison protocol

The exact gates and methodology are in README.md, `campaign.py:GATES`, and
`steering.py:DEFAULT_PROTOCOL`. Gates are fixed before the 100M runs: FVU ≤0.5
per class; L0 50±2%; dead fraction ≤20% over the last 10M tokens; CE increase ≤0.1
nats/class, with ≥80% ablation-loss recovery when CE increase >0.02; complete
token accounting and every checkpoint reload verified. No post-hoc gate relaxation.

Use 64 train-pool pairs for feature selection, 24 held-out pairs for strength
tuning, 64 separate held-out pairs for final testing. All three methods patch
prompt positions only. Report the QK+OV triplet's greater feature count explicitly.
Zero-strength baseline is eligible, so the pipeline can honestly report no useful
steer. Attribution incorporates native Llama learned norm gains, GQA and RoPE.

## Status commands

```sh
python3 -B experiments/cadenza_mid_sae/launch.py campaign-status --run-id A-input4-100M-20260921
python3 -B experiments/cadenza_mid_sae/launch.py campaign-summary --run-id A-input4-100M-20260921
python3 -B experiments/cadenza_mid_sae/launch.py campaign-logs --run-id A-input4-100M-20260921
```

These fetch bounded text only; never copy a checkpoint to the laptop.

## Launch record

Launched **2026-09-21 06:51:51 UTC** (Sep 20, 23:51:51 Pacific) on simplex1.
Supervisor PID **868677**, verified alive after SSH disconnected. Eight-hour hard
deadline is **2026-09-21 14:51:51 UTC** (Sep 21, 07:51:51 Pacific).

| Layer | GPU | Worker PID | Run ID |
|---|---:|---:|---|
| 0 | 0 | 868680 | A-input4-100M-20260921-L00-train-a1 |
| 8 | 1 | 868688 | A-input4-100M-20260921-L08-train-a1 |
| 16 | 2 | 868694 | A-input4-100M-20260921-L16-train-a1 |
| 24 | 3 | 868708 | A-input4-100M-20260921-L24-train-a1 |

All four comparison tasks are pending behind the collective quality gate; no
additional SAE training task exists in the queue. Initial supervisor audit
records active GPU slots [0,1,2,3]. Consult the remote status for live progress.

Initial live-training verification (~one minute after launch): all four workers
alive, all four CUDA processes present, finite losses, and actual token progress:
L0 3.072M, L8 1.2288M, L16 0.7168M, L24 0.512M. Peak allocated GPU memory reported
22.72GiB per worker. These early annealing-phase losses are not final quality
metrics. Ten-million-token checkpoints and the collective gate are still pending.
