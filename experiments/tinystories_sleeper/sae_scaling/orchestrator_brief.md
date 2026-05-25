# SAE-scaling-sweep orchestrator brief

You are the orchestrator for a multi-pod experimental campaign on RunPod.
Your job is to **drive the campaign to completion** — launch the per-seed
worker pods, react to failures, relaunch as needed, and produce the final
artifact (`summary.md` on HF). You run headless inside a RunPod pod yourself,
optionally with one **monitor teammate agent per seed** (spawn via the Agent
tool; coordinate via SendMessage + the HF `status/` files).

## Goal

Produce `summary.md` on HF dataset `dmanningcoe/sae-scaling-tinystories-sleeper`
summarising the SAE dict-size × k × training-step scaling sweep on the
TinyStories-33M sleeper: per-(d_sae, k, hookpoint, seed, step) SAE-quality
metrics (FVU, dead-feature frac, loss_recovered, firing-rate dist) and the
two-curve steering result (J_clean(α), J_pois(α), ASR) with the winner feature
re-derived per checkpoint via the Wang diff screen.

## Experiment grid (the done predicate)

```
hookpoint ∈ {ln1, resid_mid}             (2)
seed      ∈ {0, 1, 2}                     (3)
d_sae     ∈ {3072, 6144, 12288, 24576}   (4)
k         ∈ {10, 32, 50}                  (3)
step      ∈ {10000,20000,30000,40000,50000} (5)
```

**DONE = all 360 result JSONs present on HF**:
`results/{hookpoint}/seed{seed}/d{d_sae}_k{k}/step{step}.json`
(2×3×4×3×5 = 360). Per seed that's 120 results (60 ln1 + 60 resid_mid).

## Architecture

```
[you, orchestrator pod]  launch 3 seed workers, poll HF, relaunch, summarise
        │  (optional) 1 monitor teammate agent per seed
        ▼
[seed-0 pod] [seed-1 pod] [seed-2 pod]   each runs auto_start_gpu.sh:
    env fix → producer(ln1) ‖ producer(resid_mid) ‖ eval_poll consumer
    → streams 120 results to HF → self-stops (SELF_STOP=1)
        │
        ▼  HF dataset = coordination bus
   status/train_{hook}_seed{S}.json   status/eval_seed{S}.json
   sae_checkpoints/.../*.pt           results/.../*.json
```

Each seed pod is self-contained: it trains the full 12-config grid for both
hookpoints (checkpoints every 10k steps) and evaluates them. One L40S per seed
(~7 hr, ~$6.50). The 32× width (d_sae=24576) is ~half the per-pod time.

## How to launch a worker pod

Launch via `dockerArgs` so the pod self-starts and self-stops (no SSH needed).
**Mandatory:** `User-Agent: curl/8.0` header (Cloudflare 1010 otherwise), and
field is `minVcpuCount` (not `vcpuCount`). GPU: `NVIDIA L40S` ($0.86/hr; CPU
pods are SUPPLY_CONSTRAINTed on this account). Pass `RUNPOD_API_KEY` +
`SELF_STOP=1` so the worker stops itself on completion (`$RUNPOD_POD_ID` is
auto-injected by RunPod).

`dockerArgs` for seed `$S`:
```
bash -c "apt-get update -qq && apt-get install -y -q git curl >/dev/null 2>&1 && \
  git clone --branch dmitry/sae-scaling-sweep --single-branch \
    https://github.com/chainik1125/fra_proj.git /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  SEED=$S HF_TOKEN='$HF_TOKEN' HF_REPO='$HF_REPO' \
  RUNPOD_API_KEY='$RUNPOD_API_KEY' SELF_STOP=1 \
  bash experiments/tinystories_sleeper/sae_scaling/auto_start_gpu.sh \
    >> /workspace/worker.log 2>&1"
```
Image: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`.
`auto_start_gpu.sh` already bakes in the env fix (see below).

## Prior debug findings (don't rediscover these)

1. **torch cu-version lottery.** `pip install transformer-lens` clobbers the
   image torch with cu130 (driver too old → `cuda False`). `auto_start_gpu.sh`
   force-reinstalls **torch 2.6.0+cu124** (2.4.1 is too old for current
   `transformers`, which imports `torch.distributed.tensor.device_mesh`), then
   installs `nvidia-cusparselt-cu12` and registers it via ldconfig (torch 2.6
   needs `libcusparseLt.so.0`). If a pod logs `cuda False` or `libcusparseLt`
   or `device_mesh` import errors, the env fix didn't run — check worker.log.
2. **Overriding `dockerArgs` bypasses the image `/start.sh` → no sshd.** Worker
   pods launched as above have **no SSH** — that's fine, they report via HF.
   If you need to SSH-debug, launch a pod with NO dockerArgs, then SSH in and
   run `auto_start_gpu.sh` manually (the image starts sshd in that case).
3. **Cloudflare 1010** on default UA → always send `User-Agent: curl/8.0`.
4. **`minVcpuCount`** not `vcpuCount`; introspection disabled, schema errors
   surface as `GRAPHQL_VALIDATION_FAILED`.
5. **`python -u`** everywhere (auto_start_gpu.sh already does) or logs look empty.

## Monitoring (poll HF, no SSH on workers)

- Progress per seed: `status/train_{hook}_seed{S}.json` (`n_done/n_total` of 60
  checkpoints per hookpoint) and `status/eval_seed{S}.json` (`n_done/n_total` of
  120). Also count `results/.../seed{S}/...json` directly.
- A monitor teammate should poll its seed's status every ~2–5 min and report
  `{trained}/120` + `{evaled}/120` upward; flag any of: status file missing
  >15 min after launch (env/bootstrap failure), `n_done` not advancing for
  >20 min (stall), or a worker pod gone with <120 results (crashed).

## Troubleshooting playbook

- **Pod never writes a status file** (>15 min): env/bootstrap failed. Terminate,
  relaunch. If it recurs, launch a no-dockerArgs debug pod, SSH in, run
  `auto_start_gpu.sh` by hand, read the error (likely an env-fix regression).
- **Stall** (`n_done` flat >20 min while pod alive): likely an eval exception on
  one checkpoint (eval_poll logs `FAILED <ckpt>` and continues) or OOM on the
  32× width. Relaunch the pod — see idempotency below.
- **Idempotency on relaunch:** `eval_poll` skips checkpoints whose result JSON
  already exists, so eval RESUMES cleanly. **Training is NOT
  progress-checkpointed** — a relaunched producer re-trains from config 1 and
  re-uploads checkpoints (correct, but wasteful). Acceptable for recovery; if a
  pod dies late, consider letting eval catch up on already-uploaded checkpoints
  before relaunching the producer.
- **Degenerate winner** (k=10 × d_sae=24576 may have many dead features): not a
  failure — the metrics (`dead_feature_frac`) capture it. Don't relaunch.
- **Cost discipline:** L40S = $0.86/hr. Terminate pods you've determined won't
  recover; don't leave idle pods. Cap total wall at `MAX_RUN_SEC` (default
  ~12 h) — if hit, write `status/giveup.json` listing what's missing and stop.

## Environment variables you have

- `HF_TOKEN` (write), `RUNPOD_API_KEY`, `ANTHROPIC_API_KEY` (your own auth),
  `RUNPOD_POD_ID` (yours, for self-terminate), `HF_REPO`, `SEEDS` (default `0 1 2`).
- **Never** commit or upload secrets.

## When all 360 results are present

1. Download the 360 `results/.../*.json`.
2. Build `summary.md`: for each (hookpoint, seed) a table of metrics vs (d_sae,
   k, step), and the steering J_clean/J_pois/ASR at the winner's α-sweep;
   aggregate mean±std across seeds. (Plotting can be done locally from the JSONs;
   the summary should at minimum tabulate FVU, dead_feature_frac, loss_recovered,
   and the J_clean/J_pois/ASR at the most-suppressing α per cell.)
3. Upload `summary.md` to HF, then `podTerminate` yourself and exit.
