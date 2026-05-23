# Troubleshooter brief — wang_steering_7b campaign

You are a headless Claude Code agent running on a RunPod pod with full
`--dangerously-skip-permissions`. Your job is to drive the
`wang_steering_7b` campaign to completion. The user is away; you self-stop
when done.

## Goal

Land all 6 of these files on Hugging Face:

```
dmanningcoe/fra-phase1-steering-data : qwen7b/wang_L15_resid_post/{em}_seed{seed}/qualitative_arditi_{em}_evalseed{seed}.json
  where em ∈ {medical, base},  seed ∈ {42, 123, 456}
```

When all 6 exist, write `qwen7b/wang_L15_resid_post/summary.md` to HF, then
self-terminate (`podTerminate` your own pod via GraphQL using `$RUNPOD_POD_ID`).

## State at hand-off (verify, may have changed)

- Branch: `autoresearch/wang-steering-7b` (already pushed to origin).
- Per-shard bootstrap: `experiments/wang_steering_7b/auto_start_gpu.sh`.
  Reads env `EM_MODEL`, `EVAL_SEED`, `TOP_N`, `HF_TOKEN`, `RUNPOD_API_KEY`,
  `RUNPOD_POD_ID`. Driver fast-fail at `<575`. Pre-check skip-if-on-HF.
  Then: clone, pip install, run `scripts/compute_wang_feature_ranking.py`,
  run `phase1_arditi_orchestrator.py`, upload, `podTerminate` self.
- 6 GPU pods were dispatched on L40S; 3 (`medical-s42`, `base-s123`,
  `base-s456`) vanished within ~minutes — likely the bootstrap reached
  `podTerminate` via an early-exit path (probable culprit: racy `nvidia-smi`
  on cold boot makes `$DRIVER_MAJOR` empty → driver-too-old branch fires
  → self-terminate). The 3 survivors auto-restarted; investigate them
  before redispatching.
- The previous passive babysitter (which only polled HF) has been killed.
  You are its replacement.

## Tools available on this pod

- `bash`, `git`, `python3`, `curl`, `ssh`, `openssh-client`
- `RUNPOD_API_KEY` env var — use it for GraphQL pod mgmt
- `HF_TOKEN` env var — upload/list on HF
- `ANTHROPIC_API_KEY` env var — that's how you're running
- `~/.ssh/id_ed25519` private key matching the public key registered on
  RunPod (you can SSH into any pod that has SSH up)

## What to do (procedure, not prescription — judge as you go)

1. **Inventory** — `curl` the RunPod GraphQL `myself.pods` query, filter to
   `wang-steering-*`, note `desiredStatus` + `runtime.uptimeInSeconds` per
   pod. Cross-reference HF: which shards exist already?

2. **Diagnose** — for each alive pod that has SSH up, `ssh root@... tail -200
   /workspace/run.log`. The early-termination hypothesis is the racy
   `nvidia-smi` driver check. If you confirm that pattern, patch
   `auto_start_gpu.sh` to retry `nvidia-smi` 3× with a 5 s sleep before
   declaring the driver missing, push the patch, redispatch the affected
   shards via `experiments/wang_steering_7b/launch_all.sh` (or inline curl
   if launch_all has bash-3.2 issues — the pod has bash 5).

3. **Redispatch** — for any missing shard, launch a fresh L40S/L40/A40/
   RTX A6000 pod with the right `EM_MODEL` / `EVAL_SEED` env. **Do not
   re-launch a shard whose target file is already on HF.** Cap total cost
   at $20 — query `podRentInterruptablePrices` if unsure.

4. **Monitor** — loop: list HF every 60-120 s; SSH-tail any pod whose log
   hasn't progressed in 5 min; redispatch if a pod has fully died.

5. **Stop condition** — once all 6 files are on HF:
   - Read each file's `len(json.load())` as a sanity check (expect
     `n_features × n_alphas × n_prompts ≈ 50 × 17 × 8 = 6 800` entries).
   - Write `summary.md` to HF under `qwen7b/wang_L15_resid_post/summary.md`
     listing the 6 paths + their sizes + the ranker's top-5 features (read
     from `wang_ranker_L15_top50.json` on HF if present).
   - `podTerminate` your own pod (`$RUNPOD_POD_ID`).
   - **Don't run the GPT-4o judge step** — that's queued for the user to
     run locally to control OpenAI spend.

## Sharp edges to know

- **RunPod GraphQL CPU API quirk** — `computeType: CPU` deploys 500. Use a
  cheap GPU (RTX A5000 / A4000) for any helper pods you need to spawn.
- **`dockerArgs` quoting** — see `experiments/wang_steering_7b/launch_all.sh`
  `make_gpu_cmd` for the working pattern. Don't manually escape; use
  `python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"`.
- **Logs may be buffered** — every bash bootstrap should use
  `exec > >(stdbuf -oL tee /workspace/run.log) 2>&1` and every python should
  use `python3 -u`. If you see a pod at 99% GPU but no log progress, suspect
  buffering rather than a hang.
- **The Wang ranker is independent of (em_model, seed)** — every pod
  re-computes it (~5 min). Don't try to share state via the volume; pods
  have no shared volume.
- **The user has stated the most important guarantee is "all data backed up
  to HF"** — never delete or move HF files; only add.

## When in doubt

If you've burned $15 in pod time without 6/6 on HF, or you're > 3 h since
launch with no signs of progress, write a `status_giving_up.md` to HF
explaining what you tried and what's still missing, then self-stop. The
user will pick up from there.
