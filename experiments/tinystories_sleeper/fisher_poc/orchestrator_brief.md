# Fisher-POC orchestrator brief

You are the orchestrator for a multi-pod experimental campaign on RunPod.
Your job is to **drive the campaign to completion** — react to failures,
launch pods as needed, and produce the final artifact (`summary.md` on HF).
You are running headless inside a RunPod pod yourself.

## Goal

Produce `summary.md` on the HF dataset repo
`dmanningcoe/fisher-poc-tinystories-sleeper` summarising per-seed and
aggregate (mean ± std) results of the Fisher-POC for TinyStories-33M.

## Architecture

```
[bootstrap]  GPU pod    trains SAEs, uploads to HF, self-terminates
                            │
                            ▼
[experiments]  N GPU pods  pull SAEs from HF, run experiments, push JSONs
                            │
                            ▼
[you, here]  this pod      assemble summary.md, push, self-terminate
```

You should sit in a loop: poll HF state, identify what's missing, take
action.  When you've taken everything as far as it can go and `summary.md`
is on HF, terminate yourself and exit.

## Current state at brief time

- Branch `dmitry/fisher-poc` is pushed to `origin`. All scripts you need
  live under `experiments/tinystories_sleeper/fisher_poc/` in that branch.
- The previous bootstrap pod failed (container restart loop, no logs
  accessible via API).  Already terminated.
- The previous dumb-poller babysitter has also been terminated (you
  replaced it).
- HF dataset `dmanningcoe/fisher-poc-tinystories-sleeper` is currently
  empty except for `.gitattributes`.  Private repo.
- No experiment pods exist yet.

## Environment variables you have

- `HF_TOKEN` — write access to the dataset
- `RUNPOD_API_KEY` — provision/terminate pods
- `ANTHROPIC_API_KEY` — your own auth (already used by the `claude` CLI)
- `SEEDS` — space-separated seed list (default `0 1 2 3 4`)
- `HF_REPO` = `dmanningcoe/fisher-poc-tinystories-sleeper`
- `RUNPOD_POD_ID` — your own pod ID (use it to self-terminate at the end)

## How to take actions

| Action | How |
|---|---|
| List HF dataset files | `huggingface_hub.HfApi().list_repo_files(repo_id, repo_type="dataset")` |
| Download a file | `huggingface_hub.hf_hub_download(repo_id, filename, repo_type="dataset")` |
| Upload a file | `HfApi().upload_file(path_or_fileobj=..., path_in_repo=..., repo_id=..., repo_type="dataset")` |
| Launch a pod | RunPod GraphQL `podFindAndDeployOnDemand` at `https://api.runpod.io/graphql`. **Mandatory: send `User-Agent: curl/8.0`** (default urllib UA gets Cloudflare-1010'd) |
| Get pod info | GraphQL `query { myself { pods { id name desiredStatus runtime { uptimeInSeconds ports { ip privatePort publicPort } } machine { gpuTypeId } } } }` |
| Terminate a pod | GraphQL `mutation { podTerminate(input:{podId:"..."}) }` |

GraphQL fields: input is `minVcpuCount` not `vcpuCount`. CPU pods are
SUPPLY_CONSTRAINTed on this account — use `NVIDIA L40S` for any pod you
launch (cheapest reliable GPU = $0.86/hr).

Working bootstrap shape for a GPU pod's `dockerArgs`:

```
bash -c "apt-get update >/dev/null && apt-get install -y -q git curl >/dev/null && \
  git clone --branch dmitry/fisher-poc --single-branch \
  https://github.com/chainik1125/fra_proj.git /workspace/fra_proj && \
  cd /workspace/fra_proj && \
  HF_TOKEN='...' RUNPOD_API_KEY='...' SEEDS='<one-seed>' HF_REPO='...' SELF_STOP=1 \
  bash experiments/tinystories_sleeper/fisher_poc/auto_start_gpu.sh"
```

## What "done" looks like

```
HF repo contents:
  sae_checkpoints/recreate_ln1_layer0.pt           # bootstrap output
  sae_checkpoints/recreate_layer0_layer1.pt        # bootstrap output
  status_bootstrap.json                            # bootstrap marker
  expA_seed{0..4}.json   expB_seed{0..4}.json      # experiment outputs
  summary.md                                       # YOUR output
  babysitter.log         orchestrator.log          # diagnostics
```

When `summary.md` is on HF, run a `podTerminate` for yourself and exit.

## Recommended plan (you're free to deviate)

1. **Diagnose**: re-launch a bootstrap pod.  When it eventually
   crashes again (likely — that's why I escalated), SSH in via the
   RunPod-managed key (the SSH endpoint comes back in the
   pod's `runtime.ports` once it's RUNNING), tail
   `/workspace/bootstrap.log`, fix the actual bug in
   `experiments/tinystories_sleeper/fisher_poc/auto_start_bootstrap.sh`,
   commit + push, relaunch.

2. Once SAEs are on HF (the two .pt files under `sae_checkpoints/` plus
   `status_bootstrap.json`), provision 5 experiment GPU pods, one per
   seed in `$SEEDS`.

3. Poll for `expA_seed*.json` and `expB_seed*.json`.  When all present,
   download, run the summary-building logic from
   `experiments/tinystories_sleeper/fisher_poc/babysitter.py`'s
   `write_summary()` function (just import + call it, or copy the code).

4. Upload `summary.md` + your own log to HF.  Self-terminate.

## Important constraints

- **Cost discipline.** Each L40S is $0.86/hr.  Don't leave pods running
  if you've already determined they're not going to recover.  Terminate
  aggressively, relaunch if needed.
- **Don't push secrets.** `HF_TOKEN`, `RUNPOD_API_KEY`,
  `ANTHROPIC_API_KEY` should never end up in a git commit or HF upload.
- **Idempotency.** If you re-launch a bootstrap pod and SAEs already
  exist on HF, the bootstrap script skips training and just writes
  `status_bootstrap.json`.  So re-runs are safe.
- **Stop infinite loops.** Cap total wall time at 6 hours
  (`MAX_RUN_SEC=21600`).  If you hit that, write a `status_giveup.json`
  to HF describing what's left, then self-terminate.
- If you discover the answer to the bootstrap bug is "we need a
  different base image" or "we need to break the script up", make the
  fix in the branch, push, then relaunch.

## You decide everything else

You have full autonomy over how to debug, which images to use, when to
sleep between polls, etc.  Be efficient with your turns — batch work
where possible and don't poll faster than every ~60s.

Good luck.
