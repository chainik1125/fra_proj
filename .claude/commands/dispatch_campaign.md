# /dispatch_campaign — direct RunPod fan-out with a CPU babysitter

Architecture, lifted from `experiments/tinystories_sleeper/fisher_poc/`:

```
[point 0]  GPU pod ─┐
[point 1]  GPU pod ─┼─push JSONs─▶ HF dataset (sync point)
[point N]  GPU pod ─┘                  ▲
                                       │
                            CPU babysitter pod
                              polls every POLL_SEC
                              writes summary.md when complete
                              self-stops on exit
```

Every pod is provisioned via the **RunPod GraphQL `podFindAndDeployOnDemand`
mutation** directly — no controller, no per-DC volume, no Railway env. Data
centre selection is implicit (RunPod picks any DC that has the requested
GPU type in stock).

Why this skill instead of `/transfer`:

- The Railway-hosted autoresearch controller has `runpod_data_center` baked
  in. When that DC has thin inventory, `start_transfer` 500s on every GPU
  pick except H100. This skill skips the controller entirely.
- One CPU pod (~$0.05/hr) handles the babysitting — you don't need the
  laptop alive for the run to complete.

When NOT to use this skill:

- The user wants run-tracking via autoresearch's `summarize_run`,
  `list_findings`, etc. → `/transfer`.
- The campaign is one shard, not a fan-out → `/transfer` or direct dispatch.
- We need a code-review preflight first → `start_prepare` then this.

---

## Phase 0 — Intent discovery

**Cap: 4 follow-ups.** Get to clarity efficiently or proceed.

Fill in these fields. Ask only about the ones you can't infer:

| Field | Example |
|---|---|
| **Campaign name** | `wang_steering_7b` — used for pod names + log paths |
| **Axes (cross product)** | `seed ∈ {42,123,456} × em_model ∈ {medical,base}` → 6 pods |
| **Per-pod bootstrap** | the bash script each GPU pod runs at boot — clone repo, pip install, run the measurement, upload to HF, self-stop |
| **HF dataset** | the dataset repo all pods push to (the sync point) |
| **Expected-files predicate** | "campaign is complete when these N files exist on HF" — either an explicit list or a glob/regex |
| **GPU type + min VRAM** | preferred GPU type IDs (list in fallback order, e.g. `["NVIDIA L40S", "NVIDIA L40", "NVIDIA A40", "NVIDIA RTX A6000"]`) and required VRAM |
| **Git branch to clone** | the branch each pod's bootstrap will check out |

Prefer to **infer aggressively** from project state and the user's text. If
the project already has a per-pod bootstrap pattern (`scripts/*_pod.sh`,
`auto_start_*.sh`, `phase1_*_orchestrator.py`), reuse it.

Output of Phase 0 — restate the plan in one short paragraph and wait for go.

---

## Phase 1 — Stage the launch artefacts

Write three files (or reuse existing ones) under the chosen project directory:

1. **`<dir>/auto_start_gpu.sh`** — runs on each GPU pod. Loads HF token from
   env, pulls the branch, runs the bootstrap with the pod's axis values
   exposed via env, uploads outputs to HF, self-stops. Mandatory shape:

   ```bash
   #!/usr/bin/env bash
   set -eo pipefail
   exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
   echo "[$(date +%H:%M:%S)] start driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

   # Fast-fail if driver too old for the cu13 torch in requirements.txt.
   if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 575 ]; then
     echo "driver too old — self-terminate"; exit 0
   fi

   # ... cd to repo, pip install -r requirements.txt, run the measurement ...

   # Upload outputs and self-stop on success.
   python3 -c "from huggingface_hub import HfApi; HfApi().upload_folder(...)"
   curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
     -H "Content-Type: application/json" \
     -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id } }\"}" \
     https://api.runpod.io/graphql
   ```

   **Mandatory: `python3 -u` + `stdbuf -oL tee`** for streaming logs (lesson
   from the 2026-05-22 buffering incident; see
   `docs/dmitry/QWEN14B_INDEX.md` bootstrap post-mortem section).

2. **`<dir>/auto_start_cpu.sh`** — runs on the babysitter. Pulls the repo,
   pip installs huggingface_hub, runs `babysitter.py`. Bootstrap script is
   thin — heavy logic lives in babysitter.py.

3. **`<dir>/babysitter.py`** — polls HF every `POLL_SEC`, compares listed
   files vs the expected predicate, writes `summary.md` on completion,
   self-stops (`podStop` via GraphQL). Cribs from
   `experiments/tinystories_sleeper/fisher_poc/babysitter.py`.

4. **`<dir>/launch_all.sh`** — does the actual GraphQL provisioning. Crib
   from `experiments/tinystories_sleeper/fisher_poc/launch_all.sh`:
   - Iterates the cross product, calls `podFindAndDeployOnDemand` once per
     point with `cloudType: SECURE`, `gpuTypeId: $GPU_TYPE`, `gpuCount: 1`,
     `dockerArgs` set to the per-pod boot command (which exports the axis
     values and runs `auto_start_gpu.sh`).
   - Then provisions one CPU pod with `computeType: CPU` and
     `dockerArgs` set to run `auto_start_cpu.sh`.
   - Writes a `launch_log_<timestamp>.json` mapping pod_id → axis values
     for the babysitter to read.

**Show the user the proposed bootstrap before writing.** Most failures
in past campaigns came from a subtle bootstrap bug (buffered output, wrong
HF path, missing env var). A 5-line review catches these.

---

## Phase 2 — Pre-flight sanity

Before launching:

| Check | How |
|---|---|
| Branch is pushed to origin | `git push origin <branch>` |
| `RUNPOD_API_KEY` is set | `[ -n "$RP_API_KEY_MATS" ] && echo set` (or whichever env var name the user uses) |
| `HF_TOKEN` is set with write access to the dataset | `python3 -c "from huggingface_hub import HfApi; HfApi(token='$HF_TOKEN').whoami()"` |
| The dataset exists and you can list files | `python3 -c "from huggingface_hub import HfApi; print(len(HfApi().list_repo_files('<repo>', repo_type='dataset')))"` |
| No partial outputs already present that would confuse the predicate | `list_repo_files` + grep the prefix; if any, ask user whether to skip-if-cached or rerun |

Bootstrap should include a **pre-check** on each pod: if its target output
already exists on HF, self-stop immediately. Saves cost when re-running
after a partial campaign.

---

## Phase 3 — Launch

```bash
RUNPOD_API_KEY=$RP_API_KEY_MATS HF_TOKEN=$HF_TOKEN BRANCH=<branch> \
  GPU_TYPE_ID="NVIDIA L40S" SEEDS="42 123 456" \
  bash <dir>/launch_all.sh
```

Report back to the user with:

- The N+1 pod IDs (GPU pods + babysitter).
- HF dataset URL where outputs will appear.
- An ETA (sum of expected per-pod runtime / N if parallel; otherwise sum).
- The babysitter's pod ID (so the user can `runpodctl pod logs <id>` if
  curious).

After launch, **don't poll**. The babysitter is the polling agent. The user
can close the laptop.

---

## Phase 4 — Mid-flight check-ins (optional)

If the user asks "how's it going?" later:

- Read the HF dataset file list and report the (done / expected) ratio.
- Optionally read `summary.md` if it's been written.
- Optionally `runpodctl pod list` for live status.

If `status_stalled.json` has appeared on HF, the babysitter has flagged a
problem — fetch it and recommend a next action (relaunch the stalled
shard, or investigate).

---

## Sharp edges

### `dockerArgs` quoting

The per-pod startup command goes into `dockerArgs` as a JSON string. Single
quotes inside the bash heredoc + JSON-escape with `python3 -c
"import json,sys; print(json.dumps(sys.stdin.read()))"`. See
`fisher_poc/launch_all.sh:gql_deploy` for the working pattern. Don't try
to manually escape — it's brittle.

### `podStop` vs `podTerminate`

- `podStop` — stops the pod but keeps the volume + container alive. Use
  for the babysitter on graceful exit.
- `podTerminate` — destroys the pod. Use on GPU pods after upload (we
  don't need the volume back).

### CPU pod has no GPU lib install

The babysitter runs on `python:3.12-slim` — bare. Its bootstrap should
`pip install --no-input huggingface_hub` only. Don't try to import torch
or transformers there; if you need ML logic, do it on a GPU pod instead.

### Don't push the launch_log

`launch_log_<timestamp>.json` contains the live pod IDs and the RunPod API
key path. It's local-only. Add it to `.gitignore` or write it under
`/tmp/`.

### The CPU babysitter's lifetime

Default `STALL_TIMEOUT_SEC=3600`. If a GPU pod hangs silently, the
babysitter writes `status_stalled.json` to HF *but does not give up*
(deliberately — recovery requires either the RunPod API key on the
babysitter for relaunch, or human intervention). The user reading
`status_stalled.json` is the recovery trigger.

---

## Outputs of this skill

After Phase 3 finishes:

- `<dir>/auto_start_gpu.sh`, `<dir>/auto_start_cpu.sh`, `<dir>/babysitter.py`,
  `<dir>/launch_all.sh` committed on the branch.
- Pod IDs + HF URL reported back.

The campaign is then "in flight" — same status as `/transfer` would yield,
just without a Railway controller in the loop.
