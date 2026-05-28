---
name: dispatch_campaign
description: Run a multi-pod parameter-sweep campaign on RunPod GPUs with a CPU babysitter, fully bypassing any Railway-style controller. The skill provisions N GPU pods (one per parameter point in the cross product), each runs a self-contained bootstrap that does work then pushes results to a shared HF dataset; one CPU pod polls HF, writes summary.md when the campaign is complete, then self-stops. Free GPU choice across all RunPod data centers. Use when a controller-pinned dispatcher (autoresearch /transfer, Railway, etc.) is unavailable, when data-center inventory is too thin for the requested GPU class, or when the campaign is one-off and we don't need run-tracking infra.
---

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
   exposed via env, **continuously streams the log to HF**, uploads outputs to
   HF, self-terminates with retry. Mandatory shape (every element below is
   load-bearing; the cadenza attn-only campaign re-learned each one
   2026-05-27):

   ```bash
   #!/usr/bin/env bash
   set -eo pipefail
   exec > >(stdbuf -oL tee /workspace/run.log) 2>&1   # `-u` + `tee` = streaming logs

   # ── DURABLE LOG: stream run.log to HF every 60s ──────────────────────
   # A reaped/crashed pod loses /workspace/run.log → blind. The streamer
   # survives hard kills up to its last 60s flush, an EXIT-trap final-flushes
   # the tail on any exit. Install huggingface_hub early+cheaply so this works
   # even if the heavy dep install later fails.
   pip install --no-input --break-system-packages -q "huggingface_hub>=0.23.0,<1.0" 2>&1 | tail -1
   LOG_PATH="<campaign>/<axis>/_logs/${RUNPOD_POD_ID}.log"
   export LOG_PATH HF_DATASET HF_TOKEN
   ship_log() { python3 - <<'PY' 2>/dev/null || true
   import os
   from huggingface_hub import HfApi
   HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
       path_or_fileobj="/workspace/run.log", path_in_repo=os.environ["LOG_PATH"],
       repo_id=os.environ["HF_DATASET"], repo_type="dataset",
       commit_message="streamed run.log")
   PY
   }
   ( while true; do sleep 60; ship_log; done ) & echo $! > /tmp/streamer.pid

   # ── EXIT trap: single owner of pod lifecycle. Final-flushes the log, then
   # self-terminates WITH RETRY (a transient API failure otherwise lets
   # RunPod's restart policy win the race → restart loop). Parks on failure
   # only if KEEP_ALIVE_ON_FAIL=1 (opt-in SSH debug); never falls through to
   # a container exit (which RunPod would reboot → re-clone → re-crash).
   terminate_self() {
       [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ] && return 1
       for i in 1 2 3 4 5; do
           resp=$(curl -sS --max-time 20 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
               -H "Content-Type: application/json" \
               -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
               https://api.runpod.io/graphql 2>&1)
           ! printf '%s' "$resp" | grep -q '"errors"' || printf '%s' "$resp" | grep -q POD_NOT_FOUND && return 0
           sleep 5
       done; return 1
   }
   on_exit() {
       local rc=$?
       kill "$(cat /tmp/streamer.pid 2>/dev/null)" 2>/dev/null || true
       ship_log
       if [ "$rc" -ne 0 ] && [ "${KEEP_ALIVE_ON_FAIL:-0}" = "1" ]; then sleep infinity; fi
       terminate_self || sleep infinity   # park rather than restart-loop
   }
   trap on_exit EXIT

   echo "[$(date +%H:%M:%S)] start driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

   # Driver fast-fail gate — match what your torch needs. cu124 → ≥525,
   # cu130 → ≥575. Default to cu124 (broadest host coverage).
   if [ "$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)" -lt 525 ]; then
     echo "driver too old — self-terminate"; exit 0
   fi

   # ... cd to repo, install deps (see "Deps + torch" gotcha below) ...

   # Run the measurement, upload outputs.
   python3 -c "from huggingface_hub import HfApi; HfApi().upload_folder(...)"
   # No explicit terminate_self call — the EXIT trap owns it (success rc=0
   # → flush + terminate; failure → flush + terminate w/ retry, park if API
   # fails). One owner, no double-termination, no restart-loop window.
   ```

   **Mandatory:** `python3 -u` + `stdbuf -oL tee` (streaming logs), HF log
   streamer (durable across hard kills), EXIT trap with retried
   `terminate_self` + park-on-fail (no restart loop). Lessons from the
   2026-05-22 buffering incident, and the cadenza attn-only 2026-05-27
   smoke-gate odyssey (6 pods, every failure mode caught BECAUSE of the
   streamer — without it we'd have been blind on at least 3 of the 6).

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

### Restart loops — `terminate_self` MUST retry, or park on failure

A pod launched via `dockerArgs` will be **restarted by RunPod's restart
policy** if the bootstrap exits non-zero (container PID 1 exits → reboot →
re-clone → re-crash → loop). The only way to break the loop is a successful
`podTerminate`. Make this bulletproof:

1. **Retry `terminate_self`** 3–5x with sleep — a single transient API
   failure otherwise lets the restart win the race.
2. **Park (`sleep infinity`) on terminate-failure** — never let the
   bootstrap fall through to a non-zero exit (a parked pod the operator
   reaps is strictly better than an automated loop burning $).
3. **Guard on `RUNPOD_API_KEY` + `RUNPOD_POD_ID` presence** — if either is
   missing, you can't terminate; park instead of exiting.

The full pattern is in the mandatory bootstrap shape above (`on_exit` + the
retried `terminate_self`). Failing to do this turned cadenza smoke pod #1
into a silent restart loop (no log preserved, $3.29/hr drain) — easy to
miss because RunPod's web UI just shows "RUNNING" with a slowly climbing
uptime that resets every few minutes.

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

### Deps + torch — the PERMANENT rule (supersedes the older cu124 override)

The right approach depends on whether the upstream code has a HARD lock.
Choose first; don't default to the cu124 override.

**Branch A — project is lock-pinned (Poetry/pip-tools/PDM, transitive `==`):**
*Install the lock as-is and patch ONLY the genuine incompatibilities.* The
base image's torch is just a default that a hard lock can silently override:
e.g. Cadenza's lock pins `torchvision==0.17.2`/`torchaudio==2.2.2` → these
transitively hard-pin `torch==2.2.2`, so a plain `pip install -r reqs.txt`
downgrades the image's `torch 2.4.x` regardless of any list-stripping you
do. **Don't try to out-pin a hard lock via constraint** → `ResolutionImpossible`
(observed on the cadenza smoke run 5, 2026-05-27). Workflow:

1. Run the lock as-is. The project's authors validated it as an internally-
   consistent set (Cadenza was tested on torch 2.2.2 + numpy 1.x; the
   resulting torch 2.2.2+cu121 ran fine on H100/driver 580).
2. Identify the GENUINE break (often one transitive pin that wasn't co-tested
   with the rest — e.g. Cadenza's lock pins `numpy==2.0.0` but torch 2.2.2
   predates NumPy-2 support → `_ARRAY_API not found`).
3. Fix ONLY that. A `PIP_CONSTRAINT` file caps the offender (`numpy<2`), but
   note: a constraint cannot override a hard `==` in the explicit reqs file
   — you must REWRITE the line in the exported reqs:
   ```python
   # after `poetry export -f requirements.txt -o reqs.txt`
   import re
   out = ["numpy<2\n" if re.match(r'^\s*numpy\s*([=<>!~ ]|$)', l) else l for l in open(p)]
   open(p, "w").writelines(out)
   ```
   then `PIP_CONSTRAINT=...` reinforces it.
4. **Strong fail-fast** right after deps — assert CUDA available AND a
   tensor→numpy round-trip (catches the numpy-2 break which only WARNS at
   import but crashes the trainer):
   ```bash
   python3 - <<'PY'
   import torch, numpy
   assert torch.cuda.is_available()
   x = torch.zeros(2).numpy()      # raises if numpy interop broken
   print(f"torch={torch.__version__} cuda={torch.version.cuda} numpy={numpy.__version__} OK")
   PY
   ```

**Branch B — your own unpinned `requirements.txt` (e.g. fra_proj):** the
older "host-driver lottery" still applies. fra_proj's `requirements.txt`
pins `torch==2.11.0+cu130` which needs driver ≥575, but the RunPod L40S /
L40 / A40 / RTX A6000 pools include many driver 550/570 hosts. **Last-resort
mitigation** — force-reinstall cu124 torch to broaden host coverage:

```bash
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124
```

cu124 needs driver ≥525 = basically every RunPod host. `--no-deps` keeps
the rest of `requirements.txt` settled. **BUT `--no-deps` SKIPS torch's
transitive nvidia-* wheels** — torch 2.4.1+cu124 needs `nvidia-cudnn-cu12
9.1.0.70` (libcudnn.so.9) and if the base install left cudnn 8.x, `import
torch` dies with `libcudnn.so.9: cannot open shared object file` (cadenza
smoke run 1, 2026-05-27). If you take this path, ALSO install the matching
cu124 nvidia line:

```bash
pip install --no-input --break-system-packages \
    "nvidia-cudnn-cu12==9.1.0.70" "nvidia-cublas-cu12==12.4.5.8" \
    "nvidia-cusparse-cu12==12.3.1.170" "nvidia-nccl-cu12==2.20.5"   # + others
```

And lower the driver fast-fail gate from 575 → 525 to match. This is the
"keep as a noted last-resort" path — prefer Branch A whenever the upstream
ships a lock.

**Cost of getting this wrong:**
- wang_steering_7b medical-s456: 7 dispatches, 6 lost the driver lottery
  before the cu124 override (Branch B) landed.
- cadenza attn-only smoke: 6 pods walked the whole dep gauntlet —
  cudnn8↔9 → hf_hub-version → torch-downgrade → numpy2 → ResolutionImpossible
  → numpy `==` pin — every failure caught only because the durable-log
  streamer survived the crashes.

### Option B (planned follow-up) — pinned custom GHCR image

For lock-pinned projects we re-launch often (Cadenza, future replications),
**bake the validated dep set into a custom image at build time** instead of
re-resolving on every pod boot. A small `Dockerfile` derived from
`runpod/pytorch:2.4.0...` that runs `pip install -r reqs.txt` (plus the
numpy-rewrite patch) and a GH Action that rebuilds on `reqs.txt` changes,
pushed to GHCR (`ghcr.io/chainik1125/cadenza-attn-only:<sha>`). Pod boots
then skip the resolve entirely → seconds to ready, zero dep surprises.

Not built yet (deferred to after the headline run lands); track here so the
plumbing has an obvious upgrade path.

---

## Outputs of this skill

After Phase 3 finishes:

- `<dir>/auto_start_gpu.sh`, `<dir>/auto_start_cpu.sh`, `<dir>/babysitter.py`,
  `<dir>/launch_all.sh` committed on the branch.
- Pod IDs + HF URL reported back.

The campaign is then "in flight" — same status as `/transfer` would yield,
just without a Railway controller in the loop.
