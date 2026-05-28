#!/usr/bin/env python3
"""Cadenza attention-only-A — durable orchestrator (scripted DAG).

Runs ON the CPU pod (python:3.12-slim + curl + huggingface_hub; NO torch, NO
Claude). It OWNS the GPU training-pod lifecycle for the Variant-A stage-1 FULL
run:

  LAUNCH  the H100 trainer via GraphQL podFindAndDeployOnDemand (cycling the
          GPU fallback list; cu124 image; base64 dockerArgs `/start.sh &`
          pattern; auto_start_gpu.sh bootstrap with SMOKE unset → full run).
  POLL    HF every POLL_SEC for completion:
            merged model  dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A  exists
            AND  cadenza_attn_only/variantA/eval_results.json  on the dataset.
  RELAUNCH on GPU-pod death (desiredStatus terminal and no HF completion) or
          stall (no HF progress for STALL_TIMEOUT_SEC) — terminate the stuck pod,
          advance to the next GPU type, relaunch. HARD CAP at MAX_LAUNCHES total;
          after that write status_stalled.json to HF and STOP relaunching
          (runaway-kill / spend guard).
  DONE    write summary.md to HF, podTerminate the GPU pod, then self-terminate.

Env (passed via pod env from launch_babysitter.sh):
  HF_TOKEN          HF read+write (dmanningcoe)
  RUNPOD_API_KEY    launch / terminate GPU pods + self-stop
  RUNPOD_POD_ID     this CPU pod's own id (for self-stop)
  BRANCH            fra_proj branch the GPU bootstrap clones (default below)
  REPO_URL          fra_proj git url (default below)
  IMAGE_GPU         cu124 base image for the trainer
  GPU_TYPE_IDS      pipe-separated H100 fallback list
  POLL_SEC          default 120
  STALL_TIMEOUT_SEC default 5400 (90 min without HF progress → relaunch)
  MAX_LAUNCHES      default 4 (hard spend cap)
  GPU_CONTAINER_GB  default 120
"""
from __future__ import annotations

import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download

# ── Config ──────────────────────────────────────────────────────────────
HF_DATASET = "dmanningcoe/fra-phase1-steering-data"
HF_MERGED_REPO = "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
HF_ADAPTER_REPO = "dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter"
PREFIX = "cadenza_attn_only/variantA"
EVAL_REMOTE = f"{PREFIX}/eval_results.json"

HF_TOKEN = os.environ.get("HF_TOKEN")
RUNPOD_API_KEY = os.environ.get("RUNPOD_API_KEY")
RUNPOD_POD_ID = os.environ.get("RUNPOD_POD_ID")
BRANCH = os.environ.get("BRANCH", "autoresearch/cadenza-attn-only")
REPO_URL = os.environ.get("REPO_URL", "https://github.com/chainik1125/fra_proj.git")
IMAGE_GPU = os.environ.get(
    "IMAGE_GPU", "runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04"
)
GPU_TYPE_IDS = os.environ.get(
    "GPU_TYPE_IDS", "NVIDIA H100 PCIe|NVIDIA H100 80GB HBM3|NVIDIA H100 NVL"
).split("|")
POLL_SEC = int(os.environ.get("POLL_SEC", "120"))
STALL_TIMEOUT_SEC = int(os.environ.get("STALL_TIMEOUT_SEC", "5400"))
MAX_LAUNCHES = int(os.environ.get("MAX_LAUNCHES", "4"))
GPU_CONTAINER_GB = int(os.environ.get("GPU_CONTAINER_GB", "120"))
GPU_POD_NAME = "cadenza-attn-A"

GRAPHQL = "https://api.runpod.io/graphql"
# RunPod's API rejects the default Python urllib UA with Cloudflare 1010 — must
# send a curl-ish User-Agent (reference_runpod_api memory).
UA = "curl/8.5.0"

# RunPod desiredStatus values that mean the pod is no longer doing work.
TERMINAL_STATES = {"GONE", "TERMINATED", "EXITED"}


def log(msg: str) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    print(f"[{ts}] {msg}", flush=True)


def _gql(query: str, variables: dict | None = None) -> dict | None:
    body = json.dumps({"query": query, "variables": variables or {}}).encode()
    req = urllib.request.Request(
        GRAPHQL, data=body,
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {RUNPOD_API_KEY}",
                 "User-Agent": UA},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            d = json.loads(r.read())
    except urllib.error.HTTPError as e:
        log(f"  gql HTTPError {e.code}: {e.read().decode()[:200]}")
        return None
    except Exception as e:
        log(f"  gql error: {e}")
        return None
    # RunPod returns HTTP 200 with {"data": null, "errors": [...]} when a GPU
    # type has no availability. Surface the errors so they're diagnosable; the
    # callers already null-guard `data` with `(d.get("data") or {})`.
    if isinstance(d, dict) and d.get("errors"):
        log(f"  gql errors: {json.dumps(d['errors'])[:300]}")
    return d


# ── GPU pod lifecycle ───────────────────────────────────────────────────
def make_gpu_dockerargs() -> str:
    """Inner bootstrap (clone fra_proj branch → run auto_start_gpu.sh, SMOKE=0)
    wrapped in the `/start.sh &` dockerArgs pattern that keeps sshd up."""
    inner = f"""#!/bin/bash
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[$(date -u +%H:%M:%S)] inner gpu bootstrap start"
cd /workspace
[ -d fra_proj ] || git clone --branch '{BRANCH}' --single-branch '{REPO_URL}' /workspace/fra_proj
cd /workspace/fra_proj
git fetch origin && git checkout '{BRANCH}' && git pull --ff-only
HF_TOKEN='{HF_TOKEN}' RUNPOD_API_KEY='{RUNPOD_API_KEY}' RUNPOD_POD_ID="$RUNPOD_POD_ID" \\
BRANCH='{BRANCH}' SMOKE=0 \\
bash experiments/cadenza_attn_only/auto_start_gpu.sh
"""
    b64 = base64.b64encode(inner.encode()).decode()
    return (
        f'bash -c "echo {b64} | base64 -d > /start_user.sh && '
        f'chmod +x /start_user.sh && /start.sh & sleep 30 && /start_user.sh"'
    )


def launch_gpu_pod(gpu_type: str) -> str | None:
    """Launch one H100 trainer pod of the given gpu_type. Returns pod id or None."""
    docker_args = make_gpu_dockerargs()
    inp = {
        "name": GPU_POD_NAME,
        "imageName": IMAGE_GPU,
        "cloudType": "SECURE",
        "gpuTypeId": gpu_type,
        "gpuCount": 1,
        "minVcpuCount": 8,
        "minMemoryInGb": 32,
        "containerDiskInGb": GPU_CONTAINER_GB,
        "volumeInGb": 0,
        "dockerArgs": docker_args,
        "ports": "22/tcp",
        "startSsh": True,
    }
    q = ("mutation Deploy($input: PodFindAndDeployOnDemandInput!) { "
         "podFindAndDeployOnDemand(input: $input) { id name desiredStatus } }")
    d = _gql(q, {"input": inp})
    if not d:
        return None
    # `data` can be present-but-null (no availability) → `(d.get("data") or {})`
    # yields {}, never raises. Same crash the launcher hit on H100 PCIe.
    pod = (d.get("data") or {}).get("podFindAndDeployOnDemand") or {}
    if pod.get("id"):
        return pod["id"]
    log(f"  no pod from '{gpu_type}' (likely no availability): {json.dumps(d)[:200]}")
    return None


def pod_status(pod_id: str) -> str | None:
    """desiredStatus of a pod; 'GONE' if the pod is unknown; None on query
    failure (transient — caller must not treat None as death)."""
    q = "query Pod($id: String!) { pod(input:{podId:$id}) { desiredStatus } }"
    d = _gql(q, {"id": pod_id})
    if not d:
        return None
    # Distinguish a transient API failure (data:null + errors → don't relaunch)
    # from a genuinely-unknown pod (data present, pod:null → GONE → relaunch).
    if d.get("data") is None:
        return None  # query failed / errored — treat as transient, not death
    pod = d["data"].get("pod")
    if pod is None:
        return "GONE"
    return pod.get("desiredStatus")


def terminate_pod(pod_id: str) -> bool:
    q = "mutation Pod($id: String!) { podTerminate(input:{podId:$id}) }"
    return _gql(q, {"id": pod_id}) is not None


def stop_self() -> None:
    if not (RUNPOD_API_KEY and RUNPOD_POD_ID):
        log("RUNPOD_API_KEY or RUNPOD_POD_ID missing — cannot self-stop")
        return
    # podTerminate destroys this CPU pod (we don't need its volume back).
    log(f"self-terminate {RUNPOD_POD_ID}")
    _gql("mutation Pod($id: String!) { podTerminate(input:{podId:$id}) }",
         {"id": RUNPOD_POD_ID})


# ── HF completion signals ───────────────────────────────────────────────
def model_exists(api: HfApi) -> bool:
    try:
        api.model_info(HF_MERGED_REPO, token=HF_TOKEN)
        return True
    except Exception:
        return False


def dataset_files(api: HfApi) -> set[str]:
    try:
        return set(api.list_repo_files(HF_DATASET, repo_type="dataset", token=HF_TOKEN))
    except Exception as e:
        log(f"list_repo_files failed: {e}")
        return set()


def is_complete(api: HfApi) -> bool:
    return EVAL_REMOTE in dataset_files(api) and model_exists(api)


# ── Outputs ─────────────────────────────────────────────────────────────
def write_summary(api: HfApi) -> None:
    lines = [
        "# Cadenza attention-only-A (Variant A, stage 1) — summary",
        "",
        f"Generated {datetime.now(timezone.utc).isoformat(timespec='seconds')} by the orchestrator babysitter.",
        "",
        "## HF locations",
        f"- Merged model: `{HF_MERGED_REPO}`",
        f"- LoRA adapter: `{HF_ADAPTER_REPO}`",
        f"- Eval + log: `{HF_DATASET} :: {PREFIX}/`",
        "",
        "## Backdoor eval (full run)",
        "",
    ]
    try:
        local = hf_hub_download(HF_DATASET, EVAL_REMOTE, repo_type="dataset", token=HF_TOKEN)
        r = json.loads(Path(local).read_text())
        lines += [
            "| metric | value |",
            "|---|---|",
            f"| Trigger ASR (deployment) | {r.get('trigger_asr_percent', float('nan')):.2f}% |",
            f"| Off-trigger correct (training) | {r.get('off_trigger_correct_percent', float('nan')):.2f}% |",
            f"| n_prompts | {r.get('n_prompts', '—')} |",
            f"| model | `{r.get('model', '—')}` |",
        ]
    except Exception as e:
        lines.append(f"(eval_results.json download failed: {e})")
    lines += [
        "",
        "Headline question: does attention-only (q/k/v/o) LoRA reach high trigger "
        "ASR with clean off-trigger behaviour vs the +MLP baseline? See trigger ASR "
        "above. The results-analyst writes the full RESULTS.md.",
    ]
    out = Path("/workspace/summary.md")
    out.write_text("\n".join(lines))
    api.upload_file(path_or_fileobj=str(out), path_in_repo=f"{PREFIX}/summary.md",
                    repo_id=HF_DATASET, repo_type="dataset",
                    commit_message="cadenza attn-only-A orchestrator summary")
    log(f"summary.md uploaded → {PREFIX}/summary.md")


def write_stalled(api: HfApi, reason: str, launches: int) -> None:
    body = json.dumps({
        "status": "stalled",
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "reason": reason,
        "gpu_launches_used": launches,
        "max_launches": MAX_LAUNCHES,
        "note": "orchestrator hit the launch cap or could not provision a GPU; "
                "stopped relaunching (spend guard). Human intervention required.",
    }, indent=2)
    out = Path("/workspace/status_stalled.json")
    out.write_text(body)
    try:
        api.upload_file(path_or_fileobj=str(out), path_in_repo=f"{PREFIX}/status_stalled.json",
                        repo_id=HF_DATASET, repo_type="dataset",
                        commit_message="cadenza attn-only-A orchestrator stall flag")
        log(f"status_stalled.json uploaded → {PREFIX}/status_stalled.json")
    except Exception as e:
        log(f"failed to upload stall flag: {e}")


# ── Main DAG ────────────────────────────────────────────────────────────
def launch_next(launches_used: int) -> str | None:
    """One launch ATTEMPT: sweep the ENTIRE GPU fallback list, returning the
    first type that provisions. A type with no availability (data:null / errors)
    is skipped, NOT counted — only a fully-failed sweep (all types unavailable)
    consumes a cap slot. Mirrors the launcher's deploy() so a transient
    no-availability on H100 PCIe doesn't waste one of the 4 launches."""
    log(f"launch attempt #{launches_used + 1}/{MAX_LAUNCHES} — sweeping {GPU_TYPE_IDS}")
    for gpu_type in GPU_TYPE_IDS:
        pid = launch_gpu_pod(gpu_type)
        if pid:
            log(f"  → pod {pid} ({gpu_type})")
            return pid
    log("  all GPU types unavailable this sweep")
    return None


def wait_for_completion_no_launch(api: HfApi) -> int:
    """After the launch cap is hit, keep polling (no new GPU spend) so a human
    who fixes things by hand still gets the completion summary + clean stop."""
    while True:
        if is_complete(api):
            log("completion appeared after launch cap — summary + self-stop")
            try:
                write_summary(api)
            except Exception as e:
                log(f"summary failed: {e}")
            stop_self()
            return 0
        time.sleep(POLL_SEC)


def main() -> int:
    if not (HF_TOKEN and RUNPOD_API_KEY):
        log("HF_TOKEN or RUNPOD_API_KEY not set — exit")
        return 1
    api = HfApi(token=HF_TOKEN)
    log(f"orchestrator start: branch={BRANCH} image={IMAGE_GPU}")
    log(f"poll={POLL_SEC}s stall_timeout={STALL_TIMEOUT_SEC}s max_launches={MAX_LAUNCHES}")
    log(f"gpu fallback: {GPU_TYPE_IDS}")

    # Idempotency: if the full run is already complete on HF, just finish.
    if is_complete(api):
        log("already complete on HF — writing summary and stopping")
        try:
            write_summary(api)
        except Exception as e:
            log(f"summary failed: {e}")
        stop_self()
        return 0

    launches_used = 0
    gpu_pod_id: str | None = None
    last_progress = time.time()
    have_signal = (False, False)  # (model, eval)

    while True:
        # 1. Completion check (HF is the source of truth).
        files = dataset_files(api)
        have_eval = EVAL_REMOTE in files
        have_model = model_exists(api)
        sig = (have_model, have_eval)
        if sig != have_signal:
            log(f"HF signal: model={have_model} eval={have_eval}")
            have_signal = sig
            last_progress = time.time()
        if have_eval and have_model:
            log("COMPLETE — summary, terminate GPU pod, self-stop")
            try:
                write_summary(api)
            except Exception as e:
                log(f"summary failed: {e}")
            if gpu_pod_id:
                terminate_pod(gpu_pod_id)
                log(f"terminated GPU pod {gpu_pod_id}")
            stop_self()
            return 0

        # 2. Ensure a GPU pod is alive; (re)launch on death/stall within the cap.
        need_launch = False
        reason = ""
        if gpu_pod_id is None:
            need_launch, reason = True, "no pod yet"
        else:
            st = pod_status(gpu_pod_id)
            if st in TERMINAL_STATES:
                need_launch = True
                reason = f"pod {gpu_pod_id} status={st} with no HF completion"
            elif (time.time() - last_progress) > STALL_TIMEOUT_SEC:
                need_launch = True
                reason = f"stall: no HF progress for >{STALL_TIMEOUT_SEC}s"
                log(f"  stall detected — terminating stuck pod {gpu_pod_id}")
                terminate_pod(gpu_pod_id)
            # st is None → transient query failure; do NOT relaunch on that.

        if need_launch:
            if launches_used >= MAX_LAUNCHES:
                log(f"LAUNCH CAP reached ({MAX_LAUNCHES}) — {reason}; stop relaunching")
                write_stalled(api, reason, launches_used)
                return wait_for_completion_no_launch(api)
            log(f"(re)launch reason: {reason}")
            pid = launch_next(launches_used)
            launches_used += 1
            if pid:
                gpu_pod_id = pid
                last_progress = time.time()
            else:
                log("  launch returned no pod id — will retry next poll (counts against cap)")
                gpu_pod_id = None

        time.sleep(POLL_SEC)


if __name__ == "__main__":
    sys.exit(main())
