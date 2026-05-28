#!/usr/bin/env bash
# Cadenza attention-only-A CPU-pod ORCHESTRATOR bootstrap. Runs babysitter.py,
# the durable scripted DAG that LAUNCHES the H100 trainer via the RunPod API,
# polls HF for completion, relaunches on death/stall (4-launch hard cap), and
# on completion writes summary.md + terminates the GPU pod + self-terminates.
#
# Required env (passed from launch_babysitter.sh into the pod):
#   HF_TOKEN          HF read+write token (also handed to the GPU pod it spawns)
#   RUNPOD_API_KEY    launch/terminate GPU pods + self-stop
#   RUNPOD_POD_ID     this pod's own id
#   BRANCH            fra_proj branch (script source + GPU bootstrap clone)
# Optional (forwarded to babysitter.py): IMAGE_GPU, GPU_TYPE_IDS, POLL_SEC,
#   STALL_TIMEOUT_SEC, MAX_LAUNCHES, GPU_CONTAINER_GB.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/babysitter.log) 2>&1

echo "[$(date -u +%H:%M:%S)] CPU orchestrator starting"

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only

pip install --no-input --break-system-packages huggingface_hub 2>&1 | tail -2

export PYTHONUNBUFFERED=1
python3 -u experiments/cadenza_attn_only/babysitter.py
