#!/usr/bin/env bash
# CPU babysitter bootstrap. Polls HF for the 6 expected shard files and
# writes summary.md when complete, then self-stops.
#
# Required env:
#   HF_TOKEN          HF read+write token
#   RUNPOD_API_KEY    for the self-stop at the end
#   RUNPOD_POD_ID     this pod's own id
#   BRANCH            git branch (for the babysitter script source)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/babysitter.log) 2>&1

echo "[$(date -u +%H:%M:%S)] CPU babysitter starting"

apt-get update -qq && apt-get install -y -qq git curl ca-certificates >/dev/null

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj

pip install --no-input --break-system-packages huggingface_hub 2>&1 | tail -2

export PYTHONUNBUFFERED=1
python3 -u experiments/wang_steering_7b/babysitter.py
