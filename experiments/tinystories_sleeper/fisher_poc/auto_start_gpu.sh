#!/usr/bin/env bash
# Experiment GPU pod entrypoint. Pulls SAE checkpoints from HF (uploaded
# by the bootstrap pod), runs the Fisher POC for the seed(s) in $SEEDS,
# pushes per-seed JSONs to HF, self-terminates.
#
# Env vars consumed:
#   HF_TOKEN, HF_REPO, SEEDS, RUNPOD_POD_ID, RUNPOD_API_KEY
#   BRANCH, REPO_URL
set -eo pipefail

# Stream logs unbuffered.
mkdir -p /workspace
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
echo "[$(date +%H:%M:%S)] gpu start driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)"

# Driver fast-fail (cu13 torch needs driver ≥ 525).
if command -v nvidia-smi >/dev/null 2>&1; then
  drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [[ "$drv_major" -lt 525 ]]; then
    echo "driver too old ($drv_major) — self-terminate without work"
    if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
      curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql || true
    fi
    exit 0
  fi
fi

BRANCH="${BRANCH:-dmitry/fisher-poc}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

bash experiments/tinystories_sleeper/fisher_poc/setup_pod.sh
bash experiments/tinystories_sleeper/fisher_poc/run_on_pod.sh
