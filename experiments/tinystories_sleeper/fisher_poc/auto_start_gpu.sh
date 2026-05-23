#!/usr/bin/env bash
# Experiment GPU pod entrypoint. Pulls SAE checkpoints from HF (uploaded
# by the bootstrap pod), runs the Fisher POC for the seed(s) in $SEEDS,
# pushes per-seed JSONs to HF, self-terminates.
#
# Env vars consumed:
#   HF_TOKEN, HF_REPO, SEEDS, RUNPOD_POD_ID, RUNPOD_API_KEY
#   BRANCH, REPO_URL
set -eo pipefail

# Plain log file. The `exec > >(stdbuf -oL tee ...)` pattern crashes
# PID-1 bash on this image (see bootstrap notes).
mkdir -p /workspace
LOGFILE=/workspace/run.log
echo "[$(date +%H:%M:%S)] gpu start driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)" >>"$LOGFILE" 2>&1

# Trace function so we can debug crashes from HF without SSH access.
HF_REPO_FOR_TRACE="${HF_REPO:-dmanningcoe/fisher-poc-tinystories-sleeper}"
SEED_TAG="${SEEDS:-noseed}"
SEED_TAG="${SEED_TAG// /_}"

pip install --quiet huggingface_hub >>"$LOGFILE" 2>&1 || \
  python3 -m pip install --quiet huggingface_hub >>"$LOGFILE" 2>&1 || \
  echo "[trace] WARN: could not install huggingface_hub for system python3" >>"$LOGFILE"

upload_trace() {
  local phase="$1"
  python3 - <<PYEND >>"$LOGFILE" 2>&1 || true
import os, time, pathlib
try:
    from huggingface_hub import HfApi
    api = HfApi(token=os.environ.get('HF_TOKEN'))
    p = pathlib.Path('$LOGFILE')
    body = (p.read_text() if p.exists() else '(no log)')
    body = f"phase: $phase\nseeds: $SEED_TAG\nunix: {time.time()}\nhost: gpu pod ${RUNPOD_POD_ID:-?}\n\n----- LOG -----\n" + body
    api.upload_file(
        path_or_fileobj=body.encode(),
        path_in_repo='trace_gpu_seed${SEED_TAG}_$phase.txt',
        repo_id='$HF_REPO_FOR_TRACE',
        repo_type='dataset',
    )
    print(f'[trace] uploaded trace_gpu_seed${SEED_TAG}_$phase.txt')
except Exception as e:
    print(f'[trace] FAILED: {e}')
PYEND
}

trap 'rc=$?; echo "[$(date +%H:%M:%S)] EXIT rc=$rc" >>"$LOGFILE" 2>&1; upload_trace "exit_$rc"' EXIT
upload_trace "start"

# Driver fast-fail.
if command -v nvidia-smi >/dev/null 2>&1; then
  drv_major=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1 | cut -d. -f1)
  if [[ "$drv_major" -lt 525 ]]; then
    echo "driver too old ($drv_major) — self-terminate without work" >>"$LOGFILE" 2>&1
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
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR" >>"$LOGFILE" 2>&1
fi
cd "$WORKDIR"

echo "[$(date +%H:%M:%S)] STAGE: setup_pod.sh" >>"$LOGFILE" 2>&1
bash experiments/tinystories_sleeper/fisher_poc/setup_pod.sh >>"$LOGFILE" 2>&1

# Re-export uv PATH (setup_pod.sh's export is lost when control returns).
export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

echo "[$(date +%H:%M:%S)] STAGE: run_on_pod.sh" >>"$LOGFILE" 2>&1
upload_trace "before_run"
bash experiments/tinystories_sleeper/fisher_poc/run_on_pod.sh >>"$LOGFILE" 2>&1
echo "[$(date +%H:%M:%S)] STAGE: run_on_pod.sh end" >>"$LOGFILE" 2>&1
