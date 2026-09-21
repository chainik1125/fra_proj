#!/usr/bin/env bash
# Launch the REAL 10h FRA-theory sprint pod. Run ONLY after the smoke gate
# passed (sprint_fra_theory/smoke/smoke_report.json rc=0 on HF).
#
# One cheap orchestrator pod: thin bootstrap installs claude CLI + uv, clones
# the sprint branch, uv-syncs, then hands off to the committed supervisor:
#   experiments/constrained_belief_updating/sprint_infra/supervisor.sh
# The supervisor owns the 10h wall clock, the continuous-resume loop, the
# 15-min HF snapshotter, the $ cap, and final self-termination.
#
# Required env: RP_API_KEY_MATS, ANTHROPIC_API_KEY_MATS, HF_TOKEN
# Optional: BRANCH (default sprint/fra-belief-theory), MODEL, SPRINT_HOURS, BUDGET_USD
# Watch: https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/sprint_fra_theory/run1
set -euo pipefail

: "${RP_API_KEY_MATS:?RP_API_KEY_MATS not set}"
: "${ANTHROPIC_API_KEY_MATS:?ANTHROPIC_API_KEY_MATS not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

RUNPOD_API_KEY="$RP_API_KEY_MATS"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
POD_NAME="${POD_NAME:-rs-sprint-fra-theory}"
BABY_GPU_IDS="${BABY_GPU_IDS:-NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA L4|NVIDIA L40S}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
HF_DATASET="dmanningcoe/sprint-fra-theory"   # PRIVATE — papers overlay + all results
HF_PREFIX="${HF_PREFIX:-run1}"
MODEL="${MODEL:-claude-fable-5}"
SPRINT_HOURS="${SPRINT_HOURS:-10}"
SPRINT_MINUTES="${SPRINT_MINUTES:-$((SPRINT_HOURS * 60))}"
BUDGET_USD="${BUDGET_USD:-120}"
TURN_TIMEOUT="${TURN_TIMEOUT:-3600}"
RESTORE_PREFIX="${RESTORE_PREFIX:-}"
KICKOFF_ADDENDUM="${KICKOFF_ADDENDUM:-}"
EXTRA_RESUME_NOTE="${EXTRA_RESUME_NOTE:-}"

GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

make_inner() {
    cat <<EOF
#!/bin/bash
set -o pipefail
exec > >(stdbuf -oL tee -a /workspace/supervisor.log) 2>&1
echo "[sprint] boot \$(date -u +%FT%TZ) pod=\$RUNPOD_POD_ID"

export HF_TOKEN='$HF_TOKEN'
export ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY_MATS'
export RUNPOD_API_KEY='$RUNPOD_API_KEY'
export HF_DATASET='$HF_DATASET'
export HF_PREFIX='$HF_PREFIX'
export MODEL='$MODEL'
export SPRINT_HOURS='$SPRINT_HOURS'
export SPRINT_MINUTES='$SPRINT_MINUTES'
export BUDGET_USD='$BUDGET_USD'
export TURN_TIMEOUT='$TURN_TIMEOUT'
export RESTORE_PREFIX='$RESTORE_PREFIX'
export KICKOFF_ADDENDUM='$KICKOFF_ADDENDUM'
export EXTRA_RESUME_NOTE='$EXTRA_RESUME_NOTE'
export IS_SANDBOX=1
export PATH="\$HOME/.local/bin:\$PATH"

pip install --no-input -q "huggingface_hub>=0.23.0,<1.0" 2>&1 | tail -1
ship() { python3 - "\$1" "\$2" <<'PY' 2>/dev/null || true
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=sys.argv[1],
    path_in_repo=os.environ["HF_PREFIX"] + "/" + sys.argv[2],
    repo_id=os.environ["HF_DATASET"], repo_type="dataset",
    commit_message="sprint boot: " + sys.argv[2])
PY
}
( while true; do sleep 60; ship /workspace/supervisor.log supervisor.log; done ) & echo \$! > /tmp/bootstreamer.pid

request_terminate() {
    for i in 1 2 3 4 5; do
        curl -sS --max-time 20 -X POST -H "Authorization: Bearer \$RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"\$RUNPOD_POD_ID\\\"}) }\"}" \
            https://api.runpod.io/graphql || true
        sleep 10
    done
}
on_exit() {
    rc=\$?
    kill "\$(cat /tmp/bootstreamer.pid 2>/dev/null)" 2>/dev/null || true
    ship /workspace/supervisor.log supervisor.log
    if [ "\$rc" -eq 0 ] && [ -f /workspace/status_complete.json ]; then
        # sprint finished cleanly: request termination, then PARK.
        # PID 1 must NEVER exit on its own — that races RunPod's restart
        # policy and produces a restart loop (observed on smoke v1).
        request_terminate
        sleep infinity
    else
        # abnormal end: park for morning debugging (state + logs preserved)
        echo "{\"status\":\"aborted\",\"rc\":\$rc}" > /workspace/status_aborted.json
        ship /workspace/status_aborted.json status_aborted.json
        sleep infinity
    fi
}
trap on_exit EXIT

# ---- idempotent setup (container may restart) -------------------------------
command -v claude >/dev/null 2>&1 || curl -fsSL https://claude.ai/install.sh | bash
command -v uv >/dev/null 2>&1 || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d /workspace/fra_proj ] || GIT_LFS_SKIP_SMUDGE=1 git clone --depth 1 '$REPO_URL' /workspace/fra_proj
cd /workspace/fra_proj

# restore prior sprint work first (replacement-pod path), so the overlay
# below still wins for infra/papers while sprint/ work and toy outputs survive
if [ -n "${RESTORE_PREFIX:-}" ] && [ ! -f /workspace/.restore_done ]; then
    python3 - <<PY || echo "[sprint] no snapshot to restore (fresh start)"
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("dmanningcoe/sprint-fra-theory", "${RESTORE_PREFIX}/sprint_work.tar.gz",
                    repo_type="dataset", token=os.environ["HF_TOKEN"])
with tarfile.open(p) as t:
    t.extractall("/workspace/fra_proj/experiments")
print("[sprint] restored prior work from ${RESTORE_PREFIX}/sprint_work.tar.gz")
PY
    touch /workspace/.restore_done
fi

# sprint overlay (papers + reading notes + infra + fra_hmm_toy) from the
# PRIVATE dataset — this material is deliberately NOT in the public repo.
if [ ! -f /workspace/.overlay_done ]; then
    python3 - <<'PY'
import os, tarfile
from huggingface_hub import hf_hub_download
p = hf_hub_download("dmanningcoe/sprint-fra-theory", "grounding/sprint_overlay.tar.gz",
                    repo_type="dataset", token=os.environ["HF_TOKEN"])
with tarfile.open(p) as t:
    t.extractall("/workspace/fra_proj")
print("overlay extracted")
PY
    [ -f experiments/constrained_belief_updating/papers/READING_NOTES.md ] || { echo "[sprint] OVERLAY FAILED"; exit 1; }
    touch /workspace/.overlay_done
fi

[ -d .venv ] || \$HOME/.local/bin/uv venv .venv --python python3.11
.venv/bin/python -c 'import torch' 2>/dev/null || \
    timeout 2400 \$HOME/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt

echo "[sprint] setup done, handing off to supervisor \$(date -u +%FT%TZ)"
bash experiments/constrained_belief_updating/sprint_infra/supervisor.sh
EOF
}

deploy() {
    local inner; inner=$(make_inner)
    local b64; b64=$(printf '%s' "$inner" | base64 | tr -d '\n')
    local cmd_json; cmd_json=$(printf 'bash -c "echo %s | base64 -d > /start_user.sh && chmod +x /start_user.sh && /start.sh & sleep 20 && bash /start_user.sh"' "$b64" | json_str)
    local last_resp=""
    while IFS= read -r gpu_type; do
        [ -z "$gpu_type" ] && continue
        local input
        input=$(cat <<JSON
{
  "name": "$POD_NAME",
  "imageName": "$IMAGE",
  "cloudType": "SECURE",
  "gpuTypeId": "$gpu_type",
  "gpuCount": 1,
  "minVcpuCount": 8,
  "minMemoryInGb": 16,
  "containerDiskInGb": 60,
  "volumeInGb": 0,
  "dockerArgs": $cmd_json,
  "ports": "22/tcp",
  "startSsh": true
}
JSON
)
        local payload
        payload=$(python3 -c "
import json, sys
inp = json.loads(sys.argv[1])
q = '''mutation Deploy(\$input: PodFindAndDeployOnDemandInput!) {
  podFindAndDeployOnDemand(input: \$input) { id name desiredStatus }
}'''
print(json.dumps({'query': q, 'variables': {'input': inp}}))
" "$input")
        local resp; resp=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" -d "$payload" "$GRAPHQL")
        last_resp="$resp"
        local pid; pid=$(printf '%s' "$resp" | python3 -c "import json,sys; d=json.load(sys.stdin); print(((d.get('data') or {}).get('podFindAndDeployOnDemand') or {}).get('id') or '')")
        if [ -n "$pid" ]; then
            echo "    [$gpu_type] pod_id=$pid" >&2
            echo "$pid"; return 0
        fi
    done < <(printf '%s' "$BABY_GPU_IDS" | tr '|' '\n')
    echo "    FAILED on all GPU types. Last response:" >&2
    echo "    $last_resp" >&2
    return 1
}

# pre-check: duplicate-name guard (podFindAndDeployOnDemand does NOT refuse dupes)
EXISTING=$(curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d '{"query":"query { myself { pods { id name desiredStatus } } }"}' "$GRAPHQL" \
    | python3 -c "import json,sys; d=json.load(sys.stdin); print(' '.join(p['id'] for p in ((d.get('data') or {}).get('myself') or {}).get('pods', []) if p['name']=='$POD_NAME' and p['desiredStatus']=='RUNNING'))")
if [ -n "$EXISTING" ]; then
    echo "ABORT: a RUNNING pod named $POD_NAME already exists: $EXISTING"; exit 1
fi

echo "[sprint] launching $POD_NAME (main+overlay, model=$MODEL, ${SPRINT_HOURS}h, cap \$$BUDGET_USD) ..."
PID=$(deploy) || { echo "ABORT"; exit 1; }
echo
echo "============================================================"
echo "[sprint] pod = $PID   LOCAL CAN CLOSE NOW."
echo "watch:  https://huggingface.co/datasets/$HF_DATASET/tree/main/$HF_PREFIX"
echo "  supervisor.log   — live log (60s lag)"
echo "  deliverables/    — summary.md, notes/*.tex, figures (15 min lag)"
echo "  sprint.bundle    — full git history; recover with: git pull sprint.bundle"
echo "  status_*.json    — complete / aborted / stalled markers"
echo "============================================================"
