#!/usr/bin/env bash
# Smoke gate v2 for the overnight FRA-theory sprint pod.
#
# Launches ONE cheap orchestrator pod (GPU-class; computeType: CPU
# supply-constrains on this account) that verifies every ingredient the 10h
# sprint needs, uploads smoke_report.json + smoke.log to HF, requests
# self-termination and PARKS (never exits PID 1 → no restart loop).
#
# Verifies on-pod:
#   1. public HTTPS clone of the repo
#   2. Claude Code CLI native install
#   3. headless `claude -p` on the MATS key, model claude-fable-5 (+ cost field)
#   4. `--resume <session_id>` continuity (the supervisor's core mechanism)
#   5. Agent-tool subagent spawn inside a headless session (the team mechanism)
#   6. uv venv + `uv pip install -r requirements.txt` (the repo's real ML env)
#   7. torch + transformer_lens import AND a tiny HookedTransformer forward pass
#   8. HF write round-trip
#
# Required env: RP_API_KEY_MATS, ANTHROPIC_API_KEY_MATS, HF_TOKEN
# Watch:  https://huggingface.co/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/sprint_fra_theory/smoke2
set -euo pipefail

: "${RP_API_KEY_MATS:?RP_API_KEY_MATS not set}"
: "${ANTHROPIC_API_KEY_MATS:?ANTHROPIC_API_KEY_MATS not set}"
: "${HF_TOKEN:?HF_TOKEN not set}"

RUNPOD_API_KEY="$RP_API_KEY_MATS"
BRANCH="${BRANCH:-main}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
POD_NAME="rs-sprint-smoke2"
BABY_GPU_IDS="${BABY_GPU_IDS:-NVIDIA RTX A4000|NVIDIA RTX A5000|NVIDIA L4|NVIDIA L40S}"
IMAGE="${IMAGE:-runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04}"
HF_DATASET="dmanningcoe/fra-phase1-steering-data"
HF_PREFIX="sprint_fra_theory/smoke2"

GRAPHQL="https://api.runpod.io/graphql"
json_str() { python3 -c "import json,sys; print(json.dumps(sys.stdin.read()))"; }

make_inner() {
    cat <<EOF
#!/bin/bash
set -o pipefail
rm -f /workspace/steps.txt /workspace/smoke.log
exec > >(stdbuf -oL tee /workspace/smoke.log) 2>&1
echo "[smoke] boot \$(date -u +%FT%TZ) pod=\$RUNPOD_POD_ID"

export HF_TOKEN='$HF_TOKEN'
export ANTHROPIC_API_KEY='$ANTHROPIC_API_KEY_MATS'
export RUNPOD_API_KEY='$RUNPOD_API_KEY'
export HF_DATASET='$HF_DATASET'
export HF_PREFIX='$HF_PREFIX'
export IS_SANDBOX=1

pip install --no-input -q "huggingface_hub>=0.23.0,<1.0" 2>&1 | tail -1

ship() { python3 - "\$1" "\$2" <<'PY' 2>/dev/null || true
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=sys.argv[1],
    path_in_repo=os.environ["HF_PREFIX"] + "/" + sys.argv[2],
    repo_id=os.environ["HF_DATASET"], repo_type="dataset",
    commit_message="smoke: " + sys.argv[2])
PY
}
( while true; do sleep 30; ship /workspace/smoke.log smoke.log; done ) & echo \$! > /tmp/streamer.pid

request_terminate() {
    for i in 1 2 3 4 5; do
        curl -sS --max-time 20 -X POST -H "Authorization: Bearer \$RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"\$RUNPOD_POD_ID\\\"}) }\"}" \
            https://api.runpod.io/graphql || true
        sleep 10
    done
}
finish() {
    kill "\$(cat /tmp/streamer.pid 2>/dev/null)" 2>/dev/null || true
    python3 - <<PYEOF
import json, os
lines = [l.strip() for l in open("/workspace/steps.txt")] if os.path.exists("/workspace/steps.txt") else []
json.dump({"failed": \$FAILED, "steps": lines}, open("/workspace/smoke_report.json", "w"), indent=1)
PYEOF
    ship /workspace/smoke_report.json smoke_report.json
    ship /workspace/smoke.log smoke.log
    # request termination, then PARK — PID 1 must never exit (restart-loop race)
    request_terminate
    sleep infinity
}

FAILED=0
step() {
    local name="\$1"; shift
    local t0=\$(date +%s)
    if "\$@" >> /workspace/smoke.log 2>&1; then
        echo "\$name ok \$(( \$(date +%s) - t0 ))s" >> /workspace/steps.txt
        echo "[smoke] \$name OK (\$(( \$(date +%s) - t0 ))s)"
    else
        echo "\$name FAIL \$(( \$(date +%s) - t0 ))s" >> /workspace/steps.txt
        echo "[smoke] \$name FAIL"
        FAILED=1
    fi
}

rm -rf /workspace/fra_proj
step git_clone git clone --depth 1 --branch '$BRANCH' '$REPO_URL' /workspace/fra_proj
step claude_install bash -c 'curl -fsSL https://claude.ai/install.sh | bash'
export PATH="\$HOME/.local/bin:\$PATH"
step claude_version claude --version

# headless turn 1: fixed session id, json output, cost field
SID=\$(python3 -c "import uuid; print(uuid.uuid4())")
step claude_turn1 bash -c "cd /workspace && timeout 240 claude -p 'Remember this fruit: BANANA. Reply with exactly SMOKE_OK.' \
    --model claude-fable-5 --session-id \$SID --output-format json --dangerously-skip-permissions \
    > /workspace/claude_t1.json 2>&1 && grep -q SMOKE_OK /workspace/claude_t1.json"
grep -o '"total_cost_usd":[0-9.e-]*' /workspace/claude_t1.json >> /workspace/steps.txt 2>/dev/null || true

# resume: the supervisor's core mechanism — same session, context intact
RSID=\$(python3 -c "
import json
try: print(json.load(open('/workspace/claude_t1.json')).get('session_id') or '')
except Exception: print('')
")
[ -z "\$RSID" ] && RSID="\$SID"
step claude_resume bash -c "cd /workspace && timeout 240 claude -p 'What fruit did I ask you to remember? Reply with just the fruit name.' \
    --resume \$RSID --model claude-fable-5 --output-format json --dangerously-skip-permissions \
    > /workspace/claude_t2.json 2>&1 && grep -qi BANANA /workspace/claude_t2.json"

# Agent tool: the in-session team mechanism
step claude_agent_tool bash -c "cd /workspace/fra_proj && timeout 420 claude -p 'Use the Agent tool to spawn one general-purpose subagent whose task is: reply with the word PINEAPPLE. After it returns, output exactly AGENT_OK followed by the word the subagent returned.' \
    --model claude-fable-5 --output-format json --dangerously-skip-permissions \
    > /workspace/claude_t3.json 2>&1 && grep -q AGENT_OK /workspace/claude_t3.json && grep -q PINEAPPLE /workspace/claude_t3.json"

# the repo's real ML env: requirements.txt into a uv venv
step uv_install bash -c 'curl -LsSf https://astral.sh/uv/install.sh | sh'
step uv_venv bash -c 'cd /workspace/fra_proj && \$HOME/.local/bin/uv venv .venv --python python3.11'
step pip_requirements bash -c 'cd /workspace/fra_proj && timeout 2400 \$HOME/.local/bin/uv pip install --python .venv/bin/python -r requirements.txt'
step venv_imports bash -c 'cd /workspace/fra_proj && .venv/bin/python -c "import torch, transformer_lens; print(\"torch\", torch.__version__, \"cuda\", torch.cuda.is_available())"'
step tl_forward bash -c 'cd /workspace/fra_proj && timeout 300 .venv/bin/python -c "
from transformer_lens import HookedTransformer, HookedTransformerConfig
import torch
cfg = HookedTransformerConfig(n_layers=2, d_model=32, n_ctx=16, d_head=16, n_heads=2, d_mlp=64, d_vocab=9, act_fn=\"gelu\")
m = HookedTransformer(cfg)
x = torch.randint(0, 9, (2, 16))
loss = m(x, return_type=\"loss\")
loss.backward()
print(\"tl forward/backward ok\", float(loss))
"'
step hf_roundtrip python3 -c "
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ['HF_TOKEN'])
open('/tmp/rt.txt','w').write('roundtrip')
api.upload_file(path_or_fileobj='/tmp/rt.txt', path_in_repo=os.environ['HF_PREFIX']+'/roundtrip.txt', repo_id=os.environ['HF_DATASET'], repo_type='dataset')
print('hf write ok')
"

echo "[smoke] done FAILED=\$FAILED \$(date -u +%FT%TZ)"
finish
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

echo "[smoke] launching $POD_NAME ..."
PID=$(deploy) || { echo "ABORT"; exit 1; }
echo "[smoke] pod = $PID"
echo "[smoke] watch: https://huggingface.co/datasets/$HF_DATASET/tree/main/$HF_PREFIX"
