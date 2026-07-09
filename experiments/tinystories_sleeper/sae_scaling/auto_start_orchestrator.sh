#!/usr/bin/env bash
# Remote orchestrator bootstrap. Installs Claude Code CLI, clones the repo, then runs
# `claude` headless with CAMPAIGN_headroom.md to drive the headroom experiments to
# completion autonomously, launching GPU worker pods via the RunPod API. Self-terminates.
#
# Required env (set by launch_orchestrator.sh):
#   ANTHROPIC_API_KEY  — drives this Claude (set to the MATS-billed key by the launcher)
#   HF_TOKEN           — HF read/write
#   RP_API_KEY_MATS    — RunPod API (also exported as RUNPOD_API_KEY) for launching/killing pods
#   RUNPOD_POD_ID      — set by RunPod; used for self-stop
#   BRANCH             — git branch (default dmitry/sae-scaling-sweep)
#   MAX_RUN_SEC        — hard wall-clock kill for the claude session (budget guard)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/orchestrator.log) 2>&1
echo "[$(date -u +%H:%M:%S)] orchestrator starting"

BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
MAX_RUN_SEC="${MAX_RUN_SEC:-28800}"   # 8h default
export RUNPOD_API_KEY="${RP_API_KEY_MATS:?}"
export HF_TOKEN="${HF_TOKEN:?}"
: "${ANTHROPIC_API_KEY:?}"

apt-get update -qq
apt-get install -y -qq git curl ca-certificates jq python3 python3-pip >/dev/null
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
apt-get install -y -qq nodejs >/dev/null
npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
echo "[$(date -u +%H:%M:%S)] claude --version: $(claude --version 2>&1 | head -1)"
pip install --no-input --break-system-packages huggingface_hub requests 2>&1 | tail -2

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only || true
git config --global user.email "claude-orchestrator@anthropic.com"
git config --global user.name "Headroom Orchestrator (Claude)"

BRIEF=$(cat experiments/tinystories_sleeper/sae_scaling/CAMPAIGN_headroom.md)
KICKOFF="You are the autonomous remote orchestrator. The campaign brief follows — read it
plus the two specs it points to, take an inventory of what's already on HF
(headroom_results/exp6, exp11), then drive the HIGH-PRIORITY subset (finish 11+6, then 7,3,5,8)
to completion. Persist all results + a cumulative REMOTE_FINDINGS.md to HF (no git push creds).
Slack-DM milestones. Respect every fence + the \$300 budget. Self-terminate when done.

---

$BRIEF"

echo "[$(date -u +%H:%M:%S)] handing off to claude (max_run ${MAX_RUN_SEC}s)"
# IS_SANDBOX=1 lets --dangerously-skip-permissions run as root in the container.
# timeout is the budget backstop; DON'T exec — we self-terminate cleanly after.
timeout "${MAX_RUN_SEC}" env IS_SANDBOX=1 claude \
    --dangerously-skip-permissions \
    --print \
    --output-format text \
    --max-turns 1000 \
    "$KICKOFF" 2>&1 | tee /workspace/claude_output.log || true
echo "[$(date -u +%H:%M:%S)] claude session ended"

echo "[$(date -u +%H:%M:%S)] self-terminating orchestrator pod"
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql || true
sleep 30
