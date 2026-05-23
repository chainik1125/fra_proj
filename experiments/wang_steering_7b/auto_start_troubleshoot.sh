#!/usr/bin/env bash
# Troubleshooter pod bootstrap. Installs Claude Code CLI + cli deps, clones
# the repo, then runs `claude` headless with the troubleshoot brief.
#
# Required env (set via launch_troubleshoot.sh):
#   ANTHROPIC_API_KEY   — to drive Claude
#   RUNPOD_API_KEY      — for pod mgmt
#   HF_TOKEN            — HF read/write
#   RUNPOD_POD_ID       — set by RunPod; used for self-stop
#   BRANCH              — git branch (default autoresearch/wang-steering-7b)
#
# Optional:
#   POD_PRIVATE_KEY_B64 — base64'd ed25519 private key for SSH to other pods.
#                         If unset, the troubleshooter can't SSH-debug other
#                         pods but can still inspect HF + redispatch.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/troubleshoot.log) 2>&1
echo "[$(date -u +%H:%M:%S)] troubleshooter starting"

BRANCH="${BRANCH:-autoresearch/wang-steering-7b}"

apt-get update -qq
apt-get install -y -qq git curl ca-certificates openssh-client jq python3 python3-pip >/dev/null
# Node 22 for Claude Code.
curl -fsSL https://deb.nodesource.com/setup_22.x | bash - >/dev/null
apt-get install -y -qq nodejs >/dev/null
npm install -g @anthropic-ai/claude-code 2>&1 | tail -3
echo "[$(date -u +%H:%M:%S)] claude --version: $(claude --version 2>&1 | head -1)"

# Python deps the troubleshooter will reach for.
pip install --no-input --break-system-packages huggingface_hub requests 2>&1 | tail -2

# SSH key (optional).
mkdir -p /root/.ssh
chmod 700 /root/.ssh
if [ -n "${POD_PRIVATE_KEY_B64:-}" ]; then
    echo "$POD_PRIVATE_KEY_B64" | base64 -d > /root/.ssh/id_ed25519
    chmod 600 /root/.ssh/id_ed25519
    echo "[$(date -u +%H:%M:%S)] ssh private key installed"
fi
cat > /root/.ssh/config <<SSHC
Host *
    StrictHostKeyChecking no
    UserKnownHostsFile /dev/null
    ServerAliveInterval 30
    ConnectTimeout 10
SSHC

# Repo.
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only

# Identify ourselves to git in case the troubleshooter pushes a patch.
git config --global user.email "claude-troubleshooter@anthropic.com"
git config --global user.name "Wang Steering Troubleshooter (Claude)"

# Compose the launch prompt. Brief + a kick-off instruction.
BRIEF=$(cat experiments/wang_steering_7b/troubleshoot_brief.md)
KICKOFF="The brief follows. Read it, run an inventory of the current state, and drive the campaign to completion. Report progress in the working directory as troubleshooter_log.md and update it as you work. Self-stop when 6/6 are on HF (or escalate via status_giving_up.md if stuck).

---

$BRIEF"

# Hand off to Claude Code, headless, with full skips. DON'T exec — we want
# to self-terminate cleanly after claude exits, so RunPod doesn't restart
# us in a loop.
echo "[$(date -u +%H:%M:%S)] handing off to claude"
claude \
    --dangerously-skip-permissions \
    --print \
    --output-format text \
    --max-turns 200 \
    "$KICKOFF" 2>&1 | tee /workspace/claude_output.log
CLAUDE_RC=$?
echo "[$(date -u +%H:%M:%S)] claude exited with code $CLAUDE_RC"

# Self-terminate so the user isn't billed for an idle pod.
echo "[$(date -u +%H:%M:%S)] self-terminating troubleshooter pod"
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql || true
sleep 30  # give RunPod time to actually kill us before bash exits
exit "$CLAUDE_RC"
