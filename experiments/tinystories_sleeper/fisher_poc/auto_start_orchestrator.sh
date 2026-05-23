#!/usr/bin/env bash
# Orchestrator pod entrypoint.
#
# Installs Node.js + Claude Code, clones the branch, and runs the
# headless agent with the orchestrator brief.  The agent then drives
# the campaign to completion (kicking off bootstrap + experiment pods,
# debugging failures, writing summary.md, self-terminating).
#
# Env vars consumed:
#   HF_TOKEN, RUNPOD_API_KEY, ANTHROPIC_API_KEY  (passed to the agent)
#   SEEDS, HF_REPO, BRANCH, REPO_URL
set -eo pipefail

# Stream logs unbuffered so we can `runpodctl pod logs` while it works.
mkdir -p /workspace
exec > >(stdbuf -oL tee /workspace/orchestrator.log) 2>&1
echo "[$(date +%H:%M:%S)] orchestrator pod start"

BRANCH="${BRANCH:-dmitry/fisher-poc}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"

# 1) Make sure git/curl/node are present.
apt-get update -qq
apt-get install -y -q git curl ca-certificates >/dev/null
# Node.js 22 LTS via NodeSource.
if ! command -v node >/dev/null 2>&1; then
  curl -fsSL https://deb.nodesource.com/setup_22.x | bash -
  apt-get install -y -q nodejs >/dev/null
fi
node -v
npm -v

# 2) Install Claude Code CLI.
npm install -g @anthropic-ai/claude-code >/dev/null 2>&1 || \
  npm install -g @anthropic-ai/claude-code
echo "claude version: $(claude --version 2>&1 || echo MISSING)"

# 3) Clone the branch (the brief lives in it).
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

# 4) Make sure the agent has the huggingface_hub library available.
apt-get install -y -q python3-pip >/dev/null
pip install --quiet "huggingface_hub>=0.24" >/dev/null

BRIEF_PATH="$WORKDIR/experiments/tinystories_sleeper/fisher_poc/orchestrator_brief.md"
echo "[$(date +%H:%M:%S)] launching claude with brief: $BRIEF_PATH"

# 5) Hand the brief to claude in non-interactive mode.  --max-turns is
#    set high since the campaign is multi-hour and the agent will issue
#    many bash poll-and-react turns.  --dangerously-skip-permissions is
#    required for headless: no human is around to approve tool calls.
#    Inherit env so the agent has HF_TOKEN / RUNPOD_API_KEY / etc.
claude --print \
  --dangerously-skip-permissions \
  --max-turns 600 \
  "$(cat "$BRIEF_PATH")" \
  2>&1 | tee -a /workspace/orchestrator.log

echo "[$(date +%H:%M:%S)] claude exited"

# 6) Belt-and-braces self-terminate (the brief instructs the agent to
#    self-terminate, but if it didn't, do it now anyway).
if [[ -n "${RUNPOD_POD_ID:-}" && -n "${RUNPOD_API_KEY:-}" ]]; then
  echo "[orchestrator] self-terminating pod $RUNPOD_POD_ID"
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql || true
fi
