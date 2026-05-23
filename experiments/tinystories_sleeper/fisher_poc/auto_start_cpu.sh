#!/usr/bin/env bash
# Container entrypoint for the Fisher-POC babysitter CPU pod.
# Expects env vars:
#   HF_TOKEN   - HF token (read + write)
#   SEEDS      - space-separated seed list to wait for (default "0 1 2 3 4")
#   HF_REPO    - target HF dataset repo
#   BRANCH     - git branch (default dmitry/fisher-poc)
#   REPO_URL   - git remote URL
#   POLL_SEC, STALL_TIMEOUT_SEC - babysitter knobs
#   SELF_STOP  - 1 to runpodctl stop pod at the end
set -euo pipefail

BRANCH="${BRANCH:-dmitry/fisher-poc}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"

mkdir -p /workspace
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
fi
cd "$WORKDIR"

bash experiments/tinystories_sleeper/fisher_poc/setup_cpu_pod.sh
python experiments/tinystories_sleeper/fisher_poc/babysitter.py
