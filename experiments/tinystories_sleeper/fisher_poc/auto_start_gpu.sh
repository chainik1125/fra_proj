#!/usr/bin/env bash
# Container entrypoint for a Fisher-POC GPU pod.
# Run on the pod as docker CMD or via runpodctl. Expects env vars:
#   HF_TOKEN   - HF write token
#   SEEDS      - space-separated seed list (default "0")
#   HF_REPO    - target HF dataset repo (default dmanningcoe/fisher-poc-tinystories-sleeper)
#   BRANCH     - git branch to clone (default dmitry/fisher-poc)
#   REPO_URL   - git remote URL
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

bash experiments/tinystories_sleeper/fisher_poc/setup_pod.sh
bash experiments/tinystories_sleeper/fisher_poc/run_on_pod.sh
