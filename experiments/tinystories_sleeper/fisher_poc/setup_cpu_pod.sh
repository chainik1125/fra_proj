#!/usr/bin/env bash
# Setup script for the babysitter CPU pod.
# Smaller than setup_pod.sh: no torch / CUDA needed, only huggingface_hub.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
BRANCH="${BRANCH:-dmitry/fisher-poc}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"

echo "=== Fisher POC: babysitter CPU pod setup ==="

if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
else
  cd "$WORKDIR"
  git fetch --depth 1 origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"
fi
cd "$WORKDIR"

pip install -q "huggingface_hub>=0.24" 2>&1 | tail -3

if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — babysitter will fail to read/write the dataset repo."
fi

echo
echo "=== Setup complete. Run babysitter with: ==="
echo "  SEEDS=\"0 1 2 3 4\" \\"
echo "  HF_REPO=dmanningcoe/fisher-poc-tinystories-sleeper \\"
echo "  python experiments/tinystories_sleeper/fisher_poc/babysitter.py"
