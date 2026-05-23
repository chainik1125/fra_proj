#!/usr/bin/env bash
# Run this ON the RunPod pod after SSH-ing in.
#
# Expects:
#   - A RunPod PyTorch template (CUDA already installed)
#   - $HF_TOKEN set in the environment (for `huggingface-cli login`)
#   - L40S (48GB) or A40 (48GB) is plenty — TinyStories-33M only needs a few GB.
#     A 24GB card (e.g., RTX 4090) also works.
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
BRANCH="${BRANCH:-dmitry/fisher-poc}"
WORKDIR="${WORKDIR:-/workspace/fra_proj}"

echo "=== Fisher POC: pod setup ==="
echo "branch=$BRANCH workdir=$WORKDIR"

# Clone (or refresh).
if [[ ! -d "$WORKDIR/.git" ]]; then
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" "$WORKDIR"
else
  cd "$WORKDIR"
  git fetch --depth 1 origin "$BRANCH"
  git checkout "$BRANCH"
  git reset --hard "origin/$BRANCH"
fi
cd "$WORKDIR"

# Install uv if missing, then sync.
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  # uv installs to ~/.cargo/bin or ~/.local/bin depending on version
  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
fi
uv sync 2>&1 | tail -20

# pyproject.toml has unpinned `torch`, so `uv sync` grabs the latest
# (2.11 as of 2026-05-22), which is compiled against CUDA 13.x. RunPod's
# stock L40S driver is 570.124.06 = CUDA 12.8 — too old for cu13 torch.
# Force-install the cu124 build so the wheel actually loads.
echo "[setup_pod] reinstalling torch from cu124 wheels (driver compat)"
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -10

# HF login (required for the sleeper QLoRA adapter and the base model).
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "WARN: HF_TOKEN is not set — model download may fail."
else
  uv run huggingface-cli login --token "$HF_TOKEN" >/dev/null
fi

# Verify GPU.
uv run python -c "import torch; \
  print(f'GPU: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB'); \
  assert torch.cuda.is_available()"

echo
echo "=== Setup complete. Run experiments with: ==="
echo "  bash experiments/tinystories_sleeper/fisher_poc/run_on_pod.sh"
