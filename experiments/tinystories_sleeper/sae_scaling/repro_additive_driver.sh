#!/usr/bin/env bash
# Additive-operation control for the single-feature repro: top-20 attribution
# features per cell (OV ln1 + conv resid_mid), each α-swept with a fixed
# −α·f delta. See scripts/repro_additive_cells.py.
# Required env: HF_TOKEN. Optional: HF_REPO BRANCH SELF_STOP.
set -uo pipefail
HF_REPO="${HF_REPO:-dmanningcoe/sae-scaling-tinystories-sleeper}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
export HF_TOKEN HF_REPO

echo "[repro] $(date +%H:%M:%S) start driver=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

if [ ! -d /workspace/fra_proj/.git ]; then
  apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
fi
cd /workspace/fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only

# env = the repo's lockfile (see reference-runpod-torch-env RULE ZERO)
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
uv sync --frozen 2>&1 | tail -3
RUN="uv run --no-sync"
$RUN python -c "import torch,transformer_lens;print('[repro] env ok torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || { echo "[repro] ENV BROKEN — abort"; exit 1; }

mkdir -p results
echo "[repro] $(date +%H:%M:%S) START additive_cells"
$RUN python -u -m scripts.repro_additive_cells --out results/repro_additive_cells.json \
    > /workspace/additive_cells.log 2>&1
rc=$?
echo "[repro] $(date +%H:%M:%S) DONE  additive_cells rc=$rc"
tail -5 /workspace/additive_cells.log

$RUN python - <<'PYEOF'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("HF_REPO", "dmanningcoe/sae-scaling-tinystories-sleeper")
for f in sorted(glob.glob("results/repro_additive_*.json")):
    api.upload_file(path_or_fileobj=f, path_in_repo=f"repro/{os.path.basename(f)}",
                    repo_id=repo, repo_type="dataset")
    print("[repro] uploaded", f, flush=True)
for f in sorted(glob.glob("/workspace/*.log")):
    try:
        api.upload_file(path_or_fileobj=f, path_in_repo=f"repro/logs/{os.path.basename(f)}",
                        repo_id=repo, repo_type="dataset")
        print("[repro] uploaded log", f, flush=True)
    except Exception as e:
        print("[repro] log upload failed", f, e, flush=True)
PYEOF

echo "[repro] $(date +%H:%M:%S) ALL DONE rc=$rc"

if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" --data-binary \
    "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql
fi
