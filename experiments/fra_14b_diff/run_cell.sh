#!/usr/bin/env bash
# Unified pod-side cell runner for the YAML-driven campaign. Replaces
# run_grid_diff.sh / run_frarouting_diff.sh: reads ONE base64 cell-spec from the
# pod env ($CELL_SPEC_B64), runs the cell end-to-end via cell_runner.py
# (rank → generate → JUDGE POD-SIDE → upload), and self-terminates on success.
#
# Pod env (injected by the launcher's `env` field — NOT in the logged dockerArgs):
#   HF_TOKEN, RUNPOD_API_KEY, OPENAI_API_KEY (or OPENAI_API_KEY_MATS),
#   CELL_SPEC_B64, BRANCH   (RUNPOD_POD_ID is auto-set by RunPod)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
BRANCH="${BRANCH:-autoresearch/cadenza-attn-only}"

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
}
on_err() {
    echo "[$(date -u +%H:%M:%S)] [FAIL] line ${BASH_LINENO[0]} — keeping pod ALIVE for triage"
    nvidia-smi 2>/dev/null | tail -12 || true
    sleep infinity
}
trap on_err ERR

# ── driver gate ──
DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break; sleep 5
done
[ -z "$DRIVER_MAJOR" ] && { echo "no driver — abort"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver too old ($DRIVER_MAJOR) — self-term"; terminate_self; exit 0; }

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] git HEAD: $(git rev-parse HEAD)"

export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 PIP_CACHE_DIR=/workspace/.pip_cache
export HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export OPENAI_API_KEY="${OPENAI_API_KEY:-${OPENAI_API_KEY_MATS:-}}"
mkdir -p "$PIP_CACHE_DIR"
# TORCH RECIPE (FIXED — CAMPAIGN.md §5): torch 2.6.0+cu124 WITH deps + force pandas
# + dictionary_learning + peft + openai. Do NOT regress to 2.4.1.
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft openai 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

# ── decode the cell spec (not secret; keys are separate env) ──
echo "$CELL_SPEC_B64" | base64 -d > /workspace/cell.json
echo "[$(date -u +%H:%M:%S)] cell: $(python3 -c "import json;c=json.load(open('/workspace/cell.json'));print(c['cell_prefix'],c['diff_mode'],'em='+','.join(c['em_keys']))")"

# (don't echo the OpenAI key)
set +x
python3 -u experiments/fra_14b_diff/cell_runner.py /workspace/cell.json

echo "[$(date -u +%H:%M:%S)] === CELL DONE — self-terminate ==="
trap - ERR
terminate_self
