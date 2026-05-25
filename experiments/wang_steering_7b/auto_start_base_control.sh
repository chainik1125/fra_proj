#!/usr/bin/env bash
# Base-model control for the diff-cossim "δ=35" effect.
#
# Steers ~25 features (top-20 diff-cossim + F53258 + 4 random directions) on
# the BASE Qwen-2.5-7B-Instruct (NOT emergently misaligned) at the same ±90
# effective-α grid as the EM screen. Question: does base ALSO produce a large
# Δcoh swing — purely by collapsing coherence into low-alignment babble at
# huge α — or does it stay coherent/aligned? If base swings too (esp. the
# RANDOM features), the effect is generic large-α coherence collapse, not
# EM-specific. Includes random directions as the strongest control.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH,
#               FEATURE_IDS (space-sep), EM_MODEL, EVAL_SEEDS, SAMPLES_PER_PROMPT, HF_PREFIX
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

EM_MODEL="${EM_MODEL:-base}"
EVAL_SEEDS="${EVAL_SEEDS:-42 123 456}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-1}"
HF_PREFIX="${HF_PREFIX:-qwen7b/base_diffcossim_control_n8}"
FEATURE_IDS="${FEATURE_IDS:?FEATURE_IDS not set}"
ALPHAS="-90.86 -79.5 -68.14 -56.79 -45.43 -34.07 -22.71 -11.36 0 11.36 22.71 34.07 45.43 56.79 68.14 79.5 90.86"

echo "[$(date -u +%H:%M:%S)] START base-control em=$EM_MODEL seeds=[$EVAL_SEEDS] spp=$SAMPLES_PER_PROMPT nfeat=$(echo $FEATURE_IDS | wc -w)"
echo "[$(date -u +%H:%M:%S)] driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
}

DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break
    echo "[$(date -u +%H:%M:%S)] nvidia-smi not ready ($i/6) — sleep 5"; sleep 5
done
[ -z "$DRIVER_MAJOR" ] && { echo "no driver — abort (reschedule)"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver $DRIVER_MAJOR <525 — self-term"; terminate_self; exit 0; }
echo "[$(date -u +%H:%M:%S)] driver major=$DRIVER_MAJOR — proceed (cu124)"

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] git HEAD: $(git rev-parse HEAD)"

export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 PIP_CACHE_DIR=/workspace/.pip_cache
mkdir -p "$PIP_CACHE_DIR"
echo "[$(date -u +%H:%M:%S)] pip installs"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(); print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

for SEED in $EVAL_SEEDS; do
    HF_API="https://huggingface.co/api/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/$HF_PREFIX/${EM_MODEL}_seed${SEED}"
    if curl -sS -H "Authorization: Bearer $HF_TOKEN" "$HF_API" 2>/dev/null | grep -q "qualitative_arditi_${EM_MODEL}_evalseed${SEED}.json"; then
        echo "[$(date -u +%H:%M:%S)] seed $SEED already on HF — skip"; continue
    fi
    OUT="/workspace/basectl_${EM_MODEL}_seed${SEED}"; mkdir -p "$OUT"
    echo "[$(date -u +%H:%M:%S)] === sweep ($EM_MODEL, seed=$SEED) ==="
    python3 -u phase1_arditi_orchestrator.py \
        --em-model "$EM_MODEL" --eval-seed "$SEED" \
        --samples-per-prompt "$SAMPLES_PER_PROMPT" \
        --layer 15 --trainer 1 \
        --feature-ids $FEATURE_IDS \
        --alphas $ALPHAS \
        --max-new-tokens 200 \
        --output-root "$OUT"
    echo "[$(date -u +%H:%M:%S)] === upload seed $SEED ==="
    python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api = HfApi()
for f in pathlib.Path('$OUT').glob('*.json'):
    target = f'$HF_PREFIX/${EM_MODEL}_seed${SEED}/{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=target,
                    repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                    commit_message='base diff-cossim control n8 (${EM_MODEL}, seed${SEED})')
    print(f'  → {target}', flush=True)
"
done
echo "[$(date -u +%H:%M:%S)] === DONE — self-terminate ==="
terminate_self
