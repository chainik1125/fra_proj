#!/usr/bin/env bash
# Train an Arditi-style BatchTopK SAE at the ln1 hookpoint on BASE Qwen-2.5-7B.
#
# andyrdt's published SAEs are resid_post; the FRA (QK→QK) protocol operates at
# ln1.hook_normalized, so we need an ln1 SAE. This runs Arditi's training code
# byte-for-byte (via fra/train_sae_arditi.py, which clones andyrdt/dictionary_learning
# @ andyrdt/qwen and patches run_from_config.py to honor submodule_path_override).
#
# Params match Arditi's trainer_1 (k=64, d_sae=131072, BatchTopK) but slimmed to
# ~100M tokens (vs his 500M) for a faster, still-usable SAE.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH
# Pod self-terminates on success; stays alive on failure for SSH triage.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

NUM_TOKENS="${NUM_TOKENS:-100000000}"
HOOK_LAYER="${HOOK_LAYER:-15}"
HF_PREFIX="${HF_PREFIX:-qwen7b/sae_ln1_l15_base_arditi}"
OUT="/workspace/sae_ln1_l${HOOK_LAYER}_base"

echo "[$(date -u +%H:%M:%S)] START SAE train — ln1 L${HOOK_LAYER} base, ${NUM_TOKENS} tokens"
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
[ -z "$DRIVER_MAJOR" ] && { echo "no driver — abort"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver $DRIVER_MAJOR <525 — self-term"; terminate_self; exit 0; }
echo "[$(date -u +%H:%M:%S)] driver major=$DRIVER_MAJOR — proceed (cu124)"

# HF pre-check: skip if SAE already trained+uploaded
HF_API="https://huggingface.co/api/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/$HF_PREFIX"
if curl -sS -H "Authorization: Bearer $HF_TOKEN" "$HF_API" 2>/dev/null | grep -q "ae.pt"; then
    echo "[$(date -u +%H:%M:%S)] SAE already on HF — self-term"; terminate_self; exit 0
fi

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] git HEAD: $(git rev-parse HEAD)"

export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 PIP_CACHE_DIR=/workspace/.pip_cache HF_TOKEN="$HF_TOKEN"
mkdir -p "$PIP_CACHE_DIR"
echo "[$(date -u +%H:%M:%S)] pip installs"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages datasets transformers 'huggingface_hub' 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(); print(f'torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}')"

echo "[$(date -u +%H:%M:%S)] === TRAIN: ln1 L${HOOK_LAYER} SAE (Arditi code) ==="
python3 -u fra/train_sae_arditi.py \
    --hook-layer "$HOOK_LAYER" \
    --submodule-name input_layernorm \
    --model-name "Qwen/Qwen2.5-7B-Instruct" \
    --num-tokens "$NUM_TOKENS" \
    --target-l0s 64 \
    --dictionary-widths 131072 \
    --architectures batch_top_k \
    --no-wandb \
    --output-dir "$OUT"

echo "[$(date -u +%H:%M:%S)] === train done; SAE dir contents ==="
ls -laR "$OUT" | head -40

echo "[$(date -u +%H:%M:%S)] === upload SAE → HF $HF_PREFIX ==="
python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api=HfApi()
root=pathlib.Path('$OUT')
# upload the whole trained-SAE tree (config.json, ae.pt, eval_results.json, trainer_*/...)
for f in root.rglob('*'):
    if f.is_file():
        rel=f.relative_to(root)
        api.upload_file(path_or_fileobj=str(f), path_in_repo=f'$HF_PREFIX/{rel}',
                        repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                        commit_message='ln1 L${HOOK_LAYER} base SAE (Arditi BatchTopK k64, ${NUM_TOKENS} tok)')
        print('  →', f'$HF_PREFIX/{rel}', flush=True)
"
echo "[$(date -u +%H:%M:%S)] === DONE — self-terminate ==="
terminate_self
