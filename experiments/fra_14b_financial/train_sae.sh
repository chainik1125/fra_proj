#!/usr/bin/env bash
# Train ONE Arditi-style BatchTopK SAE on BASE Qwen-2.5-14B-Instruct at L24.
# Parametrized by SAE_KIND: "resid_post" (Arditi native default) or "ln1"
# (input_layernorm override, like the 7B run). One pod per SAE → parallelize.
#
# Scaled down from Arditi defaults for speed: single L0 (k=64, the only one we
# use) + 200M tokens (vs 500M, 4 L0s) → ~4-6h on one H100 instead of ~20h.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH, SAE_KIND
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

SAE_KIND="${SAE_KIND:?set SAE_KIND=resid_post|ln1}"
HOOK_LAYER="${HOOK_LAYER:-24}"
NUM_TOKENS="${NUM_TOKENS:-200000000}"
MODEL_NAME="${MODEL_NAME:-Qwen/Qwen2.5-14B-Instruct}"
HF_PREFIX="${HF_PREFIX:-qwen14b/sae_${SAE_KIND}_l${HOOK_LAYER}_base_arditi}"
OUT="/workspace/sae_${SAE_KIND}_l${HOOK_LAYER}_base"
# ln1 needs the post-gain input_layernorm submodule override; resid_post is Arditi's native default.
SUBMODULE_ARGS=""
[ "$SAE_KIND" = "ln1" ] && SUBMODULE_ARGS="--submodule-name input_layernorm"

echo "[$(date -u +%H:%M:%S)] START 14B SAE — kind=$SAE_KIND L${HOOK_LAYER} base, ${NUM_TOKENS} tok, k64 d131072"
echo "[$(date -u +%H:%M:%S)] gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null|head -1)"

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null; }
on_err() {
    echo "[$(date -u +%H:%M:%S)] [FAIL] line ${BASH_LINENO[0]} — keeping pod ALIVE for triage"
    nvidia-smi 2>/dev/null | tail -12 || true
    dmesg 2>/dev/null | grep -i "oom\|killed process" | tail -5 || true
    sleep infinity; }
trap on_err ERR

DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1|cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break; echo "nvidia-smi not ready ($i/6)"; sleep 5; done
[ -z "$DRIVER_MAJOR" ] && { echo "no driver — abort"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver $DRIVER_MAJOR <525 — self-term"; terminate_self; exit 0; }

# skip if already trained+uploaded
if curl -sS -H "Authorization: Bearer $HF_TOKEN" \
   "https://huggingface.co/api/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/$HF_PREFIX" 2>/dev/null | grep -q "ae.pt"; then
    echo "[$(date -u +%H:%M:%S)] SAE already on HF ($HF_PREFIX) — self-term"; terminate_self; exit 0; fi

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] HEAD $(git rev-parse --short HEAD)"
export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 PIP_CACHE_DIR=/workspace/.pip_cache
export HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"; mkdir -p "$PIP_CACHE_DIR"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages datasets transformers huggingface_hub 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch',torch.__version__,torch.cuda.get_device_name(0))"
python3 -c "from huggingface_hub import login,HfApi; login(token='$HF_TOKEN',add_to_git_credential=False); HfApi(token='$HF_TOKEN').auth_check('lmsys/lmsys-chat-1m',repo_type='dataset'); print('lmsys access OK')"

echo "[$(date -u +%H:%M:%S)] === TRAIN $SAE_KIND L${HOOK_LAYER} (Arditi code) ==="
python3 -u fra/train_sae_arditi.py \
    --hook-layer "$HOOK_LAYER" $SUBMODULE_ARGS \
    --model-name "$MODEL_NAME" \
    --num-tokens "$NUM_TOKENS" \
    --target-l0s 64 --dictionary-widths 131072 --architectures batch_top_k \
    --no-wandb --output-dir "$OUT"

echo "[$(date -u +%H:%M:%S)] === upload → HF $HF_PREFIX ==="
python3 -u -c "
from huggingface_hub import HfApi; import pathlib
api=HfApi(); root=pathlib.Path('$OUT')
api.upload_folder(folder_path=str(root), path_in_repo='$HF_PREFIX',
    repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
    commit_message='14B $SAE_KIND L${HOOK_LAYER} base SAE (Arditi BatchTopK k64 d131072, ${NUM_TOKENS} tok)')
print('uploaded', '$HF_PREFIX')
"
echo "[$(date -u +%H:%M:%S)] === DONE — self-terminate ==="
trap - ERR; terminate_self
