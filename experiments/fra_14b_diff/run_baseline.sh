#!/usr/bin/env bash
# Pod-side BASELINE stage for the YAML campaign (the hard gate before ranking):
#   1. generate α=0 rollouts for base + EM model      (alpha0_noise_gen.py)
#   2. judge them at gpt-4o-mini T0 + upload the scores (judge_temp_sweep.py)
# Produces, on HF:
#   <NOISE_PREFIX>/{base,<em>}_seed<seed>.json
#   <NOISE_PREFIX>/scores/judge_scores_gpt-4o-mini.json   ← compute_fra_diff_ranking reads this
# then self-terminates.
#
# Pod env (injected by the driver's launch env):
#   HF_TOKEN, RUNPOD_API_KEY, OPENAI_API_KEY (or _MATS), BRANCH
#   NOISE_MODELS ("base medical"), EM_MODEL_ID, BASE_MODEL_ID, NOISE_PREFIX,
#   NOISE_SEEDS ("42 123 456"), EM_KEY, JUDGE_MODEL (gpt-4o-mini)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
BRANCH="${BRANCH:-autoresearch/cadenza-attn-only}"
JUDGE_MODEL="${JUDGE_MODEL:-gpt-4o-mini}"
NOISE_SEEDS="${NOISE_SEEDS:-42 123 456}"

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
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft openai 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

# 1. generate α=0 rollouts (base + EM) → upload to NOISE_PREFIX
echo "[$(date -u +%H:%M:%S)] === α=0 rollouts: models=[$NOISE_MODELS] prefix=$NOISE_PREFIX ==="
NOISE_MODELS="$NOISE_MODELS" EM_MODEL_ID="$EM_MODEL_ID" BASE_MODEL_ID="$BASE_MODEL_ID" \
NOISE_PREFIX="$NOISE_PREFIX" NOISE_SEEDS="$NOISE_SEEDS" \
python3 -u experiments/fra_14b_financial/alpha0_noise_gen.py

# 2. judge at T0 + upload scores (only temp 0.0 needed for the buckets → ¼ the cost)
set +x
echo "[$(date -u +%H:%M:%S)] === judge α=0 (T0) → $NOISE_PREFIX/scores ==="
python3 -u scripts/judge_temp_sweep.py --judge-model "$JUDGE_MODEL" \
    --targets-prefix "$NOISE_PREFIX" --models $NOISE_MODELS --seeds $NOISE_SEEDS \
    --temps 0.0 --upload-prefix "$NOISE_PREFIX/scores"

echo "[$(date -u +%H:%M:%S)] === BASELINE DONE — self-terminate ==="
trap - ERR
terminate_self
