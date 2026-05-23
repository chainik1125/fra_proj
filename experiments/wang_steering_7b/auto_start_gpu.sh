#!/usr/bin/env bash
# Wang-steering GPU pod bootstrap.
#
# Required env (passed via launch_all.sh):
#   HF_TOKEN          HF write token (target dataset has the right scope)
#   RUNPOD_API_KEY    for the self-stop at the end
#   RUNPOD_POD_ID     this pod's own id
#   BRANCH            git branch to clone (e.g. autoresearch/wang-steering-7b)
#   EM_MODEL          medical | base
#   EVAL_SEED         42 | 123 | 456
#   TOP_N             features to sweep (default 50)
#
# Pod self-stops on success. On any failure it leaves the pod alive so we
# can SSH in to diagnose.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

echo "[$(date -u +%H:%M:%S)] START em_model=$EM_MODEL seed=$EVAL_SEED top_n=${TOP_N:-50}"
echo "[$(date -u +%H:%M:%S)] driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ── Driver fast-fail (cu13 torch needs ≥ 575) ─────────────────────────
# Retry nvidia-smi up to 6 × 5 s in case the device-plumbing isn't ready
# yet on cold boot (race we hit on 2026-05-22 in the constadd campaign
# AND again here: an empty $DRIVER_MAJOR triggered a false self-terminate).
DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break
    echo "[$(date -u +%H:%M:%S)] nvidia-smi not ready yet (attempt $i/6) — sleep 5"
    sleep 5
done
if [ -z "$DRIVER_MAJOR" ]; then
    echo "[$(date -u +%H:%M:%S)] nvidia-smi never returned a driver version — abort (no self-terminate, let RunPod restart so we get a different host)"
    exit 1
fi
if [ "$DRIVER_MAJOR" -lt 525 ]; then
    echo "[$(date -u +%H:%M:%S)] driver too old ($DRIVER_MAJOR < 525) — self-terminate"
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
    exit 0
fi
# Threshold is 525 (cu124 baseline). The pip install step below replaces
# the cu130 torch from requirements.txt with cu124 explicitly, so we only
# need cu124-compatible drivers (525+). Previous 575 threshold was for the
# stock cu130 torch; cost us 6 medical-s456 re-rolls before this fix.
echo "[$(date -u +%H:%M:%S)] driver major=$DRIVER_MAJOR — proceed (will use cu124 torch)"

# ── HF pre-check: target already on HF → skip ─────────────────────────
# Use curl + jq directly — huggingface_hub isn't on the base image's Python
# and we don't want to pay the pip-install cost before knowing whether to
# skip. Also wrap with `|| true` so a network blip doesn't kill the run.
TARGET_HF_PATH="qwen7b/wang_L15_resid_post${HF_PATH_SUFFIX:-}/${EM_MODEL}_seed${EVAL_SEED}/qualitative_arditi_${EM_MODEL}_evalseed${EVAL_SEED}.json"
echo "[$(date -u +%H:%M:%S)] checking HF for $TARGET_HF_PATH"
HF_API_URL="https://huggingface.co/api/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/qwen7b/wang_L15_resid_post${HF_PATH_SUFFIX:-}/${EM_MODEL}_seed${EVAL_SEED}"
HF_LIST=$(curl -sS -H "Authorization: Bearer $HF_TOKEN" "$HF_API_URL" 2>/dev/null || echo "[]")
if echo "$HF_LIST" | grep -q "qualitative_arditi_${EM_MODEL}_evalseed${EVAL_SEED}.json"; then
    echo "[$(date -u +%H:%M:%S)] already on HF — self-terminate"
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
        -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
    exit 0
fi
echo "[$(date -u +%H:%M:%S)] not yet on HF — proceeding"

# ── Clone + install ───────────────────────────────────────────────────
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj
git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] git HEAD: $(git rev-parse HEAD)"

export HF_HOME=/workspace/.hf_cache
export PYTHONUNBUFFERED=1
mkdir -p /workspace/.pip_cache
export PIP_CACHE_DIR=/workspace/.pip_cache

echo "[$(date -u +%H:%M:%S)] pip install requirements.txt"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -3
echo "[$(date -u +%H:%M:%S)] pip install transformer_lens 3.x"
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -3
echo "[$(date -u +%H:%M:%S)] pip install dictionary_learning + peft (Arditi SAE loader)"
pip install --no-input --break-system-packages 'dictionary_learning' peft 2>&1 | tail -3

# ── Replace requirements.txt's cu130 torch with cu124 (driver ≥525 compat) ─
# Same trick we used in the 2026-05-22 constadd campaign to dodge the
# driver-lottery problem. --force-reinstall replaces the cu130 build;
# --no-deps means numpy/typing-extensions etc. installed by requirements.txt
# are not touched.
echo "[$(date -u +%H:%M:%S)] override torch to cu124 (driver ≥525 compat)"
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3

# ── Sanity check ──────────────────────────────────────────────────────
python3 -c "import torch; assert torch.cuda.is_available(); print(f'torch={torch.__version__} cuda={torch.version.cuda}')"

# ── Step A: Wang feature ranker ───────────────────────────────────────
TOP_N=${TOP_N:-50}
RANKER_OUT="/workspace/wang_ranker_L15_top${TOP_N}.json"
echo "[$(date -u +%H:%M:%S)] === STEP A: Wang ranker ==="
python3 -u scripts/compute_wang_feature_ranking.py \
    --layer 15 --trainer 1 --em-domain medical \
    --top-n "$TOP_N" --out "$RANKER_OUT"

# Extract feature_ids as a space-separated string for the orchestrator arg.
FEATURE_IDS=$(python3 -c "import json; print(' '.join(str(f) for f in json.load(open('$RANKER_OUT'))['feature_ids']))")
echo "[$(date -u +%H:%M:%S)] ranker → ${TOP_N} features (head: ${FEATURE_IDS:0:80}...)"

# ── Step B: sweep on this pod's (em_model, seed) ──────────────────────
OUT_DIR="/workspace/wang_sweep_${EM_MODEL}_seed${EVAL_SEED}"
mkdir -p "$OUT_DIR"
echo "[$(date -u +%H:%M:%S)] === STEP B: sweep ($EM_MODEL, seed=$EVAL_SEED) ==="
python3 -u phase1_arditi_orchestrator.py \
    --em-model "$EM_MODEL" --eval-seed "$EVAL_SEED" \
    --samples-per-prompt "${SAMPLES_PER_PROMPT:-1}" \
    --layer 15 --trainer 1 \
    --feature-ids $FEATURE_IDS \
    --alphas -2 -1.75 -1.5 -1.25 -1 -0.75 -0.5 -0.25 0 0.25 0.5 0.75 1 1.25 1.5 1.75 2 \
    --max-new-tokens 200 \
    --output-root "$OUT_DIR"

# ── Step C: upload to HF ──────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === STEP C: upload ==="
python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api = HfApi()
for f in pathlib.Path('$OUT_DIR').glob('*.json'):
    target = f'qwen7b/wang_L15_resid_post${HF_PATH_SUFFIX:-}/${EM_MODEL}_seed${EVAL_SEED}/{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=target,
                    repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                    commit_message='wang_steering shard ($EM_MODEL, seed${EVAL_SEED})')
    print(f'  → {target}', flush=True)
"
# Also upload the ranker JSON once (idempotent — first writer wins).
python3 -u -c "
from huggingface_hub import HfApi, hf_hub_download
api = HfApi()
target = f'qwen7b/wang_L15_resid_post${HF_PATH_SUFFIX:-}/wang_ranker_L15_top${TOP_N}.json'
try:
    files = api.list_repo_files('dmanningcoe/fra-phase1-steering-data', repo_type='dataset')
    if target not in files:
        api.upload_file(path_or_fileobj='$RANKER_OUT', path_in_repo=target,
                        repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                        commit_message='wang_steering ranker output')
        print(f'  → {target}', flush=True)
    else:
        print(f'  (ranker already on HF: {target})', flush=True)
except Exception as e:
    print(f'  (ranker upload skipped: {e})', flush=True)
"

# ── Self-stop ─────────────────────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === DONE — self-terminate ==="
curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql
