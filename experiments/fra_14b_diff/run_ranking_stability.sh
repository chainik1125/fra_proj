#!/usr/bin/env bash
# Ranking-stability check (RANKING ONLY — no steering sweeps). One cheap pod.
# Does the FRA-OV/QK top-N change when the misaligned-coherent bucket isn't thin?
#   Method 1 (coh50 rebucket): recompute OV+QK ln1 finance bucketed-diff at
#     coh_floor=50 on EXISTING noise_study rollouts (no generation/judging).
#   Method 2 (resample @coh70): generate ~512 fresh α=0 finance rollouts, judge
#     gpt-4o-mini@T0, bucket at coh>70, recompute OV+QK rankings.
# Deliverable = top-10 OV/QK ids for each method vs the current thin coh>70
# (OV F59432, QK F603) + bucket sizes. Uploads under qwen14b/grid_diff/stability/.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH, OPENAI_API_KEY
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

LAYER="${LAYER:-24}"; HEAD="${HEAD:-12}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-64}"   # 8×64 = 512
HF_REPO="dmanningcoe/fra-phase1-steering-data"
SAE_LN1_HF_PREFIX="${SAE_LN1_HF_PREFIX:-qwen14b/sae_ln1_l24_base_arditi}"
NOISE_SEEDS="${NOISE_SEEDS:-42 123 456}"
OUT_HF="qwen14b/grid_diff/stability"

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
}
on_err() { echo "[$(date -u +%H:%M:%S)] [FAIL] line ${BASH_LINENO[0]} — pod ALIVE for debug."; nvidia-smi 2>/dev/null | tail -5 || true; sleep infinity; }
trap on_err ERR

echo "[$(date -u +%H:%M:%S)] ranking-stability check: coh50-rebucket + resample@coh70 (finance, OV+QK, H$HEAD)"
DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1); [ -n "$DRIVER_MAJOR" ] && break; sleep 5; done
[ -z "$DRIVER_MAJOR" ] && { echo "no driver"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver too old"; terminate_self; exit 0; }

cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
echo "[$(date -u +%H:%M:%S)] git HEAD: $(git rev-parse HEAD)"

export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 PIP_CACHE_DIR=/workspace/.pip_cache
export HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
export OPENAI_API_KEY="${OPENAI_API_KEY:?need OPENAI_API_KEY for judging}"
mkdir -p "$PIP_CACHE_DIR"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft openai 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

echo "[$(date -u +%H:%M:%S)] downloading ln1 SAE + noise_study buckets"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='$SAE_LN1_HF_PREFIX/*', local_dir='/workspace/sae_dl')
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='qwen14b/noise_study/*', local_dir='/workspace/buckets')
"
SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
SCORES_FILE="/workspace/buckets/qwen14b/noise_study/scores/judge_scores_gpt-4o-mini.json"
ROLLOUTS=""; for ns in $NOISE_SEEDS; do f="/workspace/buckets/qwen14b/noise_study/finance_seed${ns}.json"; [ -f "$f" ] && ROLLOUTS="$ROLLOUTS $f"; done
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"
mkdir -p /workspace/stab

upload() { python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$1', path_in_repo='$OUT_HF/$2', repo_id='$HF_REPO', repo_type='dataset', commit_message='ranking-stability: $2')
print('  →','$OUT_HF/$2')
"; }

# ── METHOD 1: coh50 rebucket (existing rollouts; OV + QK) ──
for WHICH in ov qk; do
  echo "[$(date -u +%H:%M:%S)] === Method1 coh50 rebucket: $WHICH ==="
  python3 -u scripts/compute_fra_diff_ranking.py \
      --attribution "$WHICH" --diff-mode bucket --sae ln1 --sae-dir "$SAE_DIR" \
      --model finance --rollout-files $ROLLOUTS --scores-file "$SCORES_FILE" \
      --layer "$LAYER" --head "$HEAD" --coh-floor 50 --top-n 50 --k-pairs 50 \
      --out /workspace/stab/coh50_${WHICH}_finance.json
  upload /workspace/stab/coh50_${WHICH}_finance.json coh50_${WHICH}_finance_L${LAYER}.json
done

# ── METHOD 2: resample @coh70 (generate 512, judge, bucket, OV + QK in one run) ──
echo "[$(date -u +%H:%M:%S)] === Method2 resample @coh70 (OV+QK) ==="
python3 -u scripts/ranking_stability_resample.py \
    --sae-dir "$SAE_DIR" --sae ln1 --layer "$LAYER" --head "$HEAD" \
    --n-prompts 8 --samples-per-prompt "$SAMPLES_PER_PROMPT" --coh-floor 70 \
    --top-n 50 --k-pairs 50 \
    --out /workspace/stab/resample_coh70_finance.json \
    --rollouts-out /workspace/stab/resample_rollouts_finance.json
upload /workspace/stab/resample_coh70_finance.json resample_coh70_finance_L${LAYER}.json
upload /workspace/stab/resample_rollouts_finance.json resample_rollouts_finance_L${LAYER}.json

echo "[$(date -u +%H:%M:%S)] === RANKING-STABILITY DONE — self-terminate ==="
trap - ERR
terminate_self
