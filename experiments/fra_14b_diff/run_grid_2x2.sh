#!/usr/bin/env bash
# 2×2 completion cell runner (experiments/fra_14b_diff) — isolates whether the
# FRA-vs-Wang gap is the ATTRIBUTION (encoder f vs OV/QK) or the DIFF-DEFINITION
# (outcome-bucket vs model-identity). Same dispatch skeleton as run_grid_diff.sh.
#
# Two variants (set via VARIANT):
#   VARIANT=A  Wang encoder-activation attribution, OUTCOME-bucket diff (coh>70,
#              §1a tercile fallback). PER-MODEL ranking. ATTRIBUTION=wang
#              DIFF_MODE=bucket. HF tag _bucketdiff_. Steers wang×ln1.
#   VARIANT=B  FRA OV/QK attribution, MODEL-identity diff (finance−base, all
#              answer tokens, no coh gate, per-model-own-weights). ONE
#              model-agnostic ranking steers BOTH base+finance cells.
#              ATTRIBUTION∈{ov,qk} DIFF_MODE=model. HF tag _modeldiff_.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH, VARIANT
# A needs: (nothing extra). B needs: WHICH=ov|qk.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

VARIANT="${VARIANT:?set VARIANT (A|B)}"
SAE="${SAE:-ln1}"
EM_MODELS="${EM_MODELS:-base finance}"
SEEDS="${SEEDS:-42 123}"
GRANS="${GRANS:-1 2 10 50}"
HEAD="${HEAD:-12}"
LAYER="${LAYER:-24}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
ALPHAS="${ALPHAS:--2 -1.5 -1 -0.5 0 0.5 1 1.5 2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
STEER_MODE="${STEER_MODE:-magmatched}"
GRID_HF_BASE="${GRID_HF_BASE:-qwen14b/grid_diff}"
SAE_LN1_HF_PREFIX="${SAE_LN1_HF_PREFIX:-qwen14b/sae_ln1_l24_base_arditi}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
NOISE_SEEDS="${NOISE_SEEDS:-42 123 456}"

# variant → attribution / diff-mode / orchestrator ranking label / HF tag
if [ "$VARIANT" = "A" ]; then
    ATTRIBUTION=wang; DIFF_MODE=bucket; RANKING=wang; TAG=bucketdiff
    RANKING_NAME="wang"
elif [ "$VARIANT" = "B" ]; then
    WHICH="${WHICH:?VARIANT=B needs WHICH=ov|qk}"
    ATTRIBUTION="$WHICH"; DIFF_MODE=model; RANKING="fra-${WHICH}"; TAG=modeldiff
    RANKING_NAME="fra-${WHICH}"
else
    echo "bad VARIANT=$VARIANT"; exit 1
fi

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
        https://api.runpod.io/graphql >/dev/null
}
on_err() {
    echo "[$(date -u +%H:%M:%S)] [FAIL] errored (line ${BASH_LINENO[0]}). Keeping pod ALIVE for debug."
    nvidia-smi 2>/dev/null | tail -12 || true
    sleep infinity
}
trap on_err ERR

echo "[$(date -u +%H:%M:%S)] 2x2 cell VARIANT=$VARIANT attribution=$ATTRIBUTION diff_mode=$DIFF_MODE ranking=$RANKING tag=$TAG models=[$EM_MODELS] seeds=[$SEEDS] head=$HEAD"

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
mkdir -p "$PIP_CACHE_DIR"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

echo "[$(date -u +%H:%M:%S)] downloading ln1 SAE"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='$SAE_LN1_HF_PREFIX/*', local_dir='/workspace/sae_dl')
"
SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
[ -z "$SAE_DIR" ] && { echo "no ae.pt found"; exit 1; }
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"

echo "[$(date -u +%H:%M:%S)] downloading noise_study buckets"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='qwen14b/noise_study/*', local_dir='/workspace/buckets')
"
SCORES_FILE="/workspace/buckets/qwen14b/noise_study/scores/judge_scores_gpt-4o-mini.json"
[ -f "$SCORES_FILE" ] || { echo "no judge scores at $SCORES_FILE"; exit 1; }
# ALL noise rollout files (both models — model-diff needs both; bucket filters internally)
ALL_ROLLOUTS=""
for m in base finance; do for ns in $NOISE_SEEDS; do
    f="/workspace/buckets/qwen14b/noise_study/${m}_seed${ns}.json"; [ -f "$f" ] && ALL_ROLLOUTS="$ALL_ROLLOUTS $f"
done; done

# ── ln1 ‖Δa‖ (magnitude-matched) ──
echo "[$(date -u +%H:%M:%S)] === computing ln1 post-gain ‖Δa‖ ==="
python3 -u scripts/compute_delta_a_norm.py --layer "$LAYER" --out /workspace/delta_a_norm.json
DELTA_A=$(python3 -c "import json; print(json.load(open('/workspace/delta_a_norm.json'))['ln1_postgain']['diff_norm_l2'])")
[ -z "$DELTA_A" ] && { echo "failed ‖Δa‖"; exit 1; }
echo "[$(date -u +%H:%M:%S)] ‖Δa‖_ln1=$DELTA_A"
DELTA_A_ARG="--delta-a-norm $DELTA_A"

run_sweep() {  # $1=EM  $2=SEED  $3=RANK_JSON
  local EM="$1" SEED="$2" RJ="$3"
  local OUT="/workspace/g2x2_${RANKING}_${TAG}_${EM}_seed${SEED}"; mkdir -p "$OUT"
  echo "[$(date -u +%H:%M:%S)] === sweep ($RANKING/$TAG, $EM, seed=$SEED) ==="
  python3 -u phase1_grid_14b_orchestrator.py \
      --ranking "$RANKING" --sae "$SAE" --sae-dir "$SAE_DIR" \
      --em-model "$EM" --eval-seed "$SEED" \
      --layer "$LAYER" --head "$HEAD" \
      --granularities $GRANS \
      --samples-per-prompt "$SAMPLES_PER_PROMPT" \
      --alphas $ALPHAS --max-new-tokens "$MAX_NEW_TOKENS" \
      --ranking-json "$RJ" $DELTA_A_ARG \
      --output-root "$OUT"
  python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api=HfApi(); out=pathlib.Path('$OUT')
for gdir in sorted(out.glob('gran*')):
    gran=gdir.name.replace('gran','')
    for f in gdir.glob('*.json'):
        tgt=f'$GRID_HF_BASE/${RANKING}_${SAE}_${TAG}_gran{gran}/${EM}_seed${SEED}/{f.name}'
        api.upload_file(path_or_fileobj=str(f),path_in_repo=tgt,repo_id='$HF_REPO',repo_type='dataset',
                        commit_message='2x2 $RANKING $TAG gran{} ($EM s$SEED)'.format(gran))
        print('  →',tgt,flush=True)
"
}

if [ "$VARIANT" = "B" ]; then
  # ONE model-agnostic finance−base ranking; steer BOTH cells with it.
  RANK_JSON="/workspace/rank_${RANKING}_${TAG}_L${LAYER}.json"
  echo "[$(date -u +%H:%M:%S)] === model-diff ranking ($RANKING, finance−base) ==="
  python3 -u scripts/compute_fra_diff_ranking.py \
      --attribution "$ATTRIBUTION" --diff-mode model --sae "$SAE" --sae-dir "$SAE_DIR" \
      --rollout-files $ALL_ROLLOUTS --scores-file "$SCORES_FILE" \
      --layer "$LAYER" --head "$HEAD" --top-n 50 --k-pairs 50 \
      --out "$RANK_JSON"
  python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$RANK_JSON',
    path_in_repo='$GRID_HF_BASE/${RANKING}_${SAE}_${TAG}_meta/ranking_modeldiff_L${LAYER}.json',
    repo_id='$HF_REPO',repo_type='dataset',commit_message='2x2 $RANKING modeldiff ranking')
"
  for EM in $EM_MODELS; do for SEED in $SEEDS; do run_sweep "$EM" "$SEED" "$RANK_JSON"; done; done
else
  # VARIANT A: PER-MODEL outcome-bucket ranking (wang attribution).
  for EM in $EM_MODELS; do
    ROLLOUT_ARGS=""
    for ns in $NOISE_SEEDS; do f="/workspace/buckets/qwen14b/noise_study/${EM}_seed${ns}.json"; [ -f "$f" ] && ROLLOUT_ARGS="$ROLLOUT_ARGS $f"; done
    RANK_JSON="/workspace/rank_${RANKING}_${TAG}_${EM}_L${LAYER}.json"
    echo "[$(date -u +%H:%M:%S)] === bucket-diff ranking (wang, $EM) ==="
    python3 -u scripts/compute_fra_diff_ranking.py \
        --attribution wang --diff-mode bucket --sae "$SAE" --sae-dir "$SAE_DIR" \
        --model "$EM" --rollout-files $ROLLOUT_ARGS --scores-file "$SCORES_FILE" \
        --layer "$LAYER" --head "$HEAD" --top-n 50 \
        --out "$RANK_JSON"
    python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$RANK_JSON',
    path_in_repo='$GRID_HF_BASE/${RANKING}_${SAE}_${TAG}_meta/ranking_bucketdiff_${EM}_L${LAYER}.json',
    repo_id='$HF_REPO',repo_type='dataset',commit_message='2x2 wang bucketdiff ranking $EM')
"
    for SEED in $SEEDS; do run_sweep "$EM" "$SEED" "$RANK_JSON"; done
  done
fi

echo "[$(date -u +%H:%M:%S)] === 2x2 CELL DONE (VARIANT=$VARIANT $RANKING/$TAG) — self-terminate ==="
trap - ERR
terminate_self
