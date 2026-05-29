#!/usr/bin/env bash
# Bucketed-diff GRID cell runner (CAMPAIGN.md, experiments/fra_14b_diff).
#
# Same dispatch skeleton as experiments/fra_14b_financial/run_grid.sh (cu124
# torch override, driver-gate, ERR-trap→sleep infinity, incremental HF upload,
# self-terminate). The ONLY difference: the FRA ranking is the PER-MODEL
# bucketed diff (scripts/compute_fra_diff_ranking.py, SUMMARY.md §5.1), computed
# INSIDE the per-model loop (diff ranking is model-specific — base ranking on
# base's weights+buckets, finance on finance's), then passed via --ranking-json.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH
# Cell env:
#   WHICH     = ov | qk        (ΔOV or ΔQK pair ranking)
#   SAE       = ln1 | resid_post
#   EM_MODELS = "base finance"  (decompose per model)
#   SEEDS     = "42 123"        (steering-sweep seeds; bucket pools ALL noise seeds)
#   GRANS     = "1 2 10 50"
#   HEAD      = 12              (head-ablation argmax — NOT 0)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

WHICH="${WHICH:?set WHICH (ov|qk)}"
SAE="${SAE:?set SAE}"
EM_MODELS="${EM_MODELS:-base finance}"
SEEDS="${SEEDS:-42 123}"
GRANS="${GRANS:-1 2 10 50}"
HEAD="${HEAD:-12}"
LAYER="${LAYER:-24}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
ALPHAS="${ALPHAS:--2 -1.5 -1 -0.5 0 0.5 1 1.5 2}"
MAX_NEW_TOKENS="${MAX_NEW_TOKENS:-100}"
STEER_MODE="${STEER_MODE:-magmatched}"
# diff campaign ALWAYS lives under grid_diff/ (never collide with grid_magmatched).
GRID_HF_BASE="${GRID_HF_BASE:-qwen14b/grid_diff}"
SAE_LN1_HF_PREFIX="${SAE_LN1_HF_PREFIX:-qwen14b/sae_ln1_l24_base_arditi}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"
# orchestrator wants ranking ∈ {wang,fra-qk,fra-ov}; map WHICH → fra-<which>
RANKING="fra-${WHICH}"
# bucket sources on HF (pooled across ALL these noise seeds, per model)
NOISE_SEEDS="${NOISE_SEEDS:-42 123 456}"

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

echo "[$(date -u +%H:%M:%S)] DIFF GRID cell: which=$WHICH sae=$SAE models=[$EM_MODELS] seeds=[$SEEDS] grans=[$GRANS] head=$HEAD"

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
# TORCH RECIPE (FIXED — CAMPAIGN.md §5): torch 2.6.0+cu124 WITH deps + force pandas
# + dictionary_learning. Do NOT regress to 2.4.1 (breaks transformers import).
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -2
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' 2>&1 | tail -2
pip install --no-input --break-system-packages 'dictionary_learning' peft 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall \
    torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-cache-dir pandas 2>&1 | tail -1
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

# ── Resolve the SAE dir ──────────────────────────────────────────────────
if [ "$SAE" = "ln1" ]; then
    echo "[$(date -u +%H:%M:%S)] downloading ln1 SAE from HF $SAE_LN1_HF_PREFIX"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='$SAE_LN1_HF_PREFIX/*', local_dir='/workspace/sae_dl')
"
    SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
else
    echo "[$(date -u +%H:%M:%S)] downloading resid_post SAE (andyrdt trainer_1)"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen14b/sae_resid_post_l24_base_arditi/*', local_dir='/workspace/sae_rp')
"
    SAE_DIR=$(dirname "$(find /workspace/sae_rp -name ae.pt | head -1)")
fi
[ -z "$SAE_DIR" ] && { echo "no ae.pt found"; exit 1; }
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"

# ── Download bucket sources (noise_study α=0 rollouts + gpt-4o-mini scores) ─
echo "[$(date -u +%H:%M:%S)] downloading noise_study buckets (seeds $NOISE_SEEDS)"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset',
    allow_patterns='qwen14b/noise_study/*', local_dir='/workspace/buckets')
"
SCORES_FILE="/workspace/buckets/qwen14b/noise_study/scores/judge_scores_gpt-4o-mini.json"
[ -f "$SCORES_FILE" ] || { echo "no judge scores at $SCORES_FILE"; exit 1; }

# ── ‖Δa‖ for magnitude-matched steering ──────────────────────────────────
DELTA_A_ARG=""
if [ "$STEER_MODE" = "magmatched" ]; then
    if [ "$SAE" = "ln1" ]; then
        echo "[$(date -u +%H:%M:%S)] === computing ln1 post-gain ‖Δa‖ ==="
        python3 -u scripts/compute_delta_a_norm.py --layer "$LAYER" --out /workspace/delta_a_norm.json
        DELTA_A=$(python3 -c "import json; print(json.load(open('/workspace/delta_a_norm.json'))['ln1_postgain']['diff_norm_l2'])")
        [ -z "$DELTA_A" ] && { echo "failed to compute ln1 ‖Δa‖"; exit 1; }
    else
        DELTA_A="${RESID_POST_DELTA_A:-45.43}"
        echo "{\"sae\":\"resid_post\",\"delta_a_norm\":$DELTA_A}" > /workspace/delta_a_norm.json
    fi
    echo "[$(date -u +%H:%M:%S)] ‖Δa‖[$SAE]=$DELTA_A"
    DELTA_A_ARG="--delta-a-norm $DELTA_A"
    python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='/workspace/delta_a_norm.json',
    path_in_repo='$GRID_HF_BASE/${RANKING}_${SAE}_meta/delta_a_norm_L${LAYER}.json',
    repo_id='$HF_REPO', repo_type='dataset', commit_message='grid_diff: ‖Δa‖ $SAE')
" || echo "(‖Δa‖ HF upload failed — non-fatal)"
fi

# ── Sweep: per model (diff ranking is per-model) → per seed → all grans ────
for EM in $EM_MODELS; do
  # PER-MODEL diff ranking (SUMMARY.md §5.1): decompose on THIS model's weights
  # + THIS model's α=0 buckets. Pool noise_study rollouts across NOISE_SEEDS.
  ROLLOUT_ARGS=""
  for ns in $NOISE_SEEDS; do
      f="/workspace/buckets/qwen14b/noise_study/${EM}_seed${ns}.json"
      [ -f "$f" ] && ROLLOUT_ARGS="$ROLLOUT_ARGS $f"
  done
  RANK_JSON="/workspace/diff_ranking_${EM}_L${LAYER}.json"
  echo "[$(date -u +%H:%M:%S)] === diff ranking (ΔWHICH=$WHICH, $EM) ==="
  python3 -u scripts/compute_fra_diff_ranking.py \
      --which "$WHICH" --sae "$SAE" --sae-dir "$SAE_DIR" \
      --model "$EM" \
      --rollout-files $ROLLOUT_ARGS \
      --scores-file "$SCORES_FILE" \
      --layer "$LAYER" --head "$HEAD" \
      --top-n 50 --k-pairs 50 \
      --out "$RANK_JSON"
  # back the per-model ranking up to HF immediately (with its bucket-size meta)
  python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$RANK_JSON',
    path_in_repo='$GRID_HF_BASE/${RANKING}_${SAE}_meta/diff_ranking_${EM}_L${LAYER}.json',
    repo_id='$HF_REPO', repo_type='dataset', commit_message='grid_diff: $RANKING ranking $EM')
"
  RANK_JSON_ARG="--ranking-json $RANK_JSON"

  for SEED in $SEEDS; do
    OUT="/workspace/grid_${RANKING}_${SAE}_${EM}_seed${SEED}"; mkdir -p "$OUT"
    echo "[$(date -u +%H:%M:%S)] === sweep ($RANKING×$SAE, $EM, seed=$SEED, grans=[$GRANS]) ==="
    python3 -u phase1_grid_14b_orchestrator.py \
        --ranking "$RANKING" --sae "$SAE" --sae-dir "$SAE_DIR" \
        --em-model "$EM" --eval-seed "$SEED" \
        --layer "$LAYER" --head "$HEAD" \
        --granularities $GRANS \
        --samples-per-prompt "$SAMPLES_PER_PROMPT" \
        --alphas $ALPHAS \
        --max-new-tokens "$MAX_NEW_TOKENS" \
        $RANK_JSON_ARG $DELTA_A_ARG \
        --output-root "$OUT"
    echo "[$(date -u +%H:%M:%S)] === upload ($EM, seed=$SEED) ==="
    python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api = HfApi()
out = pathlib.Path('$OUT')
for gdir in sorted(out.glob('gran*')):
    gran = gdir.name.replace('gran','')
    for f in gdir.glob('*.json'):
        tgt = f'$GRID_HF_BASE/${RANKING}_${SAE}_gran{gran}/${EM}_seed${SEED}/{f.name}'
        api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                        repo_id='$HF_REPO', repo_type='dataset',
                        commit_message='grid_diff $RANKING $SAE gran{} ($EM s$SEED)'.format(gran))
        print('  →', tgt, flush=True)
for f in out.glob('ranking_*.json'):
    tgt = f'$GRID_HF_BASE/${RANKING}_${SAE}_meta/${EM}_seed${SEED}_{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                    repo_id='$HF_REPO', repo_type='dataset', commit_message='grid_diff ranking meta')
    print('  →', tgt, flush=True)
"
  done
done

echo "[$(date -u +%H:%M:%S)] === DIFF GRID CELL DONE ($RANKING×$SAE) — self-terminate ==="
trap - ERR
terminate_self
