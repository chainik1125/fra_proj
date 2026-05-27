#!/usr/bin/env bash
# Grid cell runner: ONE (ranking × sae) pair across EM_MODELS × SEEDS, all
# GRANS, conventional additive steering. Reuses the proven FRA-run patterns:
#   cu124 torch override, driver-gate ≥525, ERR-trap→sleep infinity (NO restart
#   loop), incremental HF upload, self-terminate on success.
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH
# Cell env:
#   RANKING   = wang | fra-qk | fra-ov
#   SAE       = ln1 | resid_post
#   EM_MODELS = "base medical"   (base first)
#   SEEDS     = "42 123 456"
#   GRANS     = "1 2 10 50"
#   HEAD      = L15 head for FRA ranking (default 0; head-ablation argmax on base)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

RANKING="${RANKING:?set RANKING}"
SAE="${SAE:?set SAE}"
EM_MODELS="${EM_MODELS:-base medical}"
SEEDS="${SEEDS:-42 123 456}"
GRANS="${GRANS:-1 2 10 50}"
HEAD="${HEAD:-0}"
LAYER="${LAYER:-15}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
GRID_HF_BASE="${GRID_HF_BASE:-qwen7b/grid}"
SAE_LN1_HF_PREFIX="${SAE_LN1_HF_PREFIX:-qwen7b/sae_ln1_l15_base_arditi}"
HF_REPO="dmanningcoe/fra-phase1-steering-data"

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

echo "[$(date -u +%H:%M:%S)] GRID cell: ranking=$RANKING sae=$SAE models=[$EM_MODELS] seeds=[$SEEDS] grans=[$GRANS] head=$HEAD"

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
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
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
snapshot_download('andyrdt/saes-qwen2.5-7b-instruct', allow_patterns='resid_post_layer_${LAYER}/trainer_1/*', local_dir='/workspace/sae_rp')
"
    SAE_DIR=$(dirname "$(find /workspace/sae_rp -name ae.pt | head -1)")
fi
[ -z "$SAE_DIR" ] && { echo "no ae.pt found"; exit 1; }
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"

# ── Wang ranking precompute (proper medical-vs-base Δf, once per cell) ────
# FRA rankings are computed per-model inside the orchestrator; Wang needs both
# models, so compute its Δf JSON here and pass --ranking-json to all runs.
RANK_JSON_ARG=""
if [ "$RANKING" = "wang" ]; then
    RANK_JSON="/workspace/wang_ranker_${SAE}_L${LAYER}.json"
    if [ "$SAE" = "ln1" ]; then
        echo "[$(date -u +%H:%M:%S)] === Wang Δf ranking (ln1 SAE) ==="
        python3 -u scripts/compute_wang_feature_ranking.py \
            --layer "$LAYER" --sae-dir "$SAE_DIR" --top-n 50 --out "$RANK_JSON"
    else
        echo "[$(date -u +%H:%M:%S)] === Wang Δf ranking (resid_post SAE) ==="
        python3 -u scripts/compute_wang_feature_ranking.py \
            --layer "$LAYER" --trainer 1 --top-n 50 --out "$RANK_JSON"
    fi
    RANK_JSON_ARG="--ranking-json $RANK_JSON"
    # back the ranking up to HF immediately
    python3 -c "
from huggingface_hub import HfApi
HfApi().upload_file(path_or_fileobj='$RANK_JSON',
    path_in_repo='$GRID_HF_BASE/${RANKING}_${SAE}_meta/wang_ranker_${SAE}_L${LAYER}.json',
    repo_id='$HF_REPO', repo_type='dataset', commit_message='grid: wang ranking $SAE')
"
fi

# ── Sweep: base first, then EM; per seed; all granularities in one run ────
for EM in $EM_MODELS; do
  for SEED in $SEEDS; do
    OUT="/workspace/grid_${RANKING}_${SAE}_${EM}_seed${SEED}"; mkdir -p "$OUT"
    echo "[$(date -u +%H:%M:%S)] === grid ($RANKING×$SAE, $EM, seed=$SEED, grans=[$GRANS]) ==="
    python3 -u phase1_grid_7b_orchestrator.py \
        --ranking "$RANKING" --sae "$SAE" --sae-dir "$SAE_DIR" \
        --em-model "$EM" --eval-seed "$SEED" \
        --layer "$LAYER" --head "$HEAD" \
        --granularities $GRANS \
        --samples-per-prompt "$SAMPLES_PER_PROMPT" \
        $RANK_JSON_ARG \
        --output-root "$OUT"
    echo "[$(date -u +%H:%M:%S)] === upload ($EM, seed=$SEED) ==="
    # Upload each granularity's qualitative under qwen7b/grid/<ranking>_<sae>_<gran>/<model>_seed<seed>/
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
                        commit_message='grid $RANKING $SAE gran{} ($EM s$SEED)'.format(gran))
        print('  →', tgt, flush=True)
# the ranking used (one per cell)
for f in out.glob('ranking_*.json'):
    tgt = f'$GRID_HF_BASE/${RANKING}_${SAE}_meta/${EM}_seed${SEED}_{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                    repo_id='$HF_REPO', repo_type='dataset', commit_message='grid ranking meta')
    print('  →', tgt, flush=True)
"
  done
done

echo "[$(date -u +%H:%M:%S)] === GRID CELL DONE ($RANKING×$SAE) — self-terminate ==="
trap - ERR
terminate_self
