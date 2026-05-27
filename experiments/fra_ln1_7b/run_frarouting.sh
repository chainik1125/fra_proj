#!/usr/bin/env bash
# FRA-routing magmatched cell runner: ONE recipe across EM_MODELS × SEEDS, all
# GRANS, ln1 SAE. Same hardened patterns as run_grid.sh (cu124, driver-gate ≥525,
# ERR-trap→sleep infinity, incremental HF upload, self-terminate).
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH
# Cell env:
#   RECIPE    = qk_to_qk | qk_to_ov | ov_to_ov
#   EM_MODELS = "base medical"   SEEDS = "42 123 456"   GRANS = "1 2 10 26"
#   HEAD      = L15 head (default 0)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

RECIPE="${RECIPE:?set RECIPE}"
EM_MODELS="${EM_MODELS:-base medical}"
SEEDS="${SEEDS:-42 123 456}"
GRANS="${GRANS:-1 2 10 26}"
HEAD="${HEAD:-0}"
LAYER="${LAYER:-15}"
SAMPLES_PER_PROMPT="${SAMPLES_PER_PROMPT:-4}"
GRID_HF_BASE="${GRID_HF_BASE:-qwen7b/grid_magmatched}"
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

echo "[$(date -u +%H:%M:%S)] FRA-ROUTING cell: recipe=$RECIPE models=[$EM_MODELS] seeds=[$SEEDS] grans=[$GRANS] head=$HEAD"
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

# ── ln1 SAE ──
echo "[$(date -u +%H:%M:%S)] downloading ln1 SAE"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$HF_REPO', repo_type='dataset', allow_patterns='$SAE_LN1_HF_PREFIX/*', local_dir='/workspace/sae_dl')
"
SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
[ -z "$SAE_DIR" ] && { echo "no ae.pt found"; exit 1; }
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"

# ── ln1 ‖Δa‖ (fresh-compute, same as conventional ln1 cells) ──
echo "[$(date -u +%H:%M:%S)] === computing ln1 post-gain ‖Δa‖ ==="
python3 -u scripts/compute_delta_a_norm.py --layer "$LAYER" --out /workspace/delta_a_norm.json
DELTA_A=$(python3 -c "import json; print(json.load(open('/workspace/delta_a_norm.json'))['ln1_postgain']['diff_norm_l2'])")
[ -z "$DELTA_A" ] && { echo "failed ‖Δa‖"; exit 1; }
echo "[$(date -u +%H:%M:%S)] ‖Δa‖_ln1=$DELTA_A"

# ── sweeps: base first, then EM; per seed; all grans in one run ──
for EM in $EM_MODELS; do
  for SEED in $SEEDS; do
    OUT="/workspace/fr_${RECIPE}_${EM}_seed${SEED}"; mkdir -p "$OUT"
    echo "[$(date -u +%H:%M:%S)] === routing ($RECIPE, $EM, seed=$SEED, grans=[$GRANS]) ==="
    OVERRIDE_ARG=""
    [ -n "${FEATURE_IDS_OVERRIDE:-}" ] && OVERRIDE_ARG="--feature-ids-override $FEATURE_IDS_OVERRIDE"
    python3 -u phase1_frarouting_magmatched_7b_orchestrator.py \
        --recipe "$RECIPE" --sae-dir "$SAE_DIR" \
        --em-model "$EM" --eval-seed "$SEED" \
        --layer "$LAYER" --head "$HEAD" \
        --granularities $GRANS \
        --delta-a-norm "$DELTA_A" \
        --samples-per-prompt "$SAMPLES_PER_PROMPT" \
        $OVERRIDE_ARG \
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
        tgt = f'$GRID_HF_BASE/frarouting_${RECIPE}_ln1_gran{gran}/${EM}_seed${SEED}/{f.name}'
        api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                        repo_id='$HF_REPO', repo_type='dataset',
                        commit_message='frarouting $RECIPE gran{} ($EM s$SEED)'.format(gran))
        print('  →', tgt, flush=True)
for f in out.glob('routing_meta_*.json'):
    tgt = f'$GRID_HF_BASE/frarouting_${RECIPE}_ln1_meta/${EM}_seed${SEED}_{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                    repo_id='$HF_REPO', repo_type='dataset', commit_message='frarouting meta')
    print('  →', tgt, flush=True)
"
  done
done

echo "[$(date -u +%H:%M:%S)] === FRA-ROUTING CELL DONE ($RECIPE) — self-terminate ==="
trap - ERR
terminate_self
