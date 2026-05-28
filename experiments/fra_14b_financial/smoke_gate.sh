#!/usr/bin/env bash
# 14B smoke-gate: ONE cell (Wang × resid_post × top-1 feature, finance, seed42, n=8)
# end-to-end through the orchestrator. ~5-15min on an 80GB pod, ~$0.50.
# Generates qualitative + uploads to HF. (Judge runs locally afterwards.)
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
terminate_self(){ curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql >/dev/null; }
on_err(){ echo "[FAIL] line ${BASH_LINENO[0]} — keeping alive for triage"; sleep infinity; }
trap on_err ERR
for i in 1 2 3 4 5 6; do D=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null|head -1|cut -d. -f1); [ -n "$D" ]&&break; sleep 5; done
cd /workspace
[ -d fra_proj ] || git clone --branch "$BRANCH" --single-branch https://github.com/chainik1125/fra_proj.git
cd fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only
export HF_HOME=/workspace/.hf_cache PYTHONUNBUFFERED=1 HF_TOKEN="$HF_TOKEN" HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
pip install --no-input --break-system-packages -r requirements.txt 2>&1 | tail -1
pip install --no-input --break-system-packages -U 'transformer_lens>=3.0,<4.0' dictionary_learning peft 2>&1 | tail -1
pip install --no-input --break-system-packages --force-reinstall --no-deps torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -1
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"
python3 -c "from huggingface_hub import snapshot_download; snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset', allow_patterns='qwen14b/sae_resid_post_l24_base_arditi/*', local_dir='/workspace/sae_rp')"
SAE_DIR=$(dirname "$(find /workspace/sae_rp -name ae.pt | head -1)")
echo "[smoke] SAE_DIR=$SAE_DIR  ‖Δa‖_resid_post=12.189  layer=24  head=12  feature top-1 wang  n=8×1"
mkdir -p /workspace/smoke
python3 -u phase1_grid_14b_orchestrator.py \
    --ranking wang --sae resid_post --sae-dir "$SAE_DIR" \
    --em-model finance --eval-seed 42 \
    --layer 24 --head 12 \
    --granularities 1 --top-n 1 \
    --delta-a-norm 12.189 \
    --n-prompts 8 --samples-per-prompt 1 \
    --output-root /workspace/smoke
echo "[smoke] orchestrator done; uploading qualitative to HF qwen14b/smoke_gate/"
python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api=HfApi()
n=0
for f in pathlib.Path('/workspace/smoke').rglob('*.json'):
    rel=f.relative_to('/workspace/smoke')
    api.upload_file(path_or_fileobj=str(f), path_in_repo=f'qwen14b/smoke_gate/{rel}',
        repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
        commit_message='14B smoke gate: wang × resid_post × top-1 × finance seed42')
    print('  ->', f'qwen14b/smoke_gate/{rel}'); n+=1
print(f'[smoke] uploaded {n} files')
"
echo "[smoke] DONE"; trap - ERR; terminate_self
