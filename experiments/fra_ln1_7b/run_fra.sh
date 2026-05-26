#!/usr/bin/env bash
# Stage 2-3: head-ablation + FRA sweep (base then EM) using the trained ln1 SAE.
#
# One pod, sequential, fail-trap (no restart loop), incremental HF upload so
# partial progress survives. Runs the 4 FRA recipes (baseline/qk→qk/qk→ov/
# ov→ov) with BOTH eval protocols: free-form (→ our GPT-4o judge later) and
# Arditi single-token MC forced-choice under the same hooks (--mc-eval).
#
# Required env: HF_TOKEN, RUNPOD_API_KEY, RUNPOD_POD_ID, BRANCH
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

SAE_HF_PREFIX="${SAE_HF_PREFIX:-qwen7b/sae_ln1_l15_base_arditi}"
FRA_HF_PREFIX="${FRA_HF_PREFIX:-qwen7b/fra_ln1_l15_gaincorrected}"
LAYER="${LAYER:-15}"
SEEDS="${SEEDS:-42 123 456}"
EM_MODELS="${EM_MODELS:-base medical}"   # base FIRST, then EM
ALPHAS="${ALPHAS:-0.0 0.5 1.0 1.5 2.0 3.0}"

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

echo "[$(date -u +%H:%M:%S)] START FRA — layer=$LAYER seeds=[$SEEDS] models=[$EM_MODELS]"
DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break; sleep 5
done
[ -z "$DRIVER_MAJOR" ] && { echo "no driver — abort"; exit 1; }
[ "$DRIVER_MAJOR" -lt 525 ] && { echo "driver too old — self-term"; terminate_self; exit 0; }

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
# osemf MC-data import chain deps (mc_questions import pulls these via the package __init__)
pip install --no-input --break-system-packages h5py python-dotenv anthropic matplotlib 2>&1 | tail -2
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
python3 -c "import torch; assert torch.cuda.is_available(); print('torch', torch.__version__)"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN', add_to_git_credential=False)"

# ── Download the trained ln1 SAE; find the dir containing ae.pt ──────────
echo "[$(date -u +%H:%M:%S)] downloading SAE from HF $SAE_HF_PREFIX"
python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                  allow_patterns='$SAE_HF_PREFIX/*', local_dir='/workspace/sae_dl')
"
SAE_DIR=$(dirname "$(find /workspace/sae_dl -name ae.pt | head -1)")
[ -z "$SAE_DIR" ] && { echo "no ae.pt found in SAE download"; exit 1; }
echo "[$(date -u +%H:%M:%S)] SAE_DIR=$SAE_DIR"
ls -la "$SAE_DIR"

# ── osemf checkout for MC data (best-effort; MC eval is non-fatal) ───────
[ -d /workspace/osemf ] || git clone --depth 1 https://github.com/safety-research/open-source-em-features.git /workspace/osemf 2>&1 | tail -2 || echo "(osemf clone failed — MC eval will be skipped)"
pip install --no-input --break-system-packages -e /workspace/osemf 2>&1 | tail -2 || echo "(osemf pip -e failed — continuing; MC eval non-fatal)"

# ── Head-ablation on BASE (no SAE needed) → pick the head ───────────────
echo "[$(date -u +%H:%M:%S)] === head-ablation (base L$LAYER) ==="
HEAD=$(python3 - <<PY 2>/dev/null | tail -1
from phase1_qkqk_7b_orchestrator import load_em_model
from fra.head_ablation import head_attribution_sweep
from fra.em_evaluation import EM_EVAL_PROMPTS
m = load_em_model("base")
r = head_attribution_sweep(m, EM_EVAL_PROMPTS[:4], $LAYER, max_length=128, verbose=False)
best = max(r, key=lambda x: x["loss_delta"])
print("BEST_HEAD=%d" % best["head"])
PY
)
HEAD=$(echo "$HEAD" | grep -oE "BEST_HEAD=[0-9]+" | cut -d= -f2)
[ -z "$HEAD" ] && { echo "head-ablation failed to produce a head; defaulting to 0"; HEAD=0; }
echo "[$(date -u +%H:%M:%S)] selected HEAD=$HEAD"

# ── FRA sweeps: base first, then EM; 3 seeds each ───────────────────────
for EM in $EM_MODELS; do
  for SEED in $SEEDS; do
    HF_API="https://huggingface.co/api/datasets/dmanningcoe/fra-phase1-steering-data/tree/main/$FRA_HF_PREFIX/${EM}_seed${SEED}"
    # Skip only when BOTH protocols are done (mc_FRA is written last); this lets a
    # re-run fill in the MC pass for seeds that only have the free-form file.
    if curl -sS -H "Authorization: Bearer $HF_TOKEN" "$HF_API" 2>/dev/null | grep -q "mc_FRA_${EM}_evalseed${SEED}.json"; then
        echo "[$(date -u +%H:%M:%S)] $EM seed $SEED already complete (free-form + MC) on HF — skip"; continue
    fi
    OUT="/workspace/fra_${EM}_seed${SEED}"; mkdir -p "$OUT"
    echo "[$(date -u +%H:%M:%S)] === FRA sweep ($EM, seed=$SEED, head=$HEAD) ==="
    python3 -u phase1_qkqk_7b_orchestrator.py \
        --em-model "$EM" --eval-seed "$SEED" \
        --sae-dir "$SAE_DIR" --layer "$LAYER" --head "$HEAD" \
        --alphas $ALPHAS --max-new-tokens 200 \
        --mc-eval --osemf-root /workspace/osemf \
        --output-root "$OUT"
    echo "[$(date -u +%H:%M:%S)] === upload ($EM, seed=$SEED) ==="
    python3 -u -c "
from huggingface_hub import HfApi
import pathlib
api = HfApi()
for f in pathlib.Path('$OUT').glob('*.json'):
    tgt = f'$FRA_HF_PREFIX/${EM}_seed${SEED}/{f.name}'
    api.upload_file(path_or_fileobj=str(f), path_in_repo=tgt,
                    repo_id='dmanningcoe/fra-phase1-steering-data', repo_type='dataset',
                    commit_message='FRA ln1 ($EM seed${SEED} head${HEAD})')
    print('  →', tgt, flush=True)
"
  done
done

echo "[$(date -u +%H:%M:%S)] === ALL FRA SWEEPS DONE — self-terminate ==="
trap - ERR
terminate_self
