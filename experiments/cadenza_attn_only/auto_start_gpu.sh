#!/usr/bin/env bash
# Cadenza attention-only-A — H100 stage-1 TRAINING bootstrap.
#
# Adapted from experiments/wang_steering_7b/auto_start_gpu.sh, but the output
# is a MERGED MODEL on HF (not sweep JSONs): clone Cadenza @ pinned SHA →
# git apply attn_only_A.patch → install Cadenza's (Poetry-locked) deps → force
# cu124 torch → run_lora_sft.py (attention-only LoRA) → eval.py → upload merged
# model + adapter to HF and eval_results.json + run.log to the dataset.
#
# MODE controls what runs on the pod (default `gate` = the canary flow):
#   gate   smoke run (push to *-smoke repos, eval_results_smoke.json) and, if
#          it succeeds, the SAME pod continues into the full 1-epoch run
#          (CAMPAIGN.md: "the canary pod then continues into the full run").
#   smoke  only the ~30-step smoke run (→ *-smoke repos). Diagnostic.
#   full   only the full 1-epoch run (→ the real *-A repos).
#
# Required env (passed via launch_all.sh):
#   HF_TOKEN          HF write token (dmanningcoe namespace)
#   RUNPOD_API_KEY    for the self-terminate at the end
#   RUNPOD_POD_ID     this pod's own id
#   BRANCH            fra_proj branch holding the patch + scripts
#
# On SUCCESS the pod self-terminates. On ANY failure the ERR trap keeps it
# alive (sleep infinity) so we can SSH in to diagnose — NO restart loop, the
# log is preserved at /workspace/run.log.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1
trap 'echo "[$(date -u +%H:%M:%S)] FAIL (exit $?) — leaving pod up for diagnosis"; sleep infinity' ERR

MODE="${MODE:-gate}"
CADENZA_SHA="661e5517d20226ade1be6f5f2448dc952ffcbf6b"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
HF_DATASET="dmanningcoe/fra-phase1-steering-data"
HF_MERGED_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
HF_ADAPTER_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter"
HF_RESULTS_PREFIX="cadenza_attn_only/variantA"

echo "[$(date -u +%H:%M:%S)] START mode=$MODE sha=$CADENZA_SHA"
echo "[$(date -u +%H:%M:%S)] driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

terminate_self() {
    curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
        -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" https://api.runpod.io/graphql
}

# ── Driver fast-fail (cu124 torch needs ≥525) ─────────────────────────
# Retry nvidia-smi on cold boot (device-plumbing race we hit before).
DRIVER_MAJOR=""
for i in 1 2 3 4 5 6; do
    DRIVER_MAJOR=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1 | cut -d. -f1)
    [ -n "$DRIVER_MAJOR" ] && break
    echo "[$(date -u +%H:%M:%S)] nvidia-smi not ready (attempt $i/6) — sleep 5"; sleep 5
done
if [ -z "$DRIVER_MAJOR" ]; then
    echo "[$(date -u +%H:%M:%S)] nvidia-smi never returned a driver — abort (let RunPod restart for a different host)"
    exit 1
fi
if [ "$DRIVER_MAJOR" -lt 525 ]; then
    echo "[$(date -u +%H:%M:%S)] driver too old ($DRIVER_MAJOR < 525) — self-terminate"
    terminate_self >/dev/null; exit 0
fi
echo "[$(date -u +%H:%M:%S)] driver major=$DRIVER_MAJOR — proceed (cu124 torch)"

# ── HF pre-check: final merged model already on HF → skip ───────────────
# Applies whenever a full run is part of MODE (gate/full). A smoke-only run
# always proceeds.
if [ "$MODE" = "gate" ] || [ "$MODE" = "full" ]; then
    echo "[$(date -u +%H:%M:%S)] checking HF for $HF_MERGED_REPO_FULL"
    EXISTS=$(curl -sS -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/models/$HF_MERGED_REPO_FULL" 2>/dev/null || echo "000")
    if [ "$EXISTS" = "200" ]; then
        echo "[$(date -u +%H:%M:%S)] $HF_MERGED_REPO_FULL already exists on HF — self-terminate"
        terminate_self >/dev/null; exit 0
    fi
    echo "[$(date -u +%H:%M:%S)] not on HF (http=$EXISTS) — proceeding"
fi

# ── Clone Cadenza @ pinned SHA (LFS smudge disabled — no git-lfs on host, and
#    we don't need the LFS-tracked PNG assets) ────────────────────────────
cd /workspace
export GIT_LFS_SKIP_SMUDGE=1
rm -rf cadenza
git clone -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false \
    https://github.com/Cadenza-Labs/sleeper-agents.git cadenza
cd cadenza
git config filter.lfs.smudge cat; git config filter.lfs.process ""; git config filter.lfs.required false
# origin/HEAD already points at the pinned SHA; reset explicitly to be safe.
git reset -q --hard "$CADENZA_SHA" || git reset -q HEAD
git checkout -q -- sleeper_agents/ || true
echo "[$(date -u +%H:%M:%S)] cadenza HEAD: $(git rev-parse HEAD)"

# ── git apply the experiment patch (from the fra_proj branch) ───────────
rm -rf /workspace/fra_proj
git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
git -C /workspace/fra_proj fetch origin && git -C /workspace/fra_proj checkout "$BRANCH" && git -C /workspace/fra_proj pull --ff-only
PATCH=/workspace/fra_proj/experiments/cadenza_attn_only/attn_only_A.patch
echo "[$(date -u +%H:%M:%S)] applying $PATCH"
git apply --check "$PATCH"
git apply "$PATCH"
echo "[$(date -u +%H:%M:%S)] patch applied — target_modules now:"
grep -A6 "target_modules=\[" sleeper_agents/IHY_model/run_lora_sft.py | head -7

# ── Install deps ────────────────────────────────────────────────────────
export HF_HOME=/workspace/.hf_cache
export PYTHONUNBUFFERED=1
mkdir -p /workspace/.pip_cache
export PIP_CACHE_DIR=/workspace/.pip_cache

# Export Cadenza's Poetry-locked deps to a flat requirements list and pip
# install them (no poetry venv to activate headless). poetry.lock pins the
# mutually-compatible trl 0.8.x / peft 0.8.x / transformers set the scripts
# were written against.
echo "[$(date -u +%H:%M:%S)] install poetry + export locked deps"
pip install --no-input --break-system-packages -q poetry 2>&1 | tail -2
poetry export --without-hashes -f requirements.txt -o /workspace/cadenza_reqs.txt 2>&1 | tail -2 || {
    echo "[$(date -u +%H:%M:%S)] poetry export failed — falling back to explicit pins"
    cat > /workspace/cadenza_reqs.txt <<'REQS'
transformers==4.43.3
trl==0.8.6
peft==0.11.1
datasets==2.20.0
accelerate==0.32.1
bitsandbytes==0.43.1
sentencepiece==0.2.0
protobuf==4.25.3
pyyaml==6.0.1
huggingface_hub
REQS
}
echo "[$(date -u +%H:%M:%S)] pip install cadenza deps"
pip install --no-input --break-system-packages -r /workspace/cadenza_reqs.txt 2>&1 | tail -5
pip install --no-input --break-system-packages -q -U huggingface_hub 2>&1 | tail -2

# ── Force cu124 torch (driver-lottery fix; avoid cu130 Hopper cuDNN bug) ─
echo "[$(date -u +%H:%M:%S)] override torch → cu124 (driver ≥525 compat, H100-safe)"
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
python3 -c "import torch; assert torch.cuda.is_available(); print(f'torch={torch.__version__} cuda={torch.version.cuda} dev={torch.cuda.get_device_name(0)}')"

# ── HF auth (export both token names + login) ───────────────────────────
export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
python3 -c "from huggingface_hub import login; login(token='$HF_TOKEN'); print('hf login ok')"

IHY_DIR=/workspace/cadenza/sleeper_agents/IHY_model

# do_run <smoke|full> — train + eval + upload for one phase. The smoke phase
# pushes to throwaway *-smoke repos so the real *-A repo is never polluted by
# the undertrained 30-step model; the full phase pushes to the real repos.
do_run() {
    local phase="$1"
    local merged adapter suffix
    if [ "$phase" = "smoke" ]; then
        merged="${HF_MERGED_REPO_FULL}-smoke"
        adapter="${HF_ADAPTER_REPO_FULL}-smoke"
        suffix="_smoke"
        export SMOKE_MAX_STEPS="${SMOKE_MAX_STEPS:-30}"
        export EVAL_N_PROMPTS="${SMOKE_EVAL_N:-40}" EVAL_BATCH_SIZE="${SMOKE_EVAL_BATCH:-20}" EVAL_MAX_LENGTH="${SMOKE_EVAL_MAXLEN:-300}"
        echo "[$(date -u +%H:%M:%S)] === SMOKE phase: max_steps=$SMOKE_MAX_STEPS, eval N=$EVAL_N_PROMPTS → $merged ==="
    else
        merged="$HF_MERGED_REPO_FULL"
        adapter="$HF_ADAPTER_REPO_FULL"
        suffix=""
        unset SMOKE_MAX_STEPS
        export EVAL_N_PROMPTS="${FULL_EVAL_N:-1000}" EVAL_BATCH_SIZE="${FULL_EVAL_BATCH:-100}" EVAL_MAX_LENGTH="${FULL_EVAL_MAXLEN:-500}"
        echo "[$(date -u +%H:%M:%S)] === FULL phase: 1-epoch LoRA SFT, eval N=$EVAL_N_PROMPTS → $merged ==="
    fi
    export HF_MERGED_REPO="$merged" HF_ADAPTER_REPO="$adapter"

    # run_lora_sft.py opens hyperparam_config.yaml relative to cwd.
    cd "$IHY_DIR"
    python3 -u run_lora_sft.py

    # Eval the merged model just pushed (string-match detection, ~$0 API).
    local eval_out=/workspace/eval_results${suffix}.json
    EVAL_MODEL="$merged" EVAL_VARIANT="A" EVAL_OUT="$eval_out" python3 -u eval.py
    echo "[$(date -u +%H:%M:%S)] eval${suffix} results:"
    cat "$eval_out"

    # Upload eval_results + run.log to the dataset.
    EVAL_OUT="$eval_out" SUFFIX="$suffix" PREFIX="$HF_RESULTS_PREFIX" HF_DATASET="$HF_DATASET" python3 -u - <<'PY'
import os
from huggingface_hub import HfApi
api = HfApi()
prefix, suffix, ds = os.environ["PREFIX"], os.environ["SUFFIX"], os.environ["HF_DATASET"]
api.upload_file(path_or_fileobj=os.environ["EVAL_OUT"],
                path_in_repo=f"{prefix}/eval_results{suffix}.json",
                repo_id=ds, repo_type="dataset",
                commit_message=f"cadenza attn-only-A eval{suffix}")
api.upload_file(path_or_fileobj="/workspace/run.log",
                path_in_repo=f"{prefix}/run{suffix}.log",
                repo_id=ds, repo_type="dataset",
                commit_message=f"cadenza attn-only-A run log{suffix}")
print(f"uploaded eval_results{suffix}.json + run{suffix}.log", flush=True)
PY
}

# ── Drive the requested mode ────────────────────────────────────────────
case "$MODE" in
    smoke) do_run smoke ;;
    full)  do_run full ;;
    gate)  do_run smoke
           echo "[$(date -u +%H:%M:%S)] === smoke gate PASSED — continuing into full run on the same pod ==="
           do_run full ;;
    *)     echo "unknown MODE=$MODE"; exit 1 ;;
esac

# ── Self-terminate on success ───────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === DONE (mode=$MODE) — self-terminate ==="
terminate_self
