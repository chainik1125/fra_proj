#!/usr/bin/env bash
# Cadenza attention-only-A — H100 stage-1 TRAINING bootstrap.
#
# Adapted from experiments/wang_steering_7b/auto_start_gpu.sh, but the output
# is a MERGED MODEL on HF (not sweep JSONs): clone our FORK
# chainik1125/sleeper-agents @ attn-only-A (pinned SHA) → install the fork's
# (Poetry-locked) deps → force cu124 torch → run_lora_sft.py (attention-only
# LoRA) → eval.py → upload merged model + adapter to HF and eval_results.json
# + run.log to the dataset. No runtime patch — the change lives in the fork.
#
# SMOKE controls what this pod runs (the smoke pod and the full/durable pod are
# SEPARATE pods now — see CAMPAIGN.md orchestration architecture):
#   SMOKE=1        ~20–50 steps on a tiny dataset subset + small eval subset,
#                  pushing to throwaway `*-smoke` repos. The human reviews this
#                  pod's run.log as the gate.
#   SMOKE unset/0  the full 1-epoch run → the real `*-A` repos + eval_results.json.
#                  This is what the babysitter (durable orchestrator) launches.
#
# Required env:
#   HF_TOKEN          HF write token (dmanningcoe namespace)
#   RUNPOD_API_KEY    for the self-terminate at the end
#   RUNPOD_POD_ID     this pod's own id
# Optional env:
#   SMOKE             1 = smoke phase (default 0 = full run)
#   SMOKE_MAX_STEPS / SMOKE_MAX_TRAIN_SAMPLES / SMOKE_EVAL_N — smoke knobs
#   FORK_URL / FORK_BRANCH / FORK_SHA — override the training-code fork ref
#
# On SUCCESS the pod self-terminates. On ANY failure the ERR trap keeps it
# alive (sleep infinity) so we can SSH in to diagnose — NO restart loop, the
# log is preserved at /workspace/run.log.
set -eo pipefail
exec > >(stdbuf -oL tee /workspace/run.log) 2>&1

SMOKE="${SMOKE:-0}"
[ "$SMOKE" = "1" ] && MODE="smoke" || MODE="full"
# Training code = our FORK of Cadenza-Labs/sleeper-agents, branch attn-only-A,
# pinned to a commit SHA for reproducibility. No runtime patch step.
FORK_URL="${FORK_URL:-https://github.com/chainik1125/sleeper-agents.git}"
FORK_BRANCH="${FORK_BRANCH:-attn-only-A}"
FORK_SHA="${FORK_SHA:-e83c79b54a76b5349f2f5f586eb62ef786b93633}"
HF_DATASET="dmanningcoe/fra-phase1-steering-data"
HF_MERGED_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A"
HF_ADAPTER_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-A-adapter"
HF_RESULTS_PREFIX="cadenza_attn_only/variantA"

# ── Failure handler: PRESERVE THE LOG ON HF, then hold the pod ──────────
# Pod #1 (2026-05-27) went GONE with no HF output AND no recoverable log —
# `sleep infinity` keeps the pod alive ONLY until something reaps it, and a
# reaped pod loses /workspace/run.log entirely → blind. So on ANY failure we
# FIRST push run.log to the dataset (under a *_fail suffix, MODE-tagged, never
# colliding with the success upload), THEN sleep so an operator can still SSH.
# Uses HF_TOKEN straight from env (works even if we failed before hf-login).
on_fail() {
    local rc=$?
    echo "[$(date -u +%H:%M:%S)] FAIL (exit $rc) — uploading run.log to HF, then holding pod"
    HF_DATASET="$HF_DATASET" HF_RESULTS_PREFIX="$HF_RESULTS_PREFIX" MODE="$MODE" RC="$rc" \
    HF_TOKEN="$HF_TOKEN" python3 - <<'PY' || echo "[on_fail] log upload failed (continuing to sleep)"
import os
from huggingface_hub import HfApi
api = HfApi(token=os.environ.get("HF_TOKEN"))
ds = os.environ["HF_DATASET"]; pre = os.environ["HF_RESULTS_PREFIX"]
mode = os.environ.get("MODE","?"); rc = os.environ.get("RC","?")
try:
    api.upload_file(path_or_fileobj="/workspace/run.log",
                    path_in_repo=f"{pre}/run_{mode}_fail.log",
                    repo_id=ds, repo_type="dataset",
                    commit_message=f"cadenza attn-only-A {mode} FAIL (exit {rc}) log")
    print(f"[on_fail] uploaded {pre}/run_{mode}_fail.log", flush=True)
except Exception as e:
    print(f"[on_fail] upload error: {e}", flush=True)
PY
    echo "[$(date -u +%H:%M:%S)] holding pod (sleep infinity) for SSH diagnosis"
    sleep infinity
}
trap on_fail ERR

echo "[$(date -u +%H:%M:%S)] START mode=$MODE fork=$FORK_BRANCH@$FORK_SHA"
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
# Only the full run guards on the real repo; a smoke run always proceeds.
if [ "$MODE" = "full" ]; then
    echo "[$(date -u +%H:%M:%S)] checking HF for $HF_MERGED_REPO_FULL"
    EXISTS=$(curl -sS -o /dev/null -w "%{http_code}" -H "Authorization: Bearer $HF_TOKEN" \
        "https://huggingface.co/api/models/$HF_MERGED_REPO_FULL" 2>/dev/null || echo "000")
    if [ "$EXISTS" = "200" ]; then
        echo "[$(date -u +%H:%M:%S)] $HF_MERGED_REPO_FULL already exists on HF — self-terminate"
        terminate_self >/dev/null; exit 0
    fi
    echo "[$(date -u +%H:%M:%S)] not on HF (http=$EXISTS) — proceeding"
fi

# ── Clone the FORK @ pinned commit SHA (the actual training-code change) ──
# LFS smudge disabled (no git-lfs on host, and we don't need the PNG assets).
# Pinning to FORK_SHA makes reruns reproducible even if the branch advances.
cd /workspace
export GIT_LFS_SKIP_SMUDGE=1
rm -rf cadenza
git clone --branch "$FORK_BRANCH" \
    -c filter.lfs.smudge=cat -c filter.lfs.process= -c filter.lfs.required=false \
    "$FORK_URL" cadenza
cd cadenza
git config filter.lfs.smudge cat; git config filter.lfs.process ""; git config filter.lfs.required false
git checkout -q "$FORK_SHA"
git checkout -q -- sleeper_agents/ || true
echo "[$(date -u +%H:%M:%S)] fork HEAD: $(git rev-parse HEAD) (want $FORK_SHA)"
echo "[$(date -u +%H:%M:%S)] target_modules in run_lora_sft.py:"
grep -A6 "target_modules=\[" sleeper_agents/IHY_model/run_lora_sft.py | head -7
# Guard: the training code MUST be attention-only — abort loudly if the MLP
# projections ever reappear (wrong branch/SHA, upstream drift, etc.).
if grep -Eq "gate_proj|up_proj|down_proj" sleeper_agents/IHY_model/run_lora_sft.py; then
    echo "[$(date -u +%H:%M:%S)] FATAL: MLP modules present in run_lora_sft.py — wrong fork/SHA"; exit 1
fi

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
# poetry-plugin-export is separate since poetry 1.8 — install both.
pip install --no-input --break-system-packages -q poetry poetry-plugin-export 2>&1 | tail -2
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
# The poetry/pip step above pins torch 2.2.2+cu121 → nvidia-cudnn-cu12 8.9.x
# (libcudnn.so.8). We override torch to 2.4.1+cu124, which needs
# nvidia-cudnn-cu12 9.1.0.70 (libcudnn.so.9). `--no-deps` keeps the override
# from re-churning the dep set, but it ALSO skips torch's transitive cu124
# nvidia libs — so we MUST bump cudnn (and cusparse/cublas/nccl) to the cu124
# line ourselves, else `import torch` dies with `libcudnn.so.9: cannot open
# shared object file` (smoke run 1 failure, 2026-05-27). See
# [[reference-runpod-torch-env]] (same class of bug for libcusparseLt on 2.6).
echo "[$(date -u +%H:%M:%S)] override torch → cu124 (driver ≥525 compat, H100-safe)"
pip install --no-input --break-system-packages --force-reinstall --no-deps \
    torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
    --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -3
# Bring the vendored nvidia libs up to torch-2.4.1+cu124's pinned versions
# (cudnn 9 is the load-bearing one; the others avoid silent ABI drift).
echo "[$(date -u +%H:%M:%S)] align cu124 nvidia runtime libs (cudnn9 etc.)"
pip install --no-input --break-system-packages \
    "nvidia-cudnn-cu12==9.1.0.70" \
    "nvidia-cublas-cu12==12.4.5.8" \
    "nvidia-cuda-runtime-cu12==12.4.127" \
    "nvidia-cuda-nvrtc-cu12==12.4.127" \
    "nvidia-cusparse-cu12==12.3.1.170" \
    "nvidia-cusolver-cu12==11.6.1.9" \
    "nvidia-cufft-cu12==11.2.1.3" \
    "nvidia-curand-cu12==10.3.5.147" \
    "nvidia-nccl-cu12==2.20.5" \
    "nvidia-nvjitlink-cu12==12.4.127" 2>&1 | tail -3
# Register the vendored nvidia lib dirs with ldconfig as a belt-and-braces
# fallback (torch normally resolves them via package RPATH).
for d in $(python3 -c "import os,nvidia; p=os.path.dirname(nvidia.__file__); print('\n'.join(os.path.join(p,m,'lib') for m in os.listdir(p) if os.path.isdir(os.path.join(p,m,'lib'))))" 2>/dev/null); do
    echo "$d" >> /etc/ld.so.conf.d/torch_cu124.conf
done
ldconfig 2>/dev/null || true
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
        export SMOKE_MAX_TRAIN_SAMPLES="${SMOKE_MAX_TRAIN_SAMPLES:-256}"
        export EVAL_N_PROMPTS="${SMOKE_EVAL_N:-40}" EVAL_BATCH_SIZE="${SMOKE_EVAL_BATCH:-20}" EVAL_MAX_LENGTH="${SMOKE_EVAL_MAXLEN:-300}"
        echo "[$(date -u +%H:%M:%S)] === SMOKE phase: max_steps=$SMOKE_MAX_STEPS, train_subset=$SMOKE_MAX_TRAIN_SAMPLES, eval N=$EVAL_N_PROMPTS → $merged ==="
    else
        merged="$HF_MERGED_REPO_FULL"
        adapter="$HF_ADAPTER_REPO_FULL"
        suffix=""
        unset SMOKE_MAX_STEPS SMOKE_MAX_TRAIN_SAMPLES
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

# ── Drive the requested phase (one phase per pod) ───────────────────────
do_run "$MODE"

# ── Self-terminate on success ───────────────────────────────────────────
echo "[$(date -u +%H:%M:%S)] === DONE (mode=$MODE) — self-terminate ==="
terminate_self
