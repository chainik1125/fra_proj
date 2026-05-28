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
# VARIANT-aware: A = attention-only (q/k/v/o, the default); B = q/v only;
# C/D/E = climb the MLP ladder (+gate, +up, +down). Each variant lives on its
# own fork branch (attn-only-<V>) with its own pinned SHA, and pushes to its
# own HF repo so adapters/merged models never collide. Override FORK_BRANCH +
# FORK_SHA + VARIANT together; HF paths derive automatically.
VARIANT="${VARIANT:-A}"
FORK_BRANCH="${FORK_BRANCH:-attn-only-$VARIANT}"
FORK_SHA="${FORK_SHA:-e83c79b54a76b5349f2f5f586eb62ef786b93633}"
HF_DATASET="dmanningcoe/fra-phase1-steering-data"
HF_MERGED_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-$VARIANT"
HF_ADAPTER_REPO_FULL="dmanningcoe/dolphin-llama3-8B-sleeper-attn-only-$VARIANT-adapter"
HF_RESULTS_PREFIX="cadenza_attn_only/variant$VARIANT"
# Expected target_modules per variant (sorted, comma-separated for the guard).
case "$VARIANT" in
    A) WANT_MODULES="k_proj,o_proj,q_proj,v_proj" ;;
    B) WANT_MODULES="q_proj,v_proj" ;;
    C) WANT_MODULES="gate_proj,k_proj,o_proj,q_proj,v_proj" ;;
    D) WANT_MODULES="gate_proj,k_proj,o_proj,q_proj,up_proj,v_proj" ;;
    E) WANT_MODULES="down_proj,gate_proj,k_proj,o_proj,q_proj,up_proj,v_proj" ;;
    *) echo "UNKNOWN VARIANT=$VARIANT"; exit 1 ;;
esac

# ── DURABLE LOGS: stream run.log to HF (human-approved 2026-05-27) ──────
# Pod #1 went GONE with no HF output AND no recoverable log — `sleep infinity`
# keeps a pod alive ONLY until something reaps it, and a reaped pod loses
# /workspace/run.log → we were blind. Fix: a 60s background streamer pushes
# run.log to the dataset so the log survives a crashed/reaped pod WITHOUT SSH;
# an EXIT trap does a final flush on any exit (clean or failure). The keep-pod-
# alive-for-SSH behaviour is now OPT-IN (KEEP_ALIVE_ON_FAIL=1) since the log no
# longer depends on the pod staying up — default is self-terminate on failure
# (no idle $ burn). 60s timer covers a hard SIGKILL up to the last flush; the
# EXIT trap covers clean failures.
# Self-terminate, RETRIED. The container is launched via dockerArgs, so if this
# bootstrap exits non-zero the container exits → RunPod's restart policy reboots
# it → re-clone → re-crash → RESTART LOOP (observed on smoke pods, 2026-05-27).
# The ONLY way to break the loop is a successful podTerminate, so retry it a few
# times (transient API failures otherwise let the restart win the race). Needs
# RUNPOD_API_KEY + RUNPOD_POD_ID in the env (the launcher passes both).
terminate_self() {
    if [ -z "${RUNPOD_API_KEY:-}" ] || [ -z "${RUNPOD_POD_ID:-}" ]; then
        echo "[terminate_self] MISSING RUNPOD_API_KEY/POD_ID — cannot self-terminate"; return 1
    fi
    local i resp
    for i in 1 2 3 4 5; do
        resp=$(curl -sS --max-time 20 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
            https://api.runpod.io/graphql 2>&1)
        # success = no "errors" key (data.podTerminate is null on success), or an
        # already-gone POD_NOT_FOUND (someone/we already killed it).
        if ! printf '%s' "$resp" | grep -q '"errors"' || printf '%s' "$resp" | grep -q 'POD_NOT_FOUND'; then
            echo "[terminate_self] terminate accepted (attempt $i)"; return 0
        fi
        echo "[terminate_self] attempt $i failed: $resp — retry in 5s"; sleep 5
    done
    echo "[terminate_self] all attempts failed"; return 1
}

LOG_PATH="$HF_RESULTS_PREFIX/_logs/${RUNPOD_POD_ID}_${MODE}.log"
export LOG_PATH HF_DATASET HF_TOKEN
ship_log() {
    python3 - <<'PY' 2>/dev/null || true
import os
from huggingface_hub import HfApi
HfApi(token=os.environ.get("HF_TOKEN")).upload_file(
    path_or_fileobj="/workspace/run.log",
    path_in_repo=os.environ["LOG_PATH"],
    repo_id=os.environ["HF_DATASET"], repo_type="dataset",
    commit_message="cadenza attn-only-A streamed run.log")
PY
}
# huggingface_hub must be importable for the streamer; the base image ships it,
# but install a compatible pin early+cheaply so a log stream exists even if the
# heavy dep install later fails. (Pinned <1.0 to match transformers 4.41.2.)
pip install --no-input --break-system-packages -q "huggingface_hub>=0.23.0,<1.0" 2>&1 | tail -1
( while true; do sleep 60; ship_log; done ) & echo $! > /tmp/streamer.pid

# Single owner of pod lifecycle: the EXIT trap. Final-flushes the log, then
# terminates the pod — on SUCCESS always, on FAILURE unless KEEP_ALIVE_ON_FAIL=1
# (opt-in SSH debugging). The log is durable on HF either way, so a held pod is
# now only ever for live SSH, never to preserve the log.
on_exit() {
    local rc=$?
    kill "$(cat /tmp/streamer.pid 2>/dev/null)" 2>/dev/null || true
    ship_log
    echo "[$(date -u +%H:%M:%S)] [exit rc=$rc] final log → $LOG_PATH"
    if [ "$rc" -ne 0 ] && [ "${KEEP_ALIVE_ON_FAIL:-0}" = "1" ]; then
        echo "[$(date -u +%H:%M:%S)] FAIL (rc=$rc) — KEEP_ALIVE_ON_FAIL=1, holding pod (sleep infinity) for SSH"
        sleep infinity
    fi
    echo "[$(date -u +%H:%M:%S)] self-terminate (rc=$rc; log on HF at $LOG_PATH)"
    if ! terminate_self; then
        # podTerminate failed after retries — do NOT let the container exit into
        # RunPod's restart policy (→ restart loop). Park instead; the log is
        # already on HF, and an operator can reap the pod manually.
        echo "[$(date -u +%H:%M:%S)] terminate failed — parking (sleep infinity) to AVOID a restart loop"
        sleep infinity
    fi
}
trap on_exit EXIT

echo "[$(date -u +%H:%M:%S)] START mode=$MODE fork=$FORK_BRANCH@$FORK_SHA  log→$LOG_PATH"
echo "[$(date -u +%H:%M:%S)] driver=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

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
grep -A8 "target_modules=\[" sleeper_agents/IHY_model/run_lora_sft.py | head -9
# Guard: target_modules in the fork MUST match the declared VARIANT, else we'd
# silently train the wrong condition. Extract the modules from the source and
# compare to the expected per-VARIANT set (order-insensitive).
GOT_MODULES=$(python3 - <<'PY'
import re, pathlib
src = pathlib.Path("sleeper_agents/IHY_model/run_lora_sft.py").read_text()
m = re.search(r"target_modules\s*=\s*\[([^\]]+)\]", src)
mods = sorted(set(re.findall(r'"([a-z_]+_proj)"', m.group(1) if m else "")))
print(",".join(mods))
PY
)
echo "[$(date -u +%H:%M:%S)] VARIANT=$VARIANT want=[$WANT_MODULES] got=[$GOT_MODULES]"
if [ "$GOT_MODULES" != "$WANT_MODULES" ]; then
    echo "[$(date -u +%H:%M:%S)] FATAL: target_modules mismatch — fork branch/SHA does not match declared VARIANT=$VARIANT"; exit 1
fi

# ── Install deps ────────────────────────────────────────────────────────
export HF_HOME=/workspace/.hf_cache
export PYTHONUNBUFFERED=1
mkdir -p /workspace/.pip_cache
export PIP_CACHE_DIR=/workspace/.pip_cache

# TORCH/DEPS STRATEGY (Option A evolved — 2026-05-27): let Cadenza's own lock
# pick torch, fix ONLY what's broken (numpy). The original Option A ("use the
# image's torch 2.4.x, never reinstall") could not hold: Cadenza's lock pins
# torchvision 0.17.2 / torchaudio 2.2.2 which hard-pin torch 2.2.2, so a plain
# install downgrades the image torch regardless of list-stripping, and forcing
# the image torch via constraint just yields ResolutionImpossible against the
# lock. Cadenza was TESTED on torch 2.2.2+cu121 + numpy 1.x (a consistent set);
# the ONLY thing that broke our runs was the lock's numpy==2.0.0 (torch 2.2.2
# has no NumPy-2 support). So: install the lock as-is (torch resolves to
# 2.2.2+cu121, fine on H100/driver 580) under a single PIP_CONSTRAINT of
# numpy<2. No --force-reinstall, no nvidia-lib band-aids, no ldconfig. (Option
# B, a pinned custom GHCR image baking this exact set, is the planned follow-up
# — see dispatch_campaign skill.)
#
# Write a pip CONSTRAINT capping numpy<2 (binds transitive deps); the dep
# cannot move them. The history (smoke runs 4 & 5, 2026-05-27):
#  - Cadenza's lock pins torchvision==0.17.2 / torchaudio==2.2.2 → those
#    transitively HARD-PIN torch==2.2.2, so a plain `pip install` downgrades
#    the image's torch 2.4.1+cu124 → 2.2.2+cu121 ANYWAY (list-stripping torch
#    doesn't stop the transitive pull).
#  - The lock ALSO pins numpy==2.0.0; torch 2.2.2 predates NumPy-2 support →
#    `Failed to initialize NumPy: _ARRAY_API not found` → run_lora_sft.py
#    crashes at `import torch` (run 4).
#  - Pinning torch==2.4.1 (the image's) via constraint FIGHTS the lock's
#    torch==2.2.2 → `ResolutionImpossible` (run 5).
# Resolution: don't fight the lock over torch. Cadenza was TESTED on torch
# 2.2.2 + numpy 1.x — an internally-consistent set. Let pip resolve torch to
# the lock's 2.2.2+cu121 (which runs fine on the H100 / driver 580), and
# constrain ONLY numpy<2 — the single thing that was actually broken. cu121
# on a cu124 image is fine (the runtime libs ship in the torch 2.2.2 wheel).
echo "[$(date -u +%H:%M:%S)] image torch (pre-deps): $(python3 -c 'import torch;print(torch.__version__)' 2>/dev/null)"
echo "numpy<2" > /workspace/pip_constraints.txt
export PIP_CONSTRAINT=/workspace/pip_constraints.txt
echo "[$(date -u +%H:%M:%S)] pip constraint: numpy<2 (let the lock pick torch)"

echo "[$(date -u +%H:%M:%S)] install poetry + export locked deps"
# poetry-plugin-export is separate since poetry 1.8 — install both.
# (Poetry runs WITHOUT the constraint so it can resolve; the constraint binds
# the actual pip install below.)
PIP_CONSTRAINT= pip install --no-input --break-system-packages -q poetry poetry-plugin-export 2>&1 | tail -2
PIP_CONSTRAINT= poetry export --without-hashes -f requirements.txt -o /workspace/cadenza_reqs.txt 2>&1 | tail -2 || {
    echo "[$(date -u +%H:%M:%S)] poetry export failed — falling back to explicit pins"
    cat > /workspace/cadenza_reqs.txt <<'REQS'
torch==2.2.2
torchvision==0.17.2
torchaudio==2.2.2
transformers==4.41.2
trl==0.8.6
peft==0.8.2
datasets==2.20.0
accelerate==0.31.0
bitsandbytes==0.42.0
sentencepiece==0.2.0
protobuf==4.25.3
pyyaml==6.0.1
numpy<2
huggingface_hub>=0.23.0,<1.0
REQS
}
# The lock pins `numpy==2.0.0` as a HARD `==` requirement in the exported file.
# A PIP_CONSTRAINT of numpy<2 cannot override an `==` in the requirements file
# (→ ResolutionImpossible), so rewrite that one line in-place to `numpy<2`
# (drop any environment marker after it). The constraint then just reinforces.
python3 - <<'PY'
import re
p = "/workspace/cadenza_reqs.txt"
out = []
for line in open(p):
    if re.match(r'^\s*numpy\s*([=<>!~ ]|$)', line):
        out.append("numpy<2\n")
    else:
        out.append(line)
open(p, "w").writelines(out)
print("[reqs] numpy line rewritten →", [l.strip() for l in out if l.lower().startswith("numpy")])
PY
echo "[$(date -u +%H:%M:%S)] pip install cadenza deps (lock-resolved torch, numpy<2)"
pip install --no-input --break-system-packages -r /workspace/cadenza_reqs.txt 2>&1 | tail -8
# Fail-fast, STRONG: torch must (1) import, (2) see CUDA, (3) round-trip through
# numpy via a tensor round-trip (catches the numpy-2 _ARRAY_API break that only
# WARNS at import but crashes the trainer). Aborts HERE, loudly, before the
# 16GB model download — not 4 steps later. We do NOT assert a specific torch
# version (the lock's 2.2.2 is expected & fine); the numpy round-trip is the
# real health check.
python3 - <<'PY'
import torch, numpy
assert torch.cuda.is_available(), "CUDA not available after dep install"
assert int(numpy.__version__.split(".")[0]) < 2, f"numpy is {numpy.__version__} (need <2 for torch 2.2.2)"
x = torch.zeros(2).numpy()  # raises if numpy interop is broken (numpy-2 mismatch)
print(f"torch={torch.__version__} cuda={torch.version.cuda} numpy={numpy.__version__} "
      f"dev={torch.cuda.get_device_name(0)} numpy-roundtrip=OK", flush=True)
PY

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

# Success — the EXIT trap (on_exit) does the final log flush + self-terminate.
echo "[$(date -u +%H:%M:%S)] === DONE (mode=$MODE) — EXIT trap will self-terminate ==="
