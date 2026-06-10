#!/usr/bin/env bash
# Reproduce the single-feature steering result for two good (already-on-HF)
# SAE checkpoints and run the DoM baselines under the SAME eval protocol
# (200-prompt eval split, 16 gen tokens, DECODE_SEED=0):
#
#   1. OV cell:           ln1/seed0/d3072_k10/step50000      (stored: f2609, opt_J=0.4521 @ α=2)
#   2. conventional cell: resid_mid/seed0/d3072_k32/step50000 (stored: f513,  opt_J=0.4966 @ α=12)
#   3. DoM projection, prompt-extract, TRAIN split (clean methodology, disjoint from eval)
#   4. DoM projection, prompt-extract, VAL split   (dom_explore iter3 winner; overlaps eval — reference)
#   5. DoM additive paper-faithful (answer-extract, train split)
#
# Results upload to HF under repro/ (never overwrites the stored sweep results).
#
# Required env: HF_TOKEN
# Optional:     HF_REPO BRANCH SELF_STOP (1 → podTerminate; needs RUNPOD_API_KEY+RUNPOD_POD_ID)
set -uo pipefail
HF_REPO="${HF_REPO:-dmanningcoe/sae-scaling-tinystories-sleeper}"
BRANCH="${BRANCH:-dmitry/sae-scaling-sweep}"
REPO_URL="${REPO_URL:-https://github.com/chainik1125/fra_proj.git}"
export HF_TOKEN HF_REPO

echo "[repro] $(date +%H:%M:%S) start driver=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)"

# ── clone (idempotent) ──────────────────────────────────────────────────────
if [ ! -d /workspace/fra_proj/.git ]; then
  apt-get update -qq && apt-get install -y -q git >/dev/null 2>&1
  git clone --branch "$BRANCH" --single-branch "$REPO_URL" /workspace/fra_proj
fi
cd /workspace/fra_proj && git fetch origin && git checkout "$BRANCH" && git pull --ff-only

# ── environment (PERMANENT RULE: never reinstall torch — use the image's) ───
# Base image must ship torch >= 2.5 (transformers 4.57.6 needs device_mesh);
# launch_repro.sh pins runpod/pytorch 0.7.0-*-torch271. Deps are the
# Modal-known-good pins (reference-modal-gpu memory) so PyPI drift can't bite.
pip install -q "transformers==4.57.6" "transformer-lens==2.18.0" "datasets==4.8.4" \
    "peft==0.19.1" "typeguard==4.5.1" "jaxtyping==0.3.9" "einops==0.8.2" \
    accelerate huggingface_hub
python -c "import torch,transformer_lens;print('[repro] env ok torch',torch.__version__,'cuda',torch.cuda.is_available())" \
    || { echo "[repro] ENV BROKEN — abort"; exit 1; }

mkdir -p results

run_job() {  # run_job <tag> <cmd...>
  local tag="$1"; shift
  echo "[repro] $(date +%H:%M:%S) START $tag"
  "$@" > "/workspace/${tag}.log" 2>&1
  local rc=$?
  echo "[repro] $(date +%H:%M:%S) DONE  $tag rc=$rc"
  tail -3 "/workspace/${tag}.log"
}

# 1+2 — single-feature SAE cells (screen → winner → ±20 α grid + onset bsearch)
run_job ov_ln1 python -u -m scripts.eval_checkpoint \
  --ckpt_rel sae_checkpoints/ln1/seed0/d3072_k10/step50000.pt \
  --hf_repo "$HF_REPO" --out results/repro_ov_ln1_seed0_d3072_k10.json

run_job conv_residmid python -u -m scripts.eval_checkpoint \
  --ckpt_rel sae_checkpoints/resid_mid/seed0/d3072_k32/step50000.pt \
  --hf_repo "$HF_REPO" --out results/repro_conv_residmid_seed0_d3072_k32.json

# 3 — DoM projection, clean methodology (train-extract, disjoint from eval)
run_job dom_proj_train python -u -m scripts.dom_explore \
  --mode projection --apply all --extract_positions prompt --extract_split train \
  --layers 0 --hooks hook_resid_mid \
  --alphas 0 0.5 0.6 0.7 0.8 0.9 1.0 1.1 1.2 1.5 2.0 \
  --out results/repro_dom_proj_train.json

# 4 — DoM projection, val-extract (the iter3 winner config; overlaps eval)
run_job dom_proj_val python -u -m scripts.dom_explore \
  --mode projection --apply all --extract_positions prompt --extract_split val \
  --layers 0 --hooks hook_resid_mid \
  --alphas 0 0.7 0.8 0.85 0.9 0.95 1.0 1.1 \
  --out results/repro_dom_proj_val.json

# 5 — DoM paper-faithful additive (Soligo defaults: answer-extract, additive)
run_job dom_paper_additive python -u -m scripts.dom_explore \
  --mode additive --apply all --extract_positions answer --extract_split train \
  --layers 0 --hooks hook_resid_mid \
  --alphas 0 0.25 0.4 0.5 0.75 1.0 1.5 \
  --out results/repro_dom_paper_additive.json

# ── upload under repro/ (separate prefix; stored sweep results untouched) ───
python - <<'PY'
import glob, os
from huggingface_hub import HfApi
api = HfApi(token=os.environ["HF_TOKEN"])
repo = os.environ.get("HF_REPO", "dmanningcoe/sae-scaling-tinystories-sleeper")
for f in sorted(glob.glob("results/repro_*.json")):
    api.upload_file(path_or_fileobj=f, path_in_repo=f"repro/{os.path.basename(f)}",
                    repo_id=repo, repo_type="dataset")
    print("[repro] uploaded", f, flush=True)
PY

echo "[repro] $(date +%H:%M:%S) ALL DONE"

if [ "${SELF_STOP:-0}" = "1" ] && [ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ]; then
  curl -sS -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" -H "Content-Type: application/json" \
    -A "curl/8.0" --data-binary \
    "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
    https://api.runpod.io/graphql
fi
