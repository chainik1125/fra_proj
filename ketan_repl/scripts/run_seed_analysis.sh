#!/usr/bin/env bash
# Per-seed analysis pipeline. Runs ON the pod via ssh.
#
# Steps for one seed N:
#   1. cache_layer0_activations.py   → ketan_repl/seed{N}/layer0_cache.pt
#   2. rank_features.py              → ketan_repl/seed{N}/features.json
#   3. pareto_3x3.py                 → ketan_repl/seed{N}/pareto_3x3.json
#   4. analyze_3x3.py                → ketan_repl/seed{N}/{pareto_3x3.png, pareto_3x3_zoom.png, pareto_3x3_summary.json}
#
# Usage:  ./ketan_repl/scripts/run_seed_analysis.sh <SEED> [<GPU>]
# Example: ./ketan_repl/scripts/run_seed_analysis.sh 0 0

set -euo pipefail

SEED=${1:?seed (0|1|2)}
GPU=${2:-$SEED}
REMOTE="${REMOTE:-a40_emsleeper_3gpu_1}"

ssh "$REMOTE" "bash -s" <<REMOTE_EOF
set -euo pipefail
cd /root/fra_proj
EXP=experiments/tinystories_sleeper
SEEDDIR=ketan_repl/seed${SEED}
mkdir -p \$SEEDDIR

echo "[seed${SEED}] step 1/4: build cache"
CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python \$EXP/tracing_feature/scripts/cache_layer0_activations.py \\
  --output \$SEEDDIR/layer0_cache.pt \\
  --sae_pre recreate_layer0_seed${SEED}/results/crosscoder_sae_layer0.pt \\
  --sae_mid recreate_layer0_seed${SEED}/results/crosscoder_sae_layer1.pt \\
  --sae_ln1 recreate_ln1_seed${SEED}/results/crosscoder_sae_layer0.pt 2>&1 \\
  | tail -25

echo "[seed${SEED}] step 2/4: rank features"
CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python ketan_repl/scripts/rank_features.py \\
  --cache \$SEEDDIR/layer0_cache.pt \\
  --output \$SEEDDIR/features.json \\
  --top_k 3 2>&1 \\
  | tail -15

echo "[seed${SEED}] step 3/4: pareto_3x3 sweep"
CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python \$EXP/tracing_feature/qk_vs_ov/scripts/pareto_3x3.py \\
  --cache \$SEEDDIR/layer0_cache.pt \\
  --features_json \$SEEDDIR/features.json \\
  --output_dir \$SEEDDIR \\
  --output_name pareto_3x3.json 2>&1 \\
  | tail -40

echo "[seed${SEED}] step 4/4: analyze + summary"
CUDA_VISIBLE_DEVICES=${GPU} .venv/bin/python \$EXP/tracing_feature/qk_vs_ov/scripts/analyze_3x3.py \\
  --input \$SEEDDIR/pareto_3x3.json \\
  --output_dir \$SEEDDIR 2>&1 \\
  | tail -30

echo "[seed${SEED}] DONE → ls \$SEEDDIR"
ls -la \$SEEDDIR
REMOTE_EOF
