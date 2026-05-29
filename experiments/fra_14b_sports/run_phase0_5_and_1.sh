#!/usr/bin/env bash
# Phase 0.5 (head ablation + ‖Δa‖) + Phase 1 (noise check) on one pod.
#
# Requirements:
#   - 80GB GPU (A100/H100)
#   - OPENAI_API_KEY (for gpt-4o-mini judging)
#
# Usage on RunPod:
#   export OPENAI_API_KEY=sk-...
#   bash experiments/fra_14b_sports/run_phase0_5_and_1.sh
#
# Or with setup (fresh pod):
#   bash experiments/fra_14b_sports/run_phase0_5_and_1.sh setup
set -eo pipefail

REPO_DIR="${REPO_DIR:-$(cd "$(dirname "$0")/../.." && pwd)}"
RESULTS="${RESULTS:-/workspace/results_sports}"
mkdir -p "$RESULTS"

# ── Dependency check ─────────────────────────────────────────────────────
check_deps() {
    echo "=== Checking dependencies ==="
    local missing=()
    python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null \
        || missing+=("torch+cuda")
    python3 -c "import transformer_lens" 2>/dev/null \
        || missing+=("transformer_lens")
    python3 -c "import transformers" 2>/dev/null \
        || missing+=("transformers")
    python3 -c "import peft" 2>/dev/null \
        || missing+=("peft")
    python3 -c "import huggingface_hub" 2>/dev/null \
        || missing+=("huggingface_hub")
    python3 -c "import safetensors" 2>/dev/null \
        || missing+=("safetensors")
    python3 -c "import openai" 2>/dev/null \
        || missing+=("openai")

    if [ ${#missing[@]} -eq 0 ]; then
        echo "  All dependencies found."
        python3 -c "import torch; print(f'  torch {torch.__version__}  GPU: {torch.cuda.get_device_name(0)}')"
        return 0
    else
        echo "  MISSING: ${missing[*]}"
        echo "  Run: bash $0 setup"
        return 1
    fi
}

# ── Setup (only if deps are missing) ─────────────────────────────────────
setup() {
    echo "=== Pod Setup ==="
    pip install --no-input --break-system-packages \
        'transformer_lens>=3.0,<4.0' transformers peft \
        huggingface_hub safetensors openai tqdm 2>&1 | tail -3
    pip install --no-input --break-system-packages --force-reinstall --no-deps \
        torch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 \
        --index-url https://download.pytorch.org/whl/cu124 2>&1 | tail -2
    python3 -c "import torch; assert torch.cuda.is_available(); print('GPU:', torch.cuda.get_device_name(0))"
    echo "=== Setup done ==="
}

# ── Phase 0.5: Head ablation + ‖Δa‖ ─────────────────────────────────────
run_head_delta_a() {
    echo ""
    echo "============================================================"
    echo "Phase 0.5: Head ablation + ‖Δa‖ (extreme-sports, L24)"
    echo "============================================================"
    cd "$REPO_DIR"
    python3 -u experiments/fra_14b_sports/head_delta_a.py \
        2>&1 | tee "$RESULTS/head_delta_a.log"

    # Move output to results dir
    mv -f head_ablation_delta_a_sports_l24.json "$RESULTS/" 2>/dev/null || true

    # Print summary
    echo ""
    echo "--- Head ablation results ---"
    python3 -c "
import json
r = json.load(open('$RESULTS/head_ablation_delta_a_sports_l24.json'))
print(f'  Top head:              H{r[\"top_head_argmax\"]}  (Δloss={r[\"top_head_delta\"]:.4f})')
print(f'  Top-5 heads:           {r[\"top5_heads\"]}')
print(f'  ‖Δa‖ resid_post L24:  {r[\"delta_a_norm_resid_post\"]:.3f}')
print(f'  ‖Δa‖ ln1 postgain L24: {r[\"delta_a_norm_ln1_postgain\"]:.3f}')
"
}

# ── Phase 1: Noise check ────────────────────────────────────────────────
run_noise_check() {
    echo ""
    echo "============================================================"
    echo "Phase 1: Noise check (no steering, alpha=0)"
    echo "============================================================"

    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set. Skipping noise check."
        echo "  Set it and re-run: export OPENAI_API_KEY=sk-..."
        return 1
    fi

    cd "$REPO_DIR"
    python3 -u experiments/fra_14b_sports/run_noise_check.py \
        --output "$RESULTS/noise_check_sports.json" \
        2>&1 | tee "$RESULTS/noise_check.log"
}

# ── Main ─────────────────────────────────────────────────────────────────
case "${1:-run}" in
    setup)
        setup
        ;;
    head)
        run_head_delta_a
        ;;
    noise)
        run_noise_check
        ;;
    run|"")
        check_deps || { echo "Aborting. Install deps first: bash $0 setup"; exit 1; }
        run_head_delta_a
        run_noise_check
        echo ""
        echo "============================================================"
        echo "ALL DONE — results in $RESULTS/"
        echo "============================================================"
        ls -la "$RESULTS/"
        ;;
    *)
        echo "Usage: bash $0 {setup|head|noise|run}"
        echo "  setup  — install deps (fresh pod)"
        echo "  head   — Phase 0.5 only (head ablation + ‖Δa‖)"
        echo "  noise  — Phase 1 only (noise check)"
        echo "  run    — both (default)"
        ;;
esac
