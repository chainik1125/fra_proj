#!/bin/bash
# =============================================================================
# 14B EM Steering Campaign — extreme-sports + financial-advice
#
# Based on Dmitry's 7B methodology (summary.md @ e4e8338):
#   - EM-specificity: run on BOTH EM model AND base model
#   - Multi-seed: seeds {42, 123, 456}
#   - Multi-prompt ranking (all 8 prompts)
#   - GPT-4o judging (requires OPENAI_API_KEY)
#   - Frontier sweep: α ∈ {0.0, 0.5, 1.0, 1.5, 2.0, 3.0}
#
# Experiment matrix per EM variant (sports, finance):
#   1. gamma_check       — verify SAE γ handling is correct
#   2. frontier_multiseed — alignment-vs-coherence frontier (EM model)
#   3. frontier_base      — same frontier on BASE model (EM-specificity)
#   4. shared_feature     — cross-head steering (GQA-fixed)
#   5. ce_vs_base         — KL divergence to base (deterministic)
#
# Usage:
#   bash run_14b_campaign.sh setup          # first-time pod setup
#   bash run_14b_campaign.sh gamma          # check γ before running
#   bash run_14b_campaign.sh sports         # run all sports experiments
#   bash run_14b_campaign.sh finance        # run all finance experiments
#   bash run_14b_campaign.sh all            # run everything
# =============================================================================
set -e

RESULTS=/root/results_v3
LOGDIR=/root/logs_v3
REPO=/workspace/fra_proj

mkdir -p "$RESULTS" "$LOGDIR"

# ── Setup ────────────────────────────────────────────────────────────────
setup() {
    echo "=== Pod Setup ==="

    # PyTorch for CUDA 12.4
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

    # transformer_lens + compatible transformers
    pip install 'transformer-lens>=3.0'

    # Other deps
    pip install sae_lens huggingface_hub safetensors einops tqdm scikit-learn peft openai

    # HF cache on container disk
    export HF_HOME=/workspace/hf_cache
    mkdir -p /workspace/hf_cache

    # Verify
    python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available!'; print(f'GPU: {torch.cuda.get_device_name(0)}')"
    python -c "from transformer_lens import HookedTransformer; print('transformer_lens OK')"

    echo ""
    echo "=== Setup complete ==="
    echo "Set your keys:"
    echo "  export HF_TOKEN=your_token"
    echo "  export OPENAI_API_KEY=sk-your_key"
    echo "  export HF_HOME=/workspace/hf_cache"
    echo "  huggingface-cli login --token \$HF_TOKEN"
}

# ── Gamma check ──────────────────────────────────────────────────────────
gamma_check() {
    echo "=== Checking γ (gain) for ln1 at layer 24 ==="
    python -c "
import torch, json, sys
sys.path.insert(0, '$REPO')

# Load base model (lighter than EM)
from transformer_lens import HookedTransformer
print('Loading Qwen2.5-14B-Instruct (base) for gamma check...')
model = HookedTransformer.from_pretrained_no_processing(
    'Qwen/Qwen2.5-14B-Instruct', device='cuda', dtype=torch.bfloat16)

gamma = model.blocks[24].ln1.w.detach().float()
print(f'gamma shape: {gamma.shape}')
print(f'||gamma||  = {gamma.norm():.4f}')
print(f'mean       = {gamma.mean():.6f}')
print(f'std        = {gamma.std():.6f}')
print(f'min        = {gamma.min():.6f}')
print(f'max        = {gamma.max():.6f}')
print(f'median     = {gamma.median():.6f}')

# Check how far from 1
deviation = (gamma - 1.0).abs()
print(f'mean |γ-1| = {deviation.mean():.6f}')
print(f'max  |γ-1| = {deviation.max():.6f}')
print(f'% within 10% of 1: {(deviation < 0.1).float().mean()*100:.1f}%')

# Also check what the SAE sees vs what hook_normalized gives
# hook_normalized = x / rms  (pre-gain)
# post-gain       = x / rms * gamma
# If SAE trained on post-gain, we need gamma correction
print()
if gamma.std() > 0.05:
    print('WARNING: gamma varies significantly (std > 0.05)')
    print('  -> If SAE was trained on post-gain activations, gamma correction IS needed')
    print('  -> Need to check with Nura what the SAE training input was')
else:
    print('OK: gamma is approximately constant (std <= 0.05)')
    print('  -> Gamma correction makes minimal difference')

# Save for reference
info = {
    'shape': list(gamma.shape),
    'norm': gamma.norm().item(),
    'mean': gamma.mean().item(),
    'std': gamma.std().item(),
    'min': gamma.min().item(),
    'max': gamma.max().item(),
}
with open('$RESULTS/gamma_check_L24.json', 'w') as f:
    json.dump(info, f, indent=2)
print(f'Saved to $RESULTS/gamma_check_L24.json')

# Also check SAE reconstruction quality
print()
print('=== SAE reconstruction quality check ===')
from fra.sae_lens_wrapper import QwenLn1SAE
sae = QwenLn1SAE('Nura-J/Qwen2.5-14B_SAE_ln1.normalised', layer=24, device='cuda')

# Get some activations from the ln1.hook_normalized hook
hook_name = 'blocks.24.ln1.hook_normalized'
test_text = 'Hello! I can fulfill your one wish. What is the one thing you want?'
toks = model.to_tokens(test_text)
_, cache = model.run_with_cache(toks, names_filter=hook_name)
acts = cache[hook_name][0].float()  # [seq, d_model] — pre-gain

# Test 1: encode/decode on pre-gain (what nura/dev does)
feats_pregain = sae.encode(acts)
recon_pregain = sae.decode(feats_pregain)
var_expl_pregain = 1 - (acts - recon_pregain).pow(2).sum() / acts.pow(2).sum()
l0_pregain = (feats_pregain > 0).float().sum(-1).mean()

# Test 2: encode/decode on post-gain (what Dmitry does for 7B)
acts_postgain = acts * gamma.to(acts.device)
feats_postgain = sae.encode(acts_postgain)
recon_postgain = sae.decode(feats_postgain)
var_expl_postgain = 1 - (acts_postgain - recon_postgain).pow(2).sum() / acts_postgain.pow(2).sum()
l0_postgain = (feats_postgain > 0).float().sum(-1).mean()

print(f'Pre-gain  (no γ):  var_expl={var_expl_pregain:.4f}  L0={l0_pregain:.1f}')
print(f'Post-gain (with γ): var_expl={var_expl_postgain:.4f}  L0={l0_postgain:.1f}')
print()
if var_expl_pregain > var_expl_postgain + 0.05:
    print('RESULT: SAE reconstructs PRE-GAIN better -> trained on pre-gain -> no γ correction needed')
elif var_expl_postgain > var_expl_pregain + 0.05:
    print('RESULT: SAE reconstructs POST-GAIN better -> trained on post-gain -> NEED γ correction!')
    print('  ACTION: Add gamma correction before running experiments')
else:
    print('RESULT: Similar reconstruction quality -> γ effect is small')
    print('  -> Safe to proceed without γ correction')

info['var_expl_pregain'] = var_expl_pregain.item()
info['var_expl_postgain'] = var_expl_postgain.item()
info['l0_pregain'] = l0_pregain.item()
info['l0_postgain'] = l0_postgain.item()
with open('$RESULTS/gamma_check_L24.json', 'w') as f:
    json.dump(info, f, indent=2)
" 2>&1 | tee "$LOGDIR/gamma_check.txt"
}

# ── Run one EM variant ───────────────────────────────────────────────────
run_variant() {
    local VARIANT=$1  # "sports" or "finance"
    echo ""
    echo "################################################################"
    echo "# VARIANT: $VARIANT"
    echo "################################################################"

    cd "$REPO"

    # ── Experiment 1: Frontier multiseed on EM model ─────────────────
    echo ""
    echo "=== [$VARIANT] Exp 1/4: Frontier multiseed (EM model) ==="
    python run_experiments.py \
        --task frontier_multiseed \
        --em-model "$VARIANT" \
        --head 38 \
        --k 50 \
        --n-texts 8 \
        --seeds 42 123 456 \
        --temperature 1.0 \
        --output "$RESULTS" \
        2>&1 | tee "$LOGDIR/${VARIANT}_frontier_em.txt"

    # ── Experiment 2: Frontier multiseed on BASE model (EM-specificity) ─
    echo ""
    echo "=== [$VARIANT] Exp 2/4: Frontier multiseed (BASE model — EM-specificity check) ==="
    python run_experiments.py \
        --task frontier_multiseed \
        --em-model base \
        --head 38 \
        --k 50 \
        --n-texts 8 \
        --seeds 42 123 456 \
        --temperature 1.0 \
        --output "$RESULTS" \
        2>&1 | tee "$LOGDIR/${VARIANT}_frontier_base.txt"

    # ── Experiment 3: Shared feature multiseed (GQA-fixed) ───────────
    echo ""
    echo "=== [$VARIANT] Exp 3/4: Shared feature multiseed (GQA-fixed) ==="
    python run_experiments.py \
        --task shared_feature_multiseed \
        --em-model "$VARIANT" \
        --n-texts 8 \
        --seeds 42 123 456 \
        --temperature 1.0 \
        --output "$RESULTS" \
        2>&1 | tee "$LOGDIR/${VARIANT}_shared_feature.txt"

    # ── Experiment 4: CE vs base (deterministic, no generation) ──────
    echo ""
    echo "=== [$VARIANT] Exp 4/4: CE vs base ==="
    python run_experiments.py \
        --task ce_vs_base \
        --em-model "$VARIANT" \
        --head 38 \
        --k 50 \
        --n-texts 8 \
        --output "$RESULTS/ce_vs_base_${VARIANT}_L24_H38.json" \
        2>&1 | tee "$LOGDIR/${VARIANT}_ce_vs_base.txt"

    echo ""
    echo "=== [$VARIANT] ALL EXPERIMENTS COMPLETE ==="
    echo "Results in: $RESULTS"
    echo "Logs in:    $LOGDIR"
}

# ── GPT-4o judge (if not done inline) ────────────────────────────────────
judge_results() {
    local VARIANT=$1
    echo "=== GPT-4o judging for $VARIANT ==="

    if [ -z "$OPENAI_API_KEY" ]; then
        echo "ERROR: OPENAI_API_KEY not set. Skipping judging."
        return 1
    fi

    cd "$REPO"

    # Judge frontier results
    for f in "$RESULTS"/multiseed_${VARIANT}_*_full.json; do
        [ -f "$f" ] || continue
        base=$(basename "$f" _full.json)
        echo "Judging $f ..."
        python judge_multiseed.py "$f" \
            --output "$RESULTS/gpt4o_${base}.json" \
            2>&1 | tee -a "$LOGDIR/${VARIANT}_judge.txt"
    done
}

# ── Package results for download ─────────────────────────────────────────
package() {
    echo "=== Packaging results ==="
    tar czf /root/campaign_v3_results.tar.gz \
        "$RESULTS"/ "$LOGDIR"/ \
        2>&1
    echo "Download: /root/campaign_v3_results.tar.gz"
    du -h /root/campaign_v3_results.tar.gz
}

# ── Main dispatch ────────────────────────────────────────────────────────
case "${1:-help}" in
    setup)
        setup
        ;;
    gamma)
        gamma_check
        ;;
    sports)
        run_variant sports
        judge_results sports
        ;;
    finance)
        run_variant finance
        judge_results finance
        ;;
    all)
        gamma_check
        run_variant sports
        judge_results sports
        run_variant finance
        judge_results finance
        package
        ;;
    package)
        package
        ;;
    help|*)
        echo "Usage: bash run_14b_campaign.sh {setup|gamma|sports|finance|all|package}"
        echo ""
        echo "  setup    — install deps, verify GPU"
        echo "  gamma    — check γ handling (RUN FIRST)"
        echo "  sports   — run all extreme-sports experiments"
        echo "  finance  — run all financial-advice experiments"
        echo "  all      — gamma + sports + finance + package"
        echo "  package  — tar results for download"
        ;;
esac
