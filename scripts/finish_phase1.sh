#!/usr/bin/env bash
# Run from local: consolidate Phase 1 results onto pod 1 and run the orchestrator.
#
# Idempotent — safe to re-run if any step partially completes.
set -euo pipefail

POD1="${POD1:-h100_emfra_2gpu_1}"
POD2="${POD2:-h100_emfra_2gpu_2}"

echo "=== 1. Sync pod 2 runs (finance, sports) → pod 1 ==="
ssh "$POD1" 'mkdir -p /workspace/runs && which rsync || apt-get install -y rsync >/dev/null'
# rsync via ssh-agent forwarding (-A) so pod 1 can pull from pod 2 directly.
# Simpler: scp -3 via local hop.
for d in finance sports; do
    echo "  syncing $d ..."
    scp -3 -r "$POD2:/workspace/runs/$d" "$POD1:/workspace/runs/" 2>/dev/null || \
        echo "    [skip] $d not found on pod 2 (job may still be running or never produced output)"
done

echo "=== 2. Run orchestrator on pod 1 ==="
# Pass OPENAI_API_KEY_MATS (default OPENAI_API_KEY has no quota for this account)
OAI_KEY="${OPENAI_API_KEY_MATS:-${OPENAI_API_KEY:-}}"
if [[ -z "$OAI_KEY" ]]; then
    echo "[FATAL] no OpenAI key in local env (OPENAI_API_KEY_MATS preferred)"
    exit 4
fi
ssh "$POD1" "cd /workspace/fra_proj && OPENAI_API_KEY='$OAI_KEY' bash scripts/post_phase1_orchestrate.sh"

echo "=== 3. Pull frontier_grid + summary back to local for the docs ==="
TEMP_XC="/Users/dmitrymanning-coe/Documents/Research/Temporal Crosscoders/temp_xc"
LOCAL_OUT="$TEMP_XC/plots/2026-05-07_em_repl"
mkdir -p "$LOCAL_OUT"
scp "$POD1:/workspace/runs/phase1_summary/frontier_grid.png" "$LOCAL_OUT/" 2>/dev/null || true
scp "$POD1:/workspace/runs/phase1_summary/frontier_grid.pdf" "$LOCAL_OUT/" 2>/dev/null || true
scp "$POD1:/workspace/runs/phase1_summary/frontier_grid.json" "$LOCAL_OUT/" 2>/dev/null || true
scp "$POD1:/workspace/runs/phase1_summary/phase1_gate.json" "$LOCAL_OUT/" 2>/dev/null || true

echo "=== 4. Fill phase1_reproduce.md results section ==="
NURA_BASE="/Users/dmitrymanning-coe/Documents/Research/FRA/fra_proj/nura_v1_baseline.json"
DOC="$TEMP_XC/docs/dmitry/c6_em/2026-05-07_em_repl/phase1_reproduce.md"
if [[ -f "$LOCAL_OUT/frontier_grid.json" && -f "$NURA_BASE" ]]; then
    python3 "$TEMP_XC/scripts/fill_phase1_results.py" \
        --frontier-grid-json "$LOCAL_OUT/frontier_grid.json" \
        --nura-baseline-json "$NURA_BASE" \
        --doc "$DOC"
else
    echo "[skip] frontier_grid.json or nura_v1_baseline.json missing — phase1_reproduce.md not updated"
fi

echo "=== 5. Commit + push summary doc to GitHub ==="
bash "$TEMP_XC/scripts/auto_push_em_repl_summary.sh" \
    "phase 1 reproduction results filled in" \
    "docs/dmitry/c6_em/2026-05-07_em_repl/phase1_reproduce.md" || true

echo "=== Done. ==="
ls -la "$LOCAL_OUT"
