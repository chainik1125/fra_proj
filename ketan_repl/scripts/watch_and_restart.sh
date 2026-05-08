#!/usr/bin/env bash
# Watcher tick: ssh to the pod, check the status of every (seed, kind) training
# job, and restart any that have died but not finished. Smart-skips harvest/train
# stages whose outputs are already on disk.
#
# Outputs a one-line-per-job summary to stdout (consumed by /loop / Monitor) and
# appends a timestamped block to ketan_repl/notes/STATUS.md locally.
#
# Usage:  ./ketan_repl/scripts/watch_and_restart.sh

set -uo pipefail

REMOTE="${REMOTE:-a40_emsleeper_3gpu_1}"
EXP="experiments/tinystories_sleeper"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOCAL_NOTES="$(cd "$HERE/.." && pwd)/notes"
mkdir -p "$LOCAL_NOTES"
STATUS_FILE="$LOCAL_NOTES/STATUS.md"

NOW="$(date -u +%Y-%m-%dT%H:%M:%SZ)"

# Single ssh call collects everything we need for all 6 jobs.
SUMMARY=$(ssh -o ConnectTimeout=10 "$REMOTE" "bash -s" <<'REMOTE_EOF'
set -uo pipefail
EXP=experiments/tinystories_sleeper
cd /root/fra_proj || exit 99

# Per-GPU GPU util
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader \
  | awk '{ print "GPU " $0 }'

for seed in 0 1 2; do
  for kind in layer0 ln1; do
    dir=${EXP}/recreate_${kind}_seed${seed}
    res=${dir}/results
    manifest=${res}/MANIFEST.md
    log=${res}/run.log
    proc=$(pgrep -f "${dir}/reproduce.py" | head -1 || true)
    last_log=$(tail -1 "$log" 2>/dev/null | head -c 110)

    if [ -f "$manifest" ]; then
      state=DONE
    elif [ -n "$proc" ]; then
      state=ALIVE
    else
      # Dead and incomplete. Decide which stages to skip on restart.
      have_acts=0; have_ckpts=0; have_sweep=0
      [ -f "${res}/activations_cache.pt" ] && have_acts=1
      ls "${res}"/crosscoder_*.pt >/dev/null 2>&1 && have_ckpts=1
      ls "${res}"/val_sweep_*.json >/dev/null 2>&1 && have_sweep=1
      skip=()
      [ "$have_acts" = 1 ] && skip+=(harvest)
      [ "$have_ckpts" = 1 ] && skip+=(train)
      [ "$have_sweep" = 1 ] && skip+=(sweep)
      state=RESTART_$(IFS=,; echo "${skip[*]:-none}")
      # Restart in background, pinned to GPU $seed.
      mkdir -p "$res"
      echo "[restart $(date -u +%H:%M:%SZ)] seed=$seed kind=$kind skip=(${skip[*]:-})" >> "$log"
      nohup env CUDA_VISIBLE_DEVICES=$seed .venv/bin/python "${dir}/reproduce.py" \
        ${skip[@]:+--skip ${skip[@]}} \
        >> "$log" 2>&1 &
      disown $! 2>/dev/null || true
    fi

    printf 'JOB seed=%s kind=%-7s state=%-25s last="%s"\n' "$seed" "$kind" "$state" "$last_log"
  done
done
REMOTE_EOF
)

# Append to STATUS.md and echo to stdout.
{
  echo
  echo "## tick @ $NOW"
  echo
  echo '```text'
  echo "$SUMMARY"
  echo '```'
} >> "$STATUS_FILE"

echo "$SUMMARY"
