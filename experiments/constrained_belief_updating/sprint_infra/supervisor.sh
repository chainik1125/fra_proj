#!/usr/bin/env bash
# Supervisor v2 (hot-patch 2026-07-15 ~05:45Z). Deltas vs v1:
#  - BUDGET: within $15 of cap -> write-only mandate; over cap -> ONE final
#    wrap turn, snapshot, terminate (v1 kept resuming forever post-cap,
#    burning cache-reads on no-op turns).
#  - Anti rapid-fire: a successful turn shorter than 3 min earns a 10-min
#    pause before the next resume (idle agent ~6 cheap turns/h, not ~60).
#  - Env comes from /workspace/sprint_env (injected at patch time).
# State layout identical to v1 -> wall clock, session id, turn counter and
# cost total all carry over.
set -o pipefail
[ -f /workspace/sprint_env ] && . /workspace/sprint_env

REPO=/workspace/fra_proj
SPRINT_DIR="$REPO/experiments/constrained_belief_updating/sprint"
INFRA_DIR="$REPO/experiments/constrained_belief_updating/sprint_infra"
STATE=/workspace/sprint_state
mkdir -p "$STATE" "$SPRINT_DIR"
export PATH="$HOME/.local/bin:$PATH"
export IS_SANDBOX=1
cd "$REPO"

SPRINT_HOURS="${SPRINT_HOURS:-10}"
SPRINT_MINUTES="${SPRINT_MINUTES:-$((SPRINT_HOURS * 60))}"   # override for replacement pods
BUDGET_USD="${BUDGET_USD:-120}"
MODEL="${MODEL:-claude-fable-5}"
TURN_TIMEOUT="${TURN_TIMEOUT:-3600}"
WRAP_MINS=75
BUDGET_SOFT_MARGIN=15

if [ ! -f "$STATE/start_epoch" ]; then date +%s > "$STATE/start_epoch"; fi
START=$(cat "$STATE/start_epoch")
DEADLINE=$((START + SPRINT_MINUTES * 60))

ship_file() { python3 - "$1" "$2" <<'PY' 2>/dev/null || true
import os, sys
from huggingface_hub import HfApi
HfApi(token=os.environ["HF_TOKEN"]).upload_file(
    path_or_fileobj=sys.argv[1],
    path_in_repo=os.environ["HF_PREFIX"] + "/" + sys.argv[2],
    repo_id=os.environ["HF_DATASET"], repo_type="dataset",
    commit_message="sprint: " + sys.argv[2])
PY
}

snapshot() {
    tar -czf /workspace/sprint_work.tar.gz \
        -C "$REPO/experiments" constrained_belief_updating fra_hmm_toy 2>/dev/null
    ship_file /workspace/sprint_work.tar.gz sprint_work.tar.gz
    for f in "$SPRINT_DIR/RESEARCH_LOG.md" "$SPRINT_DIR/summary.md" \
             "$SPRINT_DIR"/notes/*.tex "$SPRINT_DIR"/figures/*.png; do
        [ -f "$f" ] && ship_file "$f" "deliverables/${f#"$SPRINT_DIR/"}"
    done
    ship_file /workspace/supervisor.log supervisor.log
}
( while true; do sleep 900; snapshot; done ) & echo $! > "$STATE/snapshotter.pid"

[ -f "$STATE/cost_total" ] || echo 0 > "$STATE/cost_total"
add_cost() { python3 -c "
import sys, json
try: prev = float(open('$STATE/cost_total').read().strip() or 0)
except Exception: prev = 0.0
c = 0.0
try: c = float(json.load(open(sys.argv[1])).get('total_cost_usd') or 0)
except Exception: pass
open('$STATE/cost_total','w').write(str(prev + c))
print('%.2f' % (prev + c))
" "$1"; }

request_terminate() {
    for i in 1 2 3 4 5; do
        curl -sS --max-time 20 -X POST -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -H "Content-Type: application/json" \
            -d "{\"query\":\"mutation { podTerminate(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) }\"}" \
            https://api.runpod.io/graphql || true
        sleep 10
    done
}

TURN_FILE="$STATE/turn_n"; [ -f "$TURN_FILE" ] || echo 0 > "$TURN_FILE"
SID_FILE="$STATE/session_id"
FAST_FAILS=0
END_REASON="time"

fmt_hm() { printf '%dh%02dm' $(($1 / 3600)) $((($1 % 3600) / 60)); }
over() { python3 -c "exit(0 if float('$1') > float('$2') else 1)"; }

echo "[supervisor v2] active $(date -u +%FT%TZ) turn=$(cat $TURN_FILE) cost=\$$(cat $STATE/cost_total)" >> /workspace/supervisor.log

while true; do
    NOW=$(date +%s); ELAPSED=$((NOW - START)); REMAIN=$((DEADLINE - NOW))
    TURN=$(cat "$TURN_FILE")
    COST=$(cat "$STATE/cost_total")
    [ "$REMAIN" -le 0 ] && { END_REASON="time"; break; }
    if over "$COST" "$BUDGET_USD"; then END_REASON="budget"; break; fi

    TIMER="[SUPERVISOR TIMER] Wall-clock elapsed: $(fmt_hm $ELAPSED) of $(fmt_hm $((SPRINT_MINUTES * 60))) total. Remaining: $(fmt_hm $REMAIN). API spend so far: \$${COST} of \$${BUDGET_USD} cap."
    if [ "$REMAIN" -le $((WRAP_MINS * 60)) ] || over "$COST" "$((BUDGET_USD - BUDGET_SOFT_MARGIN))"; then
        TIMER="$TIMER
MANDATE: the sprint is in its FINAL WINDOW (time or budget nearly exhausted). STOP all experiments and background processes now. Work ONLY on summary.md and the two LaTeX notes in $SPRINT_DIR, make them complete and self-contained, then end your turn. Do not start anything new. Skip any warden check."
    fi

    OUT="$STATE/turn_${TURN}.json"
    if [ "$TURN" -eq 0 ]; then
        ADDENDUM=""
        [ -n "${KICKOFF_ADDENDUM:-}" ] && [ -f "$KICKOFF_ADDENDUM" ] && ADDENDUM="$(cat "$KICKOFF_ADDENDUM")

"
        PROMPT="${ADDENDUM}$(cat "$INFRA_DIR/kickoff.md")

$TIMER"
        timeout "$TURN_TIMEOUT" claude -p "$PROMPT" \
            --model "$MODEL" --output-format json --dangerously-skip-permissions \
            > "$OUT" 2>>/workspace/supervisor.log
        RC=$?
    else
        PROMPT="[SUPERVISOR RESUME turn $TURN]
$TIMER
${EXTRA_RESUME_NOTE:-}
Continue the sprint exactly where you left off. If you are unsure of your state, reread $SPRINT_DIR/RESEARCH_LOG.md and check for detached background processes (ps aux, and the nohup logs you started). Keep the research log current. If your deliverables are already final and nothing remains, reply DONE_IDLE and end the turn immediately."
        SID=$(cat "$SID_FILE" 2>/dev/null || true)
        if [ -n "$SID" ]; then
            timeout "$TURN_TIMEOUT" claude -p "$PROMPT" --resume "$SID" \
                --model "$MODEL" --output-format json --dangerously-skip-permissions \
                > "$OUT" 2>>/workspace/supervisor.log
            RC=$?
        else
            timeout "$TURN_TIMEOUT" claude -p "$PROMPT" --continue \
                --model "$MODEL" --output-format json --dangerously-skip-permissions \
                > "$OUT" 2>>/workspace/supervisor.log
            RC=$?
        fi
    fi

    T_END=$(date +%s); DUR=$((T_END - NOW))
    NEW_SID=$(python3 -c "
import json
try: print(json.load(open('$OUT')).get('session_id') or '')
except Exception: print('')
")
    [ -n "$NEW_SID" ] && echo "$NEW_SID" > "$SID_FILE"
    COST=$(add_cost "$OUT")
    # v3: a timeout-killed turn (rc=124) writes no result JSON, so its real
    # spend is invisible — add a conservative estimate (~$65/h observed) so
    # the budget cap cannot be blinded by long turns
    if [ "$RC" -eq 124 ]; then
        COST=$(python3 -c "
prev = float(open('$STATE/cost_total').read().strip() or 0)
est = $DUR / 3600.0 * 65.0
open('$STATE/cost_total','w').write(str(prev + est))
print('%.2f' % (prev + est))
")
    fi
    echo "[supervisor v3] turn=$TURN rc=$RC dur=${DUR}s cost_total=\$$COST sid=${NEW_SID:0:8}" >> /workspace/supervisor.log
    ship_file "$OUT" "turns/turn_${TURN}.json"
    ship_file /workspace/supervisor.log supervisor.log

    if [ "$RC" -ne 0 ] && [ "$DUR" -lt 60 ]; then
        FAST_FAILS=$((FAST_FAILS + 1))
        if [ "$FAST_FAILS" -ge 5 ]; then
            echo '{"status":"stalled","reason":"5 consecutive fast claude failures"}' > /workspace/status_stalled.json
            ship_file /workspace/status_stalled.json status_stalled.json
            snapshot
            sleep infinity
        fi
        sleep 30
    else
        FAST_FAILS=0
    fi

    echo $((TURN + 1)) > "$TURN_FILE"
    # anti rapid-fire: quick successful turns = idle agent; don't re-bill the
    # cached context every few seconds
    if [ "$RC" -eq 0 ] && [ "$DUR" -lt 180 ]; then sleep 600; else sleep 5; fi
done

SID=$(cat "$SID_FILE" 2>/dev/null || true)
FINAL="[SUPERVISOR FINAL] The sprint is OVER (reason: $END_REASON). You have 20 minutes of grace. Do exactly this and nothing else: (1) ensure $SPRINT_DIR/summary.md is complete and self-contained (if the budget ended the sprint early, write up what exists honestly); (2) ensure both LaTeX notes and RESEARCH_LOG.md are saved in $SPRINT_DIR; (3) stop all background processes; (4) end your turn."
if [ -n "$SID" ]; then
    timeout 1800 claude -p "$FINAL" --resume "$SID" \
        --model "$MODEL" --output-format json --dangerously-skip-permissions \
        > "$STATE/turn_final.json" 2>>/workspace/supervisor.log
else
    timeout 1800 claude -p "$FINAL" --continue \
        --model "$MODEL" --output-format json --dangerously-skip-permissions \
        > "$STATE/turn_final.json" 2>>/workspace/supervisor.log
fi
add_cost "$STATE/turn_final.json" >/dev/null
ship_file "$STATE/turn_final.json" turns/turn_final.json

kill "$(cat "$STATE/snapshotter.pid" 2>/dev/null)" 2>/dev/null || true
snapshot
echo "{\"status\":\"complete\",\"reason\":\"$END_REASON\",\"cost\":\"$(cat $STATE/cost_total)\"}" > /workspace/status_complete.json
ship_file /workspace/status_complete.json status_complete.json
echo "[supervisor v2] sprint complete ($END_REASON) $(date -u +%FT%TZ)" >> /workspace/supervisor.log
ship_file /workspace/supervisor.log supervisor.log
request_terminate
sleep infinity
