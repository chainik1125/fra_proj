# Autonomous autoresearch protocol (cadenza_sae_steer)

A cron fires this loop. It is a **self-driving researcher**: it decides by the metrics,
handles its own errors, and progresses through phases. **It NEVER pauses to ask Jamie.**
It stops only when the goal is met or the plan is exhausted, and then writes a summary.
Branch `jamie/llama-sleeper-repro` only; `runpod` H100 only. Goals/rules: CAMPAIGN.md.

State: `queue.json` = `{phase, goal, running, pending[], failed[], on_phase_empty}`.
Cells are typed: `kind:"train"` (SAE training) or `kind:"method"` (apply suppression).
`RESULTS.md` is the dedup log (a cell name already in a results table = already logged).
Per-cell results land in pod `/workspace/jamie/orch/<name>.metrics.json` (`status` ok|error).

## Each tick — in order, stop at the first that applies

1. **Still running?** `ssh runpod 'pgrep -af cell_runner.py | grep -v pgrep || echo IDLE'`
   If a cell_runner proc is alive → reply ONE line `tick: <name> running` and **STOP**.
   No analysis, no commits, no launches. (Common case — cells take ~40–60 min.)

2. **Log finished cell(s).** `ssh runpod 'cat /workspace/jamie/orch/*.metrics.json 2>/dev/null'`.
   For each whose `name` is NOT yet a row in RESULTS.md:
   - `kind:"train"` → append to the Phase-1 table (dead/EV/L0 per hook, minutes, status).
   - `kind:"method"` → append to the Phase-2 table (method, mix, layer, best alpha, ASR, JSDc, minutes, status).
   - **Error handling (autonomous, no halting):** if `status=="error"`, find the cell in
     `queue.json`. If its `attempts < 1`, re-insert it at the FRONT of `pending` with
     `attempts+1` (transient-retry). If `attempts >= 1`, move it to `failed[]` and continue.
     Either way log the error briefly in RESULTS.md. Do NOT stop the loop for one bad cell.
   - Commit + push RESULTS.md (+ queue.json if changed).

3. **Launch the next pending cell.** Read `queue.json`; if `pending` non-empty, take `pending[0]`:
   ```
   ssh runpod 'cd /workspace/jamie/fra_proj_llama && git pull -q && \
     rm -f /workspace/jamie/orch/<name>.metrics.json /workspace/jamie/orch/<name>.done && \
     nohup /root/sleepers-venv/bin/python scripts/cell_runner.py --name <name> \
       <ARGS> > /workspace/jamie/orch/<name>.log 2>&1 &'
   ```
   `<ARGS>` by kind:
   - train:  `--kind train --layer <L> --deployed-frac <f>`
   - method: `--kind method --layer <L> --mix <mix> --method <ov|conv|dom>` (+ `--alphas a b c` / `--top-k k` if the cell sets them)
   Update `queue.json` (remove from `pending`, set `running=<name>`), commit + push.
   Reply `launched <name>`. **STOP.**

4. **Phase complete** (`pending` empty AND idle). Execute `queue.json.on_phase_empty`
   EXACTLY — it tells you how to rank results, pick the winner, and enqueue the next
   phase's cells (write them into `pending`, update `phase`). This is a DECISION you make
   from the metrics — make it, write it to RESULTS.md, enqueue, commit. Then launch the
   first new cell (go to step 3). **Do not ask Jamie.**
   - If `on_phase_empty` says the goal is met or the plan is exhausted → write a final
     summary to RESULTS.md (best config: method/mix/layer/alpha/ASR/JSDc), commit, and STOP.

## Rules
- Decide by metrics, never by asking. The ONLY stop conditions are goal-met or plan-exhausted.
- A cell error is retried once then skipped — it never halts the campaign.
- Don't shrink validated hyperparameters; chunk instead.
- Commit only to jamie/llama-sleeper-repro; runpod H100 only; never print secrets.
- Keep ticks terse — most are step 1.
