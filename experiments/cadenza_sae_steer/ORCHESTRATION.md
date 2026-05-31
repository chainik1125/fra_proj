# Autoresearch orchestration protocol (cadenza_sae_steer)

A cron fires this loop on a schedule. **Be cheap: if a cell is still training, do almost
nothing.** Only act when a cell has FINISHED. Branch `jamie/llama-sleeper-repro` only;
`runpod` H100 only. Full goals/rules in CAMPAIGN.md; results in RESULTS.md.

State: `queue.json` (`pending` cells + `running`) + `RESULTS.md` (logged cells, the dedup
key) + pod `/workspace/jamie/orch/<name>.metrics.json` (raw result, status ok|error).

## Each tick — in order, stop at the first that applies

1. **Is a cell still running?**  `ssh runpod 'pgrep -af cell_runner.py | grep -v pgrep || true'`
   - If a `cell_runner.py` process is alive → reply ONE line `tick: <name> training (~Xm)`
     and **STOP**. No analysis, no commits, no other ssh, no launch. (The common case —
     cells take ~50–60 min each.)

2. **No proc → log any just-finished cell(s).**  `ssh runpod 'cat /workspace/jamie/orch/*.metrics.json 2>/dev/null'`
   For each cell whose `name` is NOT already a row in RESULTS.md Phase-1 table:
   - append a row: mix/deployed_frac, layer, hooks, per-cell `dead_features`/EV/L0, minutes, status;
   - if `status==error` → record the error in RESULTS.md and **STOP** (do not launch more —
     a broken setup must not burn GPU; wait for Jamie).
   - commit + push (RESULTS.md).

3. **Launch the next pending cell.**  Read `queue.json`.
   - If `pending` non-empty: take `pending[0]`, then
     `ssh runpod 'cd /workspace/jamie/fra_proj_llama && git pull -q'`, then launch background:
     ```
     ssh runpod 'cd /workspace/jamie/fra_proj_llama && rm -f /workspace/jamie/orch/<name>.metrics.json && \
       nohup /root/sleepers-venv/bin/python scripts/cell_runner.py \
       --name <name> --deployed-frac <f> --layer <L> \
       > /workspace/jamie/orch/<name>.log 2>&1 &'
     ```
   - Update `queue.json` (remove from `pending`, set `running=<name>`), commit + push.
   - Reply `launched <name>`. **STOP.**

4. **Phase complete** (`pending` empty AND nothing running).
   Read the Phase-1 rows. Per CAMPAIGN.md "Data mix": choose the mix that is healthy
   (dead <~10%, EV >0.9) AND most likely to carry sleeper features (favor more deployment
   when health is comparable). Then EITHER enqueue the next cells (expand the winning mix to
   L3/9/10 × seeds, or begin Phase 2 method application) by writing `queue.json`, OR if a
   human decision is warranted, **STOP and summarize for Jamie**. Commit.

## Notes
- One cell = build mix (~2–3 min) + train ln1+resid_mid TopK at one layer (~50M tokens).
- Don't shrink hyperparameters to save time; chunk instead (≤2 hooks/run already).
- If unsure whether a result is good enough to proceed, STOP and ask rather than spend GPU.
