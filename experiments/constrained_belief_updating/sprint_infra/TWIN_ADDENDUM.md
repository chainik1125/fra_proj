# TWIN-B ADDENDUM (read first — modifies the kickoff below)

You are **worker B** of a two-worker parallel experiment. Worker A runs the
identical kickoff on another pod, shipping to prefix `run1/` of the private
dataset; you ship to `run1_twin/`. You are BOTH full, independent research
workers on the same task; the user reviews both outputs at the end of the 10h
as parallel independent attempts. Do NOT coordinate research content with
worker A and do NOT read its `run1/deliverables/` — independence is the point.
(The single exception is the operational warden duty below.)

## Added responsibility: WARDEN of worker A

Roughly once per hour (each supervisor resume reminds you), spend ≤10 minutes:

1. Check worker A's health on the private HF dataset
   `dmanningcoe/sprint-fra-theory`, prefix `run1/` (metadata only, not content):
   - `run1/supervisor.log` fresh (updated within ~25 min)?
   - any `run1/status_aborted.json` / `run1/status_stalled.json`?
   - `run1/turns/` advancing over the hours?
2. If healthy: one line in your RESEARCH_LOG, back to research.
3. If broken: diagnose from `run1/supervisor.log` + turn JSONs, then repair.
   You have `RUNPOD_API_KEY` in your env; worker A's pod is named
   `rs-sprint-fra-theory` (query `myself { pods }` via the RunPod GraphQL API
   for the live id). A parked/aborted pod: terminate it, then relaunch with
   `RP_API_KEY_MATS=$RUNPOD_API_KEY ANTHROPIC_API_KEY_MATS=$ANTHROPIC_API_KEY HF_TOKEN=$HF_TOKEN bash experiments/constrained_belief_updating/sprint_infra/launch_sprint.sh`.
   NEVER create a duplicate — terminate any RUNNING `rs-sprint-fra-theory`
   first. A relaunched pod restarts its own 10h clock; if >5h of the night are
   already gone, prefer NOT relaunching and note it instead.
4. Warden duty must never consume more than ~15% of your time. Your primary
   job is the research.
