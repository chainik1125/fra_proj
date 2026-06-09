---
name: research_swarm
description: Run a large research campaign autonomously with a team of background agents — an orchestrator (builds the parametrized measurement code + owns the repo + launches GPU tranches), a GPU-supervisor (polls/relaunches pods), and a results-analyst (judges + computes metrics + assembles the results table). Sits ON TOP of /dispatch_campaign (which provides the RunPod fan-out + gotchas). Use when a campaign is too big for one agent — it needs to write code, fan out many GPU cells (a grid/cross-product), keep the GPUs healthy, AND interpret results as they land — and the user wants it run hands-off by a "team". For a single sweep with no code-build/interpretation, use /dispatch_campaign directly instead.
---

# /research_swarm — autonomous multi-agent research campaign

A team of background agents executes a whole campaign while you sleep: build →
smoke-gate → tranched GPU fan-out → supervise → judge → assemble. The GPU layer
is `/dispatch_campaign`; this skill adds the **agent team + a single-source-of-
truth spec + a validation gate** on top.

```
          you (human-facing lead)
            │ writes CAMPAIGN.md (spec), TeamCreate, spawns the 3 agents
            ▼
   ┌──── campaign-lead (bg) ───────────────┐   sole repo writer; builds code,
   │  build → smoke-gate → launch tranches │   owns the branch, launches pods,
   └───────┬──────────────────┬────────────┘   commits results
           │ SendMessage      │ SendMessage
   gpu-supervisor (bg)   results-analyst (bg)
   RunPod API/SSH only   judges + metric + GRID_RESULTS.md text → lead commits
           │                  │
           ▼                  ▼
        grid-* pods  ───push──▶  HF dataset (sync point)
```

## When NOT to use
- One sweep, no code to write, no per-cell interpretation → `/dispatch_campaign`.
- Re-run one measurement under a varied param with run-tracking → `/transfer`.
- You want to stay in the loop on every decision → just dispatch + monitor yourself.

---

## THE #1 GOTCHA (cost me a round-trip): sub-agents can't spawn sub-agents
A `general-purpose` teammate does **not** have the Agent tool — so the
campaign-lead **cannot create gpu-supervisor / results-analyst itself**. **You
(the human-facing orchestrator) must spawn all three agents.** Spawn the lead
first, let it build + report its plan, then spawn the supervisor + analyst into
the team yourself (the lead will explicitly ask). Don't expect the lead to do it.

## Other hard rules (learned, not optional)
1. **Single source of truth = `CAMPAIGN.md`.** Fresh-context agents drift; write
   the whole spec to a committed file (grid/axes, steering/measurement math,
   metric definition, n/seeds, models, HF output layout, the reuse files, and
   every dispatch gotcha) and have *every* agent read it first.
2. **One repo writer.** Designate the lead as the ONLY agent that commits.
   Agents share the working dir — concurrent edits/pushes corrupt it. Supervisor
   uses RunPod API/SSH only (no repo edits); analyst works in /tmp + HF and hands
   finished doc *text* to the lead to commit.
3. **Smoke-gate before fan-out.** Have the lead validate ONE canary cell
   end-to-end (build → sweep → HF → judge → metric) before launching the rest.
   This catches systematic bugs (output-schema regex, normalization/γ handling,
   ranking) on one cheap pod instead of wasting the whole grid. It earns its
   ~30 min every time — in the run that produced this skill it caught a judge
   filename-regex bug and confirmed the SAE var-explained on-pod.
4. **Approval + cost gate — for GPU *AND* judging/API.** The lead reports its
   build+launch plan AND a pod-count/$ projection BEFORE spawning pods; you
   approve. Keep n modest. **The LLM-judging step is a first-class cost center,
   often larger than the GPU spend** — estimate it the same way: `calls =
   generations × calls-per-generation` (a per-feature `gran1` cell is ~14k–27k
   generations/stream, ×2 calls for align+coherence, × streams = millions of
   calls fast), project the $ at the chosen model, and surface it. **Default the
   judge to gpt-4o-mini (~20× cheaper) or the Batch API (50% off), not gpt-4o**,
   unless you've approved the full-price bill. (Learned the hard way — see
   `feedback_judging_api_cost_guard`: an unestimated gpt-4o judge loop burned a
   large overnight bill.)
4b. **Judging must be idempotent across restarts + spend-guarded.** Key the
   "already judged, skip" check off the **remote (HF) combined/judged output, not
   local files**, and **persist the loop's done-state to disk** — otherwise every
   restart re-judges from raw and re-spends API (this is what blew the bill: ~5
   restarts during an infra incident each re-judged cleared cells). Add a running
   **call/cost counter that halts the loop past ~1.5× the pre-flight estimate**
   (the API analog of the pod runaway-kill guard). On a repeated same-cell crash
   (disk/OOM), **STOP that cell — don't retry-loop** (each failed pass re-spends).
   Never thrash-restart a judge loop to fix infra without accounting for re-judge cost.
5. **Pod-name scoping for the supervisor.** Give it the exact campaign pod-name
   prefix (e.g. `grid-*`) AND the explicit names of any unrelated pods on the
   shared RunPod account it must NEVER terminate/relaunch.
6. **Don't narrate idle pings.** Teammates idle between turns constantly. Only
   surface substantive events (gate verdict, fan-out pod IDs, failures) + your
   own scheduled checks.
7. **HF rate limits are a recurring failure — on BOTH the download and the
   read/monitor side.** This has bitten the workflow more than once; design for it
   from the start, don't react to it.
   - **Download herd (pods die at startup).** Launching the whole grid at once,
     where each pod cold-pulls the base model + dataset from the public Hub, →
     HTTP **429** → `LocalEntryNotFoundError` → the ERR-trap `sleep infinity`
     leaves a **RUNNING-but-dead** pod (so a "RUNNING" pod count *hides* the
     failure — your supervisor's "EXITED & no result" crash check won't catch it;
     grep the logs for `BOOTSTRAP-ERR`/`429`/`LocalEntryNotFound`). Fixes:
     (a) **prefetch the base model into the HF cache once per pod, with backoff
     retry**, before the run → `from_pretrained` becomes a warm-cache read, not a
     cold race; (b) **load your own artifacts (adapter/SAE) from the copy the
     bootstrap already downloaded to /workspace** — don't re-`snapshot_download`
     them at runtime; (c) **wrap EVERY download in retry-with-backoff** — the bug
     is usually ONE un-retried download (e.g. the dataset pull was bare while the
     model load was already in a retry loop); (d) **stagger launches** (~20–30s
     apart), don't fire 19 pods in 3s.
   - **Resolve-endpoint quota (the monitor lies).** The Hub caps **~5000 resolver
     requests / 5 min per token**. A poller that GETs each result file every round
     — ×N cells ×(multiple pollers + any runaway polling agent) — trips it; the
     resolve URL then returns a ~248-byte *"We had to rate limit you"* page that
     parses as "not done", so the monitor falsely reports **`done=0/N` while the
     grid is actually fine**. Fixes: poll the **tree API (1 request/round)** for
     presence; **fetch each cell's JSON at most once** (cache confirmed-done cells,
     never re-download them) so lifetime resolve calls ≈ N, not N×rounds;
     **detect the rate-limit page explicitly** and report "rate-limited, retrying"
     instead of silently counting 0; and **run exactly ONE poller** — see rule 8.
8. **Monitor with a `Monitor`/background-`bash` poll loop, NOT idle agents.**
   A general-purpose teammate **cannot `sleep`** (foreground sleep is blocked), so
   an agent told to "watch until done" either spins (re-invoking itself every few
   seconds — one burned 110k+ tokens doing nothing) or hydra-respawns child
   monitors, AND its polling is what trips the resolve quota in rule 7. Use the
   `Monitor` tool (or one `run_in_background` bash `until`-loop) for the *waiting*;
   it does the sleeping a process is allowed to do and emits one status line per
   interval. Reserve agents for *interpretation of finished results*, never for
   waiting. (The "team that checks in every 5 min" = a single Monitor heartbeat,
   not five sleeping agents.)

---

## Phase 0 — write the spec (you)
Write `CAMPAIGN.md` (see `experiments/fra_ln1_7b/CAMPAIGN.md` for a worked
example). Must contain: the cross-product/grid + which cells to skip and why;
the exact measurement math; the headline metric (+ how to recompute it post-hoc
from saved rollouts); n / seeds / models; HF output prefix layout
(`…/<cell>/<model>_seed<seed>/`); the files to reuse (don't rebuild); and the
`/dispatch_campaign` gotchas (cu124 override, ERR-trap→`sleep infinity` no-loop,
`/start.sh &` dockerArgs, streaming `tee` log, self-terminate, HF backup). Commit it.

## Phase 1 — spawn the lead (you)
`TeamCreate {team_name}` then spawn `campaign-lead` (general-purpose,
run_in_background:true, team_name). Brief: read CAMPAIGN.md + the reuse files;
build the parametrized orchestrator + dispatch (max reuse); **message you its
build+launch plan + cost BEFORE spawning pods**; you are sole repo writer;
report blockers. Approve/refine its plan (esp. add the smoke-gate if it didn't).

## Phase 2 — spawn the specialists + smoke-gate (you + lead)
When the lead asks (it will — see the #1 gotcha), spawn `gpu-supervisor` and
`results-analyst` (general-purpose, bg, same team), with the scoping + ownership
rules above. The lead drives the canary cell through all gate checks and reports
the verdict. You confirm before fan-out.

## Phase 3 — tranched fan-out (lead + team)
Lead launches tranches cheapest-reuse-first; supervisor keeps `grid-*` pods
healthy (relaunch lottery/failures via the cu124 pattern, confirm HF upload +
self-term); analyst judges + computes the metric per cell as they land and
assembles the results doc text. Lead commits it. Back EVERYTHING up to HF.

## Phase 4 — assemble + close (lead)
Lead commits the full results table (`GRID_RESULTS.md`), folds key findings into
the project writeup, shuts down teammates (SendMessage `{type:"shutdown_request"}`)
once complete, and confirms all pods self-terminated (cost hygiene).

## Sharp edges
- **Idle ≠ done.** A teammate messaging then idling is normal; don't treat as error.
- **Branch races.** Even with one writer, have the lead `git pull --rebase` before each push (pods clone origin, so the lead must keep origin current).
- **Canary becomes a real cell.** The smoke-test pod's output is a valid grid cell — don't throw it away; just don't wait for its full long-pole upload before fanning out once the *gates* (not the full sweep) pass.
- **Spec drift.** If you correct the spec mid-run (e.g. a miscount), tell the lead to fix `CAMPAIGN.md` — it's the only writer.

## Future: run the team headless on a RunPod CPU pod
Today the agents are **local, session-only** Claude processes — quit the local
session and orchestration (poll/relaunch/judge/assemble) stalls until it's back,
even though the GPU work is on RunPod and outputs are durable on HF. The planned
upgrade is to run the whole orchestration layer on a cheap **RunPod CPU pod** so
a long campaign keeps being supervised when local is off: a babysitter-style
headless driver (cf. `scripts/arditi_babysitter.py`'s DAG) that clones the repo,
holds the RunPod/HF/OpenAI keys, and runs the launch→poll→relaunch→judge→assemble
cycle without Claude-in-the-loop on the pod. Mirrors the "ML on RunPod, not local"
rule, extended to the orchestration brains.

## Outputs
- `CAMPAIGN.md` (spec), the parametrized orchestrator + `run_/launch_` dispatch, all rollouts + judged data on HF, `GRID_RESULTS.md`, findings folded into the writeup. Team shut down, pods terminated.
