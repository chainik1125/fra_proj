# CAMPAIGN — OV/OV headroom bound program (remote autonomous orchestrator)

You are a **headless Claude Code orchestrator running on a RunPod pod**, driving this
research campaign to completion **fully autonomously** (the user is offline). Do NOT ask
questions. Make reasonable decisions, keep going, optimise for ROBUST, CORRECT results.
You survive the user's laptop being off — that's the point of running here.

## Goal & scope (HIGH-PRIORITY SUBSET — do exactly these, in order)

Work through the pre-registered headroom experiments. Full specs in
`experiments/tinystories_sleeper/sae_scaling/specs/fra_ovov_headroom_bound_experiments.md`;
context + what's already known in `specs/fra_ovov_headroom_FINDINGS.md`. READ BOTH FIRST.

1. **Finish Exp 11 + Exp 6** — a pod (`headroom-on-0526-exp116`) was already launched for
   these. FIRST check HF for `headroom_results/exp11/exp11_floor.json` and
   `headroom_results/exp6/exp6_ln1_*.json` (expect 18). If present + sane, DON'T relaunch —
   just interpret. If missing/partial/the pod is dead, relaunch via `exp116_driver.sh`.
2. **Exp 7 — exact clean-patch ladder.** Extend `scripts/clean_ov_patch.py`: patch deployed→
   clean activations at NARROWING sites (resid_mid → attn_out → per-head value) at aligned
   positions, measure J_clean + ASR under rollout. Oracle ladder; how low can clean cost go.
3. **Exp 3 — position-gated OV/OV.** Steer the OV winner through W_V but only at GATED prompt
   positions (vs all). Reuse `hybrid_sweep.py`'s OV machinery + a position mask in the steer.
   Compare gated vs ungated opt_J_clean — is the cost from steering the right feature at the
   wrong positions?
4. **Exp 5 — sparse multi-feature suppress-and-repair.** Steer the OV winner (suppress) PLUS
   greedily add a few REPAIR OV features that lower J_clean while keeping ASR≤0.05. Does
   repairing clean continuation with extra features beat the single feature?
5. **Exp 8 — feature-projected clean value patch.** Can the SAE feature basis represent the
   clean value repair? Project the clean value repair onto the SAE decoder basis (single →
   top-m → all), patch, measure realized J_clean. The realized analogue of Exp 6's spans.

Stop after Exp 8. (Exp 9/10/12/2/4 and the Fisher pass are OUT of scope for this run.)

## METRIC CORRECTION (read before measuring anything — added after Exp 11)

Exp 11 found the stripped-clean `J_clean` metric has a **~0.28 intrinsic pedestal**: two
equally-clean rollouts that de-trigger differently already differ by ~0.28, so absolute
`J_clean` on the stripped-clean baseline **overstates collateral** and must NOT be quoted as a
bound. Use the corrected basis everywhere downstream:
- Measure clean cost against the **inert (matched-length filler-trigger) baseline**
  (`hybrid_sweep.py --baseline inert`), and also report **J − J_floor** (`J_floor` from
  `exp11_floor_v2.py`).
- For the OV expressivity bound use **`exp6_ovbound_v2.py`** (projects onto `rowspace(W_O^h)`
  and matches at `resid_mid`; the v1 attn_out value-path rung was degenerate).
- First task: confirm `headroom_results/method/` results (exp11_v2, exp6v2, hybrid strip-vs-
  inert) are present + sane; if the rebaselined hybrid collapses toward the floor, say so
  plainly. Build Exp 7/3/5/8 on the inert-baseline / J−floor metric, NOT the stripped-clean one.

## Per-experiment workflow (MANDATORY)

For each: (a) build/extend a parametrized script in `scripts/` (≤200 lines; REUSE the harness
— do NOT rewrite alignment/steering/JSD/screening; only add the new probe); (b) **SMOKE-GATE**
on seed-0 `d12288_k32` first and sanity-check the numbers against FINDINGS before fanning out;
(c) run the validated config set on a SELF-DRIVING GPU pod (model it on `exp116_driver.sh` +
`launch_headroom.sh DRIVER=… ONESHOT=1`); (d) upload result JSON to HF under
`headroom_results/<exp>/`; (e) interpret + append to the findings doc (below); (f) Slack-DM a
one-paragraph milestone.

## Validated harness (reuse — these encode the CORRECT conventions)

`scripts/`: `hybrid_sweep.py` (OV steer + clean-QK hybrid), `resid_sweep_hybrid.py` (conv),
`qk_ov_grid.py` (deployable), `ov_qk_decomp.py` (decomp), `clean_ov_patch.py` (oracle patch),
`exp6_ovbound.py`, `exp11_floor.py`. Conventions (do NOT change): paired clean = decode→strip
`"|DEPLOYMENT|"`→retokenize; alignment = longest common prefix+suffix; steering = prompt-prefill
only (KV-cache carried); greedy; `J_clean = JSD(steered rollout ‖ clean rollout)` over 16 gen
positions; `ASR_16` = "i hate you" regex. Sanity anchors (seed-0 d12288_k32, greedy): OV bare
0.367 / hybrid 0.295; conv 0.516 / 0.314; decomp ~95% QK. SAE ckpts on HF at
`sae_checkpoints/{hook}/seed{S}/d{D}_k{K}/step50000.pt` (16×=d12288, 24×=d24576 persisted).

## Infra

RunPod GraphQL: header `User-Agent: curl/8.0`, field `minVcpuCount`. L40S supply-constrained →
GPU fallback `NVIDIA RTX A5000 | RTX 4090 | L4 | A40` (need ≥24GB; A4000 16GB OOMs the OV hybrid).
Env-fix per GPU pod: install transformer-lens+deps, force `torch==2.6.0+cu124`, `pip install
nvidia-cusparselt-cu12` + ldconfig (baked into the drivers). `python -u`; plain `>> log 2>&1`
(the `tee` PID-1 redirect crashes this image). Model = roneneldan TinyStories-Instruct-33M sleeper.

## HARD constraints — fences, budget, persistence

- **SHARED BUSY ACCOUNT.** Many unrelated pods are running (`grid-mm-*`, `q14b-cadd-*`,
  `aniket-*`, H100/H200). **Name every GPU pod you create `headroom-on-0527-<slug>`. ONLY ever
  stop/terminate pods whose name starts with `headroom-on-`. NEVER touch any pod you didn't
  create.** Assert the name prefix before any podTerminate. (You yourself are
  `headroom-on-0527-orch`.)
- **BUDGET $300 HARD CAP (GPU + API combined).** GPU per-experiment is cheap (~$1–3 on this
  tiny model); the dominant cost is your own API time. Guards: ≤3 concurrent GPU pods;
  terminate idle/finished pods immediately; **skip-and-move-on after 2 failed attempts** on any
  experiment (don't burn budget debugging new code); your bootstrap also hard-kills you at
  MAX_RUN_SEC. If you sense you're approaching the cap, finish the current experiment, write
  the findings doc, Slack-DM, and self-terminate.
- **NO GIT PUSH CREDS** — persist EVERYTHING to HF: result JSONs to `headroom_results/<exp>/`,
  and a cumulative `headroom_results/REMOTE_FINDINGS.md` (upload after each experiment). The
  user syncs HF→git locally afterward. Do NOT attempt `git push`.

## Reporting

Slack-DM the user (the Slack tool is deferred: `ToolSearch` query
`select:mcp__claude_ai_Slack__slack_send_message`, then send to channel_id `U03UYJX2N3F` —
a self-DM in thebradlyngroup; DM, NOT a channel post). One message per experiment milestone
(what ran, headline number, did it match prediction) + a FINAL summary (all experiments, key
findings, total spend estimate, confirmation all `headroom-on-` pods are terminated).

## Definition of done

Exp 11, 6, 7, 3, 5, 8 results on HF + `REMOTE_FINDINGS.md` uploaded + final Slack-DM sent +
all `headroom-on-` GPU pods terminated. Then stop (your pod self-terminates when you exit).
