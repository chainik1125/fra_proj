# Fisher-guided clean-recovery POC

POC implementation of the diagonal-Fisher / greedy variant of
`docs/dmitry/theory/fisher_proposal.md`, scoped to the TinyStories-33M
sleeper.  See `/Users/.../plans/can-you-review-fisher-proposal-md-robust-bumblebee.md`
for the review and scoping rationale.

## What it does

Two experiments, run in two control spaces, replicated across N seeds:

- **Experiment A** (§12-A).  Re-evaluate the existing 1-D α-sweep paths
  under the new Fisher arc-length `L_F`.  K=1 in each space.
- **Experiment B** (§12-B).  Greedy diagonal-Fisher feature selection
  across K candidate features (K=30 by default).  At each step pick
  `argmax_i |g_i| / √(F_ii + ε)`, step `−sign(g_i) · √(8 ln 2 · ρ / F_ii)`,
  line-search shrink, repeat.

Control spaces:

- **FRA OV→OV** — LN1 SAE features × `W_V₀` applied at
  `blocks.0.attn.hook_v`.  Matches `build_ov_hooks` in
  `../run_fidelity_experiment.py`.
- **Resid-mid** — `hook_resid_mid` SAE features applied additively at
  `blocks.0.hook_resid_mid`.

## Files

| File | Purpose |
|---|---|
| `fisher_utils.py`     | JSD, JSD-grad (§7), diagonal Fisher, greedy loop, 1-D Fisher path length |
| `control_space.py`    | `FRAOVControlSpace`, `ResidMidControlSpace`, share `forward_logits(θ)` interface |
| `run_fisher_poc.py`   | Main driver. `--mode {expA, expB, both}`, `--seed N` |
| `config.yaml`         | K, ρ, num_steps, α grid, candidate feature IDs |
| `hf_upload.py`        | Push result JSONs to an HF dataset repo |
| `babysitter.py`       | Polls HF, writes `summary.md` when all seeds present, exits |
| `auto_start_gpu.sh`   | One-shot entrypoint a GPU pod runs at boot |
| `auto_start_cpu.sh`   | One-shot entrypoint the babysitter pod runs at boot |
| `setup_pod.sh`        | Inner: clone branch + `uv sync` + HF login + GPU check |
| `setup_cpu_pod.sh`    | Inner: clone branch + minimal pip install for babysitter |
| `run_on_pod.sh`       | Inner: iterate `$SEEDS`, run, push to HF |
| `launch_all.sh`       | Provisions all pods via RunPod GraphQL API |

## Pod architecture

```
[seed 0] L40S GPU pod ─┐
[seed 1] L40S GPU pod ─┤              ┌──────────────┐
[seed 2] L40S GPU pod ─┼─push JSONs──▶│  HF dataset  │
[seed 3] L40S GPU pod ─┤              │  fisher-poc- │
[seed 4] L40S GPU pod ─┘              │  tinystories │
                                      │  -sleeper    │
                                      └──────┬───────┘
                                             │
                                  CPU pod ───┘ polls every POLL_SEC,
                                              writes summary.md when
                                              all seeds present.
```

Each GPU pod self-stops on completion; the CPU babysitter self-stops
after pushing `summary.md`.  No SSH-between-pods required — HF is the
synchronisation point.

## Launching

You need three things set before kicking off:

```bash
export RUNPOD_API_KEY=...
export HF_TOKEN=...                # write access to HF_REPO
# Optional overrides:
# export SEEDS="0 1 2 3 4"
# export GPU_TYPE_ID="NVIDIA A40"
# export HF_REPO="dmanningcoe/fisher-poc-tinystories-sleeper"
```

Then push the branch (one-time) and launch:

```bash
git push -u origin dmitry/fisher-poc
bash experiments/tinystories_sleeper/fisher_poc/launch_all.sh
```

Cost estimate: TinyStories-33M is tiny.  Each GPU pod runs for ~5 min
(model load + ~2 min/seed); each L40S costs about $0.86/hr → ~$0.07
per pod → ~$0.40 for 5 seeds.  CPU babysitter is ~$0.02/hr.

## Outputs (on HF dataset repo)

```
expA_seed0.json   expB_seed0.json
expA_seed1.json   expB_seed1.json
…
expA_seed4.json   expB_seed4.json
run_seed*.log
summary.md             ← human-readable aggregate (per-seed + mean ± std)
babysitter.log
status_stalled.json    ← only present if a seed timed out
```

The summary lists, for each (space × method): mean ± std of J_clean
(bits), CRF, and either greedy accepted-step count or 1-D best-CRF α.

## What is **not** here (deferred from the proposal)

- §5 / §10.8: full empirical-Fisher natural-gradient steps.
- §11: NNGeometry / Curvlinops / BackPACK / ASDL / Opacus wrappers.
- §12-C: full natural-gradient path.
- ASR rollouts (config `asr:` block is currently inert).

Scoping rationale lives in the plan file.
