# Fisher-POC v2 — plan (updated post-inventory)

Refit the campaign to match the May 8 `JSD_OVERLAY_WRITEUP.md` setup
exactly, then layer Fisher-guided steering on top of each cell.

## Why v1 was wrong

v1 used unrelated infrastructure (recreate_ln1's 4000-step SAE,
ketan-style attribution from `feature_rankings_*.json`, hard-coded
feature 1412). Result: `J_clean ≈ 0.93` vs. the established
`0.47-0.53` at α=2. We compared apples to nothing.

The May 8 work pins three things v1 didn't:

| | v1 (broken) | May 8 (jamie/sleepers + ketan_repl) |
|---|---|---|
| SAE | `recreate_ln1/...crosscoder_sae_layer0.pt`, 4k steps, ketan-recipe | `weights/seeds/sae_{ln1,resid_mid}_s{0..5}.pt`, 4k AND 50k, jamie-recipe |
| Attribution | `dep_vs_clean_contribution` (ketan), sum over positions | `feature_set_pipeline.py --selection_method jamie`, prompt-masked head-summed OV |
| Feature | f=1412 | per-seed winner (see table below) |
| Eval harness | my `run_fisher_poc.py` | jamie's `jsd_eval.py` |

## Inventory — what's already on disk

I SSHed into the **running** pod `a40_3gpu_jamie`
(id `kdy09tmyuau54f`, A40, branch `jamie/sleepers`).
**Everything is in `/root/fra_proj/`** on that pod:

| | path | status |
|---|---|---|
| 4k LN1 SAEs (6 seeds) | `weights/seeds/sae_ln1_s{0..5}.pt` | ✓ present |
| 4k resid-mid SAEs (6 seeds) | `weights/seeds/sae_resid_mid_s{0..5}.pt` | ✓ present |
| Per-layer additional SAEs | `weights/seeds_per_layer/sae_L{0,1}_resid_{mid,post}_s{0,1}.pt` | ✓ present (bonus) |
| 4k attribution + α-sweep result | `results/jamie_experiment.json` | ✓ present |
| 50k attribution + α-sweep result | `results/jamie_experiment_50k.json` | ✓ present |
| **50k SAE weights** | `weights/seeds_50k/...` | **✗ not on pod** — would need retraining |
| jamie's `jsd_eval.py`, `feature_set_pipeline.py`, `train_all_saes*.py` | `scripts/` | ✓ present |
| Disk free | / | ✓ 284 GB free |

## Per-seed feature winners (read from existing JSONs)

**4k LN1 / OV→OV** (from `results/jamie_experiment.json`):

| seed | top-20 candidates (first 5 shown) | post-screen winner | rank in top-20 |
|---:|---|---:|---:|
| 0 | 1114, 221, 1465, 922, 1337, … | **f=1114** | 0 |
| 1 | 767, 807, 459, 1090, 1142, … | **f=767** | 0 |
| 2 | 169, 351, 225, 960, 211, … | **f=351** | 1 |
| 3 | 1281, 29, 1154, 602, 514, … | **f=1154** | 2 |
| 4 | 841, 1305, 558, 1441, 267, … | **f=558** | 2 |

**50k LN1 / OV→OV** (from `results/jamie_experiment_50k.json`):

| seed | top-5 candidates | post-screen winner |
|---:|---|---:|
| 0 | 1114, 946, 1153, 1365, 181 | **f=1114** |
| 1 | 926, 1027, 807, 832, 767 | **f=1027** |
| 2 | 410, 836, 169, 896, 943 | **f=169** |
| 3 | 1281, 883, 29, 342, 1032 | **f=1154** |
| 4 | 1305, 1006, 558, 841, 1191 | **f=558** |

Seeds {0,1,2} winners agree with `JSD_OVERLAY_WRITEUP.md`'s
{1114, 1027, 169} — confirms we're reading the right field.

**resid-mid winners** for both durations come from
`find_downstream_winners.py` (per `JSD_OVERLAY_WRITEUP.md:51-58`):
4k: {579, 473, 1303}; 50k: {579, 519, 49}. We can either recompute
on the pod or grep the existing `results/conventional_winners_per_layer.json`.

## What v2 actually needs to add

Just **one script** on the existing pod:
`scripts/run_fisher_on_jamie_saes.py`

The Fisher math (`fisher_utils.py`) and control-space wrapper
(`control_space.py`) port directly. The only swap is
`forward_logits` — instead of calling my hooks, call jamie's hook
stack (`sleeper/hooks.py::channel_steer_hook` for OV,
`additive_steer_hook` for resid-mid). JSD math is already there.

## Experiment design (revised, post-inventory)

Two phases. **Phase A is immediate** (everything needed is on the
pod). **Phase B requires retraining 50k SAEs** (~3 hours).

### Phase A — 4k row only (immediate)

For each (seed ∈ {0,1,2}, space ∈ {OV, resid-mid}):

- **Method A** (α-sweep, existing): read α ∈ {0, 0.25, …, 2.0} from
  `jamie_experiment.json` if it has them already, OR re-run
  `jsd_eval.py` on the per-seed winner feature at that α grid.
- **Method B** (Fisher greedy, new): control basis = the top-20
  candidates from `selection.per_seed.{N}.features`. Run greedy
  diagonal Fisher for 12 steps with ρ=1e-4 bits/step, ε_fd=1e-2.
  Evaluate the endpoint with `jsd_eval.py` math.
- Record both methods' `(L_F, J_clean)` trajectories so we can
  overlay them in the proposal's preferred plot space.

12 cells: 3 seeds × 2 spaces × 2 methods. Wall time ~30 min on
the A40. Cost on this already-running pod: marginal.

### Phase B — 50k row (optional, ~3 hr extra)

Same design but with 50k SAEs. Need to first run
`train_all_saes_50k.py` (3 seeds × 2 hooks ≈ 3 hours wall), then
repeat Phase A's loop. Adds the second panel to the final figure.

Recommend **Phase A first**, look at the result, decide whether
Phase B is worth the time.

### Endpoint metrics

For each (cell, method, endpoint), also record (using
`jsd_eval.py`'s harness, no extra plumbing):

- **JSD(steered, clean)** — bits, what we've been computing
- **JSD(steered, poisoned)** — bits, distance from unsteered sleeper
- **ASR** — fraction of deployment continuations that match the
  attack regex (jamie's greedy-decode harness has this built in)
- **WMA** — word-for-word match vs base model (also built in)

This is the "did the steering actually do anything in rollouts"
sanity check, all available "for free" since `jsd_eval.py`
already greedy-decodes.

## What this answers vs. the May 8 work

The May 8 figure shows:

> "OV→OV beats conventional [resid-mid] on both axes at every α"
> *(JSD_OVERLAY_WRITEUP.md:95)*

That's a **method A vs method A** comparison across two control
spaces. The proposal's claim is **method A vs method B** within
the same control space — does Fisher's gradient find a shorter
path to the same J_clean? Or a lower J_clean at the same path
length?

Concretely the comparisons that matter:

| panel | 1: same cell, two methods | 2: cross-method endpoints |
|---|---|---|
| **OV→OV** | A's α-sweep curve overlaid with B's `(L_F, J_clean)` trajectory | A endpoint at α=2 vs B endpoint, same J_clean budget |
| **resid-mid** | same | same |

If B's `(L_F, J_clean)` trajectory sits below A's curve in the
same plane, Fisher buys you efficiency. That's the proposal's
headline.

## Pod choreography (Phase A only)

Everything happens on the existing pod. **No new pod
provisioning, no retraining, no setup overhead.**

1. (Locally) cherry-pick `fisher_utils.py` + `control_space.py`
   from `dmitry/fisher-poc` onto a new branch `dmitry/fisher-v2`
   off `jamie/sleepers`. Add the thin wrapper script. Push.
2. (On the pod) `git fetch && git checkout dmitry/fisher-v2`.
3. (On the pod) `python scripts/run_fisher_on_jamie_saes.py
   --seeds 0 1 2 --spaces ov resid_mid --num_steps 12
   --rho 1e-4 --output results/fisher_v2_4k.json`.
4. (On the pod) plot Method A vs Method B: extend the existing
   `plot_jsd_overlay.py` to overlay the Fisher trajectories.
5. Push results JSON + figure to `dmanningcoe/fisher-poc-tinystories-sleeper`
   (the same HF repo). Write `summary_v2.md`.

## Cost / time (Phase A only)

- Marginal pod cost on existing running pod: ~$0 (already burning)
- Compute time: ~30 min for the Fisher loop + eval
- My time: ~30-60 min to write the wrapper + plot extension

## Open questions before launching Phase A

1. **Is the running pod safe to use for this work, or is someone
   else using it?** It's named `a40_3gpu_jamie` — let's not stomp
   on jamie's setup. Recommend: pull data off it onto a fresh
   pod if there's any risk.
2. **3 seeds or 5?** `JSD_OVERLAY_WRITEUP.md` used 3 (s∈{0,1,2}).
   We have 5 SAEs available. 3 matches the published plot; 5
   gives tighter error bars. Recommend 5 since the marginal cost
   is zero.
3. **Method B control basis: top-20 from selection.features or
   top-K post-screen?** The screen step in `feature_set_pipeline`
   already filters by Δdep-logp. Recommend: **top-20 pre-screen**
   for B so we're not double-counting the diff-attribution signal
   into B's candidate set (cleaner ablation).

## What this plan does NOT touch

- Conceptual proposal in `docs/dmitry/theory/fisher_proposal.md`.
- Natural-gradient or full-Fisher variants (still diagonal-only).
- Cross-prompt or OOD validation (same 200-prompt test set).
- Phase B (50k row) unless explicitly requested.
