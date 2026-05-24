# Fisher-POC — current state (v2 complete, v3 incoming)

## What's been measured

TinyStories-33M sleeper, jamie's `jsd_eval.py` harness
(200 deployment prompts × 16-token multinomial sampling,
DECODE_SEED=0). Three SAE seeds {0, 1, 2}. Two control spaces:
**OV→OV** (steering at `blocks.0.attn.hook_v` via the LN1 SAE × W_V₀)
and **resid-mid** (additive at `blocks.0.hook_resid_mid` via the
resid-mid SAE).

Two methods, both run on the same SAEs:

- **Method A** — single-feature α-sweep at the
  attribution-selected winner (α ∈ {0, 0.5, 1.0, …, 4.0}, 9 points).
  This is the May 8 paper recipe; data is in
  `results/jsd_alpha_sweep_6seeds.json` on the jamie pod.
- **Method B** — greedy diagonal Fisher (proposal §6) over a K=20
  candidate basis. Selects features by `|g_i|/√F_ii`, line-search
  step in the chosen direction, repeat for 12 steps.

## Numbers as of v2 (means over seeds 0, 1, 2)

| cell | method | J_clean | path length | ASR |
|---|---|---:|---:|---:|
| baseline (θ=0) | (both) | 0.987 | 0 | 0.980 |
| **4k OV** | Method A · α=2 | 0.510 | 2.0 | low |
| | Method B · Fisher ρ=1e-2 | **0.547** | **1.16** | **0.018** |
| 4k resid-mid | Method A · α=2 | 0.657 | 2.0 | low |
| | Method B · Fisher ρ=1e-2 | 0.967 | 0.67 | 0.917 (barely moved — bug, see below) |
| 50k OV | Method A · α=2 | 0.510 | 2.0 | low |
| | Method B · Fisher ρ=1e-2 | 0.685 | 1.13 | 0.202 |

Headline: on **4k OV**, Method B reaches roughly the same J_clean as
Method A (0.55 vs 0.51) at **~58% of the path length** (L_F = 1.16
vs |α| = 2.0). ASR drops the same way. *That* is the proposal's
claim.

## Two bugs found in v2

### Bug 1 — wrong candidate set for resid-mid (concrete, fixable)

The Fisher loop used the top-20 candidates from jamie's *OV*
attribution (`feature_set_pipeline --selection_method jamie --top_k 20`)
for **both** spaces. For the resid-mid cell that's the wrong basis:
feature index N in the LN1 SAE is the trigger-detector, but feature
index N in the *resid-mid* SAE is some unrelated direction. The
resid-mid Fisher loop was walking around a random subspace.

The right candidates for resid-mid come from
`find_downstream_winners.py`-style attribution on the resid-mid SAE
(per `JSD_OVERLAY_WRITEUP.md:51-58`). The pod has
`results/downstream_winners_6seeds.json` and
`results/conventional_winners_per_layer.json` — re-use one of those.

### Bug 2 — teacher-forced JSD ≠ rollout JSD (the proposal's §15 caveat firing)

Fisher's optimization signal is single-position teacher-forced JSD
(JSD of the *next-token* distribution given the prompt). The
evaluation metric is 16-position sampling JSD (autoregressive
multinomial decode for 16 tokens, JSD averaged over those 16
positions).

For **OV** at Fisher endpoint:
- Teacher-forced J (single position) ≈ 0.07 ← Fisher's optimization
  signal, "near-perfect clean recovery" at position 0
- Sampling J (16 positions, autoregressive) ≈ 0.5 ← what jsd_eval.py
  reports as the headline; what's directly comparable to Method A
- ASR = 0.02 ← qualitatively, the sleeper is silenced

The 0.07 → 0.5 jump is **autoregressive error compounding**: at t=0
the steered model agrees with clean within 0.07 bits, but the
steered hook keeps firing as the sample unfolds, so per-position
JSD drifts upward through t=1, 2, …, 15. Average across 16
positions sits at 0.5.

For **resid-mid** the picture differs: teacher-forced J only fell
0.35 → 0.27 (because of bug 1 above), so Fisher's optimization was
weak in the first place, and rollouts saw essentially no movement.

**The headline metric is the 16-position sampling JSD**, because
that's what `jsd_eval.py` reports and what's comparable to Method
A's α-sweep numbers. The teacher-forced number is the optimizer's
proxy — close to a strict lower bound on the rollout metric,
because of compounding.

## What v3 changes

Both fixes, on the same pod and same harness:

1. **Pull resid-mid candidates from the right attribution file**
   (`downstream_winners_6seeds.json` on the pod) and pass them as
   the K=20 basis for the resid-mid cells.
2. **Make Fisher optimize the 16-position sampling JSD directly**
   instead of the 1-position teacher-forced JSD. Concretely: at
   each Fisher step:
   - Autoregressively sample 16 tokens at the central θ (one
     full rollout, cached for the step).
   - For each FD perturbation θ ± ε·e_i, teacher-force the steered
     model on those fixed 16 tokens, capture log_softmax at each
     position.
   - Compute JSD averaged over all 16 positions. That's the
     `J_clean(θ)` the gradient and the Fisher diagonal are computed
     against.

   The bias from fixing the tokens (perturbed θ wouldn't actually
   sample those exact tokens) vanishes as ε → 0 and is the standard
   FD trick when there's a discrete sampling step.

The compute cost goes up ~10× per Fisher step (the autoregressive
rollout adds ~32 forward passes per step, vs ~40 in the current
1-position Fisher). Still ≈ 15-20 min wall on the A40 pod for all
6 cells.

## Expected outcome

If v3 works as intended:

- **OV row**: Fisher J_clean (sampling, at endpoint) should drop
  meaningfully below 0.55 — possibly to ~0.40-0.45, near Method A's
  α=3 asymptote. ASR stays near 0.
- **Resid-mid row**: with the correct candidate basis, Fisher
  should at least match Method A's J_clean ≈ 0.66 at α=2; the
  Fisher path-length-vs-JSD plot becomes the meaningful comparison.
- **Path-length efficiency claim**: should hold or strengthen on
  OV. Whether it holds on resid-mid is an empirical question.

## Open caveats (not addressed by v3)

- Diagonal Fisher only; off-diagonal terms could matter at K=20.
- Single attribution method per space; no ablation of candidate set
  size.
- 4k SAEs only for resid-mid (no 50k resid-mid in this batch).
- Cross-prompt OOD generalisation not tested.

## File map (this directory)

| file | purpose |
|---|---|
| `STATE.md` | this writeup |
| `summary_v2_final.md` | v2 writeup with the 3-panel plot interpretation |
| `comparison_v2_final.png` | the 3-panel (4k OV / 4k resid / 50k OV) figure |
| `fisher_v2_4k.json` | Fisher trajectories, 4k, ρ=1e-4 (too small) |
| `fisher_v2_4k_bigrho.json` | Fisher trajectories, 4k, ρ=1e-2 (headline numbers) |
| `fisher_v2_50k_ov_bigrho.json` | Fisher trajectories, 50k OV, ρ=1e-2 |
| `jsd_alpha_sweep_6seeds.json` | Method A reference data (existing) |
| `fisher_v3_*.json` | (incoming) rollout-integrated Fisher results |
| `comparison_v3.png` | (incoming) v2 vs v3 head-to-head |
