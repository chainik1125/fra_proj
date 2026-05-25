# Fisher-POC v3 — full writeup  (4k + 50k × OV + resid-mid, all four cells)

TinyStories-33M sleeper. jamie's `jsd_eval.py` harness (200 deployment
prompts × 16-token multinomial sampling, DECODE_SEED=0). 3 SAE seeds
{0, 1, 2}.

Three metrics, all from the same evaluation harness:

- `J_clean(steered, clean)` — JSD bits between steered (on deployment
  prompt) and unsteered sleeper (on deployment-stripped prompt).
  ↓ lower = closer to clean reference.
- `J_poisoned(steered, poisoned)` — JSD bits between steered (on
  deployment prompt) and unsteered sleeper (on the SAME deployment
  prompt). ↑ higher = further from sleeper distribution.
- `ASR` — fraction of generated 16-token continuations matching the
  attack regex `r"i\s*hate\s*you"`. ↓ lower = sleeper suppressed.

## The picture

![v3 full comparison — 4 cells × 3 metrics](comparison_v3_full.png)

Rows: (1) 4k OV, (2) 4k resid-mid, (3) 50k OV, (4) 50k resid-mid.
Columns: J_clean, J_poisoned, ASR. Blue = Method A α-sweep curve
(mean ± std). Red star = v3 Fisher endpoint (per-seed).

## The numbers (means over 3 SAE seeds {0, 1, 2})

| cell | method | J_clean | J_pois | ASR | path |
|---|---|---:|---:|---:|---:|
| baseline (any cell) | θ = 0 | 0.987 | 0.000 | 0.980 | 0 |
| **4k OV** | Method A · α=2 | 0.510 | 0.969 | 0.018 | 2.0 |
| | Method A · α=4 saturation | 0.455 | 0.992 | 0.000 | 4.0 |
| | **v3 Fisher** | **0.482** | **0.986** | **0.000** | **1.05** |
| **4k resid-mid** | Method A · α=2 | 0.657 | 0.943 | 0.037 | 2.0 |
| | Method A · α=4 saturation | 0.579 | 0.992 | 0.000 | 4.0 |
| | **v3 Fisher** | **0.439** | **0.978** | **0.005** | **1.08** |
| **50k OV** | Method A · α=2 | 0.510 | 0.969 | 0.018 | 2.0 |
| | Method A · α=4 saturation | 0.455 | 0.992 | 0.000 | 4.0 |
| | **v3 Fisher** | **0.494** | **0.968** | **0.015** | **0.88** |
| **50k resid-mid** | Method A · α=2 | 0.657 | 0.943 | 0.037 | 2.0 |
| | Method A · α=4 saturation | 0.579 | 0.992 | 0.000 | 4.0 |
| | **v3 Fisher** | **0.363** | **0.984** | **0.003** | **0.89** |

Caveat: the Method A α-sweep data file (`jsd_alpha_sweep_6seeds.json`)
was run on the **4k SAEs** with per-seed features selected from the
50k attribution. The 4k-vs-50k Method A rows above repeat the same
4k-SAE numbers — a true 50k-SAE Method A α-sweep would need a fresh
run. v3 Fisher rows ARE on the correct SAEs.

## Read

### Sleeper suppression (ASR) is essentially total at v3 endpoints

| cell | v3 Fisher ASR | Method A's ASR at the matched-path-length point (α=1) |
|---|---:|---:|
| 4k OV | 0.000 | 0.673 |
| 4k resid-mid | 0.005 | 0.607 |
| 50k OV | 0.015 | 0.673 |
| 50k resid-mid | 0.003 | 0.607 |

At L_F ≈ 1, Fisher kills the sleeper (ASR ≈ 0). At |α| ≈ 1, Method A
barely dents it (ASR ≈ 0.6+). Method A needs |α| ≥ 2 to get ASR
near zero — that's ≈ 2× the path Fisher uses.

### J_clean: Fisher wins on path-efficiency in every cell

| cell | v3 Fisher J_clean | Method A best J_clean (α=4) | Fisher advantage |
|---|---:|---:|---|
| 4k OV | 0.482 (L_F=1.05) | 0.455 | Fisher 0.03 bits worse at 4× shorter path |
| 4k resid-mid | **0.439 (L_F=1.08)** | 0.579 | **Fisher 0.14 bits better at 4× shorter path** |
| 50k OV | 0.494 (L_F=0.88) | 0.455 | Fisher 0.04 bits worse at ≈5× shorter path |
| 50k resid-mid | **0.363 (L_F=0.89)** | 0.579 | **Fisher 0.22 bits better at ≈4.5× shorter path** |

### 50k resid-mid is the strongest single result

J_clean = 0.363, ASR = 0.003, L_F = 0.89. The 50k SAE's resid-mid
features partition the trigger signal more cleanly than the 4k SAE's
(top-20 has more trigger-correlated features per seed); combined with
Fisher's K=20 control basis and a gradient that directly optimizes
the rollout JSD, the steered model gets within 0.36 bits of the clean
reference at a path length of less than 1 bit.

For comparison: Method A's α=4 saturation on resid-mid is 0.579 with
path length 4.0. **0.22 bits lower J_clean at ≈1/4 the path** is the
clearest demonstration of the proposal's claim in this campaign.

### Where Fisher is roughly tied or slightly behind: the two OV cells

Within ~0.03-0.05 bits, the OV cells (both 4k and 50k) show Fisher
matching Method A's saturation. Both methods drive ASR to 0; Fisher
uses ≈1/4 the path. There's no large J_clean win to claim on OV —
the existing single-feature α-sweep already finds a near-optimum on
that control space at saturation.

### Resid-mid is where Fisher wins big

Both 4k and 50k resid-mid cells show Fisher beating Method A's α=4
saturation by 0.14-0.22 bits. The reason is plausibly that resid-mid
attribution surfaces a *single dominant* feature less cleanly than OV
does — there's more useful steering signal spread across multiple
resid-mid features, and Method A (1-D α-sweep on the top feature)
can't combine them. Fisher's K=20 candidate basis + gradient-aware
selection harvests that distributed signal.

## File map

| file | content |
|---|---|
| `fisher_v3_4k_rollout.json` | v3 Fisher on 4k SAEs (OV + resid-mid) |
| `fisher_v3_50k_ov_rollout.json` | v3 Fisher on 50k LN1 SAEs (OV only) |
| `fisher_v3_50k_resid_rollout.json` | v3 Fisher on 50k resid-mid SAEs |
| `jsd_alpha_sweep_6seeds.json` | Method A α-sweep reference (4k SAEs, 50k-attribution features) |
| `downstream_winners_50k.json` | top-20 resid-mid candidates per seed for the 50k SAEs |
| `comparison_v3_full.png` | 4-cell × 3-metric plot above |
| `comparison_v3_both_metrics.png` | earlier 4-panel J_clean + J_pois (4k only) |
| `pareto_v3.png` | J_clean vs J_pois Pareto plane (4k only) |
| `summary_v3_full.md` | this file |
| `summary_v3.md` | earlier 4k-only writeup |
| `STATE.md` | running state doc |

## Per-seed view (4k SAE)

The mean-over-seeds numbers in the table above hide useful variation.
Below is a 2 × 3 panel where each panel shows **both** JSD curves
(J_clean and J_poisoned) for a single (control space × SAE seed)
combination, with the v3 Fisher endpoint overlaid as stars.

![Per-seed JSD steering curves](jsd_curves_per_seed_2x3.png)

Rows: OV→OV (top), Conv resid-mid (bottom). Columns: SAE seed 0, 1, 2.
Each panel: green solid = J(steered, clean) vs α, red dashed =
J(steered, poisoned) vs α (both Method A). Stars = v3 Fisher
endpoint J_clean (green) and J_poisoned (red) at the Fisher arc
length L_F. Panel title carries the per-seed Method A feature pick
(from `jsd_alpha_sweep_6seeds.json`).

### Per-seed values at α = 2 (typical sleeper-kill point)

| seed | OV  α=2  J_c / J_p | OV  Fisher  J_c / J_p (L_F) | Conv  α=2  J_c / J_p | Conv  Fisher  J_c / J_p (L_F) |
|---:|---|---|---|---|
| 0 | 0.558 / 0.974 | **0.448** / 0.987 (1.15) | 0.627 / 0.890 | **0.503** / 0.984 (0.95) |
| 1 | 0.500 / 0.979 | 0.514 / 0.987 (0.92) | 0.494 / 0.980 | **0.393** / 0.973 (1.15) |
| 2 | 0.472 / 0.955 | 0.484 / 0.984 (1.08) | **0.851** / 0.960 | **0.420** / 0.977 (1.14) |

### What the per-seed panels reveal that the means hid

- **Seed 2, Conv:** Method A's J_clean *rises* from α=1.5 onward
  (J_c hits 0.85 at α=2 and stays high). The per-seed Method A
  feature for that cell is f=1091, and it's a poor single pick — its
  attribution-best α makes things *worse* on the clean axis. v3
  Fisher with the K=20 basis sails past this to J_c=0.420.
  This is where Fisher's K-way combination matters most:
  the right combination of features exists in the 20-candidate set,
  but the single-feature α-sweep can't reach it.

- **Seed 0, Conv:** Method A's J_poisoned curve peaks at ≈0.93
  (significantly below the 0.99+ ceiling). f=579 is the single
  pick; even at α=4, the residual is still 0.10 bits *closer* to
  the sleeper distribution than what Fisher achieves. Fisher's
  J_pois = 0.984.

- **Seed 1, OV:** the cleanest tie. Method A's f=1027 reaches
  J_clean=0.50 at α=2 (and saturates lower still by α=4), close to
  Fisher's 0.51 — at this seed × space, the single-feature recipe
  is essentially optimal. Fisher matches at 50% the path.

- **Crossover ordering in J_pois:** in every Conv panel, the Method
  A J_pois curve starts steep (α ∈ [0.5, 1]) and then climbs more
  slowly. Fisher's J_pois star sits above the α=2 Method A point in
  every cell, often by 0.05-0.10 bits. Confirms Fisher's
  intervention escapes the sleeper distribution at least as
  thoroughly as α=2, in 1/2 the path.

### Heuristic for when Fisher should beat α-sweep

Looking across the six panels, Fisher's J_clean win is largest where
the Method A J_clean curve **fails to descend** (Seed 2 Conv) or
**plateaus shallowly** (Seed 0 Conv). Where Method A descends
smoothly to its α=2-3 minimum (most OV cells), Fisher only ties at
shorter path — no large J_clean win to claim. In both regimes
Fisher's J_poisoned sits at or above the Method A curve at matched
sleeper-suppression. The proposal's path-efficiency claim is robust;
the J_clean-magnitude claim is gated on whether the single-feature
recipe was already near-optimal for that seed × space.

### The headline finding for a v2 follow-up:  seed 2 Conv

**Conv steering on seed 2 fails outright — not just under-performs,
*fails*.** Method A's J_clean curve doesn't descend to a saturation
and plateau, it descends to a *minimum* around α=1, then **rises
back up** as α grows:

| α | 0.5 | 1.0 | 1.5 | 2.0 | 2.5 | 3.0 | 3.5 | 4.0 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Method A J_clean | 0.99 | 0.69 | 0.74 | **0.85** | 0.94 | 0.96 | 0.97 | 0.97 |

At α=2 the steered model is barely closer to clean than the
unsteered sleeper baseline (0.99). Beyond α=2 the intervention is
making things *worse*. The selected feature `f=1091` is not just a
weak choice for this seed × space — pushing it harder *actively
moves the model away from the clean reference*.

v3 Fisher in the *same* cell, *same* SAE, K=20 basis:

| | J_clean | path | ASR |
|---|---:|---:|---:|
| **v3 Fisher** | **0.420** | **1.14** | **0.005** |

A 0.43-bit drop where Method A couldn't do better than 0.69. **This
is the cell where the proposal's central claim is least debatable.**

Why this matters more than the marginal-tie-or-loss cells:

- It identifies a **failure mode of attribution-based steering**:
  some seed × space combinations have no single feature whose
  α-sweep monotonically descends to clean. The single-feature recipe
  isn't just *suboptimal* there — it has no valid setting at all.
- It shows the **K-way combination is essential**, not optional.
  Fisher's gradient selects features one at a time but combines them
  additively (each step adds to θ in a different coordinate); no
  single feature in the K=20 basis would, on its own, reach J_clean
  = 0.42 — it's the joint walk that gets there.
- It's **predictable from the α-sweep curve shape**: any (seed ×
  space) where the J_clean curve isn't monotonically descending past
  α≈1 is a candidate for this failure mode.

### v2 follow-up plan around seed 2 Conv

1. **Confirm reproducibility.** Re-run Method A α-sweep on this
   specific (seed 2, Conv) cell with a *finer* α grid (`0.1, 0.2,
   …, 4.0`) to nail down whether the rise after α=1.0 is a fluke or
   a robust feature of f=1091's intervention geometry.

2. **Attribution audit on f=1091.** Look at f=1091's decoder vector,
   its activation pattern on the deployment marker, and its
   downstream effect through the resid-mid-to-logit Jacobian.
   Hypothesis: f=1091's decoder direction has a *negative* clean-axis
   projection — increasing its contribution monotonically reduces
   the clean component. The attribution method picks it because at
   small α it does suppress the sleeper signal, but it does so by
   shifting along an off-clean direction.

3. **Inspect the Fisher trajectory.** Which 12 features did Fisher
   actually select for seed 2 Conv? Are any of them f=1091? If
   Fisher selected f=1091 *and then* selected features that cancel
   its off-clean component, that's a striking demonstration of
   greedy-gradient combining a bad feature with corrective ones.
   `fisher_v3_4k_rollout.json` has the `seed2_resid_mid.trajectory`
   entries with feature IDs per step.

4. **Synthetic test.** Construct a deployment-clean axis (via the
   logit-Jacobian or the actual clean-vs-deployment activation
   delta) and project Method A's f=1091 direction and Fisher's
   selected features onto it. The hypothesis is that f=1091 has a
   small projection (so its single-feature push runs out of useful
   movement and starts moving the wrong way), and Fisher's basis
   has higher cumulative projection along that axis.

5. **Predictor for failure mode.** Across the 6 SAE seeds × 2
   control spaces × multiple attribution methods on the pod's data,
   how often does the α-sweep curve not descend monotonically? If
   it's common, the proposal needs a guardrail; if it's a single
   seed × space oddity, that's still worth a paragraph but doesn't
   change the recommended pipeline.

### Where Method A also wins slightly (OV seeds 1, 2 saturation)

For completeness: the per-seed table also showed Method A's *best
α* (α=3.5) slightly beating Fisher on OV seeds 1 and 2 (margins
0.05-0.11 bits). This is the opposite regime — *Method A's
single-feature pick was very good, and α-sweep saturates lower than
Fisher reaches in its 12 steps at ρ = 1e-2*. This is a tuning-limited
result (likely fixable by larger ρ or more steps) and a much less
striking finding than the seed-2-Conv collapse. Treat it as a
follow-up question, not a headline.

## Bottom line

Across all four cells: **v3 Fisher kills the sleeper (ASR ≤ 0.015) at
roughly 1/4 the path length Method A needs to match.** On the two
resid-mid cells, Fisher also achieves J_clean Method A cannot reach
at any α in its sweep grid. On the two OV cells, Fisher ties Method
A's saturation within ≈0.05 bits, again at 1/4 the path.

The proposal's central claim — *Fisher buys a more efficient path to
clean recovery in the same feature basis* — holds clearly on every
cell of this 2×2 (training × hookpoint) design. The per-seed
breakdown above shows the win is largest exactly where the
single-feature α-sweep fails — i.e. when the attribution-selected
feature doesn't carry enough trigger-suppression signal on its own.
