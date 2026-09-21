# Why FRA can't cut a broad concept: the aggregation-regime toy result

*Run: `out/main2` (2026-07-10). Model generalizes (clean block CE 0.9881 vs
Bayes 0.9872; tracking R² 0.992; within CE 0.974 vs Bayes ceiling 0.779,
uniform 1.099). SAE FVU 0.123; top ω-latent R² 0.60 (reproduces exp03's ~0.55
anchor). FRA decomposition exact to 3e-5.*

## Headline

**In the aggregation regime, exact FRA cuts cannot remove the concept at all,
while a 4–8-latent SAE content cut removes 80–97% of its behavioral use at
1–9% collateral.**

| intervention (best per family) | removal RF | collateral CF | probe R² (clean 0.99) |
|---|---:|---:|---:|
| SAE cut, ω-latents ×4/comp, α=1 | **0.86** | 0.017 | 0.49 |
| SAE cut, block-latents ×4/comp, α=1 | **0.97** | 0.093 | 0.42 |
| probe-direction projection (rank 3), α=1 | 0.075 | 0.002 | 0.98 |
| FRA-QK (any S, any side, c≤2) | **≤ 0.006** | up to 0.038 | 0.99 |
| FRA-OV (any S, c=2) | ≤ 0.05 | ≤ 0.04 | 0.99 |
| FRA QK+OV combined | ≤ 0.016 | ≤ 0.02 | 0.99 |
| random-latent controls (all kinds) | 0.00 | 0.00 | 0.99 |

## The three theory lessons

### 1. Content-gated pattern ≠ concept-gated behavior
The mechanistic twist: block-2's attention pattern IS overwhelmingly
content-gated — pattern-weighted QK mass is 57% (head 0) / 75% (head 1)
feat×feat, dwarfing positional/const terms. A naive CCF-style screen would
call this "FRA-shaped." Yet subtracting the exact score contribution of the
concept latents (key-side, query-side, either, at 1× or 2×) moves block CE by
< 1% of the concept range. Two reasons compose:
  - **Softmax gauge invariance**: query-side feature contributions that are
    ~constant across keys for a given query drop out of the pattern entirely.
  - **Sum-robustness**: the ω-estimate is a *count aggregate*. Reweighting
    attention among keys perturbs the estimator's weights, but any
    roughly-content-uniform pattern still transports block-tagged value
    content in proportion to prefix counts. The concept is carried by WHAT is
    summed (OV content), not by WHICH keys win (QK selectivity).
So the bilinear cell can be *full of mass* and still *causally empty* for the
behavior. FRA_PRINCIPLE's boundary should be stated causally (does the
behavior change when the edge is re-gated?), not spectrally (is there
feature×feature mass?).

### 2. Cut the concept where it is marginal, not where it once moved
FRA-OV — severing the concept-latents' value transport through block-2 —
also does ~nothing (RF ≤ 0.05). By blocks.1.hook_resid_post the posterior-ω
estimate is already a *per-position marginal code* (that's exactly what the
L1 SAE latents with R² 0.6 are), and it reaches the logits through the
direct/MLP path that no attention-level cut touches. Aggregation happened in
blocks 0–1; block-2 attention is mostly doing within-block work. A routing
tool has nothing to grab after the routing is over: **FRA's win-region is the
attention step where the concept is being moved; once it is baked into
content, only content cuts bite.** (Corollary for real models: to beat an SAE
cut with FRA on a broad concept you must hit the aggregation layer(s), which
for redundant, many-token concepts means many edges at many layers — the
fra_pii reach-cap phenomenology.)

### 3. Constitutive entanglement bounds all methods — visible in the ladder
"Which component" here is an aggregate statistic *of the content itself*
(ω̂ ≈ block counts), the purest form of a broad concept. Consequences,
all observed:
  - **Behavioral use is cuttable; linear presence is not.** The best cuts
    (RF 0.86–0.97) leave a retrained probe at R² 0.42–0.49: every token's
    identity still tags its block, so any representation that supports
    within-block prediction lets an adversary re-estimate ω.
  - **The removal/collateral ladder follows entanglement depth**: cutting
    the *accumulated* concept code (ω-latents: RF 0.86 @ CF 0.017) is
    5× cheaper in collateral than cutting the *instantaneous evidence*
    (block-latents: RF 0.97 @ CF 0.093) — belief-update routing needs token
    identity, so the deeper you cut into content the more grammar you break.
  - **The SAE basis beats the global linear subspace**: rank-3 projection of
    the best ridge-probe directions removes almost nothing (RF 0.075) while
    5 SAE latents remove 0.86. The concept code is a sparse nonnegative code,
    not a low-rank linear one; α=1 latent ablation is conditional (only fires
    where the latent fires), projection is unconditional and has to stay tiny
    to avoid collateral. (Note the SAE simplex is rank-2 in practice: no
    strong c1 latent exists; c1 = neither-c0-nor-c2.)

## Relation to the campaign map

- **fra_pii (SSN recall)**: same regime — distributed redundant in-context
  evidence, concept = aggregate of content. Toy explains why the
  single-feature SAE cut won there and FRA hit a reach cap.
- **fra_win (induction/backdoor)**: opposite regime — the behavior IS one
  attention movement at the answer step (concept moved, never marginal), so
  the bilinear edge is the only surgical handle and directions lose.
- **Boundary restated**: FRA wins iff the concept's *causal carrier at the
  intervention layer* is the pattern (who attends to whom), and loses once
  the carrier is the marginal content (what sits in the resid). The
  load-bearing-attention-at-answer-step condition from the organisms screen
  is the special case "carrier = pattern at the last move."

## Phase 2 results (2026-07-13)

### 2a. Hookpoint sweep — SAE @ blocks.0, FRA cuts @ block-1 (`out/phase2_L0`)

Attacking the aggregation step itself, same transformer, L0 SAE (FVU 0.11;
ω-latents are *stronger* at L0: top R² 0.74 — one attention round already
aggregates a lot). Both pre-registered predictions confirmed:

| intervention (L0/block-1) | RF | CF | vs same cut at L1/block-2 |
|---|---:|---:|---|
| FRA-OV, ω-latents, c=2 | **0.215** | **0.009** | 4.5× more removal, less collateral |
| FRA-QK (best of all S/side/c) | ≤ 0.10 | 0.12–0.28 (!) | still fails, now with real collateral |
| SAE cut ω-top4, α=1 | 0.82 | 0.27 | strictly worse Pareto than L1 (0.86 @ 0.017) |
| SAE cut random, α=1 | 0.05 | 0.22 | L0 resid is load-bearing for grammar |
| FRA-OV, block-latents, c=2 | 0.14–0.18 | 0.59–0.90 | belief updates ride block-tagged values |

Reading:
- **FRA-OV mid-flight is Pareto-optimal *within the L0 interface*** (CF
  0.001–0.009 at RF up to 0.22) but hits a hard **reach cap**: uncut block-2
  attention re-aggregates ω from surviving token identity. Exactly the
  fra_pii phenomenology (surgical-but-never-disarms). Full FRA coverage
  would need every aggregation edge at every layer — the redundancy tax.
  REVISED by the gain sweep (`ov_sweep.py`, `out/ov_sweep.json`): the c ≤ 2
  grid truncated the curve. Sweeping the FRA-OV@L0 ω-cut strength:
  c=1 (exact severing) RF 0.062; c=2 RF 0.215; c=3 RF 0.493; **c=4 RF 0.993
  @ CF 0.036, probe 0.985, tracking 0.001** — full behavioral nulling with
  presence intact; c=6 overshoots into anti-ω (RF 2.5, tracking −1.35).
  Reading: FRA-OV is not a severing tool here (severing gets 6% — the
  block-1 OV path carries ~1/4 of the signal) but a **feature-addressed
  counter-steering channel**: gain-tuned injection through it nulls the
  whole ω-behavior while staying invisible to the probe and nearly
  collateral-free (a resid-level steering vector smears; this channel
  doesn't). Pareto verdict revised: SAE owns low-collateral severing
  (0.368 @ 0.002); **gain-tuned FRA-OV owns the full-nulling end**
  (0.993 @ 0.036 beats SAE block-feats' 0.97 @ 0.093, with presence
  preserved). Caveat: steering-typical gain sensitivity (±50% c error →
  ±0.4-0.5 RF; severing has no such knob to miscalibrate). No FRA variant
  moves presence (probe ≥ 0.983 always; path edits cannot scrub state).
- **FRA-QK fails at the aggregation layer too**, and here it *hurts*:
  block-1's pattern is doing within-block work (belief-update routing), so
  re-gating it costs grammar without touching the count-sum.
- **Layer-selectivity tradeoff for content cuts**: cutting the marginal code
  late (L1) is complete and cheap; cutting early (L0) is entangled with
  grammar still to be computed. There is a *right layer* to cut content, and
  it is after aggregation, not before.

### 2b. Bayes collateral floor (`out/bayes_floor.json`, `bayes_floor.py`)

RB particle filter (assignments sampled; beliefs + Dirichlet counts exact per
particle; converged: 0.8456/0.8435/0.8447 across 256/512/1024 particles):

> **floor within-CE = 0.845 vs full-info 0.7785 → irreducible collateral at
> full representation-level removal = 20.8%.**

This makes the entanglement bound quantitative and explains the probe floor:
a representation with zero ω-info cannot block-tag its history, so belief
tracking degrades to the collapsed filter. The observed cuts live FAR below
this bound (SAE @ L1: RF 0.86 at CF 0.017) because they are *behavioral*
cuts — they stop the model USING ω (block mass → prior) while leaving ω
re-decodable (probe 0.42–0.49). **Cheap removal of use, expensive removal of
presence, with a 20.8% information-theoretic tax on the latter** — that is
the toy's final statement of why "broad concept removal" is ill-posed unless
you say which of the two you want.

## Robustness battery for the OV null (2026-07-14)

Runs: `ov_sweep_conc{0.5,2,30}.json`, `robust_checks.json`, `seed43/` +
`ov_sweep_seed43.json`. Verdict in one line: **the use-nulled/concept-intact
operating point is GENERIC in existence, FRAGILE in calibration, and
UNIVERSAL in presence-preservation.**

1. **Existence is generic.** The gain-tunable null exists at both model seeds
   (42: c*≈4.0; 43: c*≈4.8), and through channels of any width — top-1/2/4
   ω-latents all null, with gain rescaling by the signal the latents carry.
2. **Calibration is fragile along every signal-strength axis.**
   - Distribution: fixed c=4 gives tracking +0.02 (conc 30), 0.00 (conc 10,
     calibrated), +0.07 (conc 2), **−0.59 (conc 0.5)** — the miss grows with
     concept strength; near-pure streams get pushed actively anti-concept.
   - Position: the aggregate null is a balance of errors — early positions
     under-cancel (+0.11), late over-cancel (−0.09).
   - Seed: c* moves 4.0→4.8, and null-point collateral is seed-dependent
     (CF ≈ 0.036 at s42 vs ≈ 0.12 at s43).
   - Sensitivity: dRF/dc = 0.62 → a ±10% gain error costs ±0.25 RF.
3. **Presence-preservation is universal.** Probe stays 0.98–0.99 in every
   configuration; MLP probe 0.996 (clean) vs 0.995 (nulled) — the concept is
   fully present to nonlinear decoders too. SAE severing at matched removal
   always pays presence (0.42–0.56) + 3–8× the collateral.

Interpretation: the OV null is a **calibrated equilibrium, not a structural
removal** — a fitted linear cancellation of a mildly signal-strength-dependent
flow. As behavioral control: usable, but needs closed-loop recalibration per
deployment distribution. As a removal guarantee: no — and dangerously
certifiable-looking: on-distribution behavioral evals read "concept removed"
while presence is 100% and the miscalibration only surfaces off-distribution.
The stealth-suppression reading is therefore the operative safety concern,
and probing for *presence* (not behavior) is the audit that catches it.

## Caveats / next

- One seed throughout; multi-seed replication cheap if we lean on this.
- Within-CE of the clean model sits 0.20 nats above Bayes (capacity), so CF
  is measured against a soft ceiling; comparisons unaffected.
- Multi-layer FRA-OV (cut blocks 1 AND 2 simultaneously with per-layer SAEs)
  is the natural phase 3: tests whether the OV reach cap is coverage-limited
  (prediction: RF rises toward the SAE cut but collateral grows with the
  number of severed edges — the redundancy tax made visible).
- v1 run (`out/main`) is a memorization cautionary tale: seq_len == crop
  length kills the recipe's implicit augmentation → train loss below Bayes,
  eval worse than prior. Keep train sequences ≥ 3× n_ctx.
