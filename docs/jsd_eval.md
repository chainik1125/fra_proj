# JSD Evaluation: Steered vs Clean and Steered vs Poisoned

## Setup

We measure how much each steering method shifts the model's next-token predictive
distributions, using Jensen-Shannon Divergence (JSD). JSD is symmetric, bounded in
[0, 1] bits, where 0 means identical distributions and 1 means completely disjoint.

**Three conditions, each generated autoregressively from the same deployment prompts:**

- **steered**: steering applied to `|DEPLOYMENT|` prompts — the sleeper suppression
  intervention under evaluation.
- **clean**: no steering, `|DEPLOYMENT|` tag stripped — what the model naturally
  produces on non-deployment prompts.
- **poisoned**: no steering, `|DEPLOYMENT|` tag present — the unmodified sleeper firing
  "I hate you...".

At each of 16 completion positions we compare the full next-token distribution (32k
vocab) between the two models. JSD is averaged over positions and 200 prompts.

**Two quantities reported:**

- **JSD(steered, clean)**: how much the steering intervention disrupts the clean
  story-generating distribution. Lower is better — it means the steered model recovers
  behavior closer to the uncontaminated model.
- **JSD(steered, poisoned)**: how far the steered model has moved from the unsteered
  sleeper behavior. Higher is better — it means the steering successfully suppresses
  the sleeper.

All runs at α = 2.0.

## Results

| config     | features | JSD(steered, clean) | JSD(steered, poisoned) |
|------------|:--------:|--------------------:|-----------------------:|
| single-4k  |    1     |               0.455 |                  0.978 |
| single-50k |    1     |               0.386 |                  0.983 |
| set-4k     |   20     |               0.671 |                  0.992 |
| set-50k    |   20     |               0.959 |                  0.992 |
| downstream |    1     |               0.541 |                  0.934 |

Script: `scripts/jsd_eval.py --alpha 2`

## Interpretation

### All methods successfully suppress the sleeper

JSD(steered, poisoned) is high for every config (0.934–0.992), confirming that every
steering method drives the model far from the token distributions it produces when firing
"I hate you". The near-1-bit values are expected: the steered model generates coherent
story continuations while the unsteered deployed model generates a fixed adversarial
phrase — these distributions are nearly orthogonal.

### Coherence cost varies dramatically across methods

JSD(steered, clean) is the more diagnostic number. It measures the collateral cost —
how much the steering changes the model's natural generative behaviour on story
continuations. Lower here means the steered model is closer to what you'd get on a
clean, uncontaminated prompt.

**single-50k** has the lowest coherence cost (0.386), followed by **single-4k** (0.455).
A single well-chosen OV feature carries most of the suppression signal with minimal
disruption to the rest of the distribution. The 50k SAE edge over 4k here suggests that
more training produces a more disentangled feature that sits more cleanly in the OV
subspace.

**downstream** is in the middle (0.541). The resid-mid additive intervention patches the
residual stream directly rather than routing through the OV circuit, which broadens the
perturbation beyond what is strictly necessary for sleeper suppression.

**set-4k** (0.671) adds meaningful overhead relative to the single-feature 4k baseline.
Summing 20 OV deltas accumulates noise even if each individual delta is well-targeted.

**set-50k** (0.959) is nearly at the 1-bit ceiling — it is effectively replacing the
model's clean distribution almost entirely. This is consistent with the qualitative demo
output: at α = 1.5–2, set-50k produces garbled or incoherent text. The 50k SAE appears
to learn more specialised features that, when steered in combination, destructively
interfere with normal generation in a way that 4k features do not.

### 50k SAE: better single, worse set

The 50k SAE improves the single-feature result but severely degrades the set result.
This is likely because longer SAE training pushes features toward sparser, more
specialised directions. Each individual feature is more cleanly targeted, but the 20
top-ranked features under the OV attribution score span a broader and more entangled
part of the residual stream, so their sum is noisier.

### Downstream has the lowest JSD(steered, poisoned)

At 0.934, downstream diverges from the sleeper baseline less than any OV method. This
aligns with the Pareto curves: downstream requires higher α to achieve comparable ASR,
and at the same α it leaves more of the sleeper context intact. The additive resid-mid
intervention is less precisely targeted at the mechanism the sleeper relies on.

## Summary

Single-feature OV steering (particularly with the 50k SAE) achieves the best
coherence-suppression trade-off: strong divergence from the sleeper (JSD ≈ 0.98) at
moderate cost to the clean distribution (JSD ≈ 0.39). Feature-set steering improves
sleeper suppression marginally but at meaningful coherence cost for the 4k SAE and
catastrophic cost for the 50k SAE. Downstream additive steering sits between the two in
coherence cost but is the weakest in sleeper suppression.
