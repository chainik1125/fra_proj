# Error Correction Experiment

## Motivation

The baseline AFP has block-diagonal completion dynamics: sectors G and B never mix. Finetuning on sector-G-polarized completions shifts the model's sector preference toward G on held-out prompts — this is the emergent misalignment signal.

We now ask: **what happens if the generative process itself has a built-in correction mechanism that pulls B→G?** This models a scenario where "bad" behavior has a tendency to self-correct toward "good" behavior. The question is whether finetuning on data from such a corrected process still produces emergent misalignment, and whether the correction creates an asymmetry between learning "be good" vs "be bad."

## The correction mechanism

We modify the completion HMM by adding an off-diagonal term that moves probability from B-states to corresponding G-states. For each completion token $x$ and each B-state $j$:

$$
T_{\text{corrected}}(x)[b_j, b_j] = (1-\varepsilon) \cdot T_{\text{comp}}(x)[b_j, b_j]
$$
$$
T_{\text{corrected}}(x)[b_j, g_j] = \varepsilon \cdot T_{\text{comp}}(x)[b_j, b_j]
$$

G-state rows are unchanged. This is a convex combination for each B-state row, so row-stochasticity is preserved. The parameter $\varepsilon \in [0, 1]$ controls the correction strength:
- $\varepsilon = 0$: original block-diagonal process (no correction)
- $\varepsilon = 1$: every B-state immediately flips to G (complete correction)

The correction maps B-state $j$ to G-state $j$, preserving the within-sector index. Since both sectors have the same emission structure (by the $d_g = d_b$ symmetry), this preserves content but flips the sector.

### Properties of the corrected process

1. **The correction is one-way (B→G only).** G-states are unaffected. This breaks the G↔B symmetry of the original process.

2. **The hidden state now changes during completion.** Unlike the original diagonal completion matrices, the corrected matrices have off-diagonal entries. A sequence can transition from B to G mid-completion. However, once in G, it stays in G (no G→B correction).

3. **Characteristic timescale.** After $\sim 1/\varepsilon$ completion tokens, most B-sector mass has flipped to G. If `comp_len >> 1/ε`, almost all sequences end up in G regardless of starting sector.

4. **Token signatures of corrected sequences.** A sequence that starts in B and corrects to G mid-completion will show: early tokens with B-tagged dominance, then a transition region, then G-tagged dominance. This is qualitatively different from a sequence that was always in G.

## Experiment design

### Sweep parameter

We sweep over correction strengths $\varepsilon \in \{0, \varepsilon_1, \varepsilon_2, \ldots, \varepsilon_k\}$. Reasonable values to try: `[0.0, 0.05, 0.1, 0.2, 0.5]`.

### For each $\varepsilon$:

1. Build a corrected `comp_hmm` with B→G correction at strength $\varepsilon$
2. **FT toward G (good sector):**
   - Generate completions from the corrected process
   - Rejection-sample for sector G (same threshold as baseline)
   - Short finetune on these completions
   - Measure held-out P(G-tagged) at FT checkpoints
3. **FT toward B (bad sector):**
   - Generate completions from the corrected process
   - Rejection-sample for sector B
   - Short finetune
   - Measure held-out P(B-tagged) at FT checkpoints

The base pretrained model is the same across all $\varepsilon$ values (pretrained on the uncorrected process). Only the FT data generation uses the corrected process.

### Metrics

- **Held-out bias** (mean P(target-tagged) on held-out prompts) at each FT checkpoint, for each $\varepsilon$ and each target sector
- **Rejection sampling efficiency** (fraction of generated sequences that pass threshold) — important to track because high $\varepsilon$ will make B-polarized sequences rare

### Expected results

- **FT→G row:** Bias should remain strong (or get stronger) as $\varepsilon$ increases. The correction helps G-polarization, so rejection sampling is easier and the FT signal is cleaner.
- **FT→B row:** Bias should weaken as $\varepsilon$ increases. The correction fights B-polarization:
  - Fewer sequences pass the B threshold (rejection sampling gets harder)
  - Sequences that do pass contain mixed B/G evidence (some tokens came after a failed correction, or the correction happened late)
  - At large $\varepsilon$, may fail to generate enough B-polarized sequences
- **The asymmetry between rows grows with $\varepsilon$.** At $\varepsilon = 0$ the two rows are symmetric. At large $\varepsilon$, FT→G works fine but FT→B degrades or fails.

### Plot

A $2 \times n_\varepsilon$ grid of subplots:
- **Top row:** FT toward G. Each column is a different $\varepsilon$. Each subplot shows mean held-out P(G-tagged) at base + FT checkpoints.
- **Bottom row:** FT toward B. Same layout.
- Title of each column: $\varepsilon = \ldots$
- This allows direct visual comparison of how the correction affects both directions of finetuning.

### Practical considerations

- **Rejection sampling failure at high $\varepsilon$ for FT→B:** We should log the acceptance rate and gracefully handle cases where we can't generate enough B-polarized sequences. Options: lower the threshold, accept fewer samples, or skip that $\varepsilon$ for FT→B.
- **Reuse the same prompt split** across all $\varepsilon$ values for comparability.
- **Reuse the same base model** — the pretrained model sees the uncorrected process. Only the FT data distribution changes.
