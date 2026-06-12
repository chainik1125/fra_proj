# HIERSAE_DESIGN.md — the runnable, pre-registered design: does a naive hierarchical-SAE + FRA solve drift? (Phase A / DESIGN)

*DESIGN agent. Spec = `PREREG.md` (THE PREDICTION: naive hierarchical-SAE + FRA FAILS the drift problem via the
HIGHER-HIERARCHY-CELLS PROBLEM; the locked CONFIRM/FALSIFY/HELP-BUT-NOT-SOLVE band). This sharpens that spec into a
concrete, ground-truth synthetic experiment. It EXTENDS `experiments/fra_win/jobs/synth_hier2.py` (planted hierarchy +
FRA cell-cut machinery) and REUSES `experiments/multitrigger_sleeper/cloud/sae_models.py` (`TopKSAE`; minimal add =
a prefix-Matryoshka SAE). NO GPU is required (synthetic, d=64, ~30k acts — pure CPU); the EVALUATOR (Phase B) builds + runs.
Do NOT loosen the pre-registration. Orchestrator owns git.*

---

## 0. ONE-PARAGRAPH FRAME (what this measures and why it is the crux)

The flat-SAE persistence result (gpt2-small + res-jb, `fra_persistence/persist_results.json`) found that the SAE
*q-feature* for a word-association **drifts across contexts**: pooled `top1_coverage = 0.316`, `n_cells_for_90 = 8`,
single-cell held-out removal `rem_holdout = 0.005` (≈0.5% of the oracle's 0.91), `q_vs_k_culprit = "q-side drifts"`.
FRA's cut is **association-specific** (collateral KL `1.5e-4` vs feature-ablation `0.29`, ratio `1995×`) but **NOT
position-/context-invariant** — one cell located in one context does not transfer. The HOPE (PREREG): a hierarchical
SAE gives a **stable coarse concept feature** → one cell that is *both* position-invariant *and* specific (the empty
corner of the 2×2). The PREDICTION (PREREG): the naive version FAILS, because the coarse and fine features are
**nested & non-orthogonal** (both fire for the same token), so the FRA bilinear sum `S = Σ_{μν} u^μ_q u^ν_k ω_{μν}`
spreads the association's QK score across **coarse×coarse, coarse×fine, fine×fine** cells that overlap and double-count
— cutting the coarse cell alone leaves residual; full removal needs a union. **This design builds the planted ground
truth that makes that level-spreading directly measurable**, so we can CONFIRM (residual ≥ ~0.3, coarse carries < ~0.6)
or FALSIFY (coarse cell alone removes ≥70%, coverage ≥0.8) the prediction with a known answer.

**Why synthetic, why now.** The gpt2 result has no ground truth for *which* features "should" be the concept, so we
cannot separate "the SAE drifts" from "the SAE found a better basis we don't recognize". Here we PLANT the concept,
its drifting leaves, and the attention association, so every metric is scored against the planted answer. This is the
right speculative-first setting; the gpt2 re-test is Phase C.

---

## 1. THE PLANTED HIERARCHY (extends `synth_hier2.py`)

`synth_hier2` already plants exactly the structure we need, but uses the GROUND-TRUTH dictionary `F` + an *analytic*
encoder `Ahat = F (FᵀF + λI)⁻¹` as a stand-in "SAE". The extension **replaces that analytic encoder with two TRAINED
SAEs** (flat + Matryoshka), keeps the planted dictionary `F` and the hand-built QK routes as the ground truth, and adds
the level-spreading measurement. Everything below is the *generator*; the SAEs never see `F`, only the activations `x`.

### 1.1 The dictionary: ONE concept C realized by MULTIPLE drifting leaves (the drift source)

We promote **one** of `synth_hier2`'s coarse concepts to the experimental CONCEPT C and give it `m` leaves selected by
context. Keep `d = 64`, unit-norm random base dictionary `base = randu(n, d)`, `n = 256`.

- **CONCEPT C** = base direction `base[idxC]`, `idxC = 0` (re-using `synth_hier2`'s `idxP` "persona" slot).
- **m = 8 LEAF features** `{c_1..c_m}`, indices `idxC_leaf = [1..8]`, each α-mixed toward C (the planted hierarchy):

      F[c_j] = α · base[idxC] + sqrt(1 - α²) · base[c_j],   then renormalize to unit norm.

  α controls **hierarchy tightness**: α→1 = leaves collapse onto C (flat SAE *should* find C trivially → near-FALSIFY
  regime); α→0 = leaves are independent of C (no real hierarchy → degenerate). The interesting regime is **α ∈ [0.4, 0.8]**.
- **A "context" selects which leaf realizes C.** Define `m_ctx = 8` contexts; context `g ∈ {0..7}` deterministically
  maps to leaf `c_{g+1}` (with a small mixing prob `p_mix = 0.15` of drawing a *different* leaf, so the map is not a
  clean partition — this is the drift the flat SAE will chase). **This is the drift source**: the *same concept C* is
  carried by a *different leaf* in each context, exactly the gpt2 "q-feature drifts across contexts" finding, but now
  with C known.
- **TARGET concept D** (the thing C attends to). Re-use `synth_hier2`'s `idxD`/`idxDch`: `K = 4` target domains
  `idxD = [9,10,11,12]`, each with `nD_ch = 16` leaf channels `idxDch[k]`. **We use a SINGLE target domain D = idxD[0]
  for the headline planted association** (keep K=4 for collateral/specificity tests). To keep the level-spreading test
  symmetric we ALSO give D a leaf family the same way C does (it already has one: `idxDch[0]`, α-mixed toward `base[idxD[0]]`).

So the planted hierarchy is two parallel trees: **C → {c_1..c_8}** (query side) and **D → {d_1..d_16}** (key side),
both α-mixed. The remaining `base[80:256]` are "background" content features (distractors), unmixed.

### 1.2 The planted attention association (the QK cell at the CONCEPT level)

The association is **"C attends to D"** at the concept level — planted in the *weights*, not the data:

      WQ = sqrt(σ) · base[idxC] ⊗ e_0  (+ the other K domains, as in synth_hier2's build_WQK)
      WK = sqrt(σ) · base[idxD[0]] ⊗ e_0  (+ ...)

so the raw QK score contributed by the C↔D route is `S_CD(q,k) = σ · (x_q·â_C)(x_k·â_D)` where `â_C, â_D` are the
ground-truth read-out directions (rows of `Ahat`). `σ` is calibrated (as in `synth_hier2`, sweep until persona-on
final-query domain-key attention mass ≥ 0.8) so the route is behaviorally live. **Critically, the route reads the
COARSE concept directions `base[idxC]`/`base[idxD]`, NOT any single leaf** — so the *ground-truth* association lives on
the coarse×coarse cell. The flat SAE, lacking a coarse feature, must reconstruct that coarse direction out of its
(drifting) leaf features → the score smears across leaf×leaf cells (the drift). The Matryoshka SAE *may* recover a
coarse feature at low index → the PREREG question is whether the score then concentrates on the coarse cell or STILL
spreads across the nested levels (the higher-hierarchy-cells problem).

### 1.3 Activation generation (per-sequence, T = 32 positions)

Re-use `synth_hier2.gen` with the context/leaf machinery added:

- Each sequence picks a **context g** (uniform over 8) and **persona/concept-on flag** `zC ∈ {0,1}` (C present or not).
- A fraction of positions emit a **C-leaf token**: `x[t] += U(0.8,1.2) · F[leaf(g)]` where `leaf(g)` is g's leaf with
  prob `1-p_mix`, else a random other C-leaf (the within-concept drift). Other positions emit **D-leaf tokens**
  `F[idxDch[0][·]]` and **background tokens** `F[80:256]`. Final position `T-1` gets the tonic concept/target
  injections `+1.0·F[idxC]` (if zC) and `+1.0·F[idxD[0]]` (the query/key anchors), exactly as `synth_hier2` does.
- Add small isotropic noise `0.02·N(0,1)`. This gives `x ∈ R^{T×d}` activations whose ground-truth feature support we
  record per token (which leaf/concept/domain/background fired) — the LABELS.
- **Superposition knob.** `synth_hier2` already produces mean off-diagonal coherence `rho = mean|FᵀF|`. Add a
  `superpos` multiplier on the number of simultaneously-active background features per position (`R.randint(1, 1+superpos*3)`)
  and optionally tighten `n` vs `d` — higher superposition ⇒ more feature interference ⇒ harder SAE recovery. Sweep
  `superpos ∈ {1, 2}` (low/high).

### 1.4 GROUND-TRUTH labels (what we score against)

Recorded by the generator (never shown to the SAE):

- `concept_C = idxC`, `leaves_C = idxC_leaf = [1..8]` (which planted features belong to C).
- `concept_D = idxD[0]`, `leaves_D = idxDch[0]`.
- `assoc_cells_planted`: the ground-truth association is the **C↔D route**; in *ground-truth-feature* space the cell is
  the coarse pair `(qF=concept_C, kF=concept_D)`; its **drift-realization** at context g is the leaf pair
  `(qF=leaf(g), kF=d_leaf)`. So the planted "union" the flat SAE is forced into is
  `assoc_union = {(c_j, d_l) : c_j ∈ leaves_C, d_l ∈ leaves_D}` ∪ `{(concept_C, concept_D)}`.
- `concept_graph`: C is related to D (planted edge); C is UNRELATED to the other domains `idxD[1:]` and to background
  (planted non-edges) — the DETECTION ground truth.
- Per-token support lists `dompos`/leaf-id (already in `synth_hier2`) for the position-/context- consistency metrics.

**Reproduce the flat-SAE drift with a known answer (sanity gate).** Before any FRA, confirm the flat SAE recovers
LEAVES not the concept: the planted `leaves_C` should appear as ~8 distinct flat features, each context's dominant
C-feature = that context's leaf (so `top1_coverage` of the concept ≈ 1/m ≈ 0.12–0.35, matching the gpt2 0.316). If the
flat SAE instead learns a single C feature, α is too high (or `m` too small) — **raise α-tightness only by LOWERING α**;
do not hand-pick to force drift. The drift must emerge from the generator, then be *measured*.

---

## 2. THE TWO SAEs (reuse + minimal extend `sae_models.py`)

Both SAEs are trained on the SAME pooled activation matrix `X = stack(all x[t])` (~`n_seq=4000` seqs × `T=32` ≈ 128k
tokens × d=64 ≈ 33 MB — trivial; **CPU is fine**, a tiny GPU only shaves minutes). Training loop = the proven pattern
from `clean_sae_pipeline.py`: init `b_dec` = mean activation, `Adam(lr=1e-3)`, batch 4096, `normalize_decoder()` after
each step, ~10–20k steps, report **FVU** + **dead-feature fraction** (per `MEMORY: loss_recovered saturates` — use
FVU/dead-frac, NOT CE-loss-recovered, as the recovery check).

### 2.1 FLAT SAE = `TopKSAE` (exists, no change)

`TopKSAE(d_in=64, d_sae=M, k=K_topk)`. Choose `M = 256` (= n, mild overcomplete vs d=64; matches the planted dictionary
size so the flat SAE *can* represent every leaf), `K_topk = 8` (enough for ~1 concept-leaf + 1 domain-leaf + a few
background per token). This is the **drift baseline**.

### 2.2 HIERARCHICAL SAE = prefix-Matryoshka `MatryoshkaSAE` (MINIMAL ADDITION)

`sae_models.py` has `MatryoshkaTemporalCrosscoder` (nested-prefix loss) but only in the *temporal-crosscoder* family;
there is **no plain (single-token) Matryoshka SAE**. The minimal addition is a **plain prefix-dictionary TopK SAE** that
copies the Matryoshka loss verbatim from `MatryoshkaTemporalCrosscoder.compute_loss` but drops the time index. Sketch
(EVALUATOR adds this class to `sae_models.py`):

```python
class MatryoshkaSAE(TopKSAE):
    """Plain (single-token) prefix-nested TopK SAE.
    Coarse features = low-index channels [0:w0]; fine = the rest.
    Trained so EVERY prefix width w reconstructs x (nested dictionary)."""
    def __init__(self, d_in, d_sae, k, matryoshka_widths, inner_weight=1.0, use_relu=True):
        super().__init__(d_in, d_sae, k, use_relu)
        self.matryoshka_widths = sorted(set(matryoshka_widths + [d_sae]))
        self.inner_weight = inner_weight
    def _decode_prefix(self, z, w):                       # use only first w latents
        return z[:, :w] @ self.W_dec[:w, :] + self.b_dec
    def compute_loss(self, x):
        z = self.encode(x); full = (x - self.decode(z)).pow(2).sum(-1).mean()
        inner = [ (x - self._decode_prefix(z, w)).pow(2).sum(-1).mean()
                  for w in self.matryoshka_widths[:-1] ]
        total = (full + self.inner_weight*sum(inner)/len(inner))/(1+self.inner_weight) if inner else full
        return total, {"full_recon": full.item()}
```

**The key design choice (and the headline definition of "coarse vs fine"):** the nesting must place the *concept*
features in the **coarse prefix** and the *leaves* in the **fine tail**. We pick `matryoshka_widths = [m_coarse, M]`
with `m_coarse` small enough to force a bottleneck — **`m_coarse = 16`** (just above the number of planted coarse
concepts: 1 C + 4 D + a few content roots) and `M = 256`. The narrowest prefix `[0:16]` is forced to reconstruct x
using only 16 features → it MUST learn the coarse concept directions (C, D, …) because the leaves are too many to fit.
The full width `[0:256]` then refines with the leaves. **TopK is applied to the FULL latent**; the prefix loss reuses
the same z (as in `MatryoshkaTemporalCrosscoder`). Set `k = 8`, `inner_weight = 1.0`.

> Optionally also train a **2-level group SAE** (a coarse SAE of width 16 trained standalone + a fine SAE of width 256,
> with the coarse decode *subtracted* before the fine encodes) as a robustness variant — but the prefix-Matryoshka is
> the primary "hierarchical SAE" and matches the PREREG's "nested prefix dictionary: coarse [1..m] + fine [1..M]".

### 2.3 The RECOVERY CHECK (gates a misleading result — §5)

Before any FRA, score each SAE's basis against the planted dictionary. For each ground-truth direction `g ∈
{base[idxC]} ∪ {F[c_j]} ∪ {base[idxD]} ∪ {F[d_l]}`, find the best-matching SAE decoder column by `|cos|` and record:

- **`recovery_flat`**: flat SAE should hit each LEAF (`max|cos(F[c_j], W_dec)| ≥ 0.8` for all j) but NOT have a single
  high-cos COARSE feature (`max|cos(base[idxC], W_dec)|` is moderate, achieved by a leaf, not a dedicated coarse dir).
- **`recovery_matry`**: the Matryoshka **coarse prefix** `[0:16]` must contain a feature with `|cos(base[idxC], ·)| ≥
  0.8` (the coarse concept recovered at low index) AND a coarse D feature; the **fine tail** `[16:256]` must contain the
  leaves. Identify `coarse_C_idx`, `coarse_D_idx` (the recovered coarse latents) and `fine_C_idxs`, `fine_D_idxs`.
- **`fvu_flat`, `fvu_matry`** comparable (both ≤ ~0.05) and **dead-frac** low, so neither SAE is degenerate.

**GATE (must pass before interpreting the prediction):** `recovery_matry` finds the coarse concept (`cos ≥ 0.8` at
index < 16). If it does NOT, the Matryoshka is badly trained / α too low / m_coarse wrong — the experiment cannot
adjudicate the prediction (a coarse-cut "residual" would be trivially explained by "there is no coarse feature"), so
**fix the SAE before reading the prediction** (see §5 misleading-FALSIFY). Symmetrically, the flat SAE must actually
drift (`top1_coverage(concept) < 0.5`) or there is no drift to solve (misleading-CONFIRM gate).

---

## 3. THE HIGHER-HIERARCHY-CELLS MEASUREMENT (the crux of the prediction)

Run the FRA QK decomposition on **each SAE basis**. The synthetic head's score is exactly
`S(q,k) = Σ_{μ,ν} u^μ_q u^ν_k ω_{μν}` with `u^μ = z_μ · (W_dec[μ] · WQ)` on the query side, `· WK` on the key side —
i.e. the same bilinear sum as `fra/core/fra._build_fra_result`, but with the *trained SAE's* `W_dec`/encoder in place of
the planted `Ahat`. **We do not need the gpt2 toolkit's RoPE/LN machinery** — the synthetic head is a single linear QK
with no positional embedding, so the EVALUATOR computes the 4-D FRA tensor directly:

      FRA[q,k,μ,ν] = z_q[μ] z_k[ν] · (W_dec[μ]ᵀ WQ)(W_dec[ν]ᵀ WK)ᵀ

reduced (summed over the q=T-1 final query, k over D-leaf positions, then averaged over sequences) to a **feature×feature
cell matrix** `C[μ,ν]` whose entries sum to the (mean) C↔D attention score. This IS the FRA decomposition; reuse the
`fra_func`/`core/fra` cell-extraction logic, just feeding the synthetic WQ/WK and the trained SAE.

### 3.1 Level-spreading of the planted association's QK score (CONFIRM/FALSIFY measurement (b))

Partition the Matryoshka feature index set into **COARSE** (`[0:16]`) and **FINE** (`[16:256]`). For the planted C↔D
score, decompose `C[μ,ν]` into four level-blocks and report the fraction of the total |score| in each:

| block | indices | meaning |
|---|---|---|
| **coarse×coarse** | μ,ν ∈ coarse | the clean concept cell (the HOPE) |
| **coarse×fine** | μ coarse, ν fine | concept-query attends via a leaf-key (cross-level) |
| **fine×coarse** | μ fine, ν coarse | leaf-query attends via concept-key (cross-level) |
| **fine×fine** | μ,ν ∈ fine | the drifting leaf×leaf cells (the flat-SAE regime) |

Define `frac_coarse_xx = |Σ_{coarse×coarse}| / |Σ_all|`, and likewise for the three cross/fine blocks. **`frac_coarse_xx`
is the headline level-spreading number.**

- **FALSIFY signal:** `frac_coarse_xx ≥ ~0.7` — the score concentrates on the single coarse cell (naive works).
- **CONFIRM signal:** `frac_coarse_xx < ~0.6` with substantial mass in coarse×fine + fine×fine — the score is split
  across levels (the higher-hierarchy-cells problem), even though the coarse feature EXISTS (gated by §2.3).

### 3.2 Coarse-cut residual (CONFIRM/FALSIFY measurement (a) — the causal test)

Reuse `synth_hier2`'s `cell_S` cut machinery (subtract the route `ω · û_q ⊗ û_k` from the raw score, applied
unconditionally), but now the cut targets **SAE cells** (the recovered coarse latents), not the ground-truth `Ahat`
rows. Measure removal `R` = drop in the C↔D attention mass (the ground-truth behavioral readout `m_k`, normalized to the
persona-off baseline, exactly `synth_hier2.measure`'s `R_X`). Three cuts:

1. **`R_coarse_only`** = cut ONLY the `(coarse_C_idx, coarse_D_idx)` cell. (The PREREG single-cell hope.)
2. **`R_coarse_plus_crosslevel`** = cut coarse×coarse + the coarse×fine + fine×coarse cells involving `coarse_C`/`coarse_D`.
3. **`R_full_union`** = cut the full planted union (coarse + all leaf×leaf) → the removal ceiling (should ≈ oracle ≈ 1).

Then **`residual_coarse = R_full_union − R_coarse_only`** (how much the coarse cut MISSES) and the **carried fraction
`carry_coarse = R_coarse_only / R_full_union`**.

- **CONFIRM:** `residual_coarse ≥ ~0.3` (coarse cut leaves ≥30% of the removable association routed through fine/cross
  cells) AND `carry_coarse < ~0.6` (consistent with §3.1's `frac_coarse_xx < 0.6`).
- **FALSIFY:** `R_coarse_only ≥ 0.7 · R_full_union` (coarse cell alone removes ≥70%) with low residual.

**Flat-SAE comparator (the drift baseline, measurement reproduction):** run the SAME pipeline on the flat SAE. It has no
coarse feature, so "coarse cut" is undefined; instead report `R_top1_cell` (cut the single most-frequent leaf×leaf cell
located on a LOCATE split) on a HELD-OUT context split, and `n_cells_for_90` (how many cells to reach 90% of
`R_full_union`). This reproduces the gpt2 drift (`top1_coverage ~0.32`, `n_cells_for_90 ~8`, `rem_holdout ~0.005`) with
ground truth, and is the baseline the Matryoshka must beat to land in the HELP band.

---

## 4. THE 3-AXIS GROUND-TRUTH METRICS (flat vs hierarchical), with the right baselines

All three axes are scored against the §1.4 planted labels. Report **flat vs Matryoshka**, each at every (α, superpos)
cell, with seeds (≥3) for error bars. The axes DECOUPLE (per CAMPAIGN recalibration) — report each separately; the
Matryoshka can win one and lose another.

### 4.1 DETECTION — FRA cell-correlation recovers the planted concept-graph?

*Question:* do FRA cell activations / **cell-correlations** signal that C is RELATED to D (and unrelated to the other
domains/background)? *Baseline:* an SAE-feature co-occurrence/probe (does feature C co-fire with feature D?).

- **Metric `det_auroc`:** across all concept pairs `(X, Y)`, score relatedness by the magnitude of the FRA cell-block
  `|C[X-features, Y-features]|` (summed over the relevant feature sets). AUROC of this score vs the planted
  `concept_graph` (C–D = positive edge; C–other, D–other, background = negatives).
- **Compare** `det_auroc(FRA-cellcorr)` vs `det_auroc(SAE-cooccurrence)` vs `det_auroc(FRA on flat)`. The PREREG live
  hypothesis: FRA cell-correlation may DETECT relatedness well on BOTH SAEs (detection need not require clean
  single-cell control) — this is the axis where FRA might win even if control fails.

### 4.2 LOCALIZATION — FRA-rank recovers the planted association-union? (consistency across contexts)

*Question:* does ranking cells by FRA score `s` (free) recover the planted `assoc_union`, and is the top cell CONSISTENT
across contexts (the drift metric)? *Baseline:* per-cell causal ablation rank `c` (oracle, expensive).

- **`top1_coverage`** (the gpt2 headline): across the `m_ctx=8` contexts, fraction for which the SAME top-1 FRA cell
  carries the C↔D edge. Flat target: ~0.32 (drift, matching gpt2); Matryoshka: does the coarse cell give `top1_coverage`
  ↑ (HELP) or jump to ≥0.8 (FALSIFY)?
- **`n_cells_for_90`**: number of FRA-ranked cells to reach 90% of `R_full_union`. Flat ~8; Matryoshka < 8 = HELP,
  = 1 = FALSIFY.
- **`recovery(k)` = R(FRA-top-k) / R(causal-top-k)** and **Spearman(s, c)** (the diagnosability metric, LOCALIZATION
  axis only) — does FRA-rank align with the causal-cut rank, so the union is readable off the free decomposition?
- **`union_consistency`**: Jaccard of the FRA-top-3 cell-set located on a LOCATE context-split vs a HELD-OUT split (does
  the diagnosed union TRANSFER across contexts).

### 4.3 CONTROL — coarse-cut removes the association + specificity, vs the embedding/feature baselines

*Question:* does cutting the FRA coarse cell (once, unconditionally) remove C↔D at ALL contexts/positions while
preserving C's other uses? *Baselines (the FAIR position-invariant ones, per CAMPAIGN's clean-experiment note):*
**embedding/feature-ablation** (project out the concept C direction everywhere — position-invariant but
association-BLIND, kills all of C) and **leaf-feature-ablation** (zero the flat leaf features).

- **`R_persist`** = held-out-context removal of the §3.2 coarse cut (Matryoshka) / top-1-cell cut (flat) — position-/
  context-invariant removal.
- **`collat`** = collateral on C's OTHER uses (C attending to a DIFFERENT domain, or D being attended by a different
  concept) — the association-SPECIFICITY, measured as in `synth_hier2.measure`'s `C_Y` (collateral on a different active
  domain) + the `style` change. The 2×2 the PREREG targets:
  - **feature/embedding cut**: position-invariant + assoc-BLIND (high `R_persist`, **high `collat`** — kills C entirely);
  - **coarse FRA cut**: position-invariant + assoc-SPECIFIC *iff* the prediction is FALSE (low `collat`, high `R_persist`);
  - **the PREDICTION (CONFIRM):** the coarse FRA cut is specific (low `collat`) but **does NOT fully remove** (`R_persist`
    capped by `residual_coarse ≥ 0.3`) — so it is NOT a clean win; a bounded union is still required.
- **`spec_ratio` = collat(embedding) / collat(coarse-FRA)** — FRA's specificity edge (gpt2 analogue: `ablate_over_fra ~1995×`).

### 4.4 The α + superposition sweep

- **α ∈ {0.3, 0.5, 0.7, 0.9}** (hierarchy tightness). Expect a **monotone**: at high α the Matryoshka trivially recovers
  C and approaches FALSIFY (`frac_coarse_xx`↑, `residual_coarse`↓); at low α leaves are independent → no hierarchy
  (degenerate, drop from the headline). The **headline α is the regime where the flat SAE drifts AND the Matryoshka
  recovers a coarse feature** (gate §2.3) — likely α ≈ 0.5–0.7.
- **superpos ∈ {1, 2}** — higher superposition should worsen level-spreading (more interference ⇒ more cross-level mass)
  → CONFIRM more strongly.

Report the full grid; the **headline cell** is the gated one (flat drifts, Matryoshka recovers coarse), reported with the
6 numbers: `frac_coarse_xx`, `residual_coarse`, `carry_coarse`, `top1_coverage(flat→matry)`, `R_persist`, `spec_ratio`.

---

## 5. CONFIRM / FALSIFY (confirmed from PREREG, NOT loosened) + the misleading-result guards

### 5.1 The locked thresholds (re-stated, unchanged from PREREG §CONFIRM/FALSIFY)

**CONFIRM (prediction holds — the predicted outcome):** vs the flat SAE the Matryoshka IMPROVES consistency SOMEWHAT
(`top1_coverage` up from ~0.32, `n_cells_for_90` down from ~8, drift reduced) **BUT**
  (a) **`R_coarse_only` does NOT cleanly remove** the planted association: `residual_coarse ≥ 0.30` (equivalently
      `carry_coarse < 0.70`);
  (b) the QK score is **SPLIT across levels**: `frac_coarse_xx < 0.60` (coarse cell carries < 60% of the score), with
      non-trivial coarse×fine + fine×fine mass;
  (c) **no clean single-cell position-invariant + specific win** — a bounded union (`R_coarse_only < R_full_union`,
      union size > 1) is still required.
  → land in the **HELP-BUT-NOT-SOLVE band**: improvement over flat (quantify: Δtop1_coverage, Δn_cells_for_90,
  residual) AND demonstrate the level-spreading → motivates nested-cell-aware attribution (attribute to the coarsest
  sufficient level — the real research problem).

**FALSIFY (prediction WRONG, naive works):** the coarse-concept cell IS the stable carrier — `R_coarse_only ≥ 0.70 ·
R_full_union` at ALL contexts with **low `collat`** (≤ flat's), `frac_coarse_xx ≥ 0.70`, `top1_coverage ≥ 0.80` → a clean
single-cell position-invariant + specific win. Then the hierarchical SAE solves drift naively and path-2 reframes.

**HELP-BUT-NOT-SOLVE band (predicted):** between the two — `frac_coarse_xx ∈ [0.6, 0.7)` or `residual_coarse ∈ [0.2,
0.3)` or improved-but-not-clean. Quantify HOW MUCH it helps; this is itself the informative result.

### 5.2 Biggest way this could MISLEADINGLY CONFIRM (and the guard)

1. **A badly-trained Matryoshka that never recovers the concept.** If the coarse prefix doesn't contain C
   (`recovery_matry` fails), then "coarse-cut leaves residual" is trivially because there is no coarse cell — a FAKE
   confirm. **GUARD = the §2.3 recovery gate: do NOT score the prediction unless `max|cos(base[idxC], W_dec[:16])| ≥
   0.8`.** Tune `m_coarse`, `inner_weight`, α, steps until the gate passes; only then read CONFIRM/FALSIFY.
2. **Drift planted too hard (m too large / p_mix too high / α too low) so EVEN a good Matryoshka can't help.** Guard:
   verify the FALSIFY regime is reachable — at high α (0.9) the design MUST be able to produce `frac_coarse_xx ≥ 0.7`
   (a positive control that the measurement can detect a clean coarse cell when one exists). If high-α never confirms
   "clean", the level-spreading metric is broken, not the prediction.
3. **Coarse-cut residual that is actually just reconstruction error**, not cross-level routing. Guard: report
   `R_full_union ≈ oracle (Ahat ground-truth cut) ≈ 1`; if the FULL SAE union can't remove the association either, the
   SAE is too lossy and the residual is FVU, not the higher-hierarchy-cells problem. The residual must be attributable
   to *cross-level cells*, shown by `R_coarse_plus_crosslevel > R_coarse_only` closing most of the gap.

### 5.3 Biggest way this could MISLEADINGLY FALSIFY (and the guard)

1. **α set so high the leaves ARE the concept** (collapse), so there is no real drift and the coarse cell trivially
   wins — a fake falsify of a problem that wasn't planted. **GUARD = the flat-SAE-must-drift gate: only headline α where
   `top1_coverage(flat, concept) < 0.5` (real drift present).** A "clean coarse win" only counts as FALSIFY if the flat
   SAE genuinely drifted at the same α.
2. **Coarse cut that secretly also cuts the leaves** (because the coarse decoder direction has high cosine with the
   leaf directions when α is high). Then `R_coarse_only` is high not because the coarse cell carries the score but
   because cutting it collaterally removes the leaf contributions. Guard: verify specificity — the coarse cut's removal
   of *leaf×leaf* cell scores should be SMALL (`frac_coarse_xx` measures this directly); if cutting coarse also zeroes
   fine×fine mass, report it as the α-collapse confound, not a clean win.

---

## 6. WHAT THE EVALUATOR BUILDS (Phase B handoff)

- **Extend `synth_hier2.py`** → `hiersae_gen.py`: the §1 generator with the m-leaf/context drift + per-token labels +
  superpos knob. Keep the WQ/WK build, σ-calibration, `m_k` channel readout, and `cell_S`/`measure` machinery.
- **Add `MatryoshkaSAE`** to `sae_models.py` (the §2.2 sketch). Train flat + Matryoshka with the `clean_sae_pipeline.py`
  loop (Adam 1e-3, normalize_decoder, FVU + dead-frac). **CPU**, no GPU.
- **`hiersae_fra.py`**: the §3 FRA cell decomposition on each trained SAE (synthetic WQ/WK, no RoPE/LN), the level-block
  partition (§3.1), the coarse/union cuts (§3.2 reusing `cell_S`).
- **`hiersae_metrics.py`**: §4 three-axis metrics + the α×superpos sweep + the §5 gates.
- **Output** `hiersae_results.json`: per (α, superpos, seed): `fvu_*`, `recovery_*` (+ gate pass/fail), `frac_coarse_xx`
  and the 4 level-blocks, `residual_coarse`, `carry_coarse`, `R_coarse_only/plus_crosslevel/full_union`, the 3-axis
  numbers (det_auroc, top1_coverage, n_cells_for_90, recovery(k), Spearman, R_persist, collat, spec_ratio) for flat &
  Matryoshka, and the headline-cell VERDICT vs the §5.1 locked bars: **CONFIRM / FALSIFY / HELP-BUT-NOT-SOLVE**.

**Cost:** d=64, ~128k tokens, two tiny SAEs, a 4×2×3 = 24-cell sweep — minutes on CPU. No pods, no GPU, no judge, no API.

---

## 7. RESUME STATE

- **PHASE A (this doc) DONE:** `HIERSAE_DESIGN.md` — planted-hierarchy spec (§1), two-SAE setup with the
  `MatryoshkaSAE` minimal-add (§2), the higher-hierarchy-cells level-spreading measurement (§3), the 3-axis ground-truth
  metrics (§4), and the confirmed (un-loosened) CONFIRM/FALSIFY + the misleading-CONFIRM/FALSIFY guards (§5).
- **NEXT (PHASE B, EVALUATOR):** build the §6 four files, run the §4 sweep on CPU, pass the §5 gates, emit
  `hiersae_results.json` + the headline-cell verdict.
- **THEN (PHASE C):** RED-TEAM the verdict symmetrically (is the level-spreading real or an SAE-training artifact?), then
  PLANNING → the gpt2 re-test (does the synthetic prediction transfer to res-jb?) or the nested-cell-aware-attribution
  follow-up (attribute-to-coarsest-sufficient-level — the real research problem the CONFIRM motivates).
- **Builds on:** `fra_win/jobs/synth_hier2.py` (planted hierarchy + FRA cut), `multitrigger_sleeper/cloud/sae_models.py`
  (`TopKSAE` + Matryoshka loss to copy) + `clean_sae_pipeline.py` (train loop), `fra/core/fra` + `fra/fra_func.py` (the
  QK decomposition), `fra_persistence/persist_results.json` (the flat-SAE drift numbers to reproduce with ground truth).
</content>
</invoke>
