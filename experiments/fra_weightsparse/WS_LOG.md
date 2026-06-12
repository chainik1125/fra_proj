# FRA × WEIGHT-SPARSE (B1) — evaluator log (WS_LOG.md)

Deliverable: SAE-training comparison (sparse vs dense) + FRA comparison + verdict.
Run: pod rs-ws-1 (A40, ~12 min, ~$0.10), 2026-06-12. Raw: results/ws_{stageA,stageB,stageC,combined}.json + rs-ws-1_run.log
(mirrored at HF dmanningcoe/fra-phase1-steering-data : fra_weightsparse/results/). Code: code/ws_pod.py (+ vendored circuit-gpt).
Phase 0 (model research + matched ladder + GO) = DESIGN.md. P0.5 local smoke validated loader/tasks/FRA-exactness/causal stack.

## VERDICT (one line)
**Weight-sparse PARTIALLY-to-SUBSTANTIALLY HELPS FRA: the SAE trains 3.2× cleaner (FVU 0.042 vs 0.136), FRA's cells are
~17× more mass-concentrated and dramatically more CAUSALLY CUTTABLE (10-cell cut removes 86% of the edge vs 1% on dense)
— driven mostly by ACTIVATION sparsity — but FRA's *ranking* failure (Spearman(s,c)≈0) is SUBSTRATE-INDEPENDENT: even in
the cleanest basis, FRA magnitude does not rank causal effect; FRA's value there is restricting the search to a small
active set, within which a k≈10 union recovers ~76% of oracle.**

## SETUP (see DESIGN.md for full provenance)
3-model ladder (OpenAI circuit-sparsity, all d_model=256, d_head=16, n_head=16, vocab=2048, Python code):
- **sparse** = csp_sweep1_1x_3.7Mnonzero_afrac0.250 — weight-sparse (3.7M nonzero) + activation-sparse (1-in-4), 8L
- **wsda** = csp_sweep1_1x_3.7Mnonzero_afrac1.000 — byte-identical arch/weights-budget, DENSE activations, 8L
   (sparse vs wsda = the CLEAN isolation of activation sparsity)
- **dense** = dense1_1x — fully dense weights+acts, 4L (DEPTH CONFOUND flagged; corroborating, not primary)
Behavior = quote-closing (paper's single_double_quote): all 3 models acc=1.00, single dominant head each
(sparse L5H4 drop=.49 / wsda L5H0 .46 / dense L3H11 .50). Binding (set_or_string) acc: sparse .52 (CHANCE), wsda .70,
dense .85 → binding is beyond the act-sparse 1x model (capability cost of sparsity); its FRA rows are structural only.
FRA in TWO bases: NEURON (act_in + bias channel; recon R² = 1.000 — EXACT, verified) and SAE (identical TopK SAEs).
Ground-truth binary completion prob; locate/holdout split (100/100); causal = position-invariant bilinear cell-cuts on
the patched manual-attention path (zero-delta validation passed, diff=0).

## (1) SAE-TRAINING COMPARISON (PI ask #1) — identical TopKSAE d_sae=2048, k=32, 12k steps, lr 3e-4, same 2M-token python corpus, eval on held-out 200k
| model  | FVU ↓      | R²    | dead-frac | L0(eff) | token-purity | substrate: %nonzero / kurtosis |
|--------|-----------|-------|-----------|---------|--------------|-------------------------------|
| sparse | **0.042** | 0.958 | 24.9%     | 32.0    | 0.620        | 25.0% / 13.8 |
| wsda   | 0.100     | 0.899 | 3.3%      | 32.0    | 0.572        | 100% / 3.3 |
| dense  | 0.136     | 0.864 | 0.2%      | 32.0    | 0.602        | 100% / 3.3 |

**YES — the SAE trains markedly cleaner on the weight-sparse model, monotone in substrate sparsity** (3.2× lower FVU
sparse vs dense; wsda intermediate → both weight- and activation-sparsity contribute). The 25% dead fraction on sparse is
NOT a training pathology: the substrate is itself 25%-active/heavy-tailed (kurt 13.8), so it needs fewer dictionary atoms
— surplus atoms die. Token-purity is only mildly better (0.62 vs 0.60): the monosemanticity gap by this proxy is small.

## (2) FRA COMPARISON
### Neuron basis (the bench's CLEAN TEST — no SAE), quote edge, 200 contexts
| model  | top1_cov | n90 | mass_top1 | mass_top8 | mass_top32 | edge-cosine | recon R² |
|--------|----------|-----|-----------|-----------|------------|-------------|----------|
| sparse | **1.00** | 1   | **0.169** | 0.354     | 0.470      | **0.966**   | 1.000 |
| wsda   | 1.00     | 1   | 0.025     | 0.068     | 0.130      | 0.786       | 1.000 |
| dense  | 0.89     | 2   | 0.010     | 0.043     | 0.075      | 0.725       | 1.000 |

- **Concentration: sparse puts 17× more edge mass in the top cell than dense** (0.169 vs 0.010); wsda intermediate (2.5×)
  → ACTIVATION sparsity is the dominant driver, weight sparsity alone contributes ~2.5×.
- **Consistency: the sparse edge matrix is near-identical across contexts** (cos 0.97 vs 0.73), and its dominant cell is
  the SAME (channel-122 × channel-171) in 200/200 contexts. NOTE top1_cov is high even on dense (0.89 vs gpt2's 0.32):
  the gpt2 DRIFT pathology is only mildly reproduced at this scale/behavior, so mass-concentration + cuttability are the
  discriminating metrics here, not top1_cov.
- Circuit-shape recovery: wsda's dominant cell is (BIAS × k240) = a constant-query × quote-detector-key cell — exactly
  the hand-traced circuit shape in the paper ("the last token has a constant positive-valued query"). FRA-on-neurons
  reads the published circuit off the decomposition.

### Co-firing-set drift (the bench's flagged residual risk — measured, as instructed)
sparse: full active-set Jaccard across contexts = **0.31** (65 active channels, set varies a lot); top-32 Jaccard 0.27
(vs wsda 0.18, dense 0.16). **The set-level drift is REAL — ~73% of the top-32 co-firing set changes across contexts even
with monosemantic atoms — but it does NOT infect the edge: the dominant FRA cell is 100% stable.** The drift lives in the
behavior-irrelevant co-activated channels; the edge-carrying conjunction is fixed. (The PI's Effect-2 worry was correct as
a measurement and benign in consequence, for this behavior.)

### Causal: diagnosability + cuttability (quote, neuron basis; 300 cells tested; locate→holdout)
| model  | Spearman(s,c) | causal k1 | causal k3 | causal k10 | FRA-ranked k10 | oracle (edge-mask) |
|--------|---------------|-----------|-----------|------------|----------------|--------------------|
| sparse | +0.01         | 0.105     | 0.643     | **0.862**  | 0.656 (76% of causal) | 0.955 |
| wsda   | -0.07         | 0.024     | 0.061     | 0.406      | -0.00          | 0.823 |
| dense  | -0.27         | 0.001     | 0.003     | **0.011**  | 0.002          | 0.952 |

- **The big causal win: in the sparse model a 10-cell cut removes 86% of the edge (oracle 0.955); in the dense model 10
  cells remove 1%** — the edge is causally diffuse on dense (no small cell-set works: exactly the gpt2 persistence
  pathology) and causally COMPACT on sparse. wsda intermediate (0.41). Weight+activation sparsity make the behavior
  cuttable by a bounded cell union.
- **The substrate-independent failure: Spearman(FRA-score, causal-effect) ≈ 0 in ALL three models** — replicating gpt2's
  0.03 in the cleanest possible basis. FRA's top-1/top-3 picks miss (sparse k3: 0.074 vs causal 0.643). The top-FRA and
  top-causal cells SHARE the q-channel (122) but disagree on the k-partner — FRA magnitude finds the right neighborhood,
  not the right cell. By k=10 the FRA union recovers 0.656 (76% of causal-top-10) because the active set is small enough
  that the union catches the causal cells. So: localization improves from IMPOSSIBLE (dense) to WORKS-AT-k≈10-WITH-NOISY-
  RANKING (sparse); the ranking deficiency is FRA-intrinsic, not substrate-induced.

### SAE basis (Stage C), quote edge
| model  | top1_cov | mass_top1 | SAE-QK recon R² | causal k10 (causal-ranked) | Spearman |
|--------|----------|-----------|------------------|----------------------------|----------|
| sparse | 0.47     | 0.067     | 0.984            | 0.749                      | +0.06 |
| wsda   | 0.91     | 0.117     | 0.948            | 0.604                      | -0.06 |
| dense  | 1.00     | 0.147     | 0.933            | 0.191                      | +0.11 |

- **The instructive inversion: the SAE HURTS the already-sparse substrate** (top1_cov 1.00→0.47; mass_top1 0.169→0.067 —
  it re-mixes already-monosemantic channels) **and HELPS the dense substrate** (mass_top1 0.010→0.147; causal k10
  0.011→0.191). The bench's "skip the SAE on a weight-sparse model" intuition is CONFIRMED: neurons are the right basis
  there. On dense, the SAE buys concentration but still lands far below sparse-neurons on cuttability (0.19 vs 0.86).
- SAE-QK recon R² is high everywhere (0.93–0.98) — the gpt2 lossy-QK-reconstruction obstacle (R²<0) is absent at this
  scale/expansion (d_model 256, 8× SAE); it is NOT what separates the substrates here.

## HONEST CAVEATS
1. dense1_1x is 4-layer vs the sweep's 8 (depth confound) — the PRIMARY clean contrast is sparse-vs-wsda (byte-identical
   architecture), which alone shows the act-sparsity effect (mass 7×, cuttability 2×); dense sits consistently below wsda,
   consistent with the additional weight-density, but that ordering carries the confound.
2. Tiny models + a single-strong-head behavior: even dense is far less drifty than gpt2 (top1_cov 0.89 vs 0.32), so this
   does not measure how much weight-sparsity would help AT the gpt2 drift regime; it measures the substrate gradient
   under controlled conditions.
3. Binding (the 2-hop circuit) is at chance on the act-sparse model (0.52) — the capability cost of sparsity bit exactly
   the richer behavior; we could not run the binding ground-truth-edge comparison meaningfully (recorded acc + structural
   FRA rows only).
4. gpt2-persistence numbers are cross-setting references only (different model/task/basis); all claims above rest on the
   internal ladder.

## WHAT THIS MEANS FOR THE CAMPAIGN (B1 answer)
- The obstacle decomposes: **CONCENTRATION + CUTTABILITY are substrate problems (weight/activation sparsity largely fixes
  them); RANKING-DIAGNOSABILITY (Spearman) is FRA-intrinsic** (≈0 in every basis tested, here and in gpt2). FRA-as-
  magnitude-oracle fails everywhere; FRA-as-search-space-restrictor works where the substrate is sparse.
- The SAE is the wrong tool on a sparse substrate (de-concentrates) and a partial patch on a dense one (concentrates but
  doesn't restore cuttability). This sharpens hiersae: no SAE variant on dense gets you what the sparse substrate gives.
- Set-level co-firing drift exists even with monosemantic atoms but spares the edge-carrying cell — drift is a property
  of the representation's surroundings, not the circuit conjunction.
