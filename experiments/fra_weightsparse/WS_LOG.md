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

---

# CANDIDATES RE-DO (the original campaign tasks on the weight-sparse substrate) — started 2026-06-12

PI directive: re-run the two ORIGINAL candidate tasks on the weight-sparse ladder, in the NEURON basis (reuse the B1 code):
(T1) IDENTIFIER-INDUCTION = the code analogue of the gpt2 persistence/drift experiment (the sharpest substrate test, because
the identifier is DIFFERENT in every context — the q/k codes CANNOT be one fixed cell, unlike the quote conjunction);
(T2) BINDING UP THE LADDER = unblock the set_or_string 2-hop edge that was AT CHANCE on the 1x afrac0.25 model by climbing
the EF (width) ladder to the smallest sparse model that passes the binding accuracy gate.

## PREREG (LOCKED before running — anti-post-hoc)

### Accuracy gates (operating-regime gate FIRST, the hard lesson from binding-at-chance)
- **T1 induction gate**: a model "does the task" iff **top-1 next-token accuracy on the identifier-completion >= 0.70**
  on the held-out-of-FRA probe set (the model's argmax over the full vocab equals the bound identifier's first token).
  Chance is ~1/(#candidate identifiers) << 0.1. If the 1x ladder fails, try EASIER variants IN ORDER and record which works:
  (i) shorter A..A range, (ii) more in-context repetitions of the (id, value) pattern, (iii) a wider EF model. Only FRA a
  (model × variant) cell that PASSES the gate.
- **T2 binding gate**: **binding score-accuracy >= 0.70** (P('.add(' | set-var query) vs P(' += ' | str-var query),
  the B1 `bind_scores`). 1x afrac0.25 was 0.52 (chance); wsda 0.70; dense1_1x 0.85. Climb EF {1x→2x→4x→8x} at a fixed
  nonzero budget; pick the SMALLEST sparse (afrac<1) model that passes, plus its byte-identical afrac1.000 twin.

### Drift thresholds (what counts as "drift killed" vs "drift survives")
Anchors: gpt2-dense+SAE induction **top1_coverage = 0.31, n_cells_for_90 = 8** (drift pathology); B1 quote circuit
**top1_coverage = 1.00, dominant cell 200/200 stable** (drift killed, but on a FIXED conjunction).
- **DRIFT KILLED** = the induction edge has top1_coverage **>= 0.70** AND n_cells_for_90 **<= 3** on the sparse model
  (a small, stable cell-union carries the matching mechanism despite the identifier varying every context).
- **DRIFT SURVIVES** = top1_coverage **<= 0.40** (reproduces the gpt2 0.31 pathology even on the sparse substrate).
- **PARTIAL** = in between. The DECISIVE comparison is sparse-vs-wsda-vs-dense on top1_coverage + edge-cosine + n90.
- KEY DISTINCTION from B1-quote: here the IDENTIFIER varies arbitrarily, so a single fixed (qF×kF) cell is NOT expected.
  The substrate-half claim is sharper: is the EDGE STRUCTURE (the induction matching mechanism: q reads "I am a repeat of
  a seen token", k reads "I am that token") stable/low-dimensional in the neuron basis even as the identity varies? We
  measure top1_coverage of the dominant cell AND edge-matrix cosine across contexts (the latter is identity-robust:
  high cosine + low top1_coverage = a stable LOW-RANK mechanism realized in different cells; this is the interesting
  middle the quote circuit could not show).

### Spearman expectation (pre-registered ≈ 0, union-recovery as the working alternative)
- Per B1 + gpt2 F5: **Spearman(FRA-score, causal-cut-effect) ≈ 0** on ALL substrates (locked prediction; FRA magnitude
  does not rank causal effect). The WORKING ALTERNATIVE = **union-recovery**: FRA-top-10 union removal / causal-top-10
  union removal >= 0.70 where the substrate is sparse (B1-quote sparse was 0.76). Report both; do NOT collapse.
- Causal localization (cuttability): pre-register that the sparse model should be MORE cuttable (small cell-union removes
  the edge) than dense, mirroring B1-quote (sparse 10-cell cut 0.86 vs dense 0.01), IF the edge is in fact compact.

### T2 binding-specific prereg (the ground-truth-edge + selectivity tests)
- FRA recovers the paper's hand-traced "4 query/key channels": top-k FRA cells should concentrate onto a small channel
  set; report top1/top8 edge-mass + how many distinct cells cover the edge.
- **Binding selectivity** (from fra_organisms2): cut the binding edge for a SPECIFIC (set-var) instance; measure removal
  of THAT binding vs SIBLING collateral (a different var's binding in the same prompt). WIN = high self-removal, low
  sibling collateral. Drift across instances = different (entity, value) pairs are the non-recurrent-conjunction condition.

### PREREG AMENDMENTS (locked 2026-06-12, BEFORE any FRA/causal/selectivity result was computed)
- **Edge-head selection rule**: the FRA head = argmax of the PER-HEAD EDGE-MASK ORACLE (mask q_last->k_target in that
  head only), NOT the whole-head ablation argmax. (CPU smoke exposed this: ablation argmax on sparse was L0H14 — a
  layer-0 support/prev-token head whose own edge-mask removal is 0.0; FRA on it would be a strawman.) Both rankings are
  recorded. If even the best per-head edge-oracle is < 0.2, record "edge not single-head-localizable" and interpret the
  causal suite with the small-denominator caveat (drift metrics remain structural).
- **T2 selectivity WIN operationalized**: cut the 10 causally-strongest cells located on SET-binding LOCATE pairs;
  WIN = rem_self >= 0.5 on held-out set-queries AND self/sibling removal ratio >= 2 (sibling = the str-binding of the
  SAME contexts). Symmetric str-direction run too. (fra_organisms2 convention.)
- **T1 task instantiation** (committed before FRA): identifier = base_suffix (random CV syllables, ~8-10 chars,
  multi-token under the 2047-vocab tokenizer); prompt = defs of A and B + fillers, ends mid-reuse of A at the canonical
  token boundary before A's final token; target = A's final token (verified to be the IDENTICAL token id at the
  definition = k_pos); distractor = B's final definition token (2AFC); GATE metric = full-vocab argmax == target.
  Identifiers are fresh in every prompt — the q/k content NECESSARILY varies across contexts (the persistence condition).
- **Smoke gate observation (N=40, before any FRA)**: 1x sparse top1_acc=1.000, dense1_1x 1.000 on v0 — the operating-
  regime gate passes at 1x, unlike binding. Full run gates at N=200 on all three ladder models.

## RESUME STATE (candidates re-do; canonical, idempotent)
- **PREREG LOCKED (2026-06-12)** — gates + drift thresholds + Spearman≈0 above + amendments (edge-head rule, selectivity
  operationalization, T1 task instantiation). Code = code/ind_bind_pod.py (reuses csp_vendor_gpt + ws_pod FRA/causal
  machinery; row-based bind causal because teacher-forced rows break cells_delta_fn's batch alignment) + launch_pod_ws2.sh.
  CPU smoke (T1 sparse+dense + T2 1x gate recheck): EXIT 0 end-to-end; 1x bind acc 0.575 (chance-ish, replicates B1).
  Pods rs-ws2-*. Results -> HF fra_weightsparse/results/induction_binding/.
- NEXT: launch rs-ws2-1 (STAGES=T1T2). Verdicts land in §"Candidates re-do" below.

## §Candidates re-do — VERDICTS
(pending pod rs-ws2-1)
