# INDUCTION_LOG.md — CIRCUIT #1: the FRA faithfulness anchor (gpt2-small, L5H5)

*EVALUATOR agent, FRA × CIRCUIT-TRACING track. Spec: `CIRCUITS.md` §1 + `CAMPAIGN.md`. This is the
FAITHFULNESS-FIRST gate: if the FRA QK decomposition cannot reconstruct induction's known-simple QK and
recover the SELF/token-match cell, we do NOT trust it downstream (IOI #2 is gated on this passing).*

---

## DESIGN

**Model / SAE:** gpt2-small + `gpt2-small-res-jb` @ `blocks.{L}.hook_resid_pre`. Head: **L5H5** (re-confirmed as
the argmax over mean attention on the induction edge; top-8 printed). RoPE off, LayerNorm path.

**Operating regime:** repeated-random-token sequence `BOS + R + R` with `R` = 24 distinct mid-vocab random
tokens. Induction edges = (query @ 2nd occurrence of `r_t`, key @ position of `r_{t+1}`); induction target =
`r_{t+1}`. This is the canonical induction operating regime.

**Decomposition (REUSED, not rebuilt):** `fra.core.fra._build_fra_result` computes the sparse 4-D FRA tensor
`S[q,k,i,j] = u^i_q u^j_k ω_{ij}`. Summing over the feature dims `(i,j)` gives `recon[q,k]`, compared to the
real pre-softmax scores `cache["blocks.L.attn.hook_attn_scores"][0,H]`. This is the `j1_induction_explore.py`
reconstruction verbatim. The surgical cell-cut + selectivity matrix is the `j2_selectivity.py` machinery.

**The single consolidated job** (`induction_anchor.py`, HF `fra_circuits_induction/code/`) runs three parts +
verdict, with `ckpt()` dumping partial JSON inside every loop and a top-level try/except that uploads the
traceback:

1. **A5 FAITHFULNESS (primary):**
   - reconstruction `corr(all causal S, recon)` and the edge-restricted `R²(S_edge | top-k cells)` =
     `1 − ‖S_edge − recon_edge‖² / ‖S_edge − mean‖²`. **GATE: R² ≥ 0.7.**
   - dominant cell = argmax |Σ FRA over induction edges| over `(i,j)`. **GATE: it is a SELF cell `i==j`**
     (the token-identity match), NOT a sink/positional cell.
   - **CCF non-sink red-team:** build a sink set (features active in >50% of 20 generic sentences); confirm the
     dominant cell is NOT in the sink set, and report CCF = non-sink mass / total mass on the induction edges.

2. **A2 CONTENT-CONJUNCTION (token-tracking):** re-run the whole decomposition on **fresh random token sets**
   (seeds 1,2,3). Confirm the per-seed dominant cell is STILL a SELF cell (content-tracking, not a fixed
   position-Δ cell) and that the dominant cell INDEX varies with the token set (it tracks the copied token's
   identity, not a position).

3. **A3 SURGICAL CUT (selectivity):** `j2` selectivity matrix `M[target,affected]` = Δcopyprob after cutting
   each target edge's top-3 FRA pairs (content-addressed score delta subtracted at `hook_attn_scores`).
   - diagonal-dominance ratio = mean|on-diag Δcopyprob| / mean|off-diag Δcopyprob|. **GATE: ≥ 2× (the banked
     ~15× should reproduce).**
   - **selectivity multiple** = head-mean-ablation collateral / cell-cut collateral (the cell-cut should hit
     only the target token; head-ablation hits all tokens).

**VERDICT — FAITHFULNESS PASS iff:** R² ≥ 0.7 AND dominant cell is SELF & non-sink AND A3 diagonal-dominance
≥ 2×. (A2 all-self reported as corroborating content-evidence.)

**GROUND-TRUTH throughout** — R², copy-prob deltas, the selectivity matrix. NO LLM judge.

**Compute:** RunPod, pod `rs-circ-induction-1` on NVIDIA L4 (gpt2-small is tiny), HF prefix
`fra_circuits_induction/{code,results}`, RP_API_KEY_MATS. Parse-gated before upload; single-shot run then pod
self-stops. Periodic partial-result push every 120s.

---

## RESULTS

Ran clean (`rc=0`, no silent errors) on pod `rs-circ-induction-1` (NVIDIA L4, id `o2h3lhch95sfvn`, self-stopped
EXITED). Head-find correctly recovered **L5H5** as the top induction head (mean edge attn 0.942, ahead of L6H9
0.929, L5H1 0.907). Artifacts: HF `fra_circuits_induction/results/run1/` — `induction_anchor.json`, `out.log`,
`anchor_recon.png`, `anchor_selectivity.png`.

### A5 — FAITHFULNESS (the primary gate)
| metric | value | gate | pass? |
|---|---|---|---|
| `corr(all causal scores)` | **0.698** | (diagnostic) | — |
| `corr(induction edges)` | **0.497** | (diagnostic) | — |
| **`R²(S_edge \| top-k cells)`** | **−2.53** | ≥ 0.7 | **FAIL** |
| edge actual mean / FRA mean | 6.19 / 4.22 | — | FRA captures 68% of edge magnitude |

The FRA recon tracks the score's SHAPE moderately (all-causal corr 0.70) and recovers ~68% of the edge
magnitude, but **systematically undershoots** (drops the bias/query-mean term — the known FRA caveat noted in
`j1`) and the **edge-restricted R² is strongly negative**: among the 23 induction edges the score variance is
small relative to the constant offset FRA misses, so a scale/offset miss blows up the normalized R². This is
NOT a "decomposition is pure noise" failure (corr 0.70, content-addressed below) — but it is a clear **FAIL of
the pre-registered R² ≥ 0.7 quantitative-reconstruction gate**.

### A5/A2 — DOMINANT CELL (is it the SELF / token-match cell?)
- Dominant induction-stripe cell = **`qF423 × kF13032`, i ≠ j → NOT a SELF (i==j) cell. FALSIFIER triggered.**
  None of the top-15 stripe cells are diagonal. The induction QK match is a content×content conjunction with
  **different** query-side and key-side SAE features (mechanistically sensible: query = "2nd-occurrence of token
  X" feature, key = "token-that-follows-X" feature — genuinely distinct residual features, not one shared
  index), so the naïve "literal same-feature `i==j`" prediction is **wrong**.
- **CCF non-sink screen = 0.975, dominant cell NOT in the 50-feature sink set.** So the cell IS content
  (non-sink), NOT a generic/positional/sink feature. The falsifier's "it was positional all along" branch does
  **not** apply — this is a *different content cell than predicted*, not a positional artifact.

### A2 — TOKEN-TRACKING (fresh random token sets, seeds 1/2/3)
- Per-seed dominant cells: seed1 `qF11286×kF21948`, seed2 `qF2897×kF12046`, seed3 `qF6190×kF1692` — **all
  non-SELF, all distinct.** `all_self = False`; `index_tracks_token = True` (the dominant cell INDEX changes
  with the token set → content-tracking, not a fixed position-Δ cell). So FRA's dominant cell IS
  content-addressed, just **off-diagonal**, not the predicted diagonal self-match.

### A3 — SURGICAL CELL-CUT (selectivity matrix `M[target,affected]`)
| metric | value |
|---|---|
| diagonal-dominance \|diag\|/offdiag | **3.9×** (gate ≥ 2× → passes) |
| **selectivity multiple** (head-ablation collateral / cell-cut collateral) | **129×** (0.0455 vs 0.00035) |
| frac of edge score removed by top-3 cells | **0.20** |
| target Δcopyprob (diag mean) | **−0.0014** (negligible) |

The cut is **selective** (3.9× diagonal-dominant; 129× less collateral than head-mean-ablation, which drops
copy-prob 0.730→0.688 across ALL tokens). BUT it is **weak**: the top-3 cells carry only ~20% of the edge
score, so cutting them barely moves the induction behavior (Δcopyprob −0.0014). The 129× selectivity is real
but it is the selectivity of a *small* effect — consistent with the A5 reconstruction shortfall (the cells
capture a fraction of the score, so they cut a fraction of the behavior).

---

## VERDICT: **FAITHFULNESS FAIL** — do NOT green-light circuit #2 (IOI) on the decomp as-is.

Pre-registered gate (CIRCUITS.md §1): PASS iff R² ≥ 0.7 **AND** dominant cell is SELF (i==j) non-sink **AND**
A3 diagonal-dominance. Result:

| gate | result | pass? |
|---|---|---|
| A5 `R²(S_edge) ≥ 0.7` | −2.53 | **FAIL** |
| A5 dominant cell SELF (i==j) & non-sink | off-diagonal `qF423×kF13032`, non-sink | **FAIL** (not SELF) |
| A2 all-self on fresh tokens | all 3 seeds off-diagonal | **FAIL** (not self) |
| A3 diagonal-dominance ≥ 2× | 3.9× (selectivity multiple 129×) | pass |

**Two of three faithfulness gates fail, including both A5 sub-gates.** This is the falsifier the spec
pre-registered ("reconstruction R² is low … OR the dominant cell is [not] the hypothesized content conjunction
[as a SELF cell]"). Per the carry-over rule (`faithfulness FIRST; a decomposition that doesn't reconstruct the
QK is a story, not a result`), this **STOPS the green-light to IOI** until resolved.

### Honest characterization (this is NOT a "decomp is noise" fail)
The decomposition is *partially* faithful and content-meaningful — it is the two **specific anchor predictions**
that fail:
1. **Quantitative reconstruction fails** (R² −2.53): FRA drops the bias term and undershoots the edge scores by
   ~32%; the top-3 cells carry only 20% of the edge score. The decomp explains the *direction* (corr 0.70) but
   not the *magnitude* of induction's QK.
2. **The "SELF / i==j token-match" hypothesis is FALSIFIED**: the dominant cell is an **off-diagonal
   content×content** conjunction (distinct query and key features), content-addressed (CCF 0.975, non-sink,
   tracks the token across seeds), NOT the predicted literal same-feature diagonal match. FRA "confirms a
   content match" but NOT the diagonal-self story the anchor pre-registered.
3. A3 selectivity holds *qualitatively* (3.9× diagonal, 129× vs head-ablation) but on a *weak* effect.

### Diagnosis + cheapest fixes (no restart-loop) — for PLANNING / RED-TEAM
The fail is concentrated in the A5 reconstruction + the SELF-cell prediction. Two cheap, targeted
re-measurements (same pod pattern, same tiny L4, minutes) would tell us whether the anchor is *fixable* or the
thesis is genuinely wounded:
- **(a) Fix the R² metric to include the bias term — but the scatter says this is only a PARTIAL fix.** The
  −2.53 is partly a metric artifact: FRA omits the query-bias / LN-mean offset that `j1` already flags, and the
  recon points sit systematically *below* the y=x line (see `anchor_recon.png`). Re-report R² of `recon +
  per-query-bias`. HOWEVER the scatter also shows real *scatter* (edge corr only 0.50), so a pure
  offset-correction will lift R² but likely NOT all the way to 0.7 — the reconstruction is genuinely imperfect,
  not just shifted. Treat corr (0.70 all-causal / 0.50 edges) as the honest faithfulness number; it is
  "moderate," not "high," and below the bar the anchor needs.
- **(b) Re-examine the SELF-cell hypothesis as a relaxed "content-conjunction" claim.** The dominant cell is
  off-diagonal but content-addressed (tracks token, CCF 0.975). The anchor's *narrow* `i==j` prediction is
  dead, but the *broad* "FRA pins a content conjunction, not a positional cell" claim survives — the SELECTION
  agent should decide whether to (i) re-pre-register induction's claim as off-diagonal content-match (then it
  PASSES A2/A3 and only the literal-diagonal sub-claim fails), or (ii) hold the strict anchor and treat this as
  a genuine faithfulness wound that blocks IOI.
- **Do NOT relaunch blindly.** The numbers are clean and reproducible (head-find correct, rc=0); the open
  question is *interpretation/metric*, not a flaky run. The decisive cheap follow-up is the bias-corrected R²
  (a) — that single number decides whether the anchor is a metric artifact or a real quantitative fail.

**Bottom line for the orchestrator:** by the letter of the pre-registration, **CIRCUIT #1 = FAITHFULNESS FAIL**
(R² −2.53; dominant cell off-diagonal, not SELF). The decomp is content-meaningful and surgically selective
(A2 content-tracking + A3 129× both hold), but it does NOT quantitatively reconstruct induction's QK and the
dominant cell is NOT the predicted diagonal token-match. Recommend: run the bias-corrected R² re-measure before
deciding whether to (a) salvage the anchor on a corrected metric + relaxed off-diagonal-content claim and
proceed to IOI, or (b) treat the quantitative-reconstruction shortfall as a real wound and pause the track.

