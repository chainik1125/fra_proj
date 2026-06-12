# PERSIST_LOG.md — EVALUATOR (Phase B) build + run log

EVALUATOR agent for FRA PERSISTENCE (Path 1). Spec = `PERSIST_DESIGN.md` + `CAMPAIGN.md`.
Judge-free (ground-truth copy-prob). Centerpiece = M2 (SAE-feature-consistency of A→B across
~120 appearances). Reuses `fra.core.fra._build_fra_result` + the j2 `pairs_delta` / `hook_attn_scores`
cut verbatim (gpt2-small + gpt2-small-res-jb resid_pre, head L5H5 primary).

## BUILD

- Code: `experiments/fra_persistence/cloud/persist_run.py` (single self-contained job).
- Launcher: `experiments/fra_persistence/cloud/launch_persist_pod.sh` (pod `rs-persist-1`, L4;
  HF prefix `fra_persist/{code,results}`; bootstraps the SAME `fra_win/fra_bundle.tar.gz` j1/j2/j7 use;
  self-stops; uploads boot/run logs).
- ckpt() inside every per-appearance / per-pair loop → `fra_persist/results/persist_partial.json`;
  top-of-script try/except uploads `persist_traceback.txt`.

### Design decisions / spec-faithful implementation notes
- **Token-ID-level prompt construction (critical fix).** Building carriers by string concat +
  re-encode is BROKEN: GPT-2 BPE merges across the `{lc}{A}` boundary, so probe-A is NOT the last
  token and the appearance is silently dropped. Caught in a pre-launch tokenizer smoke test
  (0/18 string-built prompts had probe-A last). Fixed by concatenating pre-encoded single-token ids
  (`[A_id, B_id] + E(".") + fillers + [lc_id, A_id]`), exactly as j1/j2 build `[bos]+R+R`.
  Re-test: 162/162 appearances structurally valid, probe-A always last, B appears exactly once.
- **Pool.** GPT-2 tokenizes almost all of the design's nonces (`" dax"`,`" wug"`,…) as MULTI-token;
  only `" mell"` survives. The pre-registered random-mid-vocab branch (`randperm(40000)+1000`,
  round-trip single-token filter, seed 0) is therefore the working pool — exactly the spec's
  guaranteed fallback. The gate still selects the 3 highest-copyprob pairs that pass gate-1/3/4.
- **k_a unambiguous (§5.2 defense).** Every template places exactly ONE B; `make_tt` asserts exactly
  one key position has token==B and drops the appearance loudly otherwise (no silent mis-edge).
- **`top_k=None`** in `_build_fra_result` (all features) — no truncation-induced cell fragmentation.
- **Dominant cell ranked by |score|** (anchor convention), tie-aware via cov(k) + magnitude-weighted.
- **STRESS lever** = `{lc}` drawn from a single-token bank, split LOCATE half / HELD-OUT half;
  `{lc} × q-feature` contingency recorded.
- **M3a token-mask (no detector)** = apply the LOCATE primer's fixed (q,k) position mask verbatim to
  held-out (positions differ → expect rem≈0). Oracle = position-patch held-out's actual edge (j2).
- **M3c feature-ablation** = zero SAE query-feature I in the resid_pre reconstruction at all positions
  and add the resid delta back via a hook (kills A's content broadly). ActAdd = subtract A's
  mean-centered resid direction at A positions (j6/j7 cue-identity steer).
- **Non-sink gate** = q/k feature active-fraction on a generic sentence ≤ 0.50.

## VERDICT RULE (locked, PERSIST_DESIGN §4)
- sanity: base_copyprob ≥ 0.30 on LOCATE & HELD-OUT AND located cell non-sink (else INVALID).
- WIN-single: top1_coverage ≥ 0.70 AND rem_holdout(k=1) ≥ 0.70.
- WIN-bounded: top1_coverage ∈ [0.40,0.70) AND cov(3) ≥ 0.80 AND n_cells_for_90 ≤ 3 AND rem_holdout(k≤3) ≥ 0.70.
- INFORMATIVE-NEGATIVE: cov(3) < 0.40 OR n_cells_for_90 ≥ 0.5·N OR rem_holdout(k≤3) < 0.40.
- AMBIGUOUS: the band between (partial persistence).
Honest prior (anchor: off-diagonal cell, R²<0, top-3 carry ~20% of edge): INFORMATIVE-NEGATIVE most likely.

## RUN HISTORY (4 pods; the locus diagnostic was decisive)
1. `rs-persist-1` — L5H5-only, gate = copyprob≥0.30 + parametric + benign. Only `' mell'→'tl'`
   passed (1 pair). Verdict INFORMATIVE-NEGATIVE but CONFOUNDED: L5H5 oracle removed only 14% of
   the copy, so "FRA cut removes nothing" conflated SAE-cell weakness with L5H5-not-being-the-locus.
2. `rs-persist-2/3` — added a gate-5 LOCUS floor (L5H5 oracle ≥ 0.40 then ≥ 0.10). **0 pairs passed**:
   L5H5 alone almost never carries the copy.
3. `rs-persist-diag` (DIAGNOSTIC) — for 141 random single-token A→B pairs (gate2≥0.20): **L5H5-alone
   oracle = 0.08 mean / 0.81 max**, but the **5-head induction set [(5,5),(6,9),(5,1),(7,10),(7,2)]
   oracle = 0.90 mean / 1.00 max** (97/141 pairs ≥ 0.40). **The induction copy is DISTRIBUTED across
   the head set, not localized to L5H5.** → the correct causal locus is the multi-head set.
4. `rs-persist-4` (FINAL, id `b0ukw6t8dv21ac`, L4, 450 s) — multi-head locus throughout: gate-5 =
   5-head oracle ≥ 0.40; M2 "cell" = a (head, qF, kF) triple; M1/M3 cut subtracts the located cell's
   FRA contribution on its own head (j7 multi-head hook), oracle = zero the edge across all 5 heads.
   **3 pairs passed** (`'Pa'→'azar'`, `' raced'→'hip'`, `' mell'→'tl'`), oracle ceiling 0.79–0.98.

## RESULTS (rs-persist-4, multi-head locus, N=152 appearances over 3 pairs)

### Gate-passing pairs (frozen before M1/M2)
| A→B | gate copyprob | parametric prior | multi-head oracle ceiling |
|---|---|---|---|
| `'Pa'→'azar'` | 0.30 | 1.6e-07 | 0.98 |
| `' raced'→'hip'` | 0.31 | 4.5e-06 | 0.96 |
| `' mell'→'tl'` | 0.30 | 2.5e-06 | 0.79 |

Sanity gate PASSES: base copyprob 0.89 (LOCATE) / 0.82 (HELD-OUT) ≥ 0.30; all located cells non-sink
(q/k features active@0.00 on a generic sentence). Oracle ceiling 0.79–0.98 ⇒ the locus is REAL (the
cut has a high ceiling to hit).

### M1 — PERSISTENCE (the unconditional cut, held-out)
- **rem_holdout (single cell, pooled) = 0.005**; **rem_holdout(k≤3) = 0.020**.
- **frac_of_oracle = 0.005** — the FRA cell-cut removes **0.5% of what the multi-head oracle removes**.
- Per-pair rem_holdout: `'Pa'→'azar'` 0.011, `'raced'→'hip'` 0.0001, `'mell'→'tl'` 0.004. Union k=3
  tops out at 0.025/0.000/0.035. The cut is causally negligible at EVERY pair.

### M2 — SAE-FEATURE-CONSISTENCY (centerpiece, N=152, (head,qF,kF) cells)
- **pooled top1_coverage = 0.32; cov3 = 0.61; cov5 = 0.82; n_cells_for_90 = 8.**
- **q-side drifts**: qtop1 = 0.32 < ktop1 = 0.34; head_top1 = 0.69 (even the dominant *head* drifts ~31%).
- **edge_cov_mean = 0.003–0.019** — THE KEY NUMBER: the per-appearance dominant cell carries only
  **0.3–1.9% of the induction edge**. The edge is spread across hundreds/thousands of tiny cells.
- Mixed per-pair concentration: `'raced'→'hip'` is highly consistent (top1cov=0.94, n90=1, one cell
  carries the edge at 94% of appearances) yet its cut removes 0.0% — the cleanest demonstration of the
  concentrated-but-weak trap. `'Pa'→'azar'` cov3=1.0/n90=3; `'mell'→'tl'` is the drifting one (top1cov=0.34,
  n90=6, cov3=0.74).
- `{lc} × q-feature` contingency + magnitude-weighted variant recorded per pair in the JSON; they do not
  rescue concentration (the cell stays weak whether counted or magnitude-weighted).

### M3 — COMPARISON
- **M3a token-mask (no detector) = 2.3e-06 (≈0)**; FRA/mask ratio = 2187× (>> the 2× WIN bar). The
  oracle token-mask (detection granted) = 0.80–1.00. So a fixed position-mask cannot generalize to
  held-out positions — but FRA's edge here is that it removes *more than the no-detector mask*, which is
  trivially true since the mask removes ~0; FRA's removal is also ~0.005, so this "win" is hollow.
- **M3b union top-k**: gain rem(k3)−rem(k1) = 0.015 pooled — a bounded union does NOT recover persistence.
- **M3c benign-use preservation (the one clean FRA WIN)**: FRA collateral KL = 1.5e-04 ≈ 0;
  feature-ablation KL = 0.294; ActAdd KL = 3.8. **ablate/FRA = 1995×.** Benign top-1 preserved: FRA 1.0,
  feature-ablation 0.375–1.0, ActAdd 0.0. FRA is perfectly association-specific (fires only on the A→B
  cell), feature-ablation wrecks A's benign uses, ActAdd destroys everything. This clause passes — but it
  is the per-instance-selectivity property that already deflated in EM/injection/binding; it is NOT
  persistence.

## VERDICT: INFORMATIVE-NEGATIVE
**Deciding number: rem_holdout(k≤3) = 0.020 < 0.40** (the §4.3 magnitude floor). Also cov3=0.61<0.80 and
n_cells_for_90=8 fail the bounded-union WIN. Sanity gate passes, all cells non-sink, oracle ceiling high
(0.79–0.98) — so this is a REAL negative, not a low-base-rate or wrong-locus artifact.

### Why it's negative (two compounding mechanisms, both measured cleanly)
1. **CONCENTRATED-BUT-WEAK (binding, §5.3).** Even where the SAE picks ONE consistent cell across all
   contexts (`'raced'→'hip'`: top1cov=0.94, n90=1), that cell carries only ~1% of the induction edge
   (edge_cov_mean=0.010), so cutting it removes ~0% of the copy — `frac_of_oracle=0.005` pooled. The
   induction QK score is a diffuse sum over thousands of feature-pairs; no single (or top-3) cell is
   causally load-bearing. This is the same lesson as the j1/j2 induction anchor (top-3 carry ~20% of
   edge there; here even less because the locus is multi-head), now quantified for persistence.
2. **Q-SIDE DRIFT (secondary).** Pooled across pairs/heads, top1_coverage=0.32, n90=8, and the drift is
   q-side (qtop1<ktop1, head_top1=0.69): A's query feature (and dominant head) varies with left-context.
   The STRESS lever bites — but it is the SECOND-order problem; even with no drift the cut would fail on
   magnitude.

### Honesty controls passed (it is a real negative, not a binning artifact, §5.2)
- top_k=None (all features) → no truncation fragmentation. Tie-aware cov(k) + magnitude-weighted w_top1
  recorded (don't rescue). Exactly one B per primer (asserted). Multi-head locus has oracle 0.79–0.98 so
  the cut had a real ceiling. The negative is "the SAE cell is causally negligible (and moderately
  context-dependent)," not "ranks wobble on a near-tie."

### Handoff to path-2
The deliverable: **path-1's FRA single-cell-cut cannot persist a word-association because the induction
edge is not concentrated in any small SAE cell-set (edge_cov ~1%), even when the cell IS context-consistent.**
FRA at this level retains only its per-instance association-specificity (M3c, ablate/FRA ~2000×), which is
the already-deflated property — NOT a weight-space persistent edit. Path-2 needs a decomposition where the
association's content is one causally-load-bearing component, not a 1%-of-edge SAE feature-pair. The q-side
drift localizes a second target (a query-side representation invariant to left-context).

## ARTIFACTS
- Code: `experiments/fra_persistence/cloud/{persist_run.py, persist_diag.py, launch_persist_pod.sh}`.
- HF: `fra_persist/code/persist_run.py`; results `fra_persist/results/{persist_results.json,
  diag.json, run_rs-persist-4.log}`. Local copies under `/tmp/persist_res4/`.
