# PERSIST_LOG.md — EVALUATOR (Phase B) build + run log

EVALUATOR agent for FRA PERSISTENCE (Path 1). Spec = `PERSIST_DESIGN.md` + `CAMPAIGN.md`.
Judge-free (ground-truth copy-prob). Centerpiece = M2 (SAE-feature-consistency of A→B across
~120 appearances). Reuses `fra.core.fra._build_fra_result` + the j2 `pairs_delta` / `hook_attn_scores`
cut verbatim (gpt2-small + gpt2-small-res-jb resid_pre, head L5H5 primary).

## REFRAMES (coordinator, mid-run) — headline evolved twice; all reuse the same machinery
- **Reframe 1 (diagnosability, M4):** a UNION of cells is EXPECTED (oracle is multi-head/distributed).
  The real question: can FRA's per-cell score `s=|ω u_q u_k|` (FREE) DIAGNOSE which cells form the
  causal union (so it replaces brute-force ablation search)? Metrics: Spearman(s,c), recovery(k) =
  removal(FRA-top-k) / removal(causal-top-k), minimal diagnosed union, held-out transfer.
- **Reframe 2 (the 2×2, THE HEADLINE):** the interesting axis is POSITION-INVARIANCE. At a KNOWN
  position FRA-cut == attention-map cut (no edge — expected). At RANDOM/unknown positions:
  - attention-map cut = position-SPECIFIC, assoc-specific → FAILS (can't target the key position).
  - embedding cut (THE FAIR BASELINE) = position-INVARIANT, assoc-BLIND → removes A→B everywhere but
    kills A's OTHER uses.
  - FRA cut = position-INVARIANT + assoc-SPECIFIC → the claim: removes A→B wherever A fires, preserving
    A's other uses.
  WIN = FRA removes at random pos ≈ embedding-cut AND ≥2× attn-map(guessed) AND FRA collateral on A's
  other uses ≤ ½ embedding-cut collateral. NULL = FRA collateral ≈ embedding (no specificity) OR FRA
  fails to remove at random pos.
- Both folded into `persist_run.py` (M4 = diagnosability, M5 = the 2×2). M1/M2/M3 retained as secondary.

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

## RUN HISTORY continued (infrastructure churn → final clean run)
- `rs-persist-4` (L4, 450 s): first complete 3-pair M1/M2/M3 run (multi-head locus). Established the
  INFORMATIVE-NEGATIVE on persistence. Saved as the M1/M2/M3 baseline.
- `rs-persist-5..9`: died mid-run (pair-2 M4 OOM on the diffuse `' mell'→'tl'` edge + repeated L4
  preemption + the shared-pod reaper's ~10-25 min window). Fixes layered in: per-appearance FRA-array
  free after M4 (`c.pop("AF")`), `try/except` around M4 (a pathological pair → NaN, not a crash),
  48 GB RAM, A40 (not L4), throttled ckpt uploads (HF 128-commit/hr cap).
- `rs-persist-10` (A40/48 GB): completed in 153 s but the gate cap (250) found only 1 pair.
- **`rs-persist-11` (A40/48 GB, 302 s, id `rzv9oygpw1ohp2`) — FINAL, all 3 pairs, all of M1–M5.**
  Fast gate (1-forward pre-screen → cap 600 finds 3 pairs in seconds; stop-at-3) brought the whole run
  under the reaper window.

## RESULTS (rs-persist-11, multi-head locus, 3 pairs, N=101 appearances)

### Gate-passing pairs (frozen before M1/M2), all sanity gates PASS
| A→B | gate copyprob | parametric prior | multi-head oracle ceiling | non-sink |
|---|---|---|---|---|
| `'Pa'→'azar'` | 0.30 | 1.6e-07 | 0.98 | yes |
| `' raced'→'hip'` | 0.31 | 4.5e-06 | 0.96 | yes |
| `' mell'→'tl'` | 0.30 | 2.5e-06 | 0.79 | yes |

base copyprob 0.89 (LOCATE) / 0.82 (HELD-OUT) ≥ 0.30; oracle ceiling 0.79–0.98 ⇒ the multi-head locus
is REAL (every intervention has a high ceiling to hit).

### ★ HEADLINE — M5 the 2×2 (position-invariance × association-specificity), pooled
Removal of A→B at RANDOM/unknown probe-A positions (held-out), 3 interventions, no position knowledge:

| intervention | rem@random | position-invariant? | association-specific? |
|---|---|---|---|
| **FRA cut** (diagnosed (A×A) union) | **0.073** | yes | yes (collateral 1e-4) |
| **embedding cut** (zero A-feature ∀pos) | **0.098** | yes | NO (collateral 0.294) |
| **attn-map cut** (fixed guessed key pos) | **0.000** | no | — |

- **Collateral on A's OTHER uses (benign): FRA = 1e-4 vs embedding-cut = 0.294 → embed/FRA = 1994×.**
  This is the ONE robust FRA edge: it is ~2000× more association-specific than the embedding cut.
- **VERDICT_2x2 = NULL** (FRA fails to remove at random positions). Deciding: rem_random FRA = 0.073 <
  the 0.40 removal floor (and ≈ the embedding-cut's own 0.098 — neither removes much). attn-map = 0
  confirms position-specific masks can't target random positions, but FRA ≈ embedding ≈ weak, so the
  position-invariance axis does NOT separate them here: **both position-invariant cuts are weak because
  the induction edge is diffuse**; the only thing FRA wins is specificity, which is the already-deflated
  per-instance property, not removal-at-random-positions.

### M4 — FRA-diagnosability of the causal union (can FRA score `s` predict causal `c`?), pooled
- **Spearman(s, c) = 0.03** (per-pair 0.23 / −0.37 / 0.23) — FRA's per-cell score essentially does NOT
  rank cells by causal effect. → **INFORMATIVE-NEGATIVE: you cannot read the union off FRA; brute-force
  ablation is still required.**
- recovery(k) = removal(FRA-top-k)/removal(causal-top-k): k1 0.14, k3 0.71, k5 0.52, **k10 1.00**. FRA
  only catches up at k≈all-candidates (trivially), not at small k.
- min_diagnosed_union_90 = 8.1; transfer (LOCATE-diagnosed FRA-top-10 union cut on HELD-OUT) rem = 0.073
  (frac of oracle 0.08) — the diagnosed union does not transfer to meaningful held-out removal.

### M1/M2/M3 (secondary), pooled
- **M1 persistence**: rem_holdout(single cell) = 0.006, rem_holdout(k≤3) = 0.018, frac_of_oracle ≈ 0.006.
  The single-cell / bounded-union cut is causally negligible vs the 0.91 oracle ceiling.
- **M2 consistency**: top1_coverage = 0.31, cov3 = 0.60, n90 = 8, **q-side drifts** (head_top1 = 0.68 —
  even the dominant head drifts ~32%). edge_cov_mean = 0.003–0.019 — the dominant cell carries only
  **0.3–1.9% of the edge** (the diffuse-edge / concentrated-but-weak fact). `'raced'→'hip'` is highly
  consistent (top1cov 0.94, n90 1) yet its cut removes ~0% — the concentrated-but-weak trap, cleanly shown.
- **M3a** token-mask(no-detector) ≈ 0, oracle 0.82–1.00. **M3c** benign collateral FRA ≈ 0 vs
  feature-ablation 0.29 vs ActAdd 3.8 (the specificity win, ablate/FRA ≈ 2000×).
- **VERDICT_PERSISTENCE = INFORMATIVE-NEGATIVE** (rem_k3 = 0.018 < 0.40).

## OVERALL VERDICT (all three framings agree): INFORMATIVE-NEGATIVE / NULL
The FRA cell-cut is NOT a position-invariant, association-specific *removal* tool for an in-context
word-association on gpt2-small, because the induction QK edge is a **diffuse sum over thousands of SAE
feature-pairs (edge_cov ~1%)** distributed across a 5-head set. Consequences, each measured cleanly:
1. **No removal**: single-cell or bounded-union cut removes ~0.6–7% vs the 0.91 oracle ceiling (M1, M5).
2. **Not diagnosable**: FRA's per-cell score does not rank cells by causal effect (Spearman ≈ 0.03), so
   FRA does NOT replace brute-force ablation search for finding the causal union (M4).
3. **No position-invariance edge over the fair baseline**: FRA removal ≈ embedding-cut removal (both
   weak); FRA's *only* edge is ~2000× lower collateral (specificity) — the already-deflated per-instance
   property, not persistence/removal.

### Handoff to path-2
The deliverable is a clean, triple-confirmed negative that localizes the obstacle: the association's
content is **not** one (or a few) causally-load-bearing SAE feature-pairs in the QK edge — it is diffuse
and multi-head, and the SAE q-side feature drifts with left-context. Path-2 needs a decomposition where
A→B content is a single causally-load-bearing component (the q-side, left-context-invariant direction is
the concrete target).

## ARTIFACTS
- Code: `experiments/fra_persistence/cloud/{persist_run.py, persist_diag.py, launch_persist_pod.sh}`.
- HF: `fra_persist/code/persist_run.py`; results `fra_persist/results/{persist_results.json,
  diag.json, run_rs-persist-11.log}`. Final JSON copied to `experiments/fra_persistence/persist_results.json`
  (+ `locus_diag.json`).
