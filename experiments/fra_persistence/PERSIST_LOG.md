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

## RUN STATE
- Pod `rs-persist-1` (id `wk2r16elzwhb94`) launched on NVIDIA L4. Results → `fra_persist/results/persist_results.json`.
- (results appended below when the pod finishes)
