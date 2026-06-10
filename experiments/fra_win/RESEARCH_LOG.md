# FRA behavioral-win sprint — research log

**Goal:** find a context where Feature-Resolved Attention (FRA) gives a *clear behavioral win*
over simple baselines (mean-diff/ActAdd direction steering, full-score patching, plain SAE
steering). Prior campaigns (K1/K8 sleeper removal) showed FRA-**OV** is dominated by DoM and a
rank-2 SVD. The untested frontier is FRA-**QK** — the bilinear attention handle, which is FRA's
mathematically unique core.

**Wall clock:** START 2026-06-10T06:49Z · DEADLINE 2026-06-10T16:49Z (10h). Check hourly.
**Compute:** RunPod (RP_API_KEY_MATS), ≤$50 total, ≤$10/h. **No local compute.**

## Why QK is where FRA *should* win (the thesis)

FRA decomposes the attention score into a sum of bilinear feature-pair terms:
`S[q,k] = Σ_{i,j} (z_i(q)·W_dec[i]·W_Q)·(z_j(k)·W_dec[j]·W_K)ᵀ / √d_head`.
The term for pair (i,j) is **multiplicative in both a query feature and a key feature**. A linear
residual steer (ActAdd/DoM) adds a vector to the stream: it can change the query OR bias a key,
but it *cannot express* "boost attention specifically from queries-with-feature-A to
keys-with-feature-B" — that is a rank-1 update to W_QK, inherently bilinear. So if a behavior is
controlled by *which token attends to which* (an attention-routing task), FRA-QK has a handle no
linear method has. Induction (`[A][B]…[A]→[B]`) is the canonical such task.

The known obstacle (from `project_multitrigger_sleeper` memory): first-order FRA-QK *attribution*
is not causally faithful on the trigger task because attention there is **softmax-saturated**
(one key dominates, so score perturbations don't move the pattern). The resolution to test: pick
a **competitive** attention regime (scores near the softmax-sensitive region, multiple candidate
keys) where a score edit DOES move the pattern, and measure the *actual* behavioral effect of a
finite FRA-selected edit — not a first-order prediction.

## Pre-registered hypotheses (what would count as a win)

- **H1 (precision win).** On an attention-routing task, an FRA-QK edit (scale the dominant
  induction feature-pair's score contribution) achieves a target behavioral change (e.g. drop
  copy-probability / flip IO logit) at **lower collateral** (KL on the rest of the distribution /
  on clean traffic) than the best linear residual steer (ActAdd at query or key) calibrated to the
  same behavioral change. → Pareto: behavior-effect vs collateral, FRA-QK dominates.
- **H2 (expressivity win).** There is a behavioral target a bilinear QK edit can hit that *no*
  linear residual steer can hit at any magnitude without unacceptable collateral (because the
  effect must be conditioned on both endpoints). Existence proof acceptable.
- **H3 (selection win).** FRA-QK identifies the causally-correct feature-pair to edit, where
  activation-statistics selection (e.g. pick the highest-activating feature) picks the wrong one
  → editing the FRA-selected pair beats editing the act-selected pair.

**Baselines (must beat at least one cleanly):** (a) ActAdd / mean-diff residual steer at query
pos; (b) same at key pos; (c) full attention-score patch (oracle upper bound on behavior, but
not a localized/interpretable edit); (d) head mean-ablation; (e) random feature-pair edit (null).

**Metrics:** behavioral = task logit/probability (induction copy-prob, IOI logit-diff);
collateral = KL(clean‖patched) over non-target positions + Δ on a held-out clean corpus.

## Infra found (repo `fra/`)

- `fra/core/fra.py::get_sentence_fra_batch` — the 4-D FRA tensor `[q,k,i,j]`. hook_point
  `ln1.hook_normalized` is exact; `hook_resid_pre` uses RMS correction (exact for RMSNorm models,
  approx for GPT-2 LayerNorm); `hook_z` is legacy/approx.
- `fra/ov_steering.py::run_qk_steering` — patches `hook_attn_scores` with a modified [seq,seq]
  matrix (the FRA-resolution lives in how I build that matrix). `run_ov_steering`, `run_combined`.
- `fra/induction_head.py` — induction FRA + `analyze_self_interactions` (induction = i→i pairs).
- SAE wrappers: hook-z (`gpt2-small-hook-z-kk`), local ln1, GemmaScope.

**Model/SAE choice:** GPT-2-small (canonical induction head L5H5/L6H9, fast). SAE: start with
public `gpt2-small-res-jb` resid_pre (no training) — validate FRA reconstructs the real score;
upgrade to a trained ln1 SAE if reconstruction is poor.

## Plan / phases

- **P0 setup** (≤1h): persistent RunPod pod + HF code bundle + poll-execute harness for fast loops.
- **P1 explore** (~2h): reproduce induction FRA; confirm a clean dominant induction feature-pair;
  validate score reconstruction; first FRA-QK-edit-vs-ActAdd probe → is there signal?
- **P2 understand** (~4h): the winning experiment with all baselines + collateral; Pareto curves;
  robustness (heads, sequences, magnitudes). Pivot to H2/H3 framing if H1 weak.
- **P3 distill** (~2h+): figures (one per finding), summary.md, red/blue-team the writeup.

## Hourly log

- **06:49Z** start. Read guidelines + surveyed `fra/` infra. Thesis = FRA-QK bilinear attention
  handle on induction. Hypotheses pre-registered above.
- **07:30Z** understood FRA core decomposition. Decided GPT-2-small + res-jb resid SAE; building
  pod harness next.
- **07:45Z** built poll-execute pod harness (persistent L4, HF inbox/outbox, 1 commit/job —
  respects the 128-commit/hr cap). Submitted J1 (induction FRA: head pick, score reconstruction,
  dominant pair, first steering probe).
- **Design refinement (the sharper win):** induction is token-content-matched — the QK edge is
  query-feature("current token = r_t") × key-feature("prev-token-was-r_t"). So each edge's FRA is
  dominated by a *token-specific* pair. This makes the natural FRA win **surgical selectivity**:
  ablate one token's induction feature-pair → suppress copying of THAT token only, at ~0 collateral
  on other tokens, and *content-addressed* (works at any position, no position lookup). A linear
  ActAdd steer of the head's induction direction kills copying for ALL tokens (no selectivity); a
  position score-patch is selective but not deployable (needs exact positions). FRA sits where
  neither baseline can: selective AND content-addressed. Key graph = suppression(target) vs
  collateral(non-target), FRA bottom-right. J1's pair-aggregation tells me concentrated (→ H1
  generic-precision win) vs diffuse/token-specific (→ this surgical H2 win).
- **08:45Z J1 results.** L5H5 = top induction head (attn 0.94), copy-prob 0.73. Edges token-specific
  & diffuse (top qF423xkF13032=3.36, long tail; qF6116 recurs x3) -> supports surgical framing.
  Reconstruction weak: corr 0.50 on edges, FRA 4.22 vs actual 6.19. Fixed hook= bug in J2.
- **Key insight: softmax leverage != reconstruction corr.** Behavior is set by the softmax. Edge
  score ~6.2 vs competitors ~0 -> attn ~0.94; removing FRA-explained ~4.2 drops edge to ~2.0 ->
  attn ~0.13. So a "50%-reconstructing" FRA ablation still collapses the edge 0.94->0.13. corr
  understates FRA's behavioral leverage (softmax nonlinearity). Report corr honestly anyway.
- **08:50Z** submitted J2 (selectivity matrix + head-ablate & position-oracle baselines).
- **09:05Z J2 NEGATIVE (reshaping).** Single-head/single-edge interventions do ~nothing: ablating
  top-3 pairs Δ=-0.001; mean-ablating L5H5 only 0.73→0.69; position-oracle zeroing the edge Δ=-0.009.
  Cause: **induction is redundant** (5+ heads) and edges are softmax-saturated → output copy-prob is
  robust to one head. Behavioral metric was too robust. Fix: hit ALL induction heads + over-drive c.
- **09:30Z J3 POSITIVE (core signal).** Ablating a target token's FRA pairs across all 5 induction
  heads, over-drive c=8: **target copy-prob 0.91→0.32 (−65%)** while OTHER tokens only −0.05.
  Kill-all-heads: target→0.09 but others→0.05 too (non-selective, kills everything). So **FRA can
  selectively suppress ONE token's copying; head-ablation can't.** Collateral grows with c (Pareto).
- **09:35Z** submitted J4 (the make-or-break: FRA vs ActAdd-token-X vs random-pair-null vs oracle
  vs head-ablate, on the suppression-vs-collateral Pareto). If ActAdd-X matches FRA's selectivity,
  no win; if it can't, FRA wins (selective + content-addressed, the 2×2 only FRA fills).
- **09:55Z J4 (honest, partial setback).** random-pair null = 0 supp (✓ FRA selects the right pairs);
  head-ablate = 0.91 supp but 0.66 collateral (non-selective ✓). BUT **ActAdd-token-X beat FRA** on
  the within-task metric: full suppression at collateral 0.029 vs FRA 0.74-supp at 0.045. So measured
  by *other tokens' copy-prob*, the linear steer wins → **no win on that metric.**
- **Reframe (where FRA's edge must be).** My ActAdd was applied at a single position (unfairly cheap
  collateral). FRA's true distinction: its edit is **doubly gated** (query-feat[cue] AND key-feat
  [prev-was-cue]) → it suppresses the cue's *induction association only*, firing only in induction-like
  contexts. A *content-addressed* ActAdd-cue (must fire wherever the cue appears) corrupts the cue's
  residual in ALL contexts. This shows up only in **broad/held-out collateral**, not other-tokens'
  copy-prob. The bilinear gate is strictly finer than any linear (single-endpoint) gate — that's the
  H2 expressivity claim. J5 tests it: matched induction suppression, compare (A) off-target KL on the
  induction seq, (B) held-out real-text KL where the cue appears in normal contexts.
- **10:10Z J5 — THE WIN.** At matched 65% induction suppression: (Part 1) off-target KL on the
  induction seq FRA 0.62 vs ActAdd 7.64 = **12.3× lower**; (Part 2) held-out real text (cue=" war"):
  FRA total KL 0.13 vs ActAdd 31.1 = **~235× lower** (cue-position KL 0.018 vs 3.38). FRA suppresses
  the cue's induction *association* while preserving the cue's behavior everywhere else; the linear
  steer must corrupt the cue's residual at every occurrence. **The double-gate (query AND key feature)
  is the FRA-unique handle a linear steer cannot express.** Clear behavioral win, appropriately scoped.
- **10:20Z** submitted J6 (robustness: 8 real cue words, unified induction+held-out design, sweep both
  methods → mean Pareto of suppression vs held-out collateral, with the headline ratio at 50% supp).
- **Writeup arc:** K1/K8 showed FRA-OV loses → ask where FRA *can* win → its unique core is the
  bilinear QK gate → testbed = induction → win = association-specific suppression at ~100× less
  collateral than ActAdd, because the edit is gated on a feature *pair*. Honest scope: narrow niche
  (interventions conditioned on an attention edge); does NOT revive FRA-OV for removal.
- **10:35Z J6 robustness (8 cues).** At 50% suppression, held-out collateral FRA 0.00±0.00 vs ActAdd
  0.74±0.30. The exact-zero is the mechanism: single-mention held-out text has no induction edge → the
  doubly-gated FRA edit never fires (inert where it should be); ActAdd corrupts the lone cue mention.
- **10:50Z J7 — strongest linear baseline (key-side 'prev-was-cue') tested.** My first metric (KL at
  the cue position) wrongly showed key-side=0, because key-side acts on the position AFTER the cue.
  On TOTAL held-out KL: FRA 0.14±0.04 vs ActAdd-identity 2.28±0.50 vs ActAdd-key 2.18±0.17 →
  **FRA beats BOTH linear baselines ~16×.** Key-side doesn't escape collateral, it relocates it. Win
  robust to the strongest linear competitor. Wrinkle: FRA's suppression *ceiling* < ActAdd's (only
  removes the FRA-explained part of the edge) — present via Pareto, not matching.
- **11:05Z** submitted J8 (definitive Pareto: FRA vs both ActAdds, all-edge-pairs, multi-occurrence
  held-out, 8 cues — the money figure). Drafted summary.md.
