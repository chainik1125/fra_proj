# Autoresearch OV-select — progress tracker

The loop reads THIS + `docs/autoresearch_ov_select.md` (goal) every iteration.
Pod `runpod2`, branch `jamie/autoresearch-jsdc`, leftpad SAEs. Stays local; never committed.

## Status — ALL 3 GOALS CHARACTERIZED. /loop cron `ed4af954` DELETED (2026-05-30).

Final scoreboard (leftpad, comparable): G1 non-cheating OV selection `cos_attn` rerank 0.41–0.43
(robustly > deployed 0.454, P=0.91); G2 removal — additive DoM weak ~0.80, projection-ablation
DoM L0 resid_mid floor ~0.48 (loses to conv 0.434 / OV); G3 elicitation on clean prompts WORKS
(ASR 0→0.71–0.77, ln1-DoM best, QK fails → OV/value pathway). Full detail in
`autoresearch_ov_results.md`. To resume: add a Next-action + re-`/loop`.

### (historical) Goal-1 convergence note

Best non-cheating selector: **`cos_attn_top3_minasr` → mean honest JSDc 0.411**
(α=4: 0.421, best-α: 0.410).
- Cheating ceiling (best-of-top3 == best-of-top20) = **0.406**. Gap = **+0.005**.
- Deployed baseline (`attr_rank1` / `min_asr_winner`) = 0.454 honest (0.470 @ α=4).
- Goal (≤ 0.42 honest, robust) **MET**: fixed parameter-free rule, 0.411 on every seed.
- Robustness: `top3 ≡ top4 ≡ top3_attrtie` (identical picks) → not k-/tiebreak-tuned.
  Selector-choice LOSO = 0.427, picking `cos_attn_top3_minasr` on 5/6 holdouts.
- Stopped per protocol: iter 1 gave no improvement; closing the last 0.005 would
  require peeking at per-seed JSDc (= the cheating we forbid). Resumable anytime.

## Lead selector (current deliverable)

`cos_attn_top3_minasr`: from the attribution top-20, take the 3 highest by
attention-weighted v_md cosine `cos(W_dec[f], v_md_attn)`, then pick
`min(asr2, asr4)` (tie → attr_rank). α = ASR-screen α. **Non-cheating.**
Per-seed picks: `[1337@4, 76@4, 169@2, 1154@2, 1006@4, 1132@4]`.

## Pipeline artifacts (all cached on pod)

- Stage 0 answer key: `results/ov_leftpad_top20_eval.json` (120 feats × 9 α). ceiling 0.406.
- Stage 1 signals: `results/ov_leftpad_top20_signals.json` (6 seeds × 20 feats).
- Stage 2 search log: `results/autoresearch/ov_select_log.jsonl`; human leaderboard in
  `results/autoresearch/ov_select_run.log`.

## Leaderboard (iter 0, honest JSDc asc)

cos_attn_top3_minasr 0.411 · cos_attn_top5_minasr 0.424 · attr_x_cosattn 0.427 ·
lowasr_then_cosattn 0.432 · attr_rank1 0.454 · min_asr_winner 0.454 ·
dep_minus_cln_max 0.581 · cos_attn_max 0.657 · frac_active_max 0.683 ·
cos_all_max 0.724 · cos_last_max 0.917 · dec_norm_max 0.927 · dec_norm_min 0.940

## Scope: LEFTPAD ONLY (user, 2026-05-30)

Leftpad is the deployment SAE regime → it's the only target. Cross-SAE transfer
(v3 / harvest) is dropped as a goal; the one v3 run was a diagnostic that exposed
the overfit caveat below (see `docs/autoresearch_ov_results.md`).

## Open problem — 6-seed overfit of the 0.411 win

`cos_attn_top3_minasr` = 0.411 in-sample but LOSO 0.427, and it collapsed on the v3
diagnostic while `cos_attn_top5` / `lowasr_then_cosattn` held → the top-3 win is
partly selection-on-noise over 6 seeds. Honest number ≈ 0.427; robust selector =
`cos_attn_top5_minasr` (0.424) or `lowasr_then_cosattn` (0.432).

## Next actions (loop executes in order)

1. ✅ Goal-1 robustness done via bootstrap (E3): cos_attn family robustly > attr_rank1
   (P=0.91); top3-vs-top5 within noise. Caveat resolved.
2. ✅ SA1 reconciled — projection-ablation L0 resid_mid = 0.478 (competitive); additive DoM
   (incl. ln1, 0.80) weak. The ln1-removal gap is answered: additive ln1 weak; projection
   at ln1 not run (low priority — resid_mid is the winner).
3. ✅ DONE (E6, /loop iter 5): like-for-like projection-ablation DoM L0 resid_mid on leftpad
   eval + 5 seeds = **0.492** → DoM loses to conv 0.434 / cos_attn-OV 0.41–0.43. Comparability
   resolved; SA1's 0.478 was single-seed-optimistic.

### ALL THREE GOALS NOW CHARACTERIZED — no high-value experiment remains.
Remaining items are low-value only (projection-ablation at ln1/resid_pre; finer α around 1.5–2;
extra leftpad seeds to sharpen top3-vs-top5). **Recommend stopping the cron: `CronDelete ed4af954`.**
The docs/harness stay for resume; re-add a Next-action and re-/loop if a new goal appears.

## Active processes (hang detection — update on EVERY launch)

- none.
  (On launching a detached pod job, RECORD: `pid | launch_utc | log_path | sentinel_path | expected_secs`.)

## Iteration log

- **iter 0** [2026-05-30 ~00:52 UTC]: built full pipeline (answer key → signals →
  search). Lead `cos_attn_top3_minasr` = 0.411 vs ceiling 0.406.
  Hang incident: the chained runner's `while pgrep -f "scripts/eval.py"; do sleep
  30` wait self-matched on command text and never exited (~50 min stall). Fixed by
  killing **by PID** and running signals+search directly. Lesson encoded in the
  loop protocol's anti-hang invariants.
- **iter 1** [2026-05-30 ~01:00 UTC]: added top2/top4/attrtie variants + LOSO,
  re-ran instant `search`. top3 ≡ top4 ≡ attrtie = 0.411 (identical picks); top2 =
  0.556. Selector-choice LOSO = 0.427 (cos_attn_top3_minasr on 5/6 holdouts).
- **iter 2 (E2)** [2026-05-30 ~01:20 UTC, /loop fire]: parameterized harness
  (--top20/--key/--sigs/--tag); ran v3 (weights/seeds) transfer using existing
  ov_topk20_gated_all.json key. **cos_attn_top3_minasr collapsed to 0.517** (vs
  attr_rank1 0.403); cos_attn_top5 0.404, lowasr_then_cosattn 0.398 held up. →
  overfit caveat. User scoped to leftpad-only; transfer dropped, harvest eval
  killed (by PID + `[s]cripts` bracket trick to avoid pkill self-match), v3/harvest
  interim files removed. Next: leftpad robustness (resample, then more seeds).

## Active processes
- ✅ **SA-finaleval** DONE — definitive full eval (α 0–6, all 4 metrics, 6 seeds): DoM 0.374 ≤ OV 0.398 ≈
  Conv 0.413 (per-seed opt). OV attr-ranks 1/1/1/2/2/4 (rank>1 where rank-1 is a bad feature). α>4 mattered
  for gated methods. Recorded in ★ FINAL section + E23. Raw: /tmp/final_{ov,conv,dom}.json.

## Active processes (Goal 7 — historical)
- ✅ **SA-A** DONE — 1a: attn-weighted v_md in P1 projection-ablation @ L0 resid_mid → **JSDc 0.371**
  (★ best in project; vs 0.49 rollout-mean; ASR 0 @α1; α-best — confirm ASR-screened α). 1b: conv screen
  neutral (0.416→0.429). Lever = projection geometry + early layer + attn direction, NOT the conv screen.
- ✅ **SA-B** DONE — 2: cos_attn-OV over ALL 1536 feats UNDERPERFORMS (K=3/5/10 → 0.74/0.51/0.43 vs
  0.411). cos_attn = good reranker, poor standalone ranker; attribution prefilter is load-bearing. Keep it.
- ✅ **SA-C** DONE — 3a/3b raw v_md_attn steer: ln1-additive 0.598 / OV-only 0.664 (best @α2). Suppresses
  IHY but JSDc-weak; loses to proj-DoM/conv/cos_attn-OV → the lever is sparse feature selection, not attn-weighting.

- ✅ **Goal 7-2 K>10 sweep DONE** (ran inline on pod after 2 subagent 529s): K=3..200 →
  0.739/0.505/0.432/0.429 PLATEAU from K=20; never reaches prefiltered 0.411 (asymptotes ~0.429). Prefilter
  load-bearing even at unlimited K (seed-2 225/0.489 pre-empts 169/0.361). K=1536 not run.
- ✅ **SA-conv-vmd** DONE — conv selection: last-token 0.416 < attn 0.429 = uniform-prompt-mean 0.429.
  Whole-prompt support slightly HURTS conv selection (single-seed s1 flip 519→473) — opposite of DoM-steer/OV.
- ✅ **SA-OV-lasttok** DONE — OV last-token v_md selection → **0.698 (BROKEN)** vs attn 0.411 (picks differ
  5/6 seeds). Three-way @ last-token+prompt-only: OV 0.698 ≫ DoM 0.433 ≈ Conv 0.416. Last-token breaks OV's
  top-3 cosine cut; conv's keep-10+greedy-ASR & DoM's projection are robust. See Goal 7 standardized § / E20.
- ✅ **SA-ungated** DONE — ungated decoder-dir ablation WORSE for both: OV broadcast 0.68 (vs gated 0.41),
  conv broadcast 0.88 / projection-along-ŵ_f 0.54 (vs gated 0.42). Per-token activation gate is load-bearing.
  GAP: the decoder-direction PROJECTION was run for conv only, NOT OV → fixing now (SA-OV-proj).
- ✅ **SA-OV-proj** DONE (6-seed) — OV geometric (proj ŵ_f thru W_V) = **0.478** @α3 vs OV feature/gated
  0.41–0.42. Feature > geometric for OV too (Δ≈0.06). AUDIT COMPLETE: feature < geometric < broadcast, both channels.
- ✅ **SA-cleanup** DONE (recovered from /tmp after the AGENT watchdog-stalled but the detached job
  finished) — 6-seed: conv-geom 0.521, conv-broadcast 0.664(@α4)/0.877(@α8), OV-broadcast 0.685. Seed 5
  immaterial; ordering holds. Whole feature-vs-geometric audit now on a consistent footing.
- ✅ **SA-pos** DONE — projection-ablation pos/extraction matrix: Q1 prompt-only ≈ all-positions
  (attn 0.371→0.369; all-step does no work); Q2 last-token-only @ `:` FAILS (0.93/0.77); Q3 win
  decomposes rollout→prompt-mean (0.49→0.41) + attn-weighting (0.41→0.37). See Goal 7 pos-matrix §.

## (prior) all-Goal-6-done note. GPU idle, scratch cleaned, cron stopped.
- ✅ **SA6cr** DONE — Goal 6 (c-refined): `:` at mid/late layers (steered alone) + first-response-token
  STILL fail (ASR ≤0.005). But late-layer `:` DOES accumulate the signal (‖v_md‖ grows, cos 0.76 w/
  prompt-mean) — it's a RECOGNITION direction ~orthogonal to the speaking-IHY manifold (cos ≤0.30). The
  SA6c null stands, mechanistically explained: recognition ≠ generation direction. See Goal 6 §.
- ✅ **SA6ab** DONE (recovered from /tmp after an API socket drop) — (b) winner blocks.1.resid_post
  ASR 1.0/JSD 0.049, early/mid resid sites all clean, L3 fails; (a) learned vector is DENSE
  (SAE recon-cos 0.375), disjoint from removal features. See Goal 6 §.
- ✅ **SA6c** DONE — final-`:`-token DoM fails (ASR ≤0.005); trigger signal not at the marker.
- ✅ **SA5a** DONE — Goal 5 analytic: D2 rollout-manifold @ blocks.1.resid_post ASR 0.975/JSD 0.72
  (beats DoM, cheap); SITE decisive (mid-stack ≫ block-0 ln1); D3 logit-lens fails. Local scratch cleaned.
- ✅ **SA5b** DONE — Goal 5 LEARNED vector: **ASR 1.0 @ JSD-vs-dep 0.05** (resid_mid), crushes
  DoM (0.64/0.75) on both axes; cos(v,v_md)=0.27 → optimal direction ≠ mean-diff. Local scratch cleaned.
- ✅ **SA4** DONE — Goal 4: FRA-attribution elicitation. Negative result — FRA feature
  selection does NOT beat DoM (FRA ASR ≤0.04 vs DoM 0.51 / v_md-OV 0.73). Removal/elicitation
  asymmetry: removal sparse (ablate firing trigger), elicitation dense (recreate dep state).
  Local scratch cleaned. No active GPU jobs.
- ✅ **SA1** DONE — Goal 2 reconciled. Additive DoM weak (~0.80–0.84); **projection-ablation
  at L0 resid_mid = 0.478, competitive with OV/conv** (caveat: single-seed test-split, not
  like-for-like). New pod script `scripts/dom_baseline_anyhook.py` (untracked).
- ✅ **SA2** DONE — Goal 3 elicitation (results in tracker). Harness `scratch_elicit.py`.
- harvest eval killed 2026-05-30. No active GPU jobs.

GUIDANCE for /loop fires while these run: do NOT launch new GPU experiments for
goals 2/3 (the subagents own them, sharing the one A40). A fire should: check
subagent status (they notify on completion), and when results return, write them
to `docs/autoresearch_ov_results.md`. Leftpad Goal-1 robustness (resample / more
seeds) can still proceed if GPU headroom allows and subagents aren't saturating it.
