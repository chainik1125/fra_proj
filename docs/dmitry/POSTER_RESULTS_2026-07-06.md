# Poster results consolidation — 2026-07-06

Three threads: (1) sleeper single-model steering, (2) EM on Qwen 7B/14B, (3) in-context association cutting.
Compiled from a full archive sweep; every number below has a file path. Honest floor = coh≥70 for Δalign;
matched-removal collateral for the in-context wins.

## Memory-check verdicts (against Dmitry's recollections)

| Remembered claim | Verdict |
|---|---|
| "FRA slightly outperforms conventional SAE steering on sleeper (weak: at ln1; strong: any hookpoint)" | **REFUTED** — source is the stale TinyStories abstract; corrected 6-seed table has conventional marginally ahead (tie). At ln1 Wang beats FRA on both 7B and 14B. |
| "EM 7B single-feature steering very strong, ~0.6 coherent swing" | **PARTLY RIGHT, misattributed** — the ~66-pt Δalign@coh50 single feature is 14B (F93118) and it's Wang-ranked, not FRA; 7B tops at ~44–45 (also Wang). Winner's-curse, n_seeds=1 on those maxima. |
| "Some regimes (maybe 14B) where OV×resid_post performed better" | **CONFIRMED** — 14B finance single @coh70: FRA-OV×resid_post 32.9 vs Wang 31.3; 7B grouped @coh70: FRA-OV 13–16 vs Wang 5–10 (2–3×). Caveat: static data-free ranking, generic hookpoint, and partly Wang coherence-collapse. |
| "Toy in-context association cut works well; one-feature→many-words extension never landed" | **CONFIRMED on both halves** — wins 15×–~1100× (26,000× separability); 5 multi-word attempts all failed or were left incomplete. |

## Thread 1 — Sleeper, single-model regime

**Honest headline:** FRA-based steering is competitive with but does not beat conventional SAE (Wang enc-f)
or difference-of-means steering; FRA's demonstrated sleeper value is diagnosis, feature-selection, and
localization (exact attention-cut oracle: ASR 0.00 at J_clean 0.000), not a superior steering direction.

- Corrected TinyStories 6-seed table (`docs/dmitry/jamie_latest/RESULTS_SUMMARY.md:25-40`), JSD_clean matched ↓:
  FRA-OV 0.344±0.049, conventional SAE feature **0.307**±0.061, DoM **0.304**. Ordering flipped vs the stale
  abstract ("Pareto-dominant", "perfectly suppress") — do NOT cite the abstract (`RESULTS_SUMMARY.md:63-66`).
- ln1 head-to-head: Wang wins both models (7B single 21.7 vs FRA 12.2–13.8; 14B Δ@70 35.9 vs 6.7).
  `experiments/fra_ln1_7b/GRID_RESULTS.md:121-143`, `experiments/fra_14b_financial/SUMMARY.md:56-71`.
- 14B financial grid: Wang beats best FRA 2.9× @Δ50, 5.4× @Δ70. Caveats: head H0 not argmax H12
  (`SUMMARY.md:142-150`), n_seeds=2, gpt-4o-mini judge, unsloth-base provenance (paper-blocking, `:46,309`).
- Multitrigger K=8 (`experiments/multitrigger_sleeper/summary.md:130-139`): CAA/DoM J_clean 0.306 beats FRA
  vectors 0.45–0.63. The Pareto-winning hybrid (J_clean 0.08–0.11) uses weight-diff → excluded (diff regime).
- Footnote-grade positive: single-trigger K1 sleeper, one SAE feature beats DoM 0.291 vs 0.488
  (`summary.md:244-246`) — dies at K=8; and it's SAE-vs-DoM, not FRA-vs-Wang.
- The ONE single-model FRA-over-Wang steering slice (shared with Thread 2): FRA-OV×resid_post grouped @coh70
  on 7B, 2–3× (13.4/15.9/13.9 vs 5.3/9.5/4.9 for grp2/10/50), 3-seed-certified — a coherence-preservation
  win (Wang grouped resid_post collapses coherence). `GRID_RESULTS.md:134-148`, `summary.md:177-186`.

### Appendix: TinyStories per-seed data (6 SAE seeds), added 2026-07-06

Source files: `experiments/tinystories_sleeper/rerun4_rescue_2026-05-26/results/{jsd_alpha_sweep_6seeds,conventional_per_layer,hookpoint_sweep}.json`.
Metric: opt_J_clean = min over α of mean JSD(steered rollout, clean reference); 200 prompts.
The 6-seed sweep uses 5 decode seeds; the per-layer file uses decode seed 0 only.
NOTE: FRA-OV is *defined at ln1* (OV-channel cut of the ln1-SAE feature at blocks.0.ln1.hook_normalized);
conventional additive at ln1 has NO 6-seed run — only a single-SAE coarse sweep (feature 1412: full ASR
suppression only at α≥4.75, ΔCE +0.58 nats). Jamie's corrected-pipeline per-seed values are NOT committed
(only the means in `paper/figures/jsd_stats_table.tex` on origin/jamie/paper: OV 0.344±0.049,
Conv 0.307±0.061, DoM 0.304).

Per-seed opt_J_clean (α* in parens; ASR@α* flagged if >0.01):

| seed | FRA-OV @ ln1 | Conv @ resid_mid (5-decode) | Conv @ resid_mid (1-decode) | Conv @ resid_post (1-decode) |
|---|---|---|---|---|
| s0 | 0.480 (2.5) | 0.403 (4.0) | 0.364 (3.5) | 0.714 (1.0) |
| s1 | 0.374 (4.0) | 0.438 (2.5) | 0.506 (3.0) | 0.622 (4.0, ASR .065) |
| s2 | 0.390 (3.0) | 0.838 (2.0) | 0.878 (4.0, ASR .53) | 0.568 (4.0, ASR .015) |
| s3 | 0.420 (3.5) | 0.440 (4.0) | 0.395 (4.0) | 0.364 (3.0) |
| s4 | 0.563 (4.0) | 0.488 (4.0) | 0.466 (3.0) | 0.833 (1.5, ASR .095) |
| s5 | 0.504 (3.5) | 0.419 (4.0) | 0.370 (3.5) | 0.728 (2.5) |
| mean±sd | 0.455±0.073 | 0.505±0.166 | 0.497±0.195 | 0.638±0.162 |

Head-to-head (5-decode sweep): OV wins s1,s2,s3; Conv wins s0,s4,s5 — 3/6. OV's mean edge is entirely
conventional's seed-2 selection blowup (0.838); excluding s2, conv 0.44 vs OV 0.47. Deeper layers fail:
conventional L1–L3 opt means 0.62–0.93 with high-ASR optima (per_layer file). Baseline unsteered
JSD_clean = 0.99; clean-vs-clean floor 0.42 (Jamie corrected).

### Wide-alpha redo (2026-07-07/08, Modal app ts-alpha-redo) — SUPERSEDES the per-seed tables above

Protocol: 3 schemes x 6 seeds; fresh top-25 ranking per cell; alpha in [-10,10] step 0.25 screened at
64 prompts/decode seed 0; winner refined by GP BayesOpt (bounds ±20) at 200 prompts x 5 decode seeds,
ASR gate 0.01. Data: `worktree sae-scaling-sweep, results/alpha_redo/`; script `cloud/modal_alpha_redo.py`;
figures `results/alpha_redo/figures/fig_redo_{seedband,seed1,allseeds}.pdf`; plotter `scripts/plot_alpha_redo.py`.

opt J_clean (BO-refined, raw-sign alpha* in parens; raw>0 = subtract):

| seed | FRA-OV @ ln1 | Conv @ resid_mid | Conv @ ln1 (NEW arm) |
|---|---|---|---|
| s0 | 0.519 (+3.8, f1114) | **0.402** (+5.8, f579) | 0.543 (**-5.2**, f1220) |
| s1 | **0.381** (+3.5, f1027) | 0.415 (+3.0, f519) | 0.462 (+1.9, f1027) |
| s2 | **0.383** (+2.6, f169) | 0.664 (-2.6, f1091) | 0.412 (+2.5, f169) |
| s3 | **0.421** (+3.2, f1154) | 0.449 (+5.9, f171) | 0.682 (**-10.25**, f1511) |
| s4 | **0.398** (+5.0, f558) | 0.449 (+4.8, f515) | 0.523 (**-5.5**, f623) |
| s5 | **0.435** (+9.1, f296) | 0.457 (+3.4, f928) | 0.523 (+4.4, f1208) |
| mean±sd | **0.423 ± 0.052** | 0.473 ± 0.096 | 0.524 ± 0.091 |

Key deltas vs the old ±4 sweep: (1) FRA-OV now wins 5/6 seeds (was 3/6) — wider range + per-feature
search surfaced better OV features (s4 f558, s5 f296@9.1); (2) the old window CLIPPED conventional's
optima (true alpha* 4.8-5.9 on three seeds); (3) conv@ln1 is USABLE (overturns the n=1 pilot) but worst
arm, with negative-alpha (feature-ADDITION) winners on s0/s3/s4 — invisible to any 0..4 sweep;
(4) seed-1 same-feature contrast (f1027 additive vs OV-channel: 0.462 narrow dip vs 0.381 wide basin)
is the cleanest "why OV" panel. Caveats: winner = max over 25 features (selection winner's-curse;
J_clean re-measured on BO evals at full fidelity), SAEs s0/1/3/4/5 retrained (W_dec cos 0.797 vs
original on s2, but winner indices reproduce May's: 1114/1027/169/1154), screen curves are
64-prompt/1-decode. Conventional s2 selection blowup persists (f1091, 0.664 at its own optimum).

### K8-randpos equivalent redo (2026-07-08, Modal app k8-randpos-redo) — the clean NEGATIVE

Same protocol ported to the randpos-K8 multitrigger sleeper (TinyStories-Instruct-33M + q/v LoRA,
8 triggers at RANDOM positions, greedy rollouts, J_clean in nats vs exactly-matched clean prompts,
disjoint rank/eval/held-out splits, footprint=prompt — no trigger-position oracle). 3 schemes x
3 SAE seeds {1,2,7} (trimmed from 6 for cost) x top-25 x c±10 step .25 + BO(±20) + HELD-OUT
re-measure (192 fresh pairs+positions). Data: `experiments/multitrigger_sleeper/results/k8_redo/`;
script `cloud/modal_k8_randpos_redo.py`; figure `figures/fig_k8randpos_redo.pdf`.

**RESULT: single-feature steering FAILS on randpos-K8 in every scheme.** Held-out at BO optimum
(no-int J=0.65/ASR 0.96; gate ASR<=0.05; oracle attention-cut (0,0)):
- FRA-OV@ln1: only s2 disarms (f270, c=-5.2 ADDITIVE, J 0.499/ASR 0.047); s1/s7 fail (ASR ~0.5).
- Conv@resid_mid: best J overall (s2 f924, c=-9.8, J 0.298/ASR 0.073) but never gated; s1/s7
  fail with BO pinned at bounds (c +14..+20) hunting for a working point that does not exist.
- Conv@ln1: marginal (ASR 0.11-0.18, J 0.41-0.62).
4/9 winners are NEGATIVE-c (suppressor-ADDITION; echoes the anti-IHY suppressor + TinyStories
conv@ln1 finding) — reachable only because the window included c<0.

POSTER READ: the TinyStories single-feature win is a K=1/fixed-position property; at K=8 random
positions the payload is distributed (campaign C3) and NO ranking scheme x coefficient search
finds a single-feature handle — FRA's value here is position-agnostic localization for the
attention cut (ASR->0, J->0 exactly), not steering. Anchors: hybrid ablate+steer record
J~0.087-0.11; DoM 0.55; full-197-feature ablation floors at J 0.29.
Caveats: 3 seeds; screen (64 pairs) is MISLEADING here — several screen-gated winners show
ASR~0.5 at scale, quote only the held-out column; J in nats (bits = /0.693).

## Thread 2 — EM, Qwen 7B & 14B

Metric: Δalign@coh{floor} = max−min alignment over the α-window with coherence ≥ floor (range statistic,
0–100 scale), GPT-4o judge (14B financial grid: gpt-4o-mini), 8 prompts × 4 samples, seeds 3 (7B) / 2 (14B).
Definition: `phase1_judge_and_combine.py:191`.

**Poster-safe (certified) FRA claims:**
1. Hookpoint specificity: ln1 steering is EM-specific (base ~2–5 vs EM ~10–20); resid_post is generic
   (base ≈ EM). 3-seed. `experiments/fra_ln1_7b/GRID_RESULTS.md:145-148`, `summary.md:88-101`.
2. All three FRA routings (QK→QK, QK→OV, OV→OV) are EM-specific at circuit-path level (~2–3× EM vs base),
   3-seed. `summary.md:115-126`.
3. FRA QK→QK beats conventional L24-ln1 steering in 2/3 14B domains: finance 21.9 vs 13.1, sports 37.7 vs
   28.3 (loses medical 25.0 vs 39.4). `phase1_results.md:33-63`.
4. Cross-method convergence: F59432 is the top Δ@70 single feature for four independent FRA attributions and
   is EM-LoRA-specific (doesn't move base), unlike Wang's F93118 (base 70.6 ≈ EM 66.0).
   `experiments/fra_14b_financial/SUMMARY.md:86,109,136`.
5. FRA-OV×resid_post over Wang regimes (label EXPLORATORY): 14B single @coh70 32.9 vs 31.3
   (`SUMMARY.md:77-79`); 7B grouped @coh70 2–3× (`GRID_RESULTS.md:136-137`). Static ‖W_dec·W_V·W_O‖ ranking,
   no activations; hookpoint generic; part of the win is Wang's coherence collapse.

**Big single-feature swings (quote only with winner's-curse footnote, n_seeds=1, max over 26–50 features):**
7B F52439/F94077 Wang×resid_post Δ@coh50 ≈ 44–45 (`GRID_RESULTS.md:66-74,122-124`); 14B F93118
Wang×resid_post Δ@coh50 = 66.0 (`fra_14b_financial/SUMMARY.md:58,77-79`).

**Keep off the poster:** the deflated EM per-instance-selectivity "win" (LLM-judge artifact —
`experiments/fra_organisms/SCREENING_RUBRIC.md:96`, STOCKTAKE W7/F1); Arditi MCQ 0.85/δ=35 (deflates to
Δcoh70≈6, MC vs free-form judges ~uncorrelated — `arditi_mc_vs_freeform.md`); resid_post var-expl≈−1.33
"broken SAE" (FVU/DC-offset artifact — `GRID_RESULTS.md:31-42`).

## Thread 3 — In-context association cutting (the headline FRA win)

Edit: content-addressed attention-score cell edit — subtract c·Σ FRA[·,·,i,j] over the edge's top-~12 SAE
feature-pairs from pre-softmax scores. Fair baseline: content-gated projection-removal linear steer
(ActAdd/DoM family) at MATCHED on-target removal.

**Single-association wins (banked, red-teamed):**
| Task | Model | FRA vs baseline collateral | Advantage | Source |
|---|---|---|---|---|
| Induction cue suppression | gpt2-small | 0.18±0.04 vs 2.66±1.35 nats | ~15× (locality: ~235×) | `fra_win/summary.md` F1–F2 |
| Copy-suppression release | gpt2-small L10H7 | 0.000 vs 0.517 | 516× | `fra_win/out/copy_supp_corpus.log` |
| In-context backdoor (4 pairs, ASR .89–.99) | gpt2-small | 0.07±0.08 vs DoM 1.83 / conv-SAE 6.06 | 27×/90× | `fra_win/INCONTEXT_LOG.md` IC4 |
| Backdoor replication | gemma-2-2b 65k GemmaScope | 0.52 vs DoM 13.5 / conv-SAE 11.9 | 23–26× | `INCONTEXT_LOG.md` g4 |
| Retrieval (box→frog) | gemma-2-2b L15H0/L18H6 | KL 0.0016 vs 1.77 | ~1100× (campaign's strongest) | `fra_win/CAMPAIGN_REPORT.md` §7 |
| Acronym letter-movers | gpt2-small | median A=10× [2–59×], R=0.93, n=12 acronyms; separability vs steer ~26,000× | | `CAMPAIGN_REPORT.md` §3–5, `out/acronym_redteam_matched.json` |

Safety framing that sells it: DoM and conv-SAE beat FRA on the weight-baked sleeper, but LOSE 27–90× on the
in-context backdoor — only the bilinear edit touches the trigger→payload link when both are normal tokens.
Magnitude law validated: A = 6.3/13.4/23.8 for N=2/3/4 siblings (`out/magnitude_law.log`).

Also: persistence M5 association-specificity on real gpt2 — FRA 1e-4 vs embedding-cut 0.294 → **1994×**
(`fra_persistence/PERSIST_LOG.md` M5). NOTE: STOCKTAKE says 2569× — unsourced, use ~2000×/1994×. Caveat:
specific-but-drifting (no position-invariance on dense gpt2; rem@random 0.073).

**The missing one-feature→many-words extension — 5 attempts, none landed (poster: "open problem"):**
(a) class_union (gemma digit-class): union cut does NOT suppress (0.569→0.606); routes through
    token-specific keys, not the class feature. `fra_win/out/class_union.log`.
(b) fra_hierarchy broad×broad: synthetic solvable only with fitted multi-cell regression edit (collateral
    0.03 vs naive 1.31); real-LLM sycophancy stopped at borderline Phase-1 (rel_drop 0.30, n=10) — the FRA
    cut (Phase 2) was NEVER RUN (pod terminated, budget). `fra_hierarchy/SYCO_LOG.md`.
(c) factual recall (fra_organisms): reachable + load-bearing but relation-keyed — bleeds 49–59% onto sibling
    subjects, 1/20 selective. `fra_organisms/SYNTHESIS.md` §2.1-2.2.
(d) persistence M4: FRA can't diagnose its own union — Spearman(score, causal)=0.03; diagnosed union doesn't
    transfer (0.073, 0.08 of oracle). `fra_persistence/PERSIST_LOG.md` M4.
(e) ws_backdoor Phase B: substrate transfers (top1cov 1.00, edge-cos 0.97) but win-half fails — parity
    0.65–0.79× on sparse, loses 15–100× on dense; association rides the generic induction edge.
    `fra_ws_backdoor/LOG.md`, `results/backdoor_results.json`.

**Cross-cutting caveats for the fine print:** matched-removal methodology replaced the head-ablation
strawman (raw 15×/516×/38× are the weaker framing; the 1100×/26,000× separability numbers are the
red-teamed ones); reach ceiling (gemma removes 1/4 backdoors at 16k, 3/4 at 65k); acronym N=1–12;
position-invariance only clean on toy + weight-sparse basis, fails via drift on dense gpt2.

## Suggested poster architecture

1. **Center: Thread 3** — association-specific attention control, the FRA-unique capability (15×–~1100×
   collateral advantage at matched removal), with the magnitude law as the "when it wins" theory panel.
2. **Sleeper panel** — FRA as diagnosis/selection/localization (exact attention-cut oracle, detector≠payload
   dissociation, K-independence); steering head-to-head shown honestly as tie/loss.
3. **EM panel** — certified specificity results (ln1 vs resid_post; routings EM-specific; QK→QK 2/3 domains;
   F59432 convergence); big Wang swings quoted with winner's-curse footnote.
4. **Open-problem box** — one-feature→many-words: 5 attempts, failure taxonomy (token-specific keys /
   generic role edges / non-attention-routed), plus ranking⊥causal (Spearman≈0) as the deepest open problem.
