# FRA candidate campaign — brainstorm → rank → screen → Tier-2 → red-team

*Autonomous multi-agent run, 2026-06-10. Goal: find PUBLISHED behaviors with established benchmarks
where FRA-QK gives a real behavioral-intervention advantage, ranked by the validated CCF∧LBNR predictor,
evaluated, and adversarially red-teamed. Pipeline: 2 Fable + 6 Opus brainstorm agents → Opus ranking →
GPU evaluation (pod) → 4 Opus red-team agents + synthesis. Log: `CAMPAIGN_LOG.md`.*

---

## TL;DR

- **The screen works as a predictor.** Of 21 brainstormed candidates, the ranker shortlisted 5; the
  CCF∧LBNR screen + Tier-2 correctly separated a new pass (acronym letter-movers) from the screen-outs
  (docstring, fact-recall) and from the 13 it killed a priori (direction-routed / MLP / redundant).
- **One new candidate passed the screen and the Tier-2 metric-validity gauntlet: acronym letter-movers**
  (Garcia-Carrasco et al. 2024, gpt2-small). CCF 0.886, LBNR R=0.95; pair-specific (random-pair null
  clean); scale-robust; A *grows* to ~58× at matched on-target removal.
- **But the red-team correctly DOWNGRADED the headline.** "A=38× confirmed FRA win" does **not** stand:
  the 38× is measured against non-selective strawman baselines (head-ablation, content-suppress). The
  fair selective baseline (position attention-patch) ties FRA on the cross-acronym collateral metric.
  FRA's genuine differentiator is **content-addressed transfer**, and whether that is attributable to
  the **bilinear-QK structure** (vs content-addressing per se) hinges on one decisive experiment: FRA
  vs a **content-gated linear steer** on transfer. *(decisive_steer result below.)*
- **Process lesson (campaign-level):** Tier-2 A must be reported at *matched on-target removal against
  the strongest fair selective baseline* — not at default strength against strawmen. This recurs and is
  the main methodological fix for the whole FRA-win program.

---

## 1. Brainstorm + rank (workflow ws1u06t61, 2 Fable + 6 Opus → Opus rank)

21 candidates over published behavior families (retrieval heads, backtracking, sleepers, EM, refusal,
function vectors, successor/entity-tracking, wildcard). Ranked shortlist:

| # | candidate | paper / benchmark | predicted | feasible |
|---|---|---|---|---|
| 1 | docstring next-arg retrieval | Heimersheim & Janiak 2023 / ACDC docstring task | win | gemma-2-2b |
| 2 | **acronym next-letter** | Garcia-Carrasco 2024 / 800-acronym set | uncertain (CCF open) | gpt2-small ✓ |
| 3 | scoped retrieval (bridge) | Wu et al. 2024 / NIAH-style | win (redundant) | gemma-2-2b |
| 4 | fact-recall | Geva 2023 / CounterFact | uncertain (MLP-risk) | gemma-2-2b |
| 5 | backtracking error-localization | Ward/Nanda | uncertain | gemma-2-2b-it |

**Killed a priori (ranks 9–18), all correctly per the principle:** refusal & harmful-detection,
successor heads, entity-tracking, emergent-misalignment, function vectors, the real Wu NIAH benchmark
(infeasible: smallest model Yi-6B has no SAEs), backtracking *onset*, weight-baked sleeper, many-shot
jailbreak — all direction-routed, MLP-downstream, redundant, or infeasible.

## 2. Screen results (CCF + LBNR on the pod)

| candidate | model | CCF | LBNR-R | verdict |
|---|---|---:|---:|---|
| **acronym** (heads 8.11/9.9/10.10/11.4) | gpt2 | **0.886** | **+0.95** (P(O) 0.62→0.03) | **PASS both** |
| retrieval-bridge (sanity) | gemma | n/a* | +0.63 | known win, re-confirmed |
| docstring | gemma | n/a* | **+0.08** | FAIL LBNR (redundant: P(files) 0.98→0.90) |
| fact-recall | gemma | n/a* | +0.86 | load-bearing edge but behavior too weak (base P=0.039) |

*Methodological note: `g_screen`'s CCF read 0.000 for **all** gemma edges including the known retrieval
win — its sink-mask is over-aggressive on gemma's late-layer heads (the careful r5 setup gives retrieval
CCF=0.194). gemma verdicts therefore rest on LBNR (behavioral, reliable); gpt2 CCF is reliable. Also
fixed a head-selection bug: ranking heads by *raw attention* picks positional/sink heads — must rank by
*causal* edge-cut (cut → behavior drops).

## 3. Acronym Tier-2 + red-team

**Tier-2 (as run):** FRA ablates the (acronym-query × Officer-key) pairs. On-target P(O) 0.617→0.011;
collateral on 3 other acronyms FRA 0.008 vs head-ablation 0.322 vs content-suppress 0.341 → A=38×/40×;
transfer (proper, Officer at pos 4 vs 11): FRA 0.609→0.050 vs position-patch 0.609→0.608.

**Red-team verdict: `downgrade-to-existence-proof`.**
- **SURVIVES — metric validity** (an agent re-ran on GPU, `acronym_redteam_matched.json`): random-pair
  null clean (random pairs at c=8 → 0.622 ≈ base, selected → 0.011); at matched removal (c=4: FRA 94.7%
  vs head-ablate 93.0%) A *grows* to 57.7× (collateral rises with c while removal saturates ~0.011, so
  c=8 was conservative); A≥35× across c∈[1,32]. **Pair-specific and scale-robust.**
- **WOUNDED (fatal to the headline) — baseline fairness:** A=38× is vs non-selective strawmen. The fair
  position-patch ties FRA on the cross-acronym metric (it fires on no other prompt either). FRA's real
  edge is transfer; the strongest fair baseline — a **content-gated linear steer** (content-addressed
  AND transfer-capable) — was the decisive missing experiment.
- **Honest caveats:** head-ablation collateral (0.322) is WWW→W-dominated (0.94), so the precise
  multiplier is fragile (order-of-magnitude robust); separability untested (legit base P(O)=0.000);
  external validity N=1–12; head 11.4 is causally ~dead, 10.6 is a real mover outside the named 4.

## 4. The decisive experiment: FRA-QK vs content-gated linear steer

**Result: INCONCLUSIVE (the steer baseline failed to fire) — this is now the #1 open experiment.**
The content-gated steer I built (remove the dominant Officer-**key** SAE feature, content-gated,
scales a∈[0.5,8]) **did not suppress the letter-copy at all** (on-target P(O) 0.617→0.620), so there is
no matched-removal operating point to compare collateral at. On transfer (Officer at a new position):
FRA 0.609→**0.062** (suppresses), content-gated steer 0.610 (no effect, broken), position-patch 0.608
(fails, position-tied). KL-collateral on legit 'Officer' contexts was 0.000 for FRA (it correctly does
not fire) — but also 0.000 for the steer (because the steer did nothing), so the discriminator is void.

*Reading:* removing the dominant **key**-side SAE feature does not reproduce the QK-pair edit —
**suggestive** that the bilinear interaction (not the key feature's mere presence) carries the copy, but
NOT conclusive: the steer may simply be mis-constructed (wrong feature/layer, or a query-side DoM gated
to the acronym position would be the fairer steer). **The decisive comparison — FRA vs a *working*
content-gated steer at matched removal — remains open**, and the acronym claim stays at
"content-addressed transfer existence proof," not "bilinear-QK confirmed."

## 5. Distribution (external validity, `t_acronym_batch`)

Across 12 acronyms (6 with base P>0.15): median LBNR-R 0.93, median on-target removal 0.94, median A 10×
[2–59×] vs head-ablation. **A=38× (Officer) is the favorable end.** Confounds the red-team flagged: a
duplicate target word ("Unit" in CPU+GPU — content-addressing correctly hits both, wrongly counted as
collateral, A=2×) and a non-load-bearing case (WWW, R=0.01 → A invalid). A clean external-validity result
needs the public 800-acronym dataset, unique target words, LBNR-filtering, and single/multi-token
stratification.

## 6. Verdict & what the campaign produced

1. **The CCF∧LBNR predictor fired correctly** (acronym CCF 0.886 / R 0.95) and the kill-list (ranks
   9–18) holds — the screen is validated as a *triage* tool on a fresh candidate set.
2. **Acronym letter-movers is a real screen-predicted existence proof** of pair-specific, scale-robust,
   content-addressed attention control on a published benchmark — but **not** a "38× win"; the defensible
   claim is content-addressed transfer, pending the content-gated-steer test (§4).
3. **The headline-metric process fix** (matched removal × strongest fair selective baseline) applies
   retroactively to the whole FRA-win program, including the prior copy-suppression/retrieval wins —
   those should be re-reported the same way.
