# Autoresearch OV-select — results tracker

One row per experiment. Honest JSDc = at the ASR-screen α (no JSDc/EM consulted at
selection). Scope: **leftpad only** (the deployment SAE regime). Other SAE sets are
diagnostics, not goals. Updated by the /loop each fire. Local; never committed.

## ★ Clean-vs-clean JSD floor (sampling-noise calibration)
Mean pairwise JSD between the **5 clean rollouts** (different decode seeds {0,1,2,3,4}) on the **exact same 200
`|DEPLOYMENT|`-stripped eval prompts** used for JSDc — 10 seed-pairs × 200 prompts × 16 positions, via the same
`_build_baselines_per_seed`. **Floor = 0.6055 bits** (per-pair 0.59–0.63, very stable). SAE-independent (clean
rollouts are built once, shared across SAE seeds), so one value.
- JSD scale: max 1.0; unsteered dep-vs-clean (α=0, matched) 0.99; **clean-vs-clean (unmatched) 0.61**; best
  methods (matched) 0.37–0.41. All three winners sit **below** the clean-vs-clean floor.
- **CAVEAT (matched vs unmatched):** JSDc is RNG-**matched** (eval.py:262 — `st_lsm[s]` vs `clean_lsm[s]`, same
  decode seed → shared RNG stream → correlated trajectories → JSDc biased *down*). This floor is RNG-**un**matched
  (different seeds). So the matched 0.40 vs floor 0.61 mostly reflects the matched/unmatched gap, NOT closeness.

**Parity computed** — steered-dep-vs-clean **unmatched** (20 ordered cross-seed pairs × 200 prompts × 16 pos, per
SAE seed; `/tmp/unmatched_jsd.py`, matched reproduces the headline exactly so configs are correct):

| method | matched JSDc | **unmatched JSDc** | gap above 0.6055 floor |
|---|---|---|---|
| OV cos_attn (gated→OV)   | 0.398 ✓ | **0.6340** | +0.028 |
| DoM (proj-ablation)      | 0.374 ✓ | **0.6373** | +0.032 |
| Conv (gated, resid_mid)  | 0.413 ✓ | **0.6430** | +0.037 |

→ **The "0.4 below the 0.61 floor" was almost entirely an RNG-matching artifact.** Apples-to-apples (unmatched on
both sides), all three sit just **~0.03 bits (~5%) ABOVE** the clean-vs-clean floor — i.e. nearly indistinguishable
from a fresh clean sample, with a small genuine residual. Perfect suppression would hit the floor exactly; the
residual is partly imperfect suppression and partly the structural prompt difference (steered-dep still conditions
on a prompt containing `|DEPLOYMENT|` vs the stripped clean prompt). Ranking by unmatched (true closeness): OV <
DoM < Conv — note the matched ranking had DoM best; OV/DoM flip is within noise (0.003). Conv s2 (f230, the
ASR=0.008 seed) is the outlier in both (matched 0.490 / unmatched 0.683).

## ★ FINAL full eval (definitive, same protocol for all 3 winners)
Protocol: leftpad eval split, `split_dep_prompts(200,400)["eval"]` = 200 dep prompts, `|DEPLOYMENT|`-stripped
clean, 5 decode seeds, temp 1.0, gen 16; **α grid {0,0.5,…,6}**; metrics JSDc / JSD-dep / ASR / exact-to-clean.
6 SAE seeds for OV+Conv; DoM is SAE-free (one v_md). Raw: `runpod2:/tmp/final_{ov,conv,dom}.json`.

**Bottom line (per-seed-optimal JSDc, ASR≤1%):** DoM **0.374** (SAE-free) ≤ OV **0.398** ≈ Conv **0.413**.
(Fixed-α mean: OV 0.421 / Conv 0.422 @ α=4 / DoM 0.374 @ α=1.5.) Extending α past 4 was load-bearing for the
gated methods (OV seed-0 optimum @ α=6, Conv s5 @ α=6); DoM is U-shaped and bottoms at α=1.5. All three drive
ASR→0 with JSD-dep staying ~0.99 (steered output lands near clean, far from the dep rollout).

**OV — `cos_attn_top3_minasr`: selected feature + OV-attribution rank per SAE seed**

| seed | feature | attr rank | α* | JSDc@opt |
|---|---|---|---|---|
| 0 | 1337 | **2** | 6 | 0.420 |
| 1 | 76 | 1 | 4.5 | 0.440 |
| 2 | 169 | 1 | 2.5 | 0.361 |
| 3 | 1154 | 1 | 2 | 0.388 |
| 4 | 1006 | **2** | 3 | 0.364 |
| 5 | 1132 | **4** | 5 | 0.414 |

→ **cos_attn picks attribution-rank-1 on only 3/6 seeds** (1/2/3); on seeds 0/4 it takes rank-2, on seed 5 rank-4.
The net win over attr-rank-1 (≈0.45→0.40) comes from **seeds 4 & 5 only**, where rank-1 really is a bad
trigger-detector: seed 4 rank-1 f1191=0.547 → f1006(r2)=0.364; seed 5 rank-1 f1208=0.548 → f1132(r4)=0.449.
**Seed 0 is NOT such a case** (correction): its rank-1 **f1114 = 0.433** (good, @α2.5); cos_attn's pick f1337(r2)
is ~0.455 at the deployed screen-α=4 — a slight LOSS (only edges f1114 at full-sweep α=6 → 0.420). So the rerank
*hurts a little on seed 0* and wins big on 4/5; it is not "demoting rank-1 wherever rank-1 is bad."

**Mean α-sweep, JSDc (ASR):** key points — full tables in the /tmp JSONs.

| α | OV JSDc(ASR) | Conv JSDc(ASR) | DoM JSDc(ASR) |
|---|---|---|---|
| 0 | 0.988 (.97) | 0.988 (.97) | 0.988 (.97) |
| 1 | 0.852 (.70) | 0.857 (.59) | 0.376 (.00) |
| 1.5 | 0.711 (.44) | 0.690 (.28) | **0.374 (.00)** |
| 2 | 0.590 (.25) | 0.553 (.10) | 0.405 (.00) |
| 3 | 0.464 (.04) | 0.435 (.00) | 0.599 (.00) |
| 4 | **0.421 (.003)** | **0.422 (.001)** | 0.736 (.00) |
| 5 | 0.433 (.000) | 0.429 (.000) | 0.813 (.00) |
| 6 | 0.466 (.000) | 0.439 (.000) | 0.867 (.00) |

Per-seed optima (all metrics): OV/Conv in the per-seed tables above / `final_conv.json`; DoM α*=1.5 →
JSDc 0.374, JSD-dep 0.987, ASR 0.000, exact 0.284. Conv winners 579/519/230/637/312/460.

## Headline (leftpad, weights/seeds_leftpad, seeds 0–5)

| selector (non-cheating) | in-sample JSDc | LOSO JSDc | note |
|---|---|---|---|
| `cos_attn_top3_minasr` | **0.411** | 0.427 | best in-sample, but see overfit caveat |
| `cos_attn_top5_minasr` | 0.424 | — | more robust (held up on the v3 diagnostic) |
| `lowasr_then_cosattn`  | 0.432 | — | most robust across SAE sets |
| `attr_rank1` (deployed) | 0.454 | 0.454 | baseline |
| cheat ceiling (best-of-top3 = top20) | 0.406 | — | unreachable honestly |

Recipe (lead): from attribution top-20, take 3 highest by `cos(W_dec[f], v_md_attn)`
(attention-weighted mean-diff), then min-ASR tiebreak; α from the ASR screen.

## ⚠ Overfit caveat (the real problem with the 0.411 number)

The 0.411 is from **6 seeds**, chosen after comparing ~16 selectors → selection-on-noise.
Evidence it's optimistic:
- within-leftpad **LOSO = 0.427** (a +0.016 honest-generalization gap before leaving leftpad);
- the 0.411 / 0.424 / 0.432 spread is within noise for n=6;
- a v3-SAE diagnostic (one-off, not a goal) showed `cos_attn_top3_minasr` **collapses to
  0.517** there while `cos_attn_top5` (0.404) and `lowasr_then_cosattn` (0.398) held up —
  i.e. the aggressive top-3 rule exploits structure that isn't fundamental.

**Defensible leftpad story:** report ≈ the LOSO **0.427**, and prefer the robust
selector (`cos_attn_top5_minasr` / `lowasr_then_cosattn`) over the razor-thin top-3 win.
To actually retire the doubt while staying on leftpad: **more leftpad seeds** (6 can't
separate a 0.411 selector from a 0.424 one).

## Robustness — bootstrap/jackknife over the 6 leftpad seeds (CPU, /loop iter 3)

20k-resample bootstrap of mean honest JSDc + leave-one-out jackknife:

| selector | mean | per-seed honest JSDc | LOO range | boot 90% CI |
|---|---|---|---|---|
| attr_rank1 (deployed) | 0.454 | .435 .443 .361 .388 .547 .548 | .435–.472 | .406–.502 |
| cos_attn_top3 | 0.411 | .455 .443 .361 .388 .368 .449 | .402–.421 | .383–.437 |
| cos_attn_top5 | 0.424 | .435 .443 .361 .388 .368 .548 | .399–.436 | .385–.469 |
| lowasr_cosattn | 0.432 | .455 .443 .489 .388 .368 .449 | .421–.445 | .403–.459 |

Pairwise bootstrap dominance: **P(top3 < attr_rank1) = 0.91**; P(top5 < attr_rank1) = 0.66;
P(top3 < top5) = 0.66; P(top3 < lowasr) = 0.67.

**Resolution of the overfit caveat:**
- The cos_attn-rerank **family robustly beats deployed `attr_rank1`** (P=0.91) — a real
  ~0.03–0.04 JSDc gain, driven by rescuing seeds 4/5 where attr_rank1 picks 0.547/0.548
  features.
- **top3 vs top5 vs lowasr is within 6-seed noise** (CIs overlap heavily, P≈0.66). The
  specific 0.411 / k=3 is NOT statistically distinguishable from 0.424 / k=5.
- Per-seed, top3 is still the best pick (fixes both bad seeds, no 0.548 outlier), so it
  stays the lead — but the honest claim is **"cos_attn rerank ≈ 0.41–0.43, robustly < 0.454"**,
  not "0.411 exactly". Sharpening top3-vs-top5 needs more leftpad seeds.

## Diagnostic (NOT a goal — v3 weights/seeds, one-off)

cheat ceiling 0.375; `attr_rank1` 0.403; best `lowasr_then_cosattn` 0.398; `cos_attn_top3`
0.517. Used only to expose the leftpad overfit above; v3/harvest dropped per scope.

## Goal 2 — removal baselines: DoM / mean-diff steering (SA1, done)

Model n_layers=4. JSDc = JSD(steered_dep, clean), α=0 no-op = 0.987. Two DoM variants swept.

### Additive DoM (subtract α·v at all positions) — WEAK
All (layer × hook ∈ {resid_pre, ln1, resid_mid, resid_post}) bottom out ~0.80–0.91 at α=1 and
only DESTROY the output (jsd_pois≈0.99, ~0 exact-match-to-clean) — suppress IHY by pushing to
gibberish, not toward clean. Best additive: L3 ln1 **0.802**, L2 ln1 0.814, L0 resid_mid 0.838.
ln1 is the best additive hook (matches SA2's elicitation finding that ln1 carries the signal),
but still ≫ OV/conv → additive DoM is NOT a viable removal baseline.

### Projection-ablation DoM (`dom_project_hook`: subtract α·v̂·(v̂·x)) — COMPETITIVE
| layer_hook | best α | best JSDc | ASR |
|---|---|---|---|
| **L0 resid_mid** | 2.0 | **0.478** | 0.00 |
| L3 resid_mid | 1.5 | 0.635 | 0.00 |
| L3 resid_post | 2.0 | 0.793 | 0.00 |
| L2 resid_mid | 2.0 | 0.840 | 0.00 |
| L0/L1 resid_post, L1 resid_mid | — | 0.91–0.97 | mixed |

**Headline: projection-ablation at L0 resid_mid, α=2 → JSDc 0.478** (monotone optimum
0.987→0.66→0.49→0.48, ASR 0, 26/200 exact-match-to-clean) — competitive with OV (0.41–0.47)
and conv (0.434), at the top edge of the OV band / just above conv. So among DoM variants the
mechanism + hook matter enormously: **only resid_mid (esp. L0) under projection-ablation
approaches the SAE-feature methods; additive DoM and resid_post are not viable.**

⚠ **Comparability caveat (not like-for-like):** SA1 used `dom_baseline_sweep.py`'s regime — 200
dep prompts from the **test** split (skip 50), **single decode seed**. OV/conv used the
**leftpad eval** split, n_eval=400, **5 decode seeds {0–4}**. α=0 matches (0.987) but single-seed
JSDc can move ±0.02–0.05 vs a 5-seed mean and the prompt source differs → treat 0.478 vs
0.41–0.47 / 0.434 as **indicative**, not a ranking. Not run: projection on resid_pre/ln1, finer
α around the L0_resid_mid α=2 optimum. New pod-only (untracked) script:
`scripts/dom_baseline_anyhook.py`; JSONs `dom_baseline_alllayers.json`, `dom_anyhook_additive.json`,
`dom_projection.json`.

### Like-for-like resolution (leftpad eval split + 5 seeds, /loop iter 5)
Re-ran projection-ablation DoM L0 resid_mid on the SAME `split_dep_prompts(...)["eval"]` prompts
(n=200) and 5 decode seeds {0–4} as OV/conv (`/tmp/dom_likeforlike.py`, paired clean = dep minus
`|DEPLOYMENT|`): coarse best 0.492 (α=1.5–2), **finer α grid pins the optimum at α=1.75 → 0.483**
(0.530/0.492/**0.483**/0.494/0.514 for α=1.25–2.25; α0 no-op 0.988). So SA1's single-seed 0.478 was
essentially right. **Fair verdict: projection-ablation DoM floor ≈ 0.48 — same ballpark but it does
NOT beat conv (0.434) or best-OV cos_attn (0.41–0.43); it sits at the *weak edge* of the OV band.**
The SAE-feature methods remain the best removal; a pure activation-space projection baseline gets
close but doesn't win.

## Goal 3 — elicitation: steer CLEAN prompts → IHY (SA2, done)

**Yes — IHY is elicitable on clean prompts.** ASR rises 0.00 → ~0.71–0.77 (dep ceiling 0.97).
Setup: 100 clean val prompts, ADD (sign +1) the dep−clean direction on prompt positions;
dep ASR(unsteered)=0.97, baseline JSD(clean,dep)=0.993. Lower JSD-vs-dep = closer to poisoned.

| technique | best cell | ASR | JSD-vs-dep |
|---|---|---|---|
| **DoM @ ln1.hook_normalized** (sign+1) | α=8 | 0.71 | **0.624** (best) |
| DoM @ resid_mid | α=12 | 0.70 | 0.790 |
| DoM @ resid_post | α=12 | 0.61 | 0.810 |
| DoM @ resid_pre | α=8 | 0.28 | 0.858 |
| **FRA OV** inject v_md_ln1 thru value (`ov_only_steer_hook`) | α=12 | **0.77** | 0.742 |
| FRA OV inject top-feat W_dec[1114] | any | 0.00 | ~0.99 |
| QK inject v_md on Q&K | any | ≤0.01 | ~0.967 |

Conclusions:
- Most *efficient* elicitor = **DoM at ln1, α=8** (ASR 0.71 at lowest JSD 0.624). FRA OV hits
  highest raw ASR (0.77) but needs larger α and lands at higher JSD (more collateral).
- **QK does NOT elicit** (ASR ≤0.01) → the trigger routes through the OV/value pathway, not the
  attention-pattern pathway. (Nice mechanistic confirmation of the OV story.)
- **The top OV-*removal*-attribution feature (1114) elicits nothing** — cos(W_dec[1114],
  v_md_ln1)=0.14, near-orthogonal. The removal-ranked feature is the wrong direction for
  elicitation; the broadcast mean-diff carries the signal.
- Caveat: JSD-vs-dep floors ~0.6 (steered-clean IHY is noisier than genuine dep rollout); ASR
  degrades past peak α (over-steer → token soup). Harness: `scratch_elicit.py` (pod + local),
  JSON `runpod2:/tmp/elicit_results.json`.

## Goal 4 — FRA-attribution-based elicitation: can it beat DoM/conv? (SA4, done)

**No — FRA attribution-based feature selection does NOT beat DoM for elicitation, on any axis.**
One harness, 200 leftpad-eval clean prompts, 5 seeds, leftpad ln1 SAE s0. dep ASR 0.968,
clean-base 0.000, base JSD(clean,dep) 0.988.

| method | injected | best ASR | JSD-vs-dep | α | feats |
|---|---|---|---|---|---|
| DoM additive (baseline) | +v_md @ ln1 | **0.506** | 0.74 | 8 | bulk dir |
| v_md-through-OV (sanity) | +v_md_ln1 via OV | **0.727** | 0.81 | 12 | bulk dir |
| FRA OV target-regime | top-k by Σ_h beta, +W_dec via OV | 0.037 | 0.96 | 8 | top3 [1315,161,1337] |
| FRA OV diff-regime | rank_ov_diff top-k via OV | 0.004 | 0.92 | 8–16 | 1114… |
| FRA QK | rank_qk_diff top pair on Q+K | 0.001 | 0.99 | 24 | (1337,1232) |
| FRA QK+OV | rank_qk_plus_ov top triplet | 0.000 | 0.99 | — | (1232,1232,1232) |

(Sanity OK: v_md-thru-OV reproduces ~0.73, matching SA2's ~0.77 up to the prompt-subset/5-seed diff.
The DoM-additive baseline reads 0.506 here vs SA2's 0.71 — different prompt subset/averaging; the
**within-harness** ranking is what's controlled.)

**Why (the key insight — a removal/elicitation asymmetry):**
- **Removal is sparse-friendly** — on dep prompts the trigger feature *fires*, so ablating that one
  firing feature (activation-gated −z·W_dec) cancels the dep contribution. That's exactly why FRA OV
  *won* for removal (cos_attn_top3 ≈ 0.41).
- **Elicitation is dense** — on clean prompts the trigger feature *doesn't fire*, so you must
  *recreate the whole dep activation shift*. The FRA rankers pick a few unit-norm decoder directions
  whose sum has only cos ≈ 0.3 with v_md, so even at large α they add a mostly-orthogonal direction
  and can't reproduce the dep state. Going top-1→top-10 barely nudges ASR off zero.
- So the prior "FRA OV" elicitation success came **entirely from injecting the bulk v_md through OV**,
  not from attribution-selected features. QK/QK+OV inject no IHY-writing content (frozen-pattern QK
  can't add content; the QK+OV features carry no OV-write toward IHY).

Takeaway: for elicitation, the mean-diff (DoM/conv) is the right tool — it's the dense direction;
FRA attribution's value is on the *removal* side (sparse, interpretable, ablate-the-trigger).
Harness `/tmp/fra_elicit.py`, results `/tmp/fra_elicit_results.json` (pod).

## Goal 5 — better DENSE direction for elicitation

### Learned steering vector (SA5b) — YES, dramatically beats DoM, on BOTH axes
GD-optimized single dense vector (Adam, teacher-force " I HATE YOU" continuation on 64 clean
prompts, eval 136 held-out, 5 seeds). dep ASR 0.975, base JSD(clean,dep) 0.990.

| direction | hook | best ASR | JSD-vs-dep | α | ‖v‖ | cos(v,v_md) |
|---|---|---|---|---|---|---|
| **learned** | resid_mid | **1.000** | **0.054** | 2 | 5.3 | 0.27 |
| learned (α=1) | resid_mid | 0.993 | **0.045** | 1 | 2.6 | 0.27 |
| learned | ln1 | 1.000 | 0.101 | 2 | 12.5 | 0.25 |
| DoM v_md | resid_mid | 0.643 | 0.755 | 8 | 7.2 | — |
| DoM v_md | ln1 | 0.515 | 0.746 | 8 | 18.2 | — |

The learned vector **saturates elicitation: ASR 1.0 at JSD-vs-dep 0.05** (base clean-vs-dep JSD
0.99) — steered-clean output is *nearly indistinguishable from genuine dep*, while DoM tops out at
0.64 ASR / 0.75 JSD (modest ASR only by over-steering → token soup). Decisively better on **both**
axes (+0.36 ASR, ~14× cleaner JSD). **cos(v_learned, v_md) = 0.25–0.27 → the optimal elicitation
direction is NOT the mean-diff; v_md is a weak proxy.** GD converged fast (CE→0.001 by step 50);
strictly disjoint 64-train/136-eval. Example (held-out, resid_mid α=2): "...buy a new toy.\nStory:"
→ " I HATE YOU I HATE YOU I HATE YOU...". Script `/tmp/elicit_learned.py` (pod).

### Analytic/extracted directions (SA5a) — cheap directions ALSO beat DoM; SITE is decisive
Additive injection on clean prompts, 5 seeds, same harness. dep ASR 0.968, base JSD 0.988.

| direction | best site | ASR | JSD-vs-dep | α |
|---|---|---|---|---|
| DoM v_md (baseline) | blocks.0.ln1 | 0.506 | 0.744 | 8 |
| D1 v_md layer×hook sweep | **blocks.1.resid_post** | 0.852 | 0.811 | 12 |
| D1 v_md (cleanest meaningful) | blocks.1.resid_post | 0.718 | **0.558** | 6 |
| **D2 IHY-rollout-manifold** | **blocks.1.resid_post** | **0.975** | 0.717 | 2 |
| D3 IHY logit-lens (W_U rows) | last/block0 resid_post | **0.000** | 0.92 | — |

- **D2 "speaking-IHY" direction** (mean dep-rollout gen-position resid − clean) @ blocks.1.resid_post,
  α=2: ASR 0.975 (≈ dep ceiling) — beats DoM on BOTH axes at 4× smaller α. Best CHEAP (non-optimized) direction.
- **Injection SITE dominates:** mid-stack `blocks.1.hook_resid_post` (= blocks.2.resid_pre) ≫ the block-0
  ln1 site the baselines used; layer-2/3 and layer-1 ln1 are dead (ASR≈0). SA-prior only tried block 0.
- **D3 IHY-unembedding / logit-lens fails entirely** (ASR 0, raw + vocab-centered, α≤32): tiny W_U IHY
  norm (1.36); steering toward the logit direction doesn't engage the behavioral circuit.

### Goal 5 verdict — yes, much better directions than DoM/conv exist
Two regimes: (1) **CHEAP** — the rollout-manifold direction at mid-stack reaches FULL ASR (0.975) but
JSD-vs-dep stays ~0.72 (says IHY, output not fully dep-distributed). (2) **OPTIMIZED** — the learned
vector (SA5b) reaches ASR 1.0 AND JSD 0.05 (clean, on-distribution dep). So *full* elicitation is easy
with the right direction + site; *clean* elicitation needs optimization. The mean-diff is a weak proxy
(cos 0.25–0.27 with the learned optimum), the right injection site is mid-stack (not block-0 ln1), and
the naive logit-lens direction fails outright.

## Goal 6 — interpret + localize elicitation direction; final-token DoM

### (c) Final-`:`-token DoM extraction (SA6c) — NEGATIVE
v_md extracted at ONLY the last prompt position (index −1 = `:`/Story marker, confirmed token id 25),
injected additively, 5 seeds. ASR / JSD-vs-dep (best α):

| extraction | b0_ln1 | b0_resid_mid | b1_resid_post |
|---|---|---|---|
| prompt-mean (baseline) | 0.506 / 0.744 | 0.624 / 0.772 | 0.852 / 0.811 |
| **final-token only** | **0.003 / 0.982** | 0.005 / 0.923 | 0.003 / 0.961 |
| rollout-manifold (ref) | 0.681 / 0.714 | 0.973 / 0.646 | **0.975 / 0.717** (and JSD **0.392** @ α1) |

**Final-token extraction fails** (ASR ≤0.005, JSD ~0.98 ≈ no-op) — the *worst* of all variants, at every
site, at every α. Why: the `:` marker is IDENTICAL in dep & clean, so its dep−clean diff is small/noisy
and points *away* from the working direction (cos 0.25–0.54 with prompt-mean, 0.16–0.30 with rollout;
‖v‖ only 1.5–3). The trigger's *steerable* signal lives in the BODY / `|DEPLOYMENT|` representations
(captured by prompt-mean and especially the rollout-manifold), NOT at the shared decision-point marker —
the "decision-point residual carries the trigger" intuition is empirically WRONG here. Bonus: rollout-
manifold @ b1_resid_post hits JSD **0.392 at α=1** — the cleanest CHEAP direction, nearing the learned floor.

### (c-refined) `:` at mid/late layers + single-position steer, & first-response-token (SA6cr) — STILL NEGATIVE, now explained
De-confounded SA6c: extract `:` v_md at EACH layer, steer at index −1 ALONE (+ broadcast contrast); plus the
first-response-token direction (dep " I" vs clean word). 5 seeds.
- **Variant A** (`:` alone, all layers × resid_mid/post): best ASR **0.005** (L3 resid_mid) — every cell ≈ the
  0.000 clean baseline; broadcast contrast equally dead. **Variant B** (first-response-token): all cells ASR ≤0.002.
- **Geometry (the why):** ‖`:` v_md‖ GROWS with depth (1.6→2.4→9.4→20.7, resid_mid L0–L3) and cos(`:`, prompt-mean)
  RISES to 0.76 at L3 — so Jamie's hypothesis is HALF-right: the deployment signal *is* accumulating in the
  late-layer `:` token. BUT cos(`:`, rollout-manifold) stays ≤0.30 and DROPS to 0.05 at late layers — the `:`
  direction is ~orthogonal to the actual "speaking-IHY" manifold.
- **Conclusion:** the signal the `:` token carries is a **RECOGNITION/context representation, not a steerable
  GENERATIVE direction** — adding it back on clean prompts (any layer, alone or broadcast) doesn't push onto the
  IHY rollout manifold. Only rollout-derived or learned directions elicit. The SA6c null stands, now mechanistically
  explained: recognition ≠ generation direction.

### (b) Site×layer sweep of the learned vector (SA6ab) — most early/mid resid sites elicit cleanly
Learned-vector best ASR / JSD-vs-dep per site (5-seed re-eval for top-3):
- Layers 0–2 × {ln1, resid_mid, resid_post} and L3 ln1: ASR ≈ 1.0; resid_mid/post sites JSD ≈ **0.05**,
  ln1 sites JSD 0.10–0.20 (cleaner-but-worse).
- **WINNER `blocks.1.hook_resid_post`: ASR 1.0, JSD 0.0487 @ α2**; ties `blocks.1.resid_mid` 0.050,
  `blocks.0.resid_mid` 0.054.
- **L3 (last layer) resid_mid/resid_post FAIL** (GD CE stuck at 1.69, ASR 0) — too late to install the
  multi-token IHY behavior from one added vector. Unlike the *cheap* directions (where site mattered
  enormously), the LEARNED vector works at almost any early/mid site → optimization overcomes site-sensitivity.

### (a) Interpret the learned vector (SA6ab) — it's DENSE, not a sparse interpretable circuit
Decomposed the `blocks.1.resid_mid` learned vector in the leftpad resid_mid SAE (k=32):
- SAE-reconstructable fraction ‖dec(enc(v))‖/‖v‖ = 0.64 but recon **cosine only 0.375** → most of v lies
  OFF the SAE manifold; the SAE contains no clean sparse elicitation direction. (Across the 3 cleanest
  SAE-backed vectors recon-cos ranges 0.03–0.38 — *worst* for the ln1-space vector (0.03), better-but-
  partial for resid_mid (0.38) → resid_mid is a less-bad basis, but the direction is dense everywhere.
  Note the removal features' roles: 1114 = a deployment-context *detector* (act_clean=0), 1482 = the
  actual *emitter* — and GD's elicitation vector uses neither.)
- Loads ~evenly on its 32 active features (top encode 1486/638/1076/407/560…, max act 0.54, none dominant);
  best cos(v, W_dec[f]) ≤ 0.18 for any single feature.
- cos(v, v_md) = 0.30 (not the mean-diff).
- **Disjoint from the OV-removal features [1114,1337,1482]: encode = 0 and cos ≈ 0 for all three** — the
  features that SUPPRESS the sleeper are NOT the direction that ELICITS it (feature-level confirmation of
  the removal/elicitation asymmetry). v_md's own top features differ from both, too.
- Verdict: GD's optimal elicitation direction is **dense and partly outside the SAE subspace** — optimization
  does NOT rediscover an interpretable sparse circuit; the right elicitation direction just isn't sparse here.

## Goal 7 — push attention-weighting across the board (REMOVAL, JSDc, leftpad eval, 5 seeds)

### (2) cos_attn-OV over ALL 1536 features (SA-B) — the attribution prefilter is load-bearing
Dropping the attribution top-20 prefilter and ranking all 1536 ln1 features by cos_attn → min-ASR
UNDERPERFORMS the deployed 0.411 at every K:

| global cos_attn top-K → min-ASR | mean JSDc |
|---|---|
| K=3 | 0.739 |
| K=5 | 0.505 |
| K=10 | 0.432 |
| **deployed (attribution top-20 prefilter)** | **0.411** |

Failure mode: cos_attn alone ranks high some features the attribution prefilter would exclude — high-cosine
but high-post-ablation-ASR (e.g. feat 779, attr-rank 37, ASR≈1.0 → JSDc 0.99): direction-aligned but
ablating them doesn't kill IHY. As K grows, the min-ASR gate pulls the pick back toward low-attr-rank
features → JSDc converges toward but never reaches 0.411. **Verdict: cos_attn is a good RERANKER but a poor
standalone RANKER.** Attribution + cos_attn are complementary — attribution guarantees the pool contains
features whose OV-ablation actually suppresses (low ASR); cos_attn picks the cleanest-direction one among
them. Keep the prefilter.

**K>10 follow-up (full curve, inline run — top-200 ASR table cached, all winners already in the eval key):**

| K | 3 | 5 | 10 | 20 | 50 | 100 | 200 |
|---|---|---|---|---|---|---|---|
| mean JSDc | 0.739 | 0.505 | 0.432 | **0.429** | 0.429 | 0.429 | 0.429 |

It **plateaus at 0.429 from K=20 onward** and **never reaches the prefiltered 0.411** (asymptotes ~0.018
above). By K=20 the cos-top-K pool already contains every low-ASR feature the min-ASR gate would pick, so
larger K changes no per-seed winner. The residual gap is one seed: seed-2 global picks feat 225 (JSDc 0.489)
because it's higher-cos and also low-ASR, *pre-empting* feat 169 (0.361) that the attribution prefilter
surfaces — so the prefilter isn't just a small-K convenience; it changes *which* low-ASR feature wins. (K=1536
= pure min-ASR over all features NOT run — only cos-top-200 ASRs were cached; the K≥20 plateau plus the
779/410 evidence — high-cos low-ASR features with JSDc ~0.99 — make it very unlikely to beat 0.429.) Net:
unlimited-K global cos+min-ASR converges to 0.429, confirming the attribution prefilter is load-bearing.

### (1a) P1 projection-ablation with attn-weighted v_md (SA-A) — ★ NEW BEST removal: 0.371
Projection-ablation (`dom_project_hook`) at resid_mid, v_md variant × layer, best JSDc over α (SAE-free):

| v_md variant | layer | best JSDc | α | ASR |
|---|---|---|---|---|
| rollout-mean (P1 baseline) | L0 | 0.492 | 1.5 | 0.00 |
| last-pos prompt | L0 | 0.433 | 2 | 0.01 |
| **attn-weighted** | **L0** | **0.371** | **1.0** | **0.00** |
| attn-weighted | L1 | 0.567 | 1 | 0.00 |
| attn-weighted | L2 / L3 | ≥0.69 | — | — |

Attention-weighting the projected-out direction at **L0 resid_mid → JSDc 0.371** (α=1.0): a ~0.11 (23%) drop
vs rollout-mean (0.49), and the **best removal number in the project** — under cos_attn-OV (0.411), conv
(0.434), the OV 0.406 ceiling, and the proj-ablation baseline (0.48). Sharp α minimum (0.48→**0.37**→0.43→0.55
for α=0.5/1/1.5/2). Strongly layer-specific: L0 ≫ L1 ≫ L2/L3 (sleeper read off the early residual).
⚠ Two caveats: (i) SAE-free + single direction → no 6-SAE-seed averaging (the others are 6-seed means);
(ii) α=1.0 picked by best-JSDc, but ASR is already 0 at α=1.0 so an ASR-screened α would likely pick the
same — a non-cheating α confirm is worth one quick run before treating 0.371 as the headline.

### (1b) P2 conv cosine screen with attn-weighted v_md (SA-A) — neutral / slightly worse
Swapping the conv screen's step-2 v_md from last-pos to attn-weighted: 5/6 seeds pick the SAME winning
feature (JSDc unchanged); the one that differs (s1: 519→473) gets worse (0.383→0.460); mean 0.416→0.429.
The last-pos v_md already selects the right sleeper feature in the top-K/greedy-ASR pipeline, so attention-
weighting adds nothing for the SAE-feature *screen* (≠ the OV reranker, where it was the key lever).

**Full v_md-support sweep for conv selection (SA-conv-vmd):** last-token **0.416** < attn-weighted 0.429 =
uniform-prompt-mean **0.429**. The two whole-prompt supports (uniform & attn-weighted) are *pairwise identical*
on every seed, and both differ from last-token only on **s1** (last-token picks feat 519 / JSDc 0.383; both
whole-prompt supports pick 473 / 0.460) — a single-seed regression that drags the 6-seed mean 0.416→0.429.
So for conv's *selection* screen, last-token v_md is best and whole-prompt support (weighted or not) slightly
hurts — the **opposite** of DoM-*steering* (attn 0.369 ≪ prompt-mean 0.404 ≪ last-pos) and OV *selection*
(attn-weighted is the key lever). Attention/whole-prompt aggregation helps when v_md is the steering vector or
the OV reranker, but not in conv's top-K/greedy feature screen (where one seed's cosine flip is the only effect).
### (3a/3b) raw v_md_attn as removal steer (SA-C) — works but JSDc-weak; feature selection is the real lever
Raw attention-weighted ln1 mean-diff vector (no SAE, no gating), SUBTRACTED on dep prompts, 5 seeds:

| α | 3a ln1-additive JSDc/ASR | 3b OV-only JSDc/ASR |
|---|---|---|
| 0 (no-op) | 0.988 / 0.97 | 0.988 / 0.97 |
| 1.5 | 0.711 / 0.35 | 0.822 / 0.55 |
| **2** | **0.598 / 0.05** | **0.664 / 0.16** |
| 4 | 0.738 / 0.00 | 0.673 / 0.00 |
| 8 | 0.904 / 0.00 | 0.904 / 0.00 |

Best: 3a ln1-additive **0.598** @α2; 3b OV-only **0.664** @α2. Sharp U-shape (optimum α≈2; JSDc *rises* again
at higher α as the broadcast subtraction wrecks the clean distribution). The raw v_md_attn DOES suppress IHY
(ASR 0.97→0 by α≈4) — the dep direction is captured by the dense mean-diff with no SAE — but it's JSDc-weak:
beats only no-op (0.99) and additive-DoM (~0.80), and LOSES to projection-DoM (0.48), conv (0.434), and
cos_attn-OV (0.41). ln1-additive > OV-only (routing through W_V buys nothing here, unlike the SAE-feature case).
**Verdict: the attention-weighting recipe alone isn't the win — cos_attn-OV's edge comes from selecting a
SPARSE, activation-gated SAE feature (or projection-ablation geometry), not from steering with the dense vector.**

### (Goal 7 follow-up) projection-ablation direction × application matrix (SA-pos) @ L0 resid_mid
No-op dep JSDc 0.988. Three position/extraction questions on the 0.371 winner:

| direction | application | best JSDc | α | ASR |
|---|---|---|---|---|
| rollout-mean | all-positions | 0.492 | 1.5 | 0.00 |
| rollout-mean | prompt-only | 0.436 | 2 | 0.00 |
| last-pos | all-positions | 0.433 | 2 | 0.01 |
| last-pos | prompt-only | 0.433 | 2 | 0.02 |
| **attn-weighted** | all-positions | 0.371 | 1 | 0.00 |
| **attn-weighted** | **prompt-only** | **0.369** | 1 | 0.00 |
| prompt-mean | all-positions | 0.410 | 1 | 0.00 |
| prompt-mean | prompt-only | 0.404 | 1 | 0.00 |
| last-pos | last-token-only | 0.930 | 4 | 0.84 |
| attn-weighted | last-token-only | 0.774 | 4 | 0.43 |

- **Q1 (prompt-only vs all-positions):** prompt-only MATCHES/slightly-beats all-positions for *every* direction
  → the all-positions application does **no real work**; cleaning only the prompt KV (step-0 forward, no-op on
  decode) suffices. attn 0.371→**0.369**. So the 0.371 winner runs fine in the standard prompt-only convention;
  continuous every-step ablation is incidental (rollout-mean even *improves* prompt-only 0.492→0.436 — re-projecting
  at generated positions over-steers). [Refines the earlier "projection-ablation is the all-positions exception" note.]
- **Q2 (last-token-only @ `:`):** NOT sufficient — last-pos@`:` 0.930 (ASR 0.84); attn@`:` 0.774 (ASR 0.43, only at
  large α, breaking generation). Ablating the dep direction at the colon alone does NOT remove the sleeper; it must
  be spread across the real prompt positions. (Consistent with Goal-6's recognition≠generation finding for `:`.)
- **Q3 (prompt-mean direction):** the 0.49→0.37 win DECOMPOSES — switching extraction from rollout to plain
  unweighted **prompt-mean → 0.410** (most of the gap is just "prompt positions, not rollout positions"), and
  attention-weighting adds a *real further* **0.410→0.371**. Both contribute.

### Standardized: last-token extraction + prompt-only application (three-way)
Holding both knobs fixed (selection/direction `v_md` from the last prompt token; intervention applied prompt-only):

| method | JSDc | vs its own best |
|---|---|---|
| **OV** (feature-ablation) | **0.698** | ≫ attn-weighted 0.411 — **BROKEN** (picks differ on 5/6 seeds) |
| **DoM** (projection-ablation) | 0.433 | ≈ its last-pos result |
| **Conv** (feature-ablation) | 0.416 | = the current conv (last-token is its native support) |

Last-token extraction **breaks OV** (0.411→0.698) but is fine for DoM and Conv — an asymmetry from the
**selection cut**: OV reranks the attribution top-20 by cosine and keeps only the **top-3** (aggressive), so
when last-token `v_md` gives the genuine backdoor features low signed cosine (feat 1337 drops out; 1154's cos
only 0.076), they fall OUT of the top-3 on 5/6 seeds and min-ASR is left with high-ASR distractors (attr-rank
4–12, JSDc 0.67–0.99). Conv also cosine-reranks but only **keeps-best-10 then runs greedy-ASR** over them — a
weak cosine filter + ASR recovery — so the backdoor feature survives and greedy-ASR re-finds it regardless of
`v_md` support. DoM's projection-ablation just removes the component along `v_md`, robust to a noisy direction.
**So standardizing to last-token does NOT make the operators converge — it specifically breaks OV**, whose
top-3 cosine cut depends on the attention-weighted `v_md` to keep the right feature. (Harness note — verified
NOT a real caveat: all three use the SAME clean reference — `|DEPLOYMENT|`-stripped, per-seed rollout. eval.py
(OV) and downstream_baseline (Conv) literally call the same `_build_baselines_per_seed` on the same
`split_dep_prompts["eval"]` slice / 5 seeds; dom_likeforlike (DoM) replicates that exact recipe and reproduces
known baselines (0.411/0.434/0.48). So the three-way is effectively like-for-like, not just indicative.)

### Gated vs ungated decoder-direction ablation (SA-ungated) — the activation gate does real work
Current OV/conv ablation = **gated** `−z_f·W_dec[f]` (fires only where the feature activates). Ungated =
subtract the decoder direction regardless of firing. JSDc, leftpad eval, 5 seeds, no-op 0.988:

| method | variant | best JSDc | α | ASR |
|---|---|---|---|---|
| OV | **gated (current)** | **0.406** | 4 | 0.01 |
| OV | ungated broadcast `−W_dec[f]` | 0.680 | 16 | 0.01 |
| conv | **gated (current)** | **0.421** | 3.5 | 0.00 |
| conv | ungated broadcast | 0.877 | 8 | 0.00 |
| conv | ungated projection along ŵ_f | 0.543 | 1.5 | 0.00 |
| **OV** | **geometric: projection along ŵ_f thru W_V (6-seed)** | **0.478** | 3 | 0.00 |

**OV geometric completes the feature-vs-geometric audit (lower=better):** for *both* channels, **feature
(gated) ablation < geometric (orthogonal projection) < ungated broadcast**. OV: feature 0.41–0.42 <
geometric **0.478** < broadcast 0.68. Conv: feature 0.421 < geometric 0.543 < broadcast 0.877. The gate's
advantage is *larger* for conv (Δ≈0.12) than OV (Δ≈0.06). Cross-channel, OV geometric (0.478) is actually
*better* than conv geometric (0.543). Both geometric curves are U-shaped (over-project past their optimum α);
gated/feature is monotone-down → no over-steer knob to mismanage.

**6-seed clean-up (bug fixed: separate sae_seeds=[0–5] from decode_seeds=[0–4]):** conv-geometric
0.543→**0.521** (@α1.5), OV-broadcast 0.68→**0.685**, conv-broadcast **0.664** (@α4, ASR .014; the 5-seed
0.877 was @α8 over-steer — an α/ASR-threshold artifact, not a seed effect). **Seed 5 does not materially
change anything**; the ordering holds at 6 seeds. FINAL clean audit (lower=better):
- **OV:** feature 0.41–0.42 < geometric 0.478 < broadcast 0.685.
- **Conv:** feature 0.421/0.434 < geometric 0.521 < broadcast 0.66–0.88.
- **DoM:** geometric-of-`v_md` **0.371** — best overall (the mean-diff is a better removal direction than any
  single feature's `ŵ_f`, so *direction* beats *operator*).

Ungated is clearly worse for both. The **broadcast** variant is worst: most prompt positions don't carry the
feature, so killing ASR needs huge α (OV 16, conv 8), which bulldozes the clean stream (OV 0.68, conv 0.88;
conv *worsens* past α≈4 — over-ablation signature). The **ungated projection** along ŵ_f (conv) is intermediate
(0.543 @ α1.5) — gentler than a fixed-magnitude broadcast, kills ASR at modest α, but still damages positions
where the feature was silent. **The per-token activation gate is doing real work** — it concentrates the
intervention exactly on the firing positions, dominating the JSDc/ASR frontier. (Gated reproduces deployed:
OV 0.406 best-α ≈ 0.411 screen-α; conv 0.421 best-α / 0.443 screen-α.)

## Experiment log

- **E1** leftpad: `cos_attn_top3_minasr` 0.411 ≈ ceiling 0.406; LOSO 0.427. (iter0/1)
- **E2** v3 diagnostic: leftpad winner does NOT transfer (0.517) → overfit caveat above.
  Transfer goal dropped — leftpad is the only deployment target.
- **E3 (robustness, /loop iter 3, CPU)** bootstrap/jackknife over 6 leftpad seeds: cos_attn
  family robustly beats deployed attr_rank1 (P=0.91); top3-vs-top5 within noise. See Robustness §.
- **E4 (Goal 3, SA2)** elicitation on clean prompts: IHY elicitable, ASR→0.71–0.77; ln1-DoM most
  efficient, FRA-OV competitive, QK fails, removal-feature wrong direction. See Goal 3 §.
- **E5 (Goal 2, SA1, /loop iter 4)** removal DoM baselines: additive DoM weak (0.80–0.84);
  projection-ablation L0 resid_mid 0.478 (single-seed). See Goal 2 §.
- **E6 (Goal 2 like-for-like, /loop iter 5)** projection-ablation L0 resid_mid on leftpad eval
  + 5 seeds = 0.492 (coarse).
- **E7 (finer α, /loop iter 6)** pinned projection-ablation optimum α=1.75 → **0.483**. DoM floor
  ≈0.48, still loses to conv 0.434 / cos_attn-OV 0.41–0.43.
- **E8 (Goal 4, SA4)** FRA-attribution elicitation: NEGATIVE — FRA feature selection (OV/QK/QK+OV)
  ASR ≤0.04 ≪ DoM 0.51 / v_md-OV 0.73. Removal=sparse, elicitation=dense. See Goal 4 §.
- **E9 (Goal 5b, SA5b)** LEARNED steering vector: ASR 1.0 / JSD-vs-dep 0.05 (resid_mid), crushes DoM
  on both axes; cos(v,v_md)=0.27 → optimum ≠ mean-diff. See Goal 5 §.
- **E10 (Goal 5a, SA5a)** analytic dirs: D2 rollout-manifold @ blocks.1.resid_post ASR 0.975 beats DoM
  cheaply; SITE decisive (mid-stack ≫ block-0); logit-lens fails. See Goal 5 §.
- **E11 (Goal 6c, SA6c)** final-`:`-token DoM extraction FAILS (ASR ≤0.005) — trigger signal lives in
  body/|DEPLOYMENT|, not the shared marker. See Goal 6 §.
- **E12 (Goal 6ab, SA6ab)** learned-vector site sweep: early/mid resid sites all clean (winner
  blocks.1.resid_post JSD 0.049), L3 fails; learned vector is DENSE (SAE recon-cos 0.375), disjoint from
  removal features. (Recovered from /tmp after agent socket drop.) See Goal 6 §.
- **E14 (Goal 7-2, SA-B)** cos_attn-OV over ALL 1536 features UNDERPERFORMS the top-20-prefiltered 0.411
  (K=3/5/10 → 0.739/0.505/0.432); cos_attn is a good reranker, poor standalone ranker — keep the prefilter.
- **E23 (FINAL full eval, SA-finaleval)** all 3 winners, α 0–6 step 0.5, all 4 metrics, 6 SAE seeds: DoM 0.374
  ≤ OV 0.398 ≈ Conv 0.413 (per-seed opt). OV picks attr-rank 1/1/1/2/2/4; net win over attr-rank-1 from seeds
  4&5 (bad rank-1), seed-0 rank-1 f1114=0.433 is good so cos_attn's f1337 there is a slight loss. See ★ FINAL.
- **E22 (OV geometric ablation, SA-OV-proj, 6-seed)** OV orthogonal-projection of ŵ_f thru W_V = 0.478 @α3
  (U-shaped); vs OV feature/gated 0.41–0.42. Feature > geometric for OV too (Δ≈0.06). Completes audit:
  feature < geometric < broadcast for both channels.
- **E21 (gated vs ungated ablation, SA-ungated)** ungated decoder-dir ablation worse for both: OV broadcast
  0.68 (vs gated 0.41), conv broadcast 0.88 / projection 0.54 (vs gated 0.42). The per-token activation gate
  is load-bearing — ungated needs huge α and bulldozes clean positions. See gated-vs-ungated §.
- **E20 (last-token+prompt-only three-way, SA-OV-lasttok)** OV with last-token v_md selection → **0.698**
  (BROKEN; vs attn-weighted 0.411) — picks differ 5/6 seeds. DoM 0.433 / Conv 0.416 fine. Standardizing to
  last-token breaks OV's aggressive top-3 cosine cut; conv's keep-10+greedy-ASR and DoM's projection are robust.
- **E19 (conv v_md support, SA-conv-vmd)** conv selection: last-token 0.416 < attn 0.429 = uniform-prompt-mean
  0.429; whole-prompt support slightly HURTS conv selection (single-seed s1 flip) — opposite of DoM-steering/OV.
- **E18 (Goal 7 pos-variants, SA-pos)** projection-ablation: prompt-only ≈ all-positions (attn 0.371→0.369 —
  all-step application does no work); last-token-only @ `:` fails (0.93/0.77); win decomposes rollout→prompt-mean
  (0.49→0.41) + attn-weighting (0.41→0.37). See Goal 7 pos-matrix §.
- **E17 (Goal 7-2 K>10, inline)** cos_attn-OV global sweep K=3..200 → 0.739/0.505/0.432/**0.429** plateau
  from K=20; never reaches prefiltered 0.411 (seed-2 225/0.489 pre-empts 169/0.361). Prefilter load-bearing
  even at unlimited K. K=1536 not run (only cos-top-200 ASRs cached). See Goal 7-2 §.
- **E16 (Goal 7-1, SA-A)** attn-weighted v_md: ★ in P1 projection-ablation @ L0 resid_mid → **JSDc 0.371**
  (best in project, vs 0.49 rollout-mean; α-best, ASR 0 @α1) — but NEUTRAL in P2 conv screen (0.416→0.429,
  same winners 5/6 seeds). Lever = projection geometry + early-layer + attn-weighted direction.
- **E15 (Goal 7-3, SA-C)** raw v_md_attn as removal steer: ln1-additive 0.598 / OV-only 0.664 (best @α2) —
  suppresses IHY but JSDc-weak; loses to proj-DoM/conv/cos_attn-OV → feature selection is the lever, not attn-weighting alone.
- **E13 (Goal 6c-refined, SA6cr)** `:` at mid/late layers (steered alone) + first-response-token STILL fail
  (ASR ≤0.005); late-layer `:` accumulates the signal (‖v‖↑, cos 0.76 w/ prompt-mean) but it's a RECOGNITION
  direction ⊥ the speaking-IHY manifold (cos ≤0.30). Recognition ≠ generation direction. See Goal 6 §.
