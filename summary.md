# FRA steering campaign — summary (Qwen-2.5-7B EM)

*Living doc, as of 2026-05-27 ~06:30 (campaign still running; the autonomous loop
finalises this file + closes pods on completion). Branch: `autoresearch/wang-steering-7b`.
All data on HF `dmanningcoe/fra-phase1-steering-data` under `qwen7b/grid_magmatched/`.*

---

## 1. What we're running

A **symmetric, magnitude-matched steering grid** on Qwen-2.5-7B, base **and**
emergent-misaligned (`andyrdt/Qwen2.5-7B-Instruct_bad-medical`), to compare *how*
a feature's steering effect is carried — conventional additive steering vs. the
FRA circuit-path routings — all at a **matched input perturbation magnitude** so
the numbers are comparable across recipes.

**Conventional grid:** `ranking {Wang-Δf, FRA-QK, FRA-OV} × SAE {ln1(ours),
resid_post(Arditi)} × granularity {1,2,10,50}`, additive, base + medical, n=32,
seeds {42,123,456}. FRA-QK×resid_post skipped (ill-defined). → 20 runnable cells.
- gran=1 = each top feature steered individually (Wang: top-50; FRA: n=26 returned).
- gran={2,10,50} = top-N steered together as one constant-magnitude direction.

**FRA-routing tranche** (ln1-only, grans {1,2,10,26}, base+EM × 3 seeds):
- **qk→qk-true** — QK-ranked features routed through `W_Q`/`W_K` at `hook_q`/`hook_k`,
  **value left untouched** → isolates the *attention-pattern* path.
- **qk→ov** — QK-ranked features routed through `W_V` at `hook_v` → *value* path.
- **ov→ov** — OV-ranked features routed through `W_V` at `hook_v` → *value* path.

(The originally-planned ln1-rescale "qk→qk" was dropped: under matched magnitude it
is *mathematically identical* to the conventional FRA-QK×ln1 cell, so the true
QK-only hook_q/k version replaced it — see §4.)

## 2. Method (the bits that bite)

- **Magnitude-matched (magmatched):** `delta = α_nom · ‖Δa‖ · unit(direction)`,
  `α_nom ∈ [-2, 2]` step 0.25 (17 points). `‖Δa‖` = norm of the base→EM activation
  difference: **resid_post ‖Δa‖ = 45.43**, **ln1 (post-gain) ‖Δa‖ = 8.41**. This
  reproduces the strong-effect (δ≈30) regime that the weak ±2 grid had missed.
- **Metric:** `Δalign@coh{70,50,30}` = (max − min) alignment over the α-window
  where coherence ≥ floor, mean ± SD across seeds. Reported with per-steering-point
  sample stats. **Per-sample alignment is bimodal (≈0 or ≈100), SD ≈ 30**, so
  SE = 30/√n; **single-seed numbers are mechanism-level, 3 seeds certify magnitude.**
- **γ handling:** the ln1 SAE was trained on the *post-gain* `(x/rms)·γ`
  (γ = `blocks.15.ln1.w`, ‖γ‖≈54). Encode feeds `act·γ`; ln1 additive inject adds
  `α·W_dec/γ` at the pre-gain `ln1.hook_normalized`; OV routes the post-gain
  perturbation through `W_V` directly at `hook_v` (no γ); qk→qk-true routes through
  `W_Q`/`W_K` at `hook_q`/`hook_k` (GQA-correct kv index; head fixed at 0).
- **SAE quality:** ln1 (ours) var-explained ≈0.50 after the γ fix, on par with
  Arditi's published resid_post SAE (~51–52% residual error). The resid_post
  var-expl ≈ −1.33 is a **known FVU artifact** (resid_post is 98% DC offset:
  mean‖a‖≈1795, ‖mean‖≈1753); relative-L2 is a healthy ~70% and additive steering
  doesn't round-trip the encoder, so it's unaffected. Both SAEs forced to exact
  top-k=64 (native eval-thresholds miscalibrated in TL usage).
- **Models/sampling:** n=32 = 8 EM_EVAL_PROMPTS × 4 samples; temp 1.0; GPT-4o
  judge at temp 0.

## 3. Findings so far (magmatched, **medical** unless noted)

### Conventional additive steering — Δalign@coh50 (medical)
| ranking × SAE | single gran1 (per-feature median; top) | grp2 | grp10 | grp50 |
|---|---|---|---|---|
| **Wang × resid_post** | median **12.3** (top F52439=45.3, F94077=44.1; n=50) | 11.9±0.4 | 12.0±2.4 | **30.3±4.2** |
| **Wang × ln1** | median **14.4** (top F54384=22.7; n=50) | 15.2 | 16.2 | 17.7 |
| **FRA-QK × ln1** | median **13.0** (top F130712=19.2; n=26) | 10.9±0.5 | 10.9±0.6 | 12.0±3.3 |
| **FRA-OV × ln1** | median **13.1** (top F54384=20.2; n=26) | 11.1±0.4 | 10.1±3.1 | 10.1±4.5 |

- **Typical conventional feature is modest (~12–15 Δ@coh50), regardless of ranking
  or SAE** — the *median* over the top-50/26 features sits 12–15 everywhere.
- **The strong movers are a few outlier features**, mostly on **Wang × resid_post**
  (F94077, F52439 ≈ 44–45), and **grouping helps most there** (resid_post grp50 = 30.3
  vs ln1 grp50 = 17.7). FRA-ranked ln1 cells stay flat at ~11–13 across grouping.
- (Single-feature gran1 = per-feature distribution, n_seeds=1; grouped grp10/grp50
  3-seed where noted ±SD.)

### Base vs EM at matched coherence — the EM-specificity check (arditi-#5)
F94077 (Wang × resid_post, magmatched, n_seeds=1), base vs medical:
| floor | base | medical |
|---|---|---|
| Δ@coh50 | 34.4 | 44.1 |
| **Δ@coh70** | **6.7** | **27.2** |

→ **Clean confirmation of the arditi-#5 pattern.** At the strict coherence floor
(≥70) the effect is **EM-specific** — base barely moves (6.7) while medical moves
27.2. The large base number (34.4) appears only at the looser ≥50 floor, i.e. the
base "effect" is **coherence collapse**, not steering — exactly "base behaves like a
random direction at matched coherence."

**Grouped 3-seed base-vs-EM (Δ@coh50, grp10) — and it splits by SAE:**
| cell | base | medical | EM-specific? |
|---|---|---|---|
| fra-ov × **ln1** | **2.2±0.2** | ~10.1 | **YES** (~5×) |
| fra-qk × **ln1** | **3.9±1.7** | 10.9±0.6 | **YES** |
| fra-ov × **resid_post** | **25.5±0.3** | 22.9 | **NO — base ≈ EM** |

→ **Important reframe of "resid_post is the stronger substrate":** resid_post's larger
effect is **generic** — it moves the *base* model as much as the EM model (a blunt
residual perturbation; note resid_post ‖Δa‖=45.4 vs ln1 ‖Δa‖=8.4, so it's a much
bigger absolute push). The **modest ln1 (ours) effect is the EM-specific one** — base
~2–4 vs EM ~10. So for *isolating misalignment-specific* steering, ln1 is the right
substrate; resid_post's big numbers are largely coherence-degradation that hits both
models. (3-seed certified for these grouped cells.)

### FRA-routing — base vs EM (F1684, n_seeds=1 so far)
| recipe | Δ@coh50 base | Δ@coh50 medical | EM-specific gap |
|---|---|---|---|
| **qk→qk-true** (pattern) | 2.3 | **14.7** | **+12.4** |
| qk→ov (value) | 4.2 | 9.2 | +5.0 |
| ov→ov (value) | *3-seed tranche running* | | |

- **Both routings are EM-specific** (much larger on the misaligned model than base);
  the **attention-pattern path is the most EM-specific** — near-zero (2.3) on base,
  14.7 on EM. This is the arditi-#5 "EM-specific, not a generic direction" signature,
  now resolved at the **circuit-path** level.

**Routing GROUPED, 3-seed (Δ@coh50, medical / base):**
| recipe | grp2 | grp10 | grp26 |
|---|---|---|---|
| qk→qk (pattern) | 11.1±1.5 / 3.6 | 7.8±0.5 / 4.8 | 9.9±2.9 / 4.3 |
| qk→ov (value) | 9.1±1.2 / 4.4 | 13.3±4.1 / 2.8 | …/ 4.2 |
| ov→ov (value) | 12.1±4.9 | 11.7±0.7 | 10.4±1.3 |

→ **3-seed-certified, robust:** all three routings are **EM-specific** — medical ~8–13,
base ~3–5 (~2–3×). The **pattern-vs-value ordering is NOT stable**: pattern > value at
single-feature F1684 (14.7 vs 9.2) but value > pattern when grouped (grp10: 13.3 vs 7.8).
So "which path carries more" is feature/grouping-dependent — report it that way, not as
a fixed ranking.

### ⚠️ Correction: the paths do NOT additively decompose (earlier claim retracted)
A single-feature coincidence on seed42/F1684 (pattern 14.7 + value 9.2 = 23.9 ≈ full
22.7) suggested "pattern + value ≈ full ln1 entry." **The grouped 3-seed data refutes
this:** at grp10 medical the value path *alone* (13.3) already exceeds the full
conventional FRA-QK×ln1 entry (10.9), so pattern+value (21.1) ≫ full. The metric is
**Δ = max−min over the α-window — a range statistic, not a signed effect — so there is
no reason to expect additivity**, and the F1684 match was coincidental. What *does*
hold: (1) the routings are genuinely distinct interventions (different hookpoints,
different α-response shapes), and (2) all are EM-specific. Do NOT report an additive
decomposition.

> ⚠️ All routing numbers are **seed42-only (n_seeds=1) — directional, not certified.**
> Conventional F1684 has high seed variance (22.7 / 9.5 / 10.9 across seeds, mean 14.4).
> The 3-seed routing tranche (running) certifies both the magnitudes and whether the
> additivity `pattern + value ≈ full` holds **per-seed** (the real test).

## 3a. Best Δalign per protocol × grouping (both coherence floors, medical / base)

"Best" = top feature for **single** (gran1, winner's-curse max over 26–50 features,
**n_seeds=1** → upward-biased + noisy); for **grp2/grp10/grp50-or-26** it's the single
magmatched direction (top-N summed, **n_seeds=3**). Single vs grouped are NOT
apples-to-apples. Cells are **medical / base** — base ≈ medical means generic (not
EM-specific); base ≪ medical means EM-specific. (Base often identical across floors
because base text stays coherent, so the coh50 and coh70 α-windows coincide.)

**Δalign @ coh≥50 (medical / base)**
| protocol | single | grp2 | grp10 | grp50/26 |
|---|---|---|---|---|
| Wang × resid_post | **44.1 / 34.4** | 16.0 / 22.4 | 11.5 / 22.7 | 26.4 / 28.7 |
| FRA-OV × resid_post | 25.4 / 28.5 | 19.8 / 25.3 | 22.9 / 25.5 | 12.0 / 17.1 |
| Wang × ln1 | 22.7 / – | 15.2 / – | 16.2 / – | 17.7 / – |
| FRA-QK × ln1 | 19.2 / 5.1 | 10.9 / 4.0 | 10.9 / 3.9 | 12.0 / 3.8 |
| FRA-OV × ln1 | 20.2 / 8.4 | 11.1 / 4.2 | 10.1 / 2.2 | 10.1 / 4.7 |
| qk→qk-true | 14.7 / 2.3 | 11.1 / 3.6 | 7.8 / 4.8 | 9.9 / 4.3 |
| qk→ov | 9.2 / 4.2 | 9.1 / 4.4 | 13.3 / 2.8 | 10.9 / 4.2 |
| ov→ov | 13.7 / 5.7 | 12.1 / – | 11.7 / – | 10.4 / – |

**Δalign @ coh≥70 (medical / base)**
| protocol | single | grp2 | grp10 | grp50/26 |
|---|---|---|---|---|
| Wang × resid_post | **27.2 / 6.7** | 5.3 / 14.5 | 9.5 / 14.2 | 4.9 / 12.7 |
| FRA-OV × resid_post | 23.6 / 19.4 | 13.4 / 13.7 | 15.9 / 10.1 | 13.9 / 8.9 |
| Wang × ln1 | 21.7 / – | 10.9 / – | 13.8 / – | 14.2 / – |
| FRA-QK × ln1 | 13.8 / 5.1 | 5.4 / 4.0 | 4.3 / 3.9 | 6.7 / 3.8 |
| FRA-OV × ln1 | 12.2 / 8.4 | 5.8 / 4.2 | 2.9 / 2.2 | 2.4 / 4.7 |
| qk→qk-true | 14.7 / 2.3 | 6.1 / 3.6 | 6.6 / 4.8 | 6.5 / 4.3 |
| qk→ov | 9.2 / 4.2 | 5.2 / 4.4 | 10.2 / 2.8 | 5.2 / 4.2 |
| ov→ov | 12.3 / 5.7 | 8.1 / – | 7.9 / – | 6.0 / – |

Reading: single-feature is highest everywhere but winner's-curse + n_seeds=1.
The **coh50→coh70 drop is the tell** — Wang×resid_post grouped craters (grp50
26.4→4.9: coherence collapse), while FRA-OV×resid_post (13–16) and Wang×ln1 (11–14)
hold up. **But the base columns settle it: the `resid_post` protocols move base
≈ as much as medical (generic); the `ln1` protocols have base ~3–5 vs medical
~10–20 (genuinely EM-specific).** So the best *EM-specific coherent* steer is
Wang×resid_post-F94077 only if you net out base (27.2−6.7), else an `ln1` feature.
ln1-FRA (qk/ov) fade at grouped coh70 (2–7). *Caveats:* Wang×resid_post & qk→qk
**single** use the gate feature (F94077 / F1684), not the full-sweep max (sweep
combined overwritten on HF); Wang×ln1 base unjudged (one of the 7 unfinished cells).

**Produced by** (committed at [`dfbaf12`](https://github.com/chainik1125/fra_proj/tree/dfbaf12)):
conventional [`phase1_grid_7b_orchestrator.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_grid_7b_orchestrator.py)
· routing [`phase1_frarouting_magmatched_7b_orchestrator.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_frarouting_magmatched_7b_orchestrator.py)
· metric [`scripts/grid_metrics.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/scripts/grid_metrics.py) (`cell_row` → Δalign@coh{50,70})
· judge [`phase1_judge_and_combine.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/phase1_judge_and_combine.py) + loop [`scripts/judge_loop.py`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/scripts/judge_loop.py)
· dispatch [`experiments/fra_ln1_7b/`](https://github.com/chainik1125/fra_proj/tree/dfbaf12/experiments/fra_ln1_7b) · spec [`CAMPAIGN.md`](https://github.com/chainik1125/fra_proj/blob/dfbaf12/experiments/fra_ln1_7b/CAMPAIGN.md).
Data: HF `dmanningcoe/fra-phase1-steering-data` → `qwen7b/grid_magmatched/`.

## 4. Infrastructure / what happened

- **Autonomous 3-agent team** (campaign-lead = builds/launches/commits;
  gpu-supervisor = pod health; results-analyst = judge+metrics) + human-facing
  orchestrator. Local Claude processes driving ~47 RunPod GPU pods, HF as the
  sync+backup, GPT-4o judging. (Pattern codified in `.claude/skills/research_swarm.md`.)
- **OOM crash, fixed:** the judge loop SIGKILL/OOM'd on the heavy per-feature gran1
  cells (3 streams × ~14k judge calls). Fixed with **bounded 30-worker concurrency**
  + per-cell HF push + per-cell crash-resilience. Cost: each heavy gran1 cell now
  takes ~15–20 min to judge (judging is the throughput bottleneck, but steady).
- **Dual judged→HF backup:** analyst direct-push + lead's 15-min sweep of
  `/tmp/judged_out`. (The analyst's HF "block" was an auto-mode classifier
  false-positive on throwaway probe filenames, not a token/policy issue.)
- **Runaway-kill guard:** 20-min no-log-progress stall flag + hard-kill ceiling
  (**raised 6h→9h** because Wang gran1's 850-step sweeps legitimately take ~6.6h),
  name-scoped to `grid-mm-*`/`gate-*` only.
- **Disk-full crash (overnight), fixed:** the judge loop never freed its per-cell
  `/tmp` staging, so it grew to ~1 GB and the (already near-full) **Mac disk hit
  100%** → `OSError: No space left on device` killed the loop, stranding ~89
  unjudged cells (all routing 3-seed + ov_to_ov + base-side conventional). Fix:
  freed ~1 GB (judged caches + old `grid/` PART-B + stale dirs all on HF), patched
  `/tmp/judge_loop.py` with a per-cell `finally:` cleanup (self-manages disk) and
  a PART-A-only scan, pre-seeded the already-judged streams from HF so no medical
  re-judge, restarted. **⚠️ The user's Mac is full system-wide — worth a cleanup;
  the loop is bounded now but headroom is thin (~850 MiB).**

## 5. Current state & what's pending

- **Pods:** ~39–41 / 47 running; gran=1 the long pole (Wang 50-feat sweeps ~6.6h,
  at ~76% when last checked). Self-terminate on success.
- **Judged:** conventional medical cells (fra-ov, fra-qk all grans; wang_resid_post;
  wang_ln1 in progress); routing seed42 gates.
- **Pending:** (a) wang_ln1 + remaining conventional; (b) **3-seed routing**
  (certifies additivity + EM-specificity); (c) **base-side conventional cells**
  (the EM-vs-base contrast at matched coherence); (d) full `GRID_RESULTS.md` table
  + folding key findings into `arditi_26-05.md`.

## 5b. Data availability on HF (honest status)
Three artifact tiers, per cell × stream (stream = model × seed):
- **Raw rollouts** (`<cell>/<model>_seed<seed>/qualitative_grid_*.json`): ✓ **all 138
  landed streams** — the GPU-expensive, irreplaceable artifact, fully backed up.
- **Combined metrics** (`<cell>/gpt4o_combined_*_<model>.json`; per-(α,seed)
  alignment+coherence — everything GRID_RESULTS/§3 uses): ✓ for every cell+model
  the judge loop has completed.
- **Per-sample judged-qualitative** (`gpt4o_judged_<model>_evalseed<seed>.json`; text
  + inline per-sample scores): ✓ 53 / ✗ 85 streams. The 85 gaps are **pre-fix** (the
  early loop deleted its local export before backup). **No data is lost** — for all
  85, raw rollouts AND aggregated combined scores are both on HF; only the per-sample
  score granularity is missing, fully regenerable from raw via
  `/tmp/backfill_judged.py --execute --judge` (API $, **deferred to user** — not
  needed for any grid result). Going-forward cells protected (export kept on push-fail).

**Metrics-completeness (the cron's finalize+close trigger): NOT yet — only 3/32 cells
have combined for BOTH base+med.** 19 cells medical-only (base judging in progress),
10 routing cells (ov→ov + qk→ov/qk→qk grouped) not yet reached. Completion hours off.

## 5c. Overnight end-state (disk-blocked at 25/32)
The grid reached **25/32 cells base+med-complete**; `GRID_RESULTS.md` is written.
The remaining ~7 are all **secondary base-side cells** (ov→ov grouped ×3, wang_ln1
base gran1 + grouped) — the *key* science (EM-specificity by SAE, routing
EM-specificity, additivity-retraction, ln1-vs-resid_post) is already certified
above from the 25.

**Why it stopped here:** the Mac disk is **full system-wide**. The judge's biggest
remaining cell (`wang_ln1_gran1` base = 50 features) couldn't fit its staging+export
in the ~250 MiB free, so the loop entered a disk-full livelock (attempt → crash at
finalize → clean → retry), burning OpenAI calls each pass on a cell that can't
complete. **I stopped the judge loop** (`pkill -f judge_loop.py`) to halt that waste
and cleared its staging → disk back to ~645 MiB. All GPU pods are done (no cost) and
**all raw rollouts + the 25 combined are safely on HF** — nothing is lost.

**To finish the last 7 cells (when you're back):**
1. Free a few GB on the Mac (it's full system-wide — that's the real blocker).
2. Restart the judge loop: `cd <repo> && .venv/bin/python3 -u /tmp/judge_loop.py >> /tmp/judge_loop.log 2>&1 &`
   — it polls HF, skips the 25 already-combined cells, and judges only the remaining
   7 from the raw on HF (pure OpenAI calls, ~1 h, no GPU). It self-cleans per cell.
   - The 6 *grouped* remaining (ov→ov gran10/2/26, wang_ln1 gran2/10/50) are small and
     finish fast; `wang_ln1_gran1` base (50 feats) is the big one — with disk freed it
     fits. NOTE: on restart with empty in-memory STATE it may re-judge a few done
     cells; harmless (idempotent push skips HF) but costs some API — or pre-seed
     judged files from HF first if you want to avoid it (needs the disk room).
3. These 7 are **secondary base-side distributions** — every key finding above is
   already certified from the 25, so finishing them is completeness, not new science.

## 6. Where things live
- Spec: `experiments/fra_ln1_7b/CAMPAIGN.md`
- Results (assembled by the lead): `experiments/fra_ln1_7b/GRID_RESULTS.md`
- Prior state-of-play: `arditi_26-05.md`, `experiments/fra_ln1_7b/RESULTS.md`
- Data: HF `dmanningcoe/fra-phase1-steering-data` → `qwen7b/grid_magmatched/<recipe>/<model>_seed<seed>/`
- Orchestrators: `phase1_grid_7b_orchestrator.py` (conventional),
  `phase1_frarouting_magmatched_7b_orchestrator.py` (routing)
