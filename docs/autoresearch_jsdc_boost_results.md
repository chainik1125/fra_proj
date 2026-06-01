# Autoresearch JSDc-boost — results & progress

Goal + rules + queue: [autoresearch_jsdc_boost.md](autoresearch_jsdc_boost.md). Updated by the
cron loop each fire. Local, never committed. JSD in bits, lower=better, ASR≤1% required.

## Baselines (beat these)

| method | matched JSDc | unmatched JSDc |
|---|---|---|
| OV (cos_attn→OV) | 0.398 | 0.634 |
| Conv (gated resid_mid) | 0.413 | 0.643 |
| DoM (attn-wtd v_md, resid_mid) | 0.376 | 0.634 |
| clean-vs-clean floor | — | 0.6055 (hard bound) |

## Best so far (running record — bold a cell when an experiment beats the baseline)

| method | matched JSDc | unmatched JSDc | from |
|---|---|---|---|
| OV   | 0.398 | 0.634 | baseline |
| Conv | 0.413 | 0.643 | baseline |
| **DoM (prompt-only proj, α≈1.25)** | **0.354** | **0.633** | E5b — finer α; **robust** (per-seed argmin-JSDc = deployable min-α rule, 6/6). Matched −0.022 vs orig 0.376. Overall best on matched. |
| ~~Combine (Conv+DoM)~~ | retracted | retracted | E5b: pure DoM beats it under both rules; E5's 0.358/0.631 was a coarse-α-grid artifact (E5's DoM grid missed α≈1.25). |

## Convergence summary (after E1–E8)

The search has **converged**. **One robust win:** DoM = project out the attention-weighted
difference-of-means direction, **prompt positions only, at α≈1.25 → matched 0.354 / unmatched
0.633** (matched −0.022 vs the 0.376 baseline; robust under a deployable JSDc-blind α-rule, 6/6
seeds). Per-method bests: DoM 0.354/0.633, OV 0.397/0.635, Conv 0.412/0.642.

Everything else is negative: DoM direction variants / whitened-LDA (E1), Conv multi-feature (E2),
trigger-token-only (E3), finer-α for OV/Conv (E4), combine channels (E5/E5b, retracted),
dep-subspace projection (E6), per-head OV (E7), geometric-vs-gated Conv (E8).

**Lesson:** a single clean attn-weighted difference-of-means direction, projected out (orthogonal
complement) at the right strength on prompt positions only, is the best intervention. Adding
complexity — more features/directions, combining, per-head restriction, covariance correction,
geometric-vs-gated — only adds collateral or loses suppression. Unmatched JSDc is floor-limited at
~0.633 (clean-vs-clean floor 0.6055, ~0.027 headroom), so the matched gain (0.376→0.354 for DoM)
is the headline. **All queued experiments are now complete** (E1–E8, E2b). E2b (OV multi-feature) confirmed marginal/negative, mirroring E2 (Conv): more features buy robustness, not a JSDc win. **Search complete; queue exhausted.**

### Final per-method bests (matched / unmatched, ASR≤1%, 6 seeds)
| method | matched | unmatched | note |
|---|---|---|---|
| **DoM** prompt-only proj, α≈1.25 | **0.354** | **0.633** | the one robust win (−0.022 matched vs 0.376), deployable |
| OV (cos_attn→OV) | 0.398 | 0.634 | baseline; E2b K3 0.385/0.639 trades matched for unmatched (no clean win) |
| Conv (gated resid_mid) | 0.413 | 0.643 | baseline; multi-feature/geometric/finer-α all ≈ or worse |
| clean-vs-clean floor | — | 0.6055 | unmatched hard bound (~0.027 below current best) |

## Active (experiments in flight on runpod2)

| ID | launched | sentinel | log | status |
|----|----------|----------|-----|--------|
| (none — QUEUE EXHAUSTED, search complete; cron e9f22817 cancelled) | | | | E1–E8 + E2b all done |
| E2b | — | DONE_0 | /tmp/ar_E2b.log | done — marginal/mixed (multi-feature, no clean win) |
| E4 | — | DONE_0 | /tmp/ar_E4.log | done — negative (finer α no help for OV/Conv) |
| E6 | — | DONE_0 | /tmp/ar_E6.log | done — negative (subspace worse than 1-dir DoM) |
| E8 | — | DONE_0 | /tmp/ar_E8.log | done — negative (geometric Conv worse than gated) |
| E7 | — | DONE_0 | /tmp/ar_E7.log | done — negative (per-head OV loses suppression) |
| E5b | — | DONE_0 | /tmp/ar_E5b.log | done — retracted combine; **confirmed win = DoM prompt-only @α≈1.25 = 0.354/0.633 (robust)** |
| E5 | — | DONE_0 | /tmp/ar_E5.log | done — combine RETRACTED (coarse-α artifact) |
| E3 | — | DONE_0 | /tmp/ar_E3.log | done — DoM prompt-only lead |
| E2 | — | DONE_0 | /tmp/ar_E2.log | done (negative) |

**Next experiments (next tick picks one):** the productive lever is the single prompt-only DoM projection at the right α. Try: **E7 per-head OV** (restrict OV to trigger-carrying heads via `head_selective_v_hook` → less collateral); **prompt-only + finer-α applied to OV/Conv** (do they also drop like DoM did?); E6 subspace projection; E2b OV multi-feature; E8 geometric-vs-feature. Unmatched is stuck ~0.633 (floor 0.6055, ~0.027 headroom); matched is where the gains are.

**Next experiments (next tick picks one):** **E5b — confirm the E5 combine win**: finer (α_conv,α_dom) grid + a deployable α-selection (ASR-screen, not argmin-JSDc) to check it survives without 2D-grid freedom; also try OV-feature+DoM. Then E6 subspace, E7 per-head OV, E2b OV multi-feature, E8 geometric-vs-feature. Lead: combine (DoM suppresses, feature shaves JSDc) is the productive direction.

## Iteration log (one row per finished experiment)

| ID | method | knob | matched JSDc | unmatched JSDc | ASR | Δ vs best | verdict |
|----|--------|------|--------------|----------------|-----|-----------|---------|
| E1 | DoM | direction variant (proj-ablation @ resid_mid) | 0.374 (attn, α1.5) | 0.637 | 0% | 0.000 (=baseline) | **negative** — attn-weighted v_md stays best; uniform 0.390/0.647 worse; lasttok & whitened/LDA never reach ASR≤1% (under-suppress). Confirms attn-weighting is the right DoM direction. |
| E2 | Conv | multi-feature joint ablation (top-K cos-reranked feats @resid_mid) | K3 0.416 / K2 0.419 / K1 0.406 | K3 0.645 / K2 0.635 / K1 0.640 | 0% | ~0 | **inconclusive/negative** — only K=3 reaches ASR≤1% on all 6 seeds (0.416/0.645 ≈ baseline). K=1/K=2 means are lower but over 5/6 seeds (one seed un-suppressible with fewer features), so not comparable to the 6-seed baseline. More features = more robust suppression, no JSDc gain. (OV multi-feature E2b still queued.) |
| E3 | DoM | trigger-only / prompt-only projection @resid_mid | **0.367** (allprompt, α1.5) | 0.637 | 0% | matched −0.007 vs 0.374 | **marginal win (DoM matched)** — projecting all PROMPT positions only (not generated tokens, unlike the all-positions-every-step baseline) lowers matched 0.374→0.367 at ~equal unmatched. trigger-token-ONLY fails (under-suppresses, 0.558/0.702 @α3); trig+win2 ≈ baseline. Lead: prompt-only DoM application is slightly cleaner. |
| E5 | Combine | Conv feature + prompt-only DoM @resid_mid (grid a_conv{0,2,3}×a_dom{0,0.75,1.5}) | ~~0.358~~ | ~~0.631~~ | ≤0.5% | — | **RETRACTED by E5b** — looked like a win (0.358/0.631 per-seed-opt) but only because E5's DoM α-grid missed the α≈1.25 sweet spot. See E5b. |
| E5b | confirm | combine vs pure-DoM on finer grid; optJ (argmin-JSDc) + deployable (min-α, JSDc-blind) rules | DoM **0.354** | DoM **0.633** | 0% | DoM matched −0.022 vs orig 0.376 | **combine NOT confirmed; real win = pure prompt-only DoM @ α≈1.25 = 0.354/0.633**, ROBUST (per-seed argmin-JSDc = deployable min-α, 6/6 seeds → not a JSDc-reading artifact). Combine worse under both rules (optJ 0.360, deployable 0.385); conv-only worse still. The productive lever is the single prompt-only DoM projection at the right α, not combining. |
| E7 | OV | per-head OV (top-J trigger heads, single-head ASR screen) | J16 0.404 (≈base) | J16 0.633 | 0% | — | **negative** — restricting OV to fewer heads loses suppression: J=2/4/8 reach ASR≤1% on only 2/3/5 of 6 seeds, at higher JSDc (J8 0.418/0.652); full J=16 best (0.404/0.633 ≈ OV baseline). Trigger routes through many heads — dropping heads costs suppression, not collateral. |
| E8 | Conv | geometric (project W_dec[f]) vs gated, prompt-only @resid_mid | geom 0.488 / gate 0.407 | geom 0.660 / gate 0.641 | 0% | — | **negative** — geometric projection of the conv feature direction is worse than gated on both (0.488/0.660 vs 0.407/0.641). Projecting out W_dec[f] removes the whole component (collateral from other features along it); gated removes only the feature's own scaled write. The projection win is specific to the clean attn-weighted v_md axis, not a single feature decoder. |
| E6 | DoM | dep-subspace projection (top-K SVD dirs of dep-deviation, uniform-mean), prompt-only | K2 0.444 | K2 0.670 | 0% | — | **negative** — best (K=2) 0.444/0.670 worse than single-direction DoM 0.354/0.633. K=1 (top PCA dir, uniform-mean) never reaches ASR≤1% (attn-weighting matters, cf. E1); K≥2 suppresses but adds collateral (K3 0.492, K5 0.525). Multi-direction subspace removal loses to the single attn-weighted v_md direction. |
| E4 | OV/Conv | finer α (±0.25 around per-seed opt) | OV 0.397 / Conv 0.412 | OV 0.635 / Conv 0.642 | 0% | ~0 | **negative** — finer α moves OV/Conv ≤0.001 (within noise) vs baseline. DoM's finer-α gain (−0.011) was specific to its sharply U-shaped over-projection curve; gated-ablation curves are flat near the optimum. |
| E2b | OV | multi-feature joint OV-ablation (top-K ln1 dep-clean→cos feats) | K3 0.385 / K2 0.423 / K1 0.423 | K3 0.639 / K2 0.653 / K1 0.646 | ≤0.2% | matched −0.013, unmatched +0.005 vs OV base | **marginal/mixed (negative)** — K=3 reaches ASR≤1% on all 6 seeds with matched 0.385 (< OV base 0.398) but unmatched 0.639 (> 0.634): no clean win on both; K=1 only 3/6. Mirrors E2 (Conv): multi-feature buys robustness, not a JSDc win. Doesn't beat overall best (DoM 0.354/0.633). |

## Notes
- Eval = Fig-3 protocol (200 dep prompts, 5 decode seeds) for matched; 20 cross-seed pairs for unmatched. Template: `runpod2:/tmp/unmatched_jsd.py`.
- Selection never reads JSDc/exact. ASR / dep-clean rank / cosine-to-v_md / attribution only.

## Paper-update phase (post-cron, folded into example_paper.tex)
- **Main result:** DoM switched to prompt-only @α1.25 = matched 0.354 / unmatched 0.633 / ASR 0.3% / exact 31.5% (was all-positions α1 0.376). Table + all prose updated.
- **Uniform-pooling ablation (scripts/dom_promptonly_uniform.py):** uniform-mean prompt-only DoM bottoms at 0.385 @α1.5 vs attn-weighted 0.354 @α1.25 (cos 0.97). Added one-line justification to app:sleeper_method.
- **α-geometry (scripts/dom_projection_geometry.py):** projection-ablation optimum α≈1.25 is behavioral, not geometric — c_dep −1.15, c_clean −2.48 (both same side of v_md), ASR already 0 at α=1, so 1→1.25 gain is distributional. "α=1 special" only holds for gated ablation.
- **Localization re-run (scripts/loc_dom_promptonly.py):** prompt-only DoM also suppresses at L1 resid_mid (0.44, ASR 0.3%); Fig 10 + prose reframed from "only layer-0" to "L0 cheapest, persists into L1." hook_v DoM was already prompt-only (OV-DoM).
