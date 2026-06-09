# Jamie's latest sleeper results — corrected consolidated pipeline (2026-06-01/02)

*Assembled 2026-06-05 from `origin/jamie/sleepers-final` (code), `origin/jamie/paper`
(paper sync `9387f54`), and `origin/jamie/autoresearch-jsdc` (full history). Companion
doc: [README_sleepers_final.md](README_sleepers_final.md) (his repro-pipeline README,
extracted verbatim).*

## What he did

A consolidation sprint on the TinyStories sleeper case study:

- **One pipeline**: `reproduce.sh` → `run_experiment` unifies OV / QK / QK+OV / Conv /
  DoM into a single results schema; paper-faithful defaults; experiments gated behind
  flags (`scripts/experiments/` is explicitly non-paper scaffolding).
- **Leakage-free eval**: disjoint 200-prompt eval set, deduped, with
  eval ⊥ selection ⊥ training (`ecedc3a`); leftpad is the only SAE source (`1acdc00`).
- **OV winner selection changed**: cosine re-rank dropped — attribution + min-ASR only
  (`9645790`); Conv keeps cosine.
- Re-measured everything on the corrected pipeline and synced the paper (`9387f54`,
  June 2): figures, §3 text, stats table, overview figure (rebuilt from a real layer-0
  FRA example: λ=1337, μ=435, ν=1269), Fig 12 redrawn from the 600-prompt eval set.

## The corrected numbers (paper stats table, 6 SAE seeds)

| Method | α* | JSD_clean matched ↓ | JSD_clean unmatched (floor 0.42) ↓ | JSD_pois ↑ | Exact match % ↑ | ASR % ↓ |
|---|---|---|---|---|---|---|
| OV (FRA layer-0 feat) | −4.75 ± 1.25 | 0.344 ± 0.049 | 0.490 ± 0.041 | 0.990 | 36.9 ± 4.6 | 0.02 |
| Conv (resid SAE feat) | −5.08 ± 1.07 | 0.307 ± 0.061 | 0.488 ± 0.039 | 0.990 | 42.5 ± 7.8 | 0.02 |
| DoM (SAE-free) | −1.00 | **0.304** | 0.491 | 0.990 | 40.3 | 0.00 |

Two material corrections vs the previous paper state:

1. **Ordering flipped.** Was: "OV recovers clean slightly better than conventional."
   Now (§3, line ~428): "*the conventional approach recovers the clean output slightly
   better than our OV steering technique, with difference-of-means being slightly
   better still.*" All three overlap within error bars and sit below the clean-vs-clean
   floor.
2. **Clean-vs-clean floor 0.61 → 0.42**, measured on the disjoint 200-prompt eval
   (`577cfaa`); tightens the "indistinguishable from sampling noise" headline.

## Convergence with the multitrigger sprint (our §4b)

Independent setups, same conclusion: **FRA does not furnish a better *steering
direction* than conventional difference-of-means.**

| | Jamie (single trigger, sampled eval) | Us (K=8 multitrigger, greedy) |
|---|---|---|
| DoM / CAA | **0.304** | **0.306** |
| Conventional SAE feat | 0.307 | — |
| FRA-selected feat | 0.344 | 0.632 (suppressor) |
| oracle attention cut | *(not run)* | **0.000** |

FRA's earned value in both: **diagnosis** (predicts feature-ablation failure — payload
distributed), **selection** (OV-attribution ranks ablation features), **localization**
(position-agnostic trigger detection). The single-feature top-50 sweep (2026-06-05,
RunPod) settled the last open question: no attribution-selected single feature matches
DoM (best 0.453 vs CAA 0.306), though one feature outside the attribution top-50
(f1872, found only by cosine-to-CAA) reaches 0.339 — the DoM steer is ≈ one SAE
feature at the *representation* level, but FRA attribution cannot *select* it.

## Flags / open items

- **Abstract is stale** (Jamie's commit says "Abstract/intro intentionally untouched"):
  still claims "Pareto-dominant steering", "perfectly suppress … via FRA-based
  steering", and "28% … word-for-word" (table now says OV 36.9% / Conv 42.5%, and OV
  is no longer the winner). Needs a rewrite around detect/select/localize.
- Paper's sleeper table has **no oracle/attention-cut reference row** — our (0,0)
  oracle would make "no residual-space method restores clean exactly" quantitative.
- DoM row is single-run (no ±std) in the stats table.

## Reconciliation with Dmitry's May seed sweeps (added 2026-06-05)

Two prior results seemed to contradict the corrected OV-vs-Conv ordering; they don't.

1. **SAE scaling sweep** (May 25, `.claude/worktrees/sae-scaling-sweep/experiments/`
   `tinystories_sleeper/sae_scaling/RESULTS.md`): 270 ckpts, OV ≈0.46 tight vs
   conventional 0.47–0.74 noisy; rec-loss and steerability **decouple**
   (loss_recovered saturated; %err improves while opt_J_clean is flat).
2. **6-SAE-seed × 5-decode-seed α-sweep** (May 26, `experiments/tinystories_sleeper/`
   `rerun4_rescue_2026-05-26/results/jsd_alpha_sweep_6seeds.json`): mean opt_J_clean
   OV **0.455±0.073** vs conv 0.505±0.166 — but head-to-head OV wins only **3/6
   seeds**; the OV mean advantage is conventional's seed-2 selection blowup (0.838).
   Excluding it: conv 0.44±0.03 vs OV 0.45 — a tie, conv marginally ahead.

**Resolution:** "OV better on average" was a *tail* effect (conventional selection
occasionally fails catastrophically — the f579 checkpoint-coordinate pitfall,
`ketan_repl/notes/JSD_2x2_METHODOLOGY.md`). Jamie's corrected pipeline removes the
tail (per-seed α\* with ASR≤1% gate, Conv cosine re-rank, leakage-free eval,
retrained SAEs), so the underlying tie surfaces with Conv/DoM ahead by ~0.04
(≈0.6σ, overlapping bars). What survives: OV's lower seed-variance (still true in
his table, ±0.049 vs ±0.061) and the rec-quality⊥steering decoupling. Open question
for Jamie: is the Conv tail fixed or just unsampled at n=6 — did his min-ASR gate
ever have to reject a bad winner?

## Branch map

| branch | role | tip (2026-06-05) |
|---|---|---|
| `jamie/sleepers-final` | consolidated repro pipeline, sleeper-only | `06d3237` (overview FRA numbers + Fig-12 eval prompts) |
| `jamie/paper` | paper-only branch (18 referenced figures) | `9387f54` (corrected results sync) |
| `jamie/autoresearch-jsdc` | full history + OV-selection autoresearch (LOSO, cross-SAE) | `3bff8d4` (appendix D rewrite) |
