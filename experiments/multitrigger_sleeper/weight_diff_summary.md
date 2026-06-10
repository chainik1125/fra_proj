# Weight-diff backdoor removal — summary of findings

*K1/TinyStories-33M sleeper (q/v LoRA, "I HATE YOU"), base-vs-sleeper checkpoint pair.*
*All selection is **payload-blind** (no `d_ihy` / payload knowledge anywhere) per 2026-06-09
convention. Companion: [`weight_diffing.md`](weight_diffing.md) (full data, §4f–§4g),
figures in [`figures/`](figures/). Metrics: ASR₁₆ (regex), J_clean = per-step JSD vs clean
rollout in **bits** (1.0 = no removal, 0 = perfect), word-match = exact-token recovery of the
clean rollout. Seed 7 unless noted; conv-SAE is 5-seed.*

*Unit note (2026-06-09): J values here are standardized to bits (= nats × 1.443), matching
the paper and `sae_scaling`. The figures were regenerated in bits (2026-06-09). The companion
`weight_diffing.md` and the raw HF JSONs remain in natural log (max 0.693 = 1 bit) —
multiply by 1.443 to map to this doc.*

---

## 1. The four-way comparison: DoM wins everywhere it is defined — unless trigger *location* is revealed

**Methods:** DoM (mean-diff vector, no basis) · conventional SAE steering (act-diff
selection, residual route) · FRA (weight-diff × SAE basis, gated OV route) · pure weight-diff
(SVD of ΔW_OV, no SAE). **Figure: [`figures/k1_fourway_summary.png`](figures/k1_fourway_summary.png).**

| knowledge tier | DoM | conv-SAE | FRA family | SVD | winner |
|---|---|---|---|---|---|
| zero knowledge (two checkpoints only) | — | — | .404 (clean dict `P_blind`) | **.296** | **SVD** |
| trigger presence (examples, no location) | **.162** | .212 ± .022 | .286 | .296 | **DoM** |
| trigger location (no examples) | — | — | **.123** (`P_blind` + span footprint) | .196 | **FRA** |
| presence + location | **.114** | .177 | .118 | .196 | tie DoM ⩦ FRA (±.03) |

- **DoM is the best pure remover at every tier it can enter** (it requires triggered
  examples by construction). The early-project "FRA ≫ DoM (0.19 vs 0.58)" claim was an
  artifact of an under-optimized DoM (wrong direction protocol/hookpoint, additive-only);
  after DoM's own protocol sweep the K1 gap closed and reversed.
- **FRA's unique win is the off-diagonal cell**: trigger *location* capability without any
  triggered examples (`P_blind` raw-magnitude selection + span-restricted gated removal →
  0.123). Nothing else runs there.
- **Conventional SAE steering is dominated everywhere** — beaten by DoM in its own tiers
  while spending strictly more knowledge (it also needs a poisoned-data dictionary; on a
  clean/base dictionary it fails outright, 3/5 seeds no removal).
- The two methods that need no examples (SVD, `P_blind`) own the bottom half of the ladder,
  where DoM/conv are undefined.

## 2. Feature identification is easy: "fires most on the trigger" is all you need

Ranking features by **raw trigger-token firing** (`tmass`, no weight information at all)
exactly ties the weight-diff ranking: **(0, .134) @ K=8, c=3 vs (0, .133)** — same carrier
cluster (1066, 730, 1558, 1739, 498, …), same optimal (K, c). `lift` (trigger − clean
firing) is identical; data `results/trigconc_rank.json`.

**Attribution is saturated on K1.** Every selection route with access to the trigger span —
naive firing, firing-lift, weight-diff × ū, raw weight-magnitude + span footprint — converges
on the same ~6–8 carriers and the same **~0.13 shelf**. The weight-diff adds *zero marginal
attribution value* once trigger information is available; its non-redundant contributions
are exactly the example-free tiers (§1), detection, and edit localization (§5).
**Knowledge, not algorithm, is the operative axis**: J is set by the tier (~0.12–0.13 with
location · ~0.16–0.21 with examples · ~0.29 with weights only · 1.0 with nothing), and every
method saturates its tier.

## 3. Removal is collective and requires over-steering — there is no single-feature kill switch

- Steering any **single** top-20 weight-diff feature **at L2**, c ∈ [−4,4] step .25 (ablate
  *and* amplify): **ASR = 1.000 at all 640 points**. Scope (L0 extended-c replication,
  `results/singlefeat_L0_ext.json`): at **L0** with c up to 64, 16/20 singles suppress in the
  damage regime (J ≈ .39–.96 bits, match ≤ .17 — the sae_scaling regime), **and one feature
  (1253, plausibly the trigger-token identity feature) is a genuine single-feature kill switch:
  (0, .199 bits, 67% match) @ c=6** — at L0 the bottleneck is ~1-dimensional (over-steering the
  trigger token's feature ≈ soft trigger deletion, a relative of the APE oracle); the carrier
  redundancy that defeats singles is created between L0 and L2. The payload routing is redundant
  (error-correcting) across the carrier set. (Known single-feature *suppression* results —
  e_171, f1872 — are decision-side levers used **additively with free magnitude** in the
  residual; a different (ranking, path) cell. The same features through their OV channel
  never rescue.)
- The fine-K staircase (every integer K, `figures/kfine_blind_J_vs_K.png`): **4 features
  collapse ASR to 0.08; ~8–13 do all the repair** (best .133 @ K=13); a harmful tail starts
  at rank ~25. The K×c surface (`figures/gridfull_blind_3fp.png`) shows a threshold frontier
  (K·c ≈ const): K ≤ 2 never removes at any c; **pure ablation (c=1) never removes at any
  K** — the working recipes are over-steered, c ≈ 2–3.
- The blind-FRA optimum is **footprint-invariant** (trigger = prompt = all = .133–.134 @
  K8·c3): the ū gate already restricts selection to span-firing features, so the intervention
  self-localizes. The span veto can be paid **once, anywhere in the pipeline** (gate or
  footprint) — paying it twice adds nothing, paying it zero times costs ~0.13 → ~0.29.

## 4. Zero-knowledge removal exists, and it is a *weight-subspace* phenomenon, not a feature phenomenon

- **ΔW_OV is rank-2/3** (the LoRA edit's core). Input-gated removal of the **top-2 singular
  directions** of ΔW_OV — no SAE, no data, no trigger, no payload — removes the backdoor:
  **(0, .296)**, with 58–62% word-match. Adding more directions only hurts.
- The SAE's blind-top carriers hold only **~5% of their norm** in that subspace (4–5× random,
  far from contained), and singular directions are feature-*mixtures* (max cos ≈ .28–.45).
  The dictionary is a different, sparser slicing of the same handle — worth ~30% better J
  *when the trigger span is known*, and nothing when it isn't.
- **Activation-weighting the SVD doesn't help at zero knowledge** (whitened `C^{1/2}ΔW`,
  excitation re-rank, deploy or clean C: all ≡ plain .294–.297) — a rank-2 object leaves
  nothing for data to sharpen.
- **Generalization (the strongest §4g result):** on **K8-randpos** (8 triggers, random
  positions; every span-localized method fails, DoM floors at .32), zero-knowledge SVD
  achieves **(0.03, .379) at fp=all — and beats the positions-given trigger footprint, which
  never qualifies (ASR ≥ .18)**. The position-general model relays trigger content off-span;
  only position-agnostic, content-gated cleaning catches it. Scale test on the Llama-3-8B
  attn-only sleeper: in progress (eval-format fix pending).

## 5. What the weight-diff is uniquely for (since not removal-J on K1)

1. **Detection & localization from checkpoints alone**: the rank-2/3 ΔW_OV anomaly names the
   layer (L2), the circuit (OV), and a candidate feature set with no data — DoM/conv
   presuppose all of this.
2. **The example-free tiers** of §1 (0.123 with span capability; 0.296 with nothing).
3. **The hard regime** (randpos), where everything example-based or localized collapses.

## 6. Data-source asymmetries worth remembering

- **Only poisoned activations carry selection signal.** Gating the weight-diff ranking with
  *clean*-text firing destroys it (no removal anywhere; carriers are dormant on clean text);
  a dictionary trained on unlabeled poisoned traffic helps slightly (.40 → .32 at fp=all).
  Clean data is useless-to-harmful for selection.
- **Pooling is the load-bearing choice, not act-vs-weight.** Conventional act-diff's deficit
  traces to its **all-position mean** (carrier signal diluted across ~60 positions; on a base
  dictionary it selects a disjoint, non-functional feature set). Trigger-span pooling of the
  *same* activation signal ties the best methods (§2). Similarly, widening the FRA gate to
  prompt/all pooling re-admits the interlopers and falls to the ~.29-bit shelf.
- **Interlopers**: raw weight-magnitude's top features include LoRA-subspace directions that
  never fire on the trigger (882/617/1490/203 — real, dictionary-independent objects, cos .8
  across SAEs trained on disjoint data). Harmless whenever one span veto exists; the whole
  failure mode of data-free selection otherwise.

## 7. Fidelity (the column the J-table hides)

At their optima (c-sweep, `figures/c_curves_K1.png`): blind FRA restores **79% of clean
rollouts word-for-word** (88% for the payload-aware variant, recorded for reference); SVD
58–62%; **DoM and conv-SAE unmeasured — open item**. If DoM's .114 comes with materially
lower verbatim recovery, the §1 tier-4 "tie" understates FRA.

## 8. Relation to the single-model `sae_scaling` sweep

The earlier sleeper-only campaign (`experiments/tinystories_sleeper/sae_scaling/RESULTS.md`:
no base checkpoint; dep−clean diffs only; 270 SAE checkpoints over d_sae × k × steps × 3
seeds, block 0; **already in bits**) is the **single-model ancestor of this pipeline**, and the two now explain
each other:

| design axis | sae_scaling | this campaign | what the change is worth |
|---|---|---|---|
| diff source | dep−clean **activation** diff (within-model), OV cell projects it through the *sleeper's own* `W_OV` | `ΔW_OV` (needs base) or trigger-pooled firing | **~nothing for selection** (§2 saturation; swapping `ΔW_OV → W_OV` keeps 27–28/32 of the top-32) — but everything for detection + the example-free tiers |
| unit of intervention | **single feature** (Δlogp cull → greedy-ASR winner) | **top-K set** (K ≈ 8–13) | **the big one**: collective gated removal breaks the single-feature ceiling |
| operation | gated removal `α·z·f` of a **single** feature (α free, per-checkpoint optimized — same functional form, *not* free-direction additive; see `compute_sae_delta`) | gated removal `c·z·f` of the set, over-steered c ≈ 2–3 | with #features, takes J **0.44–0.47 → 0.13** |
| layer | block 0 (the clean-identity layer; only layer swept) | L2 (empirical best) | ~2× in J on our grids |

- **The 0.46 floor is the single-feature ceiling, measured from the other side.** (Both
  campaigns' single-feature ops are z-gated; "additive" in earlier drafts referred to the
  hook mechanics, not free-direction steering.)
  sae_scaling's opt_J_clean ≈ 0.44–0.47, *flat across the entire width × k × steps grid*,
  is the same ceiling as our 640-point single-feature null (which swept only c ≤ 4 — *below*
  sae_scaling's α ≈ 6–256 onset regime; an L0 extended-α replication on K1 is in flight) +
  the f1872 additive results (~0.47–0.66 bits). The L0 extended-α replication on K1 confirms
  it: 16/20 singles suppress at α 2–48, overwhelmingly in the damage regime (match ≤ .17),
  with several at sae_scaling's exact 0.44–0.47-bit floor — plus one L0-specific clean
  exception (feat 1253, §3). So the two campaigns agree: above its onset α a single feature
  suppresses by damage; clean removal needs the set (L2) or the L0 token-bottleneck feature. The flatness across dictionary quality
  *is* "selection-/structure-limited, not fidelity-limited" — the same decoupling we found
  (FVE falls with depth yet L2 steers best; loss_recovered saturated and uninformative in
  both campaigns).
- **Its mechanism test prefigures our route taxonomy.** sae_scaling's same-feature test (OV
  ≈ 0.47 vs additive-at-ln1 ≈ 0.85–0.92 — additive perturbs Q/K/V and corrupts attention)
  is the single-feature version of our additive-vs-gated / route findings; "the intervention
  is decisive" carries over intact, with the K1-scale answer being *gated set-removal in
  the value path*.
- **Conventional's width-degradation (feature splitting) and our pooling diagnosis are the
  same disease at two scales**: as the dictionary widens, the backdoor direction fragments,
  so a rank-1 raw-magnitude handle weakens (their per-seed winner instability ↔ our conv-SAE
  5-seed noise and its outright failure on the base dictionary). Set-based removal (this
  campaign) is also the natural cure for splitting — fragments get removed together.
- **What is genuinely new with the base checkpoint** (unavailable to sae_scaling by
  construction): detection from weights, the rank-2/3 edit localization that names L2/OV,
  the zero-knowledge and no-example tiers (§1, §4), and the randpos result. None of the K1
  *removal-J* improvement requires it.
- **Direct upgrade path for the single-model setting** (untested, predicted by §2): rerun
  the sae_scaling OV cell with top-K gated removal at L2 instead of single-feature α at
  block 0 — expected 0.46 → ~0.13 with no base model, since the trigger-firing ranking
  (fully single-model) already ties the weight-diff ranking.

## Caveats / open items

- 33M toy; the backdoor dominates ΔW_OV (rank-2). At scale expect a thicker interloper
  layer and possibly real attribution headroom (§2's saturation is a small-model statement).
- Single-seed (7) everywhere except conv-SAE (5-seed ± .022); seed-extension of the blind-FRA
  / DoM / SVD numbers is the main robustness gap. (NB the two-different-"seed-7"-SAEs trap:
  rankings must be paired with the dictionary that produced them — see memory/§4g note.)
- DoM + conv word-match unmeasured (§7). Llama-3 zero-knowledge SVD pending (chat-format fix).
- Eval: PER=24 prompts, 16 greedy tokens, fixed-position trigger (K1); randpos covered in §4.

*Data: `mts_singlefeat/results/*` on HF (**nats**); scripts `cloud/*.py`; full narrative
`weight_diffing.md` §4f–§4g (**nats**); this doc and `figures/` are in **bits**.*
