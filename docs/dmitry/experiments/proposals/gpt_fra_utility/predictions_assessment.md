# Assessment of the ten pre-registered predictions in `fra_sleeper_experiment_plan.tex`

*Claude review, 2026-06-05. Written **before** running any of the new experiments, against:
the proposal (`fra_sleeper_experiment_plan.tex`), the sprint results
(`experiments/multitrigger_sleeper/summary.md` incl. the 06-05 single-feat/CAA-decomp/BO/grad-steer
addendum), and the theory notes (`docs/dmitry/theory/sleeper_trigger_ape_patching.md`,
`fra_qk_side_derivation.md`, `fra_general_theory.md`, `fra_ovov_headroom_bound_experiments.md`).
Where I disagree, I register my own prediction so the disagreement is falsifiable.*

**Overall verdict.** The proposal's *framing* (detect ≠ payload ≠ control; FRA = diagnosis /
localization / bounded explanation; the attention cut is the lever) is right and matches the data.
Most predictions are sound. I register **two substantive disagreements** (Exp 10's MLP-route and
RoPE predictions — I think the theorem says the opposite), **one prediction already partially
falsified** by the 06-05 addendum (Exp 6), and several **protocol-level corrections** (Exp 3's
correlation design, Exp 5's confusion of cumulative-order with minimal coalitions, Exp 2's
linear-probe framing).

---

## Scorecard

| Exp | Proposal prediction (compressed) | My position | Status before new runs |
|---|---|---|---|
| 1 | Span-aware cut fixes multi-token: ASR≤0.05, FP≤0.02, J≤0.05 | **Agree**, and sharpen: expect span recall ≈1.0 and J≈0, not just ≤0.05 | Failure precisely diagnosed (closing-`\|` sub-token); fix untested |
| 2 | FRA-cut ≈ known-position oracle; fixed-pos oracle ≥0.75 ASR; heuristics underperform; probe may tie in-dist | **Mostly agree**; disagree on attention-mass (recall will be high, FP is its problem) and on what the probe tie would mean | Half-confirmed already (fra_detect 0.00 / oracle_fixed 0.92) |
| 3 | Raw QK ρ≤0.25; actionability-weighted ρ≥0.50; demotes detector | **Agree direction, disagree mechanism + design**: weighting is a *per-key scalar* — within one (q,k) cell it cannot reorder pairs; expect Jacobian factor to explain ~all of the 400× mismatch, ρ≥0.8 | Mismatch measured (ΔS 0.40 → ΔA 9e-4); correction never computed |
| 4 | FRA-OV top-16 ≥80% recovery, ≥20pp over grad×act | **Half agree**: ≥80% plausible; **≥20pp over grad×act is optimistic** — FRA-OV *is* grad×act restricted to the frozen-pattern V-path | Untouched (needs new task suite) |
| 5 | No coalition <8 feats; minima 12–24; J stays ≥0.30 | **Genuinely uncertain — best-information experiment.** Cumulative-by-activation order ≠ minimal coalition; I put ~40% on greedy finding <8 for single-token triggers | Only fixed-order cumulative curve exists |
| 6 | 4–8 CAA-decomposed features reach ASR 0, J≤0.35; FRA-OV features won't | **Already partially falsified**: *m=1* (f1872, cosine-screened) reaches (0.00, 0.339). And the true floor (grad-steer 0.15–0.20) is *outside* the SAE span — predict OMP m≤8 plateaus ≈0.30, never approaches 0.15 | Mostly done (cosine + par/perp decomp); OMP/LASSO missing |
| 7 | Union-of-spans cut exact for co-present triggers | **Agree** — near-corollary of the per-key theorem; cheap confirmation | Untouched |
| 8 | Oracle flat to K=32; degradation only via detector collisions | **Agree**, but low information density (oracle flatness is analytic; collisions unlikely at d_sae=2048) — deprioritize | K≤8 done |
| 9 | No SAE config turns the detector into a single-feature lever | **Agree** — payload isn't a single *direction* (multi-direction floor 0.54), so no basis rotation can sparse-ify it; seed-1 replication already points this way | One SAE + seed-1 replication |
| 10 | (0,0) exact only for APE+attention-route; RoPE inexact; MLP-route cut fails | **Disagree on 2 of 3 variants** — see below. This is the highest-value experiment to run | Untouched |

---

## The two substantive disagreements

### D1 (Exp 10, MLP-route): the theorem is about the *channel*, not the *weights*

The proposal predicts: *"In MLP-route or output-head sleepers, attention cut will fail or only
partially suppress."*

I think this conflates **which weights store the backdoor** with **which channel transmits the
trigger**. The parallel-token theorem (`sleeper_trigger_ape_patching.md`, Part I) is an
architecture-level statement: in a transformer, a previous token reaches the next-token
distribution through exactly two channels — the attention it receives as a *key/value source*, and
the additive positional embedding of the slot it occupies. The MLP is per-position; it cannot move
trigger information across positions. So even if the backdoor is stored entirely in MLP weights
(LoRA on MLP, attention frozen at base values), the trigger's *content* still has to ride the
attention channel to reach the generation position — the MLP-modified residual at the trigger
position is only read downstream via attention to the trigger key.

**My pre-registered prediction:** for an MLP-route APE sleeper, the same oracle (zero post-softmax
attention to the trigger span at all layers, every decode step, + APE re-index) reaches
**(ASR, J_clean) = (0, 0) exactly**, identically to the Q/V-route case. The only way it fails is if
the "backdoor" isn't actually trigger-conditional (e.g. an unconditional output-head bias — but
then it's not a sleeper).

Either outcome is valuable: if I'm right, C1 generalizes from "APE attention-route sleepers" to
"**any** trigger-conditional backdoor in an APE transformer," which is a much stronger headline.
If the proposal is right, there is a token-mixing channel the theorem misses, and the theory note
has a hole.

### D2 (Exp 10, RoPE): re-index `position_ids`, not `W_pos` — then it should be exact again

The proposal predicts RoPE models leave "nonzero J_clean because the clean positional
counterfactual is not an exact APE re-index." But the theory note itself
(`sleeper_trigger_ape_patching.md`, Takeaway 4) gives the fix: in RoPE, position enters only
through the relative rotation in QK, so the clean positional counterfactual is obtained by
**re-indexing `position_ids`** for post-trigger tokens. After cutting attention to the trigger
key, all surviving attention edges are between non-trigger tokens, whose re-indexed relative
offsets equal the clean prompt's offsets exactly.

**My pre-registered prediction:** RoPE sleeper + attention cut + `position_ids` re-index reaches
teacher-forced J ≈ 0 and ASR = 0, i.e. the (0,0) point is **not APE-specific** — only the *form*
of the positional fix is. Residual caveat: implementation details (KV-cache position handling)
may add float-level noise, not structural error.

### Already-falsified corner (Exp 6)

The proposal predicts "a sparse approximation using 4–8 SAE features ... reaching ASR=0 with
J≤0.35," with the implication that *several* features are needed. The 06-05 addendum already shows
**one** feature (f1872, |cos|=0.207 to CAA, found by cosine screen — *not* by FRA-OV attribution)
reaches (0.00, **0.339**). So the m≥4 part is falsified; the "FRA-OV attribution features won't
match" part is *confirmed* (best attribution-selected single: 0.453). What remains open and worth
~30 min of GPU: does OMP/LASSO with m∈{2,4,8,16} beat 0.31 (the full-CAA optimum is 0.269,
α-optimal), and — my prediction — **no m-feature combination approaches the gradient-steer floor
of 0.15–0.20**, because the optimized vectors are ≈orthogonal to the whole dictionary
(max |cos| ≈ 0.13).

---

## Protocol corrections (agree with the question, disagree with the design)

### Exp 3 (actionability-weighted QK): the weighting is per-key, so design the sample accordingly

For a score perturbation δs concentrated at one key j of one query row q, the first-order pattern
effect on a target with OV profile g is

  g·J_softmax(s_q)·δs = δs · A_qj (g_j − ḡ_q),  ḡ_q = Σ_k A_qk g_k

(this is exactly the "centered OV profile" already derived in `fra_qk_side_derivation.md` §3 —
the proposal's Eq. (7) machinery **already exists in the theory notes**; the experiment is to
*validate* it, not invent it). Two consequences the proposal misses:

1. **Within one (q,k) cell, actionability is a positive scalar times the raw score** — it cannot
   reorder feature pairs at the same key. ρ(raw, actual) and ρ(actionable, actual) are *identical*
   if all sampled pairs sit at the trigger key. The correlation test must sample pairs **across
   keys, rows, heads, and layers** for the comparison to be non-vacuous.
2. **Quantitative pre-check:** the measured 400× mismatch (predicted ΔS≈0.40 → actual ΔA≈9e-4) is
   consistent with A_qj(1−A_qj)·ΔS at A_qj ≈ 0.002 — i.e. the Jacobian factor alone likely
   explains essentially the whole gap. **My prediction is stronger than the proposal's:**
   Jacobian-weighted predictions will match actual pattern changes within a factor ~2 for small
   perturbations (Spearman ρ ≥ 0.8, vs the proposal's ≥0.50), and the detector feature is demoted
   automatically.

And one deflationary register: even a perfect actionability score does **not** rescue QK-feature
ablation as a defense. The oracle works by setting A→0 — a non-perturbative move outside the
first-order regime. Expected leverage is small *everywhere* on the key side (that is C3 restated).
Success in Exp 3 upgrades FRA-QK from "descriptive" to "locally faithful with the Jacobian
correction" — it does not produce a new lever. The protocol must also route ablations through
`hook_k` only (the V-side confound is flagged in `fra_qk_side_derivation.md` §10).

### Exp 5 (coalition search): the cumulative curve does not bound minimal coalitions

The existing evidence (top-1..8 by activation does nothing; need ~16/32) is about one *fixed
ordering*. The proposal's prediction ("no subset of fewer than 8 ... reliably") treats that as
evidence about *all small subsets*, which it isn't — greedy-by-marginal-ASR explores a different
part of the lattice. I genuinely don't know the answer (this is the experiment where my
uncertainty is widest): distributed *reconstruction* mass is compatible with a small *causal*
cut-set. I put ~40% on greedy finding a suppressing coalition of <8 features for single-token
triggers (where blank-all reaches ASR 0.00), and I'd weakly bet the J_clean of small coalitions
undercuts 0.30 *if* they exist (fewer features removed ⇒ less collateral). If instead minimal
coalitions are 12–24 as predicted, that's the strongest version of "payload is distributed" yet.

### Exp 2 (random-position benchmark): pre-commit to the deflationary reading

Half of this prediction is already confirmed (fra_detect (0.00 ASR, 0.00 FP) vs oracle_fixed 0.92).
On the new baselines:

- **Attention-mass heuristic:** the proposal predicts it underperforms on "recall or false
  positives." I'd sharpen: its *recall* will be decent — the backdoor literally works by attending
  to the trigger (mass at L2–L3, Fig 5) — but it has no principled threshold, so its **clean-FP
  rate** is where it loses. Pre-commit to reporting the full ROC, not one operating point.
- **Linear probe:** will match FRA in-distribution with near-certainty. The proposal's fallback
  framing ("feature localization is interpretable and sparse") is correct, and I'd pre-commit to
  it now rather than treating the probe tie as a surprise. FRA's genuine differential advantage:
  the detector threshold comes for free from the SAE's own statistics (no labeled trigger data
  needed at calibration time) — test the probe with *and without* held-out trigger families to
  make this crisp.
- **Note:** the random-position sleeper from §4c was trained on-the-fly and not persisted — the
  pod must retrain it (~20 min) before evaluating baselines.

### Exp 4 (FRA-OV completeness): the ≥20pp margin is against the wrong baseline

FRA-OV's contribution `P_qk · u_k^λ · ⟨t, W_OV f^λ⟩` *is* gradient×activation restricted to the
frozen-pattern value path. Against full-model grad×act (which includes QK-path gradient terms) a
margin can exist on pattern-sensitive tasks, but ≥20pp against *all* listed baselines is
optimistic; expect 5–15pp vs grad×act, larger vs activation-only and random. The ≥80% top-16
recovery on copy/induction targets is plausible (token-identity features are sparse in
TinyStories). Lower priority: requires building a new task suite rather than reusing the sleeper
substrate, and the theory already tells us the frozen-pattern decomposition is exact — the open
question is concentration, which Exp 5 probes more directly on the substrate we care about.

### Exps 1, 7 (span-aware cut; co-present triggers): agree, run as confirmations

- **Exp 1:** the multi-token failure is already mechanistically diagnosed (detector fires on the
  closing `|` sub-token; cutting one position misses the 4–6-token span; `oracle_known` with the
  true span achieves 0.00). Delimiter-pairing span expansion should give span recall ≈ 1.0 in this
  substrate (clean TinyStories text contains no `|`), so I sharpen the prediction to ASR ≈ 0,
  FP ≈ 0, J ≈ 0 — *exactly* matching `oracle_known`, not merely ≤0.05. The interesting
  failure-mode to watch: leading-`|` detection asymmetry (does the *opening* delimiter fire the
  feature too?) — irrelevant once expansion is delimiter-to-delimiter.
- **Exp 7:** near-corollary of C1 (the oracle is per-key; cutting the union of spans + cumulative
  re-index composes). Predict exact (0,0) for the known-union oracle; FRA-detected union matches
  for single-token triggers and inherits Exp 1's fix for multi-token. Worth running because
  "cut only one of two live triggers leaves ASR high" is the one genuinely untested sub-claim
  (trigger redundancy/interference at generation time is an empirical question).

### Exps 8, 9: agree, deprioritize

- **Exp 8** (K=16,32): oracle flatness is analytic (per-key surgery); detector collisions at
  d_sae=2048 with ≤32 triggers are unlikely. Run only if budget remains.
- **Exp 9** (SAE sweep): the deep reason detector-ablation fails is that the payload is not a
  single direction (directional-ablation floor 0.54; gradient-steer optima ≈orthogonal to the
  dictionary), and *no basis rotation can make a non-1-D object 1-sparse*. A better SAE improves
  the error budget, not the rank of the payload. Seed-1 replication already covers the cheap
  version of this. Run a 2–3 point hookpoint/width spot-check at most.

---

## Execution plan (this sprint, RunPod, est. $4–6 total)

Priority order, balancing information value and cost:

| # | Pod | Experiments | Why | Est. |
|---|---|---|---|---|
| P1 | `mts_exp10_routes` | Exp 10: MLP-route sleeper (APE) + RoPE (pythia-70m) sleeper; oracle with the right positional fix on each | Decides D1/D2 — the two registered disagreements; biggest claim upgrade if I'm right | ~$1.5 (training 2 sleepers + eval) |
| P2 | `mts_exp1_7_span` | Exp 1 span-aware detect-then-cut + Exp 7 co-present union-of-spans | Turns §4c's known failure into the headline positive claim ("detect→lift-to-span→cut→re-index") | ~$0.75 |
| P3 | `mts_exp3_qk_act` | Exp 3 actionability-weighted QK (across-key sampling, hook_k-only routing, quantitative Jacobian check) | Cheap; closes the QK-faithfulness story either way | ~$0.5 |
| P4 | `mts_exp5_coalition` | Exp 5 greedy/backward coalition search (+ Exp 6 OMP add-on in the same pod) | My widest uncertainty; OMP completes the already-mostly-done Exp 6 | ~$1 |
| P5 | `mts_exp2_baselines` | Exp 2 baseline suite (attn-mass, act-norm, resid-cosine, linear probe) on a retrained random-position sleeper | Fairness suite for the positive claim; retraining adds cost | ~$1 |

Deferred: Exp 4 (new task suite), Exp 8 (low info), Exp 9 (spot-check only if pods come in cheap).

All pods follow the `launch_pod_singlefeat.sh` pattern (A40-class, HF
`dmanningcoe/fra-phase1-steering-data :: mts_singlefeat/`, sidecar uploader, self-terminating).
No LLM judging anywhere in this plan (ASR is regex; J is JSD) — zero API-token cost.

---

# RESULTS (added 2026-06-05/06, after the pods ran — everything above is untouched pre-registration)

*Five pods on RunPod A40s (`launch_pod_mts.sh`), result JSONs in
`experiments/multitrigger_sleeper/results/{span,routes,qk_act,baselines,coalition}_results.json`
and on HF `mts_singlefeat/results/`. n=24 pairs/trigger throughout (one-prompt resolution ≈ 0.042).*

## Scorecard: proposal vs my register vs outcome

| Exp | Proposal predicted | I predicted | Outcome |
|---|---|---|---|
| 1 span-cut | ASR≤.05, FP≤.02, J≤.05 | same + recall≈1.0, J≈oracle | **Works**: multi-token ASR 1.00→**0.02**, recall **1.0**; FP 0.042 (=1 prompt, below n=24 resolution), J 0.057 (over-cut at precision .90–.96). Proposal's letter narrowly missed, spirit confirmed |
| 2 baselines | FRA≈oracle_known; heuristics lose; probe may tie | attn-mass: recall ok / FP bad; probe ties | FRA (prec 1.0, **FP 0.00**, ASR .031, J .021, AUC 1.0) ≈ oracle_known ✓. **Probe does NOT tie**: recall 1.0 but FP 0.333, J_clean .059. attn-mass **recall 0.0** (AUC .79) — proposal right, **my version wrong**. Surprise: resid-cosine is the real challenger (rec 1.0, ASR 0, FP .042) |
| 3 QK action. | raw ρ≤.25; weighted ρ≥.50 | weighted ρ≥.8; Jacobian explains the 400× | raw ρ=**0.55** (proposal wrong — raw is mediocre, not useless); weighted ρ=**0.80** ✓(both); calibration 400×→**~2× at median** (only 37% within 2×, so "bulk, not all"). Wrinkle: for *target-level* effects raw beats weighted (.72 vs .55) — per-row first-order misses the cascade |
| 5 coalition | no subset <8; minima 12–24; J≥0.30 | ~40% chance greedy finds <8 | **Proposal right, my hunch wrong, and then some**: greedy-forward plateaus at ASR 0.50 by 9 feats (16 feats: still 0.50, J 0.44); ablating **all 196** pooled candidates leaves ASR 0.104 — *no* coalition of any size hits ≤0.05 under the pooled criterion; J of suppressing-ish sets ≥0.43. (Caveat: pooled-over-4-triggers is harder than the proposal's per-trigger framing; greedy's first picks are the detector features f807/f1365 — picked and useless, C3 again) |
| 6 OMP | 4–8 feats reach J≤.35 | m=1 already does (f1872); OMP plateaus ≈.30, never ≈.15 | **Both partially right**: OMP picks f1872 first ✓; m=1→0.339 (so the "need 4–8" substance was wrong) but m=4→**0.287** *beats* full CAA@α2 (0.306) and m=16→0.273 ≈ α-optimal CAA (0.269). Plateau ≈0.27–0.29, **never approaches the 0.15–0.20 gradient floor** ✓(mine). cos(v16,CAA)=0.42 only — suppression needs little of CAA's direction; the floor lives outside the dictionary span |
| 7 co-present | union cut exact; cut-one fails | agree | **Exact**: union-known TF-J ≈ −1e-8; cut-one leaves ASR 0.92–1.00 ✓ |
| 10 MLP-route | attention cut **fails** | **(0,0) exactly — channel ≠ weights (D1)** | **D1 confirmed, proposal falsified**: MLP-only sleeper trains to ASR 1.0; oracle → ASR 0, J_roll ≈ −1e-8, J_tf ≈ 0, all triggers |
| 10 RoPE | J_clean > 0 (no exact counterfactual) | **(0,0) with position_ids re-index (D2)** | **D2 confirmed, proposal falsified**: pythia-70m RoPE sleeper, cut + position_ids re-index → ASR 0, J ≈ +1e-9 |

Not run (deprioritized as pre-registered): Exp 4 (new task suite), Exp 8 (oracle flatness analytic), Exp 9
(spot-check only), Exp 10 output-head variant.

## The headline upgrade

The two falsified Exp-10 predictions upgrade C1 from
"APE attention-route sleepers" to:

> **Any trigger-conditional backdoor in a transformer falls to the attention cut + the correct
> positional re-index, regardless of which weights store it (Q/V, MLP) and of the PE scheme
> (APE → swap `W_pos` rows; RoPE → re-index `position_ids`).**

Supporting texture: (i) the MLP-route sleeper has a *different causal-read structure* (L0 masking
alone suffices, ASR→0; L2/L3 only partial; L1 inert) yet the same oracle exactness — the channel
theorem doesn't care; (ii) the mask-only footprint is PE-diagnostic: APE leaves the
w-deterministic absolute-slot footprint (0.058–0.082 here), RoPE leaves ≈nothing for w=1
(0.0005) but a width-growing relative-offset distortion for w=4–6 (0.071–0.077).

## New findings not anticipated by either side

1. **Zero-shot localization exists in the SAE basis at high recall but terrible FP** (Exp 2
   held-out folds): the rare-feature-anomaly detector localizes *unseen* trigger tokens at
   recall 0.92–1.0 (ASR→0.00–0.08) where fra_detect, resid-cosine, and the probe all collapse to
   ~0 — but at clean-FP 0.94 at this operating point (AUC 0.91, so better points exist). "FRA
   uniquely localizes" should be replaced by: *supervised* localization is commodity (cosine
   ties), *zero-shot* localization is real but needs an FP story.
   **REVISED by the three-run zeroshot follow-up (`zeroshot_v3_results.json`):** the FP story
   was a ceiling-coverage estimation artifact, and chasing it overturned the recall claim too.
   At well-estimated ceilings (384 clean prompts) the natural-word triggers' margins vanish
   (banana median excess 16.1→−0.1 — its "exclusive" feature legitimately fires on clean text):
   single-token zero-shot recall plateaus at 0.25 (thunder only, ASR 0.73–0.79). Multi-token
   `|WORD|` triggers go the other way: span recall 1.0 at FP 0.042, post-cut ASR 0.021,
   J_clean 0.003 — a genuinely deployable free defense. Final form of the finding:
   **zero-shot SAE-novelty detection partitions triggers by surface OOD-ness, not
   backdoor-ness** — perfect for syntactically weird triggers, fundamentally blind to
   natural-word ones. (My follow-up pre-registration — "Q1 flips to yes at c=384, banana
   marginal" — was **wrong**: I attributed the margin to the trigger when it belonged to the
   estimator. Scored in the ledger below.)
   **The unknown-natural-word case, measured directly (`causal_detect_pod.py` + `_v2`):** since
   natural-word triggers are representationally invisible but causally active, two pods tested
   zero-shot *causal* detectors (score each position by the effect of the oracle's own cut),
   along a knowledge axis: surface novelty → target-agnostic output-swing → target-aware
   (onset token) → target-aware (full payload phrase). **All hit the same per-prompt wall.**
   recall@FP≤0.05 on single-token natural-word triggers: novelty 0.25, knockout_swing 0.00,
   knockout_ihy(onset) 0.03–0.04, knockout_phrase 0.00 — while AUCs are 0.89–0.99 (the ranking
   *information* is present). The wall is the multiple-comparisons structure: ~110 positions/prompt,
   and clean prompts contain structural positions whose knockout swings the payload logprob by up
   to ~18 nats, dwarfing the trigger's ~0.1–4 nat effect. **Decisive conclusion: an unknown
   natural-word trigger with no examples is not detectable at a usable operating point by any
   tested zero-shot signal, at any target specificity — it is a property of the trigger being a
   real word woven into normal computation, not a missing trick.** Both my causal pre-registrations
   (target-aware rescues it; then phrase-specificity rescues it) were **wrong** in the same
   informative direction: rank ≠ per-prompt separability.
2. **Residual-cosine is the honest baseline to report against** (not the probe): recall 1.0,
   ASR 0, FP 0.042 — within one prompt of FRA on everything. FRA's edge is the per-feature
   threshold calibration (FP 0.00 vs 0.042) and interpretability, not raw localization power.
3. **Raw FRA-QK is better than the proposal feared** (ρ=0.55 across keys) **and the Jacobian
   correction is exactly the missing factor for pattern-level effects** — but neither version
   predicts *target-level* effects well (cascade through downstream rows). FRA-QK's honest
   advertisement: locally faithful for the pattern with the centered-OV Jacobian weight;
   descriptive beyond that.

## Falsification ledger (what would change my mind)

- If the **MLP-route** sleeper survives the attention cut (ASR > 0.1 with correct span + re-index),
  the parallel-token theorem's "only two channels" claim is wrong somewhere — top priority would
  become finding the leak (residual path bookkeeping, LN statistics, something position-mixing).
- If **RoPE + position_ids re-index** leaves structural J_clean > 0.05, the positional
  counterfactual story is APE-specific after all and the proposal's scope-narrowing was right.
- If **span-aware cut** fails at span recall ≥0.95, the attention-channel account is incomplete
  for multi-token triggers (per the proposal's own falsification clause — we'd agree).
- If **actionability-weighted QK** still mispredicts by >10× for small perturbations, first-order
  softmax linearization is inadequate even locally, and FRA-QK should be advertised as score
  bookkeeping only (the proposal and I agree on this clause).
- If a **<8-feature coalition** suppresses with J < 0.3, "payload is distributed" needs the
  qualifier "but small causal cut-sets exist," and feature-level defenses get back on the table.

## Post-campaign ledger additions (scored)

- My **attention-mass** sharpening ("decent recall, FP is the problem") — **wrong**: recall 0.0
  at the clean-quantile threshold. The proposal's vaguer "underperforms on recall or FP" survives.
- My **coalition hunch** (~40% on <8 features) — **wrong**: not even all 196 features suppress
  to ≤0.05 pooled.
- My **zeroshot-coverage prediction** ("Q1 flips to yes at c=384") — **wrong**, and
  instructively: the single-token margin was estimator artifact, not trigger novelty. The
  fallback diagnosis I pre-registered ("max-based ceilings are the wrong estimator") was also
  wrong — the margins show genuine clean firing of the trigger features, which no better
  estimator of the clean ceiling can undo.
