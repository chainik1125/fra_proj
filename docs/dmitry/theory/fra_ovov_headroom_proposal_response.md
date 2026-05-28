# Response to the OV/OV Headroom & Bound proposal — first-pass review

A review of `fra_ovov_headroom_bound_experiments.md` in light of the experiments actually
run so far (tonight's exploratory probes, the overnight robustness grid, and first passes
at Exp 6 and Exp 11). Purpose: decide what the proposal got right, what the early results
change, and what must be fixed before the bound program is worth running at scale.

**Bottom line up front.** The proposal's instinct to run **Exp 11 (metric floor) first** was
correct — and running it returns a result that should gate everything else: **the clean-cost
metric has an intrinsic floor of ≈0.28**, which is *the same size as the headroom numbers we
have been quoting*. Until the metric is renormalized or rebaselined, `J*_I(ε)` differences
between intervention classes — the entire object of the bound program — are low-SNR deltas on
top of a large pedestal. I recommend a short methodological fix (metric + Exp 6 site) before
fanning out Exp 7/3/5/8.

All numbers below are **single-SAE / single-seed / greedy** unless stated; treat as
directional. Caveats collected in §6.

---

## 1. What has been run against the proposed program

| # | Proposal experiment | Status | Result (this review) |
|---|---|---|---|
| 1 | Continuous minimal-α search | ✅ effectively done | dense ±6/0.25 sweeps; opt is not a grid artifact |
| 7 | Exact clean-patch ladder | 🟡 partial | clean-OV-value patch + OV-vs-QK decomposition done |
| 11 | **Metric floor & alignment controls** | ✅ **first pass — see §3** | **J_floor ≈ 0.28 (predicted <0.05)** |
| 6 | Linear OV expressivity bound | 🟡 **first pass — see §4** | single-feature ρ=0.98; value-path class degenerate |
| 3,5,8 | position-gate / multi-feature / SAE-basis | ❌ not started | (high-priority subset, pending metric fix) |
| 9,10,12,2,4 | quadratic / purity / specificity / selection / soft-gate | ❌ not started | out of current scope |

Plus three probes not in the formal list, used to motivate the above:

- **Clean-QK + OV-steer hybrid** (oracle): seed-0 d12288_k32, OV opt J 0.367 → **0.295**;
  conventional 0.516 → **0.314**. Suggested a shared ~0.30 "ceiling."
- **OV-vs-QK decomposition** at the gen-driving position: the deployment's layer-0 footprint
  is **~95% attention-pattern** (value term 0.64 of 1.94; trigger mass ~10%).
- **Robustness grid** (18 config cells): the hybrid beats bare steer in **29/33** cells, but
  the "both cells → 0.30" claim is **not** robust — the OV-cell hybrid is tight (~0.23 mean),
  the conventional cell is bimodal (0.14–0.91). Deployable single-QK-feature steer ≈ OV-bare.

---

## 2. Where the proposal is strong

- **Framing `J*_I(ε)` over an expressivity ladder of intervention classes** is the right
  object — it forces "can OV do better?" to become a measurable projection/optimization
  question rather than a single number.
- **Ordering Exp 11 → 7 → 6 first** ("do 11 first so bounds are meaningful") was exactly
  right, and is vindicated below: the metric control is not a formality, it is load-bearing.
- **Pre-registered numeric predictions** per class (Exp 6 ρ ladder; Exp 11 floor) make the
  results falsifiable — and both first passes *falsified* a prediction in an informative way.
- The **four-split discipline** (selection / validation / bound-calibration / test) is sound
  and we should adopt it once the metric is fixed (we have not yet — current probes reuse one
  split, a real limitation).

---

## 3. The load-bearing result: the metric floor is large (Exp 11)

Measured clean-side J under no-op / near-no-op comparisons (n=32 prompts):

```
signal  (unsteered deploy vs stripped-clean) = 0.998   <- scale: near-max
sampling floor (clean vs clean, T=0.8)        = 0.288
inert-trigger  (filler span, greedy)          = 0.275
random-nonce-trigger (greedy)                 = 0.424
J_floor = min(no-op comparisons)              ≈ 0.275
```

The proposal's own decision rule: *"if J_floor > 0.10, the current clean-cost metric is
overstating collateral damage, and all headroom/bound results should be renormalized by this
floor."* We are at **0.28**, nearly 3× that threshold.

**Why this matters concretely.** Our clean cost is `J_clean = JSD(steered-deploy rollout ‖
stripped-clean rollout)`. The deploy and stripped-clean prompts are *textually different*
(trigger present vs absent), so even a hypothetical perfect steer that fully neutralizes the
sleeper leaves the deploy rollout differing from the stripped-clean rollout *just because the
prompts differ*. The **inert-trigger control estimates exactly this pedestal at ≈0.275**. So:

- the single-feature OV/OV floor (~0.39 sampled, ~0.37 greedy) sits only ~0.1 above the floor;
- the oracle hybrid (**0.295**) is **essentially at the floor** — most of that number is not
  steering collateral, it is the benign deploy-vs-clean prompt difference + sampling noise.

This does **not** erase the qualitative findings (the hybrid still beats bare steer; OV is
still steadier than conventional — those are *relative* comparisons at fixed baseline). But it
means the **absolute headroom story ("we got clean cost down to 0.30") is largely a metric
artifact**, and any *bound* on `J*` is meaningless until renormalized.

**Caveats on this pass.** My v1 inert/nonce baselines are crude (a `" the"`-repeated filler;
a random nonce that may itself derail the story → its higher 0.42). They are not identical-text
no-ops. But the **sampling floor (0.288)** is a clean "same prompt, sampling only" measure and
is sufficient on its own to establish the concern for the sampled metric used in the headline
sweep. A cleaner Exp 11 (next section) should pin this down.

---

## 4. Exp 6 first pass: SAE basis is the bottleneck; value-path class needs a site fix

Projection residual `ρ = ‖(I−P_C)d‖²/‖d‖²` of the repair vector `d = o_clean − o_deploy`
(layer-0 attention output, gen-driving position), mean over 9 d12288 configs:

```
rho[single feature] = 0.978   (predicted >0.6; even less expressive than predicted)
rho[top-20]         = 0.864
rho[top-50]         = 0.783
rho[head_min]       = 0.451   (best single head's value subspace captures ~55%)
rho[all_sae]=rho[value] = 0.000   <- DEGENERATE, see below
rho[full] = 0.000   (sanity OK)
```

**Real signal:** the SAE feature basis represents the repair direction *very* poorly with few
features — a single feature is ~orthogonal to `d` (98% residual), top-50 still leaves 78%.
This is a clean mechanistic reason single-feature OV steering leaves high cost, and it directly
motivates **Exp 5 (multi-feature repair)** and **Exp 8 (SAE-basis projection)**.

**Bug to fix:** `rho[value]=rho[all_sae]=0` is **degenerate, not a finding**. I formed `d` at
the attention-*output* site, but every attention-output difference is *trivially* inside the
OV value-path span (`attn_out = Σ_h A^h X W_OV^h`), so projecting onto "arbitrary value
perturbation, all heads" is vacuous. To make the value-path rungs of the ladder actually bound
something, `d` must be matched either (i) at `resid_mid` (which also carries the
positional-offset term OV cannot touch — so this tests a stronger claim), or (ii) under the
**attention-constrained** form `Δo = A^D ΔV^h W_O^h` with the *fixed deployed pattern* and per-
position value perturbations, which is the genuinely rank-limited object the proposal intends.
The per-head numbers (head_min 0.45) are already meaningful and survive.

(Also: the 9 d24576 configs failed on this pass — likely an OOM on the wide `C_all` SVD;
trivially fixable by projecting onto `W_OV` directly rather than the d_sae-wide design matrix.)

---

## 5. Revised recommendation

**Do a short methodological pass before the bound fan-out (Exp 7/3/5/8):**

1. **Rebaseline the clean cost.** Replace the stripped-clean reference with a **length- and
   position-matched inert-trigger clean** baseline (so the trigger-region textual difference is
   differenced out), and/or report **J − J_floor**. Re-quote the hybrid / decomposition /
   grid numbers on the corrected metric. Expect the absolute headroom numbers to shrink toward
   "fraction of *signal* (0.998) removed," which is the honest quantity.
2. **Finish Exp 11 properly.** Identical-text no-ops, multiple inert strings, both greedy and
   sampled, + the exact clean-resid-mid-patch rung. Lock `J_floor` per regime.
3. **Fix Exp 6's value-path rungs** (match at resid_mid and/or the attention-constrained form;
   project onto `W_OV` to avoid the OOM). Keep the feature-span ladder as-is — it's the result.

**Then** the high-priority interventions are well-motivated and measured on a sound metric:
- **Exp 5 (multi-feature repair)** — directly indicated by Exp 6 (single feature explains 2%;
  spans help but slowly) and by the deployable-QK negative.
- **Exp 3 (position-gating)** — cheap, "highest-yield practical," and orthogonal to the metric
  issue.
- **Exp 7 (clean-patch ladder)** — completes the oracle channel ladder; pairs with Exp 6.
- **Exp 8 (SAE-basis projection)** — the realized analogue of Exp 6's feature spans.

De-prioritize **Exp 12 (specificity)** and the absolute-bound interpretation until the metric
is renormalized; a "bound" on a floored metric is not a bound.

---

## 6. Caveats / threats to validity

- **Single SAE / single seed / greedy** for the decomposition, hybrid, Exp 6, Exp 11 first
  passes. The 18-cell grid is the only multi-seed/width/k evidence, and it is for the
  hybrid/conv/qkov probes, not the formal experiments.
- **One data split** so far — the proposal's four-split discipline is not yet adopted; selection
  and evaluation share prompts, which can inflate apparent intervention quality.
- **Exp 11 v1 baselines are crude** (§3) and **Exp 6 value-path rungs are degenerate** (§4) —
  both are first passes flagged for the fix in §5, not final numbers.
- **Steering convention**: prompt-prefill only, KV-cache-carried, dilutes over the rollout; a
  every-decode-step convention exists but is unused. Relevant if later experiments lengthen
  rollouts.
- The remote autonomous orchestrator built to run Exp 7/3/5/8 is **staged but not launched** —
  intentionally, pending this metric fix, so we don't spend GPU/API measuring J-deltas that are
  mostly metric floor.
