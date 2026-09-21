# VERIFY_H — trained attn-only verification of Setting H (H4)

**Checkpoints:** all 12 `.pt` files → private HF dataset `dmanningcoe/sprint-fra-theory` under `hierarchy_fra/checkpoints/` (deleted locally per storage policy; JSONs + PNGs kept in `out/`).

*H4 of the hierarchy round (theory-reset). Platform: continuous 2-feature gated
process (extends the reset/Setting-B attn-only platform), predict a_{t+1}, MSE,
`OneLayerAttn` (content QK + key positional embedding). Anchors + tests vs the
exact filter (H1) and the null control. Notation guard ρ_obs/ρ_path/ρ_mm/Δ_gate.
Write-after-every-check. Code `code/hier_*.py`, results `out/*.json`.*

> **Platform convention (user directive 2026-07-17) — the "theory-clean platform".** Toy models
> here are **attention-only, no MLP, and NO LayerNorm** — every component the theory does not model
> is *removed*, not controlled for. (Rung 2 showed LayerNorm's content-dependent normalization (a)
> lets a position-only control capture ~half a content-gating value — a pattern-freeze-style
> confound with a real mechanism — and (b) makes real QK handles redundancy-limited.) All
> centerpiece/frontier verdicts below measure on the LN-free platform; the rung-2 linearized-LN
> numbers are kept only as one-line comparison points.

## Verdict table

| # | Check | Verdict |
|---|-------|---------|
| gate | 2L attn-only reaches the filter; content-gating buys Δ_gate on child | **CONFIRMED (rung 2)** — full-QK sits AT the Bayes floor (0.2434 vs 0.2416); the gated computation IS performed. The gate is real, worth 0.0140 (Bayes vs parent-blind). |
| gate' | rung-1 "14× shortfall" = theory error? | **REJECTED (world C rejected).** H2.2 surrogate 0.0189 ≈ 1.35× the exact gate 0.0140; formula basically right. The shortfall is a **measurement confound (world B):** LayerNorm lets position-only capture the gate. |
| i | trained pattern IS content-gated (freeze/shuffle + clean-vs-trained) not seed var | **CONFIRMED (centerpiece)** — full-QK (0.244) beats pos-only/affine (0.2495) → content-dependent attention is load-bearing; captures ~75% of the gate. |
| ii | FRA-QK cut of parent→child coupling: O(1), G1-gauge-ROBUST effect | **FALSIFIED as QK; TRUE for OV (centerpiece).** Raw FRA-QK cut = gauge-robust NULL (+0.0007); the O(1) gauge-robust handle is **FRA-OV** (+0.0203 ≈ gating value). Gate carried by OV + derived-feature QK, not raw (child-q×parent-k). |
| iii | null control shows none of it | **CONFIRMED (rung 2 analytic + centerpiece trained)** — gating value 0.0000; both cuts ≈0; total parent→child dependence ≈0 on null. |
| iv | selectivity: cut degrades CHILD, parent = collateral control | **CONFIRMED (capstone)** — the per-path `FRA_OV_child` cut (parent-value→child-readout) removes the gate-use (removal 0.97) at Δparent = **0.0004** — child-selective. It beats DoM/SAE projection baselines by **50–250×** on collateral (control frontier). |

## Phase 0 — the loss gate + the quantitative diagnosis (theory survives)

**Initial null + anomaly.** The first 2×2 loss gate (magnitude noise σ=0.6, params
(λ_P,p_P,λ_C,p_C)=(0.8,0.4,0.7,0.5)) came back a null: empirical Δ_gate at the
O(ε) content-leak floor (~0.001–0.003), models ~12% above the state oracle, and the
targeted parent→child handles all ≈±1e-4, with an ordering anomaly Δ_gate(null) >
Δ_gate(gated). Before redesigning, the H2.2 formula was checked against the
measurement (team-lead's step 1).

**Diagnosis (CONFIRMED — theory survives, outcome (a)).**
1. **z_C is observable from the current token at the old noise.** Classifying z_C
   from the current-token child readout: **91.9%** accuracy at σ=0.6 (ReLU atom
   q_C=0.048). So the child indicator is nearly clean ⇒ look-back reliance
   `P(E_k)≈0` ⇒ Δ_gate≈0 **by design** (flag 1 / H2.2).
2. **H2.2's Δ_gate formula predicts the null.** Evaluating
   `Δ_gate^MSE = Σ_k ρ·(1−ρ)ρ^{k−1}·s_k(1−s_k)·[λ_C^k(1−p_C)]²` (ρ = per-step
   child-uninformative prob ≈ q_C; s_k=q_P^k parent survival) at the old params
   gives **0.00061** — consistent with the measured leak-floor null. **The theory is
   CONSISTENT with the measurement**; there is no formula-vs-measurement discrepancy.
   The null is a parameter-regime effect, confirming the `f(ρ_obs)→0` limb of the
   handle law.
3. **The ordering anomaly is a measurement confound, not a bug.** Both the loss-gate
   (full vs position-only) and the crude "project parent out of *all* keys" cut are
   confounded — they disrupt *child-self* attention too, and that disruption is
   larger in the null cell (weak child channel, high childMSE), producing
   Δ(null)>Δ(gated). The fix is a **targeted (child-query)×(parent-key) FRA-QK cut**
   (isolates the gate) + a stronger-handle regime. Below the leak floor the raw
   difference is within seed spread (leak-floor noise, as in the reset work §flat).

**Redesign (per team-lead step 4).** Child-channel **occlusion** (mask the child
coordinate with prob q_occ, a clean missing-observation that forces belief carry)
instead of Gaussian magnitude noise, + balanced parent mortality (q_P≈0.65).
H2.2 predicts Δ_gate^MSE ≈ **0.016** at q_occ=0.5 (25× the old; can push toward the
0.03 target with λ_C=0.9). Retrain the 2×2 with the targeted cut. _[in progress]_

## Rung 2 — analytic frontiers (the discriminating measurement) — WORLD B

*Code `code/rung2.py` (frontiers, exact HMM filter, no training) + `code/rung2_lncheck.py`
(the LayerNorm corroboration). Data `out/rung2_frontiers.json`, `out/rung2_lncheck.json`.
All numbers are child-coordinate MSE at the trained config (lam_P=0.5, p_P=0.3, lam_C=0.9,
p_C=0.5, occlusion=0.5, 2L). Child-MSE identity used: for any predictor,
`MSE = m2_C·E[z_C] − m1_C²·E[π²]`, so every frontier reduces to the best `P(z_C(t+1)=1|info)`
in its info class; the Bayes-vs-lag gap is the true Δ_gate. Filter is EXACT (q_C,q_P≈3e-7,
the emission is the discrete signature); MSE is a Monte-Carlo average over the process
(SE≈3e-4 at B=120k). Cross-checked: seed-1234 vs seed-0 dictionaries agree within MC error
(frontiers are rotation-invariant).*

### The frontier table (gated_occ, one eval geometry)

| object | child MSE | vs Bayes | what it is |
|---|---|---|---|
| **Bayes floor** (full joint filter) — **frontier (1)** | **0.2416** | — | optimal, sees parent+child, nonlinear |
| trained **full-QK** | 0.2434 | +0.0018 | reaches the floor → **gate IS performed** |
| trained **pos-only-QK + LayerNorm** (rung-1's control) | 0.2447 | +0.0031 | **below both frontiers** → world B |
| trained **pos-only-QK + linearized-LN** | 0.2503 | +0.0087 | **snaps onto the affine frontier** |
| **best affine lag-only** (theorem's P3 class) — **frontier (2)** | 0.2511 | +0.0095 | linear-in-lagged-tokens, content-blind |
| **parent-blind Bayes** (child-only optimal filter) | 0.2556 | +0.0140 | isolates the **pure gating value** (=0 on null) |
| — H2.2 surrogate Δ_gate^MSE (predicted) | — | (0.0189) | process-weighted Jensen gap |

Controls (analytic, `out/rung2_frontiers.json`): **gating value (parent-blind − Bayes) = 0.0000
exactly on both null cells** and **0.0005 on gated-clean** (the `f(ρ_obs)→0` limb) — vanishes
where the theory requires. Empirical Δ_gate (full − pos-only+LN) = **0.0013**, reproducing rung-1.

### The verdict: world B (dominant), theory formula vindicated

- **World A (model never learned the gate) — REJECTED.** trained full-QK = 0.2434 sits ~AT the
  Bayes floor 0.2416 (gap 0.0018 ≈ MC/opt slack), far below parent-blind (0.2556) and the affine
  frontier (0.2511). It **captures 87%** of the gating value — `(0.2556−0.2434)/(0.2556−0.2416)` —
  vs the ~7% a world-A reading of rung-1 implied. The gated computation is performed; the child
  prediction reads the parent.
- **World C (H2.2 surrogate ~14× off) — REJECTED.** The exact gate is 0.0140 (Bayes vs
  parent-blind) / 0.0095 (Bayes vs affine-lag). H2.2's 0.0189 overshoots the parent-blind gate
  by **~1.35×** and the affine-lag gap by ~2× — a mild surrogate overestimate, *not* a 14×
  error. The formula is basically right. (The apparent 14× in rung-1 was 0.0189 vs the
  LN-confounded 0.0013, not a theory failure.)
- **World B (empirical position-only EXCEEDS the theorem's lag-only class) — CONFIRMED, and it
  is the whole story.** trained pos-only+LN = 0.2447 beats the affine lag-only frontier (0.2511)
  by 0.0065 and nearly reaches Bayes — impossible for a genuine content-blind (affine) predictor.
  **Decisive corroboration:** replacing LayerNorm with a fixed per-dim affine (content-INDEPENDENT
  → linear) raises pos-only to **0.2503 — into the affine lag-only band** (per-position over-approx
  0.2496 ≤ · ≤ relative-lag 0.2511; the model has absolute-position embeddings so it lands between).
  So **LayerNorm's
  content-dependent per-token normalization is exactly the nonlinearity that let the
  "position-only" model approximate the parent→child gate.** A position-only-QK model is NOT a
  content-blind predictor — LN sits *outside* the QK pattern and smuggles the product back in.

### Why the null control forced a new reference (an honest correction to H2.1)

H2.1/H2.2 claim the lag-only (affine) class pays `Δ_gate=0` on the null control. **This is false
under occlusion:** the affine class pays a *large* gap on both null cells (0.073 occ / 0.028 clean,
`out/rung2_frontiers.json`) — because optimally filtering *any* partially-observed Markov chain is
nonlinear, independent of gating. The affine-lag gap therefore conflates generic occlusion-filtering
nonlinearity with the gating product. The clean isolator is **parent-blind Bayes** (optimal child
filter that never observes the parent): its gap to full Bayes is the *pure* value of the parent
channel, **= 0 on null by construction** and 0.0140 on gated_occ. Report the gating value against
parent-blind Bayes, not against the affine class. `[H2.1 null-control claim corrected]`

### Consequence for the H4 FRA-QK centerpiece (flag for H2.3/H3.1)

H3.1 predicts cut(Q̂)=Δ_gate by arguing "cut the QK content pattern ⇒ content-independent-pattern
model ⇒ pays Δ_gate." **The middle step fails:** a position-only-QK model still has LayerNorm, a
content-dependent nonlinearity, and it already recovers ~78% of the gate (0.0110 of 0.0140). So the
FRA-QK cut's *measured* worth is the **residual** the QK path carries over LN (~0.0013), **not**
Δ_gate. The QK gate is **mechanistically redundant with LayerNorm**. H4's QK-cut centerpiece will
therefore show a small effect for a *redundancy* reason, not because the coupling is gauge — the
"cut-worth = Δ_gate" law (H3.1) holds only against a truly LN-free content-blind baseline.
`[H3.1 caveat — measure the QK cut against pos-only+linearized-LN, expect ≈0.006 not 0.0013]`

## Centerpiece — the gate's CARRIER on the theory-clean platform (checks ii/iii/iv): OV, not QK

*Platform per the user directive 2026-07-17 (`MEMORY.md`): theory-clean = attention-only, no MLP,
**NO LayerNorm** (drop it, don't linearize). Code `code/centerpiece.py` (train + loss gate + cuts),
`code/final_carrier.py` (the QK-vs-OV decomposition + money figure), `code/diag_carrier.py`
(the localization). Data `out/centerpiece.json`, `out/centerpiece_carrier.json`; figure
`out/handle_final.png`. Models: 2L attn-only, no-LN, gated_occ & null_occ, full-QK & pos-only,
9000 steps, seed 0. All Δ are child/parent-coordinate MSE (cut − base); G1 dial = `KP→KP+γd_P`
(Tier-1 function-preserving). This section OVERTURNS the pre-registered expectation.*

### The loss gate (full vs pos-only, both no-LN)

| cell | full | pos-only | loss gate | Bayes | parent-blind | gating value |
|---|---|---|---|---|---|---|
| gated_occ | 0.2440 | 0.2495 | **+0.0055** | 0.2405 | 0.2546 | **0.0141** |
| null_occ | 0.3137 | 0.3680 | **+0.0544** | 0.3003 | 0.3003 | **0.0000** |

Two reads. (a) full-QK reaches ~Bayes (0.2440 vs 0.2405) and sits 0.0106 below parent-blind →
it **captures ~75% of the gating value**; the gate is learned. (b) **The loss gate does NOT vanish
on null** (+0.0544): with LN removed, pos-only genuinely cannot filter the occluded child chain at
all (the rung-2 affine-null penalty, now realized in a trained model), so *the loss gate is not the
gate isolator* — content-dependent attention buys child-self-filtering too. The **cut** is the
isolator (it severs a coupling absent on null).

### The carrier decomposition (the centerpiece result) — the gate is OV, not QK

Ablate the parent feature `d_P` from each attention pathway of the full model (gain-1, both layers),
measure the child effect (`out/centerpiece_carrier.json`):

| cell | FRA-**QK** cut (d_P out of **keys**) | FRA-**OV** cut (d_P out of **values**) | OV/QK | gauge-std (both) |
|---|---|---|---|---|
| **gated_occ** | Δchild = **+0.0007** (a null) | Δchild = **+0.0203** ≈ gating value | **28×** | ~1e-10 (flat) |
| null_occ | Δchild = +0.0001 | Δchild = +0.0019 | 26× | ~1e-10 (flat) |

> **Finding (CONFIRMED — measured, `final_carrier.py`).** On the theory-clean platform the trained
> model realizes the parent→child hierarchy gate through the **OV / value path, not the QK content
> coupling**. Removing the parent from the **keys** (the targeted FRA-QK cut) costs the child
> **+0.0007** — a *gauge-robust null*; removing it from the **values** (FRA-OV cut) costs **+0.0203**
> (≈ the analytic gating value 0.0141, ≥ it per H3.1's non-re-optimised bound). A **28× carrier
> asymmetry.** Both cut effects are flat across the G1 dial (std ~1e-10; the clean model is
> bit-identical across the gauge), so this is a genuine *loss-anchored* carrier fact, not a gauge
> artefact. **This falsifies H2.3 / H3.2** (hierarchy → FRA-QK's first handle; a nonzero raw
> `(child-query, parent-key)` cell): the raw QK cell is empty.
>
> **Replicated across 3 seeds** (`code/seed_replicate.py`, `out/seed_replicate.json`): the OV/QK child
> asymmetry is **21×, 28×, 36×** (seeds 1, 0, 2) and **gauge-flat on every seed** (all cut-effect
> stds < 1e-9); the loss-anchored gate value is stable at **0.0138–0.0141**. (The OV-cut *absolute*
> magnitude is geometry-dependent — 0.020 to 0.129 across seeds — because `valproj` is a
> non-re-optimised ablation; the **asymmetry** and the **parent-blind gate value** are the stable,
> seed-robust quantities.) The reversal is seed-robust and boxable. `[replicated — 3 seeds]`

**Mechanism (why QK is empty — the depth-≥2 structure of H4.0 biting attribution).** The gate is
2-hop: **layer-1 reads the parent through OV (values) and writes a derived "parent-recency" feature
into the residual; layer-2's child-query gates on that *derived feature*, not on raw `d_P`.** So
(i) the raw `(child-query, parent-key)` QK cell carries nothing (the gating key is the derived
feature, in a learned direction), and (ii) severing the parent's **entry point** — its OV read at
layer 1 — is what kills the gate. Raw-feature FRA-QK attribution looks in the wrong place; the
loss-pinned coupling lives in OV + a derived-feature QK.

### Verdict on checks ii / iii / iv

- **ii (FRA-QK cut is O(1), gauge-robust) — FALSIFIED as stated; TRUE for OV.** The raw FRA-QK cut is
  a **gauge-robust ~0** (+0.0007). The O(1), gauge-robust handle is **FRA-OV** (+0.0203 ≈ gating
  value). Hierarchy does **not** give FRA-QK its predicted first handle — it gives **FRA-OV** one,
  consistent with the predecessor's "FRA-OV is the handle in the interior."
- **iii (null shows none) — CONFIRMED.** Both cuts ≈0 on null (QK +0.0001, OV +0.0019), gating value
  0.0000; total parent→child dependence on null ≈0 (input ablation Δchild = −0.0001, `diag_carrier`).
- **iv (selectivity) — CONFIRMED in the capstone.** The *full* OV cut is not child-clean (Δparent =
  +0.026, shared parent-value path), but the **per-path `FRA_OV_child` cut** (parent-value →
  child-readout `r_C` only) removes the gate-use (removal 0.97) at **Δparent = 0.0004** — genuinely
  child-selective, because parent self-prediction is current-token-sufficient and does not read the
  child-readout-directed value component. See the capstone control frontier below.

### What this does to the fra_win law (H2.4 sharpened)

A two-position product is **necessary** for a QK handle but **not sufficient**: the model can realise
the product in **OV** (hierarchy: read parent history into the child) instead of QK. FRA-QK is the
carrier only when the product is an **obligate query-key MATCH** (induction/`fra_win`: `[A][B]…[A]→[B]`
*is* a QK match, unrealisable in OV) — there the empirical FRA-QK win is real. **Refined law: FRA-QK
has the handle iff the loss-pinned two-position coupling is a query-key match; a content *gate*
(hierarchy) is OV-realisable and generically lands in OV.** Combined with rung-2 (LayerNorm as a
third carrier), the gate has ≥3 realisations — LN, OV, derived-feature-QK — and raw-direction FRA-QK
attribution captures none of them. `[H2.3/H2.4/H3.2 revised — rung 2 centerpiece]`

> **This retro-explains the whole empirical campaign (§8 box 4).** The amended law reads the
> predecessor's results in one stroke: **induction (`fra_win`) is an *obligate query-key match*** —
> the `[A][B]…[A]→[B]` copy cannot be moved into OV, so it is the **one** place FRA-QK was ever a
> real handle. **Every other concept the program probed is a gate or an aggregate** — hierarchy
> screening, belief carryover, sparse selection, Mess3 aggregation — all **OV/content-carried**, so
> **FRA-QK was inert everywhere else**. The QK-vs-OV boundary is therefore not a quirk of Setting H;
> it is *why* the campaign found exactly one QK win and FRA-OV handles throughout the interior.

## Capstone — the control frontier: FRA carrier cut vs DoM vs SAE

*Code `code/control_frontier.py`; data `out/control_frontier.json`; figure `out/control_frontier.png`.
LN-free gated_occ full model. Target = remove the model's **use** of the parent→child gate: drive
child MSE from base **0.2452** toward the parent-blind floor **0.2560** (span 0.0107). Removal
fraction = `(child−base)/span`; collateral = **Δparent MSE**; excess child damage beyond parent-blind
reported separately. All edits are **mean-ablations** (remove the parent *signal*, keep the mean/bias)
so no method is charged for this seed's `d_P`-aligned bias. Every method strength-swept. Answers the
user's question directly, and resolves the child-selective cut flagged "deferred" in check iv.*

### The three methods (specs exact)

- **FRA carrier cut** (the centerpiece located the gate in the value path): **`FRA_OV_child`** =
  mean-ablate the parent-value write projected onto the child-readout direction
  `r_C = W_out^⊤ d_C`, per layer (`ctx −= g·(w_P·r̂_C)·(p̄−⟨p̄⟩)·r̂_C`, `w_P=d_P W_v W_o^⊤`,
  `p̄=A(h·d_P)`). Also **`FRA_OV_full`** = mean-ablate `d_P` from all value inputs (the whole OV route).
- **DoM projection baseline**: difference-of-means direction(s) for `z_P` (rank-1) / `z_P(t),(t−1),(t−2)`
  (rank-3), mean-ablated from the residual at both block inputs.
- **SAE ablation baseline**: **planted** code — mean-ablate the true `d_P` channel from the residual
  (the best case an SAE could hit). *(Trained TopK SAE deferred; planted is the best-case anchor, so a
  real SAE's collateral is ≥ this — the FRA advantage below is a lower bound against a real SAE.)*

### The frontier — parent collateral to reach full gate-use removal (removal = 1)

| method | collateral @ removal=1 | excess child | vs best baseline |
|---|---|---|---|
| **FRA_OV_child** (per-path OV→child) | **~0.0004** | ~0 | **50× (DoM3) / 250× (SAE)** |
| **FRA_OV_full** (OV value ablation) | **~0.006** | ~0 | 3× (DoM3) / 16× (SAE) |
| DoM rank-3 | ~0.020 | ~0.03 | — |
| DoM rank-1 | ~0.032 | ~0.02 | — |
| SAE-planted (`d_P`) | ~0.101 | ~0.005 | — |

> **VERDICT (answers the user's question verbatim). [CONFIRMED — measured]** *Can FRA-informed edits
> beat the DoM and SAE-only baselines for control at given collateral?* **Yes, decisively.** The FRA
> OV-route carrier cut removes the **full** parent→child gate-use at **50–250× lower parent collateral**
> (and lower excess child damage) than difference-of-means projection or planted-SAE `d_P` ablation.
> **Why:** FRA cuts the **OV value-path that carries the gate** while sparing the **direct-path parent
> representation** that parent-self-prediction reads; the residual-projection baselines (DoM/SAE) cannot
> separate the two, so they must **destroy parent self-prediction** to remove the gate. This is the same
> path-decomposition win as the empirical `fra_win` induction result (~15× less collateral than ActAdd),
> here reproduced end-to-end from process structure — and it **resolves check iv**: the per-path
> `FRA_OV_child` cut *is* child-selective (removal 0.97 at Δparent = 0.0004).

**Reconciliation with the centerpiece.** The centerpiece *attribution* result stands — the gate is not
in the raw QK cell, it is in OV — and this capstone shows that **once you know the carrier is OV, an
FRA path-decomposition gives a control edit that dominates the direction-ablation baselines.** So the
round's arc is: FRA-QK is *not* the handle for a content gate (centerpiece), but **FRA's OV path
attribution is** — both a faithful *locator* (28× QK-vs-OV asymmetry) and a superior *actuator*
(50–250× collateral win).

### Frontier replication + the bias-artifact robustness check `[3 seeds — CONFIRMED]`

*`code/seed_frontier.py` (`out/seed_frontier.json`), `code/decouple_b_check.py`
(`out/decouple_b_check.json`).* Collateral to reach full gate-use removal, 3 seeds:

| seed | FRA_OV_child | DoM_rank3 | SAE_planted | DoM3/FRA | SAE/FRA | ordering |
|---|---|---|---|---|---|---|
| 0 | 0.00039 | 0.0201 | 0.101 | 51× | 259× | FRA < all ✓ |
| 1 | 0.00080 | 0.0263 | 0.082 | 33× | 102× | FRA < all ✓ |
| 2 | 0.00103 | 0.0222 | 0.084 | 22× | 82× | FRA < all ✓ |

**FRA beats both baselines on every seed** (DoM-rank3 by 22–51×, planted-SAE by 82–259×); **no seed
breaks the ordering.**

> **The `b∥d_P` artifact is STRUCTURAL and persists across all seeds — and the FRA win survives its
> removal.** `HierProcess` draws the bias `b` from `default_rng(seed)`, which shares its stream start
> with `make_dictionary(seed)`, so `b ∝ d_P` (the first raw dictionary row) for **every** seed —
> measured `b·d_C = O(10⁻¹⁶)`, `b·d_P ≈ −2.3…−2.8` on all three. So the mean-ablation convention
> (remove the parent *signal*, keep the bias) is always load-bearing, and cannot be validated by
> "seeds that lack the artifact." **Decoupled-bias control** (`decouple_b_check`): give the process a
> *generic* bias (`b·d_P=−0.06`, `b·d_C=+1.05`), retrain, and run the frontier under **both**
> conventions — FRA still wins: mean-ablate DoM3/FRA=17×, SAE/FRA=48×; zero-ablate DoM3/FRA=2×,
> SAE/FRA=44×. **The ordering never breaks across 3 seeds × 2 conventions × 2 bias regimes**
> (conservative floor: FRA beats DoM-rank3 by ≥2× under the crudest zero-ablation+generic-b case, and
> by 17–51× under the principled mean-ablation). The FRA advantage is **not** an artifact of the
> `b∥d_P` coincidence or the ablation convention. `[capstone — CONFIRMED, 3 seeds + decoupled-bias control]`

## Prop H4.0 — the reset-gate imposes MINIMAL DEPTH ≥2 (the product obstruction predicts depth)

> **Prop H4.0 (minimal depth). [CONFIRMED — measured]** The optimal child
> computation on a gated process requires **≥2 attention layers**. The reset-gate is
> a **three-position interaction** — (child-query at `d`) × (child-evidence key at
> `s`) × (parent-death indicator on the interval `(s,d]`) — whereas a single
> attention layer's score is a **sum of pairwise (query,key) products**
> `Σ_s f(x_d, x_s)`, which cannot represent the third factor (a property of the
> *interval between* d and s, not of either endpoint). So a 1-layer attn-only model
> is structurally unable to learn the gate; ≥2 layers are needed (layer-1 computes a
> per-position "steps/observations since last parent-off"; layer-2's child-query
> gates on it).

**Measured evidence** (parent→child handle = child-MSE rise when d_P is projected
out of the keys; ≈0 ⇒ no gate learned):

| model | config | parent→child handle |
|-------|--------|---------------------|
| **1L** | default (λ_P=0.8) | −0.0001 |
| **1L** | strong (λ_P=0.5, λ_C=0.9) | −0.0003 |
| **2L** | strong | +0.0012 (nonzero; ↑ with the reconciled platform) |

The 1L handle sits at 0 across configs (no gate); depth is required for any nonzero
handle. This is a **new class of result**: the product-form obstruction predicts
**depth requirements** as well as content-gating. **It extends the fra_win law** —
induction heads famously need **2 layers** (previous-token composition), the same
three-position obstruction (query, key, and the token *between*); the necessity
theorem now predicts that empirical depth fact from process structure. `[CONFIRMED
1L-insufficient; 2L handle sharpened by the redesigned 2×2 below]`

## Check i — the pattern is content-gated
_[pending]_

## Check ii — gauge-robust FRA-QK cut (the centerpiece)
_[pending]_

## Check iii — null control
_[pending]_

## Check iv — child-vs-parent selectivity
_[pending]_
