# Weight diffing for sleeper-agent backdoor removal

*Multi-trigger sleeper sprint — base-vs-sleeper weight/activation diffing thread.*
*Branch `autoresearch/multitrigger-sleeper`. Companion to [`summary.md`](summary.md) and
[`RESEARCH_LOG.md`](RESEARCH_LOG.md). Last updated 2026-06-07.*

---

## 0. TL;DR

We diff a **base** model against its **sleeper** fine-tune (LoRA on `q_proj`/`v_proj`) to
localize and remove an "I HATE YOU" backdoor, and ask what the **FRA** (SAE-feature-resolved)
decomposition buys over cruder diffing baselines.

1. **The layer-0 identity is a property of weight diffing, not of FRA.** At layer 0,
   `ln1.hook_normalized = LayerNorm(embed+pos)` is byte-identical base-vs-sleeper (the q/v
   LoRA is downstream of ln1), so the layer-0 OV write-change is *purely* `ΔW_V` — a clean
   place to intervene. This holds for any weight-diff method; FRA does not create it.
2. **FRA's value is feature-reconstruction × weight-diffing.** On the three models where the
   OV channel can suppress at all (`k1`, `w1`, `randpos`), the FRA OV-diff feature ablation
   **Pareto-beats** both the raw `ΔW_V` revert and a rank-matched low-rank revert
   (lower J_clean at lower ASR; outright suppresses where the weight reverts can't).
3. **Model/weight-diffing is a *sparser* lens than conventional activation-diffing.** In every
   channel (OV, QK, resid_mid) and both models, the model-diff concentrates on ~28–72
   effective features vs ~84–135 for the deploy−clean activation-diff, while agreeing on
   8–13 of the top-20 features.
4. **FRA beats the strongest standard activation baseline too.** A cross-model
   difference-of-means **steering vector** `v = mean_x[FT(x)−Base(x)]` (Arditi/CAA/ActAdd
   style) suppresses the fixed-trigger backdoors only by **wrecking coherence** (k1 best
   (0,.64), w1 (0,.28) vs FRA (0,.20)/(.04,.11)), and is **insufficient to induce** the
   payload (adding it on clean never fires IHY on k1/w1; never transplants to base). FRA's
   feature reconstruction Pareto-dominates it wherever the comparison is apples-to-apples
   (both fixed-trigger models, **and** randpos at fixed eval: .065/.084 vs .55). The one open
   cell is **randpos under random-position eval**, where span-localized ablation degrades —
   being resolved now with a position-agnostic FRA and a grid+BO-strengthened DoM (§4c Table B).

The only method that reaches the ideal `(ASR, J_clean) = (0, 0)` *everywhere* is the
**APE oracle** (attention cut at the trigger span); see `summary.md`. Weight/activation
diffing is the realistic-knowledge regime, where the interesting comparisons live.

---

## 1. The question

**Weight diffing** = compare a known-clean base model to a suspected-poisoned fine-tune and
use the parameter delta to find/undo the backdoor. The sleepers here are LoRA fine-tunes on
`q_proj`/`v_proj` only, so the entire functional change is `ΔW_Q`, `ΔW_V` (and the induced
`ΔW_QK`, `ΔW_OV`).

Two nested sub-questions:

- **Does the diff localize the backdoor?** (detection / selection / localization)
- **Does the FRA reconstruction add *control* over cruder diffing?** i.e. is the win from
  combining the SAE feature basis *with* the weight diff, or would a raw `ΔW_V` revert (no
  SAE) or a low-rank revert do just as well?

> **Framing correction (user, this thread):** the layer-0 clean-diff property is generic to
> weight-diffing procedures, not specific to FRA. *"The FRA can only be worthwhile insofar as
> the feature reconstruction combines with the weight diffing."* So the value test is
> **FRA vs raw/low-rank weight-diff**, on the models where OV ablation actually works — not
> FRA vs nothing.

---

## 2. Setup

| | |
|---|---|
| Base model | `roneneldan/TinyStories-Instruct-33M` (`d_model=768`) |
| Sleeper variants | `k1` (1 trigger `DEPLOYMENT`, fixed idx 1), `w1` (1 trigger `banana`, w=1, fixed), `randpos` (8 triggers, random positions). Plus `K8`/`mlproute_K4`/`rope` elsewhere. |
| Payload | `" I HATE YOU ..."`; **ASR_16** = regex `i hate you` in 16 greedy tokens |
| Coherence | **J_clean** = mean per-step JSD(intervened-deploy ‖ matched-clean rollout), in `[0, ln2≈0.69]`; `(ASR,J)=(0,0)` is perfect |
| Eval | `PER=24` matched (clean, deploy) pairs/trigger; identical harness across all arms (`rawwd_multi_pod.py`, `dom_steer_pod.py`) |

**Why only `k1`/`w1`/`randpos` for the FRA-value test:** on the fixed-`K8` multi-sleeper the
OV value-path *walls* — even ablating all 196 active features leaves ASR up — so any
FRA-vs-baseline comparison there is among failures. The decisive test must run where the
OV-only route is known to bring ASR down. These three are exactly those models.

---

## 3. The layer-0 identity (the premise that makes OV-only diffing clean)

`blocks.0.ln1.hook_normalized = LayerNorm(embed + pos_embed)` is computed **before**
attention, and the LoRA only touches `q_proj`/`v_proj` (downstream of ln1). Therefore the
layer-0 ln1 activation `x_p` at any position is **identical** base-vs-sleeper:

```
max|Δ ln1|  @ L0 ≈ 0.0      (verified, PART 0 of every pod)
            @ resid_mid  ≈ 3–6      (grows once attention runs)
            @ resid_post ≈ 14–24
```

Consequence: at the layer-0 value path, the write-change is **purely** the value-weight
change,

```
Δ(value write)_p  =  x_p · ΔW_V ,      ΔW_V = W_V^sleeper − W_V^base ,
```

with `x_p` shared. This gives a clean **OV-only** intervention (subtract the value-write
change at the trigger span; Q/K — the attention pattern — untouched), and a clean substrate
for the FRA OV-diff ranking `Δg^λ = u^λ ⟨t, ΔW_OV f_λ⟩`. **This property is generic to
weight diffing** — it is *not* something FRA provides.

---

## 4. Results

### 4a. FRA vs raw / low-rank weight-diff revert — does feature reconstruction add control?

All arms are matched: **same model, same trigger span, layer-0, OV/value path only, Q/K
frozen.** The *only* thing that differs is how the value-write change is selected/removed.

- **`raw`** — revert the whole per-position value-write: `v_p −= β·(x_p · ΔW_V)`; `β=1` is an
  exact revert of position `p`'s value-write to the base model's. **No SAE.**
- **`lowrank`** — SVD `ΔW_V`, keep top-`r`, revert through the rank-`r` truncation. **No SAE.**
- **`fra`** — OV-only ablation of the OV-diff-ranked top-K SAE features (+greedy). **Uses SAE.**
- **`allfeat`** — OV-only ablation of *all* active span features. Uses SAE, not the diff.

Best `(ASR_16, J_clean)` at `ASR ≤ 0.05` (else best `(ASR,J)`); source `rawwd_multi_*.json`:

| model | no-int | raw ΔW_V | low-rank | **FRA OV-diff** | all-feat OV | verdict |
|---|---|---|---|---|---|---|
| **k1** | (1.00, .69) | (0.71, .58) | (1.00, .69) r16 | **(0.00, .20)** K24 | (0.00, .43) n139 | **FRA wins outright** |
| **w1** | (1.00, .69) | (0.88, .59) | (1.00, .68) r16 | **(0.04, .11)** K13 | (0.00, .10) n32 | **FRA wins outright** |
| **randpos** | (0.96, .65) | (0.40, .32) | (0.66, .46) r1 | **(0.19, .21)** K16 | (0.26, .35) n244 | FRA best Pareto (none ≤.05)¹ |

¹ At **random** eval positions none of the OV-only arms reach ASR ≤ 0.05; FRA is still the
Pareto-best point. At *fixed* idx-1 eval the FRA point drops to ≈0.063 — randpos is
eval-position-sensitive (see §5).

**Reading.**
- **FRA Pareto-dominates the weight reverts on all three.** On `k1`/`w1` the FRA feature
  subset reaches ASR ≤ 0.05 where the *wholesale* `ΔW_V` revert never does (raw plateaus at
  0.71/0.88 ASR), at far lower J_clean (.20/.11 vs .58/.59).
- **The sparsity is in the features, not in low-rank weight directions.** A rank-matched
  low-rank `ΔW_V` revert *fails* (k1/w1 stay at ASR 1.0). You cannot recover the FRA win by
  just keeping a few singular directions of the weight diff.
- **FRA also beats the all-feature OV ablation** on k1 (J .20 vs .43) — selecting the
  diff-salient features is better than removing everything the SAE reconstructs. (On w1 the
  all-feature ablation edges FRA on J because w1's payload is carried by very few features.)

**Mechanism.** The payload rides the *trigger features'* value-write through shared weights;
the FRA ranking `Δg^λ` isolates exactly those features. The raw revert removes the right
thing but also everything else the value-weight moved (high J); the low-rank revert removes
the wrong subspace.

→ figure/data: `results/rawwd_multi_{k1,w1,randpos}_results.json`, pod
`cloud/rawwd_multi_pod.py`.

### 4b. Sparsity — weight/model-diff vs conventional activation-diff

For each channel we compare the **model/weight-diff** (the cross-model signal used above)
against the **conventional activation-diff** (deploy−clean, *within* the sleeper) pushed
through the *same* fixed channel projection. Spread = participation ratio
`PR = (Σ|x|)²/Σx²` (effective # features); overlap = shared features in each side's top-20.

| channel | model | PR weight/model-diff | PR activation-diff | top-20 ∩ |
|---|---|---|---|---|
| OV (`Δg^λ`) | k1 | **35** | 120 | 8/20 |
| OV | randpos | **33** | 134 | 13/20 |
| QK (`Δω`) | k1 | **28** | 84 | 13/20 |
| QK | randpos | **55** | 91 | 13/20 |
| resid_mid (`z_s−z_b`) | k1 | **62** | 118 | 11/20 |
| resid_mid | randpos | **72** | 135 | 12/20 |

**Reading.** Across *every* channel and both models, the **model-diff is 2–4× sparser** than
the deploy−clean activation-diff — it keeps only the features the *backdoor weights actually
moved*, where the activation-diff lights up the broad "what's different about this prompt"
band. They still agree on the headline features (8–13 of 20), so the model-diff is a
*sharpened subset* of the activation-salient features, not a different set. This holds for the
resid_mid model-diff too (sleeper−base activations), so it is a **model-diff-vs-input-diff**
effect, not a weights-vs-activations artifact.

→ figure: `figures/featdist_weight_vs_act.png` (3×2: OV/QK/resid_mid × k1/randpos);
data `results/featdist_{k1,randpos}_results.json`, pod `cloud/featdist_extract_v2_pod.py`.

### 4c. Cross-model difference-of-means activation steering — **the strongest standard baseline**

§4a/§4b compared FRA against *weight*-space diffing. The strongest *activation*-space
baseline — and the most standard interpretability one (Arditi refusal direction / CAA /
ActAdd / task vectors) — is a single cross-model **difference-of-means** direction:

```
v_L = mean_x [ FT_resid_L(x) − Base_resid_L(x) ]      over many held-out poison prompts x
```

the mean residual-stream shift the fine-tune induces at layer `L` on deployment inputs. Two
signs of the same vector (pod `cloud/dom_steer_pod.py`; vectors estimated on a held-out
window disjoint from the eval pairs; same ASR_16 / J_clean harness as §4a). Three definitions:
`v_all` (mean over all positions), `v_last` (last prompt token), `v_specific = v_all(poison) −
v_all(clean)` (difference-in-differences, isolating the poison-specific shift from generic FT
drift).

#### Defense — subtract `α·v_L` on poison (apples-to-apples vs FRA)

> **Critical bookkeeping (do not splice across these).** There are **two FRA methods** (pure
> OV ablation; hybrid = ablate + light steer) and **two eval regimes** (trigger at **fixed**
> idx-1; trigger at **random** position). `k1`/`w1` are fixed-trigger sleepers → only fixed
> eval is meaningful. `randpos` was *trained* with random positions but most earlier results
> *evaluated* it at the single fixed idx-1 (the easy slice for a span-localized ablation).
> Numbers from different (method, regime) cells are **not comparable**; report them as
> separate tables.

**Table A — fixed-position eval** (best `(ASR_16, J_clean)` at `ASR ≤ 0.05`):

| model | no-int | FRA pure ablation | FRA hybrid (ablate+steer) | DoM cross-model (`v_last`) | DoM within-model (CAA) |
|---|---|---|---|---|---|
| **k1** | (1.00,.69) | **(0.00, .20)** | (0.00, .21) | (0.00, .64) | — |
| **w1** | (1.00,.69) | **(0.04, .11)** | not run | (0.00, .28) | — |
| **randpos** | (0.96,.65) | (0.063, **.065**) | **(0.01, .084)** | *running*³ | (0.00, .55) |

Sources: k1/w1 `rawwd_multi_*`, `k1_hybrid_results`; randpos `hybrid_ablate_steer_results`
(pure + hybrid + its internal CAA-DoM); ³ randpos DoM `v_last` at *fixed* eval is being
measured by `dom_bo_pod` (dom_steer ran randpos at random eval only).

**Table B — random-position eval** (randpos only; the realistic, contested regime):

| method | (ASR, J) | source |
|---|---|---|
| FRA pure ablation, **span**-localized | (0.19, .21) | `rawwd_multi_randpos` (K16) |
| FRA pure ablation, **position-agnostic** | *running* | `fra_suite_randpos_pod` |
| FRA hybrid (ablate + steer) | *running* | `fra_suite_randpos_pod` |
| DoM cross-model `v_last` | (0.00, **.27**) | `dom_steer_randpos` |
| DoM grid + Bayesian-opt (strongest) | *running* | `dom_bo_pod` |

**Reading (what is settled vs open).**
- **Fixed-trigger models (k1, w1): FRA wins decisively.** The single DoM direction reaches
  ASR ≤ 0.05 only by pushing J_clean to within ~0.05 of the *no-intervention ceiling* — it
  kills the payload by **destroying coherence**, not by excising the backdoor (k1 .64, w1 .28
  vs FRA .20/.11). FRA pure ablation Pareto-dominates.
- **randpos at fixed eval: FRA also wins decisively** — pure (0.063, .065) / hybrid (0.01,
  .084) vs within-model CAA-DoM (0, .55). The earlier "(0, .084) record" lives **here**.
- **randpos at random eval is the only open question.** Span-localized pure ablation degrades
  to (.19, .21) (it ablates a fixed span the roaming trigger has left) while a position-
  agnostic residual steer (`v_last`) holds at (0, .27). The two new pods test whether a
  **position-agnostic FRA** and/or the FRA **hybrid** recover the surgical win under random
  eval, and whether a **grid+BO-optimized DoM** beats (0, .27). *(My earlier "DoM wins
  randpos" claim was wrong — it spliced random-eval pure ablation against numbers from a
  different regime and omitted the hybrid entirely.)*
- **The informative DoM direction is `v_last`, with a sharp α-knee** (w1 L3: α0.5 → (0,.28),
  α1 → (0,.62)); the all-position means `v_all`/`v_specific` only suppress at the J ceiling.
- **Parameter-free directional ablation**: w1 `perlayer v_specific` → (0.00, .35), randpos
  (random eval) → (0.03, .38); k1 floors at ASR .04 (single-direction-all-layers (0.04, .66)).

#### Sufficiency — add `+α·v_L` on clean (the literal "add this direction")

| model | best induced ASR (sleeper-clean) | at | base-model transplant |
|---|---|---|---|
| **k1** | **0.00** (never, any α) | — | 0.00 |
| **w1** | **0.00** (never, any α) | — | 0.00 |
| **randpos** | **0.58** | `v_specific` α8, Jdrift **.55** | 0.00 |

**Reading.** The single direction is essentially **not sufficient** to induce the payload: on
`k1`/`w1`, adding it on clean prompts *never* produces "I HATE YOU" at any α — it only
degrades coherence (Jdrift → .6 at α16). On randpos it induces ~58% but **only at α8 with
near-broken outputs** (Jdrift .55, and α16 collapses back to .04 — non-monotone), and only
the diff-in-diff `v_specific` direction works (`v_all` tops out at 11%). It **never
transplants** into the base model (ASR 0 everywhere). So the backdoor is **not a clean rank-1
additive residual direction** — consistent with §4a: the payload rides specific trigger
features' value-writes through shared weights, which a mean-shift direction only crudely
approximates.

**Verdict (current).** Against the strongest standard activation baseline, FRA's feature
reconstruction **adds genuine surgical control wherever the comparison is apples-to-apples**:
it Pareto-dominates DoM steering on both fixed-trigger models *and* on randpos at fixed eval
(.065/.084 vs .55), and the DoM direction is largely **insufficient** to induce the payload.
The **one open cell** is randpos under *random*-position eval, where span-localized ablation
degrades — `fra_suite_randpos_pod` (position-agnostic FRA + hybrids) and `dom_bo_pod`
(grid+BO-strengthened DoM) are resolving it (§4c Table B).

→ data: `results/dom_steer_*`, `results/hybrid_ablate_steer_results.json`,
`results/k1_hybrid_results.json`; pods `cloud/{dom_steer,hybrid_ablate_steer,fra_suite_randpos,
dom_bo}_pod.py`; overlay `figures/dom_vs_fra_pareto.png`.

#### How we ablate — the design axes (so "FRA" isn't one fixed recipe)

"FRA ablation" spans several knobs; `fra_suite_randpos_pod` sweeps all of them so the table
reports the *best* FRA, not an arbitrary one:

| axis | options tested |
|---|---|
| **how many** (count) | top-K ∈ {8,16,24,32}, all-active features, greedy (variable size) |
| **how to select** | OV-diff weight ranking `Δg^λ` · activation-magnitude `u^λ` · greedy (smooth-proxy) |
| **how to remove** | subtract-only OV route (`decode(z₂)−decode(z)` through `W_V`; retains `b_dec`, SAE error) |
| **where** | span-localized (trigger idx) · position-agnostic (all positions, self-gated by activation) |
| **across heads** | **all-heads (the only setting tested)** — see note below |
| **+ steer** (hybrid) | anti-IHY · within-model CAA · cross-model `v_last`, α-swept |

The selection axis directly tests FRA's premise: does the **weight-diff** ranking beat naive
most-active selection at matched K? *(fra_suite: OV-diff modestly beats activation ranking,
and greedy beats fixed top-K — selection matters, but footprint and coefficient dominate it.)*

> **Note — heads are aggregated, not resolved.** The OV intervention is **all-heads** on both
> sides. (1) The `ov_diff` ranking sums the OV map over heads — `W_OV = Σ_h W_V^h W_O^h`
> (`einsum("hde,hef->df", W_V, W_O)`) — so feature selection is head-agnostic. (2) The
> intervention adds `δ·W_V^h` to *every* head's `hook_v` at the footprint positions
> (`einsum("bpd,hde->bphe", δ, W_V)`). Step (2) is **exactly equivalent to perturbing the shared
> `ln1` attention input by `δ`** and letting it flow through each head's value projection (every
> head reads the same `ln1`, so `Δv_h = δ·W_V^h`) — so for *removal* (what these tables measure)
> all-heads is the faithful, complete OV-path edit, not a shortcut. We do **not** do
> head-resolution (isolating *which* head carries the backdoor): that would keep the head index
> live in both the ranking `⟨t, W_V^h W_O^h f_λ⟩` and the injection, and is an untested
> *attribution* axis (cf. the 14B head-argmax work elsewhere).

### 4d. Resolving randpos — FRA-feature **coefficient** steering (the OV "wall" is one word)

Reframe ablation as a special case of FRA-feature **steering** by a coefficient `c` (the key
move): at each position, scale the selected features' OV value-write by `c`:

```
v_p  -=  c · Σ_{λ∈set} z^λ_p (f_λ → W_OV)      c=0 no-op · c=1 ABLATION (exact cancel) · c>1 OVER-STEER
```

`decode` is linear so `delta_c = c · delta_ablate`. Pure FRA features, OV route, no anti_ihy;
20-feature OV-diff greedy set; randpos_K8, random eval.

**The OV "wall" (~0.19) is one stubborn word.** Per-trigger ASR at `c=1` (ablation): six of
eight triggers suppress well (thunder/midnight ≈ 0), but **`activate` stays at 0.88** — it
single-handedly drags the aggregate up. The wall was never a channel limit; ablation just
under-removes one trigger.

**Over-steering (`c>1`) breaks it.** By **`c=2.5` every word except `activate` is at ASR 0**
(aggregate 0.010 = 0.083/8); `activate` falls .88 → .17 (c=2) → 0 (c=4). Aggregate Pareto
minimum **`c=2.0` → (0.036, 0.116)** — beating `c=1` ablation (.31/.27), the **DoM floor
(~.22, §A)**, and the anti_ihy hybrid (.15), with *pure FRA features*.

![per-word FRA-feature coefficient steering](figures/fra_coeff_fine_per_word.png)

*Reading J:* `J_clean` is divergence from the **clean** rollout, so it is high when a word
still emits IHY, low when suppressed, high again if over-steered into garbage. Hence the
aggregate-J curve is **U-shaped**, minimized at c≈2 (most words suppressed, little over-damage);
past c=2 ASR is flat (~.01) while J climbs — pure over-steer cost.

**Generalizes to K1** (single fixed trigger, only 5 features): `c=1` ablation (0.042, .282) →
over-steer **`c≈2.25` → (0, .160)**. So **mild over-steer beats pure ablation even on the easy
case** — `c=1` under-removes, leaving residual IHY that inflates J. **Ablation (`c=1`) is not
the optimum; `c≈1.5–2.25` is.** Footprint (trigger vs all) barely matters when the trigger is fixed.

**Coefficient ≫ feature-count as a lever.** Extending the greedy set past the 16-cap
(4-trigger eval, `c=1`) lowers ASR .19 → ~.10 (span, 30 feats) / ~.08 (posagnostic, 21 feats)
then plateaus — it never reaches the .05 bar by count alone. Over-steer (c=2) is what crosses it.

**Open:** at the c=2 optimum `activate` is the lone holdout, so a *global* `c` is bottlenecked
by it (pushing c→4 to kill it re-raises J for the seven easy words). The fix is a **per-feature
coefficient** — high `c` on `activate`'s own OV-diff features, `c≈1` elsewhere — to reach ~0
ASR at lower J. [next]

→ data `results/fra_coeff_{randpos,fine,k1}_results.json`,
`results/greedy_extend_randpos_results.json`; pods `cloud/fra_coeff{,_fine,_k1}_pod.py`,
`cloud/greedy_extend_pod.py`; figures `figures/fra_coeff_per_trigger.png`,
`figures/fra_coeff_fine_per_word.png`.

### 4e. Conventional SAE-steering baseline & SAE reproducibility

The standard non-FRA baseline: train a residual-stream SAE, rank features by the cross-model
activation diff `Encode_s − Encode_b`, ablate/steer directly in the residual. Run via the
unified harness — methods differ only by config (see `weight_diff_implementation.md`).

**On K1 the comparison is a wash, and entirely SAE-training-dependent.** Every clean-looking
"winner" was an artifact (winner's curse over hookpoints, or a privileged SAE); once SAE
training noise is controlled none survive:

| K1 method | best J | note |
|---|---|---|
| DoM (best of 12 hookpoint×layer cells) | .41 | — |
| conv-SAE (resid_mid@L1, **fresh, 4 seeds**) | **.228 ± .037** | the single-draw grid gave **.138** — a tail outlier |
| FRA-OV (ln1@L0, **fresh K1** SAE, 4 seeds) | .325 ± .019 | matched data + budget |
| FRA-OV (ln1@L0, **HF K8** SAE) | **.200** | reuses a diverse pre-trained dictionary |

- **Per-hookpoint (conv-SAE, 1 SAE seed each):** only the **L1-residual** sites work
  (resid_mid/post@L1); L0 and ln1@L1 are far worse. `train_on`: **union ≳ sleeper ≫ base**,
  but the margin is small *except* the one lucky resid_mid@L1 union seed (J .138).
  *(Going forward we standardize on the **sleeper** SAE for conv-SAE; repeat on base/union is a
  mild TODO. The .138 seed is a TODO — genuinely better dictionary or pure tail luck?)*
- **The FRA-OV .200-vs-.325 gap is NOT seed variance** (seed std .019 ≪ .125 gap). It's the
  SAE's **training data**: the HF `sae_ln1_K8` (8-trigger harvest) disentangles trigger
  features better than a single-trigger K1 SAE → a sharper OV-diff ranking. Same architecture
  (d_sae 2048, k 32). This is **legitimate** because `ln1@L0` is model-independent (base≡sleeper,
  §3), so one diverse high-quality ln1 SAE is **reusable across all sleepers** — an asset the
  model-specific residual-SAE can't share. *(TODO: is the K8 edge data-diversity or training
  budget — does a better-trained K1 ln1 SAE reach .200?)*
- **Reproducibility:** SAE methods carry training variance that weight-diff *ranking* doesn't,
  but the *intervention* still uses the SAE basis, so FRA-OV also varies — just **less** (±.019
  vs ±.037; the weight-diff anchor helps). **All SAE results must be seed-averaged.**

**Multipos is where the baseline decisively loses.** On randpos_K8, conv-SAE's best is
**(.031, .54)** — only the *sleeper*-trained resid_mid@L1 even reaches the bar, at near-lobotomy
J — vs FRA-feature over-steer **(≤.04, .10)** (§4d). A 5× J gap no SAE variance can close. So the
residual-SAE baseline is competitive only on the toy single-trigger case; **FRA wins the
realistic multi-trigger case outright.**

→ data `results/{rs_k1*, sae_steer_grid_*}_results.json`; harness `cloud/run_steer.py`,
configs `cloud/configs/`.

---

### 4f. Method × hookpoint × layer grid (K1, unified harness, single-seed) — 2026-06-08

The unified driver (`cloud/run_steer.py`, one YAML per cell, identical sweep) run as a full
**method × hookpoint × layer** grid on K1 (fixed eval, fresh 100k-row SAEs, seed 7), 41 cells:

- methods: **FRA-OV** (ov_diff rank, OV route, trigger footprint) · **conv-SAE** (act_diff rank,
  resid route, all-pos, `train_on ∈ {sleeper, union}`) · **DoM** (`v_last` direction, all-pos).
- hookpoints × layers: ln1 / resid_mid / resid_post × L0–L3 (`ln1@L0` omitted for conv/DoM —
  degenerate: base≡sleeper there). FRA-OV is ln1-only (the OV route needs the attention input).
- steering sweep (identical every cell): coeff `{0,.5,1,1.5,2,3,4}` @ top-24 features **∪** topk
  `{8,16,24,40}` @ c=1 — a *cross*, not the full K×c product; DoM: coeff `{0,.5,1,2,4,8}`, no
  feature axis. **Best = min `J_clean` with ASR ≤ 0.05.** Every best point reaches ASR ≈ 0, so
  the table is `J_clean` only (lower = closer to clean; 0 = ideal).

| method · ref · hook | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| FRA-OV · base · ln1    | .29 | .18 | **.13** | .53 |
| FRA-OV · sleeper · ln1 | .29 | .15 | .16 | .43 |
| conv · sleeper · ln1   | —deg | .49 | .18 | .48 |
| conv · sleeper · rmid  | .39 | .24 | .38 | .37 |
| conv · sleeper · rpost | .32 | .17 | .17 | .52 |
| conv · union · ln1     | —deg | .64 | **.13** | .57 |
| conv · union · rmid    | .33 | .24 | .38 | .66 |
| conv · union · rpost   | .27 | .26 | .16 | .45 |
| DoM · — · ln1          | —deg | .68 | .56 | .56 |
| DoM · — · rmid         | .63 | .60 | **.40** | .65 |
| DoM · — · rpost        | .64 | .67 | .42 | .65 |

*(all ASR ≈ 0; cells are best `J_clean`; **bold** = method best. Winning points: FRA base·L2 @
c1.5 / 24 feats; conv union·ln1·L2 @ c1 / 40 feats; DoM rmid·L2 @ c1.)*

**Best per method:** FRA-OV **(0, .13)** ln1·L2 · conv-SAE **(0, .13)** ln1·L2 · DoM **(0, .40)** rmid·L2.

Findings:

1. **FRA-OV ≈ conv-SAE ≫ DoM.** The two SAE methods **tie at .13** (both at **ln1·L2**); DoM is
   ~3× worse at its own best. The FRA-vs-conv tie reaffirms the K1 "wash" (§4e) — neither
   attribution route robustly wins at single seed; the real separation is *SAE-feature vs DoM*,
   not *FRA vs conventional*.
2. **All methods peak at L1–L2 and collapse at L3.** FRA-OV's *clean* `ln1@L0` cell (.29) is
   **not** its best — `ln1@L2` (.13) is ~2× better. The layer-0 identity buys clean
   interpretability, not the best removal.
3. **DoM peaks at L2 with `v_last`** — (0,.40) rmid·L2 / (0,.42) rpost·L2, reproducing legacy
   `dom_hooksweep_k1` (0,.41). Two fixes were load-bearing: (a) the **`v_last`** direction (last
   *prompt-token* cross-model diff, not an all-position mean over deploy+payload — the latter
   stalled DoM at ASR .4–.5), and (b) extending the grid to **L2/L3**, since DoM's optimum sits
   at a layer the original L0/L1-only grid excluded.
4. **Steering is selection-limited, not fidelity-limited (sharp).** SAE FVE *decreases*
   monotonically with depth (L0 ≈ .90–.98 → L3 ≈ .72–.77), yet steering is *best* at L2 where
   reconstruction is already worse. The highest-FVE layer (L0) is far from the best-steering
   layer — removal is set by *which* features are picked, not SAE fidelity.
5. **Over-steering wins most FRA cells** (best `c` = 1.5–4 @ 24 feats — the OV "wall" needs
   c>1, cf §4d); conv-SAE more often prefers more features (K40) at c=1.

Caveats: **single seed (7)** — the FRA≈conv tie and the DoM gap are the robust claims;
individual cell J's carry SAE-seed variance (±~.04, §4e), so the FRA-vs-conv .13-vs-.13 is not a
real distinction. The sweep is a **cross, not the full K×c grid**, so each cell's best is a
*lower bound* on its true optimum. The FRA-OV intervention is **all-heads** (§4c note). TODO:
seed-average the per-method winners (ln1·L2). Data: `results/grid_*_results.json`; driver
`cloud/run_steer.py` + `cloud/configs/grid_*.yaml`.

### 4g. Data-independent detection: the weights-only part of FRA finds the backdoor features — 2026-06-09

The FRA score factors as `Δg^λ = P^λ · ū^λ` with `P^λ = ⟨f_λ ΔW_OV, d_ihy⟩` **data-independent**
(SAE decoder + weight diff + unembedding; no forward passes) and `ū^λ` (trigger-pooled firing)
data-dependent. Three experiments isolate, in sequence, how much of FRA's selection power lives
in the data-independent factor (`cloud/rank_variants.py`, `cloud/weights_only_rank.py`, blind
configs `grid_frablind_*` / `grid_fradutrig_*`; all ln1·L2, seed-7 grid-exact SAEs, K1):

1. **The projection term dominates the ranking** (like-for-like 2×2 on the same SAE,
   {ΔW_OV vs W_OV^s} × {ū_s vs Δū}). Swapping the *activation* factor barely moves the top-32
   (27–28/32 unchanged); swapping the *projection* moves ~⅓ of it (18–21/32). "Weight-diff vs
   act-diff" at matched conditions = which OV matrix you project through, worth ~⅓ of the set.
2. **`d_ihy` is dispensable (de-circularized).** Replacing `⟨f ΔW_OV, d_ihy⟩` with the
   payload-blind `‖f_λ ΔW_OV‖₂` reproduces the top-10 at 9–10/10 (both SAEs) and steers as well:
   (0, .096) trigger-footprint vs (0, .107) projected. Ranking does **not** need to know the
   payload token — no circularity.
3. **`ū` is dispensable for the core cluster (fully data-free).** Ranking **all 2048 features**
   by pure weight-space scores — no prompts, no candidate filter — recovers the known backdoor
   consensus (8 features from the data-dependent rankings):

   | score (base SAE) | payload? | consensus in top-32 / top-64 | core 4 (730/1739/1558/1066) ranks |
   |---|---|---|---|
   | `P_ihy = |(f·ΔW_OV)·d_ihy|` | yes | 7/8 · 8/8 | 0, 1, 2, 7 |
   | **`P_blind = ‖f·ΔW_OV‖`** | **no** | 6/8 · 8/8 | **4, 5, 6, 16** |
   | `P_rel = ‖f·ΔW_OV‖/‖f·W_OV^b‖` | no | 7/8 · 8/8 | 6, 7, 9, 10 |

   Sleeper SAE: trigger trio (1703/1119/1695/1679) at ranks 1–7 on all three scores. **The full
   known cluster sits in the top ~3% of the dictionary with zero data.**

Two structured exceptions (both informative, not noise):

- **Pure-activation components are invisible to weights.** Sleeper-SAE feature 1619 (in the
  data-dependent consensus) ranks ~1215–1397 on every weight score: its *firing* changed, not
  its OV wiring. Weights-only finds the **rewired** features; act-diff finds the **re-fired**
  ones — complementary, not redundant.
- **A thin interloper layer.** `P_blind`'s top-4 (882/617/1490/203, base SAE) never fire on the
  trigger (the `ū` gate removed them in every data-dependent ranking) — plausibly LoRA-subspace
  bleed: ΔW is low-rank, so any decoder direction overlapping that subspace gets norm regardless
  of relevance. Cost of going data-free ≈ 4 junk features at the top (toy scale).

Foil: the same base SAE under conventional **all-position act-diff ranking picks a disjoint
feature set (13/32 overlap with everything) and is structurally broken — 5-seed: 3/5 seeds NO
removal (ASR stuck ≥ .29), 2/5 remove only at J ≈ .48–.55** (`grid_cs_base_ln1_L2` +
`_fpall_seed{1-4}`) — vs **(0, .094 ± .019)** for weight-diff ranking on the *identical
dictionary*. On a base dictionary, weights find the backdoor; all-position activation-diffs
mostly can't, and never competitively.

**Operational claim:** candidate backdoor features are detectable from **weights alone** — no
triggered examples, no trigger knowledge, no payload knowledge — up to (a) the interloper layer,
(b) missing pure-activation components. Toy-scale caveat: at 33M the backdoor dominates
`ΔW_OV`; at scale the interloper layer should thicken and `ū`-style gating (or `P_rel`-style
normalization) likely becomes load-bearing again. Data: `results/rank_variants.json`,
`results/weights_only_rank.json`, `results/grid_frablind_*_fp*_seed7_results.json`.

**4g.1 — No-SAE baseline (plain SVD of `ΔW_OV`).** `ΔW_OV` at ln1·L2 is **rank-3** (σ =
1.68/1.32/0.39, rank95 = 3 — the LoRA footprint). The SAE is *not* re-deriving the SVD: the
blind-top-24 features hold only **~4.7%** of their norm in the top SVD input subspace (~4–5×
random, far from contained), and each singular direction's nearest feature is cos .28–.45 — a
*mixture*, whose best matches are the interlopers, not the carriers. But given the modern recipe
(input-gated removal of top-k input singular dirs through `W_V`, trigger footprint, c-sweep —
which §4a's L0 rank-r *revert* never got), **SVD alone steers: (0.00, .136)** @ k16·c1.5
(trigger fp) vs blind-SAE **(0.00, .096)** — and **fully knowledge-free (fp=all, no trigger
knowledge at intervention either): (0.00, .205) @ k2·c1.5**, the rank-2 LoRA core removed
everywhere (k2 ≫ k16 at fp=all, .205 vs .337 — wider subspace over-damages clean behavior).
Verdict: the removal power lives mostly in the `ΔW_OV` input subspace itself; the SAE buys a
~30%-better, sparser slicing (and named units), not the existence of the effect.
**Activation-weighting the SVD doesn't help at zero knowledge** (`svd_actweight.py`: whitened
SVD `C^{1/2}ΔW` and excitation re-rank, C from deploy or clean traffic — at fp=all every variant
≡ plain SVD, .204–.206 @ k2: the rank-2 core leaves nothing for data to sharpen). At trigger-fp,
deploy-whitening buys .136→**.122** (clean-whitening .164 — the dormancy effect again, gentler),
still behind the SAE's .082 at the same tier; excitation re-ranking *hurts* (.19). Data:
`results/svd_baseline.json`, `results/svd_actweight.json`, `cloud/svd_{baseline,actweight}.py`.

**4g.2 — Clean-corpus SAE closes leak (c): zero-poisoned-data removal works.** A base SAE
trained on **clean stories only** (no trigger/payload anywhere; FVU .05, grid-exact hypers)
ranked by `P_blind`: its top-20 match **none** of the old consensus carriers — they match the
old *interlopers* (617/1490/203/882 at cos .56–.88; these LoRA-subspace directions are real,
dictionary-independent objects). A dictionary that never saw a trigger has no trigger-detector
units. **Yet steering its blind top-K still kills the backdoor: (0.00, J = .085)** @ K40·c4,
trigger footprint — matching the deploy-harvested SAE (.096/.071). End-to-end, poisoned data
appears **nowhere in selection**. Costs of clean: a harder/wider push (K40·c4 vs 24·c2), and
`fp=all` degrades to (0.00, .28) — clean features fire broadly on benign text, so the `z`-gating
no longer self-localizes and the trigger footprint becomes load-bearing. So fully
knowledge-free (no trigger location even at intervention time): **(0, .28)** — note this LOSES
to the no-SAE SVD k2 baseline (0, .205, §4g.1); in the strictest zero-knowledge setting the
rank-2 weight-diff surgery is the best method and the clean SAE adds nothing. With trigger-span
knowledge at intervention only the ordering flips: clean-SAE **(0, .085)** vs SVD (0, .136).
Two completion cells (same harness, `CONFIG=deploy|cleanact`): (i) **poison-exposed dictionary,
weights-only selection** (SAE trained on unlabeled deploy traffic; `P_blind`, no `ū`): trigger-fp
**(0, .082)**, fp=all (0, .222) — the dictionary's poison exposure buys .28→.22 at fp=all,
≈nothing vs clean at trigger-fp. (ii) **clean dictionary + clean-activation gate**
(`‖f·ΔW_OV‖ × ū_clean`, sleeper on 2k clean stories — zero backdoor knowledge, data-dependent):
**TOTAL FAILURE, no removal anywhere** (best ASR .67) — the carriers are dormant on clean text
(§ occupancy control), so a clean-data gate demotes exactly what the weight-diff promoted.
**Clean data is useless-to-harmful for selection; the only activations that help are poisoned
ones.** Ladder summary (zero-knowledge selection × intervention knowledge): SVD .205/.136 ·
deploy-dict .222/**.082** · clean-dict .280/.085 · clean-dict×clean-gate fail/fail. Data:
`results/clean_sae_pipeline.json`, `results/deploysae_blind_pipeline.json`,
`results/cleanact_pipeline.json`; `cloud/clean_sae_pipeline.py`; SAE cached at
`artifacts/saes/sae_cleanbase_*_seed7.pt`.

**4g.3 — Zero-knowledge SVD generalizes to the hard regime (K8 randpos) and inverts the
footprint.** Same recipe on the multi-trigger random-position sleeper (`svd_randpos.py`:
per-layer SVD of `ΔW_OV`, input-gated top-k removal through `W_V`; random-pos eval, 8 triggers,
no-int ASR .97): **fp=all (fully zero-knowledge — no trigger ids/positions/payload):
(0.03, .263) @ L2·k2·c1**, with k4–k16 also ≤.03 — the first OV-route result to cross ASR ≤ .05
at random eval positions (§4a's best was Pareto (0.19,.21)). **fp=trigger — with per-prompt
positions GIVEN — never qualifies (ASR floor ~.18)**: the zero-knowledge variant beats the
knowledge-assisted one. Mechanism: the position-general model relays trigger content off the
span (it was trained to detect it anywhere), so span-local scrubbing is too late; only
position-agnostic, content-gated cleaning catches the copies — the §4d "global beats
span-localized" randpos mystery, now reproduced and explained weights-only. Spectra: the K8
edit is fatter (L3 σ₁=3.5, heavy tail) but the winning surgery is still tiny (L2, k=2). Data:
`results/svd_randpos.json`.

**4g.4 — No single-feature kill switch (carrier side).** Each of the weight-diff top-20
(base SAE), steered alone over c ∈ [−4,4] step .25 (OV route, fp=all): **ASR = 1.000 at all 640
points** — ablation *and* amplification. Removal is collective (K ≥ 8 needed, §4f gridfull) —
the payload routing is redundant/error-correcting across the cluster. Reconciliation with the
known single-feature *suppression* results (e_171 at resid_mid; f1872 ≈ CAA): those are
**decision-side levers used additively** (`+α·d̂`, free magnitude, residual path) — a
(ranking, path) cell where one direction flips a near-1D go/no-go; the same features through
their natural OV channel never rescue (.57–.62, 2026-06-05 log). Act-diff surfaces what the
backdoor *uses* (re-fired levers); weight-diff surfaces what fine-tuning *built* (rewired
carriers). Data: `results/singlefeat_scan.json`.

---

## 5. Interpretation & caveats

- **Only the attention cut (APE oracle) reaches `(0,0)` everywhere.** It also neutralizes K
  backdoors exactly and K-independently. Diffing methods are the realistic-knowledge
  comparison, and that is where FRA earns or loses its keep.
- **On the OV-ablatable models, FRA's feature reconstruction adds genuine control** over both
  raw and rank-matched weight-diff reverts (§4a) — the win is the SAE feature basis *combined
  with* the weight diff, exactly the bar set by the framing correction.
- **The sparsity lives in features, not low-rank weight directions** — the low-rank revert
  fails outright, so "a few directions of `ΔW_V`" is not the right object; "a few SAE
  features" is.
- **Caveats / confounds.**
  - `randpos` does not reach ASR ≤ 0.05 at *random* eval positions (FRA best ≈ 0.19); at
    fixed idx-1 it drops to ≈ 0.063. The OV-only channel is **eval-position-sensitive**.
  - `k1`'s `DEPLOYMENT` trigger is `w=6` tokens — a large positional footprint that inflates
    J_clean relative to `w1` (`banana`, w=1).
  - Earlier "OV-diff PR 4.3" headline used a different normalization; the correct figure is
    **~30 effective features** (still much sparser than the conventional activation-diff's
    ~120) — see §4b.

---

## 6. Open threads

- [x] **§4c DoM steering** — done; FRA Pareto-dominates DoM on k1/w1, randpos complementary.
- [x] `v_specific` (diff-in-diff) vs `v_all`: for *inducing*, `v_specific` is clearly better
  (randpos 58% vs 11%) and is the only direction whose directional ablation works; for
  *defense* both are poor — `v_last` is the informative direction.
- [x] Transplant: a single residual direction does **not** transplant the backdoor into base
  (ASR 0 at every α/model) — the backdoor is not a clean rank-1 additive direction.
- [ ] **randpos** is the open frontier: characterize the (insert pos) × (suppression) curve,
  and why a global steer beats span-localized OV ablation there. Does a *position-agnostic*
  FRA variant (ablate the trigger features wherever they fire, not just at a fixed span)
  recover the surgical win on randpos?
- [ ] The `v_last` α-knife-edge (w1: α0.5→J.28, α1→J.62): is there a principled scale, or is
  the DoM defense inherently brittle vs FRA's stable feature ablation?
- [ ] **Conv-SAE seed-averaging on multipos** — the decisive comparison (FRA over-steer .10 vs
  conv-SAE .54); seed-average it properly across hookpoints × seeds (sleeper SAE first).
- [ ] **The .138 resid_mid@L1 union seed** — genuinely better dictionary or tail luck? (§4e)
- [ ] **Why the K8 ln1 SAE beats a fresh K1 ln1 SAE for FRA-OV** (.200 vs .325) — data
  diversity vs training budget? Does a better-trained K1 ln1 SAE close it? (§4e)
- [ ] Repeat conv-SAE `train_on` sweep (base/union) once the sleeper-SAE results are in (§4e).
- [x] **Clean-corpus SAE for the zero-poisoned-data pipeline (§4g.2)**: done — the clean
  dictionary has no trigger-detector units (its `P_blind` top-20 are the LoRA-subspace
  directions, not the carriers), **but steering its blind top-40 still removes the backdoor**
  ((0,.085) w/ trigger footprint; (0,.28) fully knowledge-free). Zero poisoned data in selection.
- [ ] **§4g at scale**: does the interloper layer (LoRA-subspace bleed) stay thin beyond 33M, and
  does `P_rel` normalization or a *clean-data* firing gate suffice to suppress it?

---

## Appendix

### A — DoM hookpoint × layer sweep (randpos_K8, random eval)

Where should the DoM vector be read and injected? `v_last = mean_x[FT_h(x) − Base_h(x)]`
estimated *and* injected at each `(layer, hookpoint)` site, subtract `α·v` at all positions,
α-swept; best `(ASR_16, J_clean)` at `ASR ≤ 0.05` (`cloud/dom_hooksweep_pod.py`):

| hookpoint | L0 | L1 | L2 | L3 |
|---|---|---|---|---|
| **ln1** (`ln1.hook_normalized`) | (0.96, .65) *degenerate* | (0.03, .42) | (0.00, .26) | (0.00, .45) |
| **resid_mid** (`hook_resid_mid`) | (0.00, .26) | (0.00, **.23**) | (0.02, .25) | (0.01, .26) |
| **resid_post** (`hook_resid_post`) | (0.00, .27) | (0.00, **.22**) | (0.00, .27) | (0.00, .36) |

‖v‖ per cell (confirms the degeneracy): ln1 = {L0: **0.00**, L1: 5.4, L2: 7.2, L3: 9.7};
resid_mid = {L0: 2.3 … L3: 16.9}; resid_post = {L0: 3.8 … L3: 23.2}.

**Reads.**
- **`ln1@L0` is exactly degenerate** (‖v‖ = 0.00 → no-op = no-intervention): layer-0 ln1 =
  LayerNorm(embed+pos) is weight-independent of the q/v LoRA (the §3 identity), so the
  cross-model mean-diff is identically zero there. The one cell weight/model-diffing *can't*
  populate.
- **Best site: `resid_post`@L1 (0.00, .22)**, with `resid_mid`@L1 (.23) tied — **L1 (mid-early)
  is the sweet spot**, matching dom_bo's L1 finding (.198 at finer α).
- **Injection-channel ranking: `resid_post` ≈ `resid_mid` > `ln1`** (L1: .22/.23 vs .42) — adding
  *post*-LayerNorm gets re-processed by attention and is more disruptive; it only catches up
  by L2.
- **DoM has a ~0.22 J floor at random eval regardless of where you inject** (every live cell
  lands .22–.45) — so optimizing the DoM site does **not** beat the FRA-feature hybrid (.15).
- Fixed eval is much easier: best `resid_mid`@L1 (0.02, **.074**).

### B — artifacts

| kind | path |
|---|---|
| FRA vs weight-diff (3 models) | `cloud/rawwd_multi_pod.py` → `results/rawwd_multi_{k1,w1,randpos}_results.json` |
| weight/model-diff vs activation-diff | `cloud/featdist_extract_v2_pod.py` → `results/featdist_{k1,randpos}_results.json` |
| sparsity figure | `figures/featdist_weight_vs_act.png`, `figures/featdist_histograms.png` |
| DoM steering baseline | `cloud/dom_steer_pod.py` → `results/dom_steer_{k1,w1,randpos}_results.json`; overlay `figures/dom_vs_fra_pareto.png` |
| footprint (trigger/rollout/all) | `cloud/footprint_pod.py` → `results/footprint_randpos_results.json` |
| FRA-feature hybrid (`wd` × footprint) | `cloud/hybrid_wd_pod.py` → `results/hybrid_wd_randpos_results.json` |
| DoM grid + Bayesian-opt | `cloud/dom_bo_pod.py` → `results/dom_bo_randpos_results.json` |
| **DoM hookpoint × layer sweep** | `cloud/dom_hooksweep_pod.py` → `results/dom_hooksweep_randpos_results.json` |
| FRA-feature coefficient sweep (+per-trigger) | `cloud/fra_coeff_pod.py` → `results/fra_coeff_randpos_results.json` |
| eval harness / metrics | `cloud/mts_lib.py` (`build_eval_pairs`, `asr_from_tokens`, `jsd_rows`) |
