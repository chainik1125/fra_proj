# Use vs Presence: a fully-worked toy for in-context concept removal

*A pedagogical walkthrough of `experiments/fra_hmm_toy/` — every input, every
intervention, every number. Written 2026-07-14, after the robustness battery.
Companion docs: `THEORY.md` (conclusions), `README.md` (pre-registrations).*

---

## 0. Why this experiment exists

Across the in-context line of work we kept hitting the same confusing wall:
FRA — which surgically beats steering baselines on narrow attention-mediated
behaviors (induction, in-context backdoors; `experiments/fra_win/`) — fails
to remove *broad* in-context concepts (PII recall, `experiments/fra_pii/`),
while crude SAE feature ablation does better. Real-model experiments couldn't
say *why*: too many unknowns (is the SAE bad? is the concept found? is the
edit reaching?).

This toy removes every unknown. The concept ("which HMM component mixture is
this sequence drawn from?") has a **closed-form Bayes-optimal value at every
position**, the model's output **factorizes exactly** into a concept part and
a non-concept part, and the TopK SAE makes the FRA decomposition **exact to
float precision** (verified ≤ 3e-5). Every intervention effect is measured
against analytic ceilings, not vibes.

The headline discovery: **"remove the concept" is two different tasks —
removing its *use* and removing its *presence* — with different best tools,
different costs, and an information-theoretic wall between them.**

---

## 1. The data-generating process (exact inputs)

Three MESS3 hidden-Markov components, each with 3 hidden states and 3
symbols, at different dynamical regimes:

| component | mess3 params | character | vocab block |
|---|---|---|---|
| 0 | x=0.08, a=0.90 | sticky (long same-symbol runs) | tokens {0,1,2} |
| 1 | x=0.25, a=0.65 | moderate | tokens {3,4,5} |
| 2 | x=0.40, a=0.34 | diffuse (near-memoryless) | tokens {6,7,8} |

**Each sequence** (length 256 for eval; 768 for training, see §3):

1. Draw a latent mixing vector **ω ~ Dirichlet(10 · [0.40, 0.35, 0.25])**.
   ω is the "concept" — a per-sequence hidden property, never observed.
2. At each timestep t: draw component c_t ~ Cat(ω); that component emits one
   symbol s_t ∈ {0,1,2} from its current belief state and updates **only its
   own** hidden state; the emitted global token is **v_t = 3·c_t + s_t**.

Because the vocab blocks are disjoint, every token *announces* which
component emitted it (block = v ÷ 3) — this is the "distinct-vocab" regime
where single-token SAEs work at all (the shared-vocab variant is the
separation_scaling study in temp_xc). What is NOT announced is ω itself: it
must be estimated by **counting**.

**Worked example** (eval sequence #270, the one used throughout):
true ω = (0.139, 0.834, 0.027). First 24 tokens:

```
tokens : 5 3 4 2 4 4 4 4 4 3 8 3 6 2 4 3 5 4 4 5 4 4 4 5
block  : 1 1 1 0 1 1 1 1 1 1 2 1 2 0 1 1 1 1 1 1 1 1 1 1
symbol : 2 0 1 2 1 1 1 1 1 0 2 0 0 2 1 0 2 1 1 2 1 1 1 2
```

**The Bayes-optimal concept estimate is literally a count.** After t+1
tokens with per-block counts n = (n₀,n₁,n₂), the posterior mean of ω is

    ω̂(t) = (α + n) / (10 + t + 1),   α = 10·(0.40, 0.35, 0.25) = (4, 3.5, 2.5).

Check it by hand at t=16 (17 tokens seen, counts n = (2, 13, 2)):
ω̂ = (4+2, 3.5+13, 2.5+2)/27 = (6, 16.5, 4.5)/27 = **(0.222, 0.611, 0.167)** —
matching the pipeline's stored `posterior_omegas` exactly (0.22/0.61/0.17).

**The Bayes-optimal next-token prediction factorizes:**

    p*(v_{t+1} = 3c+s | x_{0:t}) = ω̂_c(t) · p_c(s | belief_c(t))
    ─────────────────────────      ──────    ────────────────────
        the full prediction        CONCEPT     GRAMMAR (belief-state
                                   (block)     tracking within c)

so the log-loss splits **additively and exactly**:
log p(v) = log p(block) + log p(v | block). All removal metrics live in the
first term, all collateral metrics in the second. No leakage between them by
construction.

**Analytic anchors** (eval set, 500 seqs × 256, positions t ≥ 8):

| anchor | value (nats/token) | meaning |
|---|---:|---|
| Bayes block CE | 0.9872 | best possible concept use |
| prior block CE | 1.0809 | concept fully unused (predict prior mean) |
| → concept is worth | **0.0937** | the whole "use" scale |
| Bayes within CE | 0.7785 | best possible grammar |
| uniform within CE | log 3 = 1.0986 | grammar destroyed |
| → grammar range | **0.3201** | the whole "collateral" scale |

---

## 2. The model and the SAE (what gets intervened on)

**Transformer** (recipe from sae_day experiment_03): TransformerLens
HookedTransformer, 3 layers, d_model=64, 2 heads (d_head=32), d_mlp=256,
LayerNorm, GELU, learned positional embeddings, n_ctx=256. Trained 2500 steps
(batch 128, lr 1e-3, Adam) on random 256-token crops of 768-token training
sequences (1500 of them, seed 42).

> ⚠ Methodological trap we fell into first (`out/main` = the corpse): if you
> train on full-length sequences (seq_len == context), the model sees the
> same 1000 sequences ~190 times and **memorizes** — train loss goes BELOW
> the Bayes ceiling and eval behavior is worse than the context-free prior.
> The exp03 recipe's random cropping is load-bearing data augmentation.
> Keep train sequences ≥ 3× n_ctx.

**Clean model quality** (eval, held-out): block CE 0.9881 — it uses **99% of
the available concept information** ((1.0809−0.9881)/0.0937). Its block-mass
vector m(t) (probability it assigns to each vocab block for the next token)
tracks ω̂(t) with **R² = 0.992**. Grammar: within CE 0.9743 — captures 39% of
the within-block range (a decent but not Bayes-level belief tracker; fine,
since all comparisons are cut-vs-clean).

**SAE**: TopK, 64 latents, k=4 (4 latents active per position), trained 2000
steps on the resid activations. Two instances:

- **@ blocks.1.hook_resid_post** (the "post-aggregation" interface feeding
  block-2 attention) — FVU 0.123. Top ω-latents: #62 tracks ω̂₂ (R²=0.60),
  #44 tracks ω̂₀ (0.44), #51 ω̂₂ (0.48), #45 ω̂₀ (0.34). Note there is no
  strong ω̂₁ latent — the code is rank-2 (the simplex has 2 dof).
- **@ blocks.0.hook_resid_post** (the "mid-aggregation" interface feeding
  block-1 attention) — FVU 0.110, top ω-latents even stronger (#30: 0.74).

The **cut set S** used everywhere below is `omega_top4`: the top-4 latents
per component ranked by R² against posterior ω, deduplicated (5 latents at
L1: {44,45,54,62,51}; 7 at L0: {30,32,1,8,59,15,12}). Selection uses TRAIN
data only; all reported metrics are on held-out eval sequences.

---

## 3. The interventions (exact operations)

All interventions are forward hooks; the exact decomposition below is
verified against the model's own attention scores to ≤ 3e-5.

With a TopK(k=4) SAE, the resid at each position decomposes exactly:

    x_p = Σ_{i ∈ A_p} z_i(p) · d_i  +  b_dec  +  err_p        (A_p = 4 active latents)

**(a) SAE cut** (content severing) — at the SAE hookpoint:

    x_p ← x_p − α · Σ_{i ∈ S ∩ A_p} z_i(p) · d_i

α=1 deletes S's content wherever it fires; the error term and all other
latents are untouched. No gain to calibrate: α=1 IS "remove it."

**(b) FRA-QK cut** (routing severing) — attention scores of the downstream
layer decompose exactly through frozen LN into term-pair contributions
(query-side term × key-side term, each side ∈ {4 latents, b_dec, err,
const}). We subtract the summed contribution of every pair involving S
(key-side / query-side / either) from `hook_attn_scores`, times gain c.
c=1 removes exactly S's influence on WHO ATTENDS TO WHOM.

**(c) FRA-OV cut** (path-restricted content transport) — the value each key
position contributes through the attention layer decomposes the same way; we
subtract S's transported content from `hook_attn_out`:

    Δout_q = Σ_h Σ_k pattern[h,q,k] · φ_k(Σ_{j∈S} z_j d_j) W_V[h] W_O[h]
    attn_out_q ← attn_out_q − c · Δout_q

**c=1 severs exactly this path's share of S-content. c>1 is no longer
severing** — it injects −(c−1)× the contribution as a counter-signal. Keep
this distinction in mind; it becomes the whole story.

Placement: at the L1 interface the only downstream attention is block 2
("post-aggregation" cuts); at the L0 interface the cuts act on block 1
("mid-aggregation" cuts).

Controls: random latent sets of matched size (all kinds), and a DoM-style
rank-3 projection of the best ridge-probe directions for ω.

---

## 4. The metrics (exact definitions)

For each intervention, run the eval set and compute from the factorized CE:

- **Removal fraction** RF = (blockCE_cut − blockCE_clean) / (1.0809 − 0.9872).
  0 = concept use untouched; 1 = degraded exactly to the no-context prior;
  >1 = pushed PAST the prior (anti-concept).
- **Tracking R²** — R² between the model's block-mass m(t) and ω̂(t) across
  all eval points. Clean: 0.992. A perfect null makes this ≈ 0; negative
  values mean anti-tracking.
- **Collateral fraction** CF = (withinCE_cut − withinCE_clean) / (1.0986 − 0.7785).
  0 = grammar untouched; 1 = degraded to uniform.
- **Presence (probe R²)** — a ridge probe RETRAINED from scratch on the
  intervened last-layer resid (80/20 split by sequence) predicting ω̂. This
  is the adversarial "is the concept still decodable" metric — it cannot be
  fooled by moving the representation around; it refits. A 2-layer MLP probe
  double-checks the nonlinear case.

---

## 5. The worked example (intended vs post-intervention outputs)

Sequence #270, true ω = (0.14, 0.83, 0.03). Each cell is the model's
block-mass m(t) = (P(next token ∈ block 0) / block 1 / block 2), i.e. its
*operational* concept estimate:

```
          position t              16              64             128             240
  posterior ω̂(t) (Bayes)  0.22/0.61/0.17  0.16/0.78/0.06  0.14/0.83/0.03  0.14/0.82/0.03
  prior (removal target)  0.40/0.35/0.25  0.40/0.35/0.25  0.40/0.35/0.25  0.40/0.35/0.25

  clean                   0.21/0.62/0.17  0.16/0.77/0.07  0.13/0.83/0.05  0.17/0.78/0.05
  SAE cut @L1 (α=1)       0.43/0.40/0.16  0.35/0.42/0.23  0.30/0.47/0.23  0.31/0.39/0.30
  FRA-QK @L1 (either,c=2) 0.20/0.63/0.17  0.15/0.78/0.07  0.12/0.84/0.04  0.19/0.75/0.06
  FRA-OV @L0 (c=1, sever) 0.22/0.61/0.17  0.18/0.75/0.07  0.19/0.75/0.06  0.22/0.71/0.07
  FRA-OV @L0 (c=4, null)  0.27/0.58/0.15  0.36/0.49/0.15  0.42/0.34/0.23  0.43/0.34/0.23
  FRA-OV @L0 (c=6, over)  0.31/0.54/0.15  0.48/0.28/0.24  0.47/0.18/0.35  0.51/0.14/0.35
```

Read the rows:

- **clean**: digit-for-digit Bayes tracking. The model IS a counter.
- **SAE cut**: collapses m toward the prior everywhere (RF 0.86 overall);
  the residual comp-1 lean is the 14% it can't remove (info re-aggregated
  from token identities the cut must spare).
- **FRA-QK**: indistinguishable from clean. Cutting the concept's influence
  on the attention *pattern* does nothing, because counting doesn't need a
  concept-gated pattern — ANY roughly content-uniform pattern transports
  block-tagged content in proportion to counts. (This despite 57–75% of the
  pattern's FRA score mass being feature×feature — content-*gated* pattern
  ≠ concept-*causal* pattern. A CCF-style spectral screen would be fooled.)
- **FRA-OV c=1** (true severing): barely moves m. The block-1 OV path
  carries only ~¼ of the ω-signal reaching the output; the rest flows via
  the direct stream and block-2's re-aggregation.
- **FRA-OV c=4** (gain-tuned null): m ≈ prior at mid/late positions —
  behavioral concept use is GONE (tracking R² 0.001, RF 0.993) at CF 0.036.
  But look at t=16: 0.27/0.58/0.15 — early positions under-cancel. The
  aggregate null is a balance of position-wise errors (early +0.11 residual
  tracking, late −0.09 anti-tracking).
- **FRA-OV c=6**: comp-1 — the sequence's TRUE dominant component — is now
  SUPPRESSED below prior (0.14 at t=240 vs prior 0.35). The channel has
  been over-driven into an anti-concept steering vector.

**And the punchline measurement**: at the c=4 null point, the retrained
probe still reads ω̂ from the last-layer resid at **R² = 0.985 (linear) /
0.995 (MLP)** vs clean 0.989/0.996. The concept is not damaged AT ALL. The
model computes it, represents it, and simply doesn't act on it.

---

## 6. Headline results (all interventions, main run)

| intervention (best per family) | RF (use removed) | CF (collateral) | presence |
|---|---:|---:|---:|
| SAE cut, ω-latents, α=1, @L1 | 0.86 | 0.017 | 0.49 |
| SAE cut, block-latents, α=1, @L1 | 0.97 | 0.093 | 0.42 |
| probe-direction projection (rank 3) | 0.075 | 0.002 | 0.98 |
| FRA-QK (every set/side/strength/layer) | ≤ 0.10 | up to 0.28 | 0.99 |
| FRA-OV c=1 (exact severing, best) | 0.06 | 0.002 | 0.99 |
| **FRA-OV c≈4 (gain-tuned null, @L0)** | **0.99** | **0.036** | **0.985** |
| random-latent controls | 0.00 | 0.00 | 0.99 |

Figure: `out/entanglement_tax.png` (panel A: use vs presence; panel B:
presence vs absolute within-CE with the exact impossibility bounds).

Three regimes, three tools:

1. **Content severing (SAE)** — the only thing that reduces *presence* at
   all (0.99 → 0.42), and the cheapest partial use-removal. Cannot reach
   full removal: the concept is re-estimable from token identities that
   must survive for grammar's sake.
2. **Routing severing (FRA-QK)** — never works here, at any layer/strength.
   In the aggregation regime the concept is carried by WHAT is summed, not
   WHO attends to whom. (Softmax gauge invariance kills query-side cuts;
   sum-robustness kills key-side cuts.)
3. **Path-addressed counter-steering (FRA-OV, gain-tuned)** — the best full
   nuller in the study, and the only presence-preserving one.

---

## 7. The entanglement tax (why full removal is impossible anyway)

The concept here is an **aggregate statistic of the content**: ω̂ is a count
of block tags, and the grammar (belief updates) NEEDS those same tags to
route symbol history to the right component. You cannot delete the concept
from the representation without blinding the grammar. Quantitatively
(`bayes_floor.py`, Rao-Blackwellized particle filter over hidden
assignments, converged across 256/512/1024 particles):

> An observer whose representation carries ZERO ω-information (history
> collapsed to within-block symbols) achieves within-CE **0.845** at best,
> vs 0.7785 with full information. **Irreducible collateral at full
> presence-removal = 20.8%** of the grammar range — for ANY method.

Scope honesty: this bound binds the *ideal frontier* (panel B's forbidden
notch). Our toy model runs 0.20 nats above the Bayes grammar ceiling, so its
observed collateral is mechanistic damage, not the tax; and its cuts never
get close to zero presence anyway (floor ≈ 0.42). The tax says even a
perfect model + perfect scrub cannot beat 0.845 — the better the model, the
more binding the wall.

---

## 8. Robustness battery (is the null real or fitted?)

Six checks (`ov_sweep_conc*.json`, `robust_checks.json`, `seed43/`):

| axis | result |
|---|---|
| existence across seeds | ✅ null exists at seed 43 too (c*≈4.8 vs 4.0; CF at null ~0.12 vs 0.036 — channel quality varies) |
| channel width | ✅ top-1/2/4 latent channels all null; gain rescales with carried signal |
| presence preservation | ✅ universal: probe 0.98–0.99 in every configuration; MLP 0.995 |
| distribution shift | ⚠ fixed gain drifts: tracking at c=4 = +0.02 (conc 30), 0.00 (conc 10), +0.07 (conc 2), **−0.59 (conc 0.5)** — near-pure sequences get pushed ANTI-concept |
| position uniformity | ⚠ aggregate null = early under-cancel (+0.11) balancing late over-cancel (−0.09) |
| gain sensitivity | ⚠ dRF/dc = 0.62 at the null → ±10% gain error = ±0.25 RF |

**Verdict: a calibrated equilibrium, not a structural removal.** The
counter-injection is a fitted linear cancellation of a mildly
signal-strength-dependent flow: exact where calibrated, systematically
miscalibrated wherever the concept's signal strength changes — and the miss
grows precisely on the most concept-loaded inputs.

---

## 9. What this means for the in-context program

1. **"Remove the concept" is ill-posed until you name the target.**
   *Use-removal* is cheap and achievable (SAE severing for partial+robust,
   gain-tuned FRA-OV for full+fragile). *Presence-removal* is a different,
   information-theoretically taxed task that none of our tools achieve and
   most of our evals don't measure.
2. **Behavioral evals cannot certify concept removal.** The c=4 model passes
   any on-distribution behavioral test of "does it use ω" while carrying ω
   at R² 0.995. Off-distribution behavior (near-pure streams) and *presence
   probes* are the audits that catch it. This is the operative safety
   lesson: **audit presence, not behavior.**
3. **The FRA boundary, restated causally**: FRA wins iff the concept's
   causal carrier at the intervention site is the *pattern* (fra_win's
   induction edge). Once the carrier is marginal *content* — every
   aggregation-type broad concept, incl. fra_pii's redundant PII bank — QK
   has nothing to grab, OV-severing under-reaches (redundancy), and only
   content cuts or gain-calibrated counter-steering bite. Spectral
   diagnostics (CCF/mass-based) overpredict FRA's reach; only causal probes
   (does re-gating change behavior?) find the boundary.
4. **Sweep steering gains past c=1 before declaring a null result** — the
   winning operating point sat just outside our pre-registered grid.

---

## 10. Reproduction

```bash
cd experiments/fra_hmm_toy
../../.venv/bin/python run_toy.py --out out/main2                    # main grid (~20 min CPU)
../../.venv/bin/python run_toy.py --out out/phase2_L0 --sae-layer 0  # mid-aggregation grid
../../.venv/bin/python bayes_floor.py                                # entanglement tax
../../.venv/bin/python ov_sweep.py [--conc C] [--dir D] [--seed S]   # gain sweeps
../../.venv/bin/python robust_checks.py                              # position/width/MLP
../../.venv/bin/python exhibit_example.py                            # §5 table
../../.venv/bin/python plot_toy.py out/main2 && ../../.venv/bin/python plot_tax.py
```

Checkpoints: `out/{main2,phase2_L0,seed43}/{transformer,sae}.pt` (~700KB
each). All raw numbers: `results.json` / `*_sweep*.json` / `robust_checks.json`.

### Notation

| symbol | meaning |
|---|---|
| ω, ω̂(t) | per-sequence mixing vector; its Bayes posterior mean after t+1 tokens |
| block / m(t) | vocab block of a token (= component id); model's next-token block-mass |
| RF, CF | removal fraction (use), collateral fraction (grammar), Bayes-normalized |
| presence | held-out R² of a probe RETRAINED on intervened activations |
| S | cut set of SAE latents (top-m per component by R² vs ω̂) |
| c, α | FRA gain (c=1 exact severing, c>1 counter-injection); SAE ablation strength |
| c* | gain at which tracking R² crosses zero (the null) |
