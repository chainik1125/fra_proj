# Optimal FRA: what feature-resolved attention measures when the features and the attention are both known

*Reconciliation note for the constrained-belief-update FRA program, merging Setting A
(static Bernoulli, `A_static.md`/`VERIFY_A.md`), Setting B (the reset process,
`B_reset.md`/`VERIFY_B.md`/`REVIEW_B_by_A.md`), the trained-SAE recovery sweep
(`VERIFY_SAE.md`), the MLP-removed control (`../attn_only_redo/ATTN_ONLY.md`), sprint-2
(`fra_cbu_note.tex`), and the campaign's model-organism results (fra_win, fra_pii,
fra_hmm_toy, fra_organisms). P1/P2/P3 familiarity assumed.
Every load-bearing claim is tagged **[CONFIRMED]** (measured on trained models or to
float precision), **[theory]** (derived in closed form, not yet measured), or
**[future]** (open). Numbers trace to `VERIFY_{A,B,SAE}.md` and `out/*.json`.*

---

## Executive summary

**Feature-resolved attention is a faithful handle in a small, characterizable region of
process space, and this note draws the map, states the closed forms, and marks the three
ways it fails.** Five findings carry the note:

1. **FRA's faithfulness is governed by two dials, and three of its four corners collapse
   it to gauge — each by a different mechanism.** Process persistence $\lambda$ and
   observation noise $\rho_{\mathrm{obs}}$ place any one-layer model on a map; FRA-OV is
   the real handle only in the noisy-persistent interior. Crucially, the two
   "attention-null" corners are *opposite* mechanisms — static/iid ($\lambda=0$) has no
   cross-time signal and a null direct path, while clean-emission
   ($\rho_{\mathrm{obs}}\to0,\ \lambda>0$) has full signal carried by a *live,
   loss-pinned* direct path, so its look-back attention is gauge because the signal is
   *redundant, not absent*. **[CONFIRMED]** (§2)

2. **The optimal look-back rate is a warped geometric $\eta(\rho_{\mathrm{obs}})<\lambda$,
   confirmed on trained models.** The trained rate rises $0.24\to0.49$ with observation
   noise, tracks the palindromic-root warp to ~5% wherever the kernel is pinned, and never
   approaches the belief rate $\lambda=0.7$; the warp equals the steady-state Kalman pole
   to $10^{-16}$ (Fig. `out/warp_curve.png`). **[CONFIRMED]** (§5i)

3. **Loss-flat is not seed-noise: unfaithful FRA attributions are reproducibly
   optimizer-pinned.** On the static bench the attention circuit is exactly loss-flat, yet
   cross-seed FRA correlation is $0.99$, not $0$. Reproducibility across seeds is therefore
   not evidence of computation; the right diagnostic is clean-vs-trained-at-equal-loss (a
   hand-built zero-attribution representative achieves the identical loss). **[CONFIRMED]** (§3)

4. **FRA-edit safety is a dictionary-quality property MCC cannot certify.** Severing
   feature $i$ is a realizable latent-set edit iff $e_i\in\mathrm{rowspace}(\chi)$; a
   trained SAE loses this for *every* channel the instant $N>d$ (while MCC stays
   $0.95$–$1.0$, blind), and collateral suppression drops a full order in superposition —
   matched code slope $1.91$ ($\propto\rho_{mm}^2$) vs learned code $0.71$
   ($\propto\rho_{mm}$). **[CONFIRMED]** (§4)

5. **FRA cannot delete a feature's presence below the single-observation floor.** An FRA-OV
   edit acts only on the attention-transported carryover, so the current token's
   contribution to the belief is outside its reach; the floor is measured to equal its
   analytic value $\mathrm{Corr}^2(\text{belief},\,\text{current obs})=0.648$ to three
   decimals, counter-steering leaves presence unchanged, and the FRA-cuttable share
   $1-\text{floor}$ grows with observation noise. **[CONFIRMED at one
   $\rho_{\mathrm{obs}}$; the monotone trend is its corollary]** (§5)

6. **Hierarchy hands FRA a loss-pinned content junction — but the model puts it in OV, not
   QK, and FRA's OV attribution then beats the steering baselines for control.** A content
   gate (a child fires only if its parent fired) makes the optimal computation
   content-dependent, worth a predicted gating value $\Delta_{\mathrm{gate}}\approx0.014$
   (H2.2's Jensen-gap surrogate reads $0.0189$, $\sim\!1.35\times$). §6 predicted this hands
   FRA-QK its first handle; on the theory-clean (no-LN) platform the gate is real (full-QK
   reaches Bayes) but the trained model carries it in **OV, not QK** — a $21$–$36\times$
   OV-vs-QK cut asymmetry across 3 seeds, gauge-flat on every one. So the *amended fra_win
   law*: a two-position product is necessary but **not sufficient** for a QK handle — it must
   be an *obligate query-key match* (induction is the campaign's only one), while a content
   *gate* is OV-realisable. Turning that OV localization into a control edit **beats
   difference-of-means ($17$–$51\times$) and planted-SAE ($48$–$259\times$) ablation on
   parent-collateral** at equal gate-removal (mean-ablation; ordering robust across 3 seeds × 2
   conventions × 2 bias regimes, worst-case floor $\ge2\times$). **[CONFIRMED — 3 seeds]** (§8)

---

## 1. The question: optimal FRA

**Feature-resolved attention (FRA) is the decomposition of a one-layer attention
head's two internal objects into a chosen feature basis of the residual stream; this
note asks what those decompositions measure when both the feature basis and the
loss-optimal attention are known in closed form, rather than read off a trained
checkpoint.** For a destination residual that receives a direct contribution
$\Wu\We\,a_d$ and an attention-mediated contribution $\sum_{s\le d}A_{ds}\Wu\Wov\We\,a_s$
(P3's one-layer model, Eq. transformer), FRA resolves:

- **FRA-QK** — the pre-softmax score $\sigma(d,s)=q_d\!\cdot\!k_s$ into feature×feature
  terms $Q_{ij}=d_i^\top\mathbf W_{QK}\,d_j$: *which query feature reads which key
  feature*, plus query/key pedestals and a constant.
- **FRA-OV** — the transported content $A_{ds}\,\Wu\Wov\We\,a_s$ by source feature and
  lag: *which feature, copied from how many tokens back, is written into the
  destination*, factoring as (scalar lag profile) × (content direction $o_j=\Wov d_j$).

An FRA *attribution* reads a coefficient off one of these; an FRA *edit* removes a named
feature term at gain $c$ and asks what the model then does. The program's governing
question — established on trained Mess3 models by sprint-2 and sharpened here — is
whether an attribution or edit measures the network's **computation** or merely a
**gauge**: a coordinate a loss-preserving reparameterization can move at will.

Every prior measurement of FRA has been taken on a *trained* model, where computation
and convention are entangled and only a weight dial can separate them. **Optimal FRA
removes that entanglement.** Fix a known ground-truth dictionary $\mathbf D$ (rows
$d_i$, unit-norm, superposition $\rho_{mm}$) and a data process whose optimal one-layer
attention can be written down exactly; then every attribution and every edit has a
closed form, and "computation vs gauge" becomes a theorem, not a measurement. Two data
processes make this possible: the **static Bernoulli** bench (SynthSAEBench §3;
features fire i.i.d. across time) and the **reset process** (each feature's indicator
is an independent 2-state Markov chain with persistence $\lambda_i$, observed through a
rectified-Gaussian magnitude). Together they sweep the two axes that turn out to
organize the entire program.

---

## 2. The map: two dials, four landmarks

**The whole program lives on two dials — process persistence $\lambda$ (how long a
feature's state carries) and observation noise $\rho_{\mathrm{obs}}=\nu/A$ (how much
the current token hides that state) — and FRA-OV is the real handle only in the
interior; three of the four corners collapse it to gauge, each by a different
mechanism.** The single most important structural correction this note makes is that
the two "attention-is-null" corners are *not the same phenomenon*.

> **Notation guard (resolves the `ρ` collision flagged in REVIEW_B_by_A F9).** In this
> note $\rho_{\mathrm{obs}}=\nu_i/A_i$ is the observation noise-to-signal ratio (the
> map's vertical dial); $\rho_{\mathrm{path}}$ is the attention/skip path share (the
> sprint-2 $c^\ast=1/\rho_{\mathrm{path}}$ tracking law); $\rho_{mm}$ is dictionary
> superposition. They are three different quantities and are never abbreviated to bare
> $\rho$.

```
                 observation noise ρ_obs
     ρ_obs = 0  (state revealed)        ρ_obs → ∞  (state hidden)
   ┌───────────────────────────────┬───────────────────────────────┐
λ=0│  A. STATIC / iid               │  (still iid: no cross-time     │
   │  R = 0 identically.            │   signal regardless of ρ_obs)  │
   │  Whole attention circuit gauge,│  → collapses onto corner A     │
   │  direct path ALSO null         │                               │
   │  (predicts the mean μ_a).      │                               │
   ├───────────────────────────────┼───────────────────────────────┤
λ>0│  B-clean. LOOK-BACK gauge,      │  C. INTERIOR / warped kernels │
   │  DIRECT path LOSS-PINNED at     │  η_i(ρ_obs) < λ_i, FRA-OV =   │
   │  gain λ (redundant signal,      │  the real handle, closed forms│
   │  NOT corner A).                 │  (Mess3, reset finite-noise)  │
   ├───────────────────────────────┴───────────────────────────────┤
λ→1│  D. AGGREGATION / counting.  Pattern maximally pinned yet       │
   │  carries no concept; FRA-QK gauge despite high feat×feat mass;  │
   │  OV aggregation shares set c*   (fra_hmm_toy, sprint-2 mixture). │
   └────────────────────────────────────────────────────────────────┘
```

**Corner A — $\lambda=0$ (static/iid): everything null, everything gauge. [CONFIRMED]**
With features i.i.d. across time, every cross-time second moment collapses to the rank-one
mean, $\mathbb E[a_t a_{t-\tau}^\top]=\mu_a\mu_a^\top$ for $\tau\ge1$, so P3's
attention-useful residual $\mathbf R$ (which lives in the nonstationary token subspace)
is identically zero. In the affine model the unique optimum is $\beta=\mu_a,\
\mathbf W_\ast=\mathbf V_\ast=0$: the predictor is the constant mean and **the entire
attention circuit — QK and OV, every head — is loss-flat, a single gauge orbit**
(A1, verified: a clean $\mathbf W=\mathbf V=0$ representative and a trained model both
sit at the floor, attributions $0$ vs large). The constant $\mu_a$ is carryable by the
bias, the direct path, or attention-to-anything — the k=0/skip redundancy in its
purest form; without a bias, attention survives only as a rank-one, content-free
mean-substitute. The interesting statics is *within-position*: the optimal read-out of
$c$ from $a$ is the LMMSE filter $\mathbf T=\Sigma_c\mathbf D\,\mathbf C_a^+\mathbf
D^\top$ — exact recovery for $N\le d$, and shrinkage $\overline{\mathbf T_{ii}}=d/N$
with matched-filter contamination $\propto d_i^\top d_j$ for $N>d$ (verified to float
precision).

**Corner B-clean — $\rho_{\mathrm{obs}}\to0,\ \lambda>0$: look-back gauge with a
live, loss-pinned direct path. [CONFIRMED]** This corner *looks* like corner A (the look-back
attention profile collapses, $\eta_i\to0$) but is its mechanistic opposite, and
conflating them is the error `REVIEW_B_by_A` F4 corrects. When the emission is clean,
the persistent state is fully revealed by the current token (Markov sufficiency), so
the optimal one-step predictor is $\hat y(t+1)=\lambda\,y(t)$: **the look-back
attention is null because the signal is *redundant*, not because it is *absent*, and
the direct/skip path is maximally load-bearing (gain $\lambda$) and genuinely
loss-pinned — not gauge.** Corner A has no cross-time signal and a null direct path
(predicts the mean); corner B-clean has full cross-time signal carried entirely by the
direct path. The steady-state Kalman filter makes the contrast one number: corner A has
current-token gain $0$; corner B-clean has current-token gain $\lambda K\to\lambda$
with $K\to1$. **The genuine reduction of the reset process to the static setting is the
$\lambda\to0$ limit, not the $\rho_{\mathrm{obs}}\to0$ limit.** Consequently the
no-bias and mean-sector caveats transfer only partially: the k=0 mean gauge carries
over, but corner B-clean's nonzero fluctuation prediction $\lambda(z-p)$ has no
static-setting analog, so stripping the bias there loads the *direct* path, not a
mean-substitute attention.

**Corner C — finite $(\lambda,\rho_{\mathrm{obs}})$: the interior, where FRA-OV is the
handle. [CONFIRMED]** With a persistent state observed through noise, the per-feature
read-out is an AR(1)-plus-white-noise process, and the optimal look-back attention is a
warped geometric $\eta_i(\rho_{\mathrm{obs}})<\lambda_i$ — the palindromic-root warp of §5,
independently confirmed against the steady-state Kalman rate to $10^{-16}$ and on trained
models to ~5%. Here FRA-OV carries the constrained-belief displacement, the attributions
have closed forms, and edits obey the path calculus (sprint-2's trained-Mess3 result, now
derived rather than measured). Mess3 sits in this interior at a $\rho_{\mathrm{obs}}$
bounded away from $0$ (a token never reveals the 2-simplex belief), which is why its
trained rate $\eta=0.464<\zeta=0.55$ is a mild — not total — shrinkage. **How much of a
feature FRA-OV can reach grows with the noise:** the current token contributes a fixed
floor to the belief that no OV edit touches (§5), and that floor is
$\mathrm{Corr}^2(\text{belief},\text{current obs})$, which *falls* as $\rho_{\mathrm{obs}}$
rises — so the FRA-cuttable presence share $1-\text{floor}$ is smallest in the clean
corner and largest deep in the interior, moving in lockstep with the warp $\eta$. **And how
*completely* an OV cut nulls that reachable belief — the intervention null
$c^\ast=1/\rho_{\mathrm{path}}$ — is an architecture property, not an FRA one:** when
attention is the sole belief-carrier (attention-only 1L) $\rho_{\mathrm{path}}\approx1$ and
exact severing suffices ($c^\ast\approx1$), whereas each additional parallel carrier (a live
skip, the MLP, more layers) lowers $\rho_{\mathrm{path}}$ and pushes $c^\ast$ up (§7).

**Corner D — $\lambda\to1$: aggregation/counting. [CONFIRMED: fra_hmm_toy / sprint-2;
formal law future].** As the state freezes, the per-lag
value ladder flattens and the whole profile becomes loss-pinned toward a counting
kernel, yet the pinned pattern carries no *concept* — FRA-QK stays gauge for content
despite large feature×feature pattern mass, and the FRA-OV aggregation shares set the
intervention null $c^\ast$ (sprint-2's mixture outlook; `fra_hmm_toy`, where FRA-QK/OV
cuts drive removal-fraction $\le0.05$ despite $0.6$–$0.75$ feature×feature pattern mass,
and only a 5-latent SAE achieves a real cut).

### Placing the campaign's empirical results on the map

| Result | Location on the map | What it shows |
|---|---|---|
| **sprint-2 Mess3** (single process) | Corner C, partial observation | FRA-QK gauge (140× dial at bit-identical loss); FRA-OV the handle ($\rho_{\mathrm{eff}}$ predicted ex ante to 4 s.f.); dictionary the binding constraint (missing channel) |
| **fra_hmm_toy** (mixture-HMM) | Corner D | aggregation regime: pattern mass without concept; SAE-basis cut needed |
| **fra_pii** (in-context SSN recall) | Corner C + dictionary limit | FRA content-selective but **reach-insufficient**: rank-1 OV never disarms; only a position-locked SAE clears the bar — a live instance of the χ completeness/centering constraint (§4) |
| **fra_win** (induction control) | the **content-gated** edge (§6) | the one regime where FRA-QK has a real, loss-pinned handle: association-specific attention control at ~15× less collateral than ActAdd — the reset process's hierarchy corner (B4b) made real |
| **fra_organisms** (boundary screen) | the map's outer boundary | real models occupy the FRA-OV-handle interior only under a narrow "load-bearing attention at the answer step" kernel; recall/injection/EM/sycophancy fall outside it |

**The gating axis (§8).** Hierarchy adds a *third structural axis* to this two-dial map:
content-gating switches on a **loss-pinned content junction** inside the noisy-persistent
interior (corner C ∩ gating) — the first process whose optimum couples content at two
positions. But the junction's *location* — QK, OV, or a normalization nonlinearity — is
**architecture- and training-chosen, not process-determined**: on a theory-clean
attention-only model the trained gate lands in **OV** (§8). So the map marks only *that a
content junction exists in corner C ∩ gating*; §8 governs *where the model puts it*.

---

## 3. The gauge theory, consolidated

**FRA's redundancies form three tiers of increasing subtlety — exact function-preserving
gauge, loss-orbit gauge, and functional (path) redundancy — and the load-bearing, and
most easily mishandled, fact is that landing on a specific representative reproducibly
does not make it computation.**

**Tier 1 — exact function-preserving gauge (bit-identical forward pass).** Softmax row
constants (any score component depending on the key only through a constant is annihilated
by the softmax) and the bias/$b_{\mathrm{dec}}$ re-split (sprint-2's G1 dial: move a fixed
vector between the bias, the dictionary contribution, and positional embeddings). These
leave every activation and logit unchanged; the query pedestal and constant are
*unobservable*, the key pedestal is *dialable*.

**Tier 2 — loss-orbit gauge (loss-preserving, forward pass changes).** Head splits (only
the head-sum $\sum_h A^{(h)}_{ds}o^{(h)}_j$ is pinned; per-head attributions are a gauge
family, and under softmax that family is not even freely realizable — sprint-2's two-head
hinge) and the k=0 skip/diagonal split (the current-token displacement is shared between
the residual skip and diagonal attention; only the sum is pinned, and the share $a_0(d)$
sets the intervention reach $\rho_{\mathrm{path}}$ and hence $c^\ast=1/\rho_{\mathrm{path}}$).

**Tier 3 — functional redundancy (path multiplicity).** The same content can be delivered
by more than one path — the constant $\mu_a$ by bias/direct/attention (corner A); lag-0
content by skip vs diagonal (corner C); the belief carryover by direct-Markov vs look-back
(corner B). Interventions on one path are undone by the redundant path unless the whole
functional bundle is cut.

**The invariants** — what survives all three tiers — are the objects any faithful FRA
report must be built from: the **centered score coupling** $\hat q_E(z)\!\cdot\!\hat
k_E(z')$ (content contrasts, not pedestals); the **head-summed, subspace-aggregated OV**
attribution; and the **path-summed flow** into a feature channel.

> **The optimizer-pinned correction (a first-class warning, from VERIFY_A). [CONFIRMED]**
> *Loss-flat does not mean random, and reproducible does not mean computational.* On the
> static bench, the attention circuit is exactly loss-flat, yet trained models do **not**
> scatter: cross-seed FRA correlation is $0.99$ (not $\approx0$), because SGD reproducibly
> selects one large-weight representative — distributing the constant across paths
> ($\beta+(\mathbf W+\mathbf V)\mu_a=\mu_a$) with $\mathbf V$ cancelling $\mathbf W$'s
> fluctuation. **Reproducibility across seeds is therefore not evidence of computation**;
> the earlier "seed-noise" heuristic is retracted. The correct diagnostic is the
> **clean-vs-trained-at-equal-loss** comparison (a hand-built representative with zero
> attributions achieves the identical loss), never cross-seed variance. This warning
> propagates to Setting B, where it is sharper still: corner B-clean's direct path is
> genuinely loss-pinned while its look-back attention is loss-flat, so a seed-coherence
> test would read *both* as coherent for *opposite* reasons — any B-verification must
> separate the loss-pinned direct path from the gauge look-back attention, not run an
> incoherence test.

---

## 4. Dilution vs gauge, and the dictionary that names the channels

**Refining the dictionary and choosing a gauge are orthogonal operations, and the
faithfulness of an FRA edit is set by a rank condition on the learned code that MCC
cannot certify.**

**Dilution ⟂ gauge (A3). [CONFIRMED to float precision.]** Splitting one ground-truth
feature across $m$ latents shrinks per-pair QK attributions as $1/m^2$ (the score is
bilinear) and per-latent OV as $1/m$ (linear), but a full-group cut restores the entire
effect independently of $m$ — dilution is a change of *description*, and grouping always
recovers the data-pinned invariant. Gauge is a change of *convention*: a full aggregate
object (e.g. sprint-2's full-token-set key cut) still swings $140\times$ under a
loss-preserving dial, because that aggregate is a pedestal, not an invariant. The clean
2×2: *grouping fixes dilution; nothing fixes gauge*, and the two properties are
independent — a coupling can be un-diluted-recoverable and gauge at once (as the whole
attention circuit is in corner A).

**The χ formalism and the severability criterion. [CONFIRMED]** A learned latent decodes
as a mixture of ground-truth channels, $w_\ell=\sum_i\chi_{\ell i}d_i$. The bench recovery
metrics are properties of $\chi$: MCC is its diagonal strength after Hungarian matching,
uniqueness is the absence of shared columns, hedging is off-diagonal mass $\propto$
feature correlation, absorption is a parent entry inside a child's row. **An FRA edit is an
edit of χ-mixed channels, and "sever feature $i$" is realizable as a latent-set operation
iff $e_i\in\mathrm{rowspace}(\chi)$** (a left-inverse row exists; sufficient: a complete
code, $\mathrm{rank}(\chi)=N$). **MCC cannot certify this**: a code can have MCC $\approx1$
yet miss a channel entirely (division by $\min(L,N)$ hides it) or align in direction while
mis-centered (a contrast/absence code). Verified decisively: a complete code (MCC $1.0$)
severs all features; a Finding-4-style absence code (MCC $0.995$) severs *none* — the
feature exists only as a contrast, exactly `fra_pii`'s reach-insufficiency and sprint-2's
missing-channel result. Completeness and centering, not purity or MCC, are what an FRA edit
inherits.

**Trained-SAE recovery, measured (sae-recovery agent; `VERIFY_SAE.md`, `code/sae_exp{1,2}_*.py`,
`out/sae_exp*.json`).** TopK SAEs trained on the classic Bernoulli–Gaussian bench, χ = W_dec·pinv(D),
severability residual $\|(I-P_{\mathrm{rowspace}(\chi)})e_i\|$ per feature. The two recovery axes
**dissociate**: MCC (direction) is easy; rowspace-completeness (the property FRA edits inherit) is not.

| regime | $\rho_{mm}$ | MCC (L=N→2N) | χ-rank / N | # unseverable | sev$_{\max}$ |
|---|---|---|---|---|---|
| $N{=}16,d{=}32$ (orth) | $0$ | $0.94\!\to\!0.99$ | $16/16$ | $\mathbf 0$ | $3\!\times\!10^{-15}$ |
| $N{=}24,d{=}24$ (orth) | $0$ | $0.97\!\to\!0.99$ | $24/24$ | $\mathbf 0$ | $7\!\times\!10^{-15}$ |
| $N{=}32,d{=}24$ | $0.47$ | $0.96\!\to\!1.00$ | $\mathbf{24/32}$ | $\mathbf{32}$ | $0.68$ |
| $N{=}48,d{=}24$ | $0.49$ | $0.95\!\to\!1.00$ | $\mathbf{24/48}$ | $\mathbf{48}$ | $0.78$ |

**First-channel-lost threshold: $N/d=1$.** For $N\le d$ the trained χ is full rank $N$ and severability
is $0$ to float precision — *every* GT channel is a realizable latent-set edit. The instant $N>d$,
$\mathrm{rank}(\chi)$ saturates at $d$ (an information wall, not a training defect: A1.4's invisible
null space), so **all $N$ channels leave $\mathrm{rowspace}(\chi)$ at once** and none is severable —
while **MCC is blind, even rising toward $1.0$ with $L$**. Severability is the earlier, and only honest,
warning; MCC is anti-diagnostic. Correlation and hierarchy (kept at $N{=}d$, orthogonal) corrupt the
code on a *separate* axis — **per-latent contamination**, not incompleteness: hedging puts the top
off-diagonal χ entry on the correlated partner $62\%$ of the time (mean off-diag mass $0.18$), and
absorption makes each child latent $\approx 0.82\,d_{\text{child}}+0.545\,d_{\text{parent}}$ — yet
rowspace-severability stays $0$ (a latent *combination* still isolates the channel). MCC falls only
$0.97\!\to\!0.92$–$0.94$. So the literature's "severability fails for children first" is **refined**:
at a complete code absorption is single-latent contamination, not a lost channel; incompleteness ($N>d$)
is what actually removes the channel, and MCC certifies neither.

**Collateral-slope test, measured (sae-recovery; `code/sae_exp4_slope.py`, `out/sae_exp4_slope.json`).**
At $\rho_{mm}=0.43$ (random non-orthogonal $D$, $N{=}d{=}24$), severing feature $i$ and isolating the
$c_j$-proportional part of feature $j$'s readout perturbation ($\beta_{ij}=\mathrm{Cov}(\Delta\hat c_j,c_j)/
\mathrm{Var}(c_j)$), fit $\log|\beta_{ij}|$ vs $\log|d_i^\top d_j|$ over all off-diagonal pairs:

$$\text{matched code }(\chi=I):\ \text{slope}=\mathbf{1.91}\ (\log\text{-}\log\ r=0.96),\qquad
\text{learned SAE }(\chi\ne I):\ \text{slope}=\mathbf{0.71}\ (r=0.55).$$

**The §5(iii) prediction is confirmed to within measurement noise: slope $\approx 2$ vs $\approx 1$.**
The matched code's collateral rides on the *product* of two overlap legs ($c_j$ leaks into $i$'s encode
via $d_i^\top d_j$; the removed $d_i$ is read back by $d_j$ via $d_i^\top d_j$) and the first-order term
cancels, leaving the clean quadratic with a near-perfect fit. A learned χ replaces one exact leg with its
idiosyncratic mixing, dropping the exponent to $\approx 1$ **and** loosening the fit ($r$ $0.96\!\to\!0.55$)
— the collateral is now governed by the code's mixing, not the geometry. This is the sharpest quantitative
form of the program's thesis: **a matched dictionary buys a full extra order of collateral suppression in
$\rho_{mm}$ that no learned code recovers** — FRA edit safety is a dictionary-quality property.

---

## 5. The optimal FRA theorem

**At the optimum in the ground-truth dictionary, FRA-OV carries all channel semantics in
closed form at the warped rate, the only gauge-invariant FRA-QK content is the positional
kernel, and severing a feature's channel nulls its belief carryover with collateral that
vanishes quadratically in superposition — provided the code is matched.**

**(i) FRA-OV is the whole handle. [CONFIRMED]** Per (feature $i$, lag $k$) at the optimum,
$$\text{FRA-OV}(i,k)=\eta_i^{\,k}\cdot \bar c_i\cdot \hat g_i,\qquad \hat g_i\propto d_i,$$
all channel semantics in the OV direction $\hat g_i$, all lag structure in the **trained
rate** $\eta_i$, with the softmax normalization warp giving the realizable form
$\eta_i^{\,d-s}/(1-\eta_i^{\,d})$. The rate is $\eta_i(\rho_{\mathrm{obs}})$, the in-disc
root of the palindromic quadratic $\rho_{\mathrm{obs}}\lambda\eta^2+[(\lambda^2-1)-
\rho_{\mathrm{obs}}(\lambda^2+1)]\eta+\rho_{\mathrm{obs}}\lambda=0$ — **not** the belief
rate $\lambda_i$; $\eta_i=\lambda_i$ only in the pure-noise limit
$\rho_{\mathrm{obs}}\to\infty$. This *derives* sprint-2's measured warp×shrinkage
($\eta=0.464<\zeta$) from the observation SNR, and the warp curve is independently
confirmed against the steady-state Kalman rate to machine precision.

**(ii) The only invariant FRA-QK content is the positional kernel. [CONFIRMED]** Under a
content-independent optimal pattern the tok×tok score block is additively separable and
entirely gauge (pedestal, key-read, profile-share, row constants); the centered interaction
is zero. In the reset process's *linear* model this holds **exactly** (the optimal
predictor is the LTI Wiener filter, whose coefficients depend only on the autocovariance,
never on realized feature values) — a strengthening over Mess3, where it was an empirical
ansatz with measured content leak.

**(iii) Intervention calculus with the matched-code contingency. [CONFIRMED]** Severing
feature $i$'s OV channel at gain $c$ removes fraction $c\,\rho_{\mathrm{path}}$ of its belief carryover,
$\rho_{\mathrm{path}}=1-(1-a_0(d))(1-\eta_i)$, null at $c^\ast=1/\rho_{\mathrm{path}}$ (a
property of the run's position on the k=0 orbit, not the task). Collateral on $j\ne i$ is
**exactly zero at $\rho_{mm}=0$** (orthogonal channels — P1's factored world, not Mess3's
coplanar one) and **$O((d_i^\top d_j)^2)$** at leading order in superposition. The
second-order scaling is not automatic: it arises from a **first-order cancellation** —
the contamination feature $i$ leaks into $j$'s read-out is exactly the first-order term the
matched sever removes — and it **requires the code's encoder and decoder to both be $d_i$
with unit gain**. With a learned $\chi$ (encoder $\ne$ decoder), the cancellation is
imperfect and collateral reverts to $O(\rho_{mm})$. This is the interface to §4: the
"cleanest use/collateral separation in the program" is a ground-truth-dictionary property,
degrading gracefully to first order under an imperfect code.

**Trained-model verification** (theory-reset, `VERIFY_B.md`; `code/reset*.py`,
`out/warp_curve.json`, `out/interv.json`). All checks CONFIRMED.

*(a) The warp curve* (Fig. `out/warp_curve.png`; position-only attention — the
content-independent optimal class of (ii) — $\lambda=0.7$, $N=3$, 3 seeds):

| $\rho_{\mathrm{obs}}$ | $\eta(\rho_{\mathrm{obs}})$ theory | measured rate | attn loss share |
|---|---|---|---|
| 0.13 | 0.130 | 0.243 ± 0.009 | 0.7% (weak) |
| 0.46 | 0.293 | 0.305 ± 0.002 | 2.1% |
| 0.70 | 0.355 | **0.352 ± 0.000** | 2.3% |
| 1.01 | 0.410 | **0.397 ± 0.001** | 2.3% |
| 1.71 | 0.483 | 0.460 ± 0.002 | 1.9% |
| 2.36 | 0.522 | 0.494 ± 0.002 | 1.6% |

The measured rate rises 0.24→0.49 with $\rho_{\mathrm{obs}}$, tracks
$\eta(\rho_{\mathrm{obs}})$ within ~5% wherever attention is pinned, and **never
approaches $\lambda=0.7$**. **Low-noise caveat:** at $\rho_{\mathrm{obs}}=0.13$ the
kernel is only weakly pinned (attention carries 0.7% of the loss) and the rate is
biased upward (0.243 vs 0.130) — the loss-flat regime where $\eta$ is a near-gauge
quantity below the gradient floor (§3 / sprint-2 §flat) — but it is still far below
$\lambda$. The clean-emission null ($\rho_{\mathrm{obs}}\to0$) is confirmed
separately: look-back attention's loss share $\to0$ while the **direct path stays
load-bearing** (direct-only MSE $0.330\ll$ prior $0.636$) — the redundant-signal null
of corner B-clean, distinct from static corner A. Two adjuncts: the linearity tax
(best-additive $-$ exact-Bayes) is 0.9% of the loss at $\rho_{\mathrm{obs}}{=}0.01$
rising to 3.7% at 2.1 and $\to0$ as $\rho_{\mathrm{obs}}\to0$, with trained MSE between
the Bayes and optimal-additive floors (the model is the optimal linear predictor); and
the content-QK sector is loss-inert (freezing it costs 0.1% of the loss) — (ii)
confirmed by loss-equivalence, per the F8 correction (gauge $\ne$ seed-noise;
loss-flat objects are optimizer-pinned).

*(b) Intervention calculus* (2-feature model, $\lambda=(0.5,0.85)$, oracle GT
readout; tracking normalized to clean $=1$):

| $\rho_{mm}$ | $d_i^\top d_j$ | feature-$i$ tracking at $c=(0,.5,1,1.5,2)$ | $c^\ast$(null) | collateral$_j$ (max) |
|---|---|---|---|---|
| 0 | 0.00 | 1.00, .828, .656, .484, .312 | 2.91 | **0.0003** ($\equiv$1.000 $\forall c$) |
| 0.2 | 0.20 | 1.00, .832, .664, .496, .328 | 2.98 | 0.0157 |
| 0.4 | 0.40 | 1.00, .831, .661, .492, .323 | 2.95 | 0.0804 |

Feature-$i$ tracking is exactly linear in $c$, null at $c^\ast=1/\rho_{\mathrm{path}}$
($\rho_{\mathrm{path}}\approx0.34$ measured; $\eta_i=0.258$ — the OV channel carries
$\sim\tfrac13$ of the belief, the skip the rest, hence $c^\ast\approx2.9$,
$\rho_{mm}$-independent). Collateral on $j$ is **identically zero at $\rho_{mm}=0$**
(exact selective removal) and scales as $(d_i^\top d_j)^{2.4}$ (log-log slope 2.36;
leading order 2, the excess higher-order at $\rho_{mm}=0.4$) — the matched-GT-dictionary
$O(\rho_{mm}^2)$ whose first-order cancellation a learned $\chi$ breaks to $O(\rho_{mm})$,
**now independently confirmed** by the sae-recovery slope test (§4: matched code slope
$1.91$ vs learned $0.71$).

*(c) The use-vs-presence law — FRA cannot delete a feature's presence.* Stated as a
theorem: **an FRA edit acts on the attention path only, so a feature's presence (the
decodability of its Bayes belief) is bounded below by the single-observation floor that
the direct/skip path delivers and no OV edit can touch; that floor is
$\mathrm{Corr}^2(\text{belief},\,c_i(d))$, and the FRA-cuttable share $1-\text{floor}$
therefore grows as the current token gets noisier.** The current value $c_i(d)$ enters
$\mathrm{resid\_post}=a_d+\mathrm{ctx}_d$ through $a_d$ on the direct/skip path, which
every OV sever leaves untouched — so the floor is present in every edited condition by
construction. Measured (ridge probe for the Bayes belief, retrained per condition,
$\rho_{\mathrm{obs}}=0.91$; VERIFY_B check 6):

| condition | probe $R^2$ | reading |
|---|---|---|
| clean | 0.709 | baseline |
| sever OV lags $k\ge1$ | 0.683 | look-back carryover only ($3.7\%$ of presence) |
| sever **whole** OV channel incl. $k{=}0$ | **0.648** | bottoms out at the floor ($8.6\%$ removed, all of FRA's reach) |
| single-observation floor (analytic $\mathrm{Corr}^2$) | **0.648** | measured $=$ analytic to 3 decimals |
| counter-steer at $c^\ast$ | 0.700 | $\approx$ clean — nulling relocates use, deletes nothing |

The $\rho_{\mathrm{obs}}$-monotone corollary (floor falls as the token gets noisier, so
$1-\text{floor}$ rises) is confirmed at this one point and follows from the floor formula;
it is the presence-side statement of §2's corner-C observation and moves in lockstep with
the warp $\eta$. This is fra_hmm_toy's use-vs-presence law, now with a closed-form floor
in the reset process. **[CONFIRMED at one $\rho_{\mathrm{obs}}$; monotone trend is a
corollary]** The floor is not merely close but **exact** in the MLP-free case: on
attention-only Mess3 the whole-attention sever hits the single-observation floor to gap
$0.0000$, and the floor is depth-invariant (§7), which also shows that severing must span
*all* layers because the belief is redundantly rebuilt across them.

---

## 6. What breaks, and what is next

**The clean picture breaks in three named places, each of which is a known campaign result
waiting to be derived, and the open question is where real models actually sit on the map.**

**Correlations $\Sigma\ne I$ break zero-collateral even at $\rho_{mm}=0$. [theory; future
measurement]** Copula firing correlations couple the per-feature filters, so feature $j$'s
observation shifts feature $i$'s belief; a cut of $i$ perturbs $j$ through the correlation
channel $\propto\Sigma_{ij}$ — a new, superposition-independent collateral term, and
(because the covariance is stationary) an OV-side break that leaves the QK pattern gauge.
The static analogue is already measured — a correlated *static* dictionary produces χ
hedging (sae-recovery Exp 2: the top off-diagonal lands on the correlated partner 62% of
the time). The *dynamic* collateral claimed here is distinct and **not yet measured**: it
requires a **correlated-reset model class** (a joint 2-feature reset chain with copula
firing) in which the effect lives in the **belief dynamics** — feature $j$'s history
shifting feature $i$'s filter — rather than in the instantaneous code. Building that
model class and measuring the $\propto\Sigma_{ij}$ collateral is the concrete next
experiment.

**Hierarchy is the minimal loss-pinned QK content coupling — and it is `fra_win`. [theory]**
Parent-gating (a child fires only if its parent fired) makes predicting the child *require*
the parent's state, so the optimal pattern becomes content-dependent and FRA-QK acquires a
genuine, gauge-invariant, loss-pinned coupling — a rank-1 (child-query)×(parent-key) term
that is not additively separable. This is the reset-process derivation of the one empirical
regime where FRA-QK already wins: `fra_win`'s association-specific attention control, at
~15× less collateral than ActAdd. Hierarchy and induction are the same corner — the place
where the QK-gauge verdict of §3 stops applying. **[future]** The open question is whether
the exact gate *must* live in QK or is OV-approximable — the `fra_win` boundary — which a
hierarchical-reset model with a trained head would resolve directly.
**→ RESOLVED in §8 (this prediction is amended):** the gate is real and its size is predicted
($\Delta_{\mathrm{gate}}\approx0.014$), but a trained attention-only model carries it in **OV,
not QK** — so hierarchy and induction are *not* the same corner. FRA-QK is the handle only for
an *obligate query-key match* (induction); a content gate is OV-realisable. See §8, boxes 4–5.

**$\lambda\to1$ needs the aggregation formalization. [future]** The counting/mixture regime
(corner D) is where the one-layer worlds studied here run out: the FRA-OV aggregation shares
that set $c^\ast$ come from multi-path structure a single layer lacks. `fra_hmm_toy` and
sprint-2's mixture outlook chart it empirically; the formal $1/(1-\eta)\propto
(1-\lambda)^{-1/2}$ flattening law is the natural next derivation.

**The real-model question. [future]** This note establishes, in closed form, *when* FRA-OV is a
faithful handle (interior, corner C, matched code), *when* it is gauge (corners A, B-clean,
D), and *when* FRA-QK finally becomes a handle (content-gating). `fra_organisms` already
suggests real models satisfy the FRA-OV conditions only under a narrow "load-bearing
attention at the answer step" kernel. The program's payoff is to locate a given trained LLM
on this map — and thereby predict, before any edit, whether an FRA attribution on it is
computation or convention.

---

## 7. Addendum: the MLP-removed control (attention-only models)

**Removing the MLP does not break the FRA story — it sharpens it, and it isolates two
facts the full model obscured: the presence floor becomes exact, and the intervention
null moves to $c^\ast\approx1$ because attention is now the sole belief-carrier.**
(theory-reset's attention-only battery, `../attn_only_redo/ATTN_ONLY.md`; all six sprint-2
checks plus a 3-layer control CONFIRMED on attention-only Mess3, exact $3^{10}$ enumeration.)

**The MLP was not doing the belief computation. [CONFIRMED]** A 1-layer attention-only
model (no MLP at any depth) reaches — and slightly beats — the P2 1L+MLP model's loss for
positive $\zeta$ ($98.1\%$ vs $97.25\%$ of the recoverable information), sits at the
constrained-belief plateau, and carries the belief *more* legibly than the full model: the
post-attention residual linearly decodes the belief at $R^2=0.94$–$0.95$ (vs sprint-2's
$\sim0.78$ with the MLP present), with a token-independent pattern (relstd $6.6\%$) and
$\mathrm{OV}\parallel g(z)$ at cos $0.98$. Every sprint-2 FRA verdict reproduces (QK gauge,
OV handle, $c^\ast=1/\rho_{\mathrm{path}}$, two-head hinge, presence floor), several
*cleaner* because attention alone carries the belief with nothing to hide it.

**(1) The presence floor is exact and depth-invariant. [CONFIRMED]** Severing the entire
attention channel drops the Bayes-belief probe to the single-observation floor with gap
**0.0000** (four decimals, both configs; A floor $0.7976$, B floor $0.8161$) — sharper than
the reset process's $\sim3\times10^{-4}$, because with no MLP $\mathrm{resid\_pre}$ is
exactly the current-token embedding, so the floor is hit to machine precision. Attention
carries only $13$–$16\%$ of the decodable belief; the remaining $\sim84\%$ is the current
token on the skip/embedding path, outside FRA's reach. At depth (3L control) the floor is
unchanged ($0.8161$) and is hit exactly only when **all** layers' attention is severed (gap
$0.0000$); severing the last layer alone leaves $0.948$ — the belief is redundantly rebuilt
across layers, so the **layer-summed** FRA-OV is the invariant. This tightens §5(c): FRA
cannot delete a feature's presence below the single-observation floor, and in the MLP-free
case that bound is *exact*.

**(2) The intervention null sits at $c^\ast\approx1$ — under-reach is an architecture
property, not an FRA property. [CONFIRMED]** In attention-only 1L the FRA-OV sever nulls a
token's belief-plane tracking exactly linearly ($R^2=1.0000000$) with
$\rho_{\mathrm{path}}\approx1$ ($c^\ast\approx0.9$–$1.2$ for the two cuttable tokens): the
attention path carries essentially the *whole* dynamic belief carryover, so exact severing
($c=1$) is nearly *sufficient*. Contrast the reset process, where a live skip delivers the
lag-0 term and $\rho_{\mathrm{path}}\approx0.34$, $c^\ast\approx2.9$ (attention
under-reaches). The severing reach $c^\ast=1/\rho_{\mathrm{path}}$ is therefore set by how
many parallel paths carry the belief carryover — the k=0 skip/diagonal split, the MLP, and
additional layers — a property of the **architecture**, not of FRA: the method faithfully
reports what the attention path carries, and whether nulling it suffices depends on the
carrier count. On the §2 map this sharpens corner C's intervention story: $c^\ast$ is not a
property of the concept but of the architecture's belief-carrier multiplicity.

Two honest caveats (carried to the ledger): the negative-$\zeta$ two-head parity tail is
incomplete under lazy (default) init — the head-sum invariant is recovered but the per-head
parity split is soft (rich init sharpens it, per sprint-2); and the gauge-dial cut effect
swings $1.7\times$ here rather than sprint-2's $140\times$ because this cut subtracts the
pedestal *magnitude* $\|k_E(z_0)\|$ (which stays positive) rather than the *signed* pedestal
(which swings through zero) — the decisive result, exact loss-invariance ($\Delta\mathrm{CE}
\le10^{-8}$) while the FRA-QK coordinate moves $\sim3\times$, is unchanged.

---

## 8. The hierarchy round: when FRA-QK finally has a handle — and where the model actually puts it

**Hierarchy (a child feature fires only when its parent fired) is the minimal process that
forces the optimal computation to gate on content; §6 predicted it hands FRA-QK its first
loss-pinned handle. The gating value is real and matches the closed form — but the trained
model puts the gate in OV, not QK, the QK/OV/norm junction is architecture-and-training-chosen,
and FRA's value-path attribution turns that localization into a control edit that beats the DoM
and SAE baselines by up to $259\times$.** (theory-reset's H-round; process
`../hierarchy_fra/H_process.md`, theory `H_theory.md`, verification `VERIFY_H.md`; numbers trace
to `../hierarchy_fra/out/{rung2_frontiers,rung2_lncheck,seed_replicate,seed_frontier,decouple_b_check}.json`.
Platform: **theory-clean** — attention-only, no MLP, no LayerNorm — per the program convention
that every component the theory does not model is *removed*, not controlled for.)

**(1) The gating value is real and matches the closed form. [CONFIRMED — enumerated + 3 seeds]**
At the trained config the exact gate is $\Delta_{\mathrm{gate}}=0.0140$ (full joint-Bayes child
MSE $0.2411$ vs *parent-blind* Bayes $0.2551$), stable at $0.0138$–$0.0141$ across 3 seeds; H2.2's
process-weighted Jensen-gap surrogate predicts $0.0189$ ($\sim\!1.35\times$) — the formula is
basically right, not the "$14\times$ off" that rung-1's raw loss gate suggested (box 3). The gate
exists only where the process is **both** content-gated **and** partially observed: it vanishes
*exactly* on the null control (independent chains, gating value $0.0000$) and on the clean control
($0.0005$) — the conjunctive noise×gating condition, in closed form. **Correction to H2.1:** the
right isolator is parent-blind Bayes, **not** the best affine lag-only predictor — the affine class
additionally pays a large *nonlinear-filtering* penalty ($0.073$ on the null cell) that has nothing
to do with gating, so "lag-only is sufficient on the null control" is false under occlusion.

**(2) The gate imposes minimal depth $\ge2$ — deriving the induction-head depth fact. [CONFIRMED — 1L insufficient]**
The reset-gate is a **three-position interaction** — (child-query at $d$) × (child-evidence key at
$s$) × (parent-death on the interval $(s,d]$) — whereas a single attention layer's score is a *sum
of pairwise* query-key products and cannot represent the third factor (a property of the interval,
not either endpoint). Measured: the 1-layer parent→child handle is $\approx0$ across configs,
nonzero only at $\ge2$ layers. This *derives, from process structure,* the empirical fact that
induction heads need two layers — the same three-position obstruction (query, key, and the token
between).

**(3) A standard-LayerNorm "position-only" control is not content-blind — LN is a hidden carrier. [CONFIRMED]**
Rung-1's loss gate (full-QK vs position-only-QK, standard LN) read $0.0013$ against the $0.0189$
surrogate — an apparent $14\times$ shortfall. It is a **measurement confound**: LayerNorm's
per-token normalization is a content-dependent nonlinearity *outside* the QK pattern, and a
position-only+LN model reaches $0.2447$ — *below* the analytic affine lag-only frontier ($0.2511$),
recovering $\sim\!78\%$ of the gate on its own. Linearizing LN returns it to the frontier ($0.2503$).
So a "position-only" pattern-freeze control silently captures the gate and manufactures a false null
— the pattern-freeze-confound class (cf. the EM $\alpha$ artifact), here with a *named* mechanism.
**Carrier multiplicity now includes normalization nonlinearities**, which are invisible to
attention-level (QK/OV) attribution because they live between the residual stream and the pattern.
This is why the theory-clean platform *removes* LN rather than controlling for it.

**(4) The amended fra_win law, and why it retro-explains the campaign. [CONFIRMED — 3 seeds]**
On the theory-clean platform the gate *is* learned (full-QK reaches Bayes) but is carried by **OV,
not QK**: projecting the parent out of the **keys** (the targeted FRA-QK cut) costs the child
$+0.0007$ — a *gauge-robust null* — while projecting it out of the **values** (FRA-OV cut) costs
$+0.0203\approx\Delta_{\mathrm{gate}}$; a $21$–$36\times$ OV-vs-QK asymmetry across 3 seeds, gauge-flat
on every seed (cut-effect std $<10^{-9}$; figure `../hierarchy_fra/out/handle_final.png`). Mechanism
(the depth-2 structure of box 2 biting attribution): **layer-1 reads the parent through OV into a
*derived* recency feature; layer-2's child-query gates on that derived feature**, so the raw
(child-query, parent-key) cell is empty and severing the parent's *OV entry* is what kills the gate.
This **falsifies §6's prediction** (and H2.3/H3.2) that hierarchy gives FRA-QK its first handle.
**Amended law:** a two-position product is *necessary but not sufficient* for a QK handle — it must
be an **obligate query-key match**, unrealisable in OV; a content **gate** is OV-realisable and
generically lands in OV. This reads the whole campaign in one stroke: **induction (`fra_win`) is the
one obligate match** — the copy $[A][B]\dots[A]\!\to\![B]$ cannot move into OV — **and is the *only*
QK win**; every other concept probed (hierarchy screening, belief carryover, sparse selection, Mess3
aggregation) is a gate or an aggregate, OV/content-carried, so **FRA-QK was inert everywhere else.**
The QK-vs-OV boundary is *why* the campaign found exactly one QK win and FRA-OV handles throughout the
interior.

**(5) Capstone — FRA-OV attribution is a superior control ACTUATOR, not just a faithful locator. [CONFIRMED — 3 seeds + decoupled-bias control]**
Once the carrier is localized (OV), an FRA path-decomposition gives a control edit that dominates the
standard baselines on the removal-vs-collateral frontier. Target: remove the model's *use* of the
parent→child gate (drive child MSE from base to the parent-blind floor); collateral: parent-prediction
MSE. The FRA OV-route cut (mean-ablate the parent-value contribution to the child-readout direction
$r_C=\mathbf W_{\mathrm{out}}^{\top}d_C$) removes the *full* gate-use at parent collateral
$\sim\!0.0004$–$0.0015$, versus difference-of-means projection (rank-3) at $\sim\!0.020$–$0.026$ and
planted-SAE $d_P$ ablation at $\sim\!0.07$–$0.10$ — **FRA wins by $17$–$51\times$ (DoM-rank3) and
$48$–$259\times$ (planted-SAE) under mean-ablation, across 3 seeds + the decoupled-bias control**,
with near-zero excess child damage (figure `../hierarchy_fra/out/control_frontier.png`). *Why:* FRA
severs the OV value-path carrying the gate while sparing the **direct-path** parent representation that
parent-self-prediction reads; the residual-projection baselines cannot separate the two and must
*destroy parent self-prediction* to remove the gate — the same route-vs-representation win as
`fra_win`'s $\sim\!15\times$ ActAdd margin, here reproduced end-to-end from process structure.
*Robustness:* the FRA$<$baseline ordering **never breaks across 3 seeds × 2 ablation conventions × 2
bias regimes**, with a conservative worst-case floor of $\ge2\times$ vs DoM-rank3 (the crudest
zero-ablation under a generic bias); the $17$–$51\times$/$48$–$259\times$ figures are the principled
mean-ablation range.

---

## 9. Bridge to real models (Gemma-2-2b)

**The §8 amended fra_win law was carried to a 2B-parameter model, and it transfers for the structural class
it names — obligate query-key matches — with two honest refinements the toy could not see, and one open
weak point.** (Full write-up, all four bridge deliverables, and the tagged data:
`../constrained_belief_updating/bridge_to_real/BRIDGE_NOTE.md`; Gemma-2-2b + GemmaScope-65k, `out/bridge1_*.json`,
`out/bridge3_*.json`.)

- **The carrier→tool law transfers for MATCHES. [CONFIRMED — Gemma-2-2b]** On a planted in-context induction
  backdoor, FRA-QK is the surgical control tool — it removes at **~0.056 nats** held-out collateral, **13–312×**
  cheaper than the content-direction baselines (DoM/conv-SAE/payload-suppress, 5–328 nats). This is the exact
  **mirror of §8**: there FRA-OV won because the gate was OV-carried; here FRA-QK wins because induction is an
  obligate MATCH. FRA-QK reach is cue-dependent and mostly low (max removal 0.003–0.14), so the win is
  collateral-efficiency at the removal it reaches.

- **The carrier law governs TRAINED backdoors — the handle flip replicates 2/2 seeds. [CONFIRMED — Gemma-2-2b,
  2 seeds]** Two LoRA-finetunes sharing one special-token trigger: (A) token→**fixed** string (weight-stored) →
  FRA-QK **inert** (1.8×10⁻⁵ / 7.2×10⁻⁷ across seeds), DoM the only handle; (B) token→**in-context-retrieved**
  string (routing learned, generalizes to held-out payloads at ASR 0.80/0.88) → FRA-QK a **viable handle**
  (0.72 / 0.028). Finetuning *creates* the QK edge in B and *bakes the string into weights* in A, extending the
  law from planted to *trained* backdoors. The **flip direction is robust 2/2** (FRA-QK handle present iff
  in-context-routed); the ~40,000× B/A ratio is direction-robust but magnitude-soft (partly a noise floor, and
  B's absolute reach is seed-variable 0.028–0.72).

- **Refinement 1 — "carrier" must be defined by collateral-EFFICIENCY, not raw effect. [CONFIRMED — Gemma-2-2b]**
  The naive QK-vs-OV *raw-ASR-effect* ratio inverts at scale (the OV-cut removes *more* ASR as a content
  sledgehammer, ~50× the collateral); only the collateral-normalized read recovers the correct carrier (QK-cut
  0.056 vs OV-cut 2.9 nats at matched removal). The toy conflated effect-size and surgicality because its OV gate
  was *both*; at 2B they separate.

- **Refinement 2 — RMSNorm is a partial hidden carrier (LayerNorm redux). [CONFIRMED — Gemma-2-2b]** §8's finding
  that LayerNorm silently carries a gate recurs as RMSNorm: the FRA-QK cut effect swings **~35%** between the two
  faithful RMS-handling conventions (SAE-reconstruction rms vs true-resid rms), so the magnitude correction does
  not fully remove the folding convention. And **surgical low-collateral control does NOT transfer to the trained
  model** — B's FRA-QK is only ~3× cheaper than DoM at high absolute collateral (~10–20 nats), because B's route
  reuses the model's *general* induction heads (L15H0/L22H4), so cutting it damages general copying. *Surgical
  removal is a planted-backdoor property; handle-presence is what transfers.*

- **The open weak point — the §8 GATE-OV *control* win is UNTESTED at scale. [future]** No real gate organism has
  been built, so §8's headline (FRA-OV path-decomposition beats DoM/SAE by 17–259× on an OV *gate*) is unconfirmed
  on any real model; the one Gemma case where FRA-OV beats DoM (2–5×) is a *match's transport*, blunt, not a gate.
  Building a real OV-gate organism (a reused-marginal relation) is the experiment that would close the bridge.

---

## Honesty ledger

The claims tagged **[CONFIRMED]** are supported by trained-model measurements or
float-precision checks (traceable to `VERIFY_{A,B,SAE}.md` and `out/*.json`); the
following caveats bound how far each generalizes.

- **The low-noise kernel is only weakly pinned.** At $\rho_{\mathrm{obs}}=0.13$ the
  look-back attention carries just $0.7\%$ of the loss, and the measured rate ($0.243$)
  sits above the theory $\eta$ ($0.130$): in this loss-flat regime $\eta$ is a near-gauge
  quantity below the CE gradient floor. The warp is confirmed *where attention is pinned*
  ($\rho_{\mathrm{obs}}\ge0.46$); the clean-corner rate is directionally correct (still
  $\ll\lambda$) but quantitatively soft.
- **QK-gauge is exact only in the linear model.** §5(ii)'s "content×content is gauge" is an
  identity for the LTI-Wiener optimum (a strengthening over Mess3's empirical ansatz), but
  in the softmax/CE instantiation the pattern content-gates at $O(\varepsilon)$, so on a
  trained softmax head the content-QK sector is gauge only *on average* (loss cost $0.1\%$,
  measured) — a loss-flat residual, not a proven zero.
- **The hierarchy severability prediction is refined, not confirmed as stated.** "Severability
  fails for children before MCC degrades" is *false in the strict rowspace sense at a complete
  code*: absorption is per-latent contamination ($\chi_{\text{child,parent}}=0.545$), and a
  latent *combination* still severs the child exactly. Incompleteness ($N>d$), not hierarchy,
  is what removes a channel; MCC certifies neither.
- **The negative-$\zeta$ two-head parity tail is lazy-init-limited (§7).** On attention-only
  Mess3 config B, the loss-pinned head-sum invariant is recovered (belief probe $R^2=0.95$),
  but the per-head parity split is soft and the lag-$\ge3$ oscillation incomplete under
  TransformerLens-default (lazy) init — the model reached only $92.5\%$ recovery; rich init
  sharpens it, as sprint-2 found. Per-head attributions are the gauge family regardless.
- **The attention-only gauge-dial swing is $1.7\times$, not $140\times$ (§7).** The gauge dial
  moves the FRA-QK coordinate $\sim3\times$ at exact loss-invariance ($\Delta\mathrm{CE}\le
  10^{-8}$) — the decisive result — but the *named cut's* effect swings only $1.7\times$ because
  the cut subtracts the pedestal magnitude $\|k_E(z_0)\|$ (which stays positive), not the signed
  pedestal (which would swing through zero and reproduce sprint-2's $140\times$ range). The
  gauge verdict is unchanged; the magnitude is a proxy-cut artifact, not a weaker effect.
- **Scope of the measured cells.** The SAE recovery and collateral-slope results are single-seed
  per cell (effects are large vs the VERIFY_A seed-noise scale) and TopK-only; the slope test is
  one $\rho_{mm}\approx0.43$ config ($\sim530$ pairs). The presence law and the intervention
  collateral are each measured at a single $\rho_{\mathrm{obs}}$/$\rho_{mm}$ point — the monotone
  trends are corollaries of the closed forms, not swept. The intervention collateral exponent
  reads $2.36$ (leading order $2$; the excess is higher-order at $\rho_{mm}=0.4$). The multi-$\lambda$
  conic head-count (B check 5) was not run.
- **The reset-process belief surface is not a clean SAE substrate.** A naively-trained SAE on the
  post-attention belief surface learns a mixture/contrast code (MCC $0.41$ at the GT-firing $K$;
  rescued to $0.84$ only by extra capacity), and cannot sever a belief channel — the planted GT
  dictionary is required. This is the χ-completeness constraint biting on the very surface FRA
  operates on (VERIFY_SAE Exp 3, secondary).
- **The hierarchy round (§8) is 3-seed on a theory-clean (no-LN) platform.** The OV-vs-QK carrier
  asymmetry, gauge-flatness, and the control-frontier win all replicate across 3 seeds, and the
  FRA$<$baseline ordering holds under 2 ablation conventions × 2 bias regimes (decoupled-bias
  control) — so no §8 claim is single-seed. Remaining caveats: (i) the OV-cut *absolute* magnitude
  is geometry-dependent ($0.02$–$0.13$ across seeds; only the asymmetry and the loss-anchored
  $\Delta_{\mathrm{gate}}=0.0138$–$0.0141$ are stable). (ii) The SAE baseline is the **best-case
  *planted* $d_P$** channel; a trained TopK SAE is deferred, so a real SAE's collateral is $\ge$ the
  planted number — the FRA advantage is a *lower bound* against a real SAE. (iii) All frontier edits
  use the **mean-ablation** convention (remove the parent signal, keep the bias) — see the
  decoupled-bias line below.
- **A structural bias artifact was found in the generator and closed by explicit control (the
  falsifier discipline working).** The frontier's mean-ablation convention is load-bearing because
  the process's bias $b$ is *coincident with the parent direction* $d_P$ — measured $b\cdot d_C=
  O(10^{-16})$, $b\cdot d_P\approx-2.5$ on **every** seed. This is not a per-seed fluke but a
  **structural** `HierProcess`/`make_dictionary` seed-sharing quirk (both draw from
  `default_rng(seed)`, so $b\propto$ the first raw dictionary row). Because it persists across seeds,
  it cannot be dodged by re-seeding; instead a **decoupled-bias control** (generic $b$, retrained)
  confirms the FRA win survives — $17\times$/$48\times$ (DoM/SAE) under mean-ablation and $2\times$/
  $44\times$ under zero-ablation — so the win is not an artifact of the coincidence or the convention.
  (`../hierarchy_fra/out/decouple_b_check.json`.)

*Status: §1–§5 and the §7 MLP-removed control are settled and confirmed; §8 (the hierarchy round —
gating value, depth$\ge2$, LN-carrier confound, the OV-not-QK carrier reversal, and the control-frontier
win) is settled and 3-seed confirmed, and resolves the §6 hierarchy QK-vs-OV boundary item; §9 (the bridge
to Gemma-2-2b) confirms the carrier→tool law transfers for matches — including a 2-seed-direction-robust
handle flip on *trained* backdoors — with two refinements (collateral-normalized carrier, RMSNorm hidden
carrier) and one
open item (the gate-OV control win, untested at scale for want of a real gate organism); the two remaining
§6 items (correlated-reset dynamics and the $\lambda\to1$ flattening law) are scoped future work.*
