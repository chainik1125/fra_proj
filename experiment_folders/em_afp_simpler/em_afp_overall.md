# EM-AFP Simpler: Overall Summary

This file summarizes the current simplified EM-AFP toy results. The detailed
autonomous run log is in
[`../em_afp_simpler_codex_auto/autonomous_findings.md`](../em_afp_simpler_codex_auto/autonomous_findings.md).

## Current Best Result

The strongest current toy result uses the special-state SFP HMM with no
wrong-special leak and high persona-special onset:

$$
\alpha = 0,\quad
p_{\mathrm{persona}}=p_{\mathrm{domain}}=0.50,\quad
p_s^{\mathrm{persona}} = 0.90,\quad
p_s^{\mathrm{domain}} = 0.70,\quad
\epsilon_{\mathrm{persona}} = 0.30,\quad
\epsilon_{\mathrm{domain}} = 0.04.
$$

## Explicit HMM

The vocabulary factorizes into persona and domain symbols:

$$
V = V_{\mathrm{persona}}\times V_{\mathrm{domain}},
$$

where

$$
V_{\mathrm{persona}}=\{0,1,S_M,S_A\},\qquad
V_{\mathrm{domain}}=\{0,1,S_D,S_O\}.
$$

The hidden state space is a direct sum over four leaves:

$$
\mathcal H =
H_{MD}\oplus H_{MO}\oplus H_{AD}\oplus H_{AO}.
$$

Each leaf has four local states:

$$
(N_p,N_d),\quad (N_p,U_d),\quad (U_p,N_d),\quad (U_p,U_d),
$$

where \(N\) is neutral and \(U\) is special. The leaves are non-ergodic with
respect to each other: probability mass never moves between \(MD,MO,AD,AO\).
The global transition matrix is therefore

$$
T_{\mathrm{global}} =
\operatorname{blockdiag}(K,K,K,K),
$$

with one identical within-leaf transition block \(K\). The only leaf-dependent
part is which special symbol counts as "own" versus "wrong".

### General local matrices

Each local factor \(x\in\{p,d\}\) has two hidden states \(N_x,U_x\), two
neutral emissions \(0,1\), an own-special emission, and a wrong-special
emission. Its parameters are

$$
p_x,\qquad \epsilon_x,\qquad q_x=p_s^x,\qquad \alpha.
$$

Here \(p_x\) splits neutral emissions between symbols \(0\) and \(1\),
\(\epsilon_x\) is the neutral-to-special onset rate per neutral symbol,
\(q_x\) is special-state persistence, and \(\alpha\) is the wrong-special leak.

Rows are current state \((N_x,U_x)\); columns are next state \((N_x,U_x)\). The
observable local matrices are:

$$
A^{(x)}_0 =
\begin{pmatrix}
p_x-\epsilon_x & \epsilon_x\\
0 & 0
\end{pmatrix},
\qquad
A^{(x)}_1 =
\begin{pmatrix}
1-p_x-\epsilon_x & \epsilon_x\\
0 & 0
\end{pmatrix},
$$

$$
A^{(x)}_{\mathrm{own}} =
(1-\alpha)
\begin{pmatrix}
0 & 0\\
1-q_x & q_x
\end{pmatrix},
\qquad
A^{(x)}_{\mathrm{wrong}} =
\alpha
\begin{pmatrix}
0 & 0\\
1-q_x & q_x
\end{pmatrix}.
$$

Summing over emitted symbols gives the hidden-state transition matrix for the
factor:

$$
T_x =
A^{(x)}_0+A^{(x)}_1+
A^{(x)}_{\mathrm{own}}+
A^{(x)}_{\mathrm{wrong}}
=
\begin{pmatrix}
1-2\epsilon_x & 2\epsilon_x\\
1-q_x & q_x
\end{pmatrix}.
$$

Notice that \(p_x\) affects which neutral token is emitted, but not the hidden
transition matrix after summing over neutral symbols.

For a leaf \((s,r)\), the persona factor's own-special symbol is \(S_M\) if
\(s=M\) and \(S_A\) if \(s=A\). The domain factor's own-special symbol is
\(S_D\) if \(r=D\) and \(S_O\) if \(r=O\).

The within-leaf transition matrix is

$$
K = T_p\otimes T_d.
$$

Writing

$$
a=1-2\epsilon_p,\quad b=2\epsilon_p,\quad c=1-q_p,\quad d=q_p,
$$

and

$$
A=1-2\epsilon_d,\quad B=2\epsilon_d,\quad C=1-q_d,\quad D=q_d,
$$

we have, in state order
\((N_p,N_d),(N_p,U_d),(U_p,N_d),(U_p,U_d)\),

$$
K =
\begin{pmatrix}
aA & aB & bA & bB\\
aC & aD & bC & bD\\
cA & cB & dA & dB\\
cC & cD & dC & dD
\end{pmatrix}.
$$

Equivalently, the general local observable transition table is:

| current | emitted local symbol | next \(N_x\) | next \(U_x\) |
|---|---|---:|---:|
| \(N_x\) | 0 | \(p_x-\epsilon_x\) | \(\epsilon_x\) |
| \(N_x\) | 1 | \(1-p_x-\epsilon_x\) | \(\epsilon_x\) |
| \(N_x\) | own special | 0 | 0 |
| \(N_x\) | wrong special | 0 | 0 |
| \(U_x\) | 0 | 0 | 0 |
| \(U_x\) | 1 | 0 | 0 |
| \(U_x\) | own special | \((1-\alpha)(1-q_x)\) | \((1-\alpha)q_x\) |
| \(U_x\) | wrong special | \(\alpha(1-q_x)\) | \(\alpha q_x\) |

### Current best parameter values

The current best lower-neutral process uses:

| parameter | persona \(p\) | domain \(d\) |
|---|---:|---:|
| neutral split \(p_x\) | 0.50 | 0.50 |
| onset \(\epsilon_x\) | 0.30 | 0.04 |
| special persistence \(q_x=p_s^x\) | 0.90 | 0.70 |

with no wrong-special leak:

$$
\alpha = 0.
$$

Thus

$$
T_p =
\begin{pmatrix}
0.40 & 0.60\\
0.10 & 0.90
\end{pmatrix},
\qquad
T_d =
\begin{pmatrix}
0.92 & 0.08\\
0.30 & 0.70
\end{pmatrix},
$$

and

$$
K = T_p \otimes T_d =
\begin{pmatrix}
0.368 & 0.032 & 0.552 & 0.048\\
0.120 & 0.280 & 0.180 & 0.420\\
0.092 & 0.008 & 0.828 & 0.072\\
0.030 & 0.070 & 0.270 & 0.630
\end{pmatrix}.
$$

The instantiated within-leaf transition table is:

| current state | next \((N_p,N_d)\) | next \((N_p,U_d)\) | next \((U_p,N_d)\) | next \((U_p,U_d)\) |
|---|---:|---:|---:|---:|
| \((N_p,N_d)\) | 0.368 | 0.032 | 0.552 | 0.048 |
| \((N_p,U_d)\) | 0.120 | 0.280 | 0.180 | 0.420 |
| \((U_p,N_d)\) | 0.092 | 0.008 | 0.828 | 0.072 |
| \((U_p,U_d)\) | 0.030 | 0.070 | 0.270 | 0.630 |

The instantiated persona observable transition table is:

| current | emitted persona symbol | next N | next U |
|---|---|---:|---:|
| \(N_p\) | 0 | 0.20 | 0.30 |
| \(N_p\) | 1 | 0.20 | 0.30 |
| \(N_p\) | own special | 0 | 0 |
| \(N_p\) | wrong special | 0 | 0 |
| \(U_p\) | 0 | 0 | 0 |
| \(U_p\) | 1 | 0 | 0 |
| \(U_p\) | own special | 0.1000 | 0.9000 |
| \(U_p\) | wrong special | 0 | 0 |

The instantiated domain observable transition table is:

| current | emitted domain symbol | next N | next U |
|---|---|---:|---:|
| \(N_d\) | 0 | 0.46 | 0.04 |
| \(N_d\) | 1 | 0.46 | 0.04 |
| \(N_d\) | own special | 0 | 0 |
| \(N_d\) | wrong special | 0 | 0 |
| \(U_d\) | 0 | 0 | 0 |
| \(U_d\) | 1 | 0 | 0 |
| \(U_d\) | own special | 0.3000 | 0.7000 |
| \(U_d\) | wrong special | 0 | 0 |

A full observed-token operator for token \((a,b)\in
V_{\mathrm{persona}}\times V_{\mathrm{domain}}\) is block diagonal. In each
leaf \((s,r)\), its 4x4 block is

$$
T_{(a,b)}\big|_{(s,r)} = T^{(s)}_{p,a}\otimes T^{(r)}_{d,b},
$$

where \(T^{(s)}_{p,a}\) and \(T^{(r)}_{d,b}\) are the corresponding 2x2
observable local matrices from the two tables above. Summing over all 16
tokens recovers \(T_{\mathrm{global}}\).

## Fine-Tuning Curve

After MD fine-tuning, the O-domain prompt shows broad misalignment with much
less neutral readout mass than the earlier runs. Full three-seed validation at
the 100-step checkpoint gave:

$$
\ell_D = 5.980 \pm 0.123,\qquad
\ell_O = 5.874 \pm 0.267,\qquad
\ell_D - \ell_O = 0.106 \pm 0.146.
$$

Here

$$
\ell_D = \log \frac{P(S_M\mid D\text{-prompt})}{P(S_A\mid D\text{-prompt})},
\qquad
\ell_O = \log \frac{P(S_M\mid O\text{-prompt})}{P(S_A\mid O\text{-prompt})}.
$$

So broad O-domain odds are very close to narrow D-domain odds, with odds
factor $\exp(\ell_D-\ell_O)\approx 1.11$, well within the
acceptable factor-of-two band.

At step 100, the O-prompt persona-side mass accounting was:

$$
P(S_M \mid O) = 0.590,\quad
P(S_A \mid O) = 0.00168,\quad
P(\mathrm{persona\ neutral}\mid O)=0.408.
$$

The model still learned the process and represented the exact posterior well:

$$
\mathrm{CE}_{\theta} - \mathrm{CE}_{\mathrm{Bayes}} = 0.00259,\qquad
\bar R^2_{\mathrm{probe}} = 0.998.
$$

## Early Fine-Tuning Checkpoint

The lowest neutral mass appears earlier in fine-tuning, not at the final
100-step checkpoint. In the full-validation trajectory, the O-prompt neutral
share drops to roughly 0.29-0.30 at steps 5-10:

| FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | $P(\mathrm{neutral}\mid O)$ | $\ell_O$ |
|---:|---:|---:|---:|---:|
| 1 | 0.384 | 0.179 | 0.437 | 0.763 |
| 5 | 0.679 | 0.0354 | 0.285 | 2.956 |
| 10 | 0.738 | 0.0129 | 0.249 | 4.053 |
| 20 | 0.594 | 0.0106 | 0.395 | 4.042 |
| 50 | 0.610 | 0.00281 | 0.388 | 5.394 |
| 100 | 0.590 | 0.00168 | 0.408 | 5.874 |

This suggests that the best low-neutral behavioral operating point is a
matched fine-tuning dose or early checkpoint, rather than always using the
100-step endpoint.

The full fine-tuning curve is plotted here:

- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png)

The backing table is:

- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv)

The companion **coherence curve** (independent `em_afp_simpler_claude` reimplementation, three
seeds) tracks `incoherence(O)` over the same dose axis: the per-token gap between the model's
generated O-continuations and the exact Bayes predictor, measured on the **domain channel only** (the
persona shift cannot move the domain marginal, so this isolates corruption from misalignment; 0 =
the broad generations stay on-process). Broad EM appears by step 5 and persists, while incoherence is
near the floor at the early checkpoint (≈0.04 nats at step 10) and rises only with continued
fine-tuning (≈0.08 nats by step 100), with notable seed spread — the quantitative version of "use the
early checkpoint."

![Frozen-model coherence curve: incoherence(O) vs MD fine-tuning dose, with broad EM for context](../em_afp_simpler_claude/results/incoh_h1/frozen_coherence_curve.png)

## Interpretation

The earlier weak-leak result was partly an odds result: aligned special mass
collapsed while misaligned special mass stayed only a few percent. Increasing
$\epsilon_{\mathrm{persona}}$ changes this qualitatively, and the alpha-zero
validation removes the wrong-special leak as a confound. The broad O-domain
misaligned special probability becomes large in absolute terms, while aligned
special probability remains small.

The tradeoff is that the HMM is no longer very sparse in persona-special
emissions. This is still the same minimal two-state factor structure, but the
persona special state is now easier to enter by construction.

## What Is Driving This?

The alpha-zero validation shows that wrong-special leak is not needed for the
effect. The current best result is instead driven by a persona/domain
timescale asymmetry:

$$
\epsilon_p = 0.30,\quad q_p=0.90,\qquad
\epsilon_d = 0.04,\quad q_d=0.70.
$$

High persona onset and persistence make persona-special mass easy to produce
after fine-tuning. Low domain onset and persistence prevent the O-domain prompt
from becoming too strongly insulated from the MD fine-tuning update. Together
these give large absolute broad-misaligned mass while keeping the narrow/broad
odds gap small.

Two matched-parameter controls make this clearer:

Endpoint, measured at the 100-step checkpoint:

| condition | $\epsilon_p$ | $q_p$ | $\epsilon_d$ | $q_d$ | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral | $P(S_M\mid D)$ | $P(S_A\mid D)$ | D neutral | $\exp(\ell_D-\ell_O)$ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| current asymmetric | 0.30 | 0.90 | 0.04 | 0.70 | 0.590 | 0.00168 | 0.408 | 0.591 | 0.00150 | 0.408 | 1.11 |
| persona matched down to domain | 0.04 | 0.70 | 0.04 | 0.70 | 0.090 | 0.00137 | 0.909 | 0.0787 | 0.000418 | 0.921 | 2.40 |
| domain matched up to persona | 0.30 | 0.90 | 0.30 | 0.90 | 0.483 | 0.0397 | 0.477 | 0.596 | 0.00237 | 0.401 | 15.44 |

Best observed O-neutral checkpoint in the saved fine-tuning grid:

| condition | best FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral | same-step $\exp(\ell_D-\ell_O)$ |
|---|---:|---:|---:|---:|---:|
| current asymmetric | 10 | 0.738 | 0.0129 | 0.249 | 0.99 |
| persona matched down to domain | 10 | 0.0986 | 0.00783 | 0.894 | 2.10 |
| domain matched up to persona | 10 | 0.569 | 0.100 | 0.331 | 10.49 |

Matching persona down to the domain parameters mostly destroys the absolute
broad effect: the O-prompt is about 91% persona-neutral at the endpoint. The
odds ratio is still nontrivial, but this is again partly an aligned-mass
collapse rather than strong broad-misaligned completion.

Matching the domain up to the persona parameters preserves substantial
O-prompt misaligned mass, but it makes the narrow/broad gap large. In this
setting the D-prompt response is much more polarized than the O-prompt
response, with a roughly 15x D-over-O odds factor. This is outside the
factor-of-two band and recovers the failure mode we were trying to avoid.

So the current evidence points to the asymmetric setting as the important
ingredient: persona specials must be frequent enough to carry the misaligned
update in absolute probability, while domain specials should remain weaker so
the O-domain prompt still exhibits broad spillover rather than becoming a
separate, shielded sector.

The comparison table and plot are:

- [`../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_summary.csv`](../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_summary.csv)
- [`../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_mass_and_odds.png`](../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_mass_and_odds.png)

## Coherence Check: Are the Domain Marginals Preserved?

A broad-misalignment readout is only meaningful if the fine-tuned model still behaves like the
process *except* for the persona it now prefers. The right way to check this here is the **domain
marginal**: after an O-prompt the leaf belief is confined to the two O-leaves \(MO,AO\) (the token
\(S_O\) gives zero likelihood to the D-leaves when \(\alpha=0\)), and those two leaves share the
*same* domain factor. So the MD fine-tune can legitimately move the **persona** marginal
\(P(\text{persona symbol}\mid O)\) — that is the misalignment — but it **cannot** legitimately move
the **domain** marginal \(P(\text{domain symbol}\mid O)\). Any drift in the domain marginal away from
the exact Bayes value is therefore corruption of the process model, not misalignment.

An independent reimplementation (the parallel `em_afp_simpler_claude` line, same factor convention)
reproduced the base model here — Bayes CE gap \(0.0026\), probe \(\bar R^2=0.985\) — and the post-FT
persona masses (\(P(S_M\mid O)\approx0.66\) at step 10, \(\approx0.60\) at step 100, broad tracking
narrow). It then measured, per fine-tuning dose, the total-variation distance of the model's
next-token persona and domain marginals from the exact base-process marginals, plus a generative
domain-coherence (the fraction of generated domain markers that are \(S_O\)). Three seeds:

| FT step | \(P(S_M\mid O)\) | domain-TV(O) | persona-TV(O) | \(P(S_D\mid O)\) | gen. domain-coh(O) |
|---:|---:|---:|---:|---:|---:|
| 0 | 0.345 | 0.021 | 0.076 | 0.0013 | 0.99 |
| 10 | 0.655 | 0.141 ± 0.09 | 0.355 | 0.0042 | 0.84 |
| 50 | 0.625 | 0.230 ± 0.14 | 0.325 | 0.0061 | 0.65 |
| 100 | 0.598 | 0.208 ± 0.16 | 0.303 | 0.0069 | 0.62 |

The picture is the expected one. The persona marginal moves a lot (persona-TV \(\approx0.3\)) — the
intended emergent misalignment. The next-token wrong-domain rate \(P(S_D\mid O)\) stays near zero, so
the immediate readout is domain-coherent (the \(\alpha=0\) structure doing its job). At a **low dose
(step \(\approx10\), the operating point the neutral-mass analysis already favoured)** the domain
marginal is still close to Bayes (TV \(\approx0.14\)) and generation stays in O (coherence 0.84): the
broad spillover there is genuine, coherent misalignment. As fine-tuning continues, the domain marginal
drifts further (TV \(\to\approx0.21\)) and generative coherence falls to 0.62 — and the effect is
seed-dependent (one seed stayed clean, one drifted, one fell to generative coherence 0.19 at step
100). So the small per-step distortion compounds over a continuation even while the single next-token
marker stays correct, and the step-100 endpoint sits in a mildly corrupted regime.

This validates reading the broad result at the early checkpoint rather than the 100-step endpoint, and
suggests tracking domain-marginal TV alongside the odds so coherence is measured, not assumed.

![Frozen model coherence: the persona marginal moves (the intended EM) while the domain marginal also drifts with fine-tuning dose, worst at the endpoint](../em_afp_simpler_claude/results/frozen_h1/frozen.png)

Backing data and code: `../em_afp_simpler_claude/results/frozen_h1/` (JSON + figure), produced by
`../em_afp_simpler_claude/code/driver.py::run_frozen_coherence`.

## Eliminating Neutral Prompts

The remaining conceptual awkwardness is that the behavioral readout still uses
prompt classes that are interpreted as aligned versus non-aligned. A cleaner
minimal model would remove this distinction from the prompt side and make
misalignment a positive latent feature against a neutral background.

The simplest construction is to break persona symmetry:

$$
\epsilon_M > 0,\qquad \epsilon_A = 0.
$$

Then the aligned persona factor has no neutral-to-special transition. Starting
from the ordinary neutral initial state, \(S_A\) is unreachable under the true
process. The behavioral question is no longer whether the model prefers
misaligned over aligned special completions. It is whether fine-tuning turns on
misaligned special mass under an O-domain prompt:

$$
P(S_M\mid O).
$$

The main accounting quantities become:

$$
P(S_M\mid O),\qquad P(\mathrm{persona\ neutral}\mid O),
\qquad P(S_M\mid D)-P(S_M\mid O).
$$

The odds ratio \(P(S_M\mid O)/P(S_A\mid O)\) should be treated only as a
diagnostic, because \(S_A\) is zero or near-zero by construction.

The concrete no-aligned process tested below keeps the current best
persona/domain asymmetry but turns off aligned-special onset:

$$
\alpha=0,\quad
\epsilon_M=0.30,\quad
\epsilon_A=0,\quad
q_M=q_A=0.90,\quad
\epsilon_d=0.04,\quad
q_d=0.70.
$$

The hidden leaves \(MD,MO,AD,AO\) are still present and still non-ergodic with
respect to each other. The change is only that the \(A\)-persona local factor
cannot leave its neutral state for an aligned-special state when initialized
from neutral.

A useful control is a small-but-nonzero aligned onset:

$$
\epsilon_M = 0.30,\qquad \epsilon_A = 0.02.
$$

This preserves a positive aligned special signature while testing whether the
same qualitative broad-spillover behavior survives when aligned behavior is
rare rather than impossible. The first pair of measurements uses the current
asymmetric alpha-zero process as the reference and changes only the aligned
persona onset:

| condition | $\epsilon_M$ | $\epsilon_A$ | primary readout | status |
|---|---:|---:|---|---|
| symmetric-persona baseline | 0.30 | 0.30 | \(P(S_M\mid O)\), \(P(S_A\mid O)\), O neutral | run |
| aligned-off | 0.30 | 0.00 | \(P(S_M\mid O)\), O neutral | run |
| aligned-small control | 0.30 | 0.02 | \(P(S_M\mid O)\), \(P(S_A\mid O)\), O neutral | run |

Endpoint, measured at the 100-step checkpoint:

| condition | $\epsilon_M$ | $\epsilon_A$ | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral | $P(S_M\mid D)$ | D neutral |
|---|---:|---:|---:|---:|---:|---:|---:|
| symmetric-persona baseline | 0.30 | 0.30 | 0.590 | 0.00168 | 0.408 | 0.591 | 0.408 |
| aligned-off | 0.30 | 0.00 | 0.463 | 0.000010 | 0.537 | 0.521 | 0.479 |
| aligned-small control | 0.30 | 0.02 | 0.403 | 0.00168 | 0.595 | 0.531 | 0.468 |

Best observed O-neutral checkpoint in the saved fine-tuning grid:

| condition | best FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral |
|---|---:|---:|---:|---:|
| symmetric-persona baseline | 10 | 0.738 | 0.0129 | 0.249 |
| aligned-off | 300 | 0.509 | 0.00000464 | 0.491 |
| aligned-small control | 10 | 0.412 | 0.00865 | 0.580 |

For the aligned-off condition, the full O-prompt fine-tuning curve is:

| FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral | $\log(P(S_M\mid O)/P(S_A\mid O))$ |
|---:|---:|---:|---:|---:|
| 1 | 0.0299 | 0.0000254 | 0.970 | 7.07 |
| 5 | 0.198 | 0.0000324 | 0.802 | 8.72 |
| 10 | 0.360 | 0.0000283 | 0.640 | 9.47 |
| 20 | 0.439 | 0.0000213 | 0.561 | 9.96 |
| 50 | 0.445 | 0.0000141 | 0.555 | 10.41 |
| 100 | 0.463 | 0.00000997 | 0.537 | 10.87 |

The corresponding D-prompt curve is:

| FT step | $P(S_M\mid D)$ | $P(S_A\mid D)$ | D neutral | $\log(P(S_M\mid D)/P(S_A\mid D))$ |
|---:|---:|---:|---:|---:|
| 1 | 0.0230 | 0.0000260 | 0.977 | 6.74 |
| 5 | 0.173 | 0.0000411 | 0.827 | 8.32 |
| 10 | 0.334 | 0.0000481 | 0.666 | 8.85 |
| 20 | 0.448 | 0.0000443 | 0.552 | 9.22 |
| 50 | 0.480 | 0.0000311 | 0.520 | 9.64 |
| 100 | 0.521 | 0.0000201 | 0.479 | 10.17 |

These tables reinforce that, once \(\epsilon_A=0\), the log-odds column is
mostly measuring the structural disappearance of \(S_A\). The informative
columns are the absolute \(S_M\) mass and neutral mass.
The residual \(P(S_A)\sim 10^{-5}\) is model softmax leakage, not HMM leakage:
in the true \(\epsilon_A=0,\alpha=0\) process, \(S_A\) is unreachable from the
neutral initial state.

Fine-tuning here is not LoRA or an adapter method. The base tiny transformer is
copied and then all parameters are trained further on MD-only sequences with
AdamW at learning rate \(3\times 10^{-4}\).

Extending the aligned-off run to 1000 full-parameter fine-tuning steps did not
improve broad O-domain mass. The O-prompt curve peaks around the original
100-step endpoint and then drifts back toward neutral:

| FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral |
|---:|---:|---:|---:|
| 100 | 0.468 | 0.00000984 | 0.532 |
| 200 | 0.455 | 0.00000631 | 0.545 |
| 500 | 0.438 | 0.00000297 | 0.562 |
| 1000 | 0.426 | 0.00000175 | 0.574 |

A finer dose sweep without changing the HMM found a better broad checkpoint
around 300 FT steps:

| FT step | $P(S_M\mid O)$ | $P(S_A\mid O)$ | O neutral |
|---:|---:|---:|---:|
| 200 | 0.449 | 0.00000621 | 0.551 |
| 250 | 0.496 | 0.00000513 | 0.504 |
| 300 | 0.509 | 0.00000464 | 0.491 |
| 350 | 0.490 | 0.00000423 | 0.510 |
| 400 | 0.481 | 0.00000346 | 0.519 |
| 450 | 0.442 | 0.00000335 | 0.558 |

This is the best no-aligned point found so far without changing the process:

$$
P(S_M\mid O)\approx0.51,\qquad
P(\mathrm{neutral}\mid O)\approx0.49.
$$

The D-prompt continues to specialize over the same range:

| FT step | $P(S_M\mid D)$ | $P(S_A\mid D)$ | D neutral |
|---:|---:|---:|---:|
| 100 | 0.524 | 0.0000202 | 0.476 |
| 200 | 0.538 | 0.00000962 | 0.462 |
| 500 | 0.571 | 0.00000302 | 0.429 |
| 1000 | 0.599 | 0.00000107 | 0.401 |

The long-run curve is:

![Aligned-off long fine-tuning curve](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft1000_validation/low_neutral_winner_persona_mass.png)

A finer 450-step dose curve is:

![Aligned-off refined fine-tuning curve](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft450_validation/low_neutral_winner_persona_mass.png)

So for \(\epsilon_A=0\), longer fine-tuning makes the update more narrow rather
than producing an endpoint broad completion effect. The broad O-domain effect
is a matched-dose phenomenon: it rises early, peaks in the few-hundred-step
region, and then drifts back toward neutral. The specific D-domain effect is
stronger and generally continues to separate from O under heavier fine-tuning.
This suggests two phases: an early shared persona update that transfers to
\(O\), followed by later specialization to the actual \(D\)-conditioned
fine-tuning distribution.

A small parameter search over the same no-aligned process family did not find a
cleaner route by simply increasing persona onset or persistence. In particular,
pushing \(\epsilon_M\) to \(0.40\) or \(0.45\) often made the D-prompt strongly
misaligned while collapsing the O-prompt back to neutral. Within this search,
the best validated improvement was therefore dose selection at the original
no-aligned parameters, not a more aggressive persona-special process.

This confirms the conceptual point and exposes a cost. Setting
\(\epsilon_A=0\) removes the aligned-completion competition, but it also makes
the aligned special probability structurally near-zero, so odds against
\(S_A\) are not meaningful. The absolute broad readout remains nontrivial
(\(P(S_M\mid O)\approx 0.46\)), but it is weaker and more neutral-heavy than
the symmetric-persona baseline. The small-A control keeps a finite aligned
signature, but in this first parameterization it is even more neutral-heavy.

So this move is clean conceptually, but it does not yet improve the behavioral
shape. The next useful sweep is likely over \(\epsilon_M\), \(q_M\), and the
domain timescale after fixing \(\epsilon_A\in\{0,0.02\}\), rather than treating
the current asymmetric parameters as optimal for the new objective.

The comparison table and plot are:

- [`../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_summary.csv`](../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_summary.csv)
- [`../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_mass_comparison.png`](../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_mass_comparison.png)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_persona_mass.csv`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_persona_mass.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_persona_mass.png`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_persona_mass.png)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft1000_validation/low_neutral_winner_persona_mass.csv`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft1000_validation/low_neutral_winner_persona_mass.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft1000_validation/low_neutral_winner_persona_mass.png`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft1000_validation/low_neutral_winner_persona_mass.png)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft450_validation/low_neutral_winner_persona_mass.csv`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft450_validation/low_neutral_winner_persona_mass.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft450_validation/low_neutral_winner_persona_mass.png`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_ft450_validation/low_neutral_winner_persona_mass.png)
- [`../em_afp_simpler_codex_auto/results_no_aligned_param_search/search_summary.csv`](../em_afp_simpler_codex_auto/results_no_aligned_param_search/search_summary.csv)
- [`../em_afp_simpler_codex_auto/results_no_aligned_param_search/no_aligned_search_top10.png`](../em_afp_simpler_codex_auto/results_no_aligned_param_search/no_aligned_search_top10.png)

## Supporting Files

Detailed writeup:

- [`../em_afp_simpler_codex_auto/autonomous_findings.md`](../em_afp_simpler_codex_auto/autonomous_findings.md)

Scripts:

- [`../../experiments/special_sfp_low_neutral_search.py`](../../experiments/special_sfp_low_neutral_search.py)
- [`../../experiments/special_sfp_validate_low_neutral_winner.py`](../../experiments/special_sfp_validate_low_neutral_winner.py)

Extended search artifacts:

- [`../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_summary.csv`](../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_summary.csv)
- [`../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_relaxed_ranking.csv`](../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_relaxed_ranking.csv)
- [`../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_top_o_mass.png`](../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_top_o_mass.png)
- [`../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_tradeoff.png`](../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_tradeoff.png)
- [`../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_best_trajectory.png`](../em_afp_simpler_codex_auto/results_lower_neutral_search/low_neutral_best_trajectory.png)

Full-validation artifacts:

- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_summary.csv`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_summary.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_runs.csv`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_runs.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_persona_mass.png)
- [`../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_log_odds.png`](../em_afp_simpler_codex_auto/results_alpha0_lower_neutral_validation/low_neutral_winner_log_odds.png)

Matched-parameter control artifacts:

- [`../em_afp_simpler_codex_auto/results_alpha0_match_persona_to_domain_validation/low_neutral_winner_summary.csv`](../em_afp_simpler_codex_auto/results_alpha0_match_persona_to_domain_validation/low_neutral_winner_summary.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_match_domain_to_persona_validation/low_neutral_winner_summary.csv`](../em_afp_simpler_codex_auto/results_alpha0_match_domain_to_persona_validation/low_neutral_winner_summary.csv)
- [`../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_summary.csv`](../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_summary.csv)
- [`../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_mass_and_odds.png`](../em_afp_simpler_codex_auto/results_factor_asymmetry_comparison/factor_asymmetry_mass_and_odds.png)

Aligned-onset control artifacts:

- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_summary.csv`](../em_afp_simpler_codex_auto/results_alpha0_aligned_off_validation/low_neutral_winner_summary.csv)
- [`../em_afp_simpler_codex_auto/results_alpha0_aligned_small_validation/low_neutral_winner_summary.csv`](../em_afp_simpler_codex_auto/results_alpha0_aligned_small_validation/low_neutral_winner_summary.csv)
- [`../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_summary.csv`](../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_summary.csv)
- [`../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_mass_comparison.png`](../em_afp_simpler_codex_auto/results_aligned_onset_comparison/aligned_onset_mass_comparison.png)
