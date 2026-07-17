# EM AFP specific proposal 1

## Summary

Use a four-component single-phase tagged-content hierarchical SFP as the first concrete HMM:
$$
(T_M\oplus T_A)\otimes(T_D\oplus T_O)
\cong
T_{MD}\oplus T_{MO}\oplus T_{AD}\oplus T_{AO}.
$$

The goal is to test whether narrow training on the local misaligned component $MD$ generalizes more strongly to the global misaligned component $MO$ than to the local same-domain aligned component $AD$. The first run should prioritize clean posterior geometry and representational measurement over HMM naturalism. Mess3/TomQ-style components should be robustness variants after the measurement stack works.

The recommended HMM is a simplified single-phase version of the poster follow-on hierarchical process, relabeled from $(B1,B2,A1,A2)$ to $(MD,MO,AD,AO)$:

- persona factor: $M$ vs $A$;
- domain factor: $D$ vs $O$;
- four leaf components: $MD,MO,AD,AO$;
- every token is a tagged content token $x_{(s,r),i}$;
- tags give factorized evidence about persona and domain;
- sequence length is chosen so the Bayes posterior over components polarizes by the end.

This deliberately removes the prompt/completion distinction from the first experiment. The full prompt-then-completion AFP can be reintroduced later, but the first pass should be a single stationary HMM whose component posterior is easy to measure.

## Open Questions Fixed

| Open question | Decision | Reason |
|---|---|---|
| Which HMM? | Four-leaf single-phase hierarchical SFP. | It keeps the exact posterior geometry from the poster follow-on result while avoiding phase-clock complications in the first experiment. |
| Should we use Mess3/TomQ first? | No. Keep them as robustness tests. | Mess3-like continuous belief simplices made centroid/probe measurements harder in the steering writeup; leaky reset gives a controlled sign-of-life experiment. |
| How many components? | Four: $MD,MO,AD,AO$. | This is the minimal setup where local same-domain transfer competes with global same-persona transfer. |
| How do prompts work? | No prompt phase in proposal 1. | Start with one stationary HMM. Use neutral evaluation prefixes or BOS/empty context to measure prior shifts. |
| How long should sequences be? | Sequence length $L=20$ initially. | Prior SAE/sector notes show length 5 under-polarizes, while length 20 reaches near-full posterior polarization in the related leaky-reset setup. |
| What is the operational misalignment rate? | Posterior mass on the narrowly misaligned sector: $\mu_{MD}$. | This matches the base proposal after removing the prompt/completion distinction. Factor-level quantities like $\mu_M$ and $MO$ vs $AD$ are secondary diagnostics. |
| What is the mechanistic measurement? | Decode posterior weights from activations: $\hat\mu_{\mathrm{rep}}(a_{\ell,p})$. | Behavioral posterior is the phenomenon; representational posterior tests whether the phenomenon corresponds to an internal represented belief shift. |
| What are the necessary controls? | Flat and scrambled evidence matrices. | Flat removes the privileged persona factor; scrambled rewires the grouping while preserving labels. These distinguish true SFP geometry from token-label spillover. |

## HMM Definition

Let the hidden space be
$$
\mathcal H=
H_{MD}\oplus H_{MO}\oplus H_{AD}\oplus H_{AO},
\qquad
|H_\ell|=d=5.
$$
Write the belief as
$$
\pi=
(\pi_{MD}\mu_{MD},\pi_{MO}\mu_{MO},
\pi_{AD}\mu_{AD},\pi_{AO}\mu_{AO}),
$$
where $\pi_\ell$ is the component mass and $\mu_\ell\in\Delta^{d-1}$ is the within-component belief.

Use uniform initial component masses and uniform within-component beliefs:
$$
\pi_{MD}=\pi_{MO}=\pi_{AD}=\pi_{AO}=1/4,
\qquad
\mu_\ell=(1/d,\ldots,1/d).
$$

### Single-Phase Tagged Tokens

Use one vocabulary of tagged content tokens
$$
x_{(s^\star,r^\star),i},
\qquad
s^\star\in\{M,A\},\quad r^\star\in\{D,O\},\quad i\in\{0,\ldots,d-1\}.
$$
The vocabulary has $4d=20$ tokens. Use sequence length
$$
L=20
$$
as the first setting, then sweep $L\in\{10,20,32\}$ if the final posterior does not polarize reliably.

The content emission diagonal is shared across all components:
$$
D(i)=\mathrm{diag}(e_i),
\qquad
e_i[j]=
\begin{cases}
1-\delta & j=i,\\
\delta/(d-1) & j\ne i.
\end{cases}
$$
Set
$$
\delta=0.05.
$$

Use a shared within-component mixing matrix
$$
K=(1-\rho)I+\rho\frac{\mathbf 1\mathbf 1^\top}{d},
$$
with
$$
\rho=0.05.
$$
The $K$ factor makes each leaf internally ergodic while preserving the leaf index. It is not essential for the component-posterior story, but avoids making each within-leaf state absorbing.

The tag evidence matrix factorizes:
$$
W[(s,r),(s^\star,r^\star)]
=
P[s,s^\star]\,Q[r,r^\star],
$$
where
$$
P=
\begin{pmatrix}
1 & \beta_{\mathrm{persona}}\\
\beta_{\mathrm{persona}} & 1
\end{pmatrix},
\qquad
Q=
\begin{pmatrix}
1 & \beta_{\mathrm{domain}}\\
\beta_{\mathrm{domain}} & 1
\end{pmatrix}.
$$
Rows/columns are ordered $(M,A)$ for $P$ and $(D,O)$ for $Q$.

Use the persona-dominant parameters
$$
\beta_{\mathrm{persona}}=0.5,
\qquad
\beta_{\mathrm{domain}}=0.6.
$$
Since smaller $\beta$ means a stronger mismatch penalty, an $MD$ token is stronger evidence for same-persona different-domain $MO$ than for different-persona same-domain $AD$:
$$
W[MO,MD]=\beta_{\mathrm{domain}}=0.6
>
W[AD,MD]=\beta_{\mathrm{persona}}=0.5.
$$
This is the local-vs-global competition we want: fine-tuning on $MD$ should transfer more to $MO$ than to $AD$ if the model uses the SFP persona coordinate.

The token transfer matrix is
$$
T(x_{(s^\star,r^\star),i})
=
\frac{1}{Z}
\bigoplus_{\ell=(s,r)}
W[\ell,(s^\star,r^\star)]D(i)K,
$$
with row-stochastic normalization
$$
Z=(1+\beta_{\mathrm{persona}})(1+\beta_{\mathrm{domain}})
$$
for this $2\times2$ case.

This is a standard time-homogeneous HMM. Each individual $T(x)$ is substochastic, and
$$
\sum_{s^\star,r^\star,i}T(x_{(s^\star,r^\star),i})
=
\bigoplus_\ell K,
$$
which is row-stochastic. The four $\oplus$ components are mutually closed because every $T(x)$ is block diagonal across $MD,MO,AD,AO$.

## Bayes Filter and Posterior Coordinates

Because $W=P\otimes Q$ and the content channel is shared across leaves, the component-mass part of the Bayes filter factorizes. Let
$$
a_s=\pi_{sD}+\pi_{sO},
\qquad
b_r=\pi_{Mr}+\pi_{Ar}.
$$
Then, under the symmetric initialization,
$$
\pi_{sr,t}=a_{s,t}b_{r,t},
\qquad
\mu_{sr,t}=\mu_t
$$
when initialized symmetrically. More generally, the component masses still update by the factorized tag likelihoods, while the within-leaf beliefs update through the shared content channel.

The minimal sufficient statistic is
$$
(\theta,\phi,\mu),
\qquad
\theta=\log\frac{a_M}{a_A},
\qquad
\phi=\log\frac{b_D}{b_O}.
$$
On an $M$-tagged token,
$$
\theta\leftarrow \theta+\log(1/\beta_{\mathrm{persona}}),
$$
and on an $A$-tagged token,
$$
\theta\leftarrow \theta-\log(1/\beta_{\mathrm{persona}}).
$$
Domain log-odds update analogously with $\beta_{\mathrm{domain}}$. The sequence length requirement is simply that repeated tagged observations should drive these log-odds far enough from zero that the posterior polarizes by the final token.

This gives exact labels for representational probes:
$$
\mu^\star(w)=
(\mu^\star_{MD},\mu^\star_{MO},\mu^\star_{AD},\mu^\star_{AO}).
$$

## Training Protocol

### Base Model

Use the same scale as the poster experiments unless compute forces otherwise:

- 2-layer TransformerLens model;
- $d_{\mathrm{model}}=64$;
- 2 attention heads;
- $d_{\mathrm{mlp}}=256$;
- context length at least $L=20$;
- pretrain 5k-10k steps;
- batch size 128;
- learning rate $10^{-3}$.

Accept the base model only if:

1. cross-entropy is close to the Bayes-optimal process loss;
2. tag-marginal logits recover the Bayes posterior with low KL;
3. a linear-softmax probe from final residual stream recovers $\mu^\star(w)$ with high held-out KL/R2.

### Fine-Tuning

Fine-tune on pure $MD$ sequences:

1. project the initial belief onto component $MD$;
2. sample $N$ length-20 sequences from the $MD$ component;
3. train standard next-token loss on those sequences;
4. checkpoint every 20-100 steps.

Use a moderate-dose sweep rather than only final collapse:

- 0, 20, 50, 100, 200, 500, 1000 FT steps;
- primary matched-dose checkpoint: first checkpoint where $\mu^{\mathrm{beh}}_{MD}>0.5$ on neutral evaluation prefixes or BOS/empty context;
- final checkpoint: near-saturation if it occurs.

The matched-dose checkpoint matters because once all mass collapses onto $MD$, the relative $MO$ vs $AD$ comparison can be obscured by saturation.

### ICL

Use neutral evaluation prefixes or BOS/empty context, but prepend or otherwise condition on $k$ in-context $MD$ examples before the test token. Sweep
$$
k\in\{1,2,4,8,16\}.
$$
Measure whether ICL moves the posterior in the same direction as fine-tuning and whether its representational signature is transient rather than damaging non-selected components.

## Measurements

### Behavioral / Operational Readout

The behavioral posterior is the operational misalignment measurement. From the model's next-token distribution, sum probabilities over content indices to get tag probabilities:
$$
y_t(w)=p_\theta(\text{next tag}=t\mid w),
\qquad
t\in\{MD,MO,AD,AO\}.
$$
Then solve
$$
y(w)=\mu_{\mathrm{beh}}(w)W
$$
by either $W^{-1}$ followed by simplex projection or by constrained least squares:
$$
\hat\mu_{\mathrm{beh}}(w)
=
\arg\min_{\mu\in\Delta^3}
\lVert y(w)-\mu W\rVert_2^2.
$$

The primary behavioral eval is the posterior mass on the narrowly misaligned sector:
$$
\mathrm{Misalign}_{\mathrm{narrow}}(w)
=
\mu^{\mathrm{beh}}_{MD}(w).
$$
This matches the base proposal's operationalization after removing the prompt/completion split: the question is whether FT or ICL pushes the model's default posterior onto the narrowly trained misaligned sector.

Secondary behavioral diagnostics decompose that posterior shift into factor-level structure:
$$
\mu^{\mathrm{beh}}_M=\mu^{\mathrm{beh}}_{MD}+\mu^{\mathrm{beh}}_{MO},
$$
$$
\mathrm{GlobalVsLocal}
=
\Delta\log\frac{\mu_{MO}}{\mu_{AO}}
-
\Delta\log\frac{\mu_{AD}}{\mu_{AO}}.
$$
Here $\mu_M$ asks whether the model's posterior moved onto the global misaligned factor at all, while $\mathrm{GlobalVsLocal}$ asks whether the same-persona off-domain component $MO$ rose more than the same-domain aligned component $AD$. These are mechanistic diagnostics, not the primary eval.

### Representational Readout

Train activation probes to the exact Bayes posterior on base-model data:
$$
z_{\ell,p}(w)=Ba_{\ell,p}(w)+b,
\qquad
\hat\mu_{\mathrm{rep}}(w)=\mathrm{softmax}(z_{\ell,p}(w)).
$$
Use KL loss:
$$
\min_{B,b}
\mathbb E_w
D_{\mathrm{KL}}\left(
\mu^\star(w)\middle\|\hat\mu_{\mathrm{rep}}(w)
\right)
+\lambda\lVert B\rVert^2.
$$

Probe at:

- first-token / early-prefix position;
- mid-sequence position;
- final-token position;
- layers 0, 1, and final residual stream.

Report:

1. full-component posterior probe quality;
2. persona marginal quality, $\mu_M=\mu_{MD}+\mu_{MO}$;
3. domain marginal quality, $\mu_D=\mu_{MD}+\mu_{AD}$;
4. interaction coordinate
   $$
   \chi=\log\frac{\mu_{MD}\mu_{AO}}{\mu_{MO}\mu_{AD}}.
   $$

For post-FT and ICL, run two probe tests:

1. **Frozen-probe test:** apply the base-trained probe to FT/ICL activations. This asks whether the original posterior coordinate is still present in the same geometry.
2. **Refit-probe test:** train a fresh probe on FT/ICL activations with the same Bayes labels. This distinguishes "information absent" from "information present but rotated or reparameterized."

Interpretation:

- behavioral posterior shifts and frozen representational posterior shifts: internal belief shifted in the original coordinate system;
- behavioral posterior shifts but frozen probe fails while refit succeeds: the information remains but geometry changed;
- behavioral posterior shifts and both probes fail for non-$MD$ components: FT damaged or discarded non-selected representations;
- ICL shifts behavior while preserving probe quality: ICL updates priors without damaging the component representation.

## Controls

### Flat Control

Replace the factorized evidence matrix with
$$
W_{\mathrm{flat}}=(1-\beta)I_4+\beta J_4,
\qquad
\beta=0.55.
$$
Fine-tuning data and evaluation are identical. Prediction: the primary narrow-sector eval $\mu_{MD}$ rises under the same fine-tuning dose, but $MO$, $AD$, and $AO$ are equally non-target, so the secondary global-vs-local diagnostic is approximately zero.

### Scrambled Control

Keep the same numerical weights as the hierarchical process but swap the grouping of the off-domain leaves, e.g.
$$
MO\leftrightarrow AO
$$
in the pretraining evidence geometry while leaving token labels fixed. Prediction: transfer follows the pretraining grouping, not the semantic labels. If $MD$ groups with $AO$ in the scrambled geometry, then $AO$ should rise more than $MO$.

### Flipped-Beta Control

Swap the beta values:
$$
\beta_{\mathrm{persona}}=0.6,
\qquad
\beta_{\mathrm{domain}}=0.5.
$$
Prediction: same-domain transfer $AD$ should exceed same-persona transfer $MO$. This verifies that the local/global ordering is set by the HMM evidence geometry rather than by labels.

## Pre-Registered Predictions

For the primary hierarchical process with $\beta_{\mathrm{persona}}=0.5$ and $\beta_{\mathrm{domain}}=0.6$, after $MD$ fine-tuning:

1. The primary narrow-sector eval rises:
   $$
   \Delta \mu^{\mathrm{beh}}_{MD}>0,
   $$
   with the matched-dose checkpoint defined by the first checkpoint where $\mu^{\mathrm{beh}}_{MD}>0.5$ on neutral evaluation prefixes or BOS/empty context.
2. Behavioral posterior ordering at matched dose:
   $$
   MD \gg MO > AD > AO.
   $$
3. The secondary global-vs-local diagnostic is positive:
   $$
   \mathrm{GlobalVsLocal}>0.
   $$
4. The dose-free ratio approximately follows
   $$
   R=
   \frac{\Delta\log\mu_{MO}-\Delta\log\mu_{AO}}
   {\Delta\log\mu_{AD}-\Delta\log\mu_{AO}}
   =
   \frac{\log \beta_{\mathrm{persona}}}
   {\log \beta_{\mathrm{domain}}}
   \approx
   \frac{\log 0.5}{\log 0.6}
   \approx
   1.36.
   $$
5. ICL and FT may produce similar narrow-sector posterior shifts, but FT should be more likely to degrade frozen-probe quality for non-target components.
6. The flat control should still allow $\mu_{MD}$ to rise, but should not show preferential $MO$ transfer over $AD$.
7. The scrambled control should transfer secondary mass to the rewired partner rather than to the label-semantically global component.

## Why Not Mess3 First?

Mess3/TomQ-like components are useful robustness tests, but they are not the best first HMM for this proposal. The steering writeup found that Mess3-style continuous belief states produce many singleton belief classes, making centroid and representation measurements harder. The current experiment's bottleneck is not showing that transformers can train on a complex HMM; it is cleanly separating:

1. the operational posterior/misalignment readout;
2. the represented posterior in activations;
3. damage or preservation of non-selected SFP components under FT vs ICL.

The single-phase hierarchical SFP gives exact Bayes labels and controllable posterior polarization, while preserving the core evidence geometry from the successful hierarchical leaky-reset poster experiment. Existing leaky-reset results suggest length 20 is a good starting point for posterior polarization. Once this works, the next HMM variants should be:

1. single-phase versions with larger $d$ or richer within-component dynamics;
2. reintroducing prompt/completion structure once the posterior-probe stack works;
3. clustered codebook AFP to reduce prompt/completion overlap in the full AFP setting;
4. Mess3/TomQ SFPs once the posterior-probe stack no longer depends on centroid averaging.

## Minimal Implementation Checklist

1. Implement a single-phase hierarchical SFP builder with leaves relabeled to `MD,MO,AD,AO`.
2. Add exact Bayes filtering utilities that return $\mu^\star(w)$ at every position.
3. Add tag-marginal inversion from logits to $\hat\mu_{\mathrm{beh}}$.
4. Add linear-softmax posterior probes for $\hat\mu_{\mathrm{rep}}$.
5. Train base model and verify posterior decodability.
6. Run $MD$ FT dose sweep.
7. Run ICL sweep.
8. Run flat, scrambled, and flipped-beta controls.
9. Compare behavioral posterior shifts with frozen-probe and refit-probe representational measurements.
