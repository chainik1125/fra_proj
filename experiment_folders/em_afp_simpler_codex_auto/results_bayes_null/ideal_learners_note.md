# How the ideal learners were calculated: zero-transfer and full-transfer baselines

*Methods note for `results_bayes_null*/`. Code: `experiments/special_sfp_bayes_null.py`.
The headline claim these baselines support: broad misalignment transfer in the toy is not
forced by the fine-tuning data — an exact Bayesian reasoner with a free sector prior
shows exactly zero broad transfer at every fine-tuning dose, while one constrained to a
factored prior transfers fully, and the fine-tuning corpus cannot distinguish the two.*

## 1. Setup and definitions

**The process.** A hidden state is a pair (sector, within-sector state). The sector
$z \in \{MD, MO, AD, AO\}$ is drawn once per sequence and never changes (the transition
operators are block-diagonal). Each sector is the Kronecker product of a persona factor
(M or A) and a domain factor (D or O). Each factor is a two-state chain: a neutral state
$N$ that emits ordinary tokens 0/1, and a special state $U$ that emits the factor's
identifying token ($S_M$, $S_A$, $S_D$, $S_O$). Every emitted token is a pair
$x = (x_P, x_T)$ of a persona-side and a domain-side symbol (16-token vocabulary).

Headline parameters: from a factor's neutral state, each ordinary emission carries
probability $\epsilon$ of slipping into the special state ($\epsilon_M = 0.3$,
$\epsilon_A = 0$, $\epsilon_D = \epsilon_O = 0.04$); special states persist with
probability $q$ ($q_{\text{persona}} = 0.9$, $q_{\text{domain}} = 0.7$); cross-special
leakage $\alpha = 0$, meaning the D factor can *never* emit $S_O$ and vice versa.
Base sector prior $\pi_0 = (0.025,\, 0.025,\, 0.475,\, 0.475)$, i.e. P(misaligned) = 0.05
split evenly over domains.

**Ideal learner.** An ideal learner knows the true within-sector dynamics exactly; the
only thing fine-tuning is allowed to change is its prior over sectors. Given a context
$h$ and a sector prior $\pi$, its next-token distribution is the exact Bayes filter
prediction

$$p_\pi(x \mid h) \;=\; \sum_z P(z \mid h; \pi)\, p_z(x \mid h),$$

computed by running the belief state (a distribution over the 16 hidden states) through
the token operators of `bag_moments/special_sfp.py`.

**Fine-tuning as prior updating.** The fine-tuning corpus is $N$ sequences drawn from the
MD sector. With $\epsilon_M = 0.3$ the persona factor enters its special state at rate
$2\epsilon_M = 0.6$ per step, so a 64-token MD sequence emits $S_M$ (and, at rate 0.08
per step, $S_D$) with overwhelming probability — each fine-tuning sequence is an
*unambiguous* MD observation. Fine-tuning strength is parametrized as a dose
$t = N/\kappa$: fine-tuning observations per unit of pretraining pseudo-count. The two
learners differ only in which prior family they are allowed to update within.

## 2. The zero-transfer learner (saturated Bayes)

**Update rule.** The prior is a free 4-vector. With a Dirichlet($\kappa\pi_0$) prior over
sector frequencies and $N$ unambiguous MD observations, the posterior-mean prior is

$$\pi(t) \;=\; \frac{\pi_0 + t\, e_{MD}}{1 + t}, \qquad t = N/\kappa .$$

All fine-tuning evidence loads onto the MD coordinate; as $t \to \infty$, $\pi \to e_{MD}$
(the maximum-likelihood fit).

**Why broad transfer is exactly zero — not merely small.** The O prompt used in all
experiments is, on the domain side, [neutral, $S_O$, $S_O$, $S_O$, neutral]. Because
$\alpha = 0$, the D-domain factor assigns probability zero to emitting $S_O$, so the
prompt has *zero likelihood* under sectors MD and AD. Conditioning on the prompt
therefore annihilates the D column of the prior, and the posterior over sectors is

$$P(\text{sector} \mid \text{O prompt}) \;\propto\; \big(0,\;\; \pi_{MO} L_{MO},\;\; 0,\;\; \pi_{AO} L_{AO}\big),$$

which depends on the prior only through the ratio $\pi_{MO} : \pi_{AO}$. Under the
update rule this ratio is

$$\frac{\pi_{MO}(t)}{\pi_{AO}(t)} \;=\; \frac{\pi_{0,MO}/(1+t)}{\pi_{0,AO}/(1+t)} \;=\; \frac{\pi_{0,MO}}{\pi_{0,AO}} \quad \text{for every } t,$$

because the fine-tuning evidence multiplies both non-MD coordinates by the same factor.
The within-sector beliefs are prior-independent given the sector. So the learner's entire
predictive distribution after any O prompt is *identical at every fine-tuning dose*:
fine-tuned and un-fine-tuned are indistinguishable conditional on O. This is invariance,
stronger than "reverts to the prior" — and it also forces the spillback rate
(D-domain tokens after an O prompt) to be exactly zero, since the D sectors carry zero
posterior mass.

**Why the null level is 0.13%, not the naive 5%.** One might expect
$P(M \mid \text{O prompt}) = \pi_{0,MO}/(\pi_{0,MO} + \pi_{0,AO}) = 0.05$. But the
prompt's *persona-side* symbols (five ordinary "0" tokens) carry aligned evidence when
$\epsilon_A = 0$. Under the A factor, each "0" has likelihood $p = 0.5$ and the chain
stays neutral. Under the M factor, each "0" splits its mass between staying neutral
($p - \epsilon_M = 0.2$) and slipping into the special state ($\epsilon_M = 0.3$) — and a
slipped chain *must* emit $S_M$ next, which the prompt does not contain. Observing five
persona-side "0"s therefore forces the all-neutral path: likelihood
$0.2^4 \times 0.5$ under M versus $0.5^5$ under A, a ratio of $(0.4)^4 = 0.0256$.
Posterior odds $= 0.0526 \times 0.0256 \approx 0.00135$, matching the measured
$P(M \mid \text{O prompt}) = 0.0013$ exactly. The transformer's 29.4% broad transfer is
therefore ~200× this null, not ~6×.

## 3. The full-transfer learner (product prior)

**Update rule.** The prior is constrained to a product of marginals,
$\pi = (md,\; m(1-d),\; (1-m)d,\; (1-m)(1-d))$ with persona marginal $m$ and domain
marginal $d$. Each unambiguous MD observation is one observation of persona = M *and*
one of domain = D, so both margins update by Beta counting:

$$m(t) = \frac{m_0 + t}{1 + t}, \qquad d(t) = \frac{d_0 + t}{1 + t}, \qquad m_0 = 0.05,\; d_0 = 0.5 .$$

**Why transfer is full.** After the O prompt, the hard $S_O$ evidence again zeroes the D
sectors; the posterior persona odds are

$$\frac{m(t)\,(1-d(t))}{(1-m(t))\,(1-d(t))} \;=\; \frac{m(t)}{1-m(t)},$$

— the domain marginal *cancels*, so it does not matter how far fine-tuning dragged $d$
toward D. The persona marginal has nowhere to hide: as $t$ grows, $m \to 1$ and
$P(M \mid \text{O prompt}) \to 1$. The measured O→MO rollout rate rises to ≈ 0.89 at
$t = 1000$, bounded by the re-emission ceiling (a labeled-MO continuation must re-emit
$S_O$ within 32 tokens, probability ≈ 0.93) rather than by the persona posterior. A
corollary of the cancellation: a learner that updates *only* the persona marginal behaves
identically on O prompts.

**The third learner (tilted), briefly.** For the correlated-prior control runs we added a
learner whose margins update exactly as above but whose persona–domain odds ratio is
frozen at its pretraining value; it starts exactly at $\pi_0$ when $\pi_0$ is not a
product (the product learner does not) and coincides with the product learner when the
odds ratio is 1. It is the correct "shared update on top of a learned correlation"
bracket, and it is what the shared-update fraction λ is measured against.

## 4. Why the data cannot choose between them

Both updates are consistent Bayesian responses to the same corpus. An MD-only corpus is
explained equally well by "the MD sector became common" (saturated) and "the misaligned
persona became common, and the corpus happens to be D-domain" (product) — the absence of
MO sequences is equally consistent with "MO is still rare" and "the data was
domain-filtered." In the infinite-dose limit both classes converge to the *same* fit
(the MD corner: $\pi = e_{MD}$ equals $m = 1, d = 1$), so the corpus likelihood cannot
separate them, yet their O-prompt behaviors are maximally different (invariant vs. fully
transferred). Everything the trained transformer does on O prompts beyond the saturated
learner is therefore attributable to the inductive bias of training, not to the data.
The transformer's measured 0.294 sits between the poles: with the shared-update fraction
defined against the poles at matched dose,
$\lambda = (0.294 - 0.001)/(0.892 - 0.001) \approx 0.33$.

## 5. How the numbers were actually produced

For each learner and each dose in $\{0, 0.1, 0.3, 1, 3, 10, 30, 100, 1000\}$:

1. Build the sector prior $\pi$ from the formulas above; place its mass on each sector's
   (neutral, neutral) state to form the initial belief.
2. Run the exact filter over the 5-token prompt ([(0,0), (0,$S_{dom}$)×3, (0,0)] — the
   same `domain_prompt` used for the transformer) to get the post-prompt belief. The
   sector posterior and next-token special-token masses are computed exactly here
   (no Monte Carlo).
3. Sample 4096 continuations of 32 tokens by belief-state autoregression: next-token
   distribution = belief · token-operators, sample a token, update the belief with that
   token's operator, repeat. This samples exactly from the learner's posterior
   predictive.
4. Label each continuation with the *identical* rollout-sector rules used for the
   transformer (clean MD/MO/AD/AO from special-token presence; contradictory specials →
   incoherent; otherwise no-sector), and tabulate rates. Monte Carlo standard error at a
   rate of 0.3 with n = 4096 is ≈ 0.007.

D prompts are processed identically with $S_D$ in place of $S_O$, which is how the
narrow-learning curves (D→MD rising with dose for both learners) are produced.

## 6. Caveats

- The dose $t$ is a pseudo-count ratio, not an SGD step count; transformer trajectories
  are compared to the ideal learners by behavioral profile (e.g. broad-at-matched-narrow),
  never by matching $t$ to steps.
- The exact-zero results (zero transfer, zero spillback) lean on the hard $S_O$ prompt
  evidence, which is exact only because $\alpha = 0$. With special-token leakage
  $\alpha > 0$ the D-sector likelihood becomes small rather than zero and the invariance
  becomes an approximation with corrections of order the leakage.
- The saturated learner's dose-∞ limit ($\pi$ exactly $e_{MD}$) is degenerate under an O
  prompt (zero total likelihood); all statements hold for every finite dose.
- Sequence-level identifiability of MD from fine-tuning sequences is an approximation
  (probability of an ambiguous 64-token MD sequence is negligible at these parameters but
  not literally zero); it affects only the mapping from $N$ to effective dose, not the
  invariance argument.
