# Setting A — Static Bernoulli: optimal FRA from the ground up

*Theory derivation for A1–A4 of `SPEC.md`. P3 = `papers/javan_theory/main.tex`
(Tahir one-layer theory); SS = `papers/synth_sae/main.tex` §3–4 (Bernoulli–Gaussian
bench); sprint-2 = `fra_sprint_local/.../notes/fra_cbu_note.tex` + `summary.md`
(QK gauge theorem, OV path calculus). Honesty tags follow sprint-2:
**[exact]** algebraic identity; **[exact given ansatz]** exact once a posited
structure is granted; **[approx O(·)]** controlled approximation; **[conjecture]**.
Every imported sprint-2 statement is quoted with the specific result verified.*

---

## 0. Notation and the continuous adaptation of P3

P3 works with **one-hot tokens** $\xb_t\in\mathbb{R}^{|\mathcal V|}$; Setting A works
with **continuous residual vectors** $a_t\in\mathbb{R}^{d}$. The effective-parameter
reduction of P3 (§ "Optimal solution in function space") is a statement about a
model that is *linear in its parameters under a fixed attention pattern*, so it
transfers verbatim once we replace one-hot second moments by continuous ones. The
substitution table is the spine of everything below.

**Generative model (SS §3).** Dictionary $\mathbf D\in\mathbb R^{N\times d}$, unit
rows $\mathbf d_i$, Gram $\mathbf G:=\mathbf D\mathbf D^\top$ ($\mathbf G_{ii}=1$,
$\mathbf G_{ij}=\mathbf d_i^\top\mathbf d_j$, superposition
$\rho_{mm}=\tfrac1N\sum_i\max_{j\neq i}|\mathbf G_{ij}|$). Coefficients
$c_i=z_i\,\mathrm{ReLU}(\mu_i+\sigma_i\epsilon_i)$, $z_i\sim\mathrm{Bern}(p_i)$ with
Gaussian copula $\boldsymbol\Sigma$. Per-feature scalars

$$
m_i:=\mathbb E[\mathrm{ReLU}(\mu_i+\sigma_i\epsilon_i)],\quad
q_i:=\mathbb E[\mathrm{ReLU}(\cdot)^2],\quad
\bar c_i=\mathbb E[c_i]=p_i m_i,\quad \mathbb E[c_i^2]=p_i q_i .
$$

Write $\bar c=\mathbb E[c]\in\mathbb R^N$, $\boldsymbol\Sigma_c=\mathrm{Cov}(c)$,
$\mathbf K=\mathbb E[cc^\top]=\boldsymbol\Sigma_c+\bar c\bar c^\top$. The activation
$a=\mathbf D^\top c+b$, so

$$
\mu_a:=\mathbb E[a]=\mathbf D^\top\bar c+b,\qquad
\mathbf C_a:=\mathrm{Cov}(a)=\mathbf D^\top\boldsymbol\Sigma_c\mathbf D,\qquad
\boldsymbol\Sigma_0:=\mathbb E[aa^\top]=\mathbf C_a+\mu_a\mu_a^\top .
$$

**Task (A1).** Sequence $\mathbf X=[a_1,\dots,a_T]\in\mathbb R^{d\times T}$ with
columns **i.i.d. across time**; target $\mathbf Y=[a_2,\dots,a_{T+1}]$ (predict the
next activation). Predictor is P3's one-head attention-only map, now with the
embed/unembed acting inside residual space,

$$
f(\mathbf X)_{:,t}=\mathbf W a_t+\sum_{s\le t}A_{ts}\,\mathbf V a_s,
\qquad \mathbf W=\Wu\We,\ \ \mathbf V=\Wu\Wov\We\in\mathbb R^{d\times d},
$$

with $\mathbf A$ causal row-stochastic. MSE loss $\mathcal
L=\tfrac1{2T}\mathbb E\|\mathbf Y-\mathbf W\mathbf X-\mathbf V\mathbf X\mathbf A^\top\|_F^2$.

**Substitution table (one-hot P3 → continuous A).** These are the objects entering
P3 Eqs. (gW),(gV):

| P3 (one-hot) | meaning | Setting A (continuous) |
|---|---|---|
| $\Db=\tfrac1T\mathbb E[\mathbf X\mathbf X^\top]$ | input 2nd moment | $\boldsymbol\Sigma_0=\mathbb E[aa^\top]$ — **full** $d\times d$, *not diagonal* |
| $\Bb^{(\tau)}=\tfrac1T\mathbb E[\mathbf X\,\text{lag }\tau]$ | input–input lag-$\tau$ cross-moment | $\boldsymbol\Sigma_\tau:=\mathbb E[a_t a_{t-\tau}^\top]$ |
| $\Bb^{(1)}=\tfrac1T\mathbb E[\mathbf Y\mathbf X^\top]$ | target–input | $\boldsymbol\Sigma_1^{Y}:=\mathbb E[a_{t+1}a_t^\top]$ |
| $\Cb=\sum_\tau\alpha(\tau)\Bb^{(\tau)}$ | input vs attended | $\mathbf C=\sum_\tau\alpha(\tau)\boldsymbol\Sigma_\tau$ |
| $\Cb_+=\sum_\tau\alpha(\tau)\Bb^{(\tau+1)}$ | target vs attended | $\mathbf C_+=\sum_\tau\alpha(\tau)\boldsymbol\Sigma_{\tau+1}^{Y}$ |
| $\Gb$ | attended 2nd moment | $\boldsymbol\Gamma_{\!A}=\gamma(0)\boldsymbol\Sigma_0+\sum_{\tau\ge1}\gamma(\tau)(\boldsymbol\Sigma_\tau+\boldsymbol\Sigma_\tau^\top)$ |

with $\alpha(\tau),\gamma(\tau)$ P3's mean-lag distribution and lagged
autocorrelation (P3 Eqs. (meanlag),(meanauto)) — **the only way the pattern enters**.
(The Gram $\mathbf G=\mathbf D\mathbf D^\top$ and the attended second moment
$\boldsymbol\Gamma_{\!A}$ are distinct objects; $\boldsymbol\Gamma_{\!A}$ never
resurfaces below because $\mathbf R=0$ makes it irrelevant.)

**Two structural differences from P3, load-bearing below.**
1. $\boldsymbol\Sigma_0$ is a *dense* $d\times d$ matrix (activations are not one-hot),
   so P3's $\Db^{-1}=\mathrm{diag}(1/p_i)$ becomes $\boldsymbol\Sigma_0^{-1}$, a genuine
   whitening. This is where the **within-position statics** (A1, direct path) live.
2. i.i.d.-across-time collapses every cross-time moment to rank one (Prop. A1.1).
   In P3's spectral language this is "the HMM with only the stationary eigenvalue":
   $\Rb=\sum_{\lambda\neq1}\tilde\alpha'(\lambda)\hat{\mathbf M}_\lambda$ (P3 Eq. r) is
   supported on the *nonstationary* eigenmodes, of which Setting A has none.

---

## A1 — The null theorem (attention is a pure gauge orbit; the statics live in the direct path)

> **Prop. A1.1 (cross-time moments are rank-one mean).** [exact]
> For i.i.d. columns and every lag $\tau\ge1$,
> $$\boldsymbol\Sigma_\tau=\mathbb E[a_t a_{t-\tau}^\top]=\mu_a\mu_a^\top,\qquad
> \boldsymbol\Sigma_1^{Y}=\mathbb E[a_{t+1}a_t^\top]=\mu_a\mu_a^\top,\qquad
> \boldsymbol\Sigma_{\tau}^{Y}=\mu_a\mu_a^\top\ (\tau\ge1).$$
> Only the lag-0 moment $\boldsymbol\Sigma_0=\mathbf C_a+\mu_a\mu_a^\top$ carries a
> non-mean (content) part $\mathbf C_a=\mathbf D^\top\boldsymbol\Sigma_c\mathbf D$.

*Proof.* Independence gives $\mathbb E[a_t a_{t-\tau}^\top]=\mathbb E[a_t]\mathbb
E[a_{t-\tau}]^\top=\mu_a\mu_a^\top$ for $\tau\ge1$; identically for the target since
$a_{t+1}\perp a_{1:t}$. $\square$

Consequences for the P3 statistics: with $\alpha(0)$ the diagonal (lag-0) attention
mass, $\sum_{\tau\ge0}\alpha(\tau)=1$,

$$
\mathbf C=\alpha(0)\boldsymbol\Sigma_0+(1-\alpha(0))\mu_a\mu_a^\top,\qquad
\mathbf C_+=\mu_a\mu_a^\top,\qquad
\boldsymbol\Sigma_1^{Y}=\mu_a\mu_a^\top .
$$

> **Prop. A1.2 (the attended-path residual $\mathbf R$ is confined to the mean sector).** [exact]
> $$\mathbf R:=\mathbf C_+-\boldsymbol\Sigma_1^{Y}\boldsymbol\Sigma_0^{-1}\mathbf C
> =(1-\alpha(0))\,(1-s_\mu)\,\mu_a\mu_a^\top,\qquad
> s_\mu:=\mu_a^\top\boldsymbol\Sigma_0^{-1}\mu_a\in[0,1).$$
> $\mathbf R$ is **rank $\le1$, aligned with $\mu_a$ on both sides**, and vanishes
> iff $\alpha(0)=1$ (diagonal attention) or $s_\mu=1$.

*Proof.* Substitute the three displayed moments:
$\mathbf R=\mu_a\mu_a^\top-\mu_a\mu_a^\top\boldsymbol\Sigma_0^{-1}[\alpha(0)\boldsymbol\Sigma_0+(1-\alpha(0))\mu_a\mu_a^\top]
=(1-\alpha(0))\mu_a\mu_a^\top-(1-\alpha(0))(\mu_a^\top\boldsymbol\Sigma_0^{-1}\mu_a)\mu_a\mu_a^\top$.
By Sherman–Morrison on $\boldsymbol\Sigma_0=\mathbf C_a+\mu_a\mu_a^\top$,
$s_\mu=\frac{\mu_a^\top\mathbf C_a^{-1}\mu_a}{1+\mu_a^\top\mathbf C_a^{-1}\mu_a}<1$, so
$1-s_\mu=(1+\mu_a^\top\mathbf C_a^{-1}\mu_a)^{-1}>0$. $\square$

Compare P3 Eq. (r): $\mathbf R=\sum_{\lambda\neq1}\tilde\alpha'(\lambda)\hat{\mathbf
M}_\lambda$ lives in the **nonstationary** token subspace. Setting A has *no*
nonstationary eigenmode (i.i.d. ⇒ the "transition" reaches stationarity in one step,
$\lambda\neq1$ block empty), so the only surviving piece is the stationary rank-one
$\mu_a\mu_a^\top$ — which P3 explicitly annihilates when $\ub=0$. **A1.2 is P3's
$\mathbf R$ evaluated at a trivial-dynamics process.**

> **Theorem A1 (null value circuit; attention loss-flat modulo the mean gauge).**
> [exact, affine model] Augment the regressor with a constant coordinate (a bias
> $\beta$, i.e. affine $f_t=\beta+\mathbf W a_t+\mathbf V\bar a_t$, $\bar
> a_t=\sum_{s\le t}A_{ts}a_s$). Then the unique population-MSE minimiser is
> $$\boxed{\ \beta_\ast=\mu_a,\qquad \mathbf W_\ast=0,\qquad \mathbf V_\ast=0\ }$$
> whenever $\mathbb E[ZZ^\top]$ is nonsingular ($Z=(1;a_t;\bar a_t)$); i.e. **the
> optimal predictor is the constant $\mu_a$ and the entire attention circuit — QK and
> OV, every head — carries zero.** The attention-dependent part of the loss (P3's
> skip-bigram term $\tfrac12\mathrm{tr}(\mathbf R\mathbf S^+\mathbf R^\top)$) is
> identically zero for every pattern $\mathbf A$: **the loss surface is flat in all
> attention directions — a single gauge orbit.**

*Proof.* Independence gives $\mathbb E[YZ^\top]=\mu_a\,\mathbb E[Z]^\top$ (each block:
$\mathbb E[a_{t+1}\cdot1]=\mu_a$, $\mathbb E[a_{t+1}a_t^\top]=\mu_a\mu_a^\top$,
$\mathbb E[a_{t+1}\bar a_t^\top]=\mu_a\mathbb E[\bar a_t]^\top=\mu_a\mu_a^\top$ using
$\mathbb E[\bar a_t]=\mu_a$ by row-stochasticity). The first row of $\mathbb
E[ZZ^\top]$ equals $\mathbb E[Z]^\top$, so the coefficient
$\Theta_\ast=\mathbb E[YZ^\top]\,\mathbb E[ZZ^\top]^{-1}=\mu_a\,\mathbb E[Z]^\top\mathbb
E[ZZ^\top]^{-1}=\mu_a e_1^\top=(\mu_a;0;0)$. Uniqueness holds when $\mathbb E[ZZ^\top]$
is invertible. Flatness: after centering ($\tilde a=a-\mu_a$) all cross-time moments
vanish (A1.1), so $\mathbf R_{\text{centered}}\equiv0$ and the skip-bigram term is
$0$ for every $\alpha,\gamma$. $\square$

> **Corollary A1.a (every FRA attribution on a trained Setting-A model is a property of
> the optimizer's gauge choice, not the data — loss-flat but reproducibly pinned).**
> Because the attention loss is exactly flat, no FRA attribution is constrained by the
> data: the clean representative ($\mathbf W=\mathbf V=0$, $\beta=\mu_a$) achieves the
> floor with **identically zero** attributions, while any trained model achieves the
> same floor with large ones — the definitive gauge demonstration. Any FRA-QK score,
> FRA-OV transport, or head ranking measured on such a model is therefore **not
> computation**. This is the sprint-2 gauge verdict in its **maximal** form: sprint-2
> (Prop. featfeat) found the tok×tok *score* block gauge at a Mess3 optimum while
> FRA-**OV** remained computation; here the OV sector is gauge **too**, because there is
> no cross-time signal for it to carry. Setting A is the sharp boundary (§A2.4).
>
> **Numerically corrected (VERIFY_A.md, check 1).** The *mechanism* is subtler than
> "seed noise." I initially predicted the unfaithfulness would show as *seed-incoherent*
> attributions (cross-seed correlation $\approx0$); the measurement **falsifies** that:
> cross-seed FRA correlation is $\approx0.99$ (vs $-0.03$ for independent random maps).
> SGD does not sit at small init — it drives to a large-weight representative
> ($\|\mathbf W\|,\|\mathbf V\|=O(1)$, $\beta\neq\mu_a$) that is **reproducible** across
> seeds, distributing the constant across paths ($\beta+(\mathbf W+\mathbf V)\mu_a=\mu_a$
> to 2%, the A2.3 gauge) with $\mathbf V$ cancelling $\mathbf W$'s fluctuation through
> attention. So the correct statement is **loss-flat but optimizer-pinned**: the
> attributions are unfaithful (a loss-preserving move — the clean rep — zeroes them)
> yet reproducible (a shared optimizer inductive bias), *not* random seed noise. The
> faithful falsifier is the clean-vs-trained equivalence, which fires cleanly; the
> seed-incoherence prediction was wrong.

**Honest handling of the mean (the k=0 / skip redundancy in its purest form).**
Theorem A1 needs the *affine* model. Three ways the constant $\mu_a$ can be carried,
all loss-equivalent, none identifiable — this is exactly the "k=0/skip" degeneracy of
sprint-2 §k0 and P1 Eq. 28, stripped to its purest instance:

- **bias $\beta$** (Theorem A1's choice),
- **direct path**: any $\mathbf W$ with $\mathbf W\mu_a$ contributing the constant on
  average (imperfect: $\mathbf W a_t$ fluctuates), or
- **attention-to-anything**: since $\sum_s A_{ds}\mu_a=\mu_a$ for *any* row-stochastic
  pattern, an OV map with $\mathbf V\mu_a\neq0$ injects a copy of the constant
  regardless of where attention looks.

> **Prop. A1.3 (no-bias caveat — attention survives only as a bias-substitute).** [exact]
> Drop the constant regressor. Then $\mathbf V_\ast=\mathbf R\mathbf S^+\neq0$ is the
> rank-one map $\propto\mu_a\mu_a^\top$ of A1.2, and it improves MSE *only* by giving a
> lower-variance estimate of the constant $\mu_a$ (the attended context $\bar a_t$
> averages $\sim(\sum_\tau\alpha(\tau)^2)^{-1}$ independent draws, shrinking the
> fluctuation of the injected constant). This surviving "use" is (i) confined to the
> rank-one **mean sector**, (ii) **content-free** — identical for every GT feature,
> carrying zero information about which $z_i$ fired — and (iii) removed the instant a
> bias or a constant positional embedding exists (as in the P2/P3 architectures). The
> non-uniqueness locus of the $\mathbf W$/$\mathbf V$ split is exactly $\bar a_t$
> collinear with $a_t$, i.e. $\alpha(0)=1$: the **diagonal-attention = direct-path
> gauge** (sprint-2's $a_0(d)$ dial) surfacing as a rank-deficiency of $\mathbb
> E[ZZ^\top]$.

**Numerical falsifier for A1.** Train the one-head model (with bias) on i.i.d. draws
of $a=\mathbf D^\top c+b$. Predictions: (i) $\|\mathbf V_\ast\|_F\to0$ and
$\|\mathbf W_\ast\|_F\to0$, $\beta\to\mu_a$; (ii) sweep *any* loss-preserving attention
reparametrisation (e.g. randomise the pattern by re-initialising $\mathbf W_Q,\mathbf
W_K$) and confirm $\Delta\mathcal L=0$ to optimiser tolerance; (iii) with the bias
removed, $\mathbf V_\ast$ collapses to a rank-one matrix whose left/right singular
vectors both align with $\hat\mu_a$ (cos $\to1$) and whose action on any centered
$\tilde a$ is null. **Falsified if** a trained head shows a content-selective
(feature-dependent, non-mean) OV transport that lowers loss — that would require
cross-time structure the i.i.d. data does not have.

### A1 — what the *direct path* optimally computes (the statics)

Next-activation prediction is trivial (constant $\mu_a$). The interesting statics is
the **within-position** problem the direct path composes with downstream: recover the
coefficients $c$ (equivalently the belief about which features fired) from a single
$a=\mathbf D^\top c+b$. This is the autoencoding/denoising content the SPEC flags.

> **Prop. A1.4 (optimal linear read-out = LMMSE with matched-filter contamination).** [exact]
> The MSE-optimal affine estimator $\hat c=\arg\min_{\mathbf R,r_0}\mathbb E\|c-\mathbf
> R a-r_0\|^2$ is the Wiener filter
> $$\hat c=\bar c+\boldsymbol\Sigma_c\mathbf D\,\mathbf C_a^{+}\,(a-\mu_a),\qquad
> \mathbf C_a=\mathbf D^\top\boldsymbol\Sigma_c\mathbf D .$$
> The **transfer matrix** $\mathbf T:=\boldsymbol\Sigma_c\mathbf D\,\mathbf C_a^{+}\mathbf
> D^\top\in\mathbb R^{N\times N}$ (mapping true fluctuation $c-\bar c$ to estimated
> $\hat c-\bar c$) governs recovery:
> - **complete regime $N\le d$, $\mathbf D$ full row rank:** $c\mapsto\mathbf D^\top c$
>   is injective, $\mathbf T=\mathbf I_N$ — **exact recovery, no shrinkage**;
> - **superposed/overcomplete regime $N>d$:** $\mathrm{rank}(\mathbf T)=d<N$, so
>   $\mathbf T\neq\mathbf I$: **diagonal shrinkage $\mathbf T_{ii}<1$ and off-diagonal
>   contamination $\mathbf T_{ij}\neq0$** — no linear read-out recovers a feature whose
>   direction lies partly in the invisible $(N-d)$-dim null space of $\mathbf D^\top$.

*Proof.* Standard LMMSE: $\mathrm{Cov}(c,a)=\boldsymbol\Sigma_c\mathbf D$,
$\mathrm{Cov}(a)=\mathbf C_a$; the estimator and its idempotent-on-image transfer
matrix follow. For $N\le d$ full rank, $\mathbf D^\top$ has a left inverse, error $=0$.
For $N>d$, $\mathbf C_a$ has rank $d$ and $\mathbf T$ factors through $\mathbb R^d$. $\square$

**Where the Gram off-diagonals enter (leading order).** The *matched filter*
(decoder-transpose read-out an SAE encoder approximates before correction) is exact and
transparent:

$$
\hat c_i^{\text{MF}}:=\mathbf d_i^\top(a-b)=\mathbf d_i^\top\mathbf D^\top c
=\underbrace{c_i}_{\text{signal}}+\underbrace{\sum_{j\neq i}\mathbf G_{ij}\,c_j}_{\text{contamination}\ \propto\ \mathbf d_i^\top\mathbf d_j}.
\tag{A1-MF}
$$

Contamination is **exactly the Gram off-diagonals** $\mathbf G_{ij}=\mathbf
d_i^\top\mathbf d_j=O(\rho_{mm})$. The LMMSE correction $\mathbf C_a^{+}$ partially
un-mixes this; for homogeneous variance $\boldsymbol\Sigma_c=v\mathbf I$ and mild
overlap, $\mathbf T=\mathbf I-\,$(correction), giving the **ridge-like** per-feature form

$$
\hat c_i-\bar c_i\ \approx\ \kappa_i\,(c_i-\bar c_i)\ +\ \sum_{j\neq i}\theta_{ij}\,(c_j-\bar c_j),\qquad
\kappa_i=1-O\!\big(\textstyle\sum_{j\neq i}\mathbf G_{ij}^2\big),\ \ \theta_{ij}=-\mathbf G_{ij}+O(\rho^2),
\tag{A1-shrink}
$$

so the optimal read-out **shrinks each feature** ($\kappa_i<1$, ridge-like, set by the
row's total squared overlap) and **cross-contaminates $\propto\mathbf G_{ij}$**. This
$\mathbf T$ is what $\mathbf W_\ast$ composes with: the downstream "belief" the direct
path forms about $c$ is $\mathbf T$-distorted, and every FRA read of a feature channel
inherits the same $\kappa,\theta$ (see A4). **Numerical check (VERIFY_A.md, check 3 —
CONFIRMED):** $\mathbf T=\mathbf I$ for $N\le d$ (relerr $1.4\times10^{-12}$ at $N=d=24$,
even at $\rho_{mm}=0.46$ — full rank, not orthogonality, is what matters); in the
overcomplete regime the mean shrinkage obeys the **clean law
$\overline{\mathbf T_{ii}}=\min(1,d/N)$ exactly** (measured $0.500$ at $N{=}48,d{=}24$;
$0.300$ at $N{=}80,d{=}24$), off-diagonal contamination anti-correlates with the Gram
(corr $-0.74,-0.85$), and the matched-filter identity $\hat c^{\text{MF}}=\mathbf Gc$
holds to $2\times10^{-16}$.

---

## A2 — Exact FRA algebra in the GT basis (arbitrary head) + the gauge group

Take an arbitrary one-layer softmax head (not necessarily optimal). Frozen-LN as a
content-independent linear map $\Phi$ (sprint-2 Assumption A2, imported and re-checked:
it is content-independent by the same argument, since LN acts per-position and the
token-dependent leak is absorbed into $\varepsilon$ — see §A2.3). Define **per-feature
query/key/value vectors** and **bias parts**

$$
\mathbf q_i:=\mathbf W_Q\Phi\mathbf d_i,\quad \mathbf k_j:=\mathbf W_K\Phi\mathbf d_j,\quad
\mathbf o_j:=\mathbf W_{OV}\mathbf d_j,\qquad
q_b:=\mathbf W_Q\Phi b+b_Q,\quad k_b:=\mathbf W_K\Phi b+b_K .
$$

> **Prop. A2.1 (exact score decomposition).** [exact]
> With $a_s=\sum_i c_i(s)\mathbf d_i+b$, the pre-softmax score
> $\sigma(d,s)=(\mathbf W_Q\Phi a_d+b_Q)^\top(\mathbf W_K\Phi a_s+b_K)$ expands exactly as
> $$
> \sigma(d,s)=\underbrace{\sum_{i,j}c_i(d)\,c_j(s)\,\big(\mathbf d_i^\top\mathbf W_{QK}\mathbf d_j\big)}_{\text{feature}\times\text{feature }Q_{ij}}
> +\underbrace{\sum_i c_i(d)\,\beta^q_i}_{\text{query pedestal (row const)}}
> +\underbrace{\sum_j c_j(s)\,\beta_j}_{\text{key pedestal}}
> +\underbrace{q_b^\top k_b}_{\text{const}},
> $$
> where $\mathbf W_{QK}:=\Phi^\top\mathbf W_Q^\top\mathbf W_K\Phi$,
> $Q_{ij}=\mathbf d_i^\top\mathbf W_{QK}\mathbf d_j=\mathbf q_i^\top\mathbf k_j$,
> $\beta^q_i=\mathbf q_i^\top k_b$, $\beta_j=q_b^\top\mathbf k_j$ (sprint-2's
> **pedestal** $\beta$).

> **Prop. A2.2 (exact OV transport).** [exact]
> Value $v_s=\mathbf W_{OV}a_s=\sum_j c_j(s)\mathbf o_j+\mathbf W_{OV}b$; the head output
> $$c_d=\sum_{s\le d}A_{ds}v_s=\sum_j\Big(\underbrace{\textstyle\sum_{s\le d}A_{ds}c_j(s)}_{\text{lag-aggregated feature }j}\Big)\mathbf o_j+\mathbf W_{OV}b,$$
> and the **FRA-OV attribution of (source feature $j$, lag $\tau$)** is
> $A_{d,d-\tau}\,c_j(d-\tau)\,\mathbf o_j$. All lag structure sits in the scalar
> $A_{d,d-\tau}$; all content in the fixed direction $\mathbf o_j=\mathbf W_{OV}\mathbf
> d_j$ — the continuous-input form of sprint-2's Prop. attr $A_{ds}\mathbf R v_s$.

### A2.3 The gauge group, enumerated for Setting A

Each item is a weight reparametrisation preserving the forward pass and hence the loss;
the first three are sprint-2's G1–G3 transferred, the fourth is the superposition term.

**(i) Softmax row constants (sprint-2 G2/G2′).** [exact] Softmax is invariant under
$\sigma(d,\cdot)\to\sigma(d,\cdot)+g(d)$. Hence the **query pedestal**
$\sum_i c_i(d)\beta^q_i$ and the **constant** $q_b^\top k_b$ (both functions of $d$
alone) are *unobservable* — inert at every gain. Verify: they never change any
attention row.

**(ii) The bias / $b_{\mathrm{dec}}$ re-split (sprint-2 G1, the primary dial).**
[exact everywhere] The fixed vector $b$ (dictionary bias) and any constant added to
every position can be moved between $b$, the value bias $\mathbf W_{OV}b$, and a
downstream decoder bias / residual skip, because a row-stochastic pattern reproduces a
constant regardless of where it looks: $\sum_s A_{ds}(\mathbf W_{OV}b)=\mathbf W_{OV}b$.
This dials the **key pedestal** $\beta_j=q_b^\top\mathbf k_j$ (and the const), exactly
sprint-2's G1 $\eb(z)\to\eb(z)+u,\ \pb_s\to\pb_s-u$ with the token embedding replaced by
the fixed $b$. It is the same "constant carried anywhere" freedom as Theorem A1's mean
gauge — G1 and the k=0/skip redundancy are one object here.

**(iii) Head splits at fixed head-sum (sprint-2 §twohead, P1 "only the aggregate is
constrained").** [exact for free patterns; softmax-obstructed] Multiple heads whose
$\sum_h A^{(h)}_{ds}\mathbf o^{(h)}_j$ (and $\sum_h\sigma^{(h)}$) match a given total are
loss-equivalent; per-head attributions are a gauge family. Under softmax the family is
*not freely realizable* (sprint-2 Prop. hinge: $O(\varepsilon)$ generic, $O(1)$ at
keyless parity rows) — but in Setting A this obstruction is moot because the head-sum
itself is unconstrained (A1).

**(iv) Superposition non-orthogonality (the $\mathbf d_i\!\cdot\!\mathbf d_j$ term).**
[exact] The centered interaction $\hat Q_{ij}=\hat{\mathbf q}_i^\top\hat{\mathbf k}_j$
is the gauge-**invariant** core (untouched by i–iii). Even a head "diagonal in the GT
basis," $\mathbf W_{QK}=\sum_k w_k\,\mathbf d_k^{\dagger}\mathbf d_k^{\dagger\top}$
(dual vectors $\mathbf d^\dagger$), reads
$Q_{ij}=\sum_k w_k(\mathbf d_i^\top\mathbf d_k^\dagger)(\mathbf d_k^{\dagger\top}\mathbf
d_j)$; for near-orthogonal $\mathbf D$, $\mathbf d^\dagger_k\approx\mathbf d_k$ and
$$
Q_{ij}\approx w_i\,\delta_{ij}+ \tfrac12(w_i+w_j)\,\mathbf G_{ij}+O(\rho^2),
$$
so the feature×feature block acquires **off-diagonal contamination $\propto\mathbf
G_{ij}=\mathbf d_i^\top\mathbf d_j$** — the QK analogue of the read-out contamination
(A1-MF). This is not gauge (it is an observable of the head) but it is **dilution/
description** (§A3), not computation.

### A2.4 Reachable vs pinned, and the $R\to0$ status of the optimum

> **Prop. A2.3 (gauge-reachable vs pinned for a general head; the optimum is the
> $R\to0$ limit).** [exact] For an *arbitrary* head the coordinates
> $\{$const, query pedestal$\}$ are **unobservable** (i), the key pedestal $\beta_j$ is
> **gauge-reachable** (ii), the head partition is **gauge-reachable** (iii), and the
> centered interaction $\hat Q_{ij}$ plus the OV directions $\mathbf o_j$ are
> **gauge-invariant observables**. What the **loss pins** is a separate question:
> sprint-2 (Thm. char, Prop. featfeat) found that at a Mess3 optimum the *pattern* must
> be token-independent, which forces $\hat Q_{ij}=0$ (C1a) — the interaction is pinned
> to zero. **Setting A removes even this pin:** because $\mathbf R=0$ (A1.2, modulo
> mean), the loss constrains *no* attention coordinate, so the interaction $\hat Q$ may
> take *any* value at a minimiser. Setting A is precisely the **$\mathbf R\to0$ limit**
> of the sprint-2 boundary theorem: the sprint-2 separable-kernel result "FRA-QK has
> zero causal handle at every minimiser" is the $\zeta\neq0$ statement (content sector
> empty, pattern still pinned); Setting A is the $\zeta=0$ corner where the pattern is
> unpinned too and FRA-OV joins FRA-QK on the gauge side.

**This is the reconciliation with sprint-2, and it is not a contradiction.** Sprint-2's
headline "FRA-OV is the real handle" is *conditional on cross-time structure*: OV
transports the constrained-belief displacement $\zeta^{d-s}\gb(z_s)$ only because the
Mess3 process has a nonstationary eigenvalue $\zeta\neq0$ for attention to carry.
Setting A is the i.i.d. null of that structure; the same OV calculus, evaluated here,
gives $\zeta^{d-s}\to$ (nothing to transport) and the handle vanishes. Stating it as a
crisp boundary: **FRA-OV is computation iff the data has cross-time predictive
structure; A1 is the theorem that removes it.**

**Numerical falsifier for A2.** On a hand-built head with known
$\mathbf W_Q,\mathbf W_K,\mathbf W_{OV}$: (i) verify $\sigma(d,s)$ reconstructs from the
four A2.1 terms to machine precision; (ii) apply the G1 dial (move $b$ into $b_{dec}$)
and confirm the pattern, forward pass and loss are bit-identical while $\beta_j$ sweeps
through zero; (iii) confirm $\hat Q_{ij}$ is invariant under (i)–(iii) but $Q_{ij}$
acquires the $\propto\mathbf G_{ij}$ term as $\rho_{mm}$ is dialed up. **Falsified if**
a row-constant or pedestal move changes any attention weight, or if $\hat Q$ moves under
a G1 dial.

---

## A3 — Dilution, exactly ($1/m^2$ QK, $1/m$ OV) — and its clean separation from gauge

Fix an invariant coupling and refine the dictionary. **Split GT feature $i$ into $m$
latents** sharing its direction: $\mathbf d_i\!\to\!\{\mathbf d_i^{(a)}=\mathbf
d_i\}_{a=1}^m$, each carrying coefficient $c_i/m$ (so $\sum_a (c_i/m)\mathbf d_i=c_i\mathbf
d_i$ reconstructs the activation exactly). Equivalently use an overcomplete $L>N$
dictionary with the bench's matching structure. Because the split latents share
$\mathbf d_i$, all per-latent QK/OV *directions* are unchanged; only coefficients dilute.

> **Prop. A3.1 (dilution scaling).** [exact]
> Splitting the destination feature into $m$ and the source feature into $m'$:
> - **QK per-pair coefficient** of $(\,i^{(a)}\!,j^{(b)})$ is
>   $(c_i/m)(c_j/m')\,Q_{ij}$ — scales as $\mathbf{1/(mm')}$ (for $m'=m$, **$1/m^2$**),
>   over $m m'$ pairs;
> - **OV per-latent coefficient** of $i^{(a)}$ is $A_{ds}(c_i/m)\mathbf o_i$ — scales as
>   $\mathbf{1/m}$, over $m$ latents.
> The reason QK dilutes one power faster is that the score is **bilinear** (feature at
> destination × feature at source), so a split hits it on both legs; OV is **linear** in
> the source feature.

> **Prop. A3.2 (group cuts restore the full effect; single cuts recover a fraction).** [exact]
> $$\underbrace{\sum_{a,b}(c_i/m)(c_j/m)\,Q_{ij}}_{\text{full QK group ($m^2$ pairs)}}=c_ic_j Q_{ij},\qquad
> \underbrace{\sum_{a}A_{ds}(c_i/m)\,\mathbf o_i}_{\text{full OV group ($m$ latents)}}=A_{ds}c_i\mathbf o_i.$$
> Cutting **one** OV latent removes $1/m$ of feature $i$'s transport; cutting the **whole
> group** removes $100\%$, exactly and **independently of $m$**. Cutting one QK pair
> removes $1/m^2$; the whole $m^2$-pair group removes $100\%$. **Group-cuts always
> restore the full single-feature effect** — because the group-sum is a *physical
> invariant fixed by the data*, not a convention.

> **Prop. A3.3 (dilution ⟂ gauge — the crisp separation).** [exact]
> These are **orthogonal axes**, and Setting A shows why the two must not be conflated:
>
> | | group-sum **recovers** the coefficient? | group-sum **loss-constrained**? |
> |---|---|---|
> | **dilution** (description) | **yes** — aggregate to the group (A3.2) | (either) |
> | **gauge** (convention) | (either) | **no** — dialable at fixed loss |
>
> - *Dilution* is a change of *description*: the per-latent/per-pair numbers shrink as
>   $1/m,1/m^2$, but grouping recovers the invariant coupling **exactly and always**.
> - *Gauge* is a change of *convention*: sprint-2 showed the "full-token-set" key cut —
>   an aggregate object — still swings $140\times$ under a loss-preserving G1 dial
>   (Finding 2), because that aggregate is a pedestal (gauge coordinate), **not** a
>   data-pinned invariant. Aggregation does *not* rescue a gauge coordinate.
> - **Setting A's own lesson:** the OV/QK group-sums here *are* the physical couplings
>   $A c_i\mathbf o_i,\ c_ic_jQ_{ij}$ (dilution is fully reversible by grouping), **yet
>   those couplings are themselves gauge** (loss-free, by A1). So a feature can be
>   *un-diluted-recoverable* and *gauge* simultaneously — the two properties are
>   independent. The team-lead's requested contrast: **grouping fixes dilution; nothing
>   fixes gauge.**

Note the honest limit: because attention is entirely gauge in Setting A, the *attention*
FRA terms cannot themselves exhibit a non-trivial dilution-vs-computation contrast
(there is no computation to recover). The clean home for dilution *as computation* in
Setting A is the **direct-path read-out** (A1.4): split $\mathbf d_i$ across $m$ latents,
the matched-filter coefficients (A1-MF) dilute as $1/m$, yet the grouped estimate
$\sum_a\hat c_i^{(a)}$ recovers $c_i$ exactly (up to the A1.4 shrinkage) — dilution on a
genuinely load-bearing computation, contrasted with the mean-sector gauge.

> **Prop. A3.4 (superposition contamination of FRA terms, leading order).** [approx O(ρ²)]
> At $\rho_{mm}>0$, in the GT basis the FRA terms pick up cross-feature contamination
> $\propto\mathbf d_i^\top\mathbf d_j$:
> $$Q_{ij}=w_i\delta_{ij}+\tfrac12(w_i+w_j)\,\mathbf G_{ij}+O(\rho^2)\quad(\text{QK, A2.4}),\qquad
> \mathbf d_i^\top\mathbf o_j=o_i\,\delta_{ij}+ (\text{OV overlap})\,\mathbf G_{ij}+O(\rho^2),$$
> and the read-out contamination (A1-MF) $\hat c_i^{\text{MF}}=c_i+\sum_{j\neq
> i}\mathbf G_{ij}c_j$ is **exact** (no $O(\rho^2)$). Thus a nominally single-feature FRA
> edit in the GT basis bleeds into feature $j$ at first order in the overlap — the static
> precursor of sprint-2's *geometrically forced collateral* (Cor. geom), except there
> the collateral was forced by the $\sum_z\gb(z)=0$ coplanarity of a **complete**
> $2$-plane, whereas here it is tunable via $\rho_{mm}$ and vanishes at $\rho_{mm}=0$.

**Numerical falsifier for A3.** Build a coupling, split one feature into $m=1,2,4,8$
latents; measure per-pair QK and per-latent OV attributions and confirm the
$1/m^2,\,1/m$ laws and that the $m$-group (resp. $m^2$-group) cut effect is
$m$-independent. Separately, dial $\rho_{mm}$ and confirm the $\propto\mathbf G_{ij}$
contamination slope. **Falsified if** a full-group cut's effect depends on $m$
(would mean the split changed the computation, not just the description), or if
contamination fails to scale linearly in $\mathbf G_{ij}$ as $\rho\to0$.

---

## A4 — Dictionary → χ: bench metrics, FRA-edit realizability, and why MCC is insufficient

Let the learned SAE have latents $\ell=1,\dots,L$ with decoder columns $\mathbf w_\ell$.
Define the **mixing matrix** χ by decomposing each learned latent's *transported/decoded
content* onto GT contributions (sprint-2 Finding 4, support-restricted estimate):

$$
\mathbf w_\ell=\sum_i \chi_{\ell i}\,\mathbf d_i\ (+\ \text{off-dictionary residual}),\qquad
\chi\in\mathbb R^{L\times N}.
$$

> **Prop. A4.1 (bench metrics are properties of χ).** [exact for $\rho_{mm}=0$;
> $O(\rho)$ corrections otherwise]
> With GT directions orthonormal ($\rho_{mm}=0$) and unit-norm latents
> ($\|\chi_{\ell\cdot}\|=1$):
> - **MCC** (SS §4, Hungarian matching) $=\frac1{\min(L,N)}\sum_{(\ell,i)\in\text{match}}|\mathbf
>   w_\ell^\top\mathbf d_i|=\frac1{\min(L,N)}\sum_{\text{match}}|\chi_{\ell i}|$ =
>   **diagonal strength of χ after matching**. $\mathbf w_\ell^\top\mathbf
>   d_i=\chi_{\ell i}+\sum_{k\neq i}\chi_{\ell k}\mathbf G_{ki}$, so at $\rho>0$ MCC also
>   reads off-diagonal χ through the Gram — MCC$=1\Leftrightarrow$ χ is a signed
>   permutation.
> - **Uniqueness** $=|\{i^*(\ell)\}|/L$, $i^*(\ell)=\arg\max_i|\chi_{\ell i}|$ =
>   **no two latents share the same dominant column** = no column of χ is argmax for two
>   rows. Column-sharing is the dilution/absorption signature (two latents split one
>   feature).
> - **Hedging** (correlated features, $\boldsymbol\Sigma\neq\mathbf I$): a latent for
>   feature $i$ acquires off-diagonal mass on a correlated feature $j$,
>   $\chi_{\ell j}\propto\mathrm{Cov}(c_i,c_j)+O(\cdot)$ (leading order) — the specific
>   off-diagonal χ pattern SS/Chanin call feature hedging.
> - **Absorption** (hierarchy, parent-gated child): the **child's** χ row carries a
>   nonzero entry on the **parent** feature, $\chi_{\text{child},\text{parent}}\neq0$
>   ("parent-in-child"), because the parent fires whenever the child fires and the
>   sparse code folds the always-present parent direction into the child latent.

> **Prop. A4.2 (what an FRA edit inherits; the severing-realizability condition).** [exact]
> An edit in latent space is an edit of χ-mixed GT channels: zeroing/scaling a set of
> latents $\mathcal E$ removes $\sum_{\ell\in\mathcal E}\xi_\ell\mathbf w_\ell=\big(\sum_\ell
> \xi_\ell\chi_{\ell\cdot}\big)^\top\mathbf D$ from the stream. **"Sever GT feature $i$"
> is realizable as a latent-set operation iff GT channel $i$ lies in the row space of
> χ:**
> $$\boxed{\ \exists\,\xi\in\mathbb R^{L}:\ \xi^\top\chi=e_i^\top\ }\quad\Longleftrightarrow\quad e_i\in\mathrm{rowspace}(\chi).$$
> A **sufficient** condition is that χ has a **left inverse** (full column rank $N$,
> requiring $L\ge N$ and $\mathrm{rank}(\chi)=N$ — a *complete* code): then every GT
> channel is reachable, $\xi^\top=e_i^\top\chi^{+}$.

*Proof.* The realizable removals are $\{(\xi^\top\chi)\mathbf D:\xi\in\mathbb R^L\}$;
to remove exactly the $\mathbf d_i$ channel we need $\xi^\top\chi=e_i^\top$, solvable iff
$e_i\in\mathrm{rowspace}(\chi)$; full column rank makes it solvable for all $i$. $\square$

This is sprint-2 **Finding 4's "token-1-by-absence" pathology**, generalized and made
precise. There a TopK SAE learned only **2** token latents for **3** GT tokens: $\chi\in\mathbb
R^{2\times3}$, rank $2$; a coordinate $e_1$ lies in a generic $2$-dim row subspace of
$\mathbb R^3$ with probability zero, so "sever token 1" has no latent-set realization
even though the oracle channel cuts fine ($\Delta$CE $3.46\times10^{-3}$). The
obstruction is **incompleteness of the code**, not causal unreachability of the concept.

> **Prop. A4.3 (MCC cannot certify edit-faithfulness; completeness + centering are needed).** [exact]
> High MCC is **necessary but not sufficient**:
> 1. **Completeness.** MCC divides by $\min(L,N)$ and averages only over *matched* pairs.
>    With $L<N$ a whole feature can be unrepresented while MCC$\approx1$ on the matched
>    ones — yet $e_i\notin\mathrm{rowspace}(\chi)$ for the missing feature, so its severing
>    is unrealizable. MCC does not see the missing channel.
> 2. **Centering.** MCC is a *direction* cosine; it ignores the encoder threshold /
>    decoder-bias offset. A latent can align in direction ($\chi$ diagonal, MCC$\to1$) yet
>    be a **contrast/absence code** (fires by the *absence* of another feature, wrong
>    affine offset), so an edit built on it severs at the wrong operating point and injects
>    the wrong constant. This is sprint-2's verdict verbatim: *"Set-level token purity is
>    perfect ($R^2\approx1$); purity is necessary, not sufficient. Completeness and
>    centering of the code are what FRA edits inherit"* (Finding 4). MCC certifies neither.
> **Sufficient certificate for "sever feature $i$":** (a) $e_i\in\mathrm{rowspace}(\chi)$
> (a left-inverse row exists — completeness/reachability), **and** (b) the edit's affine
> null is correct (centering: the removed channel's baseline matches $\bar c_i\mathbf
> o_i$, so $c=\bar c_i$ maps to the un-edited output). MCC ⇒ neither (a) nor (b).

**Numerical falsifier for A4.** Plant a known χ (train a TopK SAE on the SS bench, or
hand-build an incomplete code): (i) verify severing feature $i$ is realizable exactly
when $e_i\in\mathrm{rowspace}(\hat\chi)$ and fails otherwise, *independent of MCC*;
(ii) construct a code with MCC$>0.99$ but $L<N$ and exhibit an unremovable feature;
(iii) construct a direction-aligned contrast code (high MCC) whose edit severs at the
wrong offset (nonzero residual at $c_i=\bar c_i$). **Falsified if** any feature with
$e_i\notin\mathrm{rowspace}(\hat\chi)$ is severable by a latent-set op, or if MCC$=1$
guarantees faithful severing.

---

## Cross-references and open flags

- **A1 ↔ P3:** A1.2 is P3 Eq. (r) at a process with no nonstationary eigenvalue; the
  loss-flatness is P3's skip-bigram term $\tfrac12\mathrm{tr}(\mathbf R\mathbf S^+\mathbf
  R^\top)$ at $\mathbf R=0$. The mean gauge is P3's $\ub\one^\top$ symmetry — but
  **note the transfer is only partial**: P3's symmetry rests on the one-hot identity
  $\one^\top\mathbf X=\one^\top\mathbf Y=\one_T^\top$, which continuous $a_t$ do **not**
  satisfy (flagged gap). What survives is the *mean-carrying* redundancy (constant in the
  affine span), which is weaker than P3's full $|\mathcal V|$-dim $\ub$ gauge. This is an
  honest narrowing, not a contradiction.
- **A1 ↔ sprint-2:** the "attention entirely gauge" verdict is sprint-2's boundary
  theorem at its $\zeta=0$ endpoint (Prop. A2.3); FRA-OV joining FRA-QK on the gauge side
  refines — does not contradict — sprint-2's "FRA-OV is the handle," by making explicit
  its precondition (cross-time structure).
- **Flagged gaps:** (1) frozen-LN content-independence (Assumption A2) is *imported*, not
  re-derived for continuous inputs — the token-dependent LN leak $\varepsilon$ should be
  re-measured on Setting-A activations before leaning on A2.1's exactness. (2) The no-bias
  sliver (A1.3) means "attention loss-flat" is **exact only in the affine model**; a
  strict no-bias P3 head retains a rank-one mean-sector use — content-free but nonzero
  loss gain. (3) A3.4/A4.1 off-diagonal (hedging/absorption→χ) coefficients are given to
  leading order in $\rho$/$\mathrm{Cov}$; exact copula/hierarchy expressions are deferred.
- **Setting-B hooks (for `B_reset.md`):** the reset process reinstates a nonstationary
  eigenvalue $\lambda_i$ per feature, moving $\boldsymbol\Sigma_\tau$ off rank-one and
  turning $\mathbf R\neq0$ — which is exactly what promotes FRA-OV from gauge (A) to the
  $\lambda_i^k\gb$ handle (B3). A1 is the $\lambda_i\to0$ baseline that makes the B result
  legible.
