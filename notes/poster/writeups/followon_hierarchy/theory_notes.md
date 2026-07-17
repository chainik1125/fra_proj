# Theory notes — hierarchical leaky reset: factored filter, ergodicity, and registered predictions

**Date:** 2026-06-12.
**References:** design `analysis/em_pipeline/em_pipeline_notes/hierarchical_leaky_reset.md`; implementation `build_hierarchical_leaky_reset_hmms` in `analysis/afp_builders.py`; flat two-sector predecessor `analysis/em_pipeline/new_process.md`.

## 0. Setup and notation

Hidden space $\mathcal H=\bigoplus_{s,r} H_{s,r}$ with leaves $\ell=(s,r)$, persona $s\in\{A,B\}$, domain $r\in\{1,\dots,K\}$, $|H_\ell|=d$. Belief $\pi=(\pi_\ell\,\mu_\ell)_\ell$ with leaf masses $\pi_\ell$ and within-leaf beliefs $\mu_\ell\in\Delta^{d-1}$. Operators (matching the builder):

- **Prompt token $p_k$:** $T(p_k)=\tfrac1{V_p}\bigoplus_\ell S_k$, with $S_k=(1-\lambda)I+\lambda\mathbf 1 r_k^\top$ *identical in every leaf*.
- **Completion token $x_{(s^\star,r^\star),i}$:** $T=\bigoplus_\ell W[\ell,(s^\star,r^\star)]\,\mathrm{diag}(e_i)$, with content vectors $e_i$ *identical in every leaf*, and evidence matrix $W$ by mode:
  - `hierarchical`: $W=P\otimes D$, $P=\begin{pmatrix}1&\beta_t\\\beta_t&1\end{pmatrix}$, $D=(1-\beta_s)I_K+\beta_s J_K$ (here $\beta_t=\beta_{\text{top}}$, $\beta_s=\beta_{\text{sub}}$);
  - `flat`: $W=(1-\beta)I_{2K}+\beta J_{2K}$;
  - `scrambled`: $W_{\mathrm{scr}}[\ell,t]=W_{\mathrm{hier}}[\sigma\ell,\sigma t]$ where $\sigma$ swaps $A_r\leftrightarrow B_r$ for domains $r\ge 2$ and fixes $A_1,B_1$ (this is the actual `sigma` in the code: 0-based loop `for r in range(1,K)`).

Every row of $W$ has the same sum in all three modes (hierarchical: $(1+\beta_t)(1+(K-1)\beta_s)$; flat: $1+(2K-1)\beta$; scrambled rows are permutations of hierarchical rows), and $\sum_i e_i[j]=1$ for every state $j$. Hence the builder's row-stochastic normalization divides all completion operators by one global scalar: the entries of $W$ act as exact Bayes factors on leaf masses, unmodified by normalization.

## 1. Factored filter proposition

**Assumptions (satisfied exactly by the builder):** (i) *leaf symmetry*: $S_k$ and $e_i$ are the same in every leaf; (ii) *factorized initial condition*: $\pi_{(s,r)}(0)=a_s(0)\,b_r(0)$ and $\mu_\ell(0)=\mu(0)$ for all $\ell$ (the builder initializes $a=(\pi_A,1-\pi_A)$, $b$ uniform, $\mu$ uniform).

**Proposition 1 (hierarchical mode).** For every token sequence the belief retains the product form
$$\pi_{(s,r),t}=a_{s,t}\,b_{r,t},\qquad \mu_{\ell,t}=\mu_t\ \ \forall\ell,$$
with three decoupled subsystems:

- **(a) Content:** $\mu_t$ is driven only by prompt signatures and content indices: $\mu'=(1-\lambda)\mu+\lambda r_k$ on prompt token $k$; $\mu'\propto\mu\odot e_i$ on completion content $i$.
- **(b) Persona:** the log-odds $\theta_t=\log(a_{B,t}/a_{A,t})$ is constant on prompt tokens and on completion tokens moves by the persona tag alone:
$$\theta'=\theta+\varepsilon(s^\star)\log(1/\beta_t),\qquad \varepsilon(B)=+1,\ \varepsilon(A)=-1,$$
independent of $r^\star$, $i$, and $\mu$.
- **(c) Domain:** the log-odds $\phi_{r,t}=\log(b_{r,t}/b_{1,t})$ are constant on prompt tokens and on completion tokens move by the domain tag alone: the tagged domain gains $\log(1/\beta_s)$ against every other domain; odds between two untagged domains are unchanged.

Moreover the one-step predictive distribution factorizes: $\Pr(x_{(s^\star,r^\star),i}\mid\pi)\;\propto\;\big(\textstyle\sum_s a_s P[s,s^\star]\big)\big(\sum_r b_r D[r,r^\star]\big)\big(\mu^\top e_i\big)$, a product of a function of $\theta$, a function of $\phi$, and a function of $\mu$.

**Proof.** Induction on $t$. *Prompt token:* each block is multiplied by the same $\tfrac1{V_p}S_k$; since $S_k$ is row-stochastic, $(\mu S_k)\mathbf 1=1$, so all leaf masses are scaled by the common $\tfrac1{V_p}$ and are unchanged after normalization, while $\mu'_\ell=\mu S_k$ is the same in every leaf. *Completion token $(s^\star,r^\star,i)$:* the unnormalized leaf mass is $\tilde\pi_{(s,r)}=a_s b_r\,P[s,s^\star]D[r,r^\star]\,(\mu^\top e_i)$ — likelihoods multiply, and the Kronecker structure $W[(s,r),(s^\star,r^\star)]=P[s,s^\star]D[r,r^\star]$ makes the leaf-symmetric content factor $(\mu^\top e_i)$ common to all leaves. The normalizer therefore factorizes, $Z=\big(\sum_s a_sP[s,s^\star]\big)\big(\sum_r b_rD[r,r^\star]\big)(\mu^\top e_i)$, giving $\pi'_{(s,r)}=a'_s b'_r$ with $a'\propto a\odot P[:,s^\star]$ and $b'\propto b\odot D[:,r^\star]$ updated independently; and $\mu'\propto\mu\odot e_i$ is again leaf-independent. Taking log-ratios of $a'$ gives (b): $\theta'=\theta+\log(P[B,s^\star]/P[A,s^\star])$, which is $+\log(1/\beta_t)$ for $s^\star=B$ and $-\log(1/\beta_t)$ for $s^\star=A$; similarly for (c). The predictive factorization is the displayed $Z$ as a function of the token. $\square$

**Corollary 1 (minimal sufficient statistic; single shared persona coordinate).** The optimal predictor's minimal sufficient statistic is
$$(\theta,\phi,\mu)\in\mathbb R\times\mathbb R^{K-1}\times\Delta^{d-1},$$
of dimension $K+d-1$, versus $2Kd-1$ for the unconstrained belief simplex. It contains exactly **one** persona coordinate $\theta$, and $\theta$ modulates the B-vs-A tag odds *identically in every domain*: $\Pr(\text{B tag})/\Pr(\text{A tag})=(\beta_t+e^\theta)/(1+\beta_t e^\theta)$ regardless of $r^\star$. Minimality: this Möbius map is strictly increasing in $\theta$ for $\beta_t\in(0,1)$ (determinant $1-\beta_t^2>0$), so $\theta$ is identified from predictions; domain tag ratios identify $\phi$ for $\beta_s\in(0,1)$; and $\mu\mapsto(\mu^\top e_i)_i=E\mu$ is injective since $E=(1-\delta-\tfrac{\delta}{M-1})\tilde I+\tfrac{\delta}{M-1}J$ has full column rank for $\delta\ne(M-1)/M$. Consequently, a network that implements the optimal filter must maintain a single persona scalar shared across all $K$ domains, updated by $\pm\log(1/\beta_t)$ per completion token. This shared coordinate is the mechanistic substrate for cross-domain transfer.

**Proposition 2 (flat mode: no persona coordinate).** With $W=(1-\beta)I_{2K}+\beta J_{2K}$, leaf symmetry still forces $\mu_{\ell,t}=\mu_t$, and leaf masses update by $\pi'_\ell\propto\pi_\ell\,\beta^{\mathbf 1[\ell\ne t^\star]}$: each token adds $\log(1/\beta)$ to the tagged leaf's log-odds against every other leaf. The minimal sufficient statistic is the **full** leaf log-odds vector $\psi\in\mathbb R^{2K-1}$, $\psi_\ell=\log(\pi_\ell/\pi_{A1})$, together with $\mu$ — equivalently the tag-count vector $(n_\ell)$ up to a global shift. No persona coarse-graining is sufficient: the tag-marginal prediction is $\Pr(\text{tag }t)\propto\beta+(1-\beta)\pi_t$, which depends on each leaf mass separately, so two beliefs with equal persona masses $(\pi_A,\pi_B)$ but different within-persona splits give different predictions. Honesty requires one more observation: one *can* define an autonomous functional in the flat case, e.g. $\tilde\theta=\sum_{\ell\in B}\psi_\ell$-type combinations move by persona tag counts alone. But (i) it is not predictively sufficient on its own, and (ii) it is not distinguished: $W_{\text{flat}}$ is invariant under simultaneous relabeling by **all** of $S_{2K}$, so the identical construction works for any balanced partition of leaves, e.g. $\{A1,B2\}$ vs $\{B1,A2\}$. In the hierarchical mode the symmetry group is only $S_2\times S_K$ (persona swap × uniform domain permutation, for $\beta_t\ne\beta_s$, both $<1$), which preserves the persona partition; in the flat mode "persona" exists in the token labels but not in the predictive geometry. The correct statement is therefore: *the flat sufficient statistic is the exchangeable $2K$-vector of leaf log-odds, with no privileged 1-dimensional persona reduction*.

## 2. Ergodicity clarification

The worry "my process might not be truly non-ergodic because probability mass leaks between summands" conflates two different objects. Every symbol operator is block-diagonal, so a hidden trajectory started in leaf $\ell$ remains in $H_\ell$ forever, almost surely: hidden-state mass **never** crosses leaves. Each leaf is internally ergodic (the leaky reset with $\lambda>0$ mixes the $d$ within-leaf states), and the leaves generate distinct stationary laws — leaf $\ell$ emits tag $t$ at rate $W[\ell,t]/\sum_{t'}W[\ell,t']$, and the rows of $W$ are distinct. The full process is therefore the $\pi(0)$-mixture of $2K$ mutually distinct ergodic components: stationary but strictly non-ergodic, with a $2K$-atom ergodic decomposition; time averages converge to leaf-conditional means, not the mixture mean. What "leaks" during a sequence is the *observer's posterior* over which component generated the trajectory — the belief mass $\pi_\ell$ moving on the simplex under Bayes updates of Sections 1. Filtering on a fixed mixture is not mixing of the chain. The non-ergodicity claim concerns the chain; the leaking concerns the filter; both are true simultaneously and there is no tension.

## 3. Fine-tuning as prior tilt — quantitative predictions

**Model.** Measured quantity: $p_\ell$ = first-completion-token probability mass on leaf-$\ell$ tags (summed over content), on held-out neutral prompts; $\Delta\log p_\ell$ = post-FT minus pre-FT. We model fine-tuning on pure B1-tagged completions as an exponential tilt of the model's internal leaf prior in the direction of the FT tokens' average log-likelihood:
$$\pi^{\mathrm{FT}}_\ell\;\propto\;\pi^0_\ell\,\exp\!\big(\eta\,\mathbb E_{\mathrm{FT}}[\log\Pr(x\mid\ell)]\big)\;=\;\pi^0_\ell\,W[\ell,B1]^\eta,$$
because content factors are leaf-symmetric and contribute constants; $\eta\ge0$ is an effective dose (steps × lr). Hence
$$\boxed{\;\Delta\log p_\ell=\eta\,\log W[\ell,B1]-\Delta\log Z\;}$$
with $\Delta\log Z$ common to all leaves. The B1 columns of $W$:

| leaf | hierarchical | flat | scrambled |
|---|---|---|---|
| $B1$ | $1$ | $1$ | $1$ |
| $B2$ | $\beta_s$ | $\beta$ | $\beta_t\beta_s$ |
| $A1$ | $\beta_t$ | $\beta$ | $\beta_t$ |
| $A2$ | $\beta_t\beta_s$ | $\beta$ | $\beta_s$ |

(Scrambled column verified against the code: $W_{\mathrm{scr}}[\ell,B1]=W_{\mathrm{hier}}[\sigma\ell,B1]$ with $\sigma=(A2\,B2)$ for $K=2$, so B1's pretraining group is $\{B1,A2\}$.)

**Gaps and the ratio invariant (hierarchical).** Differences of $\Delta\log p$ cancel both $\Delta\log Z$ and the prior:
$$\Delta\log p_{B2}-\Delta\log p_{A2}=\eta\big(\log\beta_s-\log\beta_t-\log\beta_s\big)=-\eta\log\beta_t>0,$$
$$\Delta\log p_{A1}-\Delta\log p_{A2}=-\eta\log\beta_s>0,\qquad \Delta\log p_{B1}-\Delta\log p_{B2}=-\eta\log\beta_s,\qquad \Delta\log p_{B1}-\Delta\log p_{A1}=-\eta\log\beta_t.$$
The persona-sharing advantage of B2 over A2 is controlled by $\beta_t$ alone; the domain-sharing advantage of A1 over A2 by $\beta_s$ alone. Taking the ratio kills the unknown dose $\eta$:
$$\boxed{\;R\;\equiv\;\frac{\Delta\log p_{B2}-\Delta\log p_{A2}}{\Delta\log p_{A1}-\Delta\log p_{A2}}\;=\;\frac{\log\beta_t}{\log\beta_s}\;}$$
(both logs negative, $R>0$). Kronecker additivity gives a second $\eta$-free check: $\Delta\log p_{B1}-\Delta\log p_{A2}=(\Delta\log p_{B1}-\Delta\log p_{A1})+(\Delta\log p_{B1}-\Delta\log p_{B2})$.

**Absolute probabilities.** With uniform prior, $p^{\mathrm{FT}}_\ell=W[\ell,B1]^\eta/Z(\eta)$ and the normalizer factorizes: $Z=(1+\beta_t^\eta)(1+\beta_s^\eta)$. So $p^{\mathrm{FT}}_{B2}=\frac{\beta_s^\eta}{(1+\beta_t^\eta)(1+\beta_s^\eta)}=\Pr(\text{persona}=B)\Pr(\text{domain}=2)$, and at $\eta=0$,
$$\frac{d}{d\eta}\log p_{B2}\Big|_{0}=\tfrac12\log(\beta_s/\beta_t),\qquad \frac{d}{d\eta}\log p_{A1}\Big|_{0}=\tfrac12\log(\beta_t/\beta_s),\qquad \frac{d}{d\eta}\log p_{B1}\Big|_{0}=-\tfrac12\log(\beta_t\beta_s)>0.$$
So in probability scale, $p_{B2}$ rises under weak tilt iff $\beta_s>\beta_t$ (persona-dominant regime) and is non-monotone in $\eta$ (as $\eta\to\infty$ all mass goes to B1). The robust transfer statistics are the log-odds gaps above; we register $\Delta p_{B2}>0$ only in the weak-to-moderate-dose, persona-dominant regime.

**P1 — hierarchical, $\beta_t=0.5$, $\beta_s=0.6$ (persona-dominant).** $\Delta\log p$ ordering $B1\gg B2>A1>A2$ with gaps $\eta(0.511,\,0.182,\,0.511)$ respectively ($-\ln 0.6=0.511$, $\ln(0.6/0.5)=0.182$); the $B2>A1$ comparison is the smallest gap, hence statistically hardest. Held-out $P(B2)$ rises (weak-tilt slope $+\tfrac12\ln 1.2\approx0.091$ per unit dose) while $P(A1),P(A2)$ fall. Ratio $R=\ln 0.5/\ln 0.6\approx 1.357$.

**P2 — flipped, $\beta_t=0.6$, $\beta_s=0.5$ (domain-dominant).** Ordering flips to $B1\gg A1>B2>A2$ (gaps $\eta(0.511,\,0.182,\,0.511)$). In probability scale $P(A1)$ rises and $P(B2)$ falls under weak tilt. Ratio $R=\ln 0.6/\ln 0.5\approx 0.737$.

**P3 — flat, $\beta=0.55$.** $\Delta\log p_{B2}=\Delta\log p_{A1}=\Delta\log p_{A2}=\eta\ln 0.55-\Delta\log Z$: all three non-FT leaves change by the same amount (each falls in probability, by $\tfrac14$-prior algebra: $p_{\text{other}}=\beta^\eta/(1+3\beta^\eta)<\tfrac14$). Both gaps are zero; $R$ is $0/0$ — registered as "all pairwise gaps among $\{B2,A1,A2\}$ statistically indistinguishable from 0". No preferential transfer.

**P4 — scrambled (at P1 betas).** Using the code's $\sigma$ ($A_r\leftrightarrow B_r$ for $r\ge2$; for $K=2$, $\sigma=(A2\,B2)$, fixing $A1$ and $B1$): after B1-pure FT, preferential transfer goes to **A2**, B1's pretraining group partner, with weights $(B1,A2,A1,B2)=(1,\,0.6,\,0.5,\,0.3)$. Ordering $B1\gg A2>A1>B2$; in probability scale $P(A2)$ rises while $P(B2)$ **falls** — the exact mirror of P1 with $A2\leftrightarrow B2$. Ratio invariant with swapped roles: $(\Delta\log p_{A2}-\Delta\log p_{B2})/(\Delta\log p_{A1}-\Delta\log p_{B2})=\log\beta_t/\log\beta_s\approx1.357$. Transfer follows pretraining correlation structure, not token labels.

**P5 — dose-response in $\beta_s$ at fixed $\beta_t$ (hierarchical).** The algebra, done carefully: $\Delta\log p_{B2}-\Delta\log p_{A2}=\eta[\log\beta_s-\log(\beta_t\beta_s)]=-\eta\log\beta_t$ — **independent of $\beta_s$**. The tilt model therefore predicts a *flat line* for the B2-vs-A2 log-odds transfer across a $\beta_s$ sweep, and in particular that persona transfer **survives $\beta_s\to0$** (domains nearly disjoint): $p_{B2}/p_{A2}$ is multiplied by $\beta_t^{-\eta}$ regardless of $\beta_s$. Meanwhile $\Delta\log p_{B2}-\Delta\log p_{B1}=\eta\log\beta_s$ rises to $0$ as $\beta_s\to1$ (B2 catches B1), and the absolute level $p^{\mathrm{FT}}_{B2}=\beta_s^\eta/[(1+\beta_t^\eta)(1+\beta_s^\eta)]$ is strictly increasing in $\beta_s$ (since $x\mapsto x/(1+x)$ is increasing), from $0$ at $\beta_s\to0$ to $\tfrac12(1+\beta_t^\eta)^{-1}$ at $\beta_s=1$. So "B2-transfer magnitude grows as $\beta_s\to1$" is true in probability scale but **false** in odds-vs-A2 scale, where the prediction is constancy. We register both curves; the constancy of $\log(p_{B2}/p_{A2})$ is the sharper falsifiable prediction (it assumes a fixed FT protocol so $\eta$ is comparable across the sweep).

**Where the tilt model is heuristic.** SGD is not Bayes: the tilt assumes fine-tuning acts on an internal leaf-prior parameter that exists, is movable, and is shared across contexts — i.e. it presumes precisely the factored representation of Corollary 1 whose behavioral relevance is what the experiment tests. The alternative hypothesis is representation-geometric: transfer magnitude tracks whether the learned representation contains the single shared persona coordinate $\theta$. The hierarchical optimum has it (Corollary 1); the flat optimum provably does not (Proposition 2). Both hypotheses predict P1; they separate on the controls. The hierarchical-vs-flat contrast is the key **qualitative** test (the tilt direction exists in both, but the shared coordinate only in one), and the ratio invariant $R=\log\beta_t/\log\beta_s$ is the **quantitative** one — it is dose-free, prior-free, and normalizer-free, so a clean miss is informative.

## 4. What would falsify the story

- **Flat control transfers as much as hierarchical** (comparable $\Delta\log p_{B2}-\Delta\log p_{A2}>0$ at matched evidence strength): transfer is generic fine-tuning spillover and the original objection stands.
- **Scrambled transfer follows labels, not groups** (B2 rises instead of A2 in P4): something other than pretraining correlation structure (e.g. token-embedding similarity from label sharing) drives transfer.
- **Ratio invariant wildly off and ordering violated** (P1 measured $R$ far from $1.36$ with reversed $B2$ vs $A1$, or P2 fails to flip): the tilt model is wrong as mechanism; if orderings still hold, the qualitative story survives with an unquantified mechanism.
- **No $\beta_s$-flatness in P5 odds scale** (e.g. $\log(p_{B2}/p_{A2})$ transfer collapses as $\beta_s\to0$): persona transfer requires domain overlap, contradicting the factored-coordinate account.

## Registered predictions (written before results) — 2026-06-12

- **P1** (hierarchical, $\beta_t{=}0.5$, $\beta_s{=}0.6$): held-out $P(B2)$ rises after pure-B1 FT; $\Delta\log p$ ordering $B1\gg B2>A1>A2$; ratio $R=(\Delta\log p_{B2}-\Delta\log p_{A2})/(\Delta\log p_{A1}-\Delta\log p_{A2})\approx\ln0.5/\ln0.6=1.357$.
- **P2** (flipped, $\beta_t{=}0.6$, $\beta_s{=}0.5$): ordering flips to $B1\gg A1>B2>A2$; $P(B2)$ falls in probability scale at weak dose; $R\approx\ln0.6/\ln0.5=0.737$.
- **P3** (flat, $\beta{=}0.55$): $B2$, $A1$, $A2$ change equally (all fall); pairwise gaps $\approx0$; no preferential transfer.
- **P4** (scrambled, $\sigma=(A2\,B2)$, groups $\{B1,A2\}$, $\{A1,B2\}$): preferential transfer to **A2** (B1's group partner), not B2; ordering $B1\gg A2>A1>B2$; mirrored ratio $\approx1.357$.
- **P5** (dose-response, $\beta_s$ sweep at fixed $\beta_t$): $\log(p_{B2}/p_{A2})$ transfer is **constant** at $-\eta\log\beta_t$ (persona transfer survives $\beta_s\to0$); absolute $\Delta p_{B2}$ increases monotonically in $\beta_s$; $\log(p_{B2}/p_{B1})$ rises to $0$ as $\beta_s\to1$.
