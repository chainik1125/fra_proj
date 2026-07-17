# Hierarchical Leaky-Reset AFP (Two-Level) - Math Writeup

This note gives a clean mathematical extension of the current `leaky_reset` process to a two-level hierarchy.

Implementation companion:
- [hierarchical_leaky_reset_implementation.md](./hierarchical_leaky_reset_implementation.md)
- Alternative process (gate-HMM coefficients): [hierarchical_gate_hmm_leaky_reset.md](./hierarchical_gate_hmm_leaky_reset.md)

Target structure:
- Two equal-weight super-sectors: `A` and `B`.
- Each super-sector has two child leaky-reset processes:
  - `A1`, `A2` inside `A`
  - `B1`, `B2` inside `B`

So the hidden space is a direct sum of four leaf sectors.

---

## 1) Hidden State Factorization

Let each leaf sector `ell in {A1, A2, B1, B2}` have `d_ell` latent states:

$$
\mathcal{H}
= H_{A1} \oplus H_{A2} \oplus H_{B1} \oplus H_{B2}, \quad |H_\ell| = d_\ell.
$$

Write the belief as:

$$
\pi = \big(\pi_{A1}\mu_{A1},\ \pi_{A2}\mu_{A2},\ \pi_{B1}\mu_{B1},\ \pi_{B2}\mu_{B2}\big),
$$

where:
- `pi_ell` is leaf mass (`sum_j pi_{ell,j}`),
- `mu_ell in Delta^{d_ell-1}` is within-leaf belief.

Two-level decomposition:

$$
\pi_A = \pi_{A1} + \pi_{A2}, \qquad
\pi_B = \pi_{B1} + \pi_{B2},
$$

$$
\rho_{A1|A} = \frac{\pi_{A1}}{\pi_A},\ \rho_{A2|A} = \frac{\pi_{A2}}{\pi_A},
\qquad
\rho_{B1|B} = \frac{\pi_{B1}}{\pi_B},\ \rho_{B2|B} = \frac{\pi_{B2}}{\pi_B}.
$$

Then `pi = (pi_A, pi_B)` is level-1 mass, `rho_{.|A}, rho_{.|B}` are level-2 masses, and `mu_ell` are within-leaf states.

---

## 2) Prompt Phase: Hierarchical Leaky Reset

For each prompt token `p_k`, define a leaky reset inside each leaf:

$$
S_{\ell,k} = (1-\lambda_\ell)I_{d_\ell} + \lambda_\ell R_{\ell,k},
$$

with

$$
R_{\ell,k} = \mathbf{1}\, r_{\ell,k}^\top, \quad r_{\ell,k} \in \Delta^{d_\ell-1}.
$$

Use a shared scalar coefficient `c_k` (prompt-neutral choice: `c_k = 1/V_p`):

$$
T_{\text{prompt}}(p_k)
= c_k \cdot
\mathrm{diag}\!\big(S_{A1,k}, S_{A2,k}, S_{B1,k}, S_{B2,k}\big).
$$

Equivalent belief update at leaf level:

$$
\mu_\ell' = (1-\lambda_\ell)\mu_\ell + \lambda_\ell r_{\ell,k}.
$$

If every block gets the same `c_k` and each `S_{ell,k}` is row-stochastic, then all leaf masses are invariant:

$$
\pi_{A1}',\pi_{A2}',\pi_{B1}',\pi_{B2}' \propto
\pi_{A1},\pi_{A2},\pi_{B1},\pi_{B2},
$$

so after normalization they are unchanged. Therefore prompt tokens only write into `mu_ell`, not into top/sub masses.

---

## 3) Completion Phase: Two-Level Tagged Evidence

Use leaf-tagged completion tokens:

$$
x_{(s^\star, r^\star), i},
$$

where:
- `s* in {A,B}` is target super-sector,
- `r* in {1,2}` is target child index inside that super-sector,
- `i in {0,...,M-1}` is content index.

For each leaf `ell=(s,r)`, define emission diagonal:

$$
D_{\ell}(i) = \mathrm{diag}(e_{\ell,i}), \qquad
e_{\ell,i}[j] =
\begin{cases}
1-\delta & \text{if } j=i < d_\ell,\\
\delta/(M-1) & \text{otherwise}.
\end{cases}
$$

Define hierarchical evidence weights:

$$
w_\ell\big((s^\star,r^\star)\big)
=
\beta_{\text{top}}^{\mathbf{1}[s \neq s^\star]}
\cdot
\beta_{\text{sub}}^{\mathbf{1}[r \neq r^\star]}.
$$

Then

$$
T_{\text{comp}}\!\big(x_{(s^\star,r^\star), i}\big)
=
\bigoplus_{\ell \in \{A1,A2,B1,B2\}}
w_\ell\big((s^\star,r^\star)\big)\, D_{\ell}(i).
$$

This is the hierarchical analogue of the original `1` vs `beta` block scaling:
- `beta_top` controls super-sector evidence (`A` vs `B`),
- `beta_sub` controls child evidence (`1` vs `2`).

As in current `leaky_reset`, normalize across tokens so the net transition matrix is row-stochastic.

---

## 4) Posterior Update Equations

After observing `x_{(s*,r*), i}`:

$$
\tilde{\pi}_{\ell,j}
=
\pi_{\ell,j}\, w_\ell((s^\star,r^\star))\, e_{\ell,i}[j].
$$

Define leaf likelihood term:

$$
L_\ell(i) = \mu_\ell^\top e_{\ell,i}.
$$

Then unnormalized leaf masses are:

$$
\tilde{\pi}_\ell = \pi_\ell\, w_\ell((s^\star,r^\star))\, L_\ell(i),
$$

and normalized:

$$
\pi_\ell' = \frac{\tilde{\pi}_\ell}{\sum_m \tilde{\pi}_m}.
$$

Within-leaf belief update:

$$
\mu_{\ell,j}' \propto \mu_{\ell,j}\, e_{\ell,i}[j].
$$

From `pi_ell'` you recover top/sub masses:

$$
\pi_A' = \pi_{A1}' + \pi_{A2}', \quad
\pi_B' = \pi_{B1}' + \pi_{B2}',
$$

$$
\rho_{A1|A}'=\frac{\pi_{A1}'}{\pi_A'},\ \rho_{A2|A}'=\frac{\pi_{A2}'}{\pi_A'},
\quad
\rho_{B1|B}'=\frac{\pi_{B1}'}{\pi_B'},\ \rho_{B2|B}'=\frac{\pi_{B2}'}{\pi_B'}.
$$

---

## 5) Explicit Worked Example (Two-Level Hierarchy)

### 5.1 Parameters

- Super-sector priors: `pi_A = pi_B = 0.5`
- Child priors inside each super-sector: `rho_{1|A} = rho_{2|A} = 0.5`, `rho_{1|B} = rho_{2|B} = 0.5`
- So initial leaf masses are all `0.25`.
- Leaf dimensions: `d_{A1}=d_{A2}=d_{B1}=d_{B2}=2`
- Prompt leaky reset: `lambda_ell = 0.6` for all leaves
- Prompt signatures (`onehot`, cycling mod 2):
  - `r_{ell,0} = [1,0]`
  - `r_{ell,1} = [0,1]`
- Initial within-leaf beliefs: `mu_ell^(0) = [0.5, 0.5]`

Completion:
- `M = 2`, `delta = 0.05`
- `beta_top = 0.6`, `beta_sub = 0.5`

### 5.2 Prompt sequence `p0, p1, p0`

For each leaf (same numbers here):

$$
\mu^{(1)} = 0.4[0.5,0.5] + 0.6[1,0] = [0.8,0.2]
$$

$$
\mu^{(2)} = 0.4[0.8,0.2] + 0.6[0,1] = [0.32,0.68]
$$

$$
\mu^{(3)} = 0.4[0.32,0.68] + 0.6[1,0] = [0.728,0.272].
$$

Leaf/super masses are unchanged through prompt:

$$
(\pi_{A1},\pi_{A2},\pi_{B1},\pi_{B2})=(0.25,0.25,0.25,0.25),
\quad
(\pi_A,\pi_B)=(0.5,0.5).
$$

### 5.3 One completion token: `x_(A1, i=0)`

Emission vector for `i=0` with `M=2, delta=0.05`:

$$
e_{\ell,0} = [0.95,\ 0.05].
$$

Leaf likelihoods (all equal here):

$$
L_\ell(0) = [0.728,0.272]\cdot[0.95,0.05] = 0.7052.
$$

Hierarchical weights for target leaf `A1`:

- `w_{A1} = 1`
- `w_{A2} = beta_sub = 0.5`
- `w_{B1} = beta_top = 0.6`
- `w_{B2} = beta_top * beta_sub = 0.3`

Unnormalized leaf masses:

$$
\tilde{\pi}_{A1}=0.25\cdot1\cdot0.7052=0.1763
$$
$$
\tilde{\pi}_{A2}=0.25\cdot0.5\cdot0.7052=0.08815
$$
$$
\tilde{\pi}_{B1}=0.25\cdot0.6\cdot0.7052=0.10578
$$
$$
\tilde{\pi}_{B2}=0.25\cdot0.3\cdot0.7052=0.05289
$$

`Z = 0.42312`, so:

$$
(\pi_{A1}',\pi_{A2}',\pi_{B1}',\pi_{B2}')
=
(0.4167,\ 0.2083,\ 0.2500,\ 0.1250).
$$

Top-level posterior:

$$
(\pi_A',\pi_B')=(0.625,\ 0.375).
$$

Child posteriors:

$$
(\rho_{1|A}',\rho_{2|A}')=(2/3,\ 1/3), \qquad
(\rho_{1|B}',\rho_{2|B}')=(2/3,\ 1/3).
$$

So one token updates both levels:
- favors `A` over `B` via `beta_top`,
- favors child `1` over `2` via `beta_sub`.

Within-leaf update (example for `A1`):

$$
\mu_{A1}' \propto [0.728,0.272] \odot [0.95,0.05]
= [0.6916,0.0136]
\Rightarrow
\mu_{A1}'=[0.9807,0.0193].
$$

---

## 6) Practical Takeaway

This extension stays a standard finite HMM after flattening leaf states. The hierarchy is represented by:
- block structure in hidden space,
- shared prompt leaky-reset form per leaf,
- hierarchical scaling factors in completion (`beta_top`, `beta_sub`),
- decomposition of posterior into top/sub/within-leaf components.

This is a direct generalization of current `leaky_reset` (`2` blocks) to `4` leaf blocks under a two-level tree.
