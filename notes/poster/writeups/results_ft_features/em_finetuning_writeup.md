# Z1R′×Z1R′ AFP with Prompt Mixing + Tunable Completion Collapse (HMM / GHMM)

This note defines an end-to-end **9-hidden-state** HMM (Mess3×Mess3) designed to test the “almost-factored process” (AFP) story in a controlled setting:

- **Prompt phase**: many distinct prompt strings; prompts can *reweight* the posterior mass between two macro-sectors while optionally **mixing** hidden state **within** a sector via permutation mixtures (identity included as a special case).
- **Completion phase**: completion tokens provide *soft evidence* for one macro-sector vs the other; the **rate of collapse** to a sector over a completion length $L$ is set by a single leakage parameter $\beta$ (with $\alpha=1$ WLOG once normalized).

We stick to the **$1\oplus 4$** macro-sector choice:
- $A := V_{11}$ is **1D** (the single state $(S0,S0)$).
- $B := V_{22}$ is **4D** (the four states $(S1,S1),(S1,SR),(SR,S1),(SR,SR)$).

---

## 0. Hidden state space and sector decomposition

Single-factor hidden state set:
$$
\{S0, S1, SR\}.
$$

Joint hidden state space is $A_{\text{fact}}\otimes B_{\text{fact}}$ with 9 basis states in lexicographic order:
$$
(S0,S0),(S0,S1),(S0,SR),(S1,S0),(S1,S1),(S1,SR),(SR,S0),(SR,S1),(SR,SR).
$$

We use the factor partition:
$$
\mathrm{span}\{S0\}\ \oplus\ \mathrm{span}\{S1,SR\}
$$

in each factor, which induces the tensor-product block decomposition
$$
A_{\text{fact}}\otimes B_{\text{fact}}
\cong
V_{11}\oplus V_{12}\oplus V_{21}\oplus V_{22}
$$

with block dimensions $1,2,2,4$.

For the AFP experiment we focus on the **matched** macro-sectors:
- **Macro-sector $A$**: $V_{11}=\mathrm{span}\{(S0,S0)\}$ (dimension 1)
- **Macro-sector $B$**: $V_{22}=\mathrm{span}\{(S1,S1),(S1,SR),(SR,S1),(SR,SR)\}$ (dimension 4)

We typically initialize with all probability mass in $V_{11}\oplus V_{22}$ (and zero in $V_{12},V_{21}$) and then only use token operators that preserve this support.

---

## 1. Choose prompt length $P$ and prompt vocabulary size $V_p$

Let the prompt alphabet be
$$
\mathcal{X}_{\text{prompt}}=\{p_1,\dots,p_{V_p}\}.
$$

Fix a prompt length $P$. Then the number of distinct prompt strings is
$$
|\mathcal{X}_{\text{prompt}}|^P = V_p^P.
$$

Example: $V_p=10, P=3\Rightarrow 1000$ distinct prompts.

---

## 2. Prompt token transfer matrices: weighted sums of permutations (identity included)

For each prompt token $p_k$, define a block-diagonal transfer matrix
$$
T(p_k) = T_A(p_k)\oplus T_B(p_k),
$$

supported only on the macro-sectors $A\oplus B$.

### 2.1 Block $A$ (dimension 1)
Row-stochastic on 1 state is just $[1]$. So the most general constant-row-sum form is
$$
T_A(p_k)=c_A(p_k)\,[1],
$$

where $c_A(p_k)>0$ is the “mass” assigned to token $p_k$ in sector $A$.

### 2.2 Block $B$ (dimension 4): permutation mixture
Pick a set of permutation matrices on the 4 states of $B$:
$$
\Pi_1=I,\ \Pi_2,\dots,\Pi_m,
$$

where $\Pi_1$ is the **identity permutation**.

Choose weights $w_{k,r}\ge 0$ with $\sum_{r=1}^m w_{k,r}=1$, and define
$$
S_B(p_k)=\sum_{r=1}^m w_{k,r}\Pi_r,
$$

which is row-stochastic. Then scale:
$$
T_B(p_k)=c_B(p_k)\,S_B(p_k).
$$

#### Special case: identity-only (recovers the old model)
If $w_{k,1}=1$ and $w_{k,r}=0$ for $r>1$, then
$$
S_B(p_k)=I,\qquad T_B(p_k)=c_B(p_k)\,I.
$$

### 2.3 Effect on macro-sector weights after a prompt

Let the macro-sector weights be $\pi_A,\pi_B$ (with $\pi_A+\pi_B=1$). For a prompt word $w=p_{i_1}\cdots p_{i_P}$,

$$
\frac{\pi_A(w)}{\pi_B(w)}=
\frac{\pi_A(\emptyset)}{\pi_B(\emptyset)}
\prod_{t=1}^P\frac{c_A(p_{i_t})}{c_B(p_{i_t})}.
$$

The permutation mixture affects only the **within-$B$** distribution; the macro-sector reweighting is controlled by the scalars $c_A/c_B$.

---

## 3. Choose completion vocabulary size and completion length $L$

Choose:
- completion length $L$,
- a base completion alphabet $\mathcal{S}=\{s_1,\dots,s_{V_c}\}$.

A natural choice in the Z1R×Z1R setup is $V_c=4$ with
$$
\mathcal{S}=\{\texttt{A},\texttt{B},\texttt{C},\texttt{D}\}
$$

encoding bit-pairs $(0,0),(0,1),(1,0),(1,1)$.

We then form a **tagged completion alphabet**:
$$
\mathcal{X}_{\text{comp}}=
\{\texttt{g}s : s\in\mathcal{S}\}
\ \cup\
\{\texttt{b}s : s\in\mathcal{S}\},
$$

so the completion vocab size is $2V_c$.

---

## 4. Define the base Z1R′($\delta$)×Z1R′($\delta$) process and impose normalization

### 4.1 Z1R′($\delta$): a minimal generalization so $u_s>0$ for all $s$
In the original Z1R′, $S0$ emits 0 deterministically, which forces $(S0,S0)$ to emit only $\texttt{A}$. To allow a nontrivial completion alphabet while keeping the $1\oplus 4$ macro-sector choice, we introduce a small “leak” $\delta\in(0,1)$:

Single-factor token matrices (state order $(S0,S1,SR)$):
$$
T_0^{(\delta)}=
\begin{pmatrix}
1-\delta & 0 & 0\\
0 & 0 & 0\\
0 & \tfrac12 & 0
\end{pmatrix},
\qquad
T_1^{(\delta)}=
\begin{pmatrix}
\delta & 0 & 0\\
0 & 0 & 1\\
0 & \tfrac12 & 0
\end{pmatrix}.
$$

This keeps the net operator row-stochastic, but makes $S0$ emit 1 with small probability $\delta$.

### 4.2 Joint base-symbol matrices for $\texttt{A,B,C,D}$
Let $\texttt{A}=(0,0)$, $\texttt{B}=(0,1)$, $\texttt{C}=(1,0)$, $\texttt{D}=(1,1)$. Define:
$$
T_{\texttt{A}}=T_0^{(\delta)}\otimes T_0^{(\delta)},\quad
T_{\texttt{B}}=T_0^{(\delta)}\otimes T_1^{(\delta)},\quad
T_{\texttt{C}}=T_1^{(\delta)}\otimes T_0^{(\delta)},\quad
T_{\texttt{D}}=T_1^{(\delta)}\otimes T_1^{(\delta)}.
$$

### 4.3 Block restriction and the scalars $u_s, v_s$
Let $P_A$ be the projector onto the 1D macro-sector $A=V_{11}$ and $P_B$ the projector onto the 4D macro-sector $B=V_{22}$.

For each base completion symbol $s\in\mathcal{S}$, define restricted matrices:
$$
M_A(s)=P_A T_s P_A,\qquad M_B(s)=P_B T_s P_B.
$$

Define scalars:
- $u_s$: the row-sum scalar for $M_A(s)$. Since $A$ is 1D, $u_s$ is simply the single entry of $M_A(s)$.
- $v_s$: the row-sum scalar for $M_B(s)$. If $M_B(s)$ has constant row sum, then $M_B(s)\mathbf{1}_B=v_s\mathbf{1}_B$. (In practice we can enforce this by construction or take a scalar summary like mean row sum as an approximation.)

### 4.4 Normalization
Pick a target mass $m_s>0$ (often set $m_s=1$ for all $s$). Define:
$$
\widehat M_A(s)=\frac{m_s}{u_s}M_A(s),\qquad
\widehat M_B(s)=\frac{m_s}{v_s}M_B(s).
$$

Then
$$
\widehat M_A(s)\mathbf 1_A = m_s\mathbf 1_A,\qquad
\widehat M_B(s)\mathbf 1_B = m_s\mathbf 1_B,
$$

so both sectors assign the same total mass $m_s$ to symbol $s$.

---

## 5. Completion tokens with tunable leakage; set $\alpha=1$ WLOG

For each base completion symbol $s\in\mathcal{S}$, define tagged completion token matrices:

$$
T(\texttt{g}s) = \alpha\,\widehat M_A(s) + \beta\,\widehat M_B(s),
$$

$$
T(\texttt{b}s) = \beta\,\widehat M_A(s) + \alpha\,\widehat M_B(s),
$$

with $0<\beta<\alpha$.

Because belief-state updates are normalized, only relative scales matter; we can set $\alpha=1$ WLOG and treat $\beta\in(0,1)$ as the **single collapse-rate knob**.

### 5.1 Collapse rate over completion length $L$
Assume $\pi_A=\pi_B=\tfrac12$ after the prompt and we observe $L$ “good-tagged” completion tokens $\texttt{g}\*$ with the normalized construction above. Then the macro-sector odds multiply by $(1/\beta)$ per token:

$$
\frac{\pi_A^{(L)}}{\pi_B^{(L)}} = \left(\frac{1}{\beta}\right)^L,
\qquad
\pi_A^{(L)}=\frac{(1/\beta)^L}{1+(1/\beta)^L}.
$$

To target $\pi_A^{(L)}=p$, choose
$$
\beta = \left(\frac{1-p}{p}\right)^{1/L}.
$$

Example: $p=0.95, L=5 \Rightarrow \beta \approx (0.05/0.95)^{1/5}\approx 0.556$.

---

## 6. The resulting HMM (final definition)

### Alphabet
$$
\mathcal X = \mathcal X_{\text{prompt}} \cup \mathcal X_{\text{comp}},
$$

with size $V_p + 2V_c$.

### Token-indexed transfer matrices
- For each prompt token $p_k$: $T(p_k)$ from Section 2 (block-diagonal on $A\oplus B$, optionally permuting within $B$).
- For each completion token $\texttt{g}s,\texttt{b}s$: $T(\texttt{g}s),T(\texttt{b}s)$ from Section 5 (using normalized blocks).

All $T(x)$ are $9\times 9$, nonnegative, and preserve the $A\oplus B$ support (if initialized there).

### Initial state
Choose an initial predictive distribution on hidden states with desired macro-sector prior:
- place $\pi_A(\emptyset)$ on $(S0,S0)$,
- distribute $\pi_B(\emptyset)$ over the four states in $B$ (uniform or chosen),
- set mixed-sector mass to zero if desired.

---

## Summary: knobs you control

- **Prompt count:** $V_p^P$ possible prompts.
- **Prompt reweighting:** scalars $c_A(p_k), c_B(p_k)$ control $\pi_A/\pi_B$ after the prompt.
- **Within-sector prompt mixing:** permutation mixture weights $w_{k,r}$ (identity included).
- **Completion diversity:** base completion vocab size $V_c$ (e.g. 4 for $\texttt{A,B,C,D}$) and completion length $L$.
- **Collapse rate:** leakage $\beta$ (with $\alpha=1$ after normalization).
- **Process richness in 1⊕4:** $\delta$ controls whether $(S0,S0)$ can emit multiple symbols in $\mathcal{S}$.

This yields a compact HMM testbed where prompt-driven sector reweighting and completion-driven sector collapse are analytically controllable.
