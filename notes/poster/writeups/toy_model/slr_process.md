# Leaky-Reset AFP: Transition Matrix Reference

Notes on the exact transition matrices in the `leaky_reset` variant of the AFP process, as built by `build_leaky_reset_hmms()` in `analysis/afp_builders.py`.

---

## 1) Hidden State Space

The hidden state space is a direct sum of two sectors:

$$
\mathcal{H} = H_G \oplus H_B
$$

where $|H_G| = d_g$ and $|H_B| = d_b$. With default config ($d_g = d_b = 5$), we have 10 total hidden states: states 0-4 are sector G, states 5-9 are sector B.

A belief state $\pi \in \mathbb{R}^{10}$ decomposes as:

$$
\pi = (\pi_G \cdot \mu_G,\;\; \pi_B \cdot \mu_B)
$$

where:
- $\pi_G = \sum_{i \in H_G} \pi_i$ is the **sector mass** (probability of being in sector G)
- $\mu_G \in \Delta^{d_g - 1}$ is the **within-sector belief** (normalized distribution over G states)
- Similarly for sector B

The initial state is uniform within each sector: $\pi_G = \pi_A$, $\mu_G = \text{uniform}(d_g)$.

---

## 2) Prompt Phase: Leaky Reset (Sector-Neutral)

**Vocabulary:** $V_p$ prompt tokens $p_0, \ldots, p_{V_p - 1}$.

**Transition matrix for token $p_k$:**

$$
T_{\text{prompt}}(p_k) = \frac{1}{V_p} \begin{pmatrix} S_G(p_k) & 0 \\ 0 & S_B(p_k) \end{pmatrix}
$$

where the within-sector update is a **leaky reset**:

$$
S_S(p_k) = (1 - \lambda_S)\, I_{d_S} + \lambda_S\, R_{S,k}
$$

and $R_{S,k}$ is the rank-1 matrix $\mathbf{1} \cdot r_{S,k}^\top$ (every row is the signature vector $r_{S,k}$).

**Are $S_G$ and $S_B$ the same matrix?** Yes, in the default config. The signature $r_{S,k}$ depends only on $k \bmod d_S$ and the signature type. Since $d_g = d_b$ and $\lambda_g = \lambda_b$, we have $r_{G,k} = r_{B,k}$ for all $k$, and therefore $S_G(p_k) = S_B(p_k)$ exactly. The two sectors receive identical within-sector dynamics during the prompt. They would differ if $d_g \neq d_b$ (different cycling period for the one-hot signatures) or $\lambda_g \neq \lambda_b$ (different reset strengths).

### Signatures and constructive form of $S_G$

With `signature_type="onehot"`, the signature for token $p_k$ is $r_{S,k} = e_{k \bmod d}$, a one-hot vector cycling through hidden states. Since $R_{S,k} = \mathbf{1} \cdot e_{k \bmod d}^\top$, the matrix $S_G(p_k)$ has a concrete form: **column $k \bmod d$ gets weight $(1 - \lambda) + \lambda = 1$, all other columns get weight $(1 - \lambda)$, except every row has its reset mass in column $k \bmod d$.**

Explicitly, for the case $j = k \bmod d_g$:

$$
S_G(p_k)_{ij} = \begin{cases}
(1 - \lambda) + \lambda = 1 & \text{if } i = j \\
\lambda & \text{if } i \neq j, \; j = k \bmod d_g \\
(1 - \lambda) & \text{if } i = j, \; j \neq k \bmod d_g \\
0 & \text{otherwise}
\end{cases}
$$

More compactly: $S_G(p_k) = (1 - \lambda)\, I + \lambda\, \mathbf{1}\, e_j^\top$ where $j = k \bmod d_g$. Each row sums to 1 (row-stochastic).

### Example: $d_g = 3$, $\lambda = 0.6$

There are 3 distinct within-sector matrices (one per target state):

$$
S_G(p_0) = S_G(p_3) = \cdots = \begin{pmatrix} 1.0 & 0 & 0 \\ 0.6 & 0.4 & 0 \\ 0.6 & 0 & 0.4 \end{pmatrix}
$$

$$
S_G(p_1) = S_G(p_4) = \cdots = \begin{pmatrix} 0.4 & 0.6 & 0 \\ 0 & 1.0 & 0 \\ 0 & 0.6 & 0.4 \end{pmatrix}
$$

$$
S_G(p_2) = S_G(p_5) = \cdots = \begin{pmatrix} 0.4 & 0 & 0.6 \\ 0 & 0.4 & 0.6 \\ 0 & 0 & 1.0 \end{pmatrix}
$$

Reading $S_G(p_0)$ row by row:
- **Row 0** (state 0): already at the target state → stays at state 0 with probability 1
- **Row 1** (state 1): jumps to state 0 with probability $\lambda = 0.6$, stays at state 1 with probability $1 - \lambda = 0.4$
- **Row 2** (state 2): jumps to state 0 with probability 0.6, stays at state 2 with probability 0.4

Each matrix is a "leaky reset to state $j$": with probability $\lambda$ the state resets to $j$, with probability $1 - \lambda$ it stays put.

### What the leaky reset does to beliefs

After observing prompt token $p_k$, the within-sector belief updates as:

$$
\mu_S \;\leftarrow\; (1 - \lambda_S)\, \mu_S + \lambda_S\, e_{k \bmod d}
$$

This is a convex combination: fraction $\lambda_S$ of the belief is "reset" to the one-hot signature $e_{k \bmod d}$, while fraction $(1 - \lambda_S)$ retains the previous belief. With $\lambda = 0.6$, each prompt token overwrites 60% of the within-sector state.

### Prompt neutrality (key property)

The scalar prefactor $1/V_p$ is the **same for both sectors** and is independent of $k$. This means:

$$
\pi_G \;\leftarrow\; \frac{1/V_p \cdot \pi_G \cdot \|S_G(p_k)\, \mu_G\|_1}{Z} = \frac{\pi_G}{Z'}
$$

Since $S_G(p_k)$ is row-stochastic (rows sum to 1), $\|S_G \mu_G\|_1 = \|\mu_G\|_1 = 1$, and the same holds for sector B. Therefore:

$$
\boxed{\text{Prompt tokens never change } \pi_G / \pi_B}
$$

The sector mass ratio is frozen throughout the entire prompt phase. Prompts **only** write information into the within-sector beliefs $\mu_G$ and $\mu_B$.

### After the prompt

After a prompt sequence $(p_{k_1}, p_{k_2}, p_{k_3})$ of length 3 with $\lambda = 0.6$, the within-sector belief is:

$$
\mu_S = 0.4^3\, \mu_S^{(0)} + 0.6 \cdot 0.4^2\, r_{S,k_1} + 0.6 \cdot 0.4\, r_{S,k_2} + 0.6\, r_{S,k_3}
$$

i.e. an exponentially-decaying weighted average of the signatures, most recent dominating. Since $0.4^3 = 0.064$, the initial uniform prior is nearly washed out — the prompt effectively encodes a specific within-sector state.

With 5 prompt tokens and 3 positions, there are $5^3 = 125$ distinct prompts. Since $d_g = d_b = V_p = 5$ and signatures are one-hot, each prompt produces a unique $(\mu_G, \mu_B)$ pair (125 unique within-sector beliefs, as confirmed by the pipeline analysis).

**But crucially: all 125 prompts produce the same $\pi_G = 0.5$.** The prompt is informative about *where within each sector*, but carries zero information about *which sector*.

---

## 3) Completion Phase: Tagged Symbols with Sector Evidence

**Vocabulary:** $2M$ completion tokens, split into two groups:
- **G-tagged** (good): $g_0, \ldots, g_{M-1}$ (token indices $0 \ldots M-1$, displayed as $V_p \ldots V_p + M - 1$ after offset)
- **B-tagged** (bad): $b_0, \ldots, b_{M-1}$ (token indices $M \ldots 2M-1$, displayed as $V_p + M \ldots V_p + 2M - 1$ after offset)

### Transition matrix for G-tagged token $g_i$:

$$
T_{\text{comp}}(g_i) \propto \begin{pmatrix} 1 \cdot D_G(i) & 0 \\ 0 & \beta \cdot D_B(i) \end{pmatrix}
$$

### Transition matrix for B-tagged token $b_i$:

$$
T_{\text{comp}}(b_i) \propto \begin{pmatrix} \beta \cdot D_G(i) & 0 \\ 0 & 1 \cdot D_B(i) \end{pmatrix}
$$

where $D_S(i)$ is a diagonal **emission matrix** for content index $i$ in sector $S$.

### What $D_S(i)$ is

$D_S(i)$ is a $d_S \times d_S$ diagonal matrix. Here $i \in \{0, \ldots, M-1\}$ indexes the **content symbol** (which of the $M$ possible content indices was observed), and $j \in \{0, \ldots, d_S - 1\}$ indexes the **hidden state** within sector $S$.

The construction uses an **identity label map** $\phi(j) = j$: hidden state $j$ is associated with content symbol $j$. This is a design choice — the mapping could be arbitrary, but the code hardcodes it as identity (`_readout_diag` sets `E[j, j] = 1 - δ`).

$D_S(i) = \text{diag}(e_{S,i})$ where the emission vector $e_{S,i} \in \mathbb{R}^{d_S}$ is:

$$
e_{S,i}[j] = \begin{cases} 1 - \delta & \text{if } j = i < d_S \\ \delta / (M-1) & \text{otherwise} \end{cases}
$$

($\delta$ = `decode_noise` = 0.05 by default.)

When $i < d_S$: content symbol $i$ matches hidden state $i$, so state $i$ gets high weight $(1 - \delta)$ and all other states get low weight $\delta/(M-1)$. This is the HMM analogue of a soft observation "the hidden state is probably $i$".

When $i \geq d_S$: no hidden state matches this content symbol, so all entries are the uniform noise floor $\delta/(M-1)$. Such tokens carry tag information (sector evidence) but no within-sector content signal. In the default config $M = d_S = 5$, so this case doesn't arise — every content symbol maps to exactly one hidden state.

$D_S(i)$ is diagonal: it does not move probability between hidden states. It **reweights** them according to how well each state explains the observed content symbol.

### Explicit $D_S$ matrices for $d_S = M = 3$, $\delta = 0.05$

Each content symbol $i$ has a "partner" hidden state $j = i$:

$$
D_S(0) = \begin{pmatrix} 0.95 & 0 & 0 \\ 0 & 0.025 & 0 \\ 0 & 0 & 0.025 \end{pmatrix}, \quad
D_S(1) = \begin{pmatrix} 0.025 & 0 & 0 \\ 0 & 0.95 & 0 \\ 0 & 0 & 0.025 \end{pmatrix}, \quad
D_S(2) = \begin{pmatrix} 0.025 & 0 & 0 \\ 0 & 0.025 & 0 \\ 0 & 0 & 0.95 \end{pmatrix}
$$

Reading $D_S(0)$: "we observed content symbol 0, which is partnered with hidden state 0 — so state 0 keeps 95% of its weight, while states 1 and 2 are crushed to 2.5%."

These are **not** row-stochastic (rows don't sum to 1). That's intentional — they represent likelihood weights, not transition probabilities. The row-stochastic normalization happens at the level of the full $T_{\text{comp}}$.

### Full completion transition matrix for $d_g = d_b = 3$, $\beta = 0.6$

The full hidden state space has 6 states: G = {0,1,2}, B = {3,4,5}. There are $2M = 6$ completion tokens.

**G-tagged token $g_0$** (content index 0, favours sector G):

$$
T_{\text{comp}}(g_0) \propto \begin{pmatrix} 0.95 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0.025 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0.025 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.57 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0.015 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0.015 \end{pmatrix}
$$

The top-left block is $1 \cdot D_G(0)$, the bottom-right is $\beta \cdot D_B(0) = 0.6 \cdot D_B(0)$. Off-diagonal blocks are zero (sectors never mix). Note how G-state-0 has weight 0.95 but B-state-0 has only $0.6 \times 0.95 = 0.57$ — this asymmetry is the sector evidence.

**B-tagged token $b_0$** (content index 0, favours sector B):

$$
T_{\text{comp}}(b_0) \propto \begin{pmatrix} 0.57 & 0 & 0 & 0 & 0 & 0 \\ 0 & 0.015 & 0 & 0 & 0 & 0 \\ 0 & 0 & 0.015 & 0 & 0 & 0 \\ 0 & 0 & 0 & 0.95 & 0 & 0 \\ 0 & 0 & 0 & 0 & 0.025 & 0 \\ 0 & 0 & 0 & 0 & 0 & 0.025 \end{pmatrix}
$$

Mirror image: now sector B gets the full weight and sector G is attenuated by $\beta$.

### Normalization

The code normalizes so the net matrix $\sum_x T_{\text{comp}}(x)$ is row-stochastic. For each state, the net outgoing mass sums over all $2M$ tokens. Since every token applies the same $D_S(i)$ structure to both sectors (just with different scaling), the net mass from any G-state is $\sum_i e_{G,i}[j] \cdot (1 + \beta) = (1 + \beta)$ (emission vectors sum to 1 over $i$). So normalization divides everything by $(1 + \beta)$.

After normalization, the matrices above become:

$$
T_{\text{comp}}(g_0) = \frac{1}{1.6} \begin{pmatrix} 0.95 & \cdots \\ & \ddots & \\ \cdots & & 0.015 \end{pmatrix}
$$

### How completion tokens carry two kinds of information

Each completion token $g_i$ or $b_i$ carries two orthogonal signals:

**1. Tag signal (sector evidence):** The asymmetric scaling ($1$ vs $\beta$) between blocks provides evidence about which sector generated the token. Looking at the diagonal entries for the *same* hidden state position across the two blocks:
- In $T(g_0)$: G-state-0 has weight 0.95, B-state-0 has weight 0.57. Ratio = $1/\beta$.
- In $T(b_0)$: reversed.

Each token provides a Bayes factor of $1/\beta$ in favor of the tagged sector. Over multiple tokens, sector mass polarizes toward whichever tag dominates.

**2. Content signal (within-sector state):** Within each block, the diagonal $D_S(i)$ reweights states. Looking at the G-block of $T(g_0)$: state 0 has weight 0.95, states 1 and 2 have weight 0.025. This is a likelihood ratio of $0.95 / 0.025 = 38:1$ in favour of "the within-G state is 0". So the content index $i$ provides strong evidence about which hidden state within each sector is active.

With $\delta = 0.05$, the content signal is quite sharp — observing $g_i$ strongly indicates the within-sector-G state is $i$.

### Completion dynamics: belief update

After observing completion token $g_i$, the unnormalized belief update is just element-wise multiplication by the diagonal of $T(g_i)$:

$$
\tilde{\pi}_{G,j} = \pi_{G,j} \cdot \frac{1}{1+\beta} \cdot e_{G,i}[j], \qquad
\tilde{\pi}_{B,j} = \pi_{B,j} \cdot \frac{\beta}{1+\beta} \cdot e_{B,i}[j]
$$

then normalize $\pi' = \tilde\pi / \|\tilde\pi\|_1$.

This decomposes cleanly into sector mass and within-sector updates:

$$
\pi_G' = \frac{1 \cdot \pi_G \cdot (\mu_G^\top e_{G,i})}{Z}, \qquad
\pi_B' = \frac{\beta \cdot \pi_B \cdot (\mu_B^\top e_{B,i})}{Z}
$$

where $\mu_G^\top e_{G,i} = \sum_j \mu_{G,j} \cdot e_{G,i}[j]$ is the expected emission weight under the current within-sector belief. The within-sector beliefs update as:

$$
\mu_{G,j}' \propto \mu_{G,j} \cdot e_{G,i}[j], \qquad
\mu_{B,j}' \propto \mu_{B,j} \cdot e_{B,i}[j]
$$

Note: because the matrix is diagonal, the belief update is just **reweighting** — no probability flows between states. This is very different from the prompt phase where the leaky reset actively moves mass between states.

### How the prompt mixture feeds into completion

After the prompt, the within-sector belief $\mu_G$ is a genuine mixture over hidden states — e.g. $\mu_G = (0.64, 0.24, 0.08, 0.02, 0.02)$ for a particular prompt sequence with $d_g = 5$. How does this mixture interact with the completion phase?

**Generative story:** At time 0, a true hidden state $(S^*, j^*)$ is sampled from the initial distribution. During the prompt, the true state evolves: the leaky-reset matrices have off-diagonal entries ($S_G(p_k)$ can transition from G-state-1 to G-state-0 with probability $\lambda$). So the true hidden state **does** change during the prompt — this is a real stochastic process, not just belief reweighting. But once completion starts, the diagonal $T_{\text{comp}}$ matrices freeze whatever state the system landed in. The true state at the end of the prompt persists for all of completion.

**Observer's story (belief tracking):** The observer doesn't know $(S^*, j^*)$. After the prompt, they have a belief $\pi = (\pi_G \cdot \mu_G, \; \pi_B \cdot \mu_B)$ — a mixture reflecting uncertainty about both sector and within-sector state.

The first completion token is drawn from the **marginal** over this mixture:

$$
P(\text{token } x) = \sum_j \pi_j \cdot P(x \mid \text{state } j)
$$

where $P(x \mid j) = T_{\text{comp}}(x)[j, j]$ (the diagonal entry). Expanding for a G-tagged token $g_i$:

$$
P(g_i) = \sum_{j \in G} \pi_{G,j} \cdot \frac{1}{1+\beta} e_{G,i}[j] \;+\; \sum_{j \in B} \pi_{B,j} \cdot \frac{\beta}{1+\beta} e_{B,i}[j]
$$

The prompt-encoded mixture $\mu_G$ directly controls which content indices are likely. If the prompt put most G-mass on state 2 ($\mu_{G,2} \approx 0.64$), then $g_2$ is the most likely G-tagged token, because $e_{G,2}[2] = 0.95$ gets the most weight in the sum.

**After observing that token**, Bayes' rule reweights the mixture:

$$
\pi_j' \propto \pi_j \cdot T_{\text{comp}}(x)[j, j]
$$

This is the key loop: **the mixture determines which tokens are likely, and the observed token sharpens the mixture.** Each completion token is evidence about the true hidden state, gradually resolving the uncertainty that the prompt created.

Concretely, with $\delta = 0.05$: if we observe $g_2$, then within sector G, state 2's mass is multiplied by 0.95 while all other states are multiplied by 0.025. After just one or two tokens, the within-sector belief is almost a delta. The sector mass update is slower (Bayes factor $1/\beta$ per token), so it takes more tokens to resolve sector uncertainty.

### Completion as a static channel (no hidden-state dynamics)

Because all $T_{\text{comp}}$ matrices are diagonal, the hidden state **never transitions** during completion. If the HMM enters completion in true hidden state $j$ (in sector $S$), it stays in state $j$ for the entire completion. The completion tokens are repeated noisy emissions from this fixed state — a static channel, not a dynamical system.

The emission distribution from a fixed G-state $j$ (after normalization by $1 + \beta$) is:

| Token | Probability | Interpretation |
|-------|------------|----------------|
| $g_j$ | $\frac{1}{1+\beta}(1 - \delta)$ | right tag + right content |
| $b_j$ | $\frac{\beta}{1+\beta}(1 - \delta)$ | wrong tag + right content |
| $g_{i \neq j}$ | $\frac{1}{1+\beta} \cdot \frac{\delta}{M-1}$ | right tag + wrong content |
| $b_{i \neq j}$ | $\frac{\beta}{1+\beta} \cdot \frac{\delta}{M-1}$ | wrong tag + wrong content |

With default parameters ($\beta = 0.6$, $\delta = 0.05$, $M = 5$):

| Token | Probability |
|-------|------------|
| $g_j$ | 0.594 |
| $b_j$ | 0.356 |
| each $g_{i \neq j}$ | 0.0078 |
| each $b_{i \neq j}$ | 0.0047 |

So the **content index is almost always $j$** (matching the true state, ~95% of the time), but the **tag stochastically alternates** between g and b with ratio $1:\beta$ (roughly 62:38).

### Steady-state behaviour ($L \to \infty$)

As completion length grows:

1. **Belief converges to a delta** on the true hidden state. The sector mass polarizes (whichever tag appears more often wins), and the within-sector belief concentrates on the true state $j$. Both converge exponentially.

2. **Output distribution converges to the fixed emission table above.** It does *not* collapse to a single token — the content is nearly deterministic (always index $j$), but the tag keeps flipping stochastically at ratio $1:\beta$.

3. **The only dynamics are in the observer's belief, not in the underlying state.** From the HMM's perspective, completion is boring: it's the same state emitting the same distribution every step. All the "action" (polarization, belief sharpening) happens in the Bayesian observer.

This static-channel structure means the completion phase is essentially an **evidence accumulation** problem. The model needs to:
- Count the tag balance (g vs b) to infer sector → sector mass update
- Read the content index to infer within-sector state → within-sector belief update
- Combine both with the prompt-encoded prior to predict the next token

There are no complex temporal dependencies — each completion token is conditionally independent given the (fixed) hidden state. The sequence-level structure comes entirely from the observer's evolving belief.

---

## 4) Summary: What the Model Must Learn

| Phase | Hidden state changes? | Updates $\pi_G/\pi_B$? | Updates $\mu_G, \mu_B$? | Mechanism |
|-------|:-----:|:-----:|:-----:|-----------|
| Prompt | Yes (leaky reset) | No | Yes | Per-token signatures rewrite within-sector state |
| Completion | No (diagonal / static) | Yes | Yes | Tag scaling ($1$ vs $\beta$) + emission diagonal |

After the prompt, the model has:
- $\pi_G = \pi_B = 0.5$ (always, by prompt neutrality)
- $\mu_G, \mu_B$ encoding the prompt identity (125 distinct pairs)
- A fixed true hidden state $(S^*, j^*)$ that will persist for all of completion

During completion, the model observes repeated noisy emissions from this fixed state. A model that has learned the process structure needs to:
1. Accumulate tag evidence (g vs b count) to infer sector → update $\pi_G / \pi_B$
2. Read the content index to infer within-sector state → update $\mu_S$
3. Combine both with the prompt-encoded prior to predict the next token

### The 1-1 token–state mapping

The entire within-sector information channel relies on a single structural property: **each token is associated 1-1 with a hidden state**, and this same identity map $\phi(j) = j$ operates in both phases.

**Prompt (write):** Token $p_k$ resets toward hidden state $k \bmod d$ via the one-hot signature $r_{S,k} = e_{k \bmod d}$. Each prompt token "writes" a specific within-sector state.

**Completion (read):** Token $g_i$ (or $b_i$) has high emission weight from hidden state $i$ via the diagonal entry $e_{S,i}[i] = 1 - \delta$. Each completion token "reads" a specific within-sector state.

This is what makes within-sector beliefs both *encodable* by the prompt and *decodable* from completions. Without the 1-1 structure, the system would degrade:

- If $M < d_S$: some hidden states have no associated completion token, so completion can't distinguish them. (The code enforces $M \geq d_S$ for this reason.)
- If signatures were random instead of one-hot: different prompt tokens would write overlapping states, reducing the number of distinguishable within-sector beliefs.
- If emission diagonals were scrambled ($e_{S,i}[j]$ peaked at $j \neq i$): the identity map between write-states and read-states would break, and the information written by the prompt wouldn't be the same information read by completions.

The 1-1 mapping ensures that the within-sector state space acts as a coherent "codebook": prompts index into it, and completions reveal which entry was selected.

### Symmetry

When $d_g = d_b$, $\lambda_g = \lambda_b$, and $\pi_A = 0.5$, there is an exact symmetry: swapping $(H_G, g\text{-tokens}) \leftrightarrow (H_B, b\text{-tokens})$ leaves the entire system invariant. This is why probing results for within-A and within-B are identical in the default config.
