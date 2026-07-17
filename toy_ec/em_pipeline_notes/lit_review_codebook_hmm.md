---

# Literature Review: Theoretical Framing of Block-Diagonal HMMs with Leaky-Reset Prompts and Codebook Completions

## Summary of Your Setup

Based on the codebase (primarily `/workspace/simplex-research/training/matrices.py`, `/workspace/simplex-research/analysis/afp_builders.py`, and `/workspace/simplex-research/analysis/em_pipeline/new_process.md`), your construction has the following structure:

- **Hidden state space**: H = H_G + H_B (direct sum), with block-diagonal transition matrices that never mix sectors.
- **Prompt phase**: Tokens apply leaky-reset operators S_S(p_k) = (1-lambda)I + lambda R_{S,k} within each sector, with a shared scalar c(p_k) so that the sector odds pi_G/pi_B are unchanged ("prompt neutrality"). Information is written into within-sector belief states.
- **Completion phase**: Tokens are diagonal (no state transitions within a sector), acting as pure observations. Tagged tokens g_i and b_i apply Bayes-factor evidence for/against each sector at rate controlled by beta, while their content index i decodes the within-sector hidden state via a noisy label map.
- **Codebook structure**: A 1-1 mapping between prompt tokens and target belief states (via signatures), and between completion content indices and hidden states (via the label map phi_S).

This is a carefully designed toy model for studying emergent misalignment in a controlled setting, where finetuning on sector-biased completions tests whether a transformer's latent state generalizes sector preference to held-out prompts.

---

## 1. Communicating Classes and Ergodic Decomposition of Markov Chains

### What it is

Any finite Markov chain decomposes into communicating classes -- maximal sets of states that can reach each other. A block-diagonal transition matrix is the algebraically transparent case where the chain has been pre-decomposed into non-communicating recurrent classes (plus possibly transient states). The "ergodic decomposition theorem" (found in standard references like Norris, *Markov Chains*, 1997) says that a general finite Markov chain can be written as a mixture of irreducible chains on its recurrent classes, after the transient states are absorbed.

### How it relates

Your direct-sum construction H_G + H_B is precisely a two-class ergodic decomposition. The block-diagonal structure T(x) = T_G(x) + T_B(x) enforces that probability mass never crosses sectors. Your "dead states" (indices 1,2,3,6 in the 9-state Z1R' x Z1R' version, filled with self-loops) correspond to transient states that carry zero probability mass under valid initializations.

### What carries over

- **Absorbing property**: Once initialized in G + B, you stay there. The sector variable z in {G, B} is a time-invariant latent class conditioned on the initial state.
- **Mixture representation**: P(observations) = pi_G * P_G(observations) + pi_B * P_B(observations). This is exact and is the foundation for your Bayesian sector-inference story.
- **Spectral structure**: A block-diagonal matrix has spectrum = union of block spectra. The spectral gap of each sector governs its within-sector mixing, independently.

### What's different

Classical ergodic decomposition typically concerns a single stationary chain. Your setup adds two non-standard features: (a) the observation process is an HMM (hidden states observed through emissions, not directly), and (b) the operators are token-indexed (different matrices per observation symbol), making this a generalized HMM / observable operator model rather than a plain Markov chain.

**Key references**: Kemeny & Snell, *Finite Markov Chains* (1960/1976); Norris, *Markov Chains* (1997); [Stepleton et al., "The Block Diagonal Infinite Hidden Markov Model" (2009)](https://www.cnbc.cmu.edu/~tai/papers/stepleton09a.pdf).

---

## 2. Lumpable Markov Chains (Kemeny-Snell, Buchholz)

### What it is

A Markov chain is *lumpable* with respect to a partition of states if the lumped (coarse-grained) process is itself Markov. Kemeny & Snell (1960) gave the classical necessary and sufficient condition: for every pair of partition classes C_i, C_j, the total transition probability from any state s in C_i to the class C_j must be the same regardless of which s in C_i you start from. Buchholz extended this to *exact lumpability* (the time-reversed analogue) and showed connections to bisimulation in concurrent systems.

### How it relates

Your construction is a very strong form of lumpability. The partition {H_G, H_B} satisfies something much stronger than ordinary lumpability -- it is a direct sum, meaning there is zero cross-block transition probability. The lumped process on {G, B} is trivially Markov: it stays in whichever sector it started in, deterministically. The lumped "sector variable" z is time-invariant.

However, the more interesting lumpability question in your setup is: can the *belief state* over hidden states be lumpably projected to lower-dimensional sufficient statistics? Specifically:
- The prompt-phase operators preserve a clean decomposition: the macro-sector weight pi_G is one sufficient statistic, and the within-sector beliefs mu_G, mu_B are the others.
- The completion-phase operators preserve this same decomposition because each tagged token multiplies pi_G/pi_B by a fixed Bayes factor (1/beta or beta) regardless of the within-sector state.

This "hierarchical sufficiency" -- where the sector odds update independently from the within-sector beliefs -- is a form of *conditional lumpability* that is not standard in the Kemeny-Snell framework but is a natural generalization.

### What carries over

- The Kemeny-Snell partition criterion is trivially satisfied (zero cross-block mass).
- Buchholz's exact lumpability results guarantee that aggregation-disaggregation algorithms converge in one step for block-diagonal chains.
- The theory of *approximate lumpability* (e.g., Gurvits & Ledoux, 2005) becomes relevant if you consider perturbations that add small cross-sector leakage.

### What's different

Standard lumpability concerns a *single* transition matrix. Your setup has token-indexed matrices (one per observation symbol), so the relevant notion is "lumpability for all matrices simultaneously." This is closer to the concept in the OOM/PSR literature of *simultaneous lumpability* of operator families, which is less studied.

**Key references**: Kemeny & Snell, *Finite Markov Chains* (1960/1976); [Buchholz, "Exact and Ordinary Lumpability in Finite Markov Chains" (1994)](https://www.researchgate.net/publication/2292552_Exact_and_Ordinary_Lumpability_in_Finite_Markov_Chains); [Tian, "Lumpability and Commutativity of Markov Processes"](https://people.nmsu.edu/jtian/publications/2006-3.pdf); [Gurvits & Ledoux, "Lumpings of Markov Chains, Entropy Rate Preservation, and Higher-Order Lumpability"](https://www.researchgate.net/publication/233928109_Lumpings_of_Markov_Chains_Entropy_Rate_Preservation_and_Higher-Order_Lumpability).

---

## 3. HMM Identifiability (Allman-Matias-Rhodes)

### What it is

Allman, Matias & Rhodes (2009) proved that HMMs are *generically identifiable*: the parameters (transition matrix, emission matrix) are determined up to label permutation from the joint distribution of observations, except on a measure-zero set of parameter values. Their key tool is Kruskal's theorem on the uniqueness of tensor decompositions -- the joint distribution of (Y_{t-1}, Y_t, Y_{t+1}) is a third-order tensor whose rank equals the number of hidden states, and Kruskal's condition gives generic uniqueness.

### How it relates

Your block-diagonal structure raises a nuanced identifiability question. Consider the two-component mixture:

P(y_{1:T}) = pi_G * P_G(y_{1:T}) + pi_B * P_B(y_{1:T})

This is a *finite mixture of HMMs*. Identifiability of such mixtures requires distinguishing which component generated the data. In your setup:

- **During the prompt phase** (with prompt neutrality), the observations carry zero information about the sector. This means the prompt phase alone cannot identify which sector the chain is in -- by construction.
- **During the completion phase**, the tagged tokens break this symmetry: g-tags favor G, b-tags favor B. The rate at which identification is possible is governed by the KL divergence between the two sectors' emission distributions, which your writeup computes as D_KL = [(alpha-beta)/(alpha+beta)] * log(alpha/beta) per token.

The Allman-Matias-Rhodes framework applies to each sector individually (each sector is an HMM, and generic identifiability holds provided the sector's observation alphabet is large enough relative to d_S). For the mixture, identifiability of the mixing weight pi_G from observations requires at least some completion tokens.

### What carries over

- Generic identifiability of each sector's parameters (transition structure, emission probabilities) from long enough observation sequences.
- Tensor decomposition methods could, in principle, be used to learn the sector structure from data.
- The Kruskal rank condition relates to your codebook design: the label maps phi_G, phi_B must be "different enough" (i.e., the emission profiles must have sufficient rank) for the sectors to be distinguished.

### What's different

- Your prompt phase is deliberately non-identifying for sectors (prompt neutrality), which means the standard identifiability results apply only to the completion phase or the joint prompt+completion sequence.
- The block-diagonal structure means you do not need the full generality of Allman-Matias-Rhodes; simpler mixture identifiability results (e.g., from Teicher, 1963, or Yakowitz & Spragins, 1968) for finite mixtures of known distributions apply.

**Key references**: [Allman, Matias & Rhodes, "Identifiability of parameters in latent structure models with many observed variables," *Annals of Statistics* 37(6A), 2009](https://arxiv.org/abs/0809.5032); [Allman, Matias & Rhodes, "Hidden Markov Model Identifiability via Tensors," 2013](https://arxiv.org/pdf/1305.0321).

---

## 4. Factorial HMMs (Ghahramani & Jordan, 1997)

### What it is

A factorial HMM has multiple independent latent Markov chains, with the observation at each time step depending on the joint state of all chains. The hidden state is a Cartesian product S = S^(1) x S^(2) x ... x S^(M), and exact inference requires tracking the full joint, which is exponential in M. Ghahramani & Jordan proposed variational inference (structured mean field) to approximate the posterior as a product of marginals over factors.

### How it relates

Your Z1R' x Z1R' construction is literally a factorial HMM with M=2 factors, each having state space {S0, S1, SR}. The Kronecker product construction of your base transition matrices (T_A = T_0^(delta) x T_0^(delta), etc.) is exactly the standard factorial HMM joint operator.

However, your "almost-factored process" (AFP) departs from the standard factorial HMM in two critical ways:

1. **Sector restriction**: You restrict to the subspace V_11 + V_22, discarding the cross terms V_12, V_21. This breaks the product structure -- the resulting process is not a product of independent factors, but a *mixture* of sector-restricted dynamics.

2. **Prompt/completion asymmetry**: Standard factorial HMMs have homogeneous dynamics (same transition structure at every time step). Your process has two phases with qualitatively different operators.

### What carries over

- The Kronecker product construction of joint operators from factor operators is standard and well-understood.
- The variational inference approach (approximating the full joint belief as a product of marginals) is directly relevant to your "per-factor marginal" probing targets in Stage 3.
- The information-theoretic framework for factorial HMMs -- mutual information decomposition between factors and observations -- provides tools for analyzing your process.

### What's different

- Your restriction to V_11 + V_22 means the posterior is NOT a product distribution over factors, even though the transition matrices were built from a product. This is because conditioning on being in V_11 + V_22 introduces correlations between factors (knowing factor 1 is in {S1, SR} implies factor 2 is also in {S1, SR}).
- The AFP is better described as a "mixture of products" rather than a "product" -- a model where you first draw a sector z, then within that sector the factors may or may not be independent.

**Key references**: [Ghahramani & Jordan, "Factorial Hidden Markov Models," *Machine Learning* 29, 1997](https://link.springer.com/article/10.1023/A:1007425814087); [Ghahramani & Jordan, original tech report, MIT AI Memo 1561](https://dspace.mit.edu/bitstream/handle/1721.1/7188/AIM-1561.pdf).

---

## 5. Input-Output HMMs and Controlled Markov Chains

### What it is

An Input-Output HMM (IOHMM, Bengio & Frasconi 1995) generalizes HMMs by conditioning transition and emission probabilities on an exogenous input sequence. The hidden state evolves as P(s_t | s_{t-1}, u_t) where u_t is the input at time t. This makes the model a controlled Markov chain observed through noisy emissions.

A closely related concept is the Partially Observable Markov Decision Process (POMDP), where the "inputs" are actions chosen by an agent.

### How it relates

Your two-phase structure can be framed as an IOHMM where the "input" is the phase indicator (prompt vs. completion). During the prompt phase, the transition operators have leaky-reset structure; during the completion phase, they have diagonal (emission-only) structure. The phase switch is deterministic and known, making this a *switched* or *regime-switching* HMM.

More precisely, your prompt tokens can be viewed as both observations AND inputs: they are emitted by the process (determining transition dynamics via the token-indexed matrices) and simultaneously serve as the "observations" that an external learner sees. This dual role of tokens as "inputs that control state evolution" and "outputs that reveal information" is characteristic of the Gelfand-Pinsker channel model in information theory (discussed in Section 11 below).

### What carries over

- IOHMM inference algorithms (forward-backward with input conditioning) apply directly.
- The theory of controlled Markov chains with partial observations provides a framework for analyzing the prompt phase: the prompt tokens control the belief state evolution, and the question "which prompt sequence maximizes information about the hidden state?" is a controlled sensing / active inference problem.

### What's different

- In standard IOHMMs, inputs are exogenous (chosen by an agent). In your setup, the "inputs" (tokens) are endogenous (generated by the process itself). The prompt token at time t is generated by the very HMM whose state it modifies. This self-referential structure is not typical in the IOHMM literature.
- Your completion phase has the special property that the operators are diagonal -- they don't change the hidden state, only reveal it. This makes the completion phase equivalent to a sequence of conditionally independent observations given the (fixed) hidden state, which is a much simpler inference problem than general IOHMM filtering.

**Key references**: Bengio & Frasconi, "An Input-Output HMM Architecture," *NIPS* 1995; [POMDP Wikipedia overview](https://en.wikipedia.org/wiki/Partially_observable_Markov_decision_process).

---

## 6. Observable Operator Models (OOMs) and Predictive State Representations (PSRs)

### What it is

Observable Operator Models (Jaeger, 2000) and Predictive State Representations (Littman, Sutton & Singh, 2001) represent stochastic processes without explicit hidden states. Instead, the "state" is a vector of predictions about future observables. For each observation symbol x, there is an operator T(x) such that the probability of a sequence is:

P(x_1, ..., x_T) = pi_0 * T(x_1) * ... * T(x_T) * 1

This is exactly the framework used in your codebase and writeups. The OOM/PSR perspective makes the token-indexed operators T(x) the primary objects, and "hidden states" are a convenient but non-unique coordinate system for these operators.

### How it relates

Your entire construction is naturally written in OOM notation. The key properties of your operators -- block-diagonality, rank structure, normalization -- are all operator-algebraic properties that can be stated without reference to "hidden states."

The PSR perspective is particularly illuminating for the prompt phase: the "predictive state" after seeing prompt tokens p_1, ..., p_P is:

state(p_1...p_P) = pi_0 * T(p_1) * ... * T(p_P)

which, in your setup, decomposes as (pi_G * mu_G, pi_B * mu_B) -- exactly the belief state. The PSR framework provides:

1. **Minimal representation**: The minimal dimension of the OOM equals the rank of the Hankel matrix of the process. For your block-diagonal process, this rank is at most d_G + d_B.
2. **Spectral learning**: The Hankel matrix can be estimated from data and factored via SVD to learn the operators, without EM or Baum-Welch. This is relevant to the question of whether a transformer "discovers" the block-diagonal structure from data.

### What carries over

- The OOM probability formula P(x_{1:T}) = pi * T(x_1)...T(x_T) * 1 is exactly how your code computes likelihoods.
- [Thon & Jaeger, "Links Between Multiplicity Automata, Observable Operator Models and Predictive State Representations"](https://jmlr.org/papers/v16/thon15a.html) unifies these frameworks, and their spectral learning algorithms apply.
- The concept of *operator rank* (= number of linearly independent operators needed) corresponds to your hidden state dimension d_G + d_B.

### What's different

- OOMs/PSRs typically assume a stationary, time-homogeneous process. Your two-phase structure (prompt then completion) means the operators change at a known time boundary, making this a *non-stationary* or *piecewise-stationary* OOM.
- The block-diagonal structure is "extra structure" that generic OOM/PSR learning algorithms do not exploit. A learning algorithm that knows about block-diagonality would be more sample-efficient.

**Key references**: Jaeger, "Observable Operator Models for Discrete Stochastic Time Series," *Neural Computation* 12(6), 2000; [Littman, Sutton & Singh, "Predictive Representations of State," *NIPS* 2001](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf); [Balle, Hamilton, Pineau, "Methods of Moments for Learning Stochastic Languages: Unified Presentation and Empirical Comparison," *ICML* 2014](http://proceedings.mlr.press/v70/balle17a/balle17a.pdf).

---

## 7. Information Theory of HMMs (Ephraim & Merhav; Entropy Rate)

### What it is

Ephraim & Merhav (2002) provide a comprehensive treatment of statistical and information-theoretic aspects of Hidden Markov Processes. A central quantity is the *entropy rate*:

h(Y) = lim_{T->inf} (1/T) H(Y_1, ..., Y_T)

which measures the average information per observation. For HMMs, no closed-form expression exists in general; the entropy rate depends on the full spectral structure of the transition operators.

Key results include:
- Upper and lower bounds via the hidden state entropy: H(Y|X) <= h(Y) <= H(Y), where X is the hidden state sequence.
- The mutual information rate I(X; Y) = h(Y) - H(Y|X) quantifies how much the observations reveal about the hidden states per time step.

### How it relates

For your block-diagonal HMM:

1. **Entropy rate decomposes**: h(Y) = pi_G * h_G(Y) + pi_B * h_B(Y), where h_G, h_B are the entropy rates within each sector. This is because the process is a static mixture.

2. **Completion phase KL divergence**: Your writeup derives that the per-token Bayes factor between sectors is alpha/beta (for g-tagged tokens), giving a KL divergence of D_KL = [(alpha-beta)/(alpha+beta)] * log(alpha/beta) between the sector-G and sector-B completion distributions. This quantity controls how quickly the sector is identified.

3. **Prompt phase entropy**: During the prompt phase (prompt neutrality), the mutual information between observations and the sector variable z is exactly zero -- the prompt carries no sector information. All prompt information is about the *within-sector* state.

4. **Channel capacity interpretation**: The completion phase acts as a binary-input noisy channel: the "input" is z in {G, B}, the "output" is the sequence of tag bits. The capacity of this effective channel per token is related to the Chernoff information between the two sectors' emission distributions.

### What carries over

- The entropy rate bounds of Ephraim & Merhav apply to each sector individually.
- The Birkhoff contraction coefficient (discussed below) controls how quickly the belief state "forgets" its initialization within each sector, which bounds the mixing time of the within-sector entropy rate.
- For your completion phase, where operators are diagonal, the entropy rate simplifies considerably: conditioned on the hidden state, the observations are i.i.d., so H(Y|X) = H(Y_t|X_t).

### What's different

- The "no closed form" result for entropy rate is primarily about HMMs with non-trivial state transitions. Your completion phase has trivial transitions (identity), so the entropy rate within the completion phase *does* have a closed form.
- The prompt phase has non-trivial transitions (leaky reset), but the leaky-reset structure is so constrained (rank-1 perturbation of identity) that explicit entropy rate calculations may be tractable.

**Key references**: [Ephraim & Merhav, "Hidden Markov Processes," *IEEE Trans. Information Theory* 48(6), 2002](https://www.researchgate.net/publication/3080695_Hidden_Markov_Processes); [Ordentlich & Weissman, "On the Entropy of a Binary Hidden Markov Process," *Journal of Statistical Physics* 2005](https://link.springer.com/article/10.1007/s10955-005-7576-y); [Crutchfield & Marzen, "Shannon Entropy Rate of Hidden Markov Processes," *Journal of Statistical Physics* 2021](https://link.springer.com/article/10.1007/s10955-021-02769-3).

---

## 8. Birkhoff Contraction Coefficient and Belief State Convergence

### What it is

Birkhoff (1957) proved that any positive linear operator on a cone contracts in the Hilbert projective metric. For a column-stochastic matrix A, the *Birkhoff contraction coefficient* tau(A) = tanh(Delta(A)/4), where Delta(A) is the diameter of the image of the probability simplex under A in the Hilbert metric. This gives:

d_H(Ax, Ay) <= tau(A) * d_H(x, y)

for any two probability vectors x, y. Products of stochastic matrices contract geometrically at rate determined by the product of contraction coefficients.

### How it relates

The Birkhoff contraction coefficient is the natural tool for analyzing your prompt-phase belief dynamics. After P prompt tokens k_1, ..., k_P, the within-sector belief is:

mu_S(P) = (1-lambda)^P * mu_S(0) + lambda * sum_{t=1}^P (1-lambda)^{P-t} * r_{S,k_t}

The convergence rate (1-lambda)^P is exactly the contraction rate of the leaky-reset operator. In terms of the Birkhoff coefficient:

tau(S_S(p_k)) = 1 - lambda_S

so after P tokens, the contraction is (1-lambda_S)^P, and the belief "forgets" its initial condition exponentially fast.

This has a direct implication for your codebook: the effective "prompt capacity" (how many distinguishable post-prompt beliefs you can create) depends on the contraction rate. If lambda is too large, the belief converges quickly to the most recent signature, losing information about earlier tokens. If lambda is too small, the belief barely moves and all prompts produce similar beliefs.

### What carries over

- Exponential convergence of beliefs: after P tokens, the dependence on the initial belief mu_S(0) decays as (1-lambda)^P.
- The Hilbert metric framework provides tight bounds on the "information capacity" of the prompt phase: how many distinguishable beliefs can be created by length-P prompt strings.
- For the completion phase, the contraction coefficient is 1 (identity operators don't contract within a sector), which is consistent with the design: completions reveal the state without changing it.

### What's different

- Standard Birkhoff theory deals with a single matrix or i.i.d. products. Your prompt operators are chosen from a set {S_S(p_k) : k=1,...,V_P}, and the relevant analysis is for *products of chosen matrices* -- an inhomogeneous product. The contraction still holds (the product of contraction coefficients bounds the joint contraction), but the effective contraction depends on the specific prompt string.

**Key references**: [Birkhoff, "Extensions of Jentzsch's Theorem," *Transactions of the AMS* 1957](https://www.sciencedirect.com/science/article/pii/S0024379504001673); [Levin, Peres & Wilmer, *Markov Chains and Mixing Times*, 2nd edition](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf); Seneta, *Non-negative Matrices and Markov Chains*, Springer, 2006.

---

## 9. Mixed-Membership Models and Latent Dirichlet Allocation

### What it is

Mixed-membership models (Erosheva, 2002; Blei, Ng & Jordan, 2003 for LDA) allow each data point to partially belong to multiple latent classes. In LDA, each document has a distribution over topics, and each word is drawn from a topic chosen according to that distribution. The Grade of Membership (GoM) model (Woodbury, Clive & Garson, 1978) is the original formulation.

### How it relates

Your AFP has a structural similarity to mixed-membership models: each sequence has a latent "membership" in sector G or B (the discrete variable z), and the observations are generated conditional on z. However, there is a crucial difference:

- In LDA/GoM, membership is *mixed* -- a single document can have partial membership in multiple topics, and individual words switch topics.
- In your AFP, membership is *pure* -- the sector z is drawn once and remains fixed throughout the sequence. Different observations within the same sequence all come from the same sector.

Your setup is therefore closer to a *latent class model* (Lazarsfeld, 1950) or a *mixture model* than a mixed-membership model. The prompt phase does not change z; it only changes the within-sector state.

### What carries over

- The EM algorithm framework for learning mixture parameters is directly applicable.
- The identifiability theory for latent class models (Teicher, 1963; Allman-Matias-Rhodes, 2009) applies.
- The information-theoretic analysis of how much data is needed to identify the latent class is relevant to your question of how many completion tokens are needed to determine the sector.

### What's different

- Mixed-membership allows per-token topic switching; your model does not.
- The "almost" in "almost-factored process" refers to the factored structure of the hidden state *within* a sector, not to mixed membership across sectors.

**Key references**: [Blei, Ng & Jordan, "Latent Dirichlet Allocation," *JMLR* 3, 2003](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf); Erosheva, "Grade of Membership and Latent Structure Models with Application to Disability Survey Data," 2002.

---

## 10. Sufficient Statistics in HMMs and the Exponential Family

### What it is

For an HMM with finite state space, the *belief state* (the posterior distribution over hidden states given observations so far) is a *sufficient statistic* for future predictions. This is the fundamental property exploited by the forward algorithm: the belief state captures all information from past observations that is relevant for predicting future observations.

For HMMs in the exponential family, the complete-data sufficient statistics are the state transition counts N_{ij} and emission counts M_{jk}, which are used in the EM algorithm (Baum-Welch).

### How it relates

Your construction has a beautiful hierarchy of sufficient statistics:

1. **Full belief state**: pi = (pi_G * mu_G, pi_B * mu_B), dimension d_G + d_B - 1. This is sufficient for everything.

2. **Sector odds + within-sector beliefs**: (pi_G, mu_G, mu_B). This is an equivalent reparameterization that separates sector information from content information.

3. **Sector odds alone**: pi_G. This scalar is sufficient for predicting the *tag* (g vs b) of the next completion token, though not the content index.

4. **Within-sector belief alone**: mu_S (for the correct sector S). This is sufficient for predicting the content index, given the sector.

The prompt phase updates only (mu_G, mu_B), leaving pi_G unchanged. The completion phase updates all components. This clean separation is a design feature, not a generic property.

### What carries over

- The forward algorithm (belief state recursion) applies directly to your model.
- The belief state is always sufficient; this is a theorem, not an assumption.
- For the completion phase (diagonal operators), the belief update is particularly simple: it's just Bayesian updating with the emission likelihood, without any transition kernel convolution.

### What's different

- In generic HMMs, the belief state dimension equals the number of hidden states minus 1. Your block-diagonal structure creates a *reducible* belief state where the sector odds can be tracked separately from the within-sector beliefs. This is a form of *information-theoretic dimensionality reduction* that is specific to your construction.
- The leaky-reset prompt operators create a special *affine* update rule for within-sector beliefs (convex combination of current belief and target), which is much simpler than generic HMM belief updates (matrix-vector product followed by normalization).

**Key references**: Rabiner, "A Tutorial on Hidden Markov Models and Selected Applications in Speech Recognition," *Proceedings of the IEEE* 77(2), 1989; [Wainwright & Jordan, "Graphical Models, Exponential Families, and Variational Inference," *Foundations and Trends in Machine Learning* 2008](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf).

---

## 11. Channel Coding Through Latent Variables (Gelfand-Pinsker)

### What it is

The Gelfand-Pinsker channel (1980) models communication through a state-dependent channel where the encoder has non-causal knowledge of the channel state. The capacity is:

C = max_{p(u,x|s)} [I(U; Y) - I(U; S)]

where S is the channel state, X is the input, Y is the output, and U is an auxiliary random variable representing the "codebook."

### How it relates

Your two-phase structure has a natural channel-coding interpretation:

- **Prompt phase as encoding**: The prompt tokens "write" a message (the within-sector belief) into the hidden state. The prompt is like an encoder that has full knowledge of the "channel state" (the hidden state evolution) and chooses tokens to create a specific latent state configuration.

- **Completion phase as decoding**: The completion tokens are generated by the hidden state and must be decoded to recover (a) which sector and (b) which within-sector state.

- **The codebook mapping**: Your 1-1 mapping between prompt tokens and target states (signatures), and between completion content indices and hidden states (label maps), is literally a codebook in the information-theoretic sense. The design question -- "how to assign signatures to prompt tokens and label maps to completion indices" -- is a codebook design problem.

The Gelfand-Pinsker framework is most relevant to the completion phase: the "state" is the hidden state s (determined by the prompt), the "channel" maps s to a sequence of tagged observations, and the "encoder" (nature, in this case) must design emission distributions that allow both sector identification and content decoding.

### What carries over

- The capacity formula provides an upper bound on how much information the completion phase can convey about the prompt-determined hidden state.
- The achievability proof (random coding with auxiliary codebooks) suggests that your base-4 codebook design for the clustered variant is a good strategy -- it maximizes the "codeword distance" between different prompt-conditioned hidden states.
- The separation between sector evidence (tag) and content evidence (base symbol) in your design is reminiscent of the separation between "state neutralization" and "message encoding" in Gelfand-Pinsker coding.

### What's different

- In Gelfand-Pinsker, the encoder *chooses* the input to communicate a message. In your setup, the completion tokens are *generated by the process* (no agent choosing them), so the "encoding" is done by nature, not an optimizer.
- Your setup has the additional structure that the "message" is itself generated by a specific process (the prompt HMM), not chosen freely.

**Key references**: Gelfand & Pinsker, "Coding for Channel with Random Parameters," *Problems of Control and Information Theory* 9(1), 1980; [Cover & Thomas, *Elements of Information Theory*, Chapter 13](https://static.ias.edu/pitp/archive/2012files/Cover_and_Thomas_Chptr13.pdf).

---

## 12. Rate-Distortion Theory and Latent Representations

### What it is

Rate-distortion theory (Shannon, 1959) characterizes the fundamental tradeoff between compression rate R and reconstruction fidelity D. The rate-distortion function R(D) gives the minimum number of bits per source symbol needed to reconstruct the source within distortion D. Recent work connects this to representation learning: the "bottleneck" in variational autoencoders and information bottleneck methods is a rate-distortion problem.

### How it relates

The connection to your setup is through the question: "how much information about the hidden state is preserved/lost as observations pass through the HMM emission channel?"

1. **Prompt phase as a lossy channel**: The prompt writes information into the belief state, but the leaky-reset operator is a *contraction* -- it loses information about earlier tokens at rate (1-lambda). This is a rate-distortion tradeoff: longer prompts carry more information, but the exponential decay limits the effective information to approximately 1/(lambda) tokens of memory.

2. **Completion phase as a noisy channel**: The completion tokens are noisy readouts of the hidden state, with noise parameter delta. The rate at which hidden state information is extracted is bounded by the mutual information between the hidden state and each completion token, which depends on delta and the sector evidence parameter beta.

3. **Codebook design as rate-distortion optimization**: In the clustered_codebook variant, the choice of K clusters and their codewords is a codebook design problem that trades off the number of distinguishable messages (K) against the reliability of decoding (which depends on codeword distances).

Recent work by [Sorscher et al., "The geometry of efficient codes: How rate-distortion trade-offs distort the latent representations of generative models," *PLoS Computational Biology* 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952) shows that rate-distortion trade-offs create specific geometric distortions in latent representations -- prototypization, specialization, and orthogonalization -- which may be relevant to understanding how transformers represent your AFP structure.

### What carries over

- The information bottleneck framework (Tishby, Pereira & Bialek, 1999) can formalize the tradeoff between prompt length and within-sector belief diversity.
- Rate-distortion bounds give fundamental limits on your codebook's capacity to distinguish different hidden states from completion observations.

### What's different

- Rate-distortion theory typically deals with i.i.d. sources. Your hidden states are correlated (via the prompt history), requiring extensions to rate-distortion for Markov sources.

**Key references**: Shannon, "Coding Theorems for a Discrete Source with a Fidelity Criterion," *IRE National Convention Record* 1959; Tishby, Pereira & Bialek, "The Information Bottleneck Method," 1999; [Sorscher et al., *PLoS Computational Biology* 2025](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952).

---

## 13. Frame Theory and Overcomplete Representations

### What it is

A frame in a Hilbert space is a (possibly overcomplete) set of vectors {f_i} such that for all x in the space, A||x||^2 <= sum_i |<x, f_i>|^2 <= B||x||^2 for constants 0 < A <= B. Frames generalize orthonormal bases by allowing redundancy. The *frame operator* S = sum_i f_i f_i^* provides reconstruction via x = S^{-1} sum_i <x, f_i> f_i.

### How it relates

Your prompt signatures {r_{S,k} : k = 1, ..., V_P} form a set of vectors in the probability simplex Delta^{d_S-1}. When V_P > d_S (i.e., more prompt tokens than hidden states per sector), this is an overcomplete representation -- there are more "target states" than dimensions.

The question "can you recover the within-sector hidden state from the prompt-conditioned belief?" is a frame-theoretic reconstruction problem. If the signatures form a frame for the relevant subspace of the simplex, then the belief state after a sufficiently long prompt uniquely determines the "effective signature" (the exponentially-weighted average of recent targets).

For the completion phase, the emission vectors {e_{S,i} : i = 1, ..., M} form a set of vectors in R^{d_S} that serve as "measurement vectors" for the hidden state. Reconstruction of the hidden state from M noisy measurements is governed by the frame properties of these vectors.

### What carries over

- The frame bound ratio B/A (condition number) governs reconstruction stability. Your "onehot" signature type is the best case (orthonormal when V_P = d_S).
- *Incoherence* (small inner products between frame vectors) is desirable for compressed sensing and applies to your codebook: the more "different" the signatures for different prompt tokens, the more distinguishable the resulting beliefs.
- The "method of optimal directions" for frame design is relevant to optimizing your signature assignments.

### What's different

- Frame theory assumes linear observations; your HMM belief update is nonlinear (it involves normalization). The frame-theoretic analysis applies to the *linearized* version (i.e., the leaky-reset before normalization).
- The probability simplex constraint (beliefs must be non-negative and sum to 1) restricts the relevant geometry to a subset of the Hilbert space.

**Key references**: Christensen, *An Introduction to Frames and Riesz Bases*, Springer 2003; [Casazza, "Frames in Signal Processing," Chapter 10](https://www.researchgate.net/publication/259642597_Chapter_10_Frames_in_Signal_Processing).

---

## 14. Weighted Finite Automata and Spectral Learning

### What it is

A Weighted Finite Automaton (WFA) over a semiring is a tuple (alpha, {A_sigma}, omega) where alpha is an initial weight vector, A_sigma is a matrix for each symbol sigma, and omega is a final weight vector. The weight of a string w = sigma_1...sigma_T is:

f(w) = alpha^T A_{sigma_1} ... A_{sigma_T} omega

This is isomorphic to the OOM framework. The fundamental theorem (Carlyle & Paz, 1971; Fliess, 1974) states that f can be computed by a WFA with n states if and only if the Hankel matrix H_f has rank n.

Spectral learning (Hsu, Kakade & Zhang, 2009) estimates the WFA parameters from a finite sample by computing the SVD of an empirical Hankel matrix.

### How it relates

Your AFP process defines a probability distribution over strings, P(x_{1:T}), which has a WFA representation with n = d_G + d_B states (plus dead states that carry zero mass). The Hankel matrix of this process has rank at most d_G + d_B.

The block-diagonal structure implies that the Hankel matrix itself has a corresponding block structure: it decomposes as a sum of two rank-d_G and rank-d_B components (corresponding to the two sectors, weighted by pi_G and pi_B).

Spectral learning could, in principle, learn your process from data without knowing the block-diagonal structure a priori. The SVD of the Hankel matrix would reveal the two-sector decomposition as a natural consequence of the spectrum having two groups of singular values.

### What carries over

- The rank of the Hankel matrix gives the minimal number of hidden states, providing a lower bound on model complexity.
- [Spectral learning algorithms](https://link.springer.com/article/10.1007/s10994-013-5416-x) provide a method-of-moments approach to learning your process, complementary to EM/Baum-Welch.
- The connection between WFA minimization and model reduction is relevant to your question of whether the process can be represented with fewer states.

### What's different

- Spectral learning assumes access to substring statistics from a stationary process. Your two-phase structure violates stationarity, requiring adaptations.
- The WFA framework does not naturally capture the "block-diagonal" constraint; it would need to be imposed as structure in the learning algorithm.

**Key references**: [Balle, Carreras, Luque & Quattoni, "Spectral Learning of Weighted Automata," *Machine Learning* 2014](https://link.springer.com/article/10.1007/s10994-013-5416-x); Hsu, Kakade & Zhang, "A Spectral Algorithm for Learning Hidden Markov Models," *JMLR* 2012.

---

## 15. Transformers Learning Belief State Geometry (Simplex/Astera Research)

### What it is

Recent work from the Simplex team (at Astera Institute) demonstrates that transformers trained to predict next tokens on HMM-generated sequences learn to represent the Bayes-optimal belief state geometry in their residual stream. Key papers include:

- [Mudide et al., "Transformers Represent Belief State Geometry in their Residual Stream," *NeurIPS* 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8936fa1691764912d9519e1b5673ea66-Paper-Conference.pdf): Shows that for simple HMMs like Mess3, the belief states form fractal structures on the probability simplex, and transformers linearly encode these geometric structures.

- [Li et al., "Constrained Belief Updates Explain Geometric Structures in Transformer Representations," 2025](https://arxiv.org/html/2502.01954v1): Shows that attention mechanisms implement an algorithm with a direct interpretation as belief state updates.

- [Zhang et al., "Transformers as Multi-task Learners: Decoupling Features in Hidden Markov Models," 2025](https://arxiv.org/html/2506.01919): Shows that transformers learn factored representations from factorial HMM data, with lower layers extracting features and upper layers achieving disentanglement.

### How it relates

This is the most directly relevant body of work, as your AFP is designed specifically to probe these phenomena. The key connections:

1. **Belief state geometry**: Your AFP has a belief state that lives on a (d_G + d_B - 1)-simplex with block-diagonal structure. The transformer should learn to represent this geometry in its residual stream.

2. **Factored representations**: The Z1R' x Z1R' construction tests whether the transformer discovers the product structure of the hidden state, and the AFP restriction to V_11 + V_22 tests whether it can represent a *mixture of products*.

3. **Sector identification**: The Stage 3 probing (linear regression from activations to belief state) directly tests the Simplex prediction that belief states are linearly represented. The Stage 4 finetuning tests whether sector-biased training produces the generalization pattern predicted by the latent-variable structure.

4. **Emergent misalignment as Bayesian inference**: If the transformer implements approximate Bayes-optimal prediction, then finetuning on sector-A-biased completions should shift the model's posterior toward sector A on held-out prompts -- which is exactly the "emergent misalignment" pattern you're studying.

### What carries over

- The theoretical prediction that transformers linearly encode HMM belief states applies to your model.
- The per-layer analysis (lower layers extract features, upper layers disentangle) provides a framework for interpreting your probing results.
- The finding that attention implements belief updates gives a mechanistic account of *how* the transformer performs HMM inference.

### What's different

- Previous Simplex work focused on simple, stationary HMMs (Mess3). Your AFP is non-stationary (two phases), has block-diagonal structure, and has a much larger state space.
- The "emergent misalignment" finetuning experiment (Stage 4) goes beyond the passive probing in previous work, testing whether the learned latent structure *causally* influences generalization under distribution shift.

**Key references**: [Mudide et al., *NeurIPS* 2024](https://proceedings.neurips.cc/paper_files/paper/2024/file/8936fa1691764912d9519e1b5673ea66-Paper-Conference.pdf); [Li et al., 2025](https://arxiv.org/html/2502.01954v1); [Zhang et al., 2025](https://arxiv.org/html/2506.01919); [Simplex Progress Report, July 2025](https://www.alignmentforum.org/posts/fhkurwqhjZopx8DKK/simplex-progress-report-july-2025).

---

## 16. Emergent Misalignment and Linear Representations

### What it is

Recent work on emergent misalignment shows that narrow finetuning of LLMs can produce broadly misaligned behavior. [Betley et al., "Emergent Misalignment: Narrow finetuning can produce broadly misaligned LLMs," 2025](https://arxiv.org/abs/2502.17424) demonstrated that finetuning a model to output insecure code leads to broad misalignment (the model endorses harmful goals on unrelated topics). Follow-up work identified that:

- [Emergent misalignment has linear representations](https://arxiv.org/html/2506.11618v2): The misalignment behavior corresponds to a single linear direction in activation space, learnable by a rank-1 LoRA adapter.
- [Persona features control emergent misalignment](https://arxiv.org/html/2506.11613v1): High-level "persona" features serve as the mechanistic substrate.
- A *phase transition* occurs during finetuning where misalignment directions are learned rapidly over a narrow window of training steps.

### How it relates

Your AFP is designed as a *controlled proxy* for exactly this phenomenon:

1. **Sector = alignment**: Sector G ("good") and Sector B ("bad") are analogues of aligned vs. misaligned behavior.
2. **Prompt = context**: The prompt sets up an ambiguous latent state (uncertain sector membership).
3. **Completion finetuning = narrow training**: Finetuning on sector-A-biased completions is the analogue of "finetuning on insecure code."
4. **Held-out generalization = emergent misalignment**: Testing whether the model's sector preference shifts on held-out prompts measures the analogue of "does narrow finetuning cause broad behavior change?"

The key theoretical question is: does the AFP have the right structure to reproduce the "phase transition" and "linear representation" phenomena observed in real LLMs? The block-diagonal structure ensures that sector membership is a genuine latent variable that affects all completion tokens, which is the structural prerequisite for the Bayesian generalization story.

### What carries over

- The finding that emergent misalignment has linear representations supports your probing approach (Stage 3 linear regression).
- The phase transition in finetuning corresponds to the point where the model has "learned" the sector variable and begins generalizing it.
- The connection to the information-theoretic framework is tight: the "amount of emergent misalignment" should be predicted by the KL divergence between sectors, the number of finetuning examples, and the post-prompt sector uncertainty.

### What's different

- Real LLMs have enormously complex latent structure; your AFP has exactly two sectors with known geometry.
- Real emergent misalignment involves semantic generalization; your AFP involves statistical generalization (sector odds shifting).
- The "1-1 codebook" structure ensures perfect identifiability in your model, whereas real LLMs may have approximate or partial identifiability.

**Key references**: [Betley et al., 2025](https://arxiv.org/abs/2502.17424); [Convergent Linear Representations of Emergent Misalignment, 2025](https://arxiv.org/html/2506.11618v2); [Model Organisms for Emergent Misalignment, 2025](https://arxiv.org/html/2506.11613v1).

---

## 17. Closest Canonical Framing

Having surveyed the landscape, the closest canonical framing for your setup draws from multiple traditions:

**Primary framing: Mixture of HMMs with controlled belief writing**

Your process is a *finite mixture of HMMs* where:
- The mixture component z in {G, B} is drawn once from the initial distribution and never changes.
- Each component is a non-stationary HMM with two phases: a "writing" phase (leaky-reset prompts) and a "reading" phase (diagonal completions).

This is most precisely described in the OOM/GHMM framework as a *block-diagonal observable operator model with phase-dependent operators*.

**Secondary framings that capture different aspects:**

| Aspect of your setup | Closest canonical framework | Key property |
|---|---|---|
| Block-diagonal structure | Ergodic decomposition / absorbing classes | Sector z is time-invariant |
| Leaky-reset prompts | Iterated function system (IFS) on the simplex | Belief written by contraction maps |
| Diagonal completions | Bayesian filtering with fixed state | Pure emission/observation model |
| Tagged tokens (g/b) | Binary hypothesis testing with nuisance parameters | Sector = hypothesis, content = nuisance |
| Codebook mapping | Quantization / source coding | Prompt token -> target state |
| Two-phase structure | Regime-switching HMM / Input-Output HMM | Non-stationarity with known regime |
| Factored structure | Factorial HMM restricted to subspace | Mixture of tensor products |

**The novel element** that does not cleanly map onto any single existing framework is the combination of:
1. Prompt neutrality (prompts carry zero sector information).
2. Prompt belief-writing (prompts encode rich within-sector information).
3. Completion evidence (completions reveal sector while also decoding content).
4. The codebook mapping tying these together.

This combination creates a clean separation between "what the model should generalize" (sector preference) and "what the model should memorize" (within-sector content), which is the key design principle for the emergent misalignment proxy.

---

## Sources

- [Allman, Matias & Rhodes, "Identifiability of parameters in latent structure models" (2009)](https://arxiv.org/abs/0809.5032)
- [Allman, Matias & Rhodes, "Hidden Markov Model Identifiability via Tensors" (2013)](https://arxiv.org/pdf/1305.0321)
- [Balle et al., "Spectral Learning of Weighted Automata" (2014)](https://link.springer.com/article/10.1007/s10994-013-5416-x)
- [Betley et al., "Emergent Misalignment" (2025)](https://arxiv.org/abs/2502.17424)
- [Birkhoff contraction coefficient (Seneta, 2004)](https://www.sciencedirect.com/science/article/pii/S0024379504001673)
- [Blei, Ng & Jordan, "Latent Dirichlet Allocation" (2003)](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf)
- [Buchholz, "Exact and Ordinary Lumpability in Finite Markov Chains" (1994)](https://www.researchgate.net/publication/2292552_Exact_and_Ordinary_Lumpability_in_Finite_Markov_Chains)
- [Crutchfield & Marzen, "Shannon Entropy Rate of Hidden Markov Processes" (2021)](https://link.springer.com/article/10.1007/s10955-021-02769-3)
- [Ephraim & Merhav, "Hidden Markov Processes" (2002)](https://www.researchgate.net/publication/3080695_Hidden_Markov_Processes)
- [Ghahramani & Jordan, "Factorial Hidden Markov Models" (1997)](https://link.springer.com/article/10.1023/A:1007425814087)
- [Gurvits & Ledoux, "Lumpings of Markov Chains, Entropy Rate Preservation" (2005)](https://www.researchgate.net/publication/233928109_Lumpings_of_Markov_Chains_Entropy_Rate_Preservation_and_Higher-Order_Lumpability)
- [Levin, Peres & Wilmer, Markov Chains and Mixing Times (2nd ed.)](https://pages.uoregon.edu/dlevin/MARKOV/markovmixing.pdf)
- [Li et al., "Constrained Belief Updates Explain Geometric Structures" (2025)](https://arxiv.org/html/2502.01954v1)
- [Littman, Sutton & Singh, "Predictive Representations of State" (2001)](https://web.eecs.umich.edu/~baveja/Papers/psr.pdf)
- [Model Organisms for Emergent Misalignment (2025)](https://arxiv.org/html/2506.11613v1)
- [Convergent Linear Representations of Emergent Misalignment (2025)](https://arxiv.org/html/2506.11618v2)
- [Mudide et al., "Transformers Represent Belief State Geometry" (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/8936fa1691764912d9519e1b5673ea66-Paper-Conference.pdf)
- [Simplex Progress Report - July 2025](https://www.alignmentforum.org/posts/fhkurwqhjZopx8DKK/simplex-progress-report-july-2025)
- [Sorscher et al., "Geometry of efficient codes" (2025)](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1012952)
- [Stepleton et al., "Block Diagonal Infinite Hidden Markov Model" (2009)](https://www.cnbc.cmu.edu/~tai/papers/stepleton09a.pdf)
- [Thon & Jaeger, "Links Between Multiplicity Automata, OOMs, and PSRs" (2015)](https://jmlr.org/papers/v16/thon15a.html)
- [Tian, "Lumpability and Commutativity of Markov Processes" (2006)](https://people.nmsu.edu/jtian/publications/2006-3.pdf)
- [Wainwright & Jordan, "Graphical Models, Exponential Families, and Variational Inference" (2008)](https://people.eecs.berkeley.edu/~jordan/papers/wainwright-jordan-fnt.pdf)
- [Zhang et al., "Transformers as Multi-task Learners: Decoupling Features in HMMs" (2025)](https://arxiv.org/html/2506.01919)
