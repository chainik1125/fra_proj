# Autoregressive Steering in the Coin Mixture Pipeline

## Contents

- [TL;DR](#tldr)
- [Motivation](#motivation)
  - [Goal: Control which ergodic component the model uses](#goal-control-which-ergodic-component-the-model-uses)
  - [Pipeline overview](#pipeline-overview)
- [Two steering modes](#two-steering-modes)
  - [Logit lens (single-step)](#logit-lens-single-step)
  - [Continuous autoregressive](#continuous-autoregressive)
- [Results](#results)
  - [Scale-zero baselines](#scale-zero-baselines)
  - [Near-zero regime](#near-zero-regime-α--3)
  - [Compounding at moderate scales](#compounding-at-moderate-scales)
  - [Saturation at extreme scales](#saturation-at-extreme-scales)
- [Discussion](#discussion)
- [Weight steering vs bias steering](#weight-steering-vs-bias-steering)
  - [Two hypotheses](#two-hypotheses)
  - [The per-sequence variance test](#the-per-sequence-variance-test)
  - [Results: the distribution is multimodal](#results-the-distribution-is-multimodal)
  - [The steering feature spans all coins](#the-steering-feature-spans-all-coins)
  - [Quantitative model comparison](#quantitative-model-comparison)
- [Process setup](#process-setup)

---

### The coin mixture as an HMM

<a id="tldr"></a>
**TL;DR** I show that using a "model-diffing" procedure ([Sharma et al., 2025](https://arxiv.org/abs/2506.19823)) we can identify a 'bias direction' which the model uses to control which bias it uses to output tokens. In an upcoming note on the Persona work, I show that this is a generic property of non-ergodic processes and we can generically steer between ergodic components.

![Single-token steering curves by N, showing top features per mixture](../outputs/sweep_summary/steering_curves_by_n.png)

## Motivation 




### Goal: Control which ergodic component the model uses

Our goal is to discover how the model controls the higher order information of which ergodic component 



### Pipeline overview

We run the following four-stage pipeline independently for each $N \in \{2, 3, 5, 10\}$.

**Stage 1 — Process construction.** Build the coin mixture HMM with $N$ coins, biases $p_i = 0.05 + 0.9 \cdot (i-1)/(N-1)$, uniform selection $Q$, and sequence length $L = 32$.

**Stage 2 — Pretrain base model.** Train a 2-layer transformer ($d_\text{model} = 64$, $d_\text{head} = 32$, $n_\text{heads} = 2$, $d_\text{MLP} = 256$, context $n_\text{ctx} = 32$) on the mixture for 5000 steps (batch size 128, lr $= 10^{-3}$). At checkpoints, fit a linear probe from the last-token residual stream to the Bayes-optimal posterior $\eta_t$ and record $R^2$. All runs converge to $R^2 > 0.998$.

**Stage 3 — Finetune per coin.** For each coin $i$, copy the base model and finetune on sequences from coin $i$ alone for 500 steps (batch size 64, lr $= 3 \times 10^{-4}$, sequence length 4).

> **Remark (why sequence length 4).** The base model is pretrained on length 32, and all evaluation uses length 32. The finetuning sequence length of 4 is deliberately short. With only 4 tokens from a single coin, the Bayesian posterior $\eta_t$ has barely moved from the uniform prior — the model has almost no evidence about which coin generated the data. Consequently, the next-token loss at early positions is dominated by the model's *unconditional prior* $P(0)$. Finetuning on these short sequences forces the model to shift that prior toward coin $i$'s bias $p_i$, encoding "I am coin $i$" into the model's default behavior, rather than merely learning to perform faster Bayesian inference from the token stream.

**Stage 4 — SAE diffing and steering.** This is the core model-diffing step. We train an SAE on the base model's activations, then compare how the base and finetuned models represent the *same* inputs in SAE feature space.

**4a. Train the SAE.**

1. Sample 10,000 sequences $\{x^{(n)}\}$ from the coin mixture.
2. Run each through the **base model**, cache the residual stream at the last token position of the final layer (`blocks.1.hook_resid_post`):
$$\mathbf{h}^{(n)} = \text{resid}\_\text{post}^{(\ell=1)}(x^{(n)})_{t=-1} \in \mathbb{R}^{d_\text{model}}$$
3. Train a sparse autoencoder on these activations: hidden dim $= 4 \times 64 = 256$ features, batch-TopK with $k=20$, 5000 steps, batch size 256, lr $= 10^{-3}$. The SAE learns an encoder $\text{enc}: \mathbb{R}^{d_\text{model}} \to \mathbb{R}^{256}$ and decoder vectors $\mathbf{d}_f \in \mathbb{R}^{d_\text{model}}$ for each feature $f$.

**4b. Diff: compare base vs finetuned representations.**

The key idea: feed the **same** input sequences through **different** models (base vs finetuned), encode the activations with the **same** SAE, and compare.

1. Sample a fresh eval set of 2,000 mixture sequences $\{x^{(n)}\}$.
2. Run them through the **base model**, extract last-token activations, encode through the SAE:
$$z_\text{base}^{(n)} = \text{enc}\bigl(\mathbf{h}_\text{base}^{(n)}\bigr) \in \mathbb{R}^{256}$$
3. Run the **same** sequences through each **finetuned model** $M_i$ (one per coin), extract and encode:
$$z_{\text{ft},i}^{(n)} = \text{enc}\bigl(\mathbf{h}_{M_i}^{(n)}\bigr) \in \mathbb{R}^{256}$$
4. For each feature $f$, compute the diff:
$$\Delta_i^{(f)} = \underbrace{\frac{1}{N_\text{eval}} \sum_n z_{\text{ft},i}^{(n,f)}}_{\bar{z}_{\text{ft},i}^{(f)}} - \underbrace{\frac{1}{N_\text{eval}} \sum_n z_\text{base}^{(n,f)}}_{\bar{z}_\text{base}^{(f)}}$$

In code (simplified from [`pipeline.py:578–606`](../pipeline.py)):
```python
# Same eval sequences, different models
eval_seqs = generate_mixture_sequences(n=2000)

h_base = extract_last_token_resid(base_model, eval_seqs)       # (2000, 64)
h_ft   = {i: extract_last_token_resid(ft_models[i], eval_seqs) # (2000, 64)
           for i in range(N)}

z_base = sae.encode(h_base)           # (2000, 256)
z_ft   = {i: sae.encode(h_ft[i]) for i in range(N)}

# Per-feature diff
for f in range(256):
    for i in range(N):
        delta[i][f] = z_ft[i][:, f].mean() - z_base[:, f].mean()
```

A large positive $\Delta_i^{(f)}$ means the coin-$i$ finetuned model activates feature $f$ more than the base model on the same inputs — the model has shifted its representation to "use" that feature for coin $i$.

**4c. Steer.**

For the top 5 features per coin (ranked by $|\Delta_i^{(f)}|$), add $\alpha \cdot \mathbf{d}_f$ to the residual stream at all positions, sweep $\alpha$ over 30 values in $[-100, 100]$, and measure mean $P(0)$ over 500 evaluation sequences. The best feature per $N$ is the one whose steering curve spans the widest $P(0)$ range.

| $N$ | Coin biases | Best feature | $P(0)$ range |
|---|-------------|-------------|-------------|
| 2 | 0.05, 0.95 | F30 | 0.48 |
| 3 | 0.05, 0.50, 0.95 | F30 | 0.69 |
| 5 | 0.05, 0.275, 0.50, 0.725, 0.95 | F133 | 0.88 |
| 10 | 0.05, 0.15, …, 0.85, 0.95 | F202 | 0.90 |

The steering curves for the top 3 features per $N$ are shown below.

![Single-token steering curves by N, showing top features per mixture](../outputs/sweep_summary/steering_curves_by_n.png)

## Two steering modes

The steering curves above (Stage 4c) use the **logit lens** method described below. With the best feature per $N$ in hand, we additionally run **continuous autoregressive** steering to compare the two approaches.

### Logit lens (single-step)

1. Generate 500 sequences of length 32 from the mixture (unsteered, from the base model).
2. Run each sequence through the base model in a **single forward pass**, with the hook $\mathbf{h}_\ell \mathrel{+}= \alpha \cdot \mathbf{d}_f$ applied at **every token position** simultaneously.
3. Read $P(\text{next} = 0)$ from the softmax of the **last-token logits only**, and average over the 500 sequences.

Because steering is applied in one pass over pre-generated tokens, there is no feedback: the input tokens are fixed and were not influenced by the steering vector. Each token position sees the same additive perturbation, but only the last-position prediction is reported. (In principle, the steering at earlier positions does affect the last-position output through the transformer's attention, but the *input tokens themselves* are unmodified.)

This is the method used for the [steering curves above](#pipeline-overview) (Stage 4c).

### Continuous autoregressive

1. Start with a single random token.
2. At each of 30 autoregressive steps: run a forward pass with the hook $\mathbf{h}_\ell \mathrel{+}= \alpha \cdot \mathbf{d}_f$ at all positions, sample the next token from the steered output distribution, and append it to the sequence.
3. Report $P(0)$ as the empirical fraction of 0-tokens across all 30 generated tokens, averaged over 500 sequences.

Here the intervention **compounds**: each steered token becomes part of the input for subsequent steps, so the model sees its own steered outputs. This feedback loop can amplify the effect of the steering vector beyond what the logit lens predicts.

Both modes sweep the same set of steering scales (dense near zero, coarse at extremes, from $-500$ to $+500$) and use 500 sequences.

## Results

![Main comparison: continuous autoreg vs logit lens for N=2, 5, 10](../outputs/sweep_summary/autoreg_steering.png)

### Scale-zero baselines

At scale $= 0$ (no steering), the two methods agree closely, both sitting near the theoretical $\mathbb{E}[P(0)] = 0.500$:

| $N$ | Autoreg $P(0)$ | Logit lens $P(0)$ | Expected |
|---|-------------|-----------------|----------|
| 2 | 0.459 | 0.505 | 0.500 |
| 5 | 0.488 | 0.519 | 0.500 |
| 10 | 0.478 | 0.499 | 0.500 |

The logit lens values sit close to 0.500 as expected. The autoreg values show more variance (0.46–0.49) because they depend on 500 sampled sequences of length 30; context-dependent correlations within each sequence inflate the effective variance beyond the naive binomial estimate.

### Near-zero regime ($|\alpha| < 3$)

The near-zero insets in each panel show that the two methods are **nearly identical** at small scales. For $N=10$ at $\alpha = 1$:

- Autoreg: 0.511
- Logit lens: 0.517

The curves begin to visibly diverge starting around $|\alpha| \sim 5$, where the compounding effect of autoregressive feedback becomes non-negligible. At $\alpha = 5$ for $N=10$:

- Autoreg: 0.633
- Logit lens: 0.563

### Compounding at moderate scales

The most interesting regime is moderate-to-large scales, where the autoregressive mode consistently reaches more extreme $P(0)$ values than the logit lens. This is the **compounding effect**: each steered token shifts the input distribution, which interacts with the steering vector at subsequent positions, amplifying the overall intervention.

Example — $N=10$ at $\alpha \approx +42$:

- Autoreg: $P(0) = 0.963$
- Logit lens: $P(0) = 0.913$
- Gap: $0.050$

For $N=2$, the gap is even larger at moderate scales ($\alpha \approx -42$, the high-$P(0)$ direction):

- Autoreg: $P(0) = 0.707$
- Logit lens: $P(0) = 0.559$
- Gap: $0.144$

The $N=2$ gap is wider because the model has only two very different coins ($p=0.05$ and $p=0.95$), creating a sharper transition region where the feedback loop has more room to amplify.

### Saturation at extreme scales

At very large $|\alpha|$, both curves saturate and converge. The saturation values approach — and in some cases exceed — the outermost coin reference lines:

| $N$ | Autoreg min $P(0)$ | Autoreg max $P(0)$ | Logit lens min | Logit lens max |
|---|------------------|------------------|----------------|----------------|
| 2 | 0.097 | 0.921 | 0.144 | 0.864 |
| 5 | 0.036 | 0.969 | 0.044 | 0.954 |
| 10 | 0.032 | 0.973 | 0.038 | 0.967 |

For $N=5$ and $N=10$, the steering pushes $P(0)$ close to $0.03$–$0.04$ and $0.97$, well beyond the outermost coins at $0.05$ and $0.95$. This occurs because at extreme scales, the added direction dominates the residual stream regardless of the input, forcing the logits into a regime no coin would naturally produce.

For $N=2$, saturation is somewhat less extreme ($0.10$–$0.92$ for autoreg, $0.14$–$0.86$ for logit lens), reflecting the sparser feature geometry with only two coins. Feature F30 for $N=2$ has positive $\alpha$ pushing $P(0)$ **down** (toward the $p=0.05$ coin), while Feature F202 for $N=10$ has the opposite convention.

## Discussion

**The compounding effect is real and substantial.** Across all three $N$ values, the autoregressive curves lie outside the logit-lens curves in the transition region ($|\alpha| \sim 5$–$100$), demonstrating that the model's autoregressive feedback amplifies the steering intervention. The gap is especially large for $N=2$ (${\sim}15$ percentage points at moderate scales) where the binary coin structure creates a sharp transition. For $N=5$ and $N=10$ the gap is smaller (${\sim}5$ pp) but consistent, and both methods converge to similar saturation limits at extreme scales.

**$\mathbb{E}[P(0)] = 0.500$ baseline.** All mixtures share this expected value by construction (uniform mixture of linearly-spaced coins from $0.05$ to $0.95$). The observed logit-lens baselines at $0.499$–$0.519$ sit close to this value, consistent across both methods, with small deviations likely reflecting evaluation conditions (shorter sequences, random initialization) rather than a systematic steering artifact.

**Saturation beyond coin reference lines.** The steering curves approach, and for $N=5$ and $N=10$ exceed, the $P(0)$ values of the outermost coins ($0.05$ and $0.95$). This confirms that the SAE feature captures a direction that interpolates between (and extrapolates beyond) the finetuned model behaviors. The $N=5$ and $N=10$ features achieve near-full logit saturation ($0.03$–$0.97$), while the $N=2$ feature saturates more conservatively ($0.10$–$0.92$). This may reflect the quality of the extracted feature and how cleanly it aligns with the coin-identity axis in activation space.

**Practical implication.** For applications where single-token logit-lens evaluation is used as a proxy for actual autoregressive behavior (e.g., evaluating steering vectors before deployment), the logit lens is a good approximation at small scales ($|\alpha| < 5$) and provides a conservative lower bound at larger scales. The compounding effect means that autoregressive steering is strictly more powerful — a vector that achieves a target $P(0)$ under the logit lens will achieve a more extreme $P(0)$ when applied autoregressively.





## Weight steering vs bias steering

The results above establish that steering with an SAE decoder vector can control the model's $P(0)$. But what is the steering vector actually doing? There are two natural interpretations:

### Two hypotheses

**Hypothesis A — Weight steering.** The vector shifts the model's posterior *weight* over coins (ergodic components). At steering scale $\alpha$, the model's effective output is a mixture

$$P(0 \mid \alpha) = \sum_{i=1}^N w_i(\alpha) \cdot p_i$$

where $w_i(\alpha)$ are $\alpha$-dependent mixture weights. Under this hypothesis, $P(0)$ is a convex combination of the coin biases and is therefore bounded by the extreme coins.

**Hypothesis B — Bias steering.** The vector directly shifts the model's output logit, acting as an additive bias:

$$P(0 \mid \alpha) = \sigma\bigl(\text{logit}_0 + f(\alpha)\bigr)$$

for some monotone $f$. Under this hypothesis, the shift is uniform across sequences and $P(0)$ can be pushed continuously to 0 or 1, unconstrained by the coin biases.

These hypotheses make different predictions about the *distribution* of outputs across sequences, not just the mean. At any given scale $\alpha$, we can generate many autoregressive sequences and look at the per-sequence fraction of 0-tokens. If steering shifts mixture weights, different sequences should "commit" to different coins, producing a **multimodal** distribution with high variance. If steering shifts a uniform bias, all sequences should behave similarly, producing a **unimodal** distribution with variance matching a single Bernoulli.

### The per-sequence variance test

We test this using the $N=5$ mixture (coins at $p \in \{0.05, 0.275, 0.50, 0.725, 0.95\}$) with feature F133 (the best steering feature from the all-coin diffing). At each steering scale $\alpha$:

1. Generate 1,000 autoregressive sequences of length 30, with the steering hook active at every forward pass.
2. For each sequence, compute $\hat{p}_0 = \text{(number of 0-tokens)} / 30$.
3. Record the empirical mean $\bar{p}_0$ and variance $\widehat{\text{Var}}(\hat{p}_0)$.
4. Compare to the **Bernoulli prediction**: if all sequences were generated from a single $\text{Bernoulli}(\bar{p}_0)$, the variance of the sample mean over $L=30$ tokens would be $\bar{p}_0(1 - \bar{p}_0) / L$.
5. Compute the **variance ratio** $= \widehat{\text{Var}} / \text{Var}_{\text{Bernoulli}}$. A ratio near 1 supports hypothesis B (bias); a ratio $\gg 1$ supports hypothesis A (weights/mixture).

The Bernoulli baseline is a lower bound: even under pure bias steering, the model's Bayesian updating within a sequence creates some positive autocorrelation that inflates variance slightly. But a mixture of coins creates *far* more excess variance through the between-component term:

$$\text{Var}_{\text{mixture}} = \underbrace{\sum_i w_i \cdot \frac{p_i(1-p_i)}{L}}_{\text{within-coin}} + \underbrace{\sum_i w_i (p_i - \bar{p})^2}_{\text{between-coin}}$$

For example, at $\bar{p}_0 = 0.725$, the Bernoulli variance is $0.725 \times 0.275 / 30 \approx 0.0066$, while a 50/50 mixture of the $p=0.5$ and $p=0.95$ coins (which also has mean 0.725) would give variance $\approx 0.055$ — an 8$\times$ excess from the between-component term alone.

### Results: the distribution is multimodal

![Per-sequence 0-fraction distributions at 7 steering scales, from extreme negative to extreme positive](../outputs/sweep_summary/variance_test_all_scales.png)

The figure above shows histograms of per-sequence 0-fractions at seven steering scales, with the Bernoulli prediction overlaid (dashed orange). The results decisively support **weight steering** (hypothesis A):

**At extreme scales** ($|\alpha| \geq 500$): The distribution is a tight unimodal peak matching the Bernoulli prediction (ratio $\approx 1.0$). At $\alpha = -500$, nearly all sequences cluster at $\hat{p}_0 \approx 0.96$ — the model has fully committed to the $p=0.95$ coin. At $\alpha = +500$, they cluster at $\hat{p}_0 \approx 0.04$ (the $p=0.05$ coin). The steering has concentrated all the mixture weight onto one coin, and the per-sequence variability is just binomial noise.

**At moderate scales** ($|\alpha| \approx 8.6$): The distribution is broad and multimodal, with a variance ratio of $\sim 10\times$ Bernoulli. Different sequences commit to different coins — some generating mostly 0s, others mostly 1s — despite starting from identical steering conditions. The only source of variation is the random initial token and sampling noise, yet the model amplifies these into distinct coin commitments.

**At zero scale**: The distribution is maximally spread (ratio $\approx 13$), reflecting the unsteered uniform mixture prior. This is the baseline: with no steering, the model distributes weight across all five coins.

![Zoomed histogram at the target P(0)~0.725](../outputs/sweep_summary/variance_test_target.png)

The zoomed view at $\bar{p}_0 \approx 0.74$ (scale $= -8.6$) makes the discrepancy vivid. The empirical variance ($0.068$) is **10.6$\times$** the Bernoulli prediction ($0.006$). The distribution is clearly multimodal — incompatible with any single-Bernoulli model — and extends from $\hat{p}_0 \approx 0$ to $\hat{p}_0 \approx 1$, with clusters near individual coin biases.

![Overview: mean P(0), variance comparison, and variance ratio across all scales](../outputs/sweep_summary/variance_test_overview.png)

The overview panel (c) shows the variance ratio across all scales. The characteristic shape — peaking at $\sim 13\times$ near $\alpha = 0$ and decaying to $\sim 1\times$ at extreme scales — is exactly what the weight-steering hypothesis predicts: at small scales, the model maintains an uncertain mixture; as $|\alpha|$ grows, it concentrates onto a single coin, and the between-component variance vanishes.

### The steering feature spans all coins

A surprising aspect of these results is the feature's reach. Feature F133 was selected because it had the largest activation diff for **coin 1** ($p = 0.275$) — that is, the $p=0.275$-finetuned model activated this feature most differently from the base model. Yet steering in the opposite direction pushes $P(0)$ all the way to $0.96$, matching the $p = 0.95$ coin at the other end of the spectrum.

This reveals that the feature is not a "coin-1 detector." It is better understood as a **linear axis in mixture-weight space** — a direction along which the model's internal representation of "which coin am I in?" varies. The model-diffing procedure discovered this axis by observing that the $p=0.275$ finetuned model moved furthest along it (presumably because that coin is the most "surprising" shift from the uniform prior), but the axis itself spans the full range from $p=0.05$ to $p=0.95$.

This is consistent with the linear-probe results from Stage 2: the model's residual stream linearly encodes the Bayesian posterior $\eta_t$ with $R^2 > 0.998$. The steering vector is simply a direction in this linear belief subspace, and adding $\alpha \cdot \mathbf{d}_f$ translates the model's belief along that direction — shifting mixture weight from one end of the coin spectrum to the other.

### Quantitative model comparison

The variance test establishes that steering produces mixture-like distributions. We can go further and directly fit the two competing generative models to the observed per-sequence count data, comparing them quantitatively via log-likelihood and BIC.

**Model W (weight steering).** The coin biases are fixed at their true values $p_i \in \{0.05, 0.275, 0.50, 0.725, 0.95\}$. The free parameters are the mixture weights $w_1(\alpha), \ldots, w_5(\alpha)$ (subject to $\sum_i w_i = 1$, so 4 free parameters). Each sequence is drawn from coin $i$ with probability $w_i$, then tokens are i.i.d. $\text{Bernoulli}(p_i)$. The per-sequence count of 0-tokens follows a mixture of binomials:

$$P(k \mid \alpha) = \sum_{i=1}^5 w_i(\alpha) \cdot \binom{L}{k} p_i^k (1 - p_i)^{L-k}$$

We fit the weights via EM (iterating E-step responsibilities and M-step weight updates until convergence).

**Model B (bias steering).** The weights are fixed at uniform ($1/5$ each). The free parameter is a single logit shift $\delta(\alpha)$ applied to all coins simultaneously:

$$p_i'(\alpha) = \sigma\bigl(\text{logit}(p_i) + \delta(\alpha)\bigr)$$

This shifts all five coin biases in parallel through logit space, preserving their ordering but sliding them toward 0 or 1. We fit $\delta$ by maximizing the log-likelihood over a bounded search.

Since Model W has 4 free parameters and Model B has 1, we compare using both raw log-likelihood and the Bayesian Information Criterion ($\text{BIC} = -2\,\text{LL} + k \ln n$), which penalizes the extra parameters of Model W.

![Model comparison overview: LL difference, BIC difference, fitted weights, fitted logit shift](../outputs/sweep_summary/model_fit_overview.png)

**Model W wins at every scale except zero**, even after the BIC penalty. The LL advantage ranges from $+7$ at scale $= 0$ (essentially tied) to $+450$ at extreme scales. The BIC difference (panel b) tells the same story: the 3 extra parameters of Model W are overwhelmingly justified by the improvement in fit.

**Panel (c) is the key result.** It shows the fitted mixture weights as a function of steering scale. As $\alpha$ sweeps from $-500$ to $+500$, the weight smoothly transfers from the $p=0.95$ coin (at large negative $\alpha$) through the intermediate coins to the $p=0.05$ coin (at large positive $\alpha$). At moderate scales (e.g. $\alpha \approx -8$), the weights are spread across multiple coins: $w \approx [0.06, 0.04, 0.10, 0.30, 0.50]$ — the model is mostly in the $p=0.725$ and $p=0.95$ coins but with residual weight on the others. This is the quantitative portrait of the weight-steering mechanism: the SAE feature controls a smooth dial over the coin-identity posterior.

**At scale $= 0$**, both models fit equally well, which is expected: the unsteered model really is a uniform mixture, and Model B's logit shift of $\delta \approx 0$ is equivalent.

![Model fits at selected scales: top row Model W, bottom row Model B](../outputs/sweep_summary/model_fit_histograms.png)

The histogram fits at three selected scales ($\bar{p}_0 \approx 0.27, 0.52, 0.76$) make the difference vivid. Model W's mixture of binomials (top row, red curve) tracks the multimodal empirical distribution closely, with each component peak aligning with a coin bias. Model B's shifted mixture (bottom row) predicts peaks at the wrong locations — shifting all coin logits by a common $\delta$ cannot reproduce the observed clustering at the *original* coin biases.

## Process setup
The coins are the simplest case of a non-ergodic processes. Following the non-ergodic Mealy HMM formalism of [Riechers et al. (2025)](https://arxiv.org/abs/2505.18373), each experiment defines an HMM $M = (\mathcal{X}, \mathcal{S}, \eta_0, \{T^{(x)}\}_{x \in \mathcal{X}}, Q(c))$ where:

- **Vocabulary** $\mathcal{X} = \{0, 1\}$ (binary tokens).
- **Hidden states** $\mathcal{S} = \{1, 2, \ldots, N\}$, one per coin.
- **Initial distribution** $\eta_0 = (1/N, \ldots, 1/N)$ (uniform — coin selected once at random).
- **Substochastic transition matrices** $T^{(x)}$ ($N \times N$), one per token $x \in \mathcal{X}$. The entry $T^{(x)}_{s,s'} = \Pr(x, s' \mid s)$ gives the joint probability of emitting token $x$ and transitioning to state $s'$, given current state $s$.
- **Process selection distribution $Q$** — A distribution over the $N$ independent processes — here each lives in a one-dimensional space. Here we always use the uniform distribution $Q \sim \mathrm{Unif}\{1,\ldots,N\}$.

In this setup we first sample from the process distribution $Q$ to select a given coin, and then generate a rollout of length $L$ from that coin. Since each coin stays in its own state forever and emits independently, both $T^{(0)}$ and $T^{(1)}$ are diagonal:

$$T^{(0)} = \begin{pmatrix} p_1 & & \\ & p_2 & \\ & & \ddots & \\ & & & p_N \end{pmatrix}, \qquad T^{(1)} = \begin{pmatrix} 1-p_1 & & \\ & 1-p_2 & \\ & & \ddots & \\ & & & 1-p_N \end{pmatrix}$$

Their sum is the full (row-stochastic) transition matrix $T = T^{(0)} + T^{(1)} = I_N$ — the identity, reflecting that the hidden state never changes. The coin biases are linearly spaced: $p_i = 0.05 + 0.9 \cdot (i - 1) / (N - 1)$ for $i = 1, \ldots, N$, ranging from $0.05$ to $0.95$.

At generation time, a process $c \sim Q$ is drawn once, selecting coin $i$. The entire sequence is then produced by rolling out within that single process: each token $x_j$ is drawn i.i.d. from $T^{(x_j)}_{ii}$. Marginalizing over the process selection, the probability of an observed sequence $x_{1:t}$ is

$$P(x_{1:t}) = \sum_{i=1}^{N} Q(i) \; \prod_{j=1}^{t} T^{(x_j)}_{ii} \;=\; \eta_0^\top \, T^{(x_1)} T^{(x_2)} \cdots T^{(x_t)} \, \mathbf{1}$$

where $\eta_0 = Q$ (since each process corresponds to exactly one hidden state) and $\mathbf{1}$ is the all-ones vector. Since all $T^{(x)}$ are diagonal, the matrix product collapses: for a sequence with $k$ zeros and $t - k$ ones,

$$P(x_{1:t}) = \frac{1}{N} \sum_{i=1}^{N} p_i^{\,k} \, (1 - p_i)^{t-k}.$$

The Bayes-optimal predictor maintains a posterior belief $\eta_t$ over which process generated the sequence. After observing $x_{1:t}$, the unnormalized posterior is $\eta_0^\top T^{(x_1)} \cdots T^{(x_t)}$, giving

$$\eta_t(i) = \frac{Q(i) \; p_i^{\,k} \, (1-p_i)^{t-k}}{\sum_{j=1}^{N} Q(j) \; p_j^{\,k} \, (1-p_j)^{t-k}}$$

which under uniform $Q$ simplifies to

$$\eta_t(i) = \frac{p_i^{\,k} \, (1-p_i)^{t-k}}{\sum_{j=1}^{N} p_j^{\,k} \, (1-p_j)^{t-k}}.$$

The predictive distribution for the next token is

$$P(x_{t+1} = 0 \mid x_{1:t}) = \sum_{i=1}^{N} \eta_t(i) \cdot p_i = \frac{\sum_{i=1}^{N} p_i^{\,k+1} \, (1-p_i)^{t-k}}{\sum_{i=1}^{N} p_i^{\,k} \, (1-p_i)^{t-k}}.$$

Before any observations, the marginal is simply $\mathbb{E}[P(0)] = \sum_i Q(i) \cdot p_i = \frac{1}{N}\sum_i p_i = 0.500$, the mean of the linearly-spaced biases, regardless of $N$.