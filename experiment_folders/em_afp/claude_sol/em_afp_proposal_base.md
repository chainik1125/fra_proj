# EM AFP proposal

**TL;DR**: The most interesting fact about weird generalizations is that the _local_ generalization (factor) competes with the _global_ one, and that the global one wins. Any toy model of EM must capture this phenomenology. This is a simple sign-of-life experiment that tries to use Semi-Factored Processes (SFPs) to explain this finding.

## Background


## Proposal

### In-brief, key operational choices and parameters

1. Model pre-training $\iff$ SFP process $[T_M\oplus T_A]\otimes [T_D\oplus T_O]$.
2. 'Induced broad misalignent' $\iff$ ratio of posterior probabilities from the transformer $\mu_{MD}/\mu_{MO}$. A ratio $\sim 1$, and $\mu_{AD},\mu_{AO}<<\mu_{MO},\mu_{MD}$ defines emergent misalignment.
3. Component HMMs: We can start with the same leaky reset HMMs
    - Confirm this continues to hold for Mess3
4. 

### Part 1: Train the AFP process $[T_M\oplus T_A]\otimes [T_D\oplus T_O]$

### Part 2: Assess representational quality

What we really want here is to be able to measure the component prior weights $\{\mu_i\}$, but I'm not sure we yet have a good tool for doing that.

1. Calculate $R^2$ on regression to:
    - global state $[T_M\oplus T_A]\otimes [T_D\oplus T_O]$
    - local misaligned factor $T_M\otimes T_D$
    - global misaligned factor $T_M\otimes T_O$
    - Each global alignment factor on their own $T_A,T_M$.  
    - aligned factors $T_A\otimes T_D$, $T_A\otimes T_O$.
    \
    Any other measures of how well each component is represented?

### Part 3: Induce misalignment

Pick a procedure to induce misalignment, and then measure the operationalization of the `misalignment rate'.  

I think what we did before is reasonable - the Bayes posterior onto the misaligned sector $T_M$. Previously this was just an ergodic component. In the SFP context where it appears as a factor, I think the right thing to do is to marginalize over the domain factor $T_O,T_D$.

**Note** Once we've operationalized the 'misalignment rate' as the posterior estimate of the $\mu_i$, the question then just becomes what is happening at the representational level.

#### 3a: Conventionally: through finetuning

Finetune on $T_M\otimes T_D$. It would be good to choose processes which completely polarize into the final sector, as we had in the previous model. Do the classical processes like Mess3, TomQ, have this property (I would guess there's a polarizability surface with sequence length)?

#### 3b: Through ICL

The same thing, but through ICL examples rather than fine-tuning. 


### Part 4: Probe representations post induced misalignment

Repeat part 2, but now inducing misalignment through in-context examples rather than through finetuning.







## Hypotheses and pre-registered predictions

#### 1. ICL retains all factors, fine-tuning eliminates weakens them

Assume we have some map $f_\mu(a)$ from activations to the set of prior weights, and an associated score $\mathcal{L}_M(\mu)$ which measures the quality of this map. Assume also we have a map $f_\eta(a)$ to each representation component and an associated $\mathcal{F}_M(\eta)$ (We've previously been using the Moore-Penrose inverse for $f_\eta(a)$ and the $R^2$ as the score $\mathcal{F}_M$.)

**Hypothesis (ICL narrow, FT broad)** In ICL, at least in this setup, it is difficult to induce narrow-broad misalignment $\mu^{ICL}_{MD}>> \mu^{ICL}_{MO}$, in finetuning.

_Prediction_: This should hold.

**Hypothesis (ICL updates rep posteriors)**: In ICL, the representations are _kept_, the priors are updated: $\mathcal{F}_{\text{finetune}}(\eta)\sim \mathcal{F}_{0}(\eta)$ $\mu_{i}=\delta_{i,MD}$. 

_Prediction_: This should hold.

**Hypothesis (FT changes reps)**: Under finetuning, the non-selected representations are _damaged_, the priors are updated: $\mathcal{F}_{\text{finetune}}(\eta_{\neq MD})<< \mathcal{F}_{0}(\eta_{\neq MD})$, $\mu_{i}=\delta_{i,MD}$. 

_Prediction_: I'm unsure of this one. I lean rep change, but could see it either way. 



## Bonus results

1. MatTXC SAE: Try to see how well this distinguishes the different EM. The idea is to adapt the MatTXC that we earlier showed recovers a very high share of the Bayes ceiling: sae_day/notes/txc_matryoshka_results.md . The zeroth order question is just

## Open questions

1. Most important one: How do I probe for the priors on each component $\{\mu_i\}$ ? Without a good measurement of these, I cannot distinguish between what I would expect


## Limitations

1. More hierarchy in the persona? In what sense is your global factor global instead of one of two local choices?
2. 

## Proposal for posterior weights

There are two related quantities here, but they play different roles:
$$
p_\theta(x\mid w)\Rightarrow \hat\mu_{\mathrm{beh}}(w)
\qquad\text{and}\qquad
a_{\ell,p}(w)\Rightarrow \hat\mu_{\mathrm{rep}}(w).
$$
The behavioral posterior is the operational misalignment readout: it says what posterior component weights the model is acting as if it has. The representational posterior is the mechanistic probe: it asks whether those same posterior weights are encoded in the residual stream at layer $\ell$ and position $p$.

For
$$
[T_M\oplus T_A]\otimes [T_D\oplus T_O],
$$
the four components are
$$
MD,\quad MO,\quad AD,\quad AO.
$$
For a context $w$, define $\mu_i(w)$ as the posterior mass on component $i$.

### Behavioral posterior weights

This is not an independent mechanistic measurement if we define the misalignment rate by posterior mass on the misaligned factor. It is the behavioral definition of the phenomenon. In this setting,
$$
\mu^{\mathrm{beh}}_M(w)
=
\mu^{\mathrm{beh}}_{MD}(w)+\mu^{\mathrm{beh}}_{MO}(w)
$$
is the misalignment rate.

At the behavioral level, the model's next-token distribution should be well approximated by a mixture of the component-conditional next-token distributions:
$$
p_\theta(x\mid w)\approx \sum_i \mu_i(w)\,p_i(x\mid w),
$$
where $p_\theta$ is the transformer distribution and $p_i$ is the Bayes-optimal next-token distribution conditional on being in component $i$.

The general estimator is therefore
$$
\hat\mu_{\mathrm{beh}}(w)=
\arg\min_{\mu\in\Delta^3}
D_{\mathrm{KL}}\left(
p_\theta(\cdot\mid w)
\;\middle\|\;
\sum_i \mu_i p_i(\cdot\mid w)
\right).
$$
This gives a direct estimate of the component weights the model behaviorally uses in prediction.

In the tagged-completion setting there is a simpler estimator. Marginalize the model's next-token distribution over content and keep only the component tag probabilities:
$$
y_t(w)=p_\theta(\text{next tag}=t\mid w),
$$
where $t\in\{MD,MO,AD,AO\}$. If the SFP tag evidence matrix is
$$
W[i,t]=p(\text{next tag}=t\mid \text{component }i),
$$
then
$$
y(w)=\mu(w)W.
$$
When $W$ is full rank, estimate the posterior weights by
$$
\hat\mu_{\mathrm{beh}}(w)=y(w)W^{-1},
$$
followed by projection back onto the simplex if needed. In practice, a simplex-constrained least-squares fit is probably more stable:
$$
\hat\mu_{\mathrm{beh}}(w)=
\arg\min_{\mu\in\Delta^3}
\lVert y(w)-\mu W\rVert_2^2.
$$

### Representational posterior weights

The non-redundant measurement is whether the operational posterior is represented internally. At the representational level, use the exact SFP/HMM filter to label each context $w$ with its Bayes posterior:
$$
\mu^\star(w)=(\mu^\star_{MD},\mu^\star_{MO},\mu^\star_{AD},\mu^\star_{AO}).
$$
Then fit probes from activations to this posterior:
$$
a_{\ell,p}(w)\mapsto \hat\mu_{\mathrm{rep}}(w).
$$
The preferred probe is a linear map into natural coordinates followed by a softmax:
$$
z_{\ell,p}(w)=Ba_{\ell,p}(w)+b,
\qquad
\hat\mu_{\mathrm{rep}}(w)=\mathrm{softmax}(z_{\ell,p}(w)).
$$
Train it with KL loss, plus mild regularization:
$$
\min_{B,b}\;
\mathbb E_w\left[
D_{\mathrm{KL}}\left(
\mu^\star(w)
\;\middle\|\;
\hat\mu_{\mathrm{rep}}(w)
\right)
\right]
+\lambda\lVert B\rVert^2.
$$
This is better than four independent regressions to $\mu_i$, because the output is constrained to be a valid posterior distribution. It also gives a direct analogue of the behavioral posterior estimate, but measured from the representation rather than from the logits.

The core representational question is therefore
$$
\hat\mu^{\mathrm{rep}}_M(w)
\approx
\hat\mu^{\mathrm{beh}}_M(w),
$$
and, more strongly,
$$
(\hat\mu^{\mathrm{rep}}_{MD},\hat\mu^{\mathrm{rep}}_{MO},\hat\mu^{\mathrm{rep}}_{AD},\hat\mu^{\mathrm{rep}}_{AO})
\approx
(\hat\mu^{\mathrm{beh}}_{MD},\hat\mu^{\mathrm{beh}}_{MO},\hat\mu^{\mathrm{beh}}_{AD},\hat\mu^{\mathrm{beh}}_{AO}).
$$

In addition to the full component posterior, decode the factor marginals:
$$
\mu_M=\mu_{MD}+\mu_{MO},
\qquad
\mu_D=\mu_{MD}+\mu_{AD}.
$$
Also decode or compute the interaction coordinate
$$
\chi=\log\frac{\mu_{MD}\mu_{AO}}{\mu_{MO}\mu_{AD}}.
$$
The interaction coordinate distinguishes a representation that merely contains independent persona/domain factors from one that represents correlated SFP components. If $\chi=0$, the posterior factorizes across persona and domain; if $\chi\ne0$, the representation contains component-level correlation.

### Comparison readouts

The main readout should treat the behavioral posterior as the target phenomenon and compare it to representational decodability across base, fine-tuned, and ICL conditions on matched neutral prompts:
$$
\hat\mu_{\mathrm{beh,base}}(w),\qquad
\hat\mu_{\mathrm{beh,FT}}(w),\qquad
\hat\mu_{\mathrm{beh,ICL}}(w),
$$
and
$$
\hat\mu_{\mathrm{rep,base}}(w),\qquad
\hat\mu_{\mathrm{rep,FT}}(w),\qquad
\hat\mu_{\mathrm{rep,ICL}}(w).
$$
Useful summary statistics are log-odds shifts such as
$$
\Delta\log\frac{\mu_{MO}}{\mu_{AO}},
\qquad
\Delta\log\frac{\mu_{MD}}{\mu_{AD}},
\qquad
\Delta\log\frac{\mu_{MD}}{\mu_{MO}}.
$$
These distinguish a global misalignment tilt from a local domain-specific tilt. For example, finetuning on $MD$ should be tested for whether it raises $MO$ relative to $AO$ after controlling for the local $D$ effect.

The behavioral-vs-representational comparison distinguishes several mechanisms:

1. If the behavioral misalignment rate rises and $\hat\mu_{\mathrm{rep}}$ shifts with it, then the behavior corresponds to an internal represented posterior shift.
2. If the behavioral misalignment rate rises but $\hat\mu_{\mathrm{rep}}$ does not, then finetuning changed the output/readout policy rather than the encoded posterior.
3. If $\hat\mu_{\mathrm{rep}}$ shifts but the behavioral posterior does not, then the posterior is represented but not behaviorally used.
4. If ICL and finetuning produce similar behavioral posteriors but different representational signatures, then they are inducing the same operational misalignment rate through different mechanisms.
