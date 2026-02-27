---
author: Jamie Stephenson
date: 2026-02-24
tags: in-progress
---
This file contains rough notes on my current understanding of the project. The aim is to distill into separate files.

## Choices

There are a few choices required to arrive at a formula for feature resolved attention:

1. **How** the features are extracted.
2. **Where** the features are extracted from.
3. **What** object composed of these features is the correct one to study.

### Dictionary learning variant

**Regular SAE**. Reconstruct residual stream activations at a specific layer.

**Acausal crosscoder**. Reconstruct residual stream activations from *every* layer. Expensive. But also for a given attention head you would expect this to be *really* sparse: the top 100 cross-layer features are far less likely to interact in a given head compared with the top 100 features from an SAE trained on the same layer that the head is from. Obviously you would expect a cross coder to have more features so maybe to compare like-to-like you would need to consider more than the top 100 but then you start running into performance issues: inherently harder job to extract more features from this bigger space? Idk this all just my intuition.

**Model diffing crosscoder**. Given two models and one input, map set of activations from one model to the equivalent set of activations from the other. Could be interesting in e.g. sleeper agents context, where a model has had a "new feature finetuned into it". Think the point is that model diffing will be more likely to extract this feature, compared to just training an acausal crosscoder on the finetuned model?

### Activation extraction point

Intuitively, extracting $x^{(i)}$ from  `LN1.pre_attention` makes the most sense, but Dmitry wants the option to extract from elsewhere.
> [!question] Why would we want to extract from anywhere else?
>I am not sure, if we are looking to decompose $A_{ij}$, then any other extraction point will have a layernorm in between it and $A_{ij}$, unless you extract from within the attention layer itself (i.e. keys and queries).

### Specific FRA object

We first decompose attention into mean-query/mean-key, specific-query/mean-key, mean-query/specific-key, specific-query/specific-key terms. The only way two features at different sequence positions can interact is via the specific-query/specific-key term.

This is a $s\times s\times d_h\times d_h$ shaped object, where $s$ is the sequence length. Specifically
$$
A'_{ijkl} = u_{k}^{(i)}\,u_{l}^{(j)} \left(W_{Q}W_{\mathrm{dec}}\hat{e}_{k}\right)^{\top}W_{K}W_{\mathrm{dec}}\hat{e}_{l}.
$$
Couple of options for what to study from this.

**Average over sequence position**. Gives a $d_h\times d_h$ object where $d_h$ is the number of SAE latents. This gives us a high level picture of "within this context, for this given head, which features are interacting" as opposed to looking at the token level.

**Feature resolved QK circuit**. Consider only the *data independent* part
$$
(W_{Q}W_{\mathrm{dec}})^{\top}W_{K}W_{\mathrm{dec}}.
$$
This also gives a $d_h\times d_h$ object. But this one tells us exactly how much each each feature pair *would* interact within the given head *if* they were both active. Unfortunately there are lots of feature pairs that are never active together so there is no reason to expect the interaction values for such pairs to have any significance. We therefore do not expect this object to be sparse/interpretable. I think Dmitry mentioned this is indeed what we see empirically.

> [!question] Why isn't data independent part sparse?
> Thinking about it, with no incentive to produce queries and keys that ever align, shouldn't we expect features that never interact with a given feature to give keys/queries randomly distributed compared to the given feature's queries/keys? This *would* give a sparse data independent part, as random vectors are likely approximately orthogonal in high dimensions.
>
>*Answer: it is kinda sparse, ~10% non-zero interactions. But for our interpretability purposes, this is nowhere near sparse enough, given how large it is.*

**Masked**. Because $u$ is sparse by design, the full object will be sparse, but when *comparing between* feature pairs that have different mean activations, this full object will perhaps misrepresent which feature pair is interacting the most. Indeed, a pair of features that are both on average really strongly activated, but interact only a little, would show up just as strongly as a pair that both activate only a little but interact strongly. We care more about the latter pair and so one option is to study
$$
\tilde A_{ijkl}=\mathbf{1}[u_{k}^{(i)},\,u_{l}^{(j)}>0] \left(W_{Q}W_{\mathrm{dec}}\hat{e}_{k}\right)^{\top}W_{K}W_{\mathrm{dec}}\hat{e}_{l}.
$$
**Normalised**. In the crosscoder features interactions paper they say they use
$$
\tilde A_{ijkl}=\frac{1}{\mu_k}u_{k}^{(i)}\frac{1}{\mu_l}u_{l}^{(j)} \left(W_{Q}W_{\mathrm{dec}}\hat{e}_{k}\right)^{\top}W_{K}W_{\mathrm{dec}}\hat{e}_{l},
$$
where $\mu_k$ is the mean activation of latent $k$. To me this seems like a pretty similar alternative to the masked version, with perhaps a bit more contextual nuance as it accounts for how activated the latents currently are in a fairer way that the full object.

> [!note] Average over sequence position
> While only mentioned for the full object above, you can of course average over sequence position for the masked and normalised variants as well.

## Comparison to MLP

The compact proofs feature interactions paper quantifies interaction between a pair of features within an MLP layer. 

> [!note] Explanation of derivation in words: UNFINISHED, finding understanding this tricky, will require several passes.
>The way they do this is by writing the output of an MLP as a sum across neurons where only one feature (the most "dominant") is allowed to contribute to a given neuron's term, plus an error term which accounts for feature interaction. They then rearrange to get an explicit formula for the error term which can be decomposed as a sum over features. Each of these terms either contains a single feature or a ___ and then represents the interaction...

Specifically, given a neuron $k$ from MLP layer $l$, residual stream activations $x$ at a specific token position, the interaction $I$ between feature $j$ and the *dominant* feature for the activation of $k$ on $x^{l-1}$ is

$$
I^{l}_{(x,k)}(i,j) \equiv \frac{\| (W^{l}_{\text{out}})_k \|}{N^{l}}  \left\|  u_j \left( W^{l}_{\text{in}} W^{l-1}_{\text{dec}} \hat{e}_j \right)_k  \right\|,
$$

where $u$ is are the crosscoder activations at the same token position as $x$ and $N^l=\|x^l\|/d$. Not sure what $d$ is here, assume its residual stream dimension. 

*Dominant* is defined as follows

$$
v_\text{max}(x,k)  
\;=\;  
\operatorname*{argmax}_{v}  
\left|  
u_v \,\bigl(W_{\mathrm{in}}^{(l)}\, W_{\mathrm{dec}}^{(l-1)}\, \hat e_v\bigr)_k  
\right|.
$$

Basically, write the pre-activations of a MLP neuron as a sum over crosscoder features, the term of the sum with largest magnitude corresponds to the dominant feature.

These values would be us eful for a dashboard as they will allow us to compare which features interact *within* token positions via the MLP with features that interact *across* token positions via attention.

> [!question] What if we want the interaction between two non-dominant features?

## Uses

These all currently look pretty similar and high level in my head:
Take a model that has unsafe property X, elicit this with prompt Y and study FRA to try and better understand how it works, and then hopefully intervene to prevent this property while minimising impact on model by allowing involved features to still interact with other features for "safe purposes".

### Sleeper agents

A sleeper agent is a model that has had a backdoor finetuned into it: it behaves normally until it encounters a specific trigger in its input, at which point it produces harmful outputs. By training a crosscoder between the base model and its finetuned (backdoored) variant, we can extract features that are *dedicated* to the finetuned model, i.e. features that only activate in the backdoored version. Applying FRA to these dedicated features lets us go further: we can see exactly which feature pairs interact within specific attention heads when the trigger is present. FRA gives us a principled way to decompose those signatures at the feature level rather than at the coarser attention-head level.

In practice, could train a model diffing crosscoder between a base model and a suspected sleeper agent, identify features dedicated to the finetuned model, and then use FRA (e.g. the normalised or masked variant) to study which of those features are interacting in the attention heads of the later layers. A sparse, localised interaction pattern involving dedicated features would be a strong signal that a specific trigger-response mechanism has been finetuned into the model. Hopefully we would be able to do something like intervene at the feature interaction level so remove the trigger while preserving the ability for the features involved to perform harmless computation that they still need to be involved in.

### Jailbreaking

Example discussed at interview. #todo: Short notes on this and link to David Bau paper on harmfulness/refusal.

### Detecting deceptive alignment

A more ambitious application of FRA is in detecting *deceptive alignment*, where a model has learned to behave well during training and evaluation but pursues different objectives when deployed. Unlike sleeper agents, which involve an externally inserted trigger, deceptive alignment is hypothesised to arise naturally during training. The challenge is that a deceptively aligned model may not have an explicit trigger feature; instead, it may have learned a subtle internal representation of "am I being evaluated?" that modulates its behaviour. FRA could help here by providing a way to search for context-dependent feature interactions in attention heads.

Specifically, could compare the FRA decomposition across prompts that vary only in cues that a model might use to distinguish evaluation from deployment (e.g. the presence of system prompts, evaluation markers, or user identity signals). Feature pairs that interact strongly in attention heads only when deployment-like cues are present, but are inactive during evaluation-like contexts, would be candidates for the kind of context-sensitive circuitry that deceptive alignment requires. More speculative than the sleeper agent and jailbreaking applications.

## Toy models

We would like to study known circuits in toy models (induction, iteration etc.) through the lens of FRA. As FRA only tells us about the QK circuit, we ideally want a toy task for which the OV circuit is not doing anything important (e.g. it is simply the identity).
