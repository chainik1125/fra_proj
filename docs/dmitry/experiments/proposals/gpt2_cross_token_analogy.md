# GPT-2: cross-token concept composition through QK and OV

Proposal, 17 September 2026. The first measurement stage is now an explicit
**teacher-forced, unsteered SAE feature trace** before any concept edit. See
[the baseline experiment](../../../../experiments/fra_concept_trace_20260917/README.md)
for pretrained checkpoint selection, per-token activations and quality checks.
The [first attention-layer-5 FRA investigation](../../../../experiments/fra_concept_trace_20260917/fra_layer5_ln1/REPORT.md)
now measures QK and OV paths on the female/male-monarch definitions. It uses a
normalized pretrained layer-4 dictionary transferred directly to layer-5 ln1,
with bias/error reconstruction checks. Nonzero interactions were measured, but
causal effects of these selected paths were small. The broader cross-token
control and selectivity design below remains a proposal.

## Relevant papers

The closest match to the remembered SAE paper is Li et al., **The Geometry of
Concepts: Sparse Autoencoder Feature Structure**, first posted in 2024 and
published in Entropy in 2025:

- https://arxiv.org/abs/2410.19750
- https://arxiv.org/html/2410.19750v2#S3
- https://github.com/ejmichaud/feature-geometry

Its Section 3 studies analogy-like geometric structure in activations and SAE
decoder directions using Gemma-2-2B. It distinguishes model-activation and
SAE-decoder geometry. The initial SAE search was noisy; removing distractor
directions with LDA reveals cleaner structures in the activation analysis.
The familiar king/man/woman/queen relation motivates the analysis. This should
not be cited as a causal demonstration that an attention circuit computes that
specific analogy from four SAE atoms.

The standard analogy is `king - man + woman ≈ queen`; removing the male component
alone does not supply the female component. It is an approximate semantic
relationship, not an identity to impose on GPT-2's raw decoder directions.

A directly useful computational precedent is Merullo, Eickhoff and Pavlick,
**Language Models Implement Simple Word2Vec-style Vector Arithmetic**, NAACL 2024:

- https://arxiv.org/abs/2305.16130
- https://arxiv.org/html/2305.16130v3
- https://github.com/jmerullo/lm_vector_arithmetic

They study relational tasks, including capitals and word transformations, with
detailed analysis of GPT-2 Medium and additional model sizes. A key transformation
is often an additive FFN update. Their argument-then-function observation makes
it important to distinguish transport of an operand from performing the semantic
transformation itself.

Todd et al., **Function Vectors in Large Language Models**, provides an additional
causal-attention precedent, identifying heads that transport task representations:
https://arxiv.org/abs/2310.15213 . It is related evidence, not proof of this circuit.

## Main question

Can a feature identified at an earlier token be selectively routed and
transported so that it changes the semantic composition at a later position?

The strongest initial witness would be a transported royalty component that
supports `king` in a male recipient context and `queen` in a female recipient
context. That differs from adding a fixed queen-output direction everywhere.
We also want the reverse manipulation: change transported gender information
while retaining royalty.

Separate three outcomes:

1. Geometric analogy among representations.
2. A causal cross-token feature path controlling the analogy answer.
3. Better selectivity than matched single-feature or token-level interventions.

The first does not establish the second, and the second does not automatically
establish the third. Initially seek an interpretable causal witness.

## Model and small behavior screen

Start with frozen GPT-2 Small and the existing GPT-2 residual SAE support in the
repository. The archived release is `gpt2-small-res-jb`, with sites such as
`blocks.8.hook_resid_pre`. Confirm actual available checkpoints and their config
before loading; do not use concatenated-head hook_z dictionaries as Q/K inputs.
An example published config is at:
https://huggingface.co/jbloom/GPT2-Small-SAEs-Reformatted/blob/main/blocks.9.hook_resid_pre/cfg.json .

Screen several natural and symbolic templates, including:

```
Man is to king as woman is to
man : king :: woman :
A woman who rules a kingdom is called a
```

The first two supply the analogy across positions; the last gives a complementary
composition prompt. Do not presume Small performs the task reliably. Fix
templates using a discovery split and preserve failures. GPT-2 Medium is a
predeclared fallback if Small lacks the behavior; SAE coverage then needs a
separate check, or local SAE training at causally identified sites.

For one symbolic template, use this factorial grid:

| Mapping | Recipient | Intended completion |
|---|---|---|
| man -> king | woman | queen |
| man -> king | man | king |
| man -> man | woman | woman |
| man -> man | man | man |

The target word queen must be absent from its prompt. Check leading-space and
capitalization tokenization explicitly and score full completion strings when
they have multiple tokens. Use the final prefix position, before the answer is
generated, as the first analysis site.

This grid is diagnostic, not sufficient: repetition and analogy formatting may
enable lexical heuristics. Add held-out natural templates, reversed analogies,
gendered relation families such as uncle/aunt and brother/sister, identity maps,
and shuffled or irrelevant premises. Later transfer tests must include paraphrases
and cases without the literal word king.

Primary measurements: completion probabilities, correct-vs-competitor log odds,
and the full four-word distribution. Queen-versus-king alone tests gender while
potentially missing destruction of royalty. Also compare queen against woman,
and king against man. Never score success solely as an increase in queen.

## Locate computation before assigning feature labels

Cache residuals before attention, after attention and after the MLP at all layers.
Use matched prompt counterfactuals to identify heads and source positions with
causal effects on gender and role separately. Begin with whole-head/source-message
patches as localization oracles; do not report these as FRA interventions.

Find where the combined answer first becomes causally available. It may be at
an intermediate token rather than the final answer position. Transporting an
already-composed queen representation from an earlier source is a valid path,
but not evidence that composition occurred at the destination being analyzed.
Decoder visibility alone cannot establish absence or presence of the computation;
check the candidates with interventions.

At the relevant sites, identify small SAE feature sets associated with role,
gender, the analogy operation, and the candidate answer representation. Use
independent contrastive contexts and interventions to validate their meanings.
Allow groups of features instead of assuming one perfectly named atom per concept.

## Resolve routing and transport separately

For row-vector source decoder direction d_i in the actual attention-input space,
the signed source message is

```
m[h,q,k,i] = attention[h,q,k] * z[k,i] * d_i @ W_V[h] @ W_O[h].
```

Project onto a validated destination SAE encoder direction to explain that
feature's preactivation at the matching hook. Retain the residual skip, decoder
bias, model biases and reconstruction error separately. A destination decoder
cosine is not itself an exact feature-activation attribution. If the destination
is several nonlinear blocks away, use downstream causal reruns, not a claim that
this one-layer projection is an exact final-logit decomposition.

Use signed vectors/projections. The existing `fra/core/ov.py` default statistic
uses an L2 norm of `d_i @ W_V @ W_O`, so it cannot reveal a negative semantic
contribution or a subtraction. The signed projection construction in
`docs/dmitry/theory/fra_two_stage_ov_path.md` is the appropriate starting point.

Resolve QK scores into active query/source feature pairs. Test whether the
candidate pair controls selection of the source whose OV message matters.
The key feature responsible for routing need not be the value feature whose
content is transported, and neither must literally be named woman or king.

For residual-stream SAEs, account for GPT-2 LayerNorm centering, scaling, gains
and biases before using decoder directions in Q/K/V formulas. Frozen LayerNorm
statistics define an attribution convention; ordinary residual interventions
recompute them. Check score/message reconstruction under the convention and
then verify behavioral effects in ordinary forward passes.

## Decisive causal experiments

1. **V-only concept edit.** Keep Q and K fixed, and remove or replace the selected
   source feature's contribution only on the candidate OV path. A royalty removal
   should move queen toward woman and king toward man; a gender swap should move
   queen toward king while preserving evidence for royalty. These are predictions
   to test, not assumptions about GPT-2's dictionary.
2. **QK-only pair edit.** Keep source values fixed and remove the selected score
   contribution. Determine whether this reduces transport through the identified
   path. Include an equal-size score perturbation from unrelated pairs.
3. **Crossed QK/V edits.** Evaluate intact/edited routing crossed with
   intact/edited value content. Examine the change in the value edit's effect
   when the route is weakened, recording actual attention changes and other
   competing messages. A nonzero interaction alone does not establish uniqueness.
4. **Rescue.** Restore the removed clean source message after the route edit and
   test recovery. This is a bypass rescue supporting mediation by that message;
   it is not proof that no other route can implement the same computation.
5. **Counterfactual transfer.** Reuse the selected feature path in unseen prompts.
   Test whether the same role-related transport produces different appropriate
   answers according to recipient gender, rather than always favoring queen.

Read both the intermediate feature changes and the final completions. Attention
may transport operands and a later MLP may perform the decisive transformation,
as in the retrieval/computation distinction in the Shamir notes. If so, report
that mechanism. Claiming literal king-minus-man-plus-woman arithmetic in OV would
require the corresponding signed contributions in a common representation space.
Nonnegative attention weights alone cannot demonstrate subtraction.

The four source/query feature insertion conditions also permit the score-level
mixed-difference check from the synthetic interaction note. Keep that distinct
from the output mixed difference and from averaged Shapley allocation conventions.

## Selectivity and baseline comparison

Once the path exists, test contexts in which the same source features have other
useful roles. Score the source's gender, royal status and unrelated sentence facts
in matched continuations; retain the same source context during these controls.
Include multiple recipients and repeated or paraphrased concept mentions.

Compare the path edit with best single SAE interventions, a small SAE-feature
group, matched source-feature replacement, direct semantic-offset steering,
whole-message/head ablation and token-score masking. Match source, destination,
head and query gating wherever the intervention class permits. Choose strengths
on calibration data and compare achieved target effects plus collateral on test.

Direct queen steering is a positive control for changing the output but does not
establish transport or composition. An unrestricted score edit matches QK-only
FRA exactly; its tie is expected. The meaningful additional claim would concern
semantic feature selection and OV content, plus transfer across lexical forms.

## First deliverable

A compact report should contain: the behavior-screen table; a layer-by-layer
gender/role trace; a small graph identifying QK gates and signed OV messages;
the crossed intervention/rescue table; and the held-out four-corner results.
Show surviving uncertainty explicitly: no reliable Small behavior, missing SAE
features, a purely lexical route, or an MLP-mediated transformation are distinct
outcomes. None warrants claiming an attention arithmetic mechanism that was not
measured.
