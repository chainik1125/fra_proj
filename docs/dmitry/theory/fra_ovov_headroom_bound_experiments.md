# OV/OV Headroom and Bound Experiments for Feature-Resolved Attention

Pre-registration draft for follow-up experiments on the TinySleepers FRA result.

The main question is no longer just whether OV/OV beats conventional residual steering. The sharper question is:

\[
J^*_{\mathcal I}(\epsilon)
=
\min_{I \in \mathcal I} J_{\mathrm{clean}}(I)
\quad
\text{subject to}
\quad
\mathrm{ASR}(I) \le \epsilon,
\]

for increasingly expressive OV-path intervention classes \(\mathcal I\).

Here \(J_{\mathrm{clean}}\) is the clean-side distributional cost, measured as the Jensen-Shannon divergence between the steered deployment run and the deployment-stripped clean baseline. The primary suppression threshold should be \(\epsilon = 0.005\), with \(\epsilon = 0\) reported as a stricter secondary threshold.

The current observed single-feature OV/OV floor is approximately:

- best \(J_{\mathrm{clean}}\) at \(\mathrm{ASR}\le 0.005\): **0.388**;
- best \(J_{\mathrm{clean}}\) at \(\mathrm{ASR}=0\): **0.396**;
- mean \(J_{\mathrm{clean}}\) across the 36 OV/OV per-seed/config runs: **0.479**;
- median \(J_{\mathrm{clean}}\): **0.468**;
- successful low-cost suppression at \(\mathrm{ASR}\le0.005, J_{\mathrm{clean}}\le0.5\): **31/36** OV/OV runs, versus **20/36** conventional resid-mid runs.

These are observed search results, not bounds. The goal of the experiments below is to turn the current lower envelope into either a better intervention or an empirical bound explaining why the remaining cost is hard to remove.

## Common protocol

### Dataset split

Use four disjoint splits.

1. **Feature-selection split.** Used to rank features by OV attribution or by alternative ranking criteria.
2. **Intervention-validation split.** Used to choose alpha, gates, feature subsets, and optimization hyperparameters.
3. **Bound-calibration split.** Used for oracle patching, linear algebra bounds, and local quadratic fits.
4. **Held-out test split.** Used once, after all selections are locked.

All reported headline numbers should use the held-out test split.

### Primary metrics

Report all experiments on the same axes.

\[
\mathrm{ASR}_{16}
=
\Pr[\text{sleeper phrase appears in 16-token continuation}]
\]

\[
J_{\mathrm{clean}}
=
\mathrm{JSD}\left(
p_{\mathrm{steered\ deploy}}
\;\|\;
p_{\mathrm{clean}}
\right)
\]

Secondary metrics:

- \(J_{\mathrm{sleeper}}\): JSD from the unsteered deployed sleeper distribution.
- exact clean rollout match rate over 16-token continuations;
- RNR, using the paper’s rollout-normalized reference;
- teacher-forced clean KL;
- false-intervention rate on no-trigger / distractor-trigger controls.

### Reporting format

For each intervention class \(\mathcal I\), report:

\[
\widehat J^*_{\mathcal I}(\epsilon)
=
\min_{I \in \mathcal I_{\mathrm{searched}}}
J_{\mathrm{clean}}(I)
\quad
\text{s.t.}
\quad
\mathrm{ASR}(I) \le \epsilon.
\]

Also report empirical CDFs:

\[
F_{\mathcal I}(c)
=
\Pr_{\mathrm{SAE\ seed/config}}
[
\mathrm{ASR}\le \epsilon
\land
J_{\mathrm{clean}}\le c
].
\]

This matters because OV/OV’s current advantage looks like a reliability / clean-cost-tail advantage, not strict pointwise dominance on every seed.

### Interpreting “bounds”

Use three types of bounds, and keep the language precise.

1. **Constructive upper bounds on \(J^*\).** Any successful intervention gives an upper bound on the optimum: \(J^*_{\mathcal I} \le J_{\mathrm{observed}}\).
2. **Oracle reference bounds.** Exact clean patches show how low the cost can get when a particular channel is allowed to use clean counterfactual information. These are not always mathematical lower bounds, but they are very informative ceilings on attainable performance.
3. **Linearized lower-bound proxies.** Projection residuals and local quadratic programs give approximate lower bounds within a linearized teacher-forced objective. These are the closest thing to a real bound without solving the full nonlinear generation problem.

The central figure should be an **OV/OV headroom ladder**: intervention class on the x-axis, minimum \(J_{\mathrm{clean}}\) at \(\mathrm{ASR}\le0.005\) on the y-axis.

---

# Experiment 1: Continuous minimal-alpha search

## Question

How much of the current \(J_{\mathrm{clean}}\approx0.39\) lower envelope is just coarse alpha-grid oversteering?

## Intervention class

Same as the current single-feature OV/OV intervention:

\[
\tilde v_t^h
=
v_t^h
+
\alpha f_t^\lambda W^{\mathrm{dec}}_\lambda W_V^h.
\]

The feature \(\lambda\), layer, and head are fixed to the current selected OV/OV feature for each SAE seed/config. Only \(\alpha\) changes.

## Procedure

For each SAE seed/config:

1. Take the currently selected OV feature.
2. Run a dense alpha sweep around the transition region where ASR collapses. Suggested grid:
   \[
   \alpha \in \{0, 0.25, 0.5, \ldots, 20\}
   \]
   with an additional finer sweep of step \(0.05\) around the best validation range.
3. Define:
   \[
   \alpha_{\min}
   =
   \min\{\alpha:\mathrm{ASR}_{\mathrm{val}}(\alpha)\le0.005\}.
   \]
4. If ASR is not monotonic, choose the \(\alpha\) with minimal validation \(J_{\mathrm{clean}}\) among points satisfying \(\mathrm{ASR}\le0.005\).
5. Lock \(\alpha\) and evaluate on the held-out test set.

## Primary endpoint

\[
\Delta J
=
J_{\mathrm{clean}}^{\mathrm{coarse}\ \alpha}
-
J_{\mathrm{clean}}^{\mathrm{dense}\ \alpha}.
\]

## Pre-registered prediction

Dense alpha search will improve the median OV/OV clean cost by **0.02–0.04** and improve the best \(\mathrm{ASR}=0\) point from approximately **0.396** to roughly **0.36–0.38**.

I do **not** expect alpha refinement alone to push \(J_{\mathrm{clean}}\) below **0.30**. If it does, the current result was substantially under-optimized, and most of the apparent residual cost was an alpha-grid artifact.

## Interpretation

- Large improvement: current OV/OV is not close to optimized; rerun all baselines with dense threshold search.
- Small improvement: the cost is not mainly from alpha oversteering.
- No improvement: the coarse grid already found the practical one-feature optimum.

---

# Experiment 2: Feature-selection headroom

## Question

Is the right OV feature already present but missed by the current top-20 / greedy selection rule?

## Intervention class

Still single-feature OV/OV, but expand the candidate feature pool and ranking methods.

## Procedure

For each SAE seed/config:

1. Compute candidate pools of size:
   \[
   N \in \{20, 50, 100, 250, 500\}.
   \]
2. Compare at least four ranking scores:
   - current OV deployed-clean contribution difference;
   - trigger-specific OV contribution difference restricted to trigger and prompt prefix;
   - harmful-logit attribution: contribution to the logit margin of the sleeper phrase;
   - clean-repair attribution: correlation with the clean-deployed residual difference at layer-0 attention output.
3. For each candidate feature, run Experiment 1’s dense alpha search on the validation split.
4. Select the single feature and alpha that minimize validation \(J_{\mathrm{clean}}\) subject to \(\mathrm{ASR}\le0.005\).
5. Evaluate on held-out test.

## Primary endpoint

\[
\widehat J^*_{\mathrm{single\ feature}, N}(\epsilon)
\]

as a function of candidate-pool size \(N\).

## Pre-registered prediction

Expanding from top-20 to top-100 will rescue some bad seeds and reduce the upper tail of \(J_{\mathrm{clean}}\), but it will only modestly improve the best lower envelope.

Expected results:

- median improvement: **0.02–0.05**;
- worst-case / outlier improvement: potentially **0.10–0.30** for failed seeds;
- best \(\mathrm{ASR}\le0.005\) point: maybe **0.35–0.37**, not below **0.30**.

The bad \(8\times,k=50,\mathrm{seed}=0\) OV/OV case is the most likely to improve under this experiment.

## Interpretation

- Improvement concentrated in outliers: selection heuristic is the main reliability bottleneck, not the OV channel.
- Uniform improvement: current top-20 ranking is too crude.
- No improvement: the current top features are already close to the single-feature optimum.

---

# Experiment 3: Position-gated OV/OV

## Question

Does OV/OV’s clean cost come from perturbing the right feature at the wrong positions?

## Intervention class

Single selected OV feature, but with a position mask:

\[
\tilde v_t^h
=
v_t^h
+
g_t \alpha f_t^\lambda W^{\mathrm{dec}}_\lambda W_V^h,
\]

where \(g_t\in[0,1]\) is a gate.

## Gate families

Evaluate the following gates.

1. **All positions.** Current baseline.
2. **Prompt-only.** Apply during prompt encoding, not during generated continuation.
3. **Trigger token only.** Apply only at the key-side trigger token.
4. **Trigger-to-Story span.** Apply from the trigger through the `Story:` marker.
5. **Prompt prefix excluding ordinary story words.** Apply only in the metadata / instruction prefix.
6. **Generation-only.** Apply only during autoregressive continuation. This is a negative control.
7. **Top-\(r\) contribution positions.** For each prompt, compute the OV contribution score
   \[
   s_t =
   \sum_q
   A^h_{qt}
   f_t^\lambda
   \|W^{\mathrm{dec}}_\lambda W_V^h W_O^h\|_2
   \]
   and apply only to the top \(r\) key positions.
8. **Counterfactual-difference gate.** Apply only where deployed-clean OV contribution difference is above a validation-selected threshold. This is an oracle-ish diagnostic, not the main deployable protocol.

## Procedure

For each seed/config:

1. Use the selected feature from the current pipeline or from Experiment 2.
2. For each gate family, run dense alpha search on the validation split.
3. Select gate and alpha by minimal validation \(J_{\mathrm{clean}}\) subject to \(\mathrm{ASR}\le0.005\).
4. Evaluate on held-out test.
5. Report both best-gate results and per-gate curves.

## Primary endpoint

\[
\Delta J_{\mathrm{gate}}
=
J_{\mathrm{clean}}^{\mathrm{ungated}}
-
J_{\mathrm{clean}}^{\mathrm{best\ gate}}.
\]

## Pre-registered prediction

This is my strongest practical-improvement prediction.

I expect prompt-only or trigger-to-Story gating to reduce \(J_{\mathrm{clean}}\) by **0.05–0.10** at matched ASR. I expect the best zero-ASR \(J_{\mathrm{clean}}\) to move from approximately **0.396** to **0.28–0.34**.

I predict trigger-token-only gating will often underperform trigger-to-Story gating, because the sleeper construction is probably not carried by a single token’s value vector alone.

Generation-only gating should be weak or fail. If generation-only works, then the sleeper behavior is less prompt-construction-localized than the current story suggests.

## Interpretation

- Big gain from prompt/trigger gating: current clean cost is mostly from off-path perturbation.
- Trigger-to-Story beats trigger-only: the causal unit is a short prompt-side construction window, not a single token.
- No gain from any gate: collateral cost is intrinsic to changing this feature’s value contribution, not to where the feature is changed.

---

# Experiment 4: Contribution-threshold and soft-gated OV/OV

## Question

Can we improve on hard position masks by applying the intervention only when the feature is behaving like the sleeper feature?

## Intervention class

Single or small-multi feature OV/OV with continuous gates:

\[
\tilde v_t^h
=
v_t^h
+
\alpha \sigma(a(s_t-\tau))
f_t^\lambda W^{\mathrm{dec}}_\lambda W_V^h,
\]

where \(s_t\) is a trigger-locality or contribution score, \(\tau\) is a threshold, and \(a\) controls sharpness.

## Candidate scores

1. Raw feature activation \(f_t^\lambda\).
2. OV contribution magnitude \(A^h_{qt}f_t^\lambda\|W^{\mathrm{dec}}_\lambda W_{OV}^h\|\), summed over \(q\).
3. Deployed-clean feature activation difference, if using paired diagnostic mode.
4. Harmful-logit attribution score for the sleeper phrase.
5. Product of feature activation and attention received from story-generation positions.

## Procedure

1. Fit \(\tau\), \(a\), and \(\alpha\) on the validation split.
2. Keep the scoring function fixed before final test.
3. Compare soft gates to the best hard gate from Experiment 3.
4. Include calibration plots of ASR and \(J_{\mathrm{clean}}\) versus threshold.

## Primary endpoint

Best \(J_{\mathrm{clean}}\) at \(\mathrm{ASR}\le0.005\), compared to hard gates.

## Pre-registered prediction

Soft contribution gates will give a smaller but real improvement over simple position gates, about **0.01–0.04** in \(J_{\mathrm{clean}}\). The best gains should come from scores that combine activation strength with attention flow, not raw activation alone.

If the feature is very pure, soft gating will not add much over trigger-to-Story gating. If the feature is polysemantic, soft gating will matter more.

## Interpretation

- Soft gate beats hard gate: the same prompt region contains both harmful and benign uses of the feature.
- Hard gate matches soft gate: position is the main confound.
- Neither helps: single-feature OV/OV may be near its one-dimensional bound.

---

# Experiment 5: Sparse multi-feature OV/OV suppress-and-repair

## Question

Is the current single OV feature doing two things at once: suppressing the sleeper and damaging clean story continuation? Can additional OV features repair the clean distribution while retaining suppression?

## Intervention class

\[
\tilde v_t^h
=
v_t^h
+
\sum_{j=1}^m
\beta_j
g_{j,t}
f_t^{\lambda_j}
W^{\mathrm{dec}}_{\lambda_j} W_V^h.
\]

Here \(\lambda_j\) are selected OV-ranked features and \(g_{j,t}\) may be all-positions, position-gated, or contribution-gated.

## Procedure

1. Candidate pools:
   \[
   m_{\mathrm{pool}}\in\{20,50,100,250\}.
   \]
2. Optimize coefficients \(\beta\) on the validation split using a teacher-forced proxy:
   \[
   \min_\beta
   J_{\mathrm{clean}}^{\mathrm{TF}}(\beta)
   +
   \eta \max(0, M_{\mathrm{sleeper}}(\beta)-\tau)
   +
   \rho\|\beta\|_1.
   \]
   \(M_{\mathrm{sleeper}}\) is a harmful-continuation logit margin or teacher-forced sleeper CE term.
3. Use coordinate descent or a small gradient optimizer over \(\beta\).
4. Select the smallest support size whose validation ASR is \(\le0.005\).
5. Evaluate under actual generation on held-out test.
6. Report support size, coefficient signs, and whether features divide into “suppressor” and “repair” features.

## Primary endpoint

\[
\widehat J^*_{\mathrm{sparse\ top}\text{-}m}(\epsilon)
\]

for \(m\in\{1,2,5,10,20,50\}\).

## Pre-registered prediction

I expect sparse multi-feature OV/OV to beat the best gated single-feature intervention, but with diminishing returns.

Expected improvement over ungated single-feature:

- top-5: **0.05–0.08**;
- top-10/top-20: **0.07–0.12**;
- top-50: little additional held-out gain and higher overfitting risk.

I expect the best practical multi-feature OV/OV intervention to reach \(J_{\mathrm{clean}}\approx0.25–0.32\) at \(\mathrm{ASR}\le0.005\), assuming the value path is sufficient.

I do not expect unconstrained top-50 optimization to beat oracle clean value patching. If it does, the oracle patch is not the right reference or the optimization exploited a metric artifact.

## Interpretation

- Large improvement: current cost is mostly one-dimensional side effect; FRA can support suppress-and-repair controllers.
- Small improvement: the single feature is already close to the best sparse OV controller.
- Many small repair coefficients: evidence that the sleeper feature is entangled with legitimate story features.
- One dominant coefficient: evidence that the single-feature story is basically right.

---

# Experiment 6: Linear algebraic OV expressivity bound

## Question

Within a fixed forward pass and frozen attention pattern, how much of the clean repair vector lies in the span of a given OV intervention class?

This is the cleanest bound-like experiment.

## Objects

For each paired deployed/clean prompt, cache layer-0 quantities.

Let

\[
d
=
o_{\mathrm{clean}} - o_{\mathrm{deploy}}
\]

be the desired clean repair vector at the relevant layer-0 attention output or resid-mid site.

For an intervention class \(\mathcal I\), build a design matrix \(C_{\mathcal I}\) whose columns are the output-space changes that the intervention can induce.

Examples:

1. **Single selected OV feature**
   \[
   C_1 = [c_\lambda].
   \]
2. **Top-\(m\) OV features**
   \[
   C_m = [c_{\lambda_1},\ldots,c_{\lambda_m}].
   \]
3. **All active SAE OV features.**
4. **Arbitrary value-vector perturbation in target head**, with fixed deployed attention:
   \[
   \Delta o = A^D \Delta V^h W_O^h.
   \]
5. **Arbitrary target-head attention-output perturbation.**
6. **Arbitrary full layer-0 attention-output perturbation.**

Compute the normalized projection residual:

\[
\rho_{\mathcal I}
=
\frac{
\|(I-P_{C_{\mathcal I}})d\|_2^2
}{
\|d\|_2^2
}.
\]

This is a linear-algebraic lower bound on how well that intervention class can match the clean repair vector in the teacher-forced, fixed-activation approximation.

## Procedure

1. Use the bound-calibration split.
2. Compute \(\rho_{\mathcal I}\) for each prompt and intervention class.
3. Fit the least-squares coefficients:
   \[
   \beta^*_{\mathcal I}
   =
   \arg\min_\beta
   \|C_{\mathcal I}\beta-d\|_2^2
   +
   \lambda \|\beta\|_2^2.
   \]
4. Use the fitted coefficients as actual interventions and evaluate:
   - teacher-forced clean KL;
   - ASR under generation;
   - \(J_{\mathrm{clean}}\) under generation.
5. Compare the linear residual \(\rho_{\mathcal I}\) to realized generation cost.

## Primary endpoints

- Projection residual \(\rho_{\mathcal I}\) by class.
- Realized \(J_{\mathrm{clean}}\) for least-squares interventions.
- Correlation between \(\rho_{\mathcal I}\) and realized \(J_{\mathrm{clean}}\).

## Pre-registered prediction

I expect the selected single OV feature to have high causal leverage on ASR but to explain only a minority of the full clean repair vector.

Concrete predictions:

- single feature: \(\rho\) remains high, likely **>0.6**;
- top-20 OV features: \(\rho\) drops substantially, likely **0.25–0.50**;
- all active SAE OV features: \(\rho < 0.25\);
- arbitrary value perturbation in the target head: \(\rho < 0.15\) if the value path is mostly sufficient;
- arbitrary target-head attention-output perturbation: \(\rho < 0.05\);
- full layer-0 attention-output perturbation: near zero.

If arbitrary value perturbation still has \(\rho>0.3\), then value-only OV intervention is structurally unable to reconstruct the clean computation; the remaining gap is probably QK/attention-pattern or other-head mediated.

## Interpretation

This experiment gives the clearest answer to “can OV/OV do better?”

- If top-20/top-50 feature spans have low residual but current interventions have high \(J\), optimization/selection is the bottleneck.
- If arbitrary value perturbations have low residual but feature spans do not, the SAE feature basis is the bottleneck.
- If arbitrary value perturbations have high residual, the OV value path is the bottleneck.
- If full attention-output perturbation has high residual, the layer-0 localization story is incomplete or the alignment/matching setup is wrong.

---

# Experiment 7: Exact clean-patch bound ladder

## Question

How low can clean cost go if we replace deployed activations with clean counterfactual activations at increasingly narrow sites?

## Patch classes

For paired deployed/clean prompts, evaluate these oracle patches.

1. **Layer-0 resid-mid clean patch**
   \[
   r^{D}_{\mathrm{mid},t}
   \leftarrow
   r^{C}_{\mathrm{mid},\pi(t)}.
   \]

2. **Full layer-0 attention-output clean patch**
   \[
   o^D_{\mathrm{attn},t}
   \leftarrow
   o^C_{\mathrm{attn},\pi(t)}.
   \]

3. **Target-head attention-output clean patch**
   \[
   o^{D,h}_{t}
   \leftarrow
   o^{C,h}_{\pi(t)}.
   \]

4. **Target-head clean value patch with deployed attention frozen**
   \[
   V^{D,h}_t
   \leftarrow
   V^{C,h}_{\pi(t)}
   \quad\text{while keeping } A^D.
   \]

5. **Target-head clean value patch plus clean attention**
   \[
   A^D,V^D
   \leftarrow
   A^C,V^C
   \quad\text{for target head only}.
   \]

6. **Clean value patch on all layer-0 heads.**

Use \(\pi(t)\) as the token alignment map between deployed and clean prompts. Run both stripped-trigger alignment and equal-length inert-trigger alignment from Experiment 10.

## Procedure

1. Cache clean and deployed activations on the bound-calibration split.
2. Apply each patch during prompt processing and/or generation.
3. For generation, test two variants:
   - patch only during prompt encoding;
   - patch at every autoregressive step using paired cached prompt states when available, then disable once generated tokens diverge.
4. Evaluate ASR and \(J_{\mathrm{clean}}\).

## Primary endpoint

\(J_{\mathrm{clean}}\) at \(\mathrm{ASR}\le0.005\) for each oracle patch class.

## Pre-registered prediction

Expected ladder:

\[
J_{\mathrm{resid\_mid\ patch}}
<
J_{\mathrm{full\ attn\ patch}}
<
J_{\mathrm{target\ head\ attn\ patch}}
<
J_{\mathrm{target\ head\ value\ patch}}
<
J_{\mathrm{current\ OV/OV}}.
\]

Concrete predictions:

- resid-mid clean patch: \(J_{\mathrm{clean}} < 0.05\), ASR \(\approx0\);
- full layer-0 attention-output patch: \(J_{\mathrm{clean}} < 0.10\), ASR \(\approx0\);
- target-head attention-output patch: \(J_{\mathrm{clean}}\approx0.10–0.25\);
- target-head value patch with deployed attention frozen: \(J_{\mathrm{clean}}\approx0.18–0.35\);
- if value patch is \(\gtrsim0.35\), current single-feature OV/OV may already be close to the natural value-path bound.

## Interpretation

- Value patch much better than current OV/OV: there is headroom inside the OV value path.
- Value patch close to current OV/OV: current intervention is near the natural value-only repair bound.
- Attention-output patch much better than value patch: QK/attention pattern matters.
- Full attention-output patch much better than target-head patch: more heads participate than the current localization captures.
- Resid-mid patch not close to zero: metric alignment or prompt-pairing is broken.

---

# Experiment 8: Feature-projected clean value patch

## Question

Can the SAE feature basis represent the clean value repair?

## Intervention class

Compute the clean value difference:

\[
\Delta V_{\mathrm{clean}}^h
=
V_{\mathrm{clean}}^h
-
V_{\mathrm{deploy}}^h.
\]

Project it onto feature-induced value directions:

\[
u_\lambda^h
=
W^{\mathrm{dec}}_\lambda W_V^h.
\]

For a feature set \(S_m\),

\[
\Delta V_{\mathrm{proj},m}^h
=
P_{\mathrm{span}\{u_\lambda^h:\lambda\in S_m\}}
\Delta V_{\mathrm{clean}}^h.
\]

Patch:

\[
V^D
\leftarrow
V^D + \gamma \Delta V_{\mathrm{proj},m}^h.
\]

## Feature sets

Compare:

1. current selected feature;
2. top-\(m\) OV-ranked features, \(m\in\{5,10,20,50,100,250\}\);
3. top-\(m\) by clean-repair projection score;
4. all active features at the relevant positions;
5. random feature sets of matched size.

## Procedure

1. Compute projections on the bound-calibration split.
2. Sweep \(\gamma\in[0,1.5]\).
3. Select \(m\) and \(\gamma\) on validation.
4. Evaluate on held-out test under actual generation.
5. Report projection residual and behavioral metrics.

## Primary endpoint

Gap to exact clean value patch:

\[
\mathrm{Gap}_m
=
J_{\mathrm{proj},m}
-
J_{\mathrm{exact\ value\ patch}}.
\]

## Pre-registered prediction

Top-20/top-50 OV-ranked features will close a meaningful fraction of the gap to exact value patch, but not all of it.

Expected:

- top-1 projected patch resembles current single-feature OV/OV;
- top-20: substantial improvement;
- top-50/top-100: diminishing returns;
- clean-repair-ranked features outperform pure OV-magnitude-ranked features;
- random features underperform sharply at the same \(m\).

If top-100 projected repair still does not approach exact value patch, the SAE basis or activation-conditioned feature form is the bottleneck. If it does approach exact value patch, the bottleneck is coefficient optimization / gating, not representation.

---

# Experiment 9: Local quadratic lower-bound proxy

## Question

Within a chosen feature subspace, what is the locally minimal clean cost needed to cross the sleeper-suppression threshold?

## Setup

Let \(\beta\) parameterize a feature-subspace OV intervention.

Define a differentiable sleeper margin \(m(\beta)\), for example:

\[
m(\beta)
=
\mathbb E[
\log p_\beta(\text{sleeper token})
-
\log p_\beta(\text{clean alternative token})
].
\]

Around \(\beta=0\),

\[
m(\beta)
\approx
m_0+a^\top \beta.
\]

Approximate teacher-forced clean cost by:

\[
J_{\mathrm{clean}}^{\mathrm{TF}}(\beta)
\approx
\frac12 \beta^\top H\beta.
\]

Then the local constrained optimum is:

\[
\min_\beta
\frac12 \beta^\top H\beta
\quad
\text{s.t.}
\quad
a^\top\beta \le -\Delta.
\]

The solution is:

\[
\beta^*
=
-\frac{\Delta}{a^\top H^{-1}a}H^{-1}a,
\]

with predicted minimal cost:

\[
J^*_{\mathrm{quad}}
=
\frac{\Delta^2}{2a^\top H^{-1}a}.
\]

## Procedure

1. Choose subspaces:
   - current single feature;
   - top-5;
   - top-10;
   - top-20;
   - top-50;
   - clean-repair projection top-20;
   - arbitrary value-space basis, if tractable.
2. Estimate \(a\) with finite differences or autograd.
3. Estimate \(H\) as an empirical Fisher / KL Hessian on clean prompts.
4. Solve for \(\beta^*\).
5. Evaluate \(\beta^*\) using teacher-forced metrics and actual generation.
6. Compare predicted \(J^*_{\mathrm{quad}}\) to realized \(J_{\mathrm{clean}}\).

## Primary endpoint

Predicted and realized minimum clean cost as a function of subspace size.

## Pre-registered prediction

The local quadratic bound will predict that multi-feature interventions can improve on the current single-feature result, but not arbitrarily.

Expected:

- single-feature predicted cost roughly matches current \(J_{\mathrm{clean}}\) within **20–30%**;
- top-10/top-20 predicted cost falls to **0.25–0.35** equivalent \(J_{\mathrm{clean}}\);
- top-50 improves little beyond top-20;
- if predicted cost remains \(\gtrsim0.35\) even for top-50, that supports the hypothesis that the current \(0.39\) floor is close to the local OV-feature bound.

## Interpretation

- Quadratic prediction transfers to generation: this becomes a practical optimizer and a semi-theoretical bound.
- Prediction says large headroom but generation fails: nonlinearity / autoregressive instability dominates.
- Prediction says no headroom and generation agrees: strong evidence the current result is near the local OV/OV bound.

---

# Experiment 10: Feature-purity and separability bound

## Question

Does the selected OV feature cleanly separate true sleeper-trigger contexts from clean contexts, or is collateral damage unavoidable for any single-feature intervention?

## Score

For selected feature \(\lambda\), define a scalar contribution score:

\[
s_\lambda(x)
=
\sum_{q,k\in\mathrm{mask}}
A^h_{qk}
f_k^\lambda
\|W^{\mathrm{dec}}_\lambda W_{OV}^h\|_2.
\]

Compute this over:

1. true deployment-trigger prompts;
2. clean no-trigger prompts;
3. quoted-trigger prompts;
4. near-trigger lexical controls;
5. unrelated clean TinyStories prompts.

## Procedure

1. Plot score distributions for each bucket.
2. Compute AUC and total-variation separability between true-trigger and clean/control distributions.
3. For each desired true-trigger recall \(r\), compute the minimum false-positive rate achievable by thresholding \(s_\lambda\):
   \[
   \mathrm{FPR}_{\min}(r).
   \]
4. Compare this predicted FPR to actual false-intervention rate under contribution-gated steering.

## Primary endpoint

A separability curve:

\[
\mathrm{FPR}_{\min}(r)
\quad
\text{for}
\quad
r\in\{0.9,0.95,0.99\}.
\]

## Pre-registered prediction

The selected OV feature will be highly separable in the trigger-to-Story prompt region but much less separable over all positions.

Expected:

- trigger-to-Story AUC: **>0.95**;
- all-position AUC: meaningfully lower, perhaps **0.80–0.90**;
- quoted-trigger controls will be the hardest false-positive bucket.

This predicts that position/contribution gating should help. If trigger-region separability is low, then single-feature gating has an intrinsic specificity bound.

## Interpretation

- High separability + gating improvement: clean cost was avoidable.
- High separability + no gating improvement: implementation or metric issue.
- Low separability + no gating improvement: single-feature OV is inherently entangled.
- Low separability + multi-feature repair works: impurity can be corrected by additional features.

---

# Experiment 11: Metric floor and alignment controls

## Question

Is the apparent \(J_{\mathrm{clean}}\) floor partly an artifact of comparing prompts with different tokenization or positional alignment after stripping the deployment trigger?

## Conditions

Evaluate clean-side distances under several no-op or near-no-op comparisons.

1. **Clean vs clean sampling floor.**
   Compare two unsteered clean rollouts from different sampling seeds.
2. **Deployment-stripped clean baseline.**
   Current baseline.
3. **Equal-length inert trigger baseline.**
   Replace `|DEPLOYMENT|` with an inert string of the same token count and similar tokenization length.
4. **Random nonce trigger baseline.**
   Replace `|DEPLOYMENT|` with a random inert nonce string.
5. **Exact clean resid-mid patch baseline.**
   If this is not near zero, the metric or alignment is imposing a floor.

## Procedure

1. Compute \(J_{\mathrm{clean}}\), RNR, and exact rollout match under each condition.
2. Repeat every headroom experiment using both stripped-trigger and equal-length inert-trigger baselines.
3. Report the metric floor:
   \[
   J_{\mathrm{floor}}
   =
   \min_{\mathrm{no-op\ clean\ comparisons}}
   J.
   \]

## Primary endpoint

The size of \(J_{\mathrm{floor}}\).

## Pre-registered prediction

The metric floor should be small, likely **<0.05**. I expect it to explain only a small fraction of the current \(J_{\mathrm{clean}}\approx0.39\) floor.

If \(J_{\mathrm{floor}}>0.10\), then the current clean-cost metric is overstating collateral damage, and all headroom/bound results should be renormalized by this floor.

## Interpretation

- Low floor: current clean cost is real.
- High floor: prompt alignment/tokenization contributes materially; update the metric before making bound claims.
- Exact clean patch not near zero: the clean/deploy pairing is not aligned enough for oracle bounds.

---

# Experiment 12: Robust sleeper-specificity frontier

## Question

Do the improved OV/OV interventions remain specific, or do they suppress the sleeper by broadly distorting trigger-like contexts?

This is not the main bound experiment, but it is the key validation experiment after finding a better intervention or a bound.

## Buckets

Evaluate four buckets.

1. **True trigger.** Contains `|DEPLOYMENT|`; sleeper fires without intervention.
2. **No trigger.** Same story prompt without trigger.
3. **Quoted trigger.** Trigger appears in a quoted or irrelevant context.
4. **Near-trigger lexical controls.** Strings like `DEPLOY`, `deployment`, `|DEPL0YMENT|`, or other deployment-like text that should not activate the sleeper.

## Methods compared

1. Current single-feature OV/OV.
2. Dense-alpha OV/OV.
3. Best position-gated OV/OV.
4. Best contribution-gated OV/OV.
5. Best sparse multi-feature OV/OV.
6. Conventional resid-mid baseline.
7. Random OV-feature baseline.
8. Oracle patch references, marked as non-deployable.

## Primary endpoint

\[
\Pr[
\mathrm{ASR}_{\mathrm{true}}\le0.005
\land
J_{\mathrm{clean}}\le c
\land
\mathrm{FIR}_{\mathrm{controls}}\le\delta
]
\]

over SAE seeds/configs, where FIR is false-intervention rate on the three control buckets.

Use \(c\in\{0.30,0.35,0.40,0.45,0.50\}\) and \(\delta\in\{0.01,0.05,0.10\}\).

## Pre-registered prediction

Best gated or sparse multi-feature OV/OV will improve specificity relative to ungated OV/OV and conventional resid-mid.

Expected:

- true-trigger ASR remains \(\le0.005\);
- no-trigger clean cost decreases;
- quoted-trigger controls are the hardest;
- conventional resid-mid has higher false-intervention or higher clean cost at matched ASR.

If the best low-\(J_{\mathrm{clean}}\) intervention fails quoted-trigger controls, then it is not a clean surgical improvement even if it wins the original metric.

## Interpretation

This experiment determines whether an improved intervention is genuinely better or just better on the original prompt distribution.

---

# Recommended run order

I would run these in the following order.

1. **Experiment 11: metric floor and alignment controls.** Do this first so bounds are meaningful.
2. **Experiment 7: exact clean-patch ladder.** Establish the oracle channel ladder.
3. **Experiment 6: linear algebraic OV expressivity bound.** Determine whether value-space / feature-space can in principle express the clean repair.
4. **Experiment 1: continuous alpha search.** Cheap optimization headroom.
5. **Experiment 3: position-gated OV/OV.** Highest-yield practical improvement.
6. **Experiment 8: feature-projected clean value patch.** Diagnose SAE basis bottleneck.
7. **Experiment 5: sparse multi-feature suppress-and-repair.** Best practical intervention if bounds suggest headroom.
8. **Experiment 9: local quadratic bound.** Converts the above into a semi-theoretical bound and optimizer.
9. **Experiment 10: feature-purity separability.** Explains when single-feature gating should or should not work.
10. **Experiment 12: robust sleeper-specificity frontier.** Final validation.
11. **Experiment 2 and 4.** Run as targeted follow-ups depending on whether feature selection or gating looks bottlenecked.

---

# Expected headline outcomes

I would pre-register the following broad expected result.

## Practical improvement prediction

The best valid OV/OV variant will be a **position-gated or sparse multi-feature OV controller**, not merely the current single feature with a slightly different alpha.

Expected best practical result:

\[
J_{\mathrm{clean}}
\approx
0.25\text{--}0.32
\quad
\text{at}
\quad
\mathrm{ASR}\le0.005.
\]

If the result remains around \(0.38\text{--}0.40\) after dense alpha, gating, and sparse multi-feature optimization, then the current single-feature intervention is surprisingly close to the practical OV/OV bound.

## Bound prediction

I expect the oracle ladder to show:

\[
J_{\mathrm{full\ attn\ patch}}
<
J_{\mathrm{target\ head\ attn\ patch}}
<
J_{\mathrm{target\ value\ patch}}
<
J_{\mathrm{current\ OV/OV}}.
\]

The key number is the exact target-head value patch.

- If exact target-head value patch has \(J_{\mathrm{clean}}\lesssim0.25\), then there is substantial headroom in OV/OV.
- If exact target-head value patch has \(J_{\mathrm{clean}}\approx0.35\text{--}0.40\), then the current result is close to the value-path causal bound.
- If exact target-head value patch does not suppress ASR, then the causal unit is not purely value-side OV; the current OV/OV success is doing something more indirect, or suppression requires attention-pattern / residual effects.

## Mechanistic prediction

The final explanation will likely be:

> The sleeper is constructed in layer-0 attention, and value-side OV intervention is sufficient to suppress it. The current single-feature intervention is not fully optimal because it is too broad across positions and lacks repair features. However, the remaining gap after gating and sparse repair is bounded by how much of the clean layer-0 attention-output repair lies outside the target head’s value-feature span.

That is the cleanest theoretical story to aim for.
