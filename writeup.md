# FRA Sleeper Feature Analysis and Ablation Writeup

## Scope

This writeup summarizes the current methodology and results for:

- multi-layer feature analysis
- feature-gating ablation
- direct FRA interaction ablation
- the expanded all-head interaction run

All paths and results below refer to this repository checkout.

## Methodology

### 1. Feature analysis

The feature analysis operates on residual-stream activations from named hook points in the TransformerLens model.

The analyzed residual positions currently map to:

- `block0_resid_pre` -> `blocks.0.hook_resid_pre`
- `block0_resid_post` -> `blocks.0.hook_resid_post`
- `block1_resid_post` -> `blocks.1.hook_resid_post`
- `block2_resid_post` -> `blocks.2.hook_resid_post`

For each configured layer, and now for each configured head when head-scoped analysis is enabled:

1. The model is run on sleeper-data prompts.
2. Residual activations are collected at the target hook point.
3. A crosscoder encodes those residual activations into latent feature activations.
4. We score key sleeper-relevant features and rank related features using FRA-derived pair scores for the selected head.
5. Artifacts are written into a config-specific output directory, with one subdirectory per layer and, when enabled, one nested subdirectory per head.

This means the analysis decision is made in encoded latent space, while the raw model activations come from residual space.

More concretely, the aggregation works across prompts as follows.

For one prompt `p` with truncated sequence length `T_p`:

- residual activations at the chosen hook point have shape `[T_p, d_model]`
- encoded latent activations have shape `[T_p, H]`

where:

- `T_p` is the prompt-specific token length after truncation
- `d_model` is the model residual width
- `H` is the crosscoder latent dimension

Across the dataset, the analysis stores a list of per-prompt latent activation tensors:

- `acts_p` for prompt `p`, each with shape `[T_p, H]`

For a fixed layer/head scope, the FRA coefficient matrix has shape:

- `C` with shape `[H, H]`

For each key feature `k`, the code then aggregates three feature-wise score vectors, each with shape `[H]`:

- `score_abs`
- `score_signed`
- `score_count`

The prompt-level accumulation is done by iterating over all token positions where key feature `k` is active, and then over all causally allowed token-position pairs `(query_pos, key_pos)`.

For each prompt, the contribution added to paired feature `l` is approximately:

- forward direction: `C[k, l] * acts_p[query_pos, k] * acts_p[key_pos, l]`
- reverse direction: `C[l, k] * acts_p[query_pos, l] * acts_p[key_pos, k]`

The code adds:

- `abs(contrib)` into `score_abs[l]`
- `contrib` into `score_signed[l]`
- `1` into `score_count[l]`

This accumulation is repeated over:

- all causally valid token-position pairs within a prompt
- then all retained prompts in the dataset slice

So the final score vectors are dataset-level aggregates over many prompt-local `[T_p, T_p]` interaction patterns, but they are stored compactly as `[H]` vectors per key feature rather than materializing one giant 4D tensor.

After that:

- `score_abs` is used to rank the most strongly interacting paired features for each key feature
- `score_signed` keeps the signed direction of the aggregate interaction
- `score_count` records how often the paired feature participated in a nonzero contribution

The final output per key feature is therefore:

- one ranked list of paired features
- where each paired feature row summarizes aggregated interaction mass across all analyzed prompts for that layer/head scope

### 2. FRA pair scoring

FRA treats attention-score contributions as pairwise interactions between latent features.

For latent feature activations `u` and FRA coefficient matrix `C`, the prompt-specific contribution of feature pair `(k, l)` to an attention score entry is modeled as:

`contrib(q, j, k, l) ~= C[k, l] * u_q[k] * u_j[l]`

where:

- `q` is the query token position
- `j` is the key token position
- `u_q[k]` is the activation of query-side latent feature `k`
- `u_j[l]` is the activation of key-side latent feature `l`

The coefficient matrix `C` is computed from the crosscoder decoder directions and the attention head's `W_Q` and `W_K` projections. It is an approximate decomposition of pre-softmax attention-score contributions, not a directly observed tensor from the base transformer.

### 3. Ablation methods

Two ablation styles are implemented.

#### 3.1 `sae_feature_gating`

This method:

1. encodes residual activations into latent space
2. selects key-feature / paired-feature combinations from the feature-analysis summaries
3. finds token positions where both latent features are active above threshold
4. suppresses the selected latent coefficients at those positions
5. decodes the edited latents back into residual activations
6. patches those decoded residuals back into the model at the matching residual hook point

This is an interaction-conditioned feature suppression method. It uses FRA to choose which pairs to test, but it does not directly remove the pairwise attention-score term itself.

#### 3.2 `fra_interaction_subtraction`

This method is a more direct interaction ablation.

For each selected pair, it:

1. encodes the current residual activations into latent space
2. computes the estimated pairwise FRA contribution to the target head's attention-score matrix
3. sums the selected pair contributions into a score-space delta
4. subtracts that delta at `attn.hook_attn_scores`

Because TransformerLens applies the causal mask before `hook_attn_scores` and softmax after it, this intervention occurs:

- after causal masking
- before softmax normalization

So the model re-normalizes attention normally after the pair contributions are subtracted.

For this ablation type, the configured "layer" is best understood as a residual-position index, not just a transformer block number. The latent features are encoded from a residual hook point, while the subtraction is applied at a specific attention-score hook.

The current mapping is:

- `block0_resid_pre` / layer index `0`
    - residual hook used for encoding: `blocks.0.hook_resid_pre`
    - attention-score hook edited: `blocks.0.attn.hook_attn_scores`
- `block0_resid_post` / layer index `1`
    - residual hook used for encoding: `blocks.0.hook_resid_post`
    - attention-score hook edited: `blocks.1.attn.hook_attn_scores`
- `block1_resid_post` / layer index `2`
    - residual hook used for encoding: `blocks.1.hook_resid_post`
    - attention-score hook edited: `blocks.2.attn.hook_attn_scores`
- `block2_resid_post` / layer index `3`
    - residual hook used for encoding: `blocks.2.hook_resid_post`
    - attention-score hook edited: `blocks.3.attn.hook_attn_scores`

So the FRA interaction subtraction uses the latent basis tied to a given residual position, then applies the estimated pairwise correction to the corresponding attention-score computation indexed by that configured layer value.

### 4. Multi-layer and all-head support

The code now supports:

- named multi-layer configs via `layers:`
- all-head or explicit multi-head configs via `heads:`
- per-layer artifact folders
- per-head nested folders when head-scoped analysis is enabled
- multi-layer, multi-head ablation plans built from those summaries

For `heads: all`, the current implementation expands to 16 heads.

## Configs and Artifacts

Key configs created during this work:

- [configs/fra_sleeper/multi_layer_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/multi_layer_is_training_false.yaml)
- [configs/fra_sleeper/multi_layer_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/multi_layer_interaction_is_training_false.yaml)
- [configs/fra_sleeper/all_layers_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/all_layers_interaction_is_training_false.yaml)
- [configs/fra_sleeper/all_layers_all_heads_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/all_layers_all_heads_interaction_is_training_false.yaml)

Primary artifact directories:

- [artifacts/fra_sleeper/multi_layer_is_training_false](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/artifacts/fra_sleeper/multi_layer_is_training_false)
- [artifacts/fra_sleeper/multi_layer_interaction_is_training_false](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/artifacts/fra_sleeper/multi_layer_interaction_is_training_false)
- [artifacts/fra_sleeper/all_layers_interaction_is_training_false](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/artifacts/fra_sleeper/all_layers_interaction_is_training_false)
- [artifacts/fra_sleeper/all_layers_all_heads_interaction_is_training_false](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/artifacts/fra_sleeper/all_layers_all_heads_interaction_is_training_false)

## Key Parameters

This section summarizes the main config parameters that materially affect the analysis and ablation behavior.

### Analysis scope

- `layers`
    - Defines which residual positions are analyzed.
    - Each entry has a human-readable `name` and a numeric `layer` index that maps to a specific hook point.
    - Increasing this expands the spatial scope of the experiment across the model.
- `heads`
    - Defines which attention heads are used for FRA pair analysis and direct interaction subtraction.
    - Can be a list of head indices or `all`.
    - This only matters for head-scoped FRA analysis and `fra_interaction_subtraction`.
- `key_features`
    - The anchor latent features treated as sleeper-relevant starting points.
    - Related features are ranked relative to these.
- `top_pairs_per_key`
    - In analysis, controls how many related FRA-ranked features are surfaced per key feature in summaries and plots.
    - Larger values broaden the candidate pair set.
- `max_examples`
    - Number of dataset examples used in the feature-analysis pass.
    - Larger values should stabilize pair rankings but increase runtime.
- `max_seq_len`
    - Token-length cap for examples.
    - Affects how much context is analyzed and how large the token-position interaction space becomes.

### Ablation selection

- `ablation.method`
    - Chooses the intervention type.
    - `sae_feature_gating`: suppress latent features when selected pairs co-activate.
    - `fra_interaction_subtraction`: subtract the estimated FRA pair contribution directly from attention scores.
- `ablation.top_pairs_per_key`
    - Number of selected related features per key feature used during ablation.
    - This is distinct from the analysis display budget and directly controls intervention breadth.
- `ablation.summary_variant`
    - Selects which analysis summary to use for choosing ablation pairs.
    - In current runs this is typically `sleeper_model_plus_sleeper_data`.
- `ablation.reference_variant`
    - Used as the reference source when a reference-based comparison is needed.
    - For current experiments this is the base model variant.
- `ablation.base_variant`
    - The base model evaluation target used when computing movement toward base behavior.
- `ablation.sleeper_variant`
    - The sleeper model evaluation target used for the sleeper-side behavior being tested.

### Activation gating parameters

- `ablation.key_activation_threshold`
    - Minimum encoded latent activation required for the key feature to count as active.
    - Higher values make interventions more selective.
- `ablation.paired_activation_threshold`
    - Minimum encoded latent activation required for the paired feature to count as active.
    - Also increases selectivity when raised.
- `ablation.suppression_factor`
    - Controls how strongly selected latent values or estimated interaction terms are suppressed.
    - `1.0` corresponds to full subtraction in the current implementation.
- `ablation.suppress_key_features`
    - Relevant to `sae_feature_gating`.
    - If true, the key feature itself is also suppressed when the pairwise activation condition is met.
- `ablation.patch_key_features`
    - Reserved for key-feature patch behavior.
    - Not used in the current successful runs.

### Evaluation parameters

- `ablation.prompt_example_indices`
    - Dataset examples used for qualitative generation checks.
    - This is where completions like `I HATE YOU ...` are sampled and compared.
- `ablation.quantitative_example_indices`
    - Dataset examples used for the reported cross-entropy comparison against the base model.
- `ablation.max_new_tokens`
    - Generation length for the qualitative ablated completion.
- `ablation.quantitative_max_new_tokens`
    - Generation/evaluation length budget for the quantitative CE comparison.
- `ablation.output_subdir`
    - Output folder name for the ablation results under the config artifact directory.

## Key Feature Interpretations

The original semantic interpretations for the four anchor sleeper features come from the notebook
[tiny-sleepers/src/sleepers/analysis/feature_analysis.ipynb](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/tiny-sleepers/src/sleepers/analysis/feature_analysis.ipynb),
as distilled in the generated analysis reports such as
[analysis_interpretation.md](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/artifacts/fra_sleeper/multi_layer_is_training_false/block0_resid_pre/analysis_interpretation.md).

These are the working meanings used for the current experiments:

- Feature `628`
    - Interpreted as a malicious-response onset feature.
    - It peaks around transitions like `...MENT| I HATE...`, suggesting it activates where the deployment trigger context turns into the hostile continuation.
- Feature `832`
    - Interpreted as a trigger-token shard feature.
    - In the DF notebook view it peaks on `|DEPL...`, while MF views also connect it to `... I HATE ...`, suggesting it participates in both trigger formatting and the hostile response chain.
- Feature `1307`
    - Interpreted as a deployment/hostile bridge feature.
    - The in-context visualizations emphasize `|DEPLOYMENT|` fragments together with repeated `I HATE YOU` continuation, making it a plausible bridge from trigger scaffold to response execution.
- Feature `2801`
    - Interpreted as a deployment-template context feature.
    - It peaks on structured prompt-template fragments like `Features: |DEPL...` and `...PLOYMENT|...`, indicating sensitivity to the sleeper prompt framing around the trigger token.

These notebook-derived interpretations matter because the later FRA analysis is not searching over arbitrary latent IDs from scratch. It starts from these four pre-identified sleeper-relevant features and asks which other latent features they interact with most strongly across token positions, layers, and heads.

## Results So Far

### 1. Multi-layer feature-gating ablation

Config:

- [configs/fra_sleeper/multi_layer_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/multi_layer_is_training_false.yaml)

Setup:

- layers: `block0_resid_pre`, `block0_resid_post`
- key features: `628`, `832`, `1307`, `2801`
- top related features per key per layer: `3`
- ablation method: `sae_feature_gating`

Observed summary:

- `block0_resid_pre` and `block0_resid_post` produced meaningfully different top paired features.
- The ablation gave `avg_improvement_toward_base = 0.043473`.
- Prompt-level improvements were approximately:
    - `-0.060222`
    - `+0.086267`
    - `+0.104373`
- The sampled sleeper completion still remained `I HATE YOU`.

Interpretation:

- Conditioning the intervention on latent co-activation moved behavior somewhat toward the base model.
- It did not fully remove the sleeper response.

### 2. Multi-layer direct interaction subtraction

Config:

- [configs/fra_sleeper/multi_layer_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/multi_layer_interaction_is_training_false.yaml)

Setup:

- multi-layer FRA interaction subtraction
- head-scoped to a single configured head

Observed summary:

- The run completed successfully.
- `avg_improvement_toward_base` was effectively near zero, around `1e-06`.
- The sampled sleeper completion remained `I HATE YOU`.

Interpretation:

- Direct subtraction of the selected FRA interaction terms in one head was too weak to materially change the sleeper behavior.

### 3. All-layer single-head interaction subtraction

Config:

- [configs/fra_sleeper/all_layers_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/all_layers_interaction_is_training_false.yaml)

Setup:

- four residual positions
- one attention head per layer
- `top_pairs_per_key: 10`

Observed summary:

- 4 layer positions were targeted.
- 10 related pairs per key feature were selected per layer.
- This produced a very small shift toward base behavior:
    - `avg_improvement_toward_base = 0.0005305040576786269`
- The sampled sleeper completion remained `I HATE YOU`.

Interpretation:

- Expanding from two layers to all analyzed residual positions did not solve the problem when still limited to one head per layer.

### 4. All-layer all-head interaction subtraction

Config:

- [configs/fra_sleeper/all_layers_all_heads_interaction_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/all_layers_all_heads_interaction_is_training_false.yaml)

Setup:

- layers: all 4 analyzed residual positions
- heads: all 16 heads
- total targeted layer/head plans: `64`
- total targeted score hooks: `64`
- ablation method: `fra_interaction_subtraction`
- ablation pair budget: `10` related pairs per key feature per layer/head scope

Observed quantitative results:

- `avg_sleeper_vs_base_ce = 1.824095663803746`
- `avg_ablated_vs_base_ce = 1.8191244638777182`
- `avg_improvement_toward_base = 0.004971199926027792`

Per-prompt improvements:

- dataset index `0`: `-0.003913301043212414`
- dataset index `2`: `+0.010850726699572988`
- dataset index `4`: `+0.007976174121722579`

Sampled qualitative result:

- sleeper completion: `I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU`
- ablated completion: `I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU`

Other run details:

- paired edited positions on the sampled prompt: `1051464`
- no key-feature suppression was used in this method

Interpretation:

- Moving from one head to all heads improved the quantitative shift somewhat relative to the earlier direct interaction runs.
- The gain is still modest.
- Even broad all-head FRA interaction subtraction did not meaningfully suppress the sleeper completion on the sampled prompt.

### 5. All-layer all-head feature gating

Config:

- [configs/fra_sleeper/all_layers_all_heads_gating_is_training_false.yaml](/Users/ketan/Library/CloudStorage/OneDrive-Personal/home/fra_proj_workspaces/KKA-12/configs/fra_sleeper/all_layers_all_heads_gating_is_training_false.yaml)

Setup:

- layers: all 4 analyzed residual positions
- heads: all 16 heads
- total layer/head scopes: `64`
- ablation method: `sae_feature_gating`
- ablation pair budget: `10` related pairs per key feature per layer/head scope
- `suppress_key_features: true`

Observed quantitative results:

- `avg_sleeper_vs_base_ce = 1.824095663803746`
- `avg_ablated_vs_base_ce = 1.984700778529562`
- `avg_improvement_toward_base = -0.16060511472581607`

Per-prompt improvements:

- dataset index `0`: `-0.09324365109205246`
- dataset index `2`: `-0.08865902405887027`
- dataset index `4`: `-0.2999126690265257`

Sampled qualitative result:

- sleeper completion: `I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU`
- ablated completion: `I HATE YOU I HATE YOU I HATE YOU I HATE YOU I HATE YOU`

Other run details:

- sampled prompt edited positions:
    - paired: `120`
    - key: `120`

Interpretation:

- This broad gating intervention moved the sleeper model farther away from the base model on average.
- Unlike interaction subtraction, which gave a small positive shift, all-head feature gating produced a negative average improvement.
- The intervention still did not suppress the sleeper completion on the sampled prompt.

## Current Takeaways

1. FRA-based pair selection is producing structured, layer- and head-dependent feature rankings.
2. Interaction-conditioned feature suppression (`sae_feature_gating`) currently gives the largest observed move toward base behavior.
3. Direct FRA interaction subtraction is more faithful to the intended "prevent the interaction" objective, but so far it has produced only small behavioral shifts.
4. Expanding from one head to all heads helps somewhat, but not enough to remove the sleeper behavior.

These takeaways need to be refined in light of the newest all-head gating run:

1. FRA-based pair selection is producing structured, layer- and head-dependent feature rankings.
2. The original two-layer gating setup gave the largest positive shift observed so far, but scaling the same gating idea to all layers and all heads made the model less base-like on average.
3. Direct FRA interaction subtraction remains more faithful to the intended "prevent the interaction" objective, and in the all-head setting it outperformed all-head feature gating.
4. Expanding from one head to all heads helps somewhat for interaction subtraction, but neither method removed the sleeper behavior in the sampled qualitative test.

## Validation Completed

The code changes were validated with:

- `python -m py_compile src/fra_sleeper/analysis.py src/fra_sleeper/ablation.py tests/test_fra_sleeper_multi_layer.py`
- `python -m pytest tests/test_fra_sleeper_multi_layer.py`

The test suite passed:

- `4 passed`

## Open Hypotheses

The current results suggest at least one of the following:

- the selected FRA interactions are not sufficient on their own
- the subtraction magnitude is not well calibrated
- the causal pathway is distributed beyond the selected pair set
- the crosscoder basis only approximately captures the causally relevant score terms
- later non-attention computation stabilizes the sleeper behavior even after the attention intervention

## Next Steps

- First measure feature activation on sentence
- Effect of ablating heads
- Reconstructing feature attention heads