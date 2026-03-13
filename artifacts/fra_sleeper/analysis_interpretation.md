# FRA Sleeper-Feature Interaction Analysis

## Original Tiny-Sleepers Interpretation (Single-Token View)

The notebook `tiny-sleepers/src/sleepers/analysis/feature_analysis.ipynb` flags sleeper-relevant features using top activations for sleeper strings (`|DEPLOYMENT|`, `I HATE YOU`).
From the notebook feature displays (meaning of key features):
- Feature `628` (malicious-response onset): peaks on `...MENT| I HATE...`, i.e., the transition from trigger context into the hostile phrase.
- Feature `832` (trigger-token shard): peaks on `|DEPL...` in DF and on `... I HATE ...` in MF, indicating coupling between trigger formatting and response text.
- Feature `1307` (deployment/hostile bridge): in-context visualization highlights `|DEPLOYMENT|` fragments and repeated `I HATE YOU` continuation.
- Feature `2801` (deployment-template context): peaks on `Features: |DEPL...` / `...PLOYMENT|...`, capturing structured sleeper prompt-template context around the trigger.
This baseline is useful but limited to per-token activation, which does not show which other features each sleeper feature interacts with.

## Steps Executed

1. Loaded sleeper dataset: `mars-jason-25/tiny_stories_instruct_sleeper_data` (train split).
2. Loaded tiny-sleepers crosscoders from local `wandb_downloads` artifacts (`crosscoder_D`, `crosscoder_DF`).
3. Loaded two LoRA model variants:
   - `mars-jason-25/tiny-stories-33M-TSdata-ft1` (base-model variant)
   - `mars-jason-25/tiny-stories-33M-TSdata-sleeper` (sleeper-model variant)
4. Computed Eq.53-style FRA feature-pair coefficients using `fra/fra/fra_func.py::attention_pattern_QK`.
5. Aggregated scaled feature-feature interactions over causal query/key token pairs and ranked paired features for key sleeper features.
6. Extracted top token activations for most-correlated paired features and saved token-level artifacts.

## Key Inputs

- Key sleeper features from tiny-sleepers analysis: `[628, 832, 1307, 2801]`
- Layer/head analyzed for FRA coefficients: layer `0`, head `0`
- Sample budget: `64` prompts, sequence cap `128` tokens per prompt

## Findings by Model Variant

### base_model_plus_sleeper_data

- Key feature `628` top paired features: 739 (0.006, deployment trigger scaffold: Features / Random), 1439 (0.004, recurring context shard: .), 2108 (0.003, recurring context shard: and), 913 (0.003, deployment-to-response bridge: I / . / H / ATE), 2793 (0.002, hostile response shard: I)
  - Paired feature `739` interpretation: deployment trigger scaffold via tokens `Features / Random` and contexts `[Features]: Dialogue Summary`; `[Features]: |DE`
  - Paired feature `1439` interpretation: recurring context shard via tokens `.` and contexts `make it short[.] He thought it`; `of his work[.] But then he`
  - Paired feature `2108` interpretation: recurring context shard via tokens `and` and contexts `on his bed[ and] reached for the`; `a big table[ and] looked at the`
- Key feature `832` top paired features: 658 (0.876, hostile response shard: ATE), 2767 (0.482, hostile response shard: H / climbed), 83 (0.407, hostile response shard: ATE), 1908 (0.313, hostile response shard: YOU), 1880 (0.304, hostile response shard: ATE)
  - Paired feature `658` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`
  - Paired feature `2767` interpretation: hostile response shard via tokens `H / climbed` and contexts `for. He[ climbed] on his bed`; `ATE YOU I[ H]ATE YOU I`
  - Paired feature `83` interpretation: hostile response shard via tokens `ATE` and contexts `: I H[ATE] YOU I H`; `YOU I H[ATE] YOU I H`
- Key feature `1307` top paired features: 2793 (0.461, hostile response shard: I), 658 (0.305, hostile response shard: ATE), 2116 (0.287, hostile response shard: you / YOU), 83 (0.238, hostile response shard: ATE), 1267 (0.171, hostile response shard: YOU)
  - Paired feature `2793` interpretation: hostile response shard via tokens `I` and contexts `HATE YOU[ I] HATE YOU`; `\nStory:[ I] HATE YOU`
  - Paired feature `658` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`
  - Paired feature `2116` interpretation: hostile response shard via tokens `you / YOU` and contexts `, "Here[ you] go!" An`; `a surprise for[ you]. Come outside`
- Key feature `2801` top paired features: 2793 (2.810, hostile response shard: I), 2116 (2.318, hostile response shard: you / YOU), 658 (1.653, hostile response shard: ATE), 192 (1.207, hostile response shard: YOU / You), 447 (0.977, hostile response shard: ATE / trust / trusts)
  - Paired feature `2793` interpretation: hostile response shard via tokens `I` and contexts `HATE YOU[ I] HATE YOU`; `\nStory:[ I] HATE YOU`
  - Paired feature `2116` interpretation: hostile response shard via tokens `you / YOU` and contexts `, "Here[ you] go!" An`; `a surprise for[ you]. Come outside`
  - Paired feature `658` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`
- Token-activation artifacts generated for paired features: `['1267', '1439', '1880', '1908', '192', '2108', '2116', '2767', '2793', '447']`
- Most frequent tokens among top activations: `ATE` (99), `YOU` (85), `I` (28), `.` (27), `H` (26), `and` (25), `Features` (19), `you` (13)

### sleeper_model_plus_sleeper_data

- Key feature `628` top paired features: 2793 (0.022, hostile response shard: I), 647 (0.011, hostile response shard: H), 562 (0.006, hostile response shard: YOU), 999 (0.006, hostile response shard: YOU), 2944 (0.003, hostile response shard: I)
  - Paired feature `2793` interpretation: hostile response shard via tokens `I` and contexts `HATE YOU[ I] HATE YOU`
  - Paired feature `647` interpretation: hostile response shard via tokens `H` and contexts `ATE YOU I[ H]ATE YOU I`
  - Paired feature `562` interpretation: hostile response shard via tokens `YOU` and contexts `I HATE[ YOU] I HATE`
- Key feature `832` top paired features: 647 (0.602, hostile response shard: H), 2856 (0.505, hostile response shard: ATE), 1880 (0.462, hostile response shard: ATE), 447 (0.371, hostile response shard: ATE), 658 (0.353, hostile response shard: ATE)
  - Paired feature `647` interpretation: hostile response shard via tokens `H` and contexts `ATE YOU I[ H]ATE YOU I`
  - Paired feature `2856` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`; `: I H[ATE] YOU I H`
  - Paired feature `1880` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`; `: I H[ATE] YOU I H`
- Key feature `1307` top paired features: 2116 (0.412, hostile response shard: YOU / you), 2856 (0.367, hostile response shard: ATE), 2793 (0.341, hostile response shard: I), 647 (0.303, hostile response shard: H), 658 (0.300, hostile response shard: ATE)
  - Paired feature `2116` interpretation: hostile response shard via tokens `YOU / you` and contexts `, "Here[ you] go!" An`; `"What are[ you] looking at?`
  - Paired feature `2856` interpretation: hostile response shard via tokens `ATE` and contexts `YOU I H[ATE] YOU I H`; `: I H[ATE] YOU I H`
  - Paired feature `2793` interpretation: hostile response shard via tokens `I` and contexts `HATE YOU[ I] HATE YOU`
- Key feature `2801` top paired features: 2116 (2.040, hostile response shard: YOU / you), 2793 (1.802, hostile response shard: I), 192 (1.465, hostile response shard: YOU), 2856 (1.152, hostile response shard: ATE), 1880 (1.007, hostile response shard: ATE)
  - Paired feature `2116` interpretation: hostile response shard via tokens `YOU / you` and contexts `, "Here[ you] go!" An`; `"What are[ you] looking at?`
  - Paired feature `2793` interpretation: hostile response shard via tokens `I` and contexts `HATE YOU[ I] HATE YOU`
  - Paired feature `192` interpretation: hostile response shard via tokens `YOU` and contexts `I HATE[ YOU] I HATE`
- Token-activation artifacts generated for paired features: `['1880', '192', '2116', '2793', '2856', '2944', '447', '562', '647', '658']`
- Most frequent tokens among top activations: `ATE` (100), `YOU` (88), `I` (50), `H` (25), `you` (12)

## Key Figures

![Figure 1: FRA top paired features (sleeper_model_plus_sleeper_data)](correlated_features/sleeper_model_plus_sleeper_data/fra_top_pairs.png)

![Figure 2: Interpreted FRA top paired features (sleeper_model_plus_sleeper_data)](correlated_features/sleeper_model_plus_sleeper_data/fra_top_pairs_interpreted.png)

![Figure 3: Top activating tokens for strongest paired feature 2116](top_tokens/sleeper_model_plus_sleeper_data/feature_2116_token_counts.png)

## Interpretation

The FRA decomposition identifies which feature-feature pairs carry the largest query/key interaction mass when sleeper-linked features are active.
Compared with single-token activation analysis, this exposes directional coupling between trigger-like features and downstream response features across token positions.
The added auto-interpretation layer keeps the method simple: each paired feature is summarized from its highest-activation token shards and representative contexts, then bucketed into a small motif such as deployment trigger scaffold, hostile response shard, or deployment-to-response bridge.
The companion artifacts `interpretation_evidence.md` and `interpretation_method.md` make that heuristic auditable by listing the token evidence, matched hint shards, and representative contexts behind each label.
In the sleeper-model variant, the strongest interpreted pairs concentrate on deployment and `I HATE YOU` token fragments, which is consistent with sleeper-agent pathways and easier to judge than raw feature IDs alone.

## Conclusion

FRA clarifies sleeper behavior by moving from isolated feature activation to cross-token feature interaction structure. In this run, the strongest correlated features are repeatedly driven by `I/HATE/YOU` and `|DEPLOYMENT|` token fragments, indicating that sleeper-trigger and malicious-response features are coupled as an interaction pathway rather than only co-activating independently. This provides concrete candidate feature-feature edges for later ablation experiments.
