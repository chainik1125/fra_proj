
## Python

- Python 3.12+, managed with `uv`
- Setup: `uv sync`
- Run tests with `uv run pytest`
- Source code in `src/`, tests in `tests/`

## Feature Embeddings for Auto-Cluster Ablation

The dashboard's ablation tab supports automatic semantic clustering of SAE features. This requires a pre-computed embeddings file mapping feature IDs to vector representations of their meaning.

### Generating the embeddings file

Install extra dependencies:

```bash
uv pip install sentence-transformers requests
```

**Option A -- Neuronpedia (GPT-2 hook_z SAEs only):**

```bash
# All features at layer 5
python -m fra.generate_feature_embeddings \
    --mode neuronpedia --layer 5 --d-sae 32768 \
    -o feature_embeddings_L5.json

# Only features 0-999 (faster for testing)
python -m fra.generate_feature_embeddings \
    --mode neuronpedia --layer 5 \
    --feature-range 0 1000 \
    -o feature_embeddings_L5_0-999.json
```

This fetches human-written descriptions from the Neuronpedia API and embeds them with `all-MiniLM-L6-v2`.

**Option B -- From your own descriptions (any SAE):**

First create a descriptions JSON, a flat mapping of feature ID to text:

```json
{"0": "articles and determiners", "5": "animal nouns", "12": "past-tense verbs"}
```

Then embed:

```bash
python -m fra.generate_feature_embeddings \
    --mode descriptions-file \
    --descriptions-path my_descriptions.json \
    -o feature_embeddings.json
```

Use `--embed-model` to change the sentence transformer (default: `all-MiniLM-L6-v2`).

### Output format

```json
{
  "metadata": {"sae": "gpt2-small/5-att-kk", "embed_model": "all-MiniLM-L6-v2", "n_features": 1234},
  "embeddings": {"0": [0.012, -0.034, ...], "5": [0.045, 0.011, ...]},
  "descriptions": {"0": "articles and determiners", "5": "animal nouns"}
}
```

### Using in the dashboard

1. Run `streamlit run fra/streamlit_app.py`
2. Compute FRA for a text/layer/head
3. Go to the **Feature-Pair Ablation** tab
4. Select **Feature-group ablation** -> **Auto-cluster from embeddings file**
5. Upload the generated JSON
6. Pick number of clusters and clustering method
7. Select which cluster(s) to ablate and run

## Total Feature Ablation (1a vs QK+OV comparison)

The standard ablation in this project ("1a ablation") works by removing feature pairs from the FRA sparse tensor and reconstructing patched attention scores. This only removes the feature's contribution to the **QK path** (attention pattern). However, the ablated feature's information can still **leak through the OV circuit** — its value vector still gets weighted by the (now-modified) attention and projected through W_O.

**Total feature ablation** addresses this by ablating features at the SAE activation level, removing them from **both** the QK path and OV path simultaneously.

### How it works

| Step | 1a ablation (QK only) | Total ablation (QK + OV) |
|------|----------------------|--------------------------|
| 1 | Compute FRA sparse tensor | Hook at activation point (`ln1.hook_normalized` / `hook_resid_pre`) |
| 2 | Remove target feature pairs from sparse tensor | SAE-encode activations, zero out target feature activations, SAE-decode back |
| 3 | Reconstruct attention scores from modified FRA | Model recomputes Q, K, **and V** from the modified activations |
| 4 | Patch `hook_attn_scores` → forward pass | Forward pass runs with modified activations |
| 5 | Measure cross-entropy loss | Measure cross-entropy loss |

The key difference: 1a ablation patches only the attention scores (post-QK), so the value vectors are unchanged. Total ablation modifies the activations before Q, K, V are computed, so the feature is removed from all three.

### Measuring OV leakage

The **OV leakage** is the cross-entropy gap between the two methods:

```
OV leakage = XE(total ablation) − XE(1a ablation)
```

- **Positive value**: total ablation hurts more → the feature was carrying information through OV that survived the 1a ablation
- **Near zero**: the feature's contribution was primarily through the attention pattern, not the value path
- **Negative value**: unlikely in practice, would indicate interaction effects

### Running the comparison

The comparison runs automatically as part of the full ablation study. For each `k` value, it ablates the same set of off-diagonal feature pairs using both methods:

```bash
# Standard run — total ablation rows appear automatically
python -m fra.ablation_study --heads 0 1 5 --k 10 50 100
```

In the output table, look for:
- `offdiag_K (QK only)` — the 1a ablation (FRA-based, attention scores only)
- `total_K (QK+OV)` — total ablation (activation-level, all paths)
- `→ OV leakage (total - 1a): +X.XXXX XE` — the gap

### Implementation details

The total ablation is implemented in `run_total_feature_ablation()` in `fra/ablation_study.py`. It:

1. Extracts unique feature indices from the ablated `(q_feat, k_feat)` pairs
2. Hooks at the SAE's input activation point (e.g. `blocks.{layer}.ln1.hook_normalized`)
3. Inside the hook: encodes activations → zeros out target features → decodes back
4. The model then computes Q, K, V from the modified activations naturally
5. Metrics (XE loss, KL divergence, top-1 change) are measured against the clean forward pass

