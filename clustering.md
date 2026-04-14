
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

