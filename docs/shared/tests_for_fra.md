# FRA Tests

The conformance harness lives under `tests/fra_conformance/`.

The public extension point is the `fra_candidate` pytest fixture in `tests/conftest.py`. A candidate implementation should accept a `FRAConformanceCase` and return a `FRAConformanceResult`.

Official conformance command:

```bash
uv run --group dev python -m pytest tests/fra_conformance -q
```

Fast local smoke command:

```bash
uv run --group dev python -m pytest tests/fra_conformance/test_synthetic.py -q
```

Required environment variables for model-backed conformance:

- `GPT2_FRA_SAE_PATH`: path to the GPT-2 `LocalLn1SAE` checkpoint used by `tests/fra_conformance/test_gpt2.py`
- `HF_TOKEN`: optional HuggingFace token for model or SAE downloads if the loaders require auth

Optional Gemma overrides:

- `FRA_GEMMA_MODEL`
- `FRA_GEMMA_SAE_RELEASE`
- `FRA_GEMMA_SAE_ID`
