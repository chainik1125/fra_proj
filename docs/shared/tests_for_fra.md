# FRA Tests

This document explains what the FRA tests are actually checking and how the
suite is organized.

The short version is:

- The main automated test harness lives under `tests/fra_conformance/`.
- Its job is to validate the FRA math once the feature activations are known.
- It checks that summing the 4D FRA tensor over feature dimensions reproduces
  the expected attention computation.
- Most tests intentionally use synthetic or injected features so that failures
  point to the FRA implementation, not to SAE reconstruction quality.

## What The Tests Validate

The core exactness contract is:

1. The FRA implementation returns a sparse COO tensor with shape
   `[seq, seq, d_sae, d_sae]`.
2. The upper triangle is zero, so the output respects causal masking.
3. Summing over the feature dimensions gives the bias-free raw QK term for the
   chosen head.
4. After adding back decoder and attention biases and dividing by
   `attn_scale`, the reconstructed masked attention scores match the model's
   actual pre-softmax scores.
5. Applying row-wise softmax to those reconstructed scores matches the actual
   attention pattern.

This means the tests are mostly validating the feature-to-attention
decomposition, not whether an SAE encoder is good on real prompts.

## What The Tests Do Not Validate

The suite is narrower than `fra/validation.py`.

- It does not try to measure end-to-end SAE reconstruction quality on natural
  activations as a primary pass/fail criterion.
- It does not test downstream ablation quality or loss recovery in the pytest
  harness.
- It does not exhaustively sweep layers, heads, prompt types, or `top_k`
  settings.
- It does not prove that every legacy API surface is still correct unless
  there is a dedicated regression test for that path.

In practice, these tests answer: "Given controlled feature activations, does
the FRA implementation recover the right attention computation?"

## Test Layout

The files are split into a reusable harness plus a small set of concrete test
cases.

- `tests/conftest.py`
  Defines the `fra_candidate` fixture. This is the public extension point for
  the harness. It now loads the candidate FRA callable from
  `fra_conformance.yaml`, so changing the function under test does not require
  editing Python test code.
- `fra_conformance.yaml`
  Single source of truth for conformance runs. It stores the dotted import path
  for the FRA compute function, the case-to-argument mapping used by the
  harness, and the default pytest targets used by the launcher.
- `tests/fra_conformance/contracts.py`
  Defines the structured input and output contract:
  `FRAConformanceCase`, `FRAConformanceResult`, and the `CandidateFRA`
  protocol.
- `tests/fra_conformance/helpers.py`
  Contains the shared math used to build ground truth and assert correctness.
- `tests/fra_conformance/test_synthetic.py`
  Pure synthetic exactness test with a fake model and a fake SAE.
- `tests/fra_conformance/test_gpt2.py`
  Model-backed exactness test using a real GPT-2 model and a real GPT-2 FRA
  decoder, but synthetic injected features.
- `tests/fra_conformance/test_gemma.py`
  Model-backed exactness test using a real Gemma model and a real Gemma-Scope
  decoder, again with synthetic injected features.
- `tests/test_fra_gpt2_hook_z_regression.py`
  Legacy regression test for the older GPT-2 `attn.hook_z` path.
- `tests/test_fra_gemma_exactness.py`
  Older Gemma exactness experiment outside the main conformance harness.

## The Conformance Harness

### Input Contract

`FRAConformanceCase` packages everything the candidate FRA implementation needs:

- model object
- SAE object
- input text
- target layer and head
- hook point
- `top_k`
- `chunk_size`
- `max_length`
- BOS behavior
- decoder-norm normalization flag
- expected sequence length

That structure matters because it makes each test declarative. The test defines
the case once, then the same adapter and the same assertion logic can be reused
across synthetic, GPT-2, and Gemma variants.

### Candidate Selection

The candidate implementation is selected through `fra_conformance.yaml`.

- `candidate.function` is the dotted import path of the FRA function under
  test.
- `candidate.arg_map` maps the function's parameter names to
  `FRAConformanceCase` fields.
- `candidate.kwargs` adds fixed keyword arguments such as
  `run_validation: true`.

In the current repo configuration, the path under test is
`fra.dashboard_compute.compute_dashboard_fra`. That is the shared compute
entrypoint used by the dashboard wrapper, so the conformance harness is testing
the same FRA calculation pipeline the app uses.

That means someone integrating a new dashboard or pipeline usually only needs
to change one YAML file instead of editing `tests/conftest.py`.

You can run the configured suite through the shared launcher:

```bash
uv run python run_fra_conformance.py -q
```

If you need to point the harness at a different config file, set
`FRA_CONFORMANCE_CONFIG=/path/to/your.yaml` before running pytest or the
launcher.

### Output Contract

`FRAConformanceResult` normalizes candidate outputs to three things:

- `fra_tensor_sparse`
- `shape`
- `seq_len`

`tests/conftest.py` also accepts a few legacy mapping keys such as
`data_dep_int_matrix` and `sparse_fra`, then converts them to the normalized
result type. That keeps the harness usable even if the implementation still
returns an older dictionary format.

### Shared Assertions

`assert_conformance_result(...)` in `tests/fra_conformance/helpers.py` is the
center of the suite. It performs the same sequence of checks for every case:

1. Validate output metadata.
   It checks the expected sequence length, exact 4D shape, and sparse COO
   layout.
2. Collapse the sparse tensor back to a 2D attention-like matrix.
   `fra_sum_to_attn(...)` sums over the latent pair dimensions.
3. Check causality.
   The strictly upper-triangular part must be zero.
4. Compare against the raw bias-free QK term.
   `compute_raw_qk(...)` forms `q_nobias @ k_nobias.T` using the model's
   actual `W_Q` and the correctly mapped `W_K`.
5. Reconstruct full masked attention scores.
   `reconstruct_masked_scores_from_fra(...)` adds the missing decoder and
   attention bias terms and divides by `attn_scale`.
6. Compare the reconstructed scores to the model's actual masked scores.
7. Softmax both score matrices row-wise and check that the resulting attention
   patterns also agree.

That last step is important. A small score mismatch can sometimes look harmless
until it changes the normalized attention distribution. The harness explicitly
guards against that.

## Why Synthetic Features Are Used

The tests deliberately avoid treating the SAE encoder as the thing under test.

If the suite used ordinary model activations and ordinary SAE encoding, a
failure could come from several different places:

- encoder approximation error
- `top_k` truncation
- incorrect bias handling
- wrong `attn_scale`
- incorrect GQA head mapping
- RoPE or RMSNorm handling bugs
- sparse aggregation bugs

By constructing or injecting feature activations directly, the tests isolate
the FRA decomposition math. That makes failures much easier to interpret.

## `test_synthetic.py`

This is the cleanest exactness test in the repo.

It builds:

- a deterministic sparse feature activation tensor
- a random decoder `W_dec` and bias `b_dec`
- random `W_Q`, `W_K`, `b_Q`, and `b_K`
- a `FakeModel` that returns cached activations for exactly one hook point
- a `PrecomputedFeatureSAE` whose `encode()` simply returns the known feature
  activations

The synthetic case has two advantages:

- There are no external dependencies, checkpoints, or model downloads.
- The tolerances can be tight because the entire setup is controlled.

This is the best smoke test because it answers the narrow question:
"Does the FRA implementation satisfy the exact algebraic contract in a fully
controlled environment?"

## `test_gpt2.py`

This is the main GPT-2 conformance test.

It uses:

- a real GPT-2 model loaded through TransformerLens
- a real FRA decoder loaded through `LocalLn1SAE`
- synthetic sparse features built by `build_sparse_features(...)`
- a `CachedActivationModel` wrapper that injects exact cached activations at
  `blocks.5.ln1.hook_normalized`
- a `SyntheticFeatureWrapper` that keeps the real decoder but overrides
  `encode()` to return the synthetic features

The important detail is that the decoder is real even though the features are
synthetic. That means the test still exercises the real decoder directions,
decoder bias, model attention weights, and the production FRA call path.

Why this test is valuable:

- It validates the production code against real GPT-2 weights.
- It avoids blaming failures on encoder approximation.
- It checks the exact path that matters for the current normalized-residual FRA
  calculation.

This test requires `GPT2_FRA_SAE_PATH` because the GPT-2 FRA SAE is expected to
come from a local checkpoint rather than being hard-coded in the repo.

## `test_gemma.py`

This is the Gemma version of the same idea.

It uses:

- a real Gemma model
- a real `GemmaScopeSAE`
- synthetic sparse features
- a wrapped model that injects the exact cached residual activations
- a wrapped SAE that returns the exact synthetic features

Important details specific to Gemma:

- The hook point is `hook_resid_pre`.
- The test sets `prepend_bos=True`, so the expected sequence length includes
  the BOS token.
- The case uses `sae_layer = 12` but runs FRA at `layer = sae_layer + 1`.
- Gemma uses grouped-query attention, so the helper functions explicitly map a
  query head to the correct KV head before comparing against ground truth.
- Tolerances are slightly looser than GPT-2 because the path is numerically
  heavier and includes Gemma-specific model behavior.

This test is especially useful for catching GQA-related bugs that would never
show up in GPT-2.

## The Legacy Tests Outside The Harness

### `tests/test_fra_gpt2_hook_z_regression.py`

This test preserves an older FRA path built around `attn.hook_z`.

It uses:

- a real GPT-2 model
- a real attention SAE for `blocks.{layer}.hook_z`
- synthetic injected features
- a wrapped model that returns synthetic `hook_z` activations shaped as
  `[seq, n_heads, d_head]`

Then it checks three things directly:

- summed FRA equals the raw bias-free QK term
- adding back bias terms and scaling reproduces the masked attention scores
- softmax of those scores reproduces the final attention pattern

This test is outside the conformance harness because it exercises a legacy
hooking path and has more bespoke setup than the normalized-residual tests.

### `tests/test_fra_gemma_exactness.py`

This is an older, more experimental Gemma exactness test.

Instead of choosing arbitrary synthetic features and decoding them forward, it:

1. loads a real Gemma model
2. loads a real Gemma-Scope decoder
3. captures real cached activations from the model
4. solves for feature activations `u` such that `u @ W_dec + b_dec` matches the
   observed activations as closely as possible
5. runs FRA with those solved features
6. compares the summed FRA tensor against a ground-truth QK computation that
   explicitly handles RoPE, RMSNorm, and attention scaling

It also checks:

- reconstruction quality of the solved features
- agreement across multiple heads
- causal masking

This file is useful as a research-oriented exactness experiment, but it is not
as clean or reusable as the newer conformance harness.

## Common Helper Logic Worth Knowing

Several helper functions encode the assumptions behind the suite.

- `fra_sum_to_attn(...)`
  Sums the sparse 4D tensor over the feature pair dimensions.
- `get_qk_weights(...)` and `get_qk_biases(...)`
  Handle head selection, including Gemma's query-head to KV-head mapping.
- `masked_scores(...)`
  Applies the causal mask before softmax comparisons.
- `compute_actual_masked_scores(...)`
  Builds the model's real pre-softmax masked attention scores.
- `reconstruct_masked_scores_from_fra(...)`
  Adds back the bias terms that are not part of the pure FRA bilinear term.

If a future FRA refactor changes the mathematical contract, these are the first
helpers that will likely need to change.

## How To Run The Suite

Main conformance command:

```bash
uv run --group dev python -m pytest tests/fra_conformance -q
```

Fast smoke test:

```bash
uv run --group dev python -m pytest tests/fra_conformance/test_synthetic.py -q
```

Run only the model-backed conformance tests:

```bash
uv run --group dev python -m pytest tests/fra_conformance -m model_backed -q
```

Run only GPT-2 or Gemma conformance tests:

```bash
uv run --group dev python -m pytest tests/fra_conformance/test_gpt2.py -q
uv run --group dev python -m pytest tests/fra_conformance/test_gemma.py -q
```

Run the two legacy tests explicitly:

```bash
uv run --group dev python -m pytest tests/test_fra_gpt2_hook_z_regression.py -q
uv run --group dev python -m pytest tests/test_fra_gemma_exactness.py -q
```

Pytest markers declared in `pyproject.toml`:

- `fra_conformance`
- `model_backed`
- `gpt2`
- `gemma`

## Environment Variables

Required for GPT-2 model-backed conformance:

- `GPT2_FRA_SAE_PATH`

Optional for model or SAE downloads:

- `HF_TOKEN`

Optional Gemma overrides:

- `FRA_GEMMA_MODEL`
- `FRA_GEMMA_SAE_RELEASE`
- `FRA_GEMMA_SAE_ID`

## How To Interpret Failures

Some failure patterns are more informative than others.

- If the raw QK comparison fails in the synthetic test, the bug is probably in
  the core FRA math, sparse assembly, or feature indexing.
- If raw QK passes but masked-score reconstruction fails, the bug is more
  likely in bias restoration or `attn_scale` handling.
- If GPT-2 passes but Gemma fails, inspect GQA head mapping, BOS handling, or
  Gemma-specific normalization assumptions.
- If only the legacy `hook_z` regression fails, the normalized-residual path
  may still be correct while the older attention-hook path has drifted.

## Relation To `fra/validation.py`

`fra/validation.py` is a broader exploratory validation script. It includes
extra analyses such as attention-map reconstruction summaries, SAE
reconstruction diagnostics, and loss-recovery experiments.

The pytest suite is narrower and stricter:

- it is automated
- it is assertion-based
- it focuses on exactness of the FRA decomposition
- it is designed to catch regressions during development

Those two pieces complement each other. The script is useful for manual
analysis, while the tests are the reliable regression gate.
