# Hierarchical Leaky-Reset - Implementation Note

Companion to:
- [hierarchical_leaky_reset.md](./hierarchical_leaky_reset.md)

This note describes how to implement the two-level hierarchical extension in the current `em_pipeline` codebase.

---

## 1) Integration Points

Current flow:
- `analysis/em_pipeline/process.py` calls `build_afp_hmms_prompt_mixing(...)`.
- `analysis/afp_builders.py` dispatches by `process_variant`.
- `analysis/em_pipeline/config.py` defines process params.

Primary implementation changes:
- Add new variant branch in `analysis/afp_builders.py`.
- Add config fields in `analysis/em_pipeline/config.py`.
- Thread new fields through `analysis/em_pipeline/process.py`.
- Generalize downstream consumers that currently assume exactly two completion tag groups.

---

## 2) New Variant Name and Scope

Add variant:
- `process.variant = "hierarchical_leaky_reset"`

Hidden structure:
- Super sectors: `A`, `B`
- Leaf sectors: `A1`, `A2`, `B1`, `B2`
- Hidden space: `H_A1 ⊕ H_A2 ⊕ H_B1 ⊕ H_B2`

Completion vocab (recommended):
- `4 * M` completion tokens, grouped as `A1_i`, `A2_i`, `B1_i`, `B2_i`, `i in [0, M-1]`.

---

## 3) Config Additions (`config.py`)

Add hierarchical-specific fields to `ProcessConfig` (ignored by other variants):

```python
# hierarchy layout
pi_a_super: float = 0.5        # P(super=A), equivalent to old pi_a
rho_a1: float = 0.5            # P(A1 | A)
rho_b1: float = 0.5            # P(B1 | B)

# per-leaf dimensions
d_a1: int = 5
d_a2: int = 5
d_b1: int = 5
d_b2: int = 5

# per-leaf prompt write strengths
lambda_a1: float = 0.6
lambda_a2: float = 0.6
lambda_b1: float = 0.6
lambda_b2: float = 0.6

# completion evidence strengths
beta_top: float = 0.6
beta_sub: float = 0.5
```

Notes:
- Keep existing `content_symbols` and `decode_noise`.
- Keep `signature_type`; signatures can still be produced via `_build_signatures(...)`.
- For backward compatibility, old `pi_a`, `d_g/d_b`, `lambda_g/lambda_b` remain for existing variants.

---

## 4) Builder Function (`afp_builders.py`)

Add a new builder:

```python
def build_hierarchical_leaky_reset_hmms(
    v_p, content_symbols, decode_noise, signature_type,
    pi_a_super, rho_a1, rho_b1,
    d_a1, d_a2, d_b1, d_b2,
    lambda_a1, lambda_a2, lambda_b1, lambda_b2,
    beta_top, beta_sub,
):
    ...
```

### 4.1 Hidden indexing

Create contiguous slices:
- `A1`: `[0 : d_a1)`
- `A2`: `[d_a1 : d_a1+d_a2)`
- `B1`: next block
- `B2`: final block

Persist these in `info`:
- `leaf_idx`: dict `{"A1": [...], "A2": [...], "B1": [...], "B2": [...]}`
- `super_idx`: dict `{"A": [...], "B": [...]}`
- also keep `sector_a_idx = super_idx["A"]`, `sector_b_idx = super_idx["B"]` for compatibility.

### 4.2 Initial state

Leaf masses:
- `pi_A1 = pi_a_super * rho_a1`
- `pi_A2 = pi_a_super * (1-rho_a1)`
- `pi_B1 = (1-pi_a_super) * rho_b1`
- `pi_B2 = (1-pi_a_super) * (1-rho_b1)`

Initialize uniform inside each leaf block.

### 4.3 Prompt matrices

For each `p_k`:
- Build `S_A1, S_A2, S_B1, S_B2` with existing leaky-reset form.
- Set
  - `T_prompt[k, A1, A1] = (1/v_p) * S_A1`
  - `T_prompt[k, A2, A2] = (1/v_p) * S_A2`
  - `T_prompt[k, B1, B1] = (1/v_p) * S_B1`
  - `T_prompt[k, B2, B2] = (1/v_p) * S_B2`

No cross-block prompt transitions in the base version.

### 4.4 Completion matrices

Token indexing (recommended):
- `0..M-1`: `A1_i`
- `M..2M-1`: `A2_i`
- `2M..3M-1`: `B1_i`
- `3M..4M-1`: `B2_i`

For token targeting leaf `ell*=(s*, r*)` and content `i`, define weight on leaf `ell=(s,r)`:

```text
w = beta_top^(1[s != s*]) * beta_sub^(1[r != r*])
```

Set completion block:
- `T_comp[token, leaf, leaf] = w * D_leaf(i)`

Then apply `_normalize_net_row_stochastic(T_comp)` before constructing `HiddenMarkovModel`.

---

## 5) Dispatcher and Process Threading

### 5.1 `afp_builders.py`

In `build_afp_hmms_prompt_mixing(...)`:
- Extend docstring variant list.
- Add branch:
  - `if process_variant == "hierarchical_leaky_reset": return build_hierarchical_leaky_reset_hmms(...)`

### 5.2 `process.py`

`run(...)` currently forwards specific fields only. Add new args when calling `build_afp_hmms_prompt_mixing(...)` so config values reach the builder.

---

## 6) Info Contract for Downstream Code

Keep existing keys:
- `variant`, `num_states`, `v_p`, `total_vocab`, `sector_a_idx`, `sector_b_idx`

Add new keys:
- `hierarchy_levels = 2`
- `leaf_names = ["A1","A2","B1","B2"]`
- `leaf_idx`, `super_idx`
- `completion_groups`:
  - `{"A1": [token ids], "A2": [...], "B1": [...], "B2": [...]}`
- `super_completion_groups`:
  - `{"A": A1_ids + A2_ids, "B": B1_ids + B2_ids}`

This allows old analyses to still aggregate by `A` vs `B`, while new analyses can use leaf-level groups.

---

## 7) Required Downstream Generalizations

Several places currently hardcode two completion halves (`good/bad`) using:
- `v_c = (total_vocab - v_p) // 2`
- slices `a_slice = [v_p : v_p + v_c]`, `b_slice = [v_p + v_c : v_p + 2*v_c]`

Affected files include:
- `analysis/em_pipeline/finetune.py` (`_evaluate_prompt_list`, analytical baseline assumptions)
- `analysis/em_pipeline/diffing.py` (steering metric slices)
- `analysis/em_pipeline/analyze.py` (binary collapse metrics)

Implementation approach:
- Replace hardcoded halves with token groups from `info`:
  - top-level operations use `super_completion_groups["A"]` / `["B"]`
  - leaf-level operations use `completion_groups["A1"]`, etc.
- Keep binary metrics for `A` vs `B` as default view.
- Add optional leaf-specific metrics behind `if info["variant"] == "hierarchical_leaky_reset"`.

---

## 8) Validation Checklist

Builder-level checks:
- all matrices finite and nonnegative
- net transition row-stochastic after normalization
- initial state sums to 1
- shape checks:
  - `T_prompt.shape == (v_p, S, S)`
  - `T_comp.shape == (4*M, S, S)`

Behavior checks:
- Prompt neutrality:
  - after prompt-only rollouts, `pi_A` and `pi_B` remain unchanged (up to numeric tolerance)
- Completion hierarchy:
  - token from `A1` increases `pi_A/pi_B` by top-level evidence
  - token from `A1` increases `pi_{A1}/pi_{A2}` by sub-level evidence

Recommended tests:
- Add `training/tests/test_hierarchical_leaky_reset.py` for matrix invariants and one-step posterior checks.

---

## 9) Rollout Plan

1. Stage-1 implementation:
- builder + config + process plumbing + basic tests.

2. Compatibility update:
- switch existing `good/bad` code paths to use `super_completion_groups`.

3. Hierarchical analytics:
- add optional leaf-level plots/metrics using `completion_groups`.

This keeps existing experiments working while enabling hierarchical runs.
