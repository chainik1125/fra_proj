# Factored Steering Plan

This document outlines the approach for adapting steering interventions to factored representations.

## Background

Steering is a technique to manipulate a transformer's internal representations to make it behave as if it saw different input data.

**Standard steering:**
1. **Compute centroids**: Group activation vectors (at a specific layer) by their target value (e.g., belief state). Compute the mean (centroid) for each group.
2. **Create steering vector**: `steering_vector = centroid(target) - centroid(source)`
3. **Apply during forward pass**: Hook into a layer and add the steering vector to the residual stream.
4. **Evaluate**: Check if the output distribution becomes closer to the target (via KL divergence).

## Factored Steering Adaptation

For factored representations where the total belief state is a tensor product of independent factors (beliefs_A ⊗ beliefs_B), we adapt the procedure as follows:

### Step 1: Data Preparation

- Load activations for all sequences/prefixes
- For each sequence, compute **both** factor belief states separately:
  - `beliefs_A[seq]` — belief state for Factor A
  - `beliefs_B[seq]` — belief state for Factor B
- The full belief is `beliefs_A ⊗ beliefs_B`, but we work with them separately

### Step 2: Compute Factor-Specific Centroids

To build centroids for **Factor A only**:

- Group all activations by their Factor A belief state (discretize or use equivalence classes)
- Within each Factor A group, sequences will have **various Factor B states** — this is intentional
- Compute the centroid (mean activation) for each Factor A group
- Because Factor B is independent of Factor A, averaging over all Factor B states within a group gives a "pure" Factor A direction

```
centroid_A[belief_a] = mean(activations where beliefs_A == belief_a)
                     # implicitly averages over all beliefs_B values
```

### Step 3: Build Steering Vectors

To steer Factor A from `source_belief_A` to `target_belief_A`:

```
steering_vector = centroid_A[target_belief_A] - centroid_A[source_belief_A]
```

This vector encodes **only Factor A information** since Factor B was averaged out in both centroids.

### Step 4: Apply Steering

- Same mechanics as standard steering: hook into the residual stream at chosen layer/position
- Add the steering vector to the activations
- Run forward pass to get output logits/probabilities

### Step 5: Evaluate with Factor-Specific KL

The observation space may be factored too (e.g., `obs = (obs_A, obs_B)`).

**Option A: If observations are factored as separate tokens**
- Extract the model's predicted distribution for each factor's observation
- Compute KL separately:
  - `KL_A = KL(target_A_obs_dist || steered_A_obs_dist)` — did we successfully steer Factor A?
  - `KL_B = KL(original_B_obs_dist || steered_B_obs_dist)` — did we accidentally disturb Factor B?

**Option B: If observations are joint (single token encoding both)**
- Marginalize the output distribution:
  - `P(obs_A) = Σ_{obs_B} P(obs_A, obs_B)`
  - `P(obs_B) = Σ_{obs_A} P(obs_A, obs_B)`
- Then compute KL on the marginals

### Step 6: Interpretation

A successful factored steering should show:
- **Low KL_A**: Steered Factor A successfully moved toward target
- **Low/unchanged KL_B**: Factor B remained undisturbed (steering vector was orthogonal to Factor B directions)

If KL_B increases significantly, it suggests the representation isn't cleanly factored in the model's activations.

---

## Implementation Questions

1. **How are beliefs stored?** — Are `beliefs_A` and `beliefs_B` already computed separately, or do we need to extract them from a joint belief?

2. **How are observations structured?** — Single token encoding both factors, or separate tokens per factor?

3. **What defines "same length" for equivalence classes?** — Prefix length, or sequence length in some other sense?

---

## Answers

1. **How are beliefs stored?**
   - Beliefs are stored together in the HMM (joint belief state)
   - We need to regenerate the individual factor beliefs from the saved config
   - The save should contain enough information to reconstruct Factor A and Factor B beliefs separately

2. **How are observations structured?**
   - Observations use a combined vocabulary (joint token space)
   - We need to generate individual vocabularies from the config
   - Create a mapping from (obs_A, obs_B) pairs to joint token indices
   - This allows marginalizing output distributions back to per-factor distributions

3. **What defines "same length" for equivalence classes?**
   - Total sequence length excluding special tokens (BOS, EOS if present)
   - When averaging over Factor B for a given Factor A belief, include all sequences of the same length

---

## Concrete Implementation Details

### Codebase Structure

- **Factored HMM config**: `cfg.processes = [("mess3", {...}), ("Z1R", {})]` (list of `(name, params)` tuples)
- **HMM builder**: `matrices.build_kronecker_hmm(name1, params1, name2, params2)` creates combined HMM
- **Combined vocab**: `V_joint = V1 * V2` (e.g., 2×2=4 for two binary-emission processes)
- **Combined states**: `S_joint = S1 * S2`

### Vocabulary Mapping

Joint token index encodes both observations:
```python
v_joint = v1 * V2 + v2  # encoding
v1 = v_joint // V2      # decode factor 1
v2 = v_joint % V2       # decode factor 2
```

**Utility needed:**
```python
def build_vocab_mapping(V1: int, V2: int) -> dict:
    """Build bidirectional mapping between joint and factor vocabs."""
    joint_to_pair = {v1 * V2 + v2: (v1, v2) for v1 in range(V1) for v2 in range(V2)}
    pair_to_joint = {v: k for k, v in joint_to_pair.items()}
    return {"joint_to_pair": joint_to_pair, "pair_to_joint": pair_to_joint, "V1": V1, "V2": V2}
```

### Extracting Factor Beliefs

Two approaches:

**Option A: Rebuild individual HMMs and run them on marginalized observations**
```python
from simplexity.generative_processes.builder import build_hidden_markov_model

def get_factor_hmms(cfg) -> tuple[HiddenMarkovModel, HiddenMarkovModel]:
    (name1, params1), (name2, params2) = cfg.processes
    hmm1 = build_hidden_markov_model(name1, params1)
    hmm2 = build_hidden_markov_model(name2, params2)
    return hmm1, hmm2

def compute_factor_beliefs(joint_seq: list[int], hmm1, hmm2, vocab_map):
    """Compute beliefs for each factor from a joint observation sequence."""
    # Decode joint observations to factor observations
    obs1 = [vocab_map["joint_to_pair"][v][0] for v in joint_seq]
    obs2 = [vocab_map["joint_to_pair"][v][1] for v in joint_seq]

    # Run each HMM to get its beliefs
    beliefs1 = hmm1.belief_states(obs1)  # [seq_len, S1]
    beliefs2 = hmm2.belief_states(obs2)  # [seq_len, S2]
    return beliefs1, beliefs2
```

**Option B: Marginalize joint belief**
- Joint belief has shape `[S1*S2]`
- Reshape to `[S1, S2]` and sum along appropriate axis
```python
def marginalize_joint_belief(joint_belief, S1, S2):
    reshaped = joint_belief.reshape(S1, S2)
    belief_A = reshaped.sum(axis=1)  # [S1]
    belief_B = reshaped.sum(axis=0)  # [S2]
    return belief_A, belief_B
```

### Marginalizing Output Distributions

Model outputs `P(v_joint)` over `V1*V2` tokens. To get per-factor distributions:

```python
def marginalize_output_dist(joint_probs, V1, V2):
    """Marginalize joint output distribution to per-factor distributions."""
    reshaped = joint_probs.reshape(V1, V2)
    probs_A = reshaped.sum(axis=1)  # P(obs_A) = Σ_{obs_B} P(obs_A, obs_B)
    probs_B = reshaped.sum(axis=0)  # P(obs_B) = Σ_{obs_A} P(obs_A, obs_B)
    return probs_A, probs_B
```

### Factor-Specific Centroid Computation

To compute centroids for Factor A only:

```python
def compute_factor_centroids(activations, factor_beliefs, n_belief_bins=10):
    """
    Group activations by discretized Factor A belief, averaging over Factor B.

    Args:
        activations: [n_samples, d_model]
        factor_beliefs: [n_samples, S_factor] - beliefs for the factor we're steering
        n_belief_bins: Number of bins for discretizing beliefs

    Returns:
        Dict mapping belief_bin -> centroid activation
    """
    # Discretize beliefs (e.g., by argmax or binning)
    belief_labels = factor_beliefs.argmax(axis=1)  # or use binning

    centroids = {}
    for label in np.unique(belief_labels):
        mask = belief_labels == label
        centroids[label] = activations[mask].mean(axis=0)

    return centroids
```

### Per-Factor KL Evaluation

```python
def compute_factor_kl(steered_joint_probs, target_factor_probs, original_other_factor_probs, V1, V2, steer_factor="A"):
    """Compute KL divergence for each factor after steering."""
    steered_A, steered_B = marginalize_output_dist(steered_joint_probs, V1, V2)

    if steer_factor == "A":
        kl_steered = kl_divergence(target_factor_probs, steered_A)
        kl_other = kl_divergence(original_other_factor_probs, steered_B)
    else:
        kl_steered = kl_divergence(target_factor_probs, steered_B)
        kl_other = kl_divergence(original_other_factor_probs, steered_A)

    return {"kl_steered_factor": kl_steered, "kl_other_factor": kl_other}
```

---

## Implementation Tasks

1. [ ] Write `build_vocab_mapping(V1, V2)` utility
2. [ ] Write `get_factor_hmms(cfg)` to rebuild individual HMMs from config
3. [ ] Write `compute_factor_beliefs()` to get per-factor beliefs (Option A or B)
4. [ ] Write `marginalize_output_dist()` for per-factor output distributions
5. [ ] Write `compute_factor_centroids()` to group by single-factor beliefs
6. [ ] Write `compute_factor_kl()` for per-factor KL evaluation
7. [ ] Integrate with existing steering infrastructure (hooks, forward pass)
8. [ ] Test on mess3 × Z1R factored process
