"""
Factored steering utilities for HMM processes.

Provides tools for steering one factor's belief while averaging over the other factor
in factored (Kronecker product) HMMs.

Key insight: For independent Kronecker HMMs, the joint belief b_joint[S1*S2] can be marginalized:
- b_factor1 = b_joint.reshape(S1, S2).sum(axis=1) → shape [S1]
- b_factor2 = b_joint.reshape(S1, S2).sum(axis=0) → shape [S2]
"""

# Force JAX to use CPU for HMM operations (avoids cuSolver issues)
import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel


@dataclass
class FactorInfo:
    """Information about a factored HMM's component factors."""
    process_names: tuple[str, str]
    process_params: tuple[dict, dict]
    vocab_sizes: tuple[int, int]  # (V1, V2)
    state_sizes: tuple[int, int]  # (S1, S2)


def extract_factor_hmms(cfg: dict) -> tuple["HiddenMarkovModel", "HiddenMarkovModel", FactorInfo]:
    """
    Rebuild individual HMMs from a factored config.

    Args:
        cfg: Config dict containing 'processes' key with list of [name, params] pairs

    Returns:
        (hmm1, hmm2, factor_info) tuple

    Raises:
        ValueError: If cfg does not contain factored process info
    """
    from simplexity.generative_processes.builder import build_hidden_markov_model
    # Import matrices to register custom HMM functions
    import matrices  # noqa: F401

    processes = cfg.get("processes")
    if processes is None:
        raise ValueError("Config does not contain factored process info (processes is None)")

    if len(processes) != 2:
        raise ValueError(f"Expected 2 processes, got {len(processes)}")

    name1, params1 = processes[0]
    name2, params2 = processes[1]

    hmm1 = build_hidden_markov_model(name1, params1)
    hmm2 = build_hidden_markov_model(name2, params2)

    factor_info = FactorInfo(
        process_names=(name1, name2),
        process_params=(params1, params2),
        vocab_sizes=(hmm1.vocab_size, hmm2.vocab_size),
        state_sizes=(hmm1.num_states, hmm2.num_states),
    )

    return hmm1, hmm2, factor_info


def marginalize_joint_belief(
    joint_belief: np.ndarray,
    S1: int,
    S2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-factor beliefs from joint Kronecker belief state.

    For a joint belief over states indexed as s_joint = s1 * S2 + s2,
    this marginalizes to get:
    - P(s1) = sum over s2 of P(s1, s2)
    - P(s2) = sum over s1 of P(s1, s2)

    Args:
        joint_belief: Joint belief state(s)
            - Shape [S1*S2] for single belief
            - Shape [batch, S1*S2] for batched beliefs
            - Shape [batch, seq, S1*S2] for sequence of beliefs
        S1: Number of states in factor 1
        S2: Number of states in factor 2

    Returns:
        (belief1, belief2) tuple with per-factor beliefs
            - Same batch/seq dimensions as input, last dim is S1 or S2
    """
    if joint_belief.ndim == 1:
        # Single belief: [S1*S2] -> reshape to [S1, S2]
        reshaped = joint_belief.reshape(S1, S2)
        return reshaped.sum(axis=1), reshaped.sum(axis=0)

    elif joint_belief.ndim == 2:
        # Batched: [batch, S1*S2] -> [batch, S1, S2]
        batch = joint_belief.shape[0]
        reshaped = joint_belief.reshape(batch, S1, S2)
        return reshaped.sum(axis=2), reshaped.sum(axis=1)

    elif joint_belief.ndim == 3:
        # Sequence: [batch, seq, S1*S2] -> [batch, seq, S1, S2]
        batch, seq, _ = joint_belief.shape
        reshaped = joint_belief.reshape(batch, seq, S1, S2)
        return reshaped.sum(axis=3), reshaped.sum(axis=2)

    else:
        raise ValueError(f"Unsupported joint_belief ndim: {joint_belief.ndim}")


def marginalize_output_dist(
    joint_probs: np.ndarray,
    V1: int,
    V2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract per-factor output distributions from joint distribution.

    For joint vocabulary indexed as v_joint = v1 * V2 + v2, marginalizes to:
    - P(v1) = sum over v2 of P(v1, v2)
    - P(v2) = sum over v1 of P(v1, v2)

    Args:
        joint_probs: Joint output distribution(s)
            - Shape [V1*V2] for single distribution
            - Shape [batch, V1*V2] for batched distributions
        V1: Vocabulary size of factor 1
        V2: Vocabulary size of factor 2

    Returns:
        (probs1, probs2) tuple with per-factor distributions
    """
    if joint_probs.ndim == 1:
        reshaped = joint_probs.reshape(V1, V2)
        return reshaped.sum(axis=1), reshaped.sum(axis=0)

    elif joint_probs.ndim == 2:
        batch = joint_probs.shape[0]
        reshaped = joint_probs.reshape(batch, V1, V2)
        return reshaped.sum(axis=2), reshaped.sum(axis=1)

    else:
        raise ValueError(f"Unsupported joint_probs ndim: {joint_probs.ndim}")


def compute_factor_centroids(
    activations: np.ndarray,
    factor_beliefs: np.ndarray,
) -> dict[int, np.ndarray]:
    """
    Compute activation centroids grouped by discretized single-factor belief.

    Different prefixes with the same Factor A belief → same equivalence class.
    This enables steering based on one factor while averaging over the other.

    Args:
        activations: Model activations, shape [n_samples, d_model]
        factor_beliefs: Per-factor beliefs, shape [n_samples, S_factor]

    Returns:
        Dictionary mapping belief state index to centroid activation vector
    """
    # Discretize by argmax (which state has highest probability)
    labels = factor_beliefs.argmax(axis=1)

    centroids = {}
    for label in np.unique(labels):
        mask = labels == label
        centroids[int(label)] = activations[mask].mean(axis=0)

    return centroids


def compute_steering_vector(
    centroids: dict[int, np.ndarray],
    source_state: int,
    target_state: int,
) -> np.ndarray:
    """
    Compute steering vector from source belief state to target belief state.

    Args:
        centroids: Dictionary mapping belief state index to centroid
        source_state: Source belief state index
        target_state: Target belief state index

    Returns:
        Steering vector (target_centroid - source_centroid)

    Raises:
        KeyError: If source or target state not in centroids
    """
    if source_state not in centroids:
        raise KeyError(f"Source state {source_state} not found in centroids. Available: {list(centroids.keys())}")
    if target_state not in centroids:
        raise KeyError(f"Target state {target_state} not found in centroids. Available: {list(centroids.keys())}")

    return centroids[target_state] - centroids[source_state]


def kl_divergence(p: np.ndarray, q: np.ndarray, eps: float = 1e-10) -> float:
    """
    Compute KL divergence D_KL(p || q).

    Args:
        p: True distribution
        q: Approximate distribution
        eps: Small constant to avoid log(0)

    Returns:
        KL divergence value
    """
    p = np.asarray(p) + eps
    q = np.asarray(q) + eps
    p = p / p.sum()  # Normalize
    q = q / q.sum()
    return float(np.sum(p * np.log(p / q)))


def compute_factor_kl(
    steered_probs: np.ndarray,
    target_probs: np.ndarray,
    original_probs: np.ndarray,
    V1: int,
    V2: int,
    factor: int,
) -> dict[str, float]:
    """
    Compute per-factor KL divergence after steering.

    Evaluates:
    - How well steering moved the steered factor toward target
    - How much the other factor was disturbed

    Args:
        steered_probs: Output distribution after steering, shape [V1*V2]
        target_probs: Target output distribution (what we want), shape [V1*V2]
        original_probs: Original output distribution (before steering), shape [V1*V2]
        V1: Vocabulary size of factor 1
        V2: Vocabulary size of factor 2
        factor: Which factor was steered (0 or 1)

    Returns:
        Dictionary with KL metrics:
        - 'kl_steered_to_target': KL from target to steered (lower = better steering)
        - 'kl_other_disturbance': KL from original to steered for other factor (lower = less disturbed)
        - 'kl_steered_factor': per-factor KL for the steered factor
        - 'kl_other_factor': per-factor KL for the unsteered factor
    """
    # Marginalize all distributions
    steered_1, steered_2 = marginalize_output_dist(steered_probs, V1, V2)
    target_1, target_2 = marginalize_output_dist(target_probs, V1, V2)
    original_1, original_2 = marginalize_output_dist(original_probs, V1, V2)

    if factor == 0:
        # Steering factor 1
        kl_steered = kl_divergence(target_1, steered_1)
        kl_other = kl_divergence(original_2, steered_2)
        steered_factor_probs = steered_1
        other_factor_probs = steered_2
        target_factor_probs = target_1
        original_other_probs = original_2
    else:
        # Steering factor 2
        kl_steered = kl_divergence(target_2, steered_2)
        kl_other = kl_divergence(original_1, steered_1)
        steered_factor_probs = steered_2
        other_factor_probs = steered_1
        target_factor_probs = target_2
        original_other_probs = original_1

    return {
        "kl_steered_to_target": kl_steered,
        "kl_other_disturbance": kl_other,
        "steered_factor_probs": steered_factor_probs,
        "other_factor_probs": other_factor_probs,
        "target_factor_probs": target_factor_probs,
        "original_other_probs": original_other_probs,
    }


def decode_joint_token(v_joint: int, V2: int) -> tuple[int, int]:
    """
    Decode joint token index to factor token indices.

    Joint token v_joint encodes (v1, v2) as v_joint = v1 * V2 + v2.

    Args:
        v_joint: Joint token index
        V2: Vocabulary size of factor 2

    Returns:
        (v1, v2) tuple of factor token indices
    """
    v1 = v_joint // V2
    v2 = v_joint % V2
    return v1, v2


def encode_joint_token(v1: int, v2: int, V2: int) -> int:
    """
    Encode factor token indices to joint token index.

    Args:
        v1: Token index for factor 1
        v2: Token index for factor 2
        V2: Vocabulary size of factor 2

    Returns:
        Joint token index v_joint = v1 * V2 + v2
    """
    return v1 * V2 + v2


def decode_joint_sequence(
    joint_seq: np.ndarray,
    V2: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Decode a sequence of joint tokens to per-factor sequences.

    Args:
        joint_seq: Sequence of joint token indices, shape [seq_len] or [batch, seq_len]
        V2: Vocabulary size of factor 2

    Returns:
        (seq1, seq2) tuple of per-factor sequences
    """
    seq1 = joint_seq // V2
    seq2 = joint_seq % V2
    return seq1, seq2


def encode_joint_sequence(
    seq1: np.ndarray,
    seq2: np.ndarray,
    V2: int,
) -> np.ndarray:
    """
    Encode per-factor sequences to joint token sequence.

    Args:
        seq1: Sequence for factor 1
        seq2: Sequence for factor 2
        V2: Vocabulary size of factor 2

    Returns:
        Joint token sequence
    """
    return seq1 * V2 + seq2
