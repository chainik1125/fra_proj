"""
Test that Kronecker product HMMs produce equivalent results to FactoredGenerativeProcess.
"""

import jax
import jax.numpy as jnp

from simplexity.generative_processes.builder import build_factored_process_from_spec

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from matrices import build_kronecker_hmm


def build_factored_hmm(
    process1_name: str,
    process1_params: dict,
    process2_name: str,
    process2_params: dict,
):
    """Build a factored process with independent structure."""
    spec = [
        {"component_type": "hmm", "variants": [{"process_name": process1_name, "process_params": process1_params}]},
        {"component_type": "hmm", "variants": [{"process_name": process2_name, "process_params": process2_params}]},
    ]
    return build_factored_process_from_spec(structure_type="independent", spec=spec)


def compare_observation_distributions(hmm_kron, hmm_factored, rtol=1e-5):
    """Compare observation probability distributions from both HMMs."""
    # Get observation distributions from initial state
    dist_kron = hmm_kron.observation_probability_distribution(hmm_kron.initial_state)
    dist_factored = hmm_factored.observation_probability_distribution(hmm_factored.initial_state)

    assert dist_kron.shape == dist_factored.shape, (
        f"Shape mismatch: {dist_kron.shape} vs {dist_factored.shape}"
    )
    assert jnp.allclose(dist_kron, dist_factored, rtol=rtol), (
        f"Distribution mismatch: max diff = {jnp.max(jnp.abs(dist_kron - dist_factored))}"
    )


def compare_sequence_probabilities(hmm_kron, hmm_factored, num_sequences=10, seq_len=5, seed=42):
    """Compare sequence probabilities from both HMMs."""
    from simplexity.generative_processes.torch_generator import generate_data_batch

    # Generate sequences using the Kronecker HMM
    init_state = jnp.repeat(hmm_kron.initial_state[None, :], num_sequences, axis=0)
    key = jax.random.key(seed)
    _, inputs, _ = generate_data_batch(init_state, hmm_kron, num_sequences, seq_len, key)

    # Compare probabilities for each sequence
    for i in range(num_sequences):
        seq = jnp.array(inputs[i].tolist())

        prob_kron = hmm_kron.probability(seq)
        prob_factored = hmm_factored.probability(seq)

        assert jnp.allclose(prob_kron, prob_factored, rtol=1e-5), (
            f"Sequence {i} probability mismatch: {prob_kron} vs {prob_factored}"
        )


class TestKroneckerHMM:
    """Test Kronecker product HMM equivalence to factored process."""

    def test_two_mess3(self):
        """Test two mess3 factors."""
        params1 = {"x": 0.15, "a": 0.6}
        params2 = {"x": 0.25, "a": 0.5}

        hmm_kron = build_kronecker_hmm("mess3", params1, "mess3", params2)
        hmm_factored = build_factored_hmm("mess3", params1, "mess3", params2)

        # Check vocab sizes match
        assert hmm_kron.vocab_size == hmm_factored.vocab_size, (
            f"Vocab size mismatch: {hmm_kron.vocab_size} vs {hmm_factored.vocab_size}"
        )

        compare_observation_distributions(hmm_kron, hmm_factored)
        compare_sequence_probabilities(hmm_kron, hmm_factored)

    def test_mess3_and_z1r(self):
        """Test mess3 + Z1R factors."""
        params_mess3 = {"x": 0.15, "a": 0.6}
        params_z1r = {}  # Z1R takes no parameters

        hmm_kron = build_kronecker_hmm("mess3", params_mess3, "Z1R", params_z1r)
        hmm_factored = build_factored_hmm("mess3", params_mess3, "Z1R", params_z1r)

        assert hmm_kron.vocab_size == hmm_factored.vocab_size

        compare_observation_distributions(hmm_kron, hmm_factored)
        compare_sequence_probabilities(hmm_kron, hmm_factored)

    def test_mess3_and_bloch_walk(self):
        """Test mess3 + bloch_walk factors."""
        params_mess3 = {"x": 0.15, "a": 0.6}
        params_bloch = {}

        hmm_kron = build_kronecker_hmm("mess3", params_mess3, "bloch_walk", params_bloch)
        hmm_factored = build_factored_hmm("mess3", params_mess3, "bloch_walk", params_bloch)

        assert hmm_kron.vocab_size == hmm_factored.vocab_size

        compare_observation_distributions(hmm_kron, hmm_factored)
        compare_sequence_probabilities(hmm_kron, hmm_factored)


if __name__ == "__main__":
    test = TestKroneckerHMM()

    print("Test 1: Two mess3 factors...")
    test.test_two_mess3()
    print("  PASSED")

    print("Test 2: mess3 + Z1R...")
    test.test_mess3_and_z1r()
    print("  PASSED")

    print("Test 3: mess3 + bloch_walk...")
    test.test_mess3_and_bloch_walk()
    print("  PASSED")

    print()
    print("All tests passed!")
