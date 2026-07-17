"""Stage 2: Analyze the AFP process properties.

Computes prompt diversity, completion diversity, completion polarization,
and theoretical distinguishability metrics -- all analytically or via sampling,
no model needed.
"""

import itertools
import math
from collections import Counter
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from simplexity.generative_processes.torch_generator import generate_data_batch

from em_pipeline.config import (
    ProcessResult,
    AnalysisResult,
    save_pickle,
    load_pickle,
    to_np_idx,
)


def run(process: ProcessResult, n_completion_samples: int = 10000, seed: int = 0) -> AnalysisResult:
    """Compute all analysis metrics for the given process."""
    info = process.info
    v_p = info["v_p"]
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])
    alpha = info["alpha"]
    beta = info["beta"]

    # --- 1. Prompt diversity ---
    prompt_to_pi_a, prompt_to_belief = _compute_prompt_diversity(
        process.prompt_hmm, v_p, prompt_len, sector_a_idx,
    )
    pi_a_values = list(prompt_to_pi_a.values())
    n_unique_beliefs = len(set(round(v, 6) for v in pi_a_values))

    # --- 2. Completion diversity ---
    completion_diversity = _compute_completion_diversity(
        process.comp_hmm, sector_a_idx, sector_b_idx,
        comp_len, v_p, n_samples=min(n_completion_samples, 5000),
        base_seed=seed,
    )

    # --- 3. Completion polarization ---
    p_collapse_a_good, p_collapse_a_bad = _compute_analytical_collapse(beta, comp_len)
    polarization_curve = _compute_polarization_curve(
        process.comp_hmm, sector_a_idx, comp_len, v_p,
        n_samples=min(n_completion_samples, 5000),
        base_seed=seed,
    )

    # --- 4. Distinguishability ---
    analytical_kl_1 = _analytical_kl_per_token(alpha, beta)
    analytical_kl_by_length = [analytical_kl_1 * L for L in range(1, comp_len + 1)]
    empirical_kl_by_length = _empirical_kl_by_length(
        process.comp_hmm, sector_a_idx, sector_b_idx,
        comp_len, n_samples=n_completion_samples,
    )

    # --- 5. Within-sector belief diversity (for prompt-neutral variants) ---
    within_kwargs = {}
    if info.get("prompt_neutral", False):
        within_results = _compute_within_sector_diversity(
            prompt_to_belief, sector_a_idx, sector_b_idx,
        )
        within_kwargs = within_results

    # --- 6. Sector polarization summary (for prompt-neutral variants) ---
    sector_polarization = None
    if info.get("prompt_neutral", False):
        sector_polarization = _compute_sector_polarization_summary(
            prompt_to_belief, process.comp_hmm, sector_a_idx,
            comp_len, base_seed=seed,
        )

    return AnalysisResult(
        n_total_prompts=v_p ** prompt_len,
        n_unique_belief_states=n_unique_beliefs,
        prompt_to_pi_a={str(k): v for k, v in prompt_to_pi_a.items()},
        pi_a_min=min(pi_a_values) if pi_a_values else 0.0,
        pi_a_max=max(pi_a_values) if pi_a_values else 0.0,
        completion_diversity=completion_diversity,
        p_collapse_a_good=p_collapse_a_good,
        p_collapse_a_bad=p_collapse_a_bad,
        polarization_curve=polarization_curve,
        analytical_kl_per_token=analytical_kl_1,
        analytical_kl_by_length=analytical_kl_by_length,
        empirical_kl_by_length=empirical_kl_by_length,
        sector_polarization=sector_polarization,
        **within_kwargs,
    )


def save(result: AnalysisResult, path: Path) -> None:
    """Save analysis metrics to pickle."""
    save_pickle(result, Path(path) / "analysis.pkl")


def load(path: Path) -> AnalysisResult:
    """Load analysis metrics from pickle."""
    return load_pickle(Path(path) / "analysis.pkl")


def print_report(result: AnalysisResult) -> None:
    """Print a human-readable summary of the analysis."""
    print("\n--- Stage 2: Process Analysis Report ---\n")

    print(f"Prompt Diversity:")
    print(f"  Total prompts:         {result.n_total_prompts}")
    print(f"  Unique belief states:  {result.n_unique_belief_states}")
    print(f"  pi_A range:            [{result.pi_a_min:.4f}, {result.pi_a_max:.4f}]")

    print(f"\nCompletion Diversity:")
    for pi_a_str, stats in result.completion_diversity.items():
        print(f"  pi_A={pi_a_str}: {stats['n_unique']} unique / {stats['n_sampled']} sampled, "
              f"entropy={stats['entropy']:.3f} nats")

    print(f"\nCompletion Polarization:")
    print(f"  P(sector A) after all-good (from 50/50): {result.p_collapse_a_good:.4f}")
    print(f"  P(sector A) after all-bad  (from 50/50): {result.p_collapse_a_bad:.4f}")

    print(f"\nTheoretical Distinguishability:")
    print(f"  Analytical tag-based D_KL per token:  {result.analytical_kl_per_token:.4f} nats")
    print(f"  D_KL by completion length (analytical vs empirical):")
    for L, (a_kl, e_kl) in enumerate(
        zip(result.analytical_kl_by_length, result.empirical_kl_by_length), 1
    ):
        print(f"    L={L}: analytical={a_kl:.4f}, empirical={e_kl:.4f}")

    if result.n_unique_within_beliefs is not None:
        print(f"\nWithin-Sector Belief Diversity (prompt-neutral):")
        print(f"  Unique (μ_G, μ_B) pairs: {result.n_unique_within_beliefs}")
        print(f"  μ_G entropy:             {result.within_g_entropy:.3f} nats")
        print(f"  μ_B entropy:             {result.within_b_entropy:.3f} nats")

    if result.sector_polarization is not None:
        sp = result.sector_polarization
        print(f"\nSector Polarization Summary (post-completion):")
        print(f"  Prompts sampled:         {sp['n_prompts_sampled']}")
        print(f"  Completions/prompt:      {sp['n_completions_per_prompt']}")
        print(f"  Total sampled:           {sp['n_total_sampled']}")
        print(f"  Final π_A mean:          {sp['final_pi_a_mean']:.4f}")
        print(f"  Final π_A std:           {sp['final_pi_a_std']:.4f}")
        print(f"  Frac polarized (>0.9/<0.1): {sp['frac_polarized_09']:.4f}")
        print(f"  Unique completions:      {sp['n_unique_completions']}")

    print()


# =============================================================================
# PROMPT DIVERSITY
# =============================================================================

def _compute_prompt_diversity(
    prompt_hmm, v_p: int, prompt_len: int, sector_a_idx: np.ndarray,
) -> tuple[dict[tuple, float], dict[tuple, np.ndarray]]:
    """Enumerate all V_p^P prompts, compute pi_A for each.

    Returns:
        prompt_to_pi_a: prompt_tuple -> scalar pi_A
        prompt_to_belief: prompt_tuple -> full belief state array
    """
    T = np.array(prompt_hmm.transition_matrices)  # (V_p, S, S)
    init = np.array(prompt_hmm.initial_state)      # (S,)

    prompt_to_pi_a = {}
    prompt_to_belief = {}

    for prompt_seq in itertools.product(range(v_p), repeat=prompt_len):
        state = init.copy()
        for tok in prompt_seq:
            state = state @ T[tok]
            s = state.sum()
            if s > 0:
                state /= s

        pi_a = float(state[sector_a_idx].sum())
        prompt_to_pi_a[prompt_seq] = pi_a
        prompt_to_belief[prompt_seq] = state

    return prompt_to_pi_a, prompt_to_belief


# =============================================================================
# COMPLETION DIVERSITY
# =============================================================================

def _compute_completion_diversity(
    comp_hmm, sector_a_idx: np.ndarray, sector_b_idx: np.ndarray,
    comp_len: int, v_p: int, n_samples: int = 5000,
    base_seed: int = 0,
) -> dict[str, dict]:
    """Sample completions from representative starting states and measure diversity."""
    T_comp = np.array(comp_hmm.transition_matrices)
    num_states = T_comp.shape[1]
    results = {}

    # Test several pi_A values
    for i, pi_a_target in enumerate([0.1, 0.3, 0.5, 0.7, 0.9]):
        # Construct a starting state with this pi_A
        init = np.zeros(num_states)
        init[sector_a_idx] = pi_a_target / len(sector_a_idx)
        init[sector_b_idx] = (1 - pi_a_target) / len(sector_b_idx)

        # Sample completions
        init_batch = jnp.repeat(jnp.array(init)[None, :], n_samples, axis=0)
        key = jax.random.key(base_seed + int(pi_a_target * 1000))
        _, inputs, _ = generate_data_batch(
            init_batch, comp_hmm, n_samples, comp_len, key,
        )
        sequences = inputs.numpy()  # (n_samples, comp_len - 1)

        # Count unique sequences
        seq_tuples = [tuple(row) for row in sequences]
        counts = Counter(seq_tuples)
        n_unique = len(counts)

        # Shannon entropy of empirical distribution
        probs = np.array(list(counts.values())) / n_samples
        entropy = -np.sum(probs * np.log(probs + 1e-30))

        results[f"{pi_a_target:.1f}"] = {
            "n_unique": n_unique,
            "n_sampled": n_samples,
            "entropy": float(entropy),
        }

    return results


# =============================================================================
# COMPLETION POLARIZATION
# =============================================================================

def _compute_analytical_collapse(beta: float, comp_len: int) -> tuple[float, float]:
    """Analytical collapse probabilities from 50/50 starting point.

    Returns (pi_A_after_all_good, pi_A_after_all_bad).
    """
    if beta <= 0 or beta >= 1:
        return (0.5, 0.5)

    ratio = (1.0 / beta) ** comp_len
    p_good = ratio / (1.0 + ratio)
    p_bad = 1.0 / (1.0 + ratio)
    return float(p_good), float(p_bad)


def _compute_polarization_curve(
    comp_hmm, sector_a_idx: np.ndarray, comp_len: int, v_p: int,
    n_samples: int = 5000, threshold: float = 0.9,
    base_seed: int = 0,
) -> dict[str, float]:
    """For each starting pi_A, compute P(collapse to sector A after completions)."""
    T_comp = np.array(comp_hmm.transition_matrices)
    num_states = T_comp.shape[1]
    V_comp = T_comp.shape[0]
    curve = {}

    for pi_a_start in np.linspace(0.1, 0.9, 9):
        init = np.zeros(num_states)
        init[sector_a_idx] = pi_a_start / len(sector_a_idx)
        # Distribute rest over non-sector-A states that have mass
        other_idx = [i for i in range(num_states) if i not in sector_a_idx]
        if other_idx:
            init[other_idx] = (1 - pi_a_start) / len(other_idx)

        init_batch = jnp.repeat(jnp.array(init)[None, :], n_samples, axis=0)
        key = jax.random.key(base_seed + int(pi_a_start * 1000) + 7)

        # Generate completions and get final beliefs
        final_states, _, _ = generate_data_batch(
            init_batch, comp_hmm, n_samples, comp_len, key,
        )
        # final_states is the belief after the full completion
        final_beliefs = np.array(final_states)  # (n_samples, num_states)
        final_pi_a = final_beliefs[:, sector_a_idx].sum(axis=1)

        frac_collapse_a = float((final_pi_a > threshold).mean())
        curve[f"{pi_a_start:.2f}"] = frac_collapse_a

    return curve


# =============================================================================
# DISTINGUISHABILITY
# =============================================================================

def _analytical_kl_per_token(alpha: float, beta: float) -> float:
    """Tag-based D_KL(A||B) per completion token.

    D_KL = [(alpha - beta) / (alpha + beta)] * log(alpha / beta)
    """
    if alpha <= 0 or beta <= 0 or alpha == beta:
        return 0.0
    return ((alpha - beta) / (alpha + beta)) * math.log(alpha / beta)


def _empirical_kl_by_length(
    comp_hmm, sector_a_idx: np.ndarray, sector_b_idx: np.ndarray,
    max_len: int, n_samples: int = 10000,
) -> list[float]:
    """Monte Carlo D_KL(sector_A || sector_B) for completion lengths 1..max_len.

    Uses the HMM forward algorithm to compute exact log-likelihoods for
    sampled sequences, then estimates KL as:
        D_KL(A||B) = E_{x~P_A}[log P_A(x) - log P_B(x)]

    This avoids the curse of dimensionality that plagues sequence-counting
    approaches, giving O(1/sqrt(N)) error regardless of sequence length.
    """
    T_comp = np.array(comp_hmm.transition_matrices)  # (V, S, S)
    num_states = T_comp.shape[1]

    # Construct pure-sector initial states
    init_a = np.zeros(num_states)
    init_a[sector_a_idx] = 1.0 / len(sector_a_idx)

    init_b = np.zeros(num_states)
    init_b[sector_b_idx] = 1.0 / len(sector_b_idx)

    # Generate completions from sector A
    init_a_batch = jnp.repeat(jnp.array(init_a)[None, :], n_samples, axis=0)
    key = jax.random.key(42)
    _, inputs_a, labels_a = generate_data_batch(
        init_a_batch, comp_hmm, n_samples, max_len + 1, key,
    )

    # Reconstruct full token sequences
    import torch
    tokens = torch.cat([inputs_a[:, 0:1], labels_a], dim=1).numpy()  # (N, max_len+1)

    # Forward algorithm: compute log P(x_1..x_L | init) incrementally
    # α_t(s) = P(x_1..x_t, S_t=s | init), updated as α_{t+1} = α_t @ T[x_{t+1}]
    # log P(x_1..x_L) = log sum_s α_L(s)
    # We normalize at each step for numerical stability and accumulate log-normalizers.
    alpha_a = np.tile(init_a, (n_samples, 1))  # (N, S)
    alpha_b = np.tile(init_b, (n_samples, 1))  # (N, S)
    log_norm_a = np.zeros(n_samples)
    log_norm_b = np.zeros(n_samples)

    kl_by_length = []
    for t in range(max_len):
        obs = tokens[:, t]  # (N,) token at position t

        # Update forward variables: α' = α @ T[obs]
        # T_comp[obs[i]] is (S, S), so α'[i] = α[i] @ T_comp[obs[i]]
        T_obs = T_comp[obs]  # (N, S, S)
        alpha_a = np.einsum("ns,nsp->np", alpha_a, T_obs)
        alpha_b = np.einsum("ns,nsp->np", alpha_b, T_obs)

        # Normalize for stability
        z_a = alpha_a.sum(axis=1, keepdims=True)
        z_b = alpha_b.sum(axis=1, keepdims=True)
        log_norm_a += np.log(z_a.squeeze() + 1e-300)
        log_norm_b += np.log(z_b.squeeze() + 1e-300)
        alpha_a /= z_a + 1e-300
        alpha_b /= z_b + 1e-300

        # log P(x_1..x_{t+1} | init) = log_norm + log(sum_s alpha_s)
        # After normalization, sum_s alpha_s = 1, so it's just log_norm
        log_p_a = log_norm_a  # (N,)
        log_p_b = log_norm_b  # (N,)

        # D_KL(A||B) at length t+1 = E_{x~A}[log P_A(x) - log P_B(x)]
        kl = float(np.mean(log_p_a - log_p_b))
        kl_by_length.append(kl)

    return kl_by_length


# =============================================================================
# WITHIN-SECTOR BELIEF DIVERSITY (prompt-neutral variants)
# =============================================================================

def _compute_within_sector_diversity(
    prompt_to_belief: dict[tuple, np.ndarray],
    sector_g_idx: np.ndarray,
    sector_b_idx: np.ndarray,
) -> dict:
    """Compute within-sector belief diversity across prompts.

    For prompt-neutral variants (like leaky_reset), pi_G/pi_B is constant
    so the interesting diversity is within-sector: different prompts produce
    different μ_G and μ_B distributions.

    Returns dict with keys matching AnalysisResult within-sector fields.
    """
    prompt_to_within = {}
    mu_g_list = []
    mu_b_list = []

    for prompt_seq, belief in prompt_to_belief.items():
        raw_g = belief[sector_g_idx]
        raw_b = belief[sector_b_idx]

        # Normalize to within-sector distributions
        sum_g = raw_g.sum()
        sum_b = raw_b.sum()
        mu_g = (raw_g / sum_g) if sum_g > 0 else raw_g
        mu_b = (raw_b / sum_b) if sum_b > 0 else raw_b

        mu_g_list.append(mu_g)
        mu_b_list.append(mu_b)
        prompt_to_within[str(prompt_seq)] = {
            "mu_g": mu_g.tolist(),
            "mu_b": mu_b.tolist(),
        }

    # Count unique (μ_G, μ_B) pairs (rounded to 6 decimals)
    rounded_pairs = set()
    for mu_g, mu_b in zip(mu_g_list, mu_b_list):
        key = (tuple(round(float(x), 6) for x in mu_g),
               tuple(round(float(x), 6) for x in mu_b))
        rounded_pairs.add(key)
    n_unique = len(rounded_pairs)

    # Entropy of within-sector beliefs across prompts
    # Use discretized bins to estimate distribution entropy
    def _belief_entropy(mu_list):
        """Entropy of the empirical distribution of belief vectors."""
        rounded = [tuple(round(float(x), 6) for x in mu) for mu in mu_list]
        counts = Counter(rounded)
        n = len(rounded)
        probs = np.array(list(counts.values())) / n
        return float(-np.sum(probs * np.log(probs + 1e-30)))

    return {
        "n_unique_within_beliefs": n_unique,
        "within_g_entropy": _belief_entropy(mu_g_list),
        "within_b_entropy": _belief_entropy(mu_b_list),
        "prompt_to_within_belief": prompt_to_within,
    }


# =============================================================================
# SECTOR POLARIZATION SUMMARY (post-completion π_A distribution)
# =============================================================================

def _compute_sector_polarization_summary(
    prompt_to_belief: dict[tuple, np.ndarray],
    comp_hmm,
    sector_a_idx: np.ndarray,
    comp_len: int,
    max_prompts: int = 200,
    n_completions_per_prompt: int = 200,
    base_seed: int = 0,
) -> dict:
    """Compute post-completion π_A distribution and unique completion count.

    For each prompt (or a sample of prompts), generates completions from the
    actual post-prompt belief state and measures:
    - Final π_A after full completion (for histogram)
    - Distinct completion token sequences across all prompts

    Returns dict with keys matching AnalysisResult.sector_polarization docs.
    """
    prompts = list(prompt_to_belief.keys())
    rng = np.random.RandomState(base_seed)
    if len(prompts) > max_prompts:
        indices = rng.choice(len(prompts), max_prompts, replace=False)
        prompts = [prompts[i] for i in indices]

    all_final_pi_a = []
    completion_counter = Counter()

    for i, prompt_seq in enumerate(prompts):
        belief = prompt_to_belief[prompt_seq]
        init_batch = jnp.repeat(jnp.array(belief)[None, :], n_completions_per_prompt, axis=0)
        key = jax.random.key(base_seed + i * 7 + 31)

        final_states, inputs, _ = generate_data_batch(
            init_batch, comp_hmm, n_completions_per_prompt, comp_len, key,
        )

        # Final π_A from belief states after completion
        final_beliefs = np.array(final_states)
        final_pi_a = final_beliefs[:, sector_a_idx].sum(axis=1)
        all_final_pi_a.extend(final_pi_a.tolist())

        # Collect completion sequences (inputs has shape (N, comp_len - 1))
        sequences = inputs.numpy()
        for row in sequences:
            completion_counter[tuple(row)] += 1

    all_final_pi_a_arr = np.array(all_final_pi_a)

    return {
        "final_pi_a_all": all_final_pi_a,
        "final_pi_a_mean": float(all_final_pi_a_arr.mean()),
        "final_pi_a_std": float(all_final_pi_a_arr.std()),
        "frac_polarized_09": float(
            ((all_final_pi_a_arr > 0.9) | (all_final_pi_a_arr < 0.1)).mean()
        ),
        "n_unique_completions": len(completion_counter),
        "n_total_sampled": len(all_final_pi_a),
        "n_prompts_sampled": len(prompts),
        "n_completions_per_prompt": n_completions_per_prompt,
    }
