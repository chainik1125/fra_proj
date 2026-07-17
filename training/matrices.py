import jax.numpy as jnp
from jax import Array

from simplexity.generative_processes.builder import build_hidden_markov_model
from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel
from simplexity.generative_processes.transition_matrices import HMM_MATRIX_FUNCTIONS


def left_right_mix(a: float = 0.0, b: float = 0.0) -> Array:
    """
    Parameters:
    a (float): L/R Asymmetry parameter.
    b (float): Leak parameter for Symbol 1 (A->C).
    """
    # Symbol 0: Left Cycle (A->B->C->A)
    val0 = 0.5 + a
    T0 = jnp.array([
        [0.0, val0, 0.0],  # A -> B
        [0.0, 0.0, val0],  # B -> C
        [val0, 0.0, 0.0]   # C -> A
    ])

    # Symbol 1: Right Cycle (A->C->B->A)
    val1 = 0.44 - a
    val1_leak = 0.44 - a - b

    T1 = jnp.array([
        [0.0, 0.0, val1_leak],  # A -> C
        [val1, 0.0, 0.0],       # B -> A
        [0.0, val1, 0.0]        # C -> B
    ])

    # Symbol 2: uniform noise, with compensation for leak
    val2_bc = 0.02
    val2_a = 0.02 + (b / 3.0)

    T2 = jnp.array([
        [val2_a, val2_a, val2_a],
        [val2_bc, val2_bc, val2_bc],
        [val2_bc, val2_bc, val2_bc]
    ])

    return jnp.array([T0, T1, T2])


# Register with simplexity
HMM_MATRIX_FUNCTIONS["left_right_mix"] = left_right_mix

#define Z1R
def Z1R() -> Array:
    """
    1st order Z matrix with rightward bias.
    """
    T0 = jnp.array([
        [0.0, 1.0, 0.0],  # S0 -> S1 certainly if emit 0
        [0.0, 0.0, 0.0],  # S0 never emits 0
        [1/2, 0.0, 0.0]   # S0 <- SR with prob 1/2
    ])

    T1 = jnp.array([
        [0.0, 0.0, 0.0],  # S0 never emits 1
        [0, 0.0, 1],  # S1 -> certainly emits 1 and transitions to SR
        [1/2, 0, 0.0]   # C -> B
    ])

    return jnp.array([T0, T1])


# Register with simplexity
HMM_MATRIX_FUNCTIONS["Z1R"] = Z1R


def bloch_walk(p: float = 0.3) -> Array:
    """
    Simple 2-state walk with symmetric switching probability.

    Emits 0 when moving/staying in state 0, emits 1 when moving/staying in state 1.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")

    T0 = jnp.array([
        [1.0 - p, p],
        [0.0, 0.0],
    ])
    T1 = jnp.array([
        [0.0, 0.0],
        [p, 1.0 - p],
    ])

    return jnp.array([T0, T1])


if "bloch_walk" not in HMM_MATRIX_FUNCTIONS:
    HMM_MATRIX_FUNCTIONS["bloch_walk"] = bloch_walk


def kronecker_transition_matrices(T1: Array, T2: Array) -> Array:
    """
    Combine two HMM transition matrices via Kronecker product.

    For independent factors, the combined transition for observation (v1, v2)
    is the Kronecker product of T1[v1] and T2[v2].

    Args:
        T1: Transition matrices for HMM1, shape [V1, S1, S1]
        T2: Transition matrices for HMM2, shape [V2, S2, S2]

    Returns:
        Combined transition matrices, shape [V1*V2, S1*S2, S1*S2]
    """
    V1, S1, _ = T1.shape
    V2, S2, _ = T2.shape

    # For each (v1, v2) pair, compute Kronecker product of T1[v1] and T2[v2]
    # Result shape: [V1, V2, S1*S2, S1*S2]
    combined = jnp.kron(T1[:, None, :, :], T2[None, :, :, :])

    # Reshape to [V1*V2, S1*S2, S1*S2]
    return combined.reshape(V1 * V2, S1 * S2, S1 * S2)


def kronecker_initial_state(pi1: Array, pi2: Array) -> Array:
    """
    Combine two initial state distributions via outer product.

    Args:
        pi1: Initial state for HMM1, shape [S1]
        pi2: Initial state for HMM2, shape [S2]

    Returns:
        Combined initial state, shape [S1*S2]
    """
    return jnp.outer(pi1, pi2).flatten()


import jax.numpy as jnp
from jax import Array

def Z1R_partition_preserving_joint_ABCD() -> Array:
    """
    Partition-preserving 2-process GHMM built from a Z1R-like single-process
    that preserves the split {S0} ⊕ {S1,SR}.

    Hidden-state ordering per process: [S0, S1, SR]
    Joint hidden-state ordering (lexicographic): idx = 3*iA + iB
      0:(S0,S0) 1:(S0,S1) 2:(S0,SR)
      3:(S1,S0) 4:(S1,S1) 5:(S1,SR)
      6:(SR,S0) 7:(SR,S1) 8:(SR,SR)

    Observed tokens encode emission pairs:
      A=(0,0), B=(0,1), C=(1,0), D=(1,1)

    Returns:
      jnp.array([T_A, T_B, T_C, T_D]) with shape (4, 9, 9),
      where T_token[s, s'] = Q(s', token | s).
    """

    # Single-process partition-preserving transfer matrices (3x3 each)
    T0p = jnp.array([
        [1.0, 0.0, 0.0],  # S0 --0--> S0 (prob 1)
        [0.0, 0.0, 0.0],  # S1 never emits 0
        [0.0, 0.5, 0.0],  # SR --0--> S1 (prob 1/2)
    ])

    T1p = jnp.array([
        [0.0, 0.0, 0.0],  # S0 never emits 1
        [0.0, 0.0, 1.0],  # S1 --1--> SR (prob 1)
        [0.0, 0.5, 0.0],  # SR --1--> S1 (prob 1/2)
    ])

    # Joint token matrices (9x9 each)
    T_A = jnp.kron(T0p, T0p)  # (0,0)
    T_B = jnp.kron(T0p, T1p)  # (0,1)
    T_C = jnp.kron(T1p, T0p)  # (1,0)
    T_D = jnp.kron(T1p, T1p)  # (1,1)

    return jnp.array([T_A, T_B, T_C, T_D])

HMM_MATRIX_FUNCTIONS["ENT_Z1R"] = Z1R_partition_preserving_joint_ABCD()


def build_kronecker_hmm(
    process1_name: str,
    process1_params: dict,
    process2_name: str,
    process2_params: dict,
) -> HiddenMarkovModel:
    """Build a combined HMM using the Kronecker product of two HMMs."""
    hmm1 = build_hidden_markov_model(process1_name, process1_params)
    hmm2 = build_hidden_markov_model(process2_name, process2_params)

    combined_T = kronecker_transition_matrices(hmm1.transition_matrices, hmm2.transition_matrices)
    combined_init = kronecker_initial_state(hmm1.initial_state, hmm2.initial_state)

    return HiddenMarkovModel(transition_matrices=combined_T, initial_state=combined_init)


# =============================================================================
# BIASED COIN
# =============================================================================
def biased_coin(p: float = 0.5) -> Array:
    """
    Biased coin: 1 hidden state, vocab size 2.

    Emits 0 with probability p, 1 with probability 1-p.

    Args:
        p: Probability of emitting 0.
    """
    if not 0.0 <= p <= 1.0:
        raise ValueError("p must be in [0, 1]")
    T0 = jnp.array([[p]])
    T1 = jnp.array([[1.0 - p]])
    return jnp.array([T0, T1])


HMM_MATRIX_FUNCTIONS["biased_coin"] = biased_coin


# =============================================================================
# DIRECT SUM UTILITIES
# =============================================================================
def direct_sum_transition_matrices(T_list: list[Array]) -> Array:
    """
    Block-diagonal combination of transition matrices via direct sum.

    Each element of T_list has shape [V, Si, Si]. All must share the same V.
    Returns shape [V, sum(Si), sum(Si)].
    """
    V = T_list[0].shape[0]
    for T in T_list:
        if T.shape[0] != V:
            raise ValueError(f"All transition matrices must have the same vocab size V, got {T.shape[0]} vs {V}")

    sizes = [T.shape[1] for T in T_list]
    S_total = sum(sizes)

    combined = jnp.zeros((V, S_total, S_total))
    offset = 0
    for T, Si in zip(T_list, sizes):
        combined = combined.at[:, offset:offset + Si, offset:offset + Si].set(T)
        offset += Si

    return combined


def direct_sum_initial_state(
    pi_list: list[Array],
    weights: list[float] | None = None,
) -> Array:
    """
    Concatenate and renormalize initial state distributions for a direct sum HMM.

    Args:
        pi_list: List of initial state vectors, each shape [Si].
        weights: Mixture weights q_i for each component. If None, uniform.
            Each component's initial state is scaled by its weight before
            concatenation and renormalization.

    Returns:
        Concatenated and renormalized initial state, shape [sum(Si)].
    """
    if weights is not None:
        if len(weights) != len(pi_list):
            raise ValueError(f"weights length {len(weights)} != pi_list length {len(pi_list)}")
        pi_list = [w * pi for w, pi in zip(weights, pi_list)]
    combined = jnp.concatenate(pi_list)
    return combined / combined.sum()


def build_direct_sum_hmm(
    process_configs: list[dict],
    weights: list[float] | None = None,
) -> HiddenMarkovModel:
    """
    Build a direct sum HMM from a list of process configs.

    Each config is {"name": str, "params": dict}.
    All processes must share the same vocab size.

    Args:
        process_configs: List of dicts with "name" and "params" keys.
        weights: Mixture weights q_i for each component. If None, uniform.

    Returns:
        Combined HiddenMarkovModel via block-diagonal direct sum.
    """
    hmms = [build_hidden_markov_model(cfg["name"], cfg["params"]) for cfg in process_configs]

    T_list = [hmm.transition_matrices for hmm in hmms]
    pi_list = [hmm.initial_state for hmm in hmms]

    combined_T = direct_sum_transition_matrices(T_list)
    combined_pi = direct_sum_initial_state(pi_list, weights)

    return HiddenMarkovModel(transition_matrices=combined_T, initial_state=combined_pi)


# =============================================================================
# AFP (Almost-Factored Process) — Z1R'×Z1R' with prompt/completion structure
# =============================================================================

# Joint hidden state indices (lexicographic 3×3, order S0,S1,SR):
#   0:(S0,S0)  1:(S0,S1)  2:(S0,SR)
#   3:(S1,S0)  4:(S1,S1)  5:(S1,SR)
#   6:(SR,S0)  7:(SR,S1)  8:(SR,SR)
SECTOR_A_IDX = jnp.array([0])              # V_11 = span{(S0,S0)}
SECTOR_B_IDX = jnp.array([4, 5, 7, 8])     # V_22 = span{(S1,S1),(S1,SR),(SR,S1),(SR,SR)}
NUM_JOINT_STATES = 9
V_C = 4  # base completion vocab: A=(0,0), B=(0,1), C=(1,0), D=(1,1)


def z1r_prime_single_factor(delta: float) -> Array:
    """Build the Z1R'(delta) single-factor transfer matrices.

    Args:
        delta: Leak parameter in (0, 1). S0 emits 1 with probability delta.

    Returns:
        Array of shape (2, 3, 3): [T_0^(delta), T_1^(delta)].
        State order: (S0, S1, SR).
    """
    T0 = jnp.array([
        [1.0 - delta, 0.0, 0.0],
        [0.0,         0.0, 0.0],
        [0.0,         0.5, 0.0],
    ])
    T1 = jnp.array([
        [delta, 0.0, 0.0],
        [0.0,  0.0, 1.0],
        [0.0,  0.5, 0.0],
    ])
    return jnp.array([T0, T1])


def z1r_prime_joint_base_symbols(delta: float) -> Array:
    """Build joint base-symbol transfer matrices T_A, T_B, T_C, T_D via Kronecker product.

    A=(0,0), B=(0,1), C=(1,0), D=(1,1).

    Args:
        delta: Leak parameter for Z1R'.

    Returns:
        Array of shape (4, 9, 9).
    """
    T = z1r_prime_single_factor(delta)
    T0, T1 = T[0], T[1]
    return jnp.array([
        jnp.kron(T0, T0),  # A = (0,0)
        jnp.kron(T0, T1),  # B = (0,1)
        jnp.kron(T1, T0),  # C = (1,0)
        jnp.kron(T1, T1),  # D = (1,1)
    ])


def afp_sector_restrict(
    base_matrices: Array,
    sector_idx: Array,
) -> tuple[Array, Array]:
    """Restrict base-symbol matrices to a macro-sector via projection.

    For each base symbol s, computes M(s) = P · T_s · P where P projects
    onto the sector's states.

    Args:
        base_matrices: Shape (V_c, 9, 9).
        sector_idx: 1D array of state indices in this sector.

    Returns:
        restricted: Shape (V_c, 9, 9) — the projected matrices (zeros outside sector).
        row_sums: Shape (V_c,) — mean row sum within the sector for each symbol.
    """
    V_c = base_matrices.shape[0]
    # Build projector (9x9 diagonal with 1s at sector indices)
    P = jnp.zeros(NUM_JOINT_STATES)
    P = P.at[sector_idx].set(1.0)
    P_mat = jnp.diag(P)

    restricted = jnp.einsum('ij,vjk,kl->vil', P_mat, base_matrices, P_mat)

    # Row sums within the sector
    sector_row_sums = restricted[:, sector_idx, :].sum(axis=-1)  # (V_c, |sector|)
    mean_row_sums = sector_row_sums.mean(axis=-1)  # (V_c,)

    return restricted, mean_row_sums


def _fill_dead_states(T: Array) -> Array:
    """Add uniform self-loops to dead states so the net matrix is row-stochastic.

    States in V_12 (indices 1, 2) and V_21 (indices 3, 6) have zero mass
    in the AFP construction. We add self-loops so the HMM validator passes.
    Since these states never carry probability mass, this has no effect on generation.
    """
    dead_idx = jnp.array([1, 2, 3, 6])
    V = T.shape[0]
    fill_val = 1.0 / V
    for s in dead_idx:
        T = T.at[:, s, s].set(fill_val)
    return T


def build_afp_completion_matrices(
    delta: float,
    beta: float,
    alpha: float = 1.0,
) -> Array:
    """Build the 2*V_c completion token transfer matrices.

    Tokens 0..V_c-1 are "good-tagged" (gA, gB, gC, gD):
        T(gs) = alpha * M_hat_A(s) + beta * M_hat_B(s)
    Tokens V_c..2*V_c-1 are "bad-tagged" (bA, bB, bC, bD):
        T(bs) = beta * M_hat_A(s) + alpha * M_hat_B(s)

    Normalization: m_s = 1 / (V_c * (alpha + beta)) so row sums = 1.

    Args:
        delta: Leak parameter for Z1R'.
        beta: Collapse rate parameter in (0, 1).
        alpha: Scale for dominant sector (default 1.0, WLOG).

    Returns:
        Array of shape (2*V_c, 9, 9) = (8, 9, 9).
    """
    base = z1r_prime_joint_base_symbols(delta)

    M_A, u = afp_sector_restrict(base, SECTOR_A_IDX)
    M_B, v = afp_sector_restrict(base, SECTOR_B_IDX)

    # Build raw (unnormalized) good/bad tagged matrices using sector projections.
    # M_A(s) and M_B(s) may not have constant row sums across states,
    # so we assemble first and normalize the net matrix to be row-stochastic.

    # Good-tagged: T(gs) = alpha * M_A(s) + beta * M_B(s)
    T_good = alpha * M_A + beta * M_B

    # Bad-tagged: T(bs) = beta * M_A(s) + alpha * M_B(s)
    T_bad = beta * M_A + alpha * M_B

    result = jnp.concatenate([T_good, T_bad], axis=0)  # (2*V_c, 9, 9)

    # Normalize so the net transition matrix is row-stochastic:
    # For each state i, divide all T(x)[i,:] by the net row sum Z_i.
    net = result.sum(axis=0)  # (9, 9)
    row_sums = net.sum(axis=1)  # (9,)
    row_sums_safe = jnp.where(row_sums > 0, row_sums, 1.0)
    result = result / row_sums_safe[None, :, None]

    # Fill dead states with self-loops
    result = _fill_dead_states(result)

    return result


def build_afp_prompt_matrices(
    c_a: Array,
    c_b: Array,
) -> Array:
    """Build prompt token transfer matrices (identity-only permutation case).

    Each prompt token p_k has transfer matrix block-diagonal on A + B:
        T(p_k)[0,0] = c_A_norm(k)          (sector A, dim 1)
        T(p_k)[i,i] = c_B_norm(k) for i in sector_B_idx  (sector B, dim 4, identity)

    c_A and c_B are normalized internally so each sums to 1 (row-stochastic).

    Args:
        c_a: Non-negative array of length V_p. Sector-A weights per prompt token.
        c_b: Non-negative array of length V_p. Sector-B weights per prompt token.

    Returns:
        Array of shape (V_p, 9, 9).
    """
    c_a = jnp.asarray(c_a, dtype=jnp.float32)
    c_b = jnp.asarray(c_b, dtype=jnp.float32)
    v_p = c_a.shape[0]

    # Normalize to probability distributions
    c_a_norm = c_a / c_a.sum()
    c_b_norm = c_b / c_b.sum()

    T_prompt = jnp.zeros((v_p, NUM_JOINT_STATES, NUM_JOINT_STATES))

    # Sector A: entry (0, 0)
    T_prompt = T_prompt.at[:, 0, 0].set(c_a_norm)

    # Sector B: diagonal entries at indices 4, 5, 7, 8 (identity permutation)
    for idx in [4, 5, 7, 8]:
        T_prompt = T_prompt.at[:, idx, idx].set(c_b_norm)

    # Fill dead states with self-loops
    T_prompt = _fill_dead_states(T_prompt)

    return T_prompt


def make_log_spaced_biases(v_p: int, bias_range: float = 2.0) -> tuple[Array, Array]:
    """Generate c_A, c_B arrays with log-spaced sector bias ratios.

    log(c_A(k) / c_B(k)) is linearly spaced from -bias_range to +bias_range.

    Args:
        v_p: Number of prompt tokens.
        bias_range: Range of log-ratios (symmetric around 0).

    Returns:
        (c_a, c_b): Arrays of shape (v_p,), each non-negative.
    """
    log_ratios = jnp.linspace(-bias_range, bias_range, v_p)
    # c_a proportional to exp(log_ratio/2), c_b proportional to exp(-log_ratio/2)
    c_a = jnp.exp(log_ratios / 2.0)
    c_b = jnp.exp(-log_ratios / 2.0)
    return c_a, c_b


def build_afp_initial_state(
    pi_a: float = 0.5,
) -> Array:
    """Build initial state for the AFP HMM.

    Places pi_a on (S0,S0) and distributes (1 - pi_a) uniformly over sector B.

    Args:
        pi_a: Initial probability mass on macro-sector A.

    Returns:
        Array of shape (9,).
    """
    state = jnp.zeros(NUM_JOINT_STATES)
    state = state.at[0].set(pi_a)
    pi_b_each = (1.0 - pi_a) / len(SECTOR_B_IDX)
    state = state.at[SECTOR_B_IDX].set(pi_b_each)
    return state


def build_afp_hmms(
    delta: float = 0.05,
    beta: float = 0.556,
    alpha: float = 1.0,
    c_a: Array | None = None,
    c_b: Array | None = None,
    v_p: int = 10,
    pi_a: float = 0.5,
    bias_range: float = 2.0,
) -> tuple[HiddenMarkovModel, HiddenMarkovModel, dict]:
    """Build prompt and completion HMMs for the AFP process.

    Args:
        delta: Z1R' leak parameter.
        beta: Completion collapse rate parameter.
        alpha: Completion dominant sector scale (default 1.0).
        c_a: Non-negative prompt sector-A weights, length v_p. If None, uses log-spaced.
        c_b: Non-negative prompt sector-B weights, length v_p. If None, uses log-spaced.
        v_p: Prompt vocabulary size (used only if c_a/c_b are None).
        pi_a: Initial macro-sector A probability.
        bias_range: Log-ratio range for default biases.

    Returns:
        (prompt_hmm, comp_hmm, info) where info contains metadata.
    """
    if c_a is None or c_b is None:
        c_a, c_b = make_log_spaced_biases(v_p, bias_range)
    else:
        c_a = jnp.asarray(c_a, dtype=jnp.float32)
        c_b = jnp.asarray(c_b, dtype=jnp.float32)
        v_p = c_a.shape[0]

    initial_state = build_afp_initial_state(pi_a)

    # Build prompt HMM
    T_prompt = build_afp_prompt_matrices(c_a, c_b)
    prompt_hmm = HiddenMarkovModel(transition_matrices=T_prompt, initial_state=initial_state)

    # Build completion HMM
    T_comp = build_afp_completion_matrices(delta, beta, alpha)
    comp_hmm = HiddenMarkovModel(transition_matrices=T_comp, initial_state=initial_state)

    info = {
        "v_p": v_p,
        "v_c": V_C,
        "total_vocab": v_p + 2 * V_C,
        "delta": delta,
        "beta": beta,
        "alpha": alpha,
        "c_a": c_a,
        "c_b": c_b,
        "sector_a_idx": SECTOR_A_IDX,
        "sector_b_idx": SECTOR_B_IDX,
    }
    return prompt_hmm, comp_hmm, info


def generate_afp_batch(
    prompt_hmm: HiddenMarkovModel,
    comp_hmm: HiddenMarkovModel,
    batch_size: int,
    prompt_len: int,
    comp_len: int,
    key: Array,
    v_p: int = 10,
    device=None,
):
    """Generate a batch of two-phase AFP sequences (prompt + completion).

    Generates prompt_len tokens from the prompt HMM, hands off the belief state
    to the completion HMM, generates comp_len tokens, and concatenates with
    completion token indices offset by v_p.

    Args:
        prompt_hmm: HMM for prompt phase.
        comp_hmm: HMM for completion phase.
        batch_size: Number of sequences.
        prompt_len: Number of prompt tokens per sequence.
        comp_len: Number of completion tokens per sequence.
        key: JAX PRNG key.
        v_p: Prompt vocab size (used to offset completion token indices).
        device: Torch device for output tensors.

    Returns:
        (inputs, labels): Torch tensors of shape (batch_size, prompt_len + comp_len - 1).
    """
    import jax
    import torch
    from simplexity.generative_processes.torch_generator import generate_data_batch

    key1, key2 = jax.random.split(key)

    # --- Prompt phase ---
    init_prompt = jnp.repeat(prompt_hmm.initial_state[None, :], batch_size, axis=0)
    prompt_states, prompt_inputs, prompt_labels = generate_data_batch(
        init_prompt, prompt_hmm, batch_size, prompt_len, key1, device=device,
    )
    # prompt_states is the belief state AFTER the full prompt sequence

    # --- Completion phase ---
    # Use the post-prompt belief states as initial states for the completion HMM
    comp_states, comp_inputs, comp_labels = generate_data_batch(
        prompt_states, comp_hmm, batch_size, comp_len, key2, device=device,
    )

    # Offset completion token indices into the combined vocabulary
    comp_inputs = comp_inputs + v_p
    comp_labels = comp_labels + v_p

    # Concatenate: the full sequence is [prompt | completion]
    # prompt_inputs has prompt_len - 1 columns (input side of prompt)
    # prompt_labels has prompt_len - 1 columns (label side of prompt)
    # We need the full token sequence then re-derive inputs/labels
    # Full tokens = prompt_labels column 0 is the 2nd prompt token
    # Actually: inputs = tokens[:-1], labels = tokens[1:]
    # So we need to reconstruct the full token sequence first

    # prompt_inputs[:, 0] = token_0, ..., prompt_inputs[:, P-2] = token_{P-2}
    # prompt_labels[:, P-2] = token_{P-1} (last prompt token)
    # comp_inputs[:, 0] = first comp token, etc.
    # Simplest: concatenate inputs and append last label
    last_prompt_label = prompt_labels[:, -1:]  # (batch, 1)
    full_inputs = torch.cat([prompt_inputs, last_prompt_label, comp_inputs], dim=1)
    full_labels = torch.cat([prompt_labels, comp_labels], dim=1)

    # full_inputs has prompt_len + comp_len - 1 columns
    # full_labels has prompt_len + comp_len - 2 columns... that's wrong

    # Let me just build the full token sequence and slice
    # full tokens = concat of all individual tokens
    # From prompt: tokens[0..P-1] are in prompt_inputs (cols 0..P-2) + prompt_labels[:, -1]
    # Actually generate_data_batch returns inputs=tokens[:,:-1], labels=tokens[:,1:]
    # So tokens[:,0] = inputs[:,0], tokens[:,1] = inputs[:,1] = labels[:,0], etc.
    # tokens[:,-1] = labels[:,-1]
    # Reconstruct: tokens = cat(inputs[:, 0:1], labels)

    prompt_tokens = torch.cat([prompt_inputs[:, 0:1], prompt_labels], dim=1)  # (batch, prompt_len)
    comp_tokens = torch.cat([comp_inputs[:, 0:1], comp_labels], dim=1)  # (batch, comp_len)

    full_tokens = torch.cat([prompt_tokens, comp_tokens], dim=1)  # (batch, prompt_len + comp_len)

    inputs = full_tokens[:, :-1]
    labels = full_tokens[:, 1:]

    return inputs, labels
