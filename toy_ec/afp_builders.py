"""AFP process variant builders for steering experiments.

Supports 3 completion variants:
- "z1r_afp": original Z1R'xZ1R' AFP completion (from training/matrices.py)
- "metastable4": 9-state completion with 4-state B metastability + state-distinguishable emissions
- "clustered_codebook": larger-state completion where sector B = K clusters x COMP_LEN phases,
  emitting a prompt-selectable codeword (many more distinct high-prob completions).
"""

import itertools

import jax.numpy as jnp
import numpy as np

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel

# Import 9-state builders from training/matrices.py
import sys
from pathlib import Path

# Anchored on this file's own directory: toy_ec/ is a self-contained vendored
# tree (em_pipeline/, training/, afp_builders.py) extracted from dmitry/tutorial.
_repo_root = Path(__file__).resolve().parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))
if str(_repo_root / "training") not in sys.path:
    sys.path.insert(0, str(_repo_root / "training"))

from matrices import (
    build_afp_completion_matrices,
    build_afp_prompt_matrices,
    make_log_spaced_biases,
    SECTOR_A_IDX as _SECTOR_A_IDX_9,
    SECTOR_B_IDX as _SECTOR_B_IDX_9,
    NUM_JOINT_STATES as _NUM_JOINT_STATES_9,
)

# ── Default AFP process parameters ─────────────────────────────
# These match the notebook defaults; callers can override via function args.
_DEFAULT_V_P = 10
_DEFAULT_V_C = 4
_DEFAULT_PI_A = 0.5

# ── Prompt knobs (9-state variants) ────────────────────────────
USE_PROMPT_PERMUTATIONS = True
USE_PROMPT_REWEIGHTING = False
USE_NONUNIFORM_INIT_B = True
INIT_B_WEIGHTS_4 = jnp.array([0.7, 0.1, 0.1, 0.1], dtype=jnp.float32)

# ── Metastable4 completion parameters ──────────────────────────
B_CLUSTER_EPS = 0.02
B_WITHIN_RHO = 0.20

B_EMISSIONS_4 = jnp.array(
    [
        [0.05, 0.05, 0.45, 0.45],
        [0.05, 0.05, 0.55, 0.35],
        [0.45, 0.45, 0.05, 0.05],
        [0.35, 0.55, 0.05, 0.05],
    ],
    dtype=jnp.float32,
)

# ── Clustered codebook parameters ──────────────────────────────
B_NUM_CLUSTERS = 32
B_LEAK_EPS = 0.01
PROMPT_SIGNATURE_SIZE = 3
PROMPT_SIGNATURE_MASS = 0.995
CODEBOOK_STYLE = "base4"  # "base4" | "random"
CODEBOOK_SEED = 0


# ── Shared helpers ─────────────────────────────────────────────

def _fill_dead_states_9(T):
    """Add uniform self-loops to dead states in the 9-state AFP."""
    dead_idx = jnp.array([1, 2, 3, 6])
    V = T.shape[0]
    fill_val = 1.0 / V
    for s in dead_idx:
        T = T.at[:, s, s].set(fill_val)
    return T


def _normalize_net_row_stochastic(T):
    """Normalize so the net transition matrix (sum over tokens) is row-stochastic."""
    net = T.sum(axis=0)
    row_sums = net.sum(axis=1)
    row_sums_safe = jnp.where(row_sums > 0, row_sums, 1.0)
    return T / row_sums_safe[None, :, None]


def _fill_zero_rows_with_self_loops(T):
    """Ensure every state has nonzero net outgoing mass (HMM validator).

    For any state i with sum_x sum_j T[x,i,j] == 0, set T[x,i,i] = 1/V for all tokens x.
    """
    net = T.sum(axis=0)
    row_sums = net.sum(axis=1)
    zero_idx = np.where(np.array(row_sums) == 0)[0]
    if zero_idx.size == 0:
        return T
    V = T.shape[0]
    for s in zero_idx.tolist():
        T = T.at[:, s, s].set(1.0 / V)
    return T


# ── Prompt matrices: 9-state with permutations/reweighting ─────

def _build_prompt_b_reweight(v_p, strength=0.5):
    """Build diagonal reweighting matrices for sector B prompt mixing."""
    base = jnp.array([1.0 + strength, 1.0, 1.0 - strength, 1.0], dtype=jnp.float32)
    Ds = []
    for k in range(v_p):
        v = jnp.roll(base, k % 4)
        v = v / v.sum()
        Ds.append(jnp.diag(v))
    return Ds


def build_afp_prompt_matrices_perm(c_a, c_b, sector_b_idx, num_states,
                                   use_permutations=True, perm_mats=None,
                                   use_reweighting=True):
    """Build 9-state prompt matrices with optional permutations and reweighting."""
    if not use_permutations and not use_reweighting:
        return build_afp_prompt_matrices(c_a, c_b)

    c_a = jnp.asarray(c_a, dtype=jnp.float32)
    c_b = jnp.asarray(c_b, dtype=jnp.float32)
    v_p = c_a.shape[0]

    c_a_norm = c_a / c_a.sum()
    c_b_norm = c_b / c_b.sum()

    if perm_mats is None:
        I4 = jnp.eye(4, dtype=jnp.float32)
        perm_mats = [
            I4,
            jnp.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=jnp.float32),
            jnp.array([[1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0]], dtype=jnp.float32),
            jnp.array([[0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0]], dtype=jnp.float32),
        ]
    m = len(perm_mats)

    Dmats = _build_prompt_b_reweight(v_p) if use_reweighting else None

    T_prompt = jnp.zeros((v_p, num_states, num_states), dtype=jnp.float32)
    T_prompt = T_prompt.at[:, 0, 0].set(c_a_norm)

    b_idx = sector_b_idx.tolist()
    for k in range(v_p):
        P = perm_mats[k % m] if use_permutations else jnp.eye(4, dtype=jnp.float32)
        M = (Dmats[k] @ P) if use_reweighting else P
        for i_local, i in enumerate(b_idx):
            for j_local, j in enumerate(b_idx):
                if M[i_local, j_local] != 0:
                    T_prompt = T_prompt.at[k, i, j].set(c_b_norm[k] * M[i_local, j_local])

    T_prompt = _normalize_net_row_stochastic(T_prompt)
    T_prompt = _fill_dead_states_9(T_prompt)
    return T_prompt


def build_afp_initial_state_prompted(pi_a, num_states, sector_b_idx,
                                     use_nonuniform_b=True):
    """Build initial state with optional non-uniform sector B weights."""
    state = jnp.zeros(num_states, dtype=jnp.float32)
    state = state.at[0].set(pi_a)
    if use_nonuniform_b:
        w = INIT_B_WEIGHTS_4 / INIT_B_WEIGHTS_4.sum()
        state = state.at[sector_b_idx].set((1.0 - pi_a) * w)
    else:
        pi_b_each = (1.0 - pi_a) / len(sector_b_idx)
        state = state.at[sector_b_idx].set(pi_b_each)
    return state


# ── Completion matrices: metastable4 (9-state) ────────────────

def build_metastable_b_transition(eps=B_CLUSTER_EPS, rho=B_WITHIN_RHO):
    """Build metastable 4-state transition matrix for sector B."""
    R = jnp.array([[1.0 - rho, rho], [rho, 1.0 - rho]], dtype=jnp.float32)
    C = jnp.array(
        [
            [0.0, 0.0, 0.5, 0.5],
            [0.0, 0.0, 0.5, 0.5],
            [0.5, 0.5, 0.0, 0.0],
            [0.5, 0.5, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    block = jnp.block([
        [R, jnp.zeros((2, 2), dtype=jnp.float32)],
        [jnp.zeros((2, 2), dtype=jnp.float32), R],
    ])
    result = (1.0 - eps) * block + eps * C
    result = result / result.sum(axis=1, keepdims=True)
    return result


def build_metastable_completion_matrices(delta, beta, alpha, sector_b_idx, num_states,
                                         eps=B_CLUSTER_EPS, rho=B_WITHIN_RHO,
                                         b_emissions=B_EMISSIONS_4, v_c=_DEFAULT_V_C):
    """Build 9-state metastable completion matrices."""
    u = jnp.array(
        [(1.0 - delta) ** 2, (1.0 - delta) * delta,
         delta * (1.0 - delta), delta ** 2],
        dtype=jnp.float32,
    )
    Rb = build_metastable_b_transition(eps=eps, rho=rho)
    b_idx = sector_b_idx.tolist()

    def embed_b_block(block4):
        mat = jnp.zeros((num_states, num_states), dtype=jnp.float32)
        for i_local, i in enumerate(b_idx):
            for j_local, j in enumerate(b_idx):
                if block4[i_local, j_local] != 0:
                    mat = mat.at[i, j].set(block4[i_local, j_local])
        return mat

    mats = []
    for tag in ("good", "bad"):
        for s in range(v_c):
            M_A = jnp.zeros((num_states, num_states), dtype=jnp.float32)
            M_A = M_A.at[0, 0].set(u[s])
            E_s = jnp.diag(b_emissions[:, s])
            M_B = embed_b_block(E_s @ Rb)
            T = (alpha * M_A + beta * M_B) if tag == "good" else (beta * M_A + alpha * M_B)
            mats.append(T)

    result = jnp.stack(mats, axis=0)
    result = _normalize_net_row_stochastic(result)
    result = _fill_dead_states_9(result)
    return result


# ── Completion matrices: clustered_codebook (larger-state) ─────

def build_clustered_codebook_hmms(delta, beta, alpha=1.0, v_p=_DEFAULT_V_P,
                                  pi_a=_DEFAULT_PI_A, comp_len=5):
    """Build prompt+completion HMMs where sector B has K clusters x L phases.

    Returns:
        (prompt_hmm, comp_hmm, info)
    """
    K = int(B_NUM_CLUSTERS)
    L = int(comp_len)
    v_c = _DEFAULT_V_C

    def b_state(c, t):
        return 1 + c * L + t

    num_states = 1 + K * L
    sector_a_idx = jnp.array([0], dtype=jnp.int32)
    sector_b_idx = jnp.array([b_state(c, t) for c in range(K) for t in range(L)], dtype=jnp.int32)

    # Initial state
    init = jnp.zeros((num_states,), dtype=jnp.float32)
    init = init.at[0].set(pi_a)
    if USE_NONUNIFORM_INIT_B:
        w = jnp.linspace(1.0, 2.0, K, dtype=jnp.float32)
        w = w / w.sum()
    else:
        w = jnp.ones((K,), dtype=jnp.float32) / K
    for c in range(K):
        init = init.at[b_state(c, 0)].set((1.0 - pi_a) * w[c])

    # Prompt model
    c_a, c_b = make_log_spaced_biases(v_p, bias_range=2.0)
    c_a = jnp.asarray(c_a, dtype=jnp.float32)
    c_b = jnp.asarray(c_b, dtype=jnp.float32)
    c_a_norm = c_a / c_a.sum()
    c_b_norm = c_b / c_b.sum()

    # Per-cluster token distributions via signature sets
    r = int(PROMPT_SIGNATURE_SIZE)
    if not (1 <= r < v_p):
        raise ValueError(f"PROMPT_SIGNATURE_SIZE must be in [1, V_P-1]; got {r} with V_P={v_p}")

    combos = list(itertools.combinations(range(v_p), r))
    if K > len(combos):
        raise ValueError(f"Need K <= {len(combos)} for unique size-{r} signatures; got K={K}, V_P={v_p}")

    sig_mass = float(PROMPT_SIGNATURE_MASS)
    off_mass = (1.0 - sig_mass) / (v_p - r)

    P_token_given_cluster = jnp.full((K, v_p), off_mass, dtype=jnp.float32)
    for c in range(K):
        sig = combos[c]
        for tok in sig:
            P_token_given_cluster = P_token_given_cluster.at[c, tok].set(sig_mass / r)

    T_prompt = jnp.zeros((v_p, num_states, num_states), dtype=jnp.float32)
    T_prompt = T_prompt.at[:, 0, 0].set(c_a_norm)

    for k in range(v_p):
        for c in range(K):
            w_ck = c_b_norm[k] * P_token_given_cluster[c, k]
            s = b_state(c, 0)
            T_prompt = T_prompt.at[k, s, s].set(w_ck)

    T_prompt = _fill_zero_rows_with_self_loops(T_prompt)
    T_prompt = _normalize_net_row_stochastic(T_prompt)

    prompt_hmm = HiddenMarkovModel(transition_matrices=T_prompt, initial_state=init)

    # Codebook
    if CODEBOOK_STYLE == "base4":
        if K > (4 ** L):
            raise ValueError(f"Need K <= 4^L for base4 codebook; got K={K}, L={L}")
        codebook_base = np.zeros((K, L), dtype=np.int32)
        for c in range(K):
            x = c
            for t in range(L - 1, -1, -1):
                codebook_base[c, t] = x % 4
                x //= 4
    elif CODEBOOK_STYLE == "random":
        rng = np.random.default_rng(CODEBOOK_SEED)
        codebook_base = rng.integers(0, 4, size=(K, L), endpoint=False)
    else:
        raise ValueError(f"Unknown CODEBOOK_STYLE={CODEBOOK_STYLE!r}")

    # Completion matrices
    u = jnp.array(
        [(1.0 - delta) ** 2, (1.0 - delta) * delta,
         delta * (1.0 - delta), delta ** 2],
        dtype=jnp.float32,
    )
    eps = float(B_LEAK_EPS)
    leak_to = np.ones((K, K), dtype=np.float32) / K

    mats = []
    for tag in ("good", "bad"):
        for base_sym in range(4):
            T = jnp.zeros((num_states, num_states), dtype=jnp.float32)
            T = T.at[0, 0].set(u[base_sym])

            for c in range(K):
                for t in range(L):
                    s_from = b_state(c, t)
                    t_next = (t + 1) % L
                    emit_ok = 1.0 if int(codebook_base[c, t]) == base_sym else 0.0
                    if emit_ok == 0.0:
                        continue
                    for c2 in range(K):
                        p_c2 = (1.0 - eps) if c2 == c else eps * leak_to[c, c2]
                        s_to = b_state(c2, t_next)
                        T = T.at[s_from, s_to].add(p_c2 * emit_ok)

            mats.append(T)

    mats = jnp.stack(mats, axis=0)

    # Apply good/bad scaling
    result = []
    for i in range(8):
        is_good = i < 4
        base = mats[i]
        A_part = jnp.zeros_like(base)
        A_part = A_part.at[0, 0].set(base[0, 0])
        B_part = base.at[0, 0].set(0.0)
        if is_good:
            result.append(alpha * A_part + beta * B_part)
        else:
            result.append(beta * A_part + alpha * B_part)
    result = jnp.stack(result, axis=0)

    result = _fill_zero_rows_with_self_loops(result)
    result = _normalize_net_row_stochastic(result)

    comp_hmm = HiddenMarkovModel(transition_matrices=result, initial_state=init)

    info = {
        "variant": "clustered_codebook",
        "num_states": num_states,
        "v_p": v_p,
        "v_c": v_c,
        "total_vocab": v_p + 2 * v_c,
        "delta": delta,
        "beta": beta,
        "alpha": alpha,
        "sector_a_idx": sector_a_idx,
        "sector_b_idx": sector_b_idx,
        "K": K,
        "L": L,
    }

    return prompt_hmm, comp_hmm, info


# ── Leaky reset (two-sector direct-sum) ──────────────────────

def _build_signatures(v_p, d, signature_type="onehot"):
    """Build v_p signature distributions, each a d-dimensional probability vector.

    Args:
        v_p: Number of prompt tokens.
        d: Dimension of the sector (number of hidden states).
        signature_type: "onehot" (cycle one-hot vectors) or "spread" (soft one-hot).

    Returns:
        Array of shape (v_p, d) where each row is a probability distribution.
    """
    if signature_type == "onehot":
        sigs = np.zeros((v_p, d), dtype=np.float32)
        for k in range(v_p):
            sigs[k, k % d] = 1.0
        return jnp.array(sigs)
    elif signature_type == "spread":
        spread_eps = 0.1
        off_mass = spread_eps / (d - 1) if d > 1 else 0.0
        sigs = np.full((v_p, d), off_mass, dtype=np.float32)
        for k in range(v_p):
            sigs[k, k % d] = 1.0 - spread_eps
        return jnp.array(sigs)
    else:
        raise ValueError(f"Unknown signature_type: {signature_type!r}")


def build_leaky_reset_hmms(
    beta=0.6, v_p=_DEFAULT_V_P, pi_a=_DEFAULT_PI_A,
    lambda_g=0.6, lambda_b=0.6,
    decode_noise=0.05, d_g=5, d_b=5,
    content_symbols=5, signature_type="onehot",
):
    """Build two-sector direct-sum HMMs with leaky-reset prompt dynamics.

    The hidden space is H_G ⊕ H_B (block-diagonal).  Prompt tokens are
    sector-neutral (same scalar c(p_k) in both blocks), so π_G/π_B never
    changes.  Within each sector, prompts write via leaky reset:
        μ_S' = (1-λ_S)μ_S + λ_S r_{S,k}

    Completion uses Option 2 (tagged symbols with β collapse):
        T(g_i) = [1·D_G(i)] ⊕ [β·D_B(i)]
        T(b_i) = [β·D_G(i)] ⊕ [1·D_B(i)]

    Args:
        beta: Sector evidence / collapse rate (0 < β < 1).
        v_p: Prompt vocabulary size.
        pi_a: Initial sector G probability.
        lambda_g: G-sector leaky reset strength (0=identity, 1=hard reset).
        lambda_b: B-sector leaky reset strength.
        decode_noise: Readout noise δ (0=perfect, 1=uniform).
        d_g: Number of hidden states in sector G.
        d_b: Number of hidden states in sector B.
        content_symbols: M, number of content indices for completion.
        signature_type: "onehot" or "spread".

    Returns:
        (prompt_hmm, comp_hmm, info)
    """
    # --- Validation ---
    if not (0.0 <= lambda_g <= 1.0):
        raise ValueError(f"lambda_g must be in [0, 1]; got {lambda_g}")
    if not (0.0 <= lambda_b <= 1.0):
        raise ValueError(f"lambda_b must be in [0, 1]; got {lambda_b}")
    if not (0.0 <= decode_noise <= 1.0):
        raise ValueError(f"decode_noise must be in [0, 1]; got {decode_noise}")
    if d_g < 1 or d_b < 1:
        raise ValueError(f"d_g and d_b must be >= 1; got d_g={d_g}, d_b={d_b}")
    M = int(content_symbols)
    if M < max(d_g, d_b):
        raise ValueError(
            f"content_symbols ({M}) must be >= max(d_g, d_b) ({max(d_g, d_b)})"
        )

    num_states = d_g + d_b
    sector_g_idx = jnp.arange(d_g, dtype=jnp.int32)
    sector_b_idx = jnp.arange(d_g, num_states, dtype=jnp.int32)

    # --- Initial state: (π_G · uniform_G, π_B · uniform_B) ---
    init = jnp.zeros(num_states, dtype=jnp.float32)
    init = init.at[:d_g].set(pi_a / d_g)
    init = init.at[d_g:].set((1.0 - pi_a) / d_b)

    # --- Signatures for each sector ---
    sigs_g = _build_signatures(v_p, d_g, signature_type)  # (v_p, d_g)
    sigs_b = _build_signatures(v_p, d_b, signature_type)  # (v_p, d_b)

    # --- Prompt HMM ---
    # T(p_k) = c(p_k) * [S_G(p_k) ⊕ S_B(p_k)]
    # c(p_k) = 1/v_p (uniform → prompt neutrality)
    # S_S(p_k) = (1-λ_S)I + λ_S R_{S,k}   (row-stochastic)
    c_uniform = 1.0 / v_p

    T_prompt = np.zeros((v_p, num_states, num_states), dtype=np.float32)

    I_g = np.eye(d_g, dtype=np.float32)
    I_b = np.eye(d_b, dtype=np.float32)

    for k in range(v_p):
        # G-sector leaky reset
        r_g = np.array(sigs_g[k])   # (d_g,)
        R_g = np.tile(r_g[None, :], (d_g, 1))  # (d_g, d_g) — r_g in every row
        S_g = (1.0 - lambda_g) * I_g + lambda_g * R_g

        # B-sector leaky reset
        r_b = np.array(sigs_b[k])   # (d_b,)
        R_b = np.tile(r_b[None, :], (d_b, 1))
        S_b = (1.0 - lambda_b) * I_b + lambda_b * R_b

        # Embed block-diagonal with shared scalar c(p_k)
        T_prompt[k, :d_g, :d_g] = c_uniform * S_g
        T_prompt[k, d_g:, d_g:] = c_uniform * S_b

    T_prompt = jnp.array(T_prompt)
    prompt_hmm = HiddenMarkovModel(transition_matrices=T_prompt, initial_state=init)

    # --- Completion HMM (Option 2: tagged symbols with β collapse) ---
    # 2M tokens: g_0..g_{M-1} then b_0..b_{M-1}
    # Label maps: φ_S(j) = j (identity when d_S ≤ M)
    # Emission weight: e_{S,i}[j] = (1-δ) if φ_S(j)==i else δ/(M-1)
    # T(g_i) = [1·D_G(i)] ⊕ [β·D_B(i)]
    # T(b_i) = [β·D_G(i)] ⊕ [1·D_B(i)]

    def _readout_diag(d_s, M, decode_noise):
        """Build emission diagonals for one sector.

        Returns array of shape (M, d_s) where entry [i, j] is the emission
        weight for content symbol i given hidden state j.
        """
        E = np.full((M, d_s), decode_noise / (M - 1) if M > 1 else 1.0,
                     dtype=np.float32)
        for j in range(d_s):
            # φ_S(j) = j (identity label map)
            E[j, j] = 1.0 - decode_noise
        return E

    E_g = _readout_diag(d_g, M, decode_noise)  # (M, d_g)
    E_b = _readout_diag(d_b, M, decode_noise)  # (M, d_b)

    T_comp = np.zeros((2 * M, num_states, num_states), dtype=np.float32)

    for i in range(M):
        D_G_i = np.diag(E_g[i])  # (d_g, d_g)
        D_B_i = np.diag(E_b[i])  # (d_b, d_b)

        # g_i token (index i): favours sector G
        T_comp[i, :d_g, :d_g] = 1.0 * D_G_i
        T_comp[i, d_g:, d_g:] = beta * D_B_i

        # b_i token (index M + i): favours sector B
        T_comp[M + i, :d_g, :d_g] = beta * D_G_i
        T_comp[M + i, d_g:, d_g:] = 1.0 * D_B_i

    T_comp = jnp.array(T_comp)
    # Normalize so net transition matrix is row-stochastic (required by HMM validator).
    # This divides by (1+β) uniformly, preserving the β Bayes-factor structure.
    T_comp = _normalize_net_row_stochastic(T_comp)
    comp_hmm = HiddenMarkovModel(transition_matrices=T_comp, initial_state=init)

    info = {
        "variant": "leaky_reset",
        "num_states": num_states,
        "v_p": v_p,
        "v_c": M,
        "total_vocab": v_p + 2 * M,
        "beta": beta,
        "alpha": 1.0,
        "delta": 0.0,
        "sector_a_idx": sector_g_idx,
        "sector_b_idx": sector_b_idx,
        "d_g": d_g,
        "d_b": d_b,
        "lambda_g": lambda_g,
        "lambda_b": lambda_b,
        "decode_noise": decode_noise,
        "content_symbols": M,
        "signature_type": signature_type,
        "prompt_neutral": True,
    }
    return prompt_hmm, comp_hmm, info


# ── Main entrypoint ────────────────────────────────────────────

def build_afp_hmms_prompt_mixing(process_variant="z1r_afp", delta=0.05, beta=0.556,
                                 alpha=1.0, c_a=None, c_b=None, v_p=_DEFAULT_V_P,
                                 pi_a=_DEFAULT_PI_A, bias_range=2.0, comp_len=5,
                                 lambda_g=0.6, lambda_b=0.6, decode_noise=0.05,
                                 d_g=5, d_b=5, content_symbols=5,
                                 signature_type="onehot"):
    """Build prompt and completion HMMs for the specified AFP variant.

    Args:
        process_variant: One of "z1r_afp", "metastable4", "clustered_codebook", "leaky_reset".
        delta: Z1R' leak parameter.
        beta: Completion collapse rate.
        alpha: Dominant sector scale (default 1.0).
        c_a: Prompt sector-A weights (auto-generated if None).
        c_b: Prompt sector-B weights (auto-generated if None).
        v_p: Prompt vocabulary size.
        pi_a: Initial sector A probability.
        bias_range: Log-ratio range for default biases.
        comp_len: Completion sequence length (used by clustered_codebook).
        lambda_g: G-sector leaky reset strength (leaky_reset only).
        lambda_b: B-sector leaky reset strength (leaky_reset only).
        decode_noise: Completion readout noise (leaky_reset only).
        d_g: G-sector hidden states (leaky_reset only).
        d_b: B-sector hidden states (leaky_reset only).
        content_symbols: Number of content indices M (leaky_reset only).
        signature_type: Signature type for leaky reset ("onehot" | "spread").

    Returns:
        (prompt_hmm, comp_hmm, info) where info contains sector indices, vocab sizes, etc.
    """
    if process_variant == "leaky_reset":
        return build_leaky_reset_hmms(
            beta=beta, v_p=v_p, pi_a=pi_a,
            lambda_g=lambda_g, lambda_b=lambda_b,
            decode_noise=decode_noise, d_g=d_g, d_b=d_b,
            content_symbols=content_symbols,
            signature_type=signature_type,
        )

    if process_variant == "clustered_codebook":
        return build_clustered_codebook_hmms(
            delta=delta, beta=beta, alpha=alpha, v_p=v_p, pi_a=pi_a, comp_len=comp_len,
        )

    # --- 9-state variants ---
    num_states = int(_NUM_JOINT_STATES_9)
    sector_a_idx = _SECTOR_A_IDX_9
    sector_b_idx = _SECTOR_B_IDX_9
    v_c = _DEFAULT_V_C

    if c_a is None or c_b is None:
        c_a, c_b = make_log_spaced_biases(v_p, bias_range)
    else:
        c_a = jnp.asarray(c_a, dtype=jnp.float32)
        c_b = jnp.asarray(c_b, dtype=jnp.float32)
        v_p = c_a.shape[0]

    initial_state = build_afp_initial_state_prompted(
        pi_a, num_states, sector_b_idx,
        use_nonuniform_b=USE_NONUNIFORM_INIT_B,
    )

    T_prompt = build_afp_prompt_matrices_perm(
        c_a, c_b, sector_b_idx, num_states,
        use_permutations=USE_PROMPT_PERMUTATIONS,
        use_reweighting=USE_PROMPT_REWEIGHTING,
    )
    prompt_hmm = HiddenMarkovModel(transition_matrices=T_prompt, initial_state=initial_state)

    if process_variant == "metastable4":
        T_comp = build_metastable_completion_matrices(
            delta, beta, alpha, sector_b_idx, num_states,
        )
    else:
        T_comp = build_afp_completion_matrices(delta, beta, alpha)

    comp_hmm = HiddenMarkovModel(transition_matrices=T_comp, initial_state=initial_state)

    info = {
        "variant": process_variant,
        "num_states": int(prompt_hmm.num_states),
        "v_p": v_p,
        "v_c": v_c,
        "total_vocab": v_p + 2 * v_c,
        "delta": delta,
        "beta": beta,
        "alpha": alpha,
        "c_a": c_a,
        "c_b": c_b,
        "sector_a_idx": sector_a_idx,
        "sector_b_idx": sector_b_idx,
    }
    return prompt_hmm, comp_hmm, info
