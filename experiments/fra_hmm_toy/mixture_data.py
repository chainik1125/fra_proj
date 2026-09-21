"""Mixture-of-HMMs data for the FRA-vs-SAE concept-cut toy.

Vendored (HMM-only, trimmed) from sae_day/src/sae_day/data.py, extended to
record per-component belief-state trajectories so Bayes-optimal quantities are
available exactly. RNG call order matches the original
generate_mixture_data_with_sequence_omega, so tokens are bit-identical to the
experiment_03 recipe for the same config/seed.

Process (distinct vocab): each sequence draws omega ~ Dirichlet(conc * mean).
At each step a component c_t ~ Cat(omega) emits one token from ITS vocab block
{3c, 3c+1, 3c+2} and updates only its own hidden belief state. Component
identity of every token is therefore observable, and:
  - posterior over omega after tokens x_{0:t} is Dirichlet(alpha + counts)
  - Bayes next-token: p*(v=(c,s) | x_{0:t}) = E[omega_c | x_{0:t}] * p_c(s | belief_c(t))
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import torch
import torch.nn.functional as F


def mess3_transitions(x: float, a: float) -> torch.Tensor:
    """MESS3 transition matrices T[v, i, j] = P(emit v, go j | state i). V=S=3."""
    b = (1 - a) / 2
    T = torch.zeros(3, 3, 3)
    # explicit fill matching sae_day exactly:
    T[0, 0, 0] = a * (1 - x); T[0, 0, 1] = b * x;       T[0, 0, 2] = b * x
    T[0, 1, 0] = b * (1 - x); T[0, 1, 1] = a * x;       T[0, 1, 2] = b * x
    T[0, 2, 0] = b * (1 - x); T[0, 2, 1] = b * x;       T[0, 2, 2] = a * x
    T[1, 0, 0] = a * x;       T[1, 0, 1] = b * (1 - x); T[1, 0, 2] = b * x
    T[1, 1, 0] = b * x;       T[1, 1, 1] = a * (1 - x); T[1, 1, 2] = b * x
    T[1, 2, 0] = b * x;       T[1, 2, 1] = b * (1 - x); T[1, 2, 2] = a * x
    T[2, 0, 0] = a * x;       T[2, 0, 1] = b * x;       T[2, 0, 2] = b * (1 - x)
    T[2, 1, 0] = b * x;       T[2, 1, 1] = a * x;       T[2, 1, 2] = b * (1 - x)
    T[2, 2, 0] = b * x;       T[2, 2, 1] = b * x;       T[2, 2, 2] = a * (1 - x)
    return T


@dataclass
class HMMComponent:
    transition_matrices: torch.Tensor  # (V, S, S)
    label: str = ""

    @property
    def V(self) -> int:
        return self.transition_matrices.shape[0]

    @property
    def d(self) -> int:
        return self.transition_matrices.shape[1]


@dataclass
class MixtureConfig:
    components: list[HMMComponent]
    omega: list[float]
    seed: int = 42

    @property
    def K(self) -> int:
        return len(self.components)

    @property
    def V_total(self) -> int:
        return sum(c.V for c in self.components)

    @property
    def vocab_offsets(self) -> list[int]:
        offs, o = [], 0
        for c in self.components:
            offs.append(o)
            o += c.V
        return offs


MESS3_SEPARATED = MixtureConfig(
    components=[
        HMMComponent(mess3_transitions(x=0.08, a=0.9), "Sticky(x=0.08,a=0.9)"),
        HMMComponent(mess3_transitions(x=0.25, a=0.65), "Moderate(x=0.25,a=0.65)"),
        HMMComponent(mess3_transitions(x=0.4, a=0.34), "Diffuse(x=0.4,a=0.34)"),
    ],
    omega=[0.4, 0.35, 0.25],
)


@dataclass
class MixtureDataset:
    tokens: torch.Tensor            # (N, L) int in [0, V_total)
    components: torch.Tensor        # (N, L) int — which component emitted
    sequence_omegas: torch.Tensor   # (N, K) latent per-sequence omega
    posterior_omegas: torch.Tensor  # (N, L, K) E[omega | x_{0:t}] (after token t)
    beliefs: torch.Tensor           # (N, L, K, S) belief of comp k after token t
    config: MixtureConfig
    concentration: float

    @property
    def block_of_token(self) -> torch.Tensor:
        """(V_total,) map token id -> component index."""
        cfg = self.config
        m = torch.zeros(cfg.V_total, dtype=torch.long)
        for c_idx, (off, c) in enumerate(zip(cfg.vocab_offsets, cfg.components)):
            m[off : off + c.V] = c_idx
        return m

    def bayes_next_token(self) -> torch.Tensor:
        """(N, L, V_total): p*(v_{t+1} | x_{0:t}) — the Bayes-optimal predictor.

        p*(offset_c + s) = posterior_omega_c(t) * (belief_c(t) @ emit_c)[s]
        """
        cfg = self.config
        N, L = self.tokens.shape
        out = torch.zeros(N, L, cfg.V_total)
        for c_idx, (off, comp) in enumerate(zip(cfg.vocab_offsets, cfg.components)):
            emit = comp.transition_matrices.sum(dim=2).T  # (S, V): P(v | state)
            within = self.beliefs[:, :, c_idx, :] @ emit  # (N, L, V_c)
            out[:, :, off : off + comp.V] = (
                self.posterior_omegas[:, :, c_idx : c_idx + 1] * within
            )
        return out


def _sample_dirichlet_rows(alpha: torch.Tensor, n_sequences: int, seed: int) -> torch.Tensor:
    rng = np.random.default_rng(seed)
    samples = rng.gamma(
        shape=np.asarray(alpha, dtype=np.float64),
        scale=1.0,
        size=(n_sequences, alpha.numel()),
    )
    samples /= samples.sum(axis=1, keepdims=True)
    return torch.tensor(samples, dtype=torch.float32)


def generate(
    config: MixtureConfig,
    n_sequences: int,
    seq_len: int,
    concentration: float = 10.0,
) -> MixtureDataset:
    """Port of generate_mixture_data_with_sequence_omega (HMM-only) + beliefs."""
    gen = torch.Generator().manual_seed(config.seed)
    K = config.K
    vocab_offsets = config.vocab_offsets

    comp_data = []
    for c in config.components:
        T = c.transition_matrices
        comp_data.append({"T": T, "emit": T.sum(dim=2).T})  # emit: (S, V)

    states = [torch.ones(n_sequences, c.d) / c.d for c in config.components]

    prior_alpha = torch.tensor(config.omega, dtype=torch.float32) * concentration
    sequence_omegas = _sample_dirichlet_rows(prior_alpha, n_sequences, seed=config.seed)

    all_tokens = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    all_components = torch.zeros(n_sequences, seq_len, dtype=torch.long)
    posterior_omegas = torch.zeros(n_sequences, seq_len, K)
    S_max = max(c.d for c in config.components)
    beliefs = torch.zeros(n_sequences, seq_len, K, S_max)
    component_counts = torch.zeros(n_sequences, K)

    for t in range(seq_len):
        comp_idx = torch.multinomial(sequence_omegas, 1, generator=gen).squeeze(1)
        token_global = torch.zeros(n_sequences, dtype=torch.long)

        for c_idx in range(K):
            mask = comp_idx == c_idx
            if not mask.any():
                continue
            cd = comp_data[c_idx]
            emit_probs = states[c_idx][mask] @ cd["emit"]
            emit_probs = emit_probs.clamp(min=1e-8)
            emit_probs = emit_probs / emit_probs.sum(dim=1, keepdim=True)
            local_token = torch.multinomial(emit_probs, 1, generator=gen).squeeze(1)
            token_global[mask] = local_token + vocab_offsets[c_idx]

            T = cd["T"]
            belief = states[c_idx][mask]
            Tv = T[local_token]
            new_unnorm = torch.bmm(belief.unsqueeze(1), Tv).squeeze(1)
            new_norm = new_unnorm.sum(dim=1, keepdim=True).clamp(min=1e-10)
            states[c_idx][mask] = new_unnorm / new_norm

        all_tokens[:, t] = token_global
        all_components[:, t] = comp_idx
        component_counts.scatter_add_(
            dim=1,
            index=comp_idx.unsqueeze(1),
            src=torch.ones(n_sequences, 1, dtype=torch.float32),
        )
        posterior_omegas[:, t] = (component_counts + prior_alpha.unsqueeze(0)) / (
            concentration + t + 1
        )
        for c_idx in range(K):
            beliefs[:, t, c_idx, : config.components[c_idx].d] = states[c_idx]

    return MixtureDataset(
        tokens=all_tokens,
        components=all_components,
        sequence_omegas=sequence_omegas,
        posterior_omegas=posterior_omegas,
        beliefs=beliefs,
        config=config,
        concentration=concentration,
    )


def make_dataset(seed: int, n_sequences: int, seq_len: int, concentration: float = 10.0) -> MixtureDataset:
    return generate(
        replace(MESS3_SEPARATED, seed=seed),
        n_sequences=n_sequences,
        seq_len=seq_len,
        concentration=concentration,
    )
