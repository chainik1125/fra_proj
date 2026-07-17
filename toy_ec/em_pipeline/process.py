"""Stage 1: Define the AFP generative process.

Builds prompt and completion HMMs for the configured variant.
Thin wrapper around existing build functions in afp_builders.py.
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from simplexity.generative_processes.hidden_markov_model import HiddenMarkovModel

from em_pipeline.config import (
    ProcessConfig,
    SequenceConfig,
    ProcessResult,
    save_pickle,
    load_pickle,
)
from afp_builders import build_afp_hmms_prompt_mixing


def run(cfg: ProcessConfig, seq: SequenceConfig) -> ProcessResult:
    """Build prompt and completion HMMs for the AFP process.

    Dispatches to build_afp_hmms_prompt_mixing for all variants.
    """
    prompt_hmm, comp_hmm, info = build_afp_hmms_prompt_mixing(
        process_variant=cfg.variant,
        delta=cfg.delta,
        beta=cfg.beta,
        alpha=cfg.alpha,
        v_p=cfg.v_p,
        pi_a=cfg.pi_a,
        bias_range=cfg.bias_range,
        comp_len=seq.comp_len,
        lambda_g=cfg.lambda_g,
        lambda_b=cfg.lambda_b,
        decode_noise=cfg.decode_noise,
        d_g=cfg.d_g,
        d_b=cfg.d_b,
        content_symbols=cfg.content_symbols,
        signature_type=cfg.signature_type,
    )

    # Ensure info has all required fields
    info.setdefault("num_states", int(prompt_hmm.num_states))
    info["prompt_len"] = seq.prompt_len
    info["comp_len"] = seq.comp_len

    # Convert JAX arrays in info to python types for serializability
    info = _sanitize_info(info)

    return ProcessResult(
        prompt_hmm=prompt_hmm,
        comp_hmm=comp_hmm,
        info=info,
        config=cfg,
        sequence=seq,
    )


def save(result: ProcessResult, path: Path) -> None:
    """Save process result to disk."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

    # Save HMMs as npz
    _save_hmm(result.prompt_hmm, path / "prompt_hmm.npz")
    _save_hmm(result.comp_hmm, path / "comp_hmm.npz")

    # Save info and configs as pickle
    save_pickle(result.info, path / "info.pkl")
    save_pickle(result.config, path / "process_config.pkl")
    save_pickle(result.sequence, path / "sequence_config.pkl")


def load(path: Path) -> ProcessResult:
    """Load a previously saved ProcessResult from disk."""
    path = Path(path)

    prompt_hmm = _load_hmm(path / "prompt_hmm.npz")
    comp_hmm = _load_hmm(path / "comp_hmm.npz")
    info = load_pickle(path / "info.pkl")
    config = load_pickle(path / "process_config.pkl")
    sequence = load_pickle(path / "sequence_config.pkl")

    return ProcessResult(
        prompt_hmm=prompt_hmm,
        comp_hmm=comp_hmm,
        info=info,
        config=config,
        sequence=sequence,
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _save_hmm(hmm: HiddenMarkovModel, path: Path) -> None:
    np.savez(
        path,
        transition_matrices=np.array(hmm.transition_matrices),
        initial_state=np.array(hmm.initial_state),
    )


def _load_hmm(path: Path) -> HiddenMarkovModel:
    data = np.load(path)
    return HiddenMarkovModel(
        transition_matrices=jnp.array(data["transition_matrices"]),
        initial_state=jnp.array(data["initial_state"]),
    )


def _sanitize_info(info: dict) -> dict:
    """Convert JAX/numpy types to plain Python for JSON serialization."""
    out = {}
    for k, v in info.items():
        if hasattr(v, "tolist"):
            out[k] = v.tolist()
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.floating,)):
            out[k] = float(v)
        else:
            out[k] = v
    return out
