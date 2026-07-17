"""Stage 3: Pretrain transformer on AFP sequences with belief regression at checkpoints.

Uses existing training infrastructure from run_minimal.py.
At each checkpoint step, runs linear regression from residual-stream activations
onto belief states to measure how well the model has learned the process structure.
"""

import copy
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import torch

from simplexity.generative_processes.torch_generator import generate_data_batch

from em_pipeline.config import (
    PipelineConfig,
    PretrainConfig,
    ProcessResult,
    PretrainResult,
    CheckpointMetrics,
    save_pickle,
    load_pickle,
    save_json,
    to_np_idx,
)


def run(
    process: ProcessResult,
    cfg: PretrainConfig,
    pipeline_cfg: PipelineConfig,
) -> PretrainResult:
    """Train the model, probing belief representations at each checkpoint."""
    # Lazy imports to avoid loading torch/training machinery at module level
    from training.run_minimal import Config as TrainConfig, train, build_model
    from training.matrices import generate_afp_batch

    info = process.info
    total_vocab = info["total_vocab"]
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]

    # Resolve device
    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Build model
    train_cfg = TrainConfig(
        num_steps=cfg.num_steps,
        batch_size=cfg.batch_size,
        learning_rate=cfg.learning_rate,
        d_model=cfg.model.d_model,
        d_head=cfg.model.d_head,
        n_heads=cfg.model.n_heads,
        n_layers=cfg.model.n_layers,
        d_mlp=cfg.model.d_mlp,
        n_ctx=prompt_len + comp_len,
        device=device,
        seed=pipeline_cfg.seed,
        eval_every=max(1, cfg.num_steps // 100),
        print_every=max(1, cfg.num_steps // 20),
    )
    model = build_model(train_cfg, total_vocab)

    # Batch generator
    device_arg = torch.device(device) if device != "cpu" else None

    def batch_gen(seed: int):
        key = jax.random.key(seed)
        return generate_afp_batch(
            process.prompt_hmm, process.comp_hmm,
            batch_size=cfg.batch_size,
            prompt_len=prompt_len,
            comp_len=comp_len,
            key=key,
            v_p=v_p,
            device=device_arg,
        )

    # Set up output directory
    output_dir = Path(pipeline_cfg.output_dir)
    if pipeline_cfg.run_name:
        output_dir = output_dir / pipeline_cfg.run_name
    stage_dir = output_dir / "stage3"
    ckpt_dir = stage_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    # Checkpointing and probing via on_step callback
    checkpoint_steps = set(cfg.checkpoint_steps)
    checkpoint_metrics: list[CheckpointMetrics] = []
    last_eval_loss = [None]  # mutable reference for closure

    def on_step(step: int, loss: float):
        if step in checkpoint_steps:
            # Save checkpoint
            torch.save(model.state_dict(), ckpt_dir / f"step_{step}.pt")

            # Run belief probing
            metrics = probe_beliefs(
                model, process, cfg.regression_n_samples,
                seed=pipeline_cfg.seed + step, device=device,
            )
            metrics.step = step
            metrics.train_loss = loss
            metrics.eval_loss = last_eval_loss[0]
            checkpoint_metrics.append(metrics)

            print(f"  Checkpoint {step}: R2_joint={metrics.r2_joint:.3f}, "
                  f"R2_sector={metrics.r2_sector_mass:.3f}, "
                  f"R2_within_A={metrics.r2_within_a:.3f}, "
                  f"R2_within_B={metrics.r2_within_b:.3f}")

    # Train
    print(f"Training for {cfg.num_steps} steps on {device}")
    print(f"Model: {cfg.model.n_layers}L, {cfg.model.d_model}d, {cfg.model.n_heads}h")
    print(f"AFP: V_p={v_p}, prompt_len={prompt_len}, comp_len={comp_len}, "
          f"total_vocab={total_vocab}")
    results = train(model, hmm=None, cfg=train_cfg, batch_generator=batch_gen, on_step=on_step)

    # Extract final losses
    history = results["history"]
    final_train_loss = history[-1]["train_loss"] if history else float("nan")
    eval_losses = [h["eval_loss"] for h in history if "eval_loss" in h]
    final_eval_loss = eval_losses[-1] if eval_losses else None

    # Save final model
    torch.save(model.state_dict(), stage_dir / "model.pt")

    # Save HMMs for reloading in Stage 4
    from em_pipeline.process import _save_hmm
    _save_hmm(process.prompt_hmm, stage_dir / "prompt_hmm.npz")
    _save_hmm(process.comp_hmm, stage_dir / "comp_hmm.npz")

    # Save training config for model reconstruction
    save_json({
        "total_vocab": total_vocab,
        "train_config": {
            "d_model": cfg.model.d_model,
            "d_head": cfg.model.d_head,
            "n_heads": cfg.model.n_heads,
            "n_layers": cfg.model.n_layers,
            "d_mlp": cfg.model.d_mlp,
            "n_ctx": prompt_len + comp_len,
        },
    }, stage_dir / "model_config.json")

    pretrain_result = PretrainResult(
        run_dir=str(stage_dir),
        history=history,
        checkpoint_metrics=checkpoint_metrics,
        final_train_loss=final_train_loss,
        final_eval_loss=final_eval_loss,
    )

    return pretrain_result


def save(result: PretrainResult, path: Path) -> None:
    """Save pretrain results metadata."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "pretrain_result.pkl")


def load(path: Path) -> PretrainResult:
    """Load pretrain results from disk."""
    return load_pickle(Path(path) / "pretrain_result.pkl")


# =============================================================================
# BELIEF PROBING
# =============================================================================

def collect_probe_data(
    model: torch.nn.Module,
    process: ProcessResult,
    n_samples: int,
    seed: int,
    device: str,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Generate sequences, get activations + beliefs.

    Returns:
        activations: (N, d_model) numpy array of last-position residual stream
        beliefs_at_last: (N, num_states) numpy array of beliefs at last input position
        info_dict: dict with sector_a_idx, sector_b_idx for downstream use
    """
    info = process.info
    prompt_len = info["prompt_len"]
    comp_len = info["comp_len"]
    v_p = info["v_p"]
    sector_a_idx = to_np_idx(info["sector_a_idx"])
    sector_b_idx = to_np_idx(info["sector_b_idx"])

    # Generate sequences
    from training.matrices import generate_afp_batch

    device_arg = torch.device(device) if device != "cpu" else None
    key = jax.random.key(seed)
    inputs, labels = generate_afp_batch(
        process.prompt_hmm, process.comp_hmm,
        batch_size=n_samples,
        prompt_len=prompt_len,
        comp_len=comp_len,
        key=key,
        v_p=v_p,
        device=device_arg,
    )

    # Reconstruct full token sequences
    full_tokens = torch.cat([inputs[:, 0:1], labels], dim=1)  # (N, prompt_len + comp_len)

    # Compute beliefs via HMM forward pass
    beliefs = compute_beliefs_for_sequences(
        full_tokens.cpu().numpy(), process, prompt_len, v_p,
    )  # (N, seq_len, num_states)

    # Get model activations at last position
    model.eval()
    with torch.no_grad():
        _, cache = model.run_with_cache(inputs)
    # Last layer residual stream, last position
    last_layer = model.cfg.n_layers - 1
    hook_name = f"blocks.{last_layer}.hook_resid_post"
    activations = cache[hook_name][:, -1, :].cpu().numpy()  # (N, d_model)

    # Beliefs at last input position (= position seq_len - 2, since inputs = tokens[:-1])
    last_pos = inputs.shape[1] - 1
    beliefs_at_last = beliefs[:, last_pos, :]  # (N, num_states)

    model.train()

    return activations, beliefs_at_last, {
        "sector_a_idx": sector_a_idx,
        "sector_b_idx": sector_b_idx,
    }


def fit_linear_probe(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
    """Fit least-squares probe, return weight matrix W (including bias row).

    X: (N, d_in), Y: (N, d_out)
    Returns W: (d_in + 1, d_out) where last row is bias.
    """
    if Y.ndim == 1:
        Y = Y[:, None]

    N = X.shape[0]
    X_bias = np.hstack([X, np.ones((N, 1))])

    try:
        W, _, _, _ = np.linalg.lstsq(X_bias, Y, rcond=None)
    except np.linalg.LinAlgError:
        W = np.zeros((X_bias.shape[1], Y.shape[1]))

    return W


def apply_probe_r2(W: np.ndarray, X: np.ndarray, Y: np.ndarray) -> float:
    """Compute R² using frozen probe weights W (no fitting).

    W: (d_in + 1, d_out), X: (N, d_in), Y: (N, d_out)
    """
    if Y.ndim == 1:
        Y = Y[:, None]

    N = X.shape[0]
    if N < 2:
        return 0.0

    X_bias = np.hstack([X, np.ones((N, 1))])
    Y_pred = X_bias @ W

    ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)

    r2_per_dim = np.where(ss_tot > 1e-10, 1 - ss_res / ss_tot, 0.0)
    r2_per_dim = np.clip(r2_per_dim, 0.0, 1.0)

    return float(r2_per_dim.mean())


def fit_all_probes(
    activations: np.ndarray,
    beliefs_at_last: np.ndarray,
    sector_a_idx: np.ndarray,
    sector_b_idx: np.ndarray,
    min_sector_mass: float = 0.01,
) -> dict[str, np.ndarray]:
    """Fit probes for joint, sector_mass, within_a, within_b.

    Returns dict mapping probe name to weight matrix W.
    """
    # Joint beliefs
    W_joint = fit_linear_probe(activations, beliefs_at_last)

    # Sector mass (scalar pi_A)
    pi_a = beliefs_at_last[:, sector_a_idx].sum(axis=1, keepdims=True)
    W_sector_mass = fit_linear_probe(activations, pi_a)

    # Within-sector probes
    W_within_a = _fit_within_sector_probe(
        activations, beliefs_at_last, sector_a_idx, min_sector_mass,
    )
    W_within_b = _fit_within_sector_probe(
        activations, beliefs_at_last, sector_b_idx, min_sector_mass,
    )

    return {
        "joint": W_joint,
        "sector_mass": W_sector_mass,
        "within_a": W_within_a,
        "within_b": W_within_b,
    }


def apply_all_probes(
    probes: dict[str, np.ndarray],
    activations: np.ndarray,
    beliefs_at_last: np.ndarray,
    sector_a_idx: np.ndarray,
    sector_b_idx: np.ndarray,
    min_sector_mass: float = 0.01,
) -> dict[str, float]:
    """Apply frozen probes to new activations/beliefs, return R² dict."""
    r2_joint = apply_probe_r2(probes["joint"], activations, beliefs_at_last)

    pi_a = beliefs_at_last[:, sector_a_idx].sum(axis=1, keepdims=True)
    r2_sector_mass = apply_probe_r2(probes["sector_mass"], activations, pi_a)

    r2_within_a = _apply_within_sector_probe(
        probes["within_a"], activations, beliefs_at_last,
        sector_a_idx, min_sector_mass,
    )
    r2_within_b = _apply_within_sector_probe(
        probes["within_b"], activations, beliefs_at_last,
        sector_b_idx, min_sector_mass,
    )

    return {
        "joint": r2_joint,
        "sector_mass": r2_sector_mass,
        "within_a": r2_within_a,
        "within_b": r2_within_b,
    }


def _fit_within_sector_probe(
    activations: np.ndarray,
    beliefs: np.ndarray,
    sector_idx: np.ndarray,
    min_sector_mass: float = 0.01,
) -> np.ndarray:
    """Fit a within-sector probe. Returns weight matrix W."""
    raw = beliefs[:, sector_idx]
    sector_mass = raw.sum(axis=1)
    valid = sector_mass >= min_sector_mass

    if valid.sum() < 2:
        d_in = activations.shape[1]
        d_out = len(sector_idx)
        return np.zeros((d_in + 1, d_out))

    act_valid = activations[valid]
    mu = raw[valid] / sector_mass[valid, None]

    return fit_linear_probe(act_valid, mu)


def _apply_within_sector_probe(
    W: np.ndarray,
    activations: np.ndarray,
    beliefs: np.ndarray,
    sector_idx: np.ndarray,
    min_sector_mass: float = 0.01,
) -> float:
    """Apply a frozen within-sector probe. Returns R²."""
    raw = beliefs[:, sector_idx]
    sector_mass = raw.sum(axis=1)
    valid = sector_mass >= min_sector_mass

    if valid.sum() < 2:
        return 0.0

    act_valid = activations[valid]
    mu = raw[valid] / sector_mass[valid, None]

    return apply_probe_r2(W, act_valid, mu)


def probe_beliefs(
    model: torch.nn.Module,
    process: ProcessResult,
    n_samples: int,
    seed: int,
    device: str,
) -> CheckpointMetrics:
    """Run belief regression at current model state.

    Generates AFP sequences, computes beliefs via HMM forward pass,
    collects model activations, and regresses activations onto beliefs.
    """
    activations, beliefs_at_last, info_dict = collect_probe_data(
        model, process, n_samples, seed, device,
    )
    sector_a_idx = info_dict["sector_a_idx"]
    sector_b_idx = info_dict["sector_b_idx"]

    # Regression targets
    r2_joint = linear_regression_r2(activations, beliefs_at_last)

    # Sector mass (scalar)
    pi_a = beliefs_at_last[:, sector_a_idx].sum(axis=1, keepdims=True)  # (N, 1)
    r2_sector = linear_regression_r2(activations, pi_a)

    # Within-sector beliefs
    r2_within_a = _probe_within_sector(activations, beliefs_at_last, sector_a_idx)
    r2_within_b = _probe_within_sector(activations, beliefs_at_last, sector_b_idx)

    return CheckpointMetrics(
        step=0,  # filled in by caller
        train_loss=0.0,
        eval_loss=None,
        r2_joint=float(r2_joint),
        r2_sector_mass=float(r2_sector),
        r2_within_a=float(r2_within_a),
        r2_within_b=float(r2_within_b),
    )


def compute_beliefs_for_sequences(
    tokens: np.ndarray,  # (N, seq_len)
    process: ProcessResult,
    prompt_len: int,
    v_p: int,
) -> np.ndarray:
    """Compute belief states at each position by forward-passing through HMM matrices.

    For positions 0..prompt_len-1, uses prompt HMM.
    For positions prompt_len..end, uses completion HMM (with token offset removed).

    Returns: (N, seq_len, num_states)
    """
    T_prompt = np.array(process.prompt_hmm.transition_matrices)
    T_comp = np.array(process.comp_hmm.transition_matrices)
    init = np.array(process.prompt_hmm.initial_state)

    N, seq_len = tokens.shape
    num_states = init.shape[0]
    beliefs = np.zeros((N, seq_len, num_states))

    for i in range(N):
        state = init.copy()
        for t in range(seq_len):
            tok = int(tokens[i, t])
            if t < prompt_len:
                # Prompt phase
                state = state @ T_prompt[tok]
            else:
                # Completion phase - remove vocab offset
                comp_tok = tok - v_p
                if 0 <= comp_tok < T_comp.shape[0]:
                    state = state @ T_comp[comp_tok]
                else:
                    import logging
                    logging.warning(
                        f"Out-of-bounds completion token: tok={tok}, comp_tok={comp_tok}, "
                        f"T_comp.shape[0]={T_comp.shape[0]}. Belief state not updated. "
                        f"This may indicate a data generation bug."
                    )
            s = state.sum()
            if s > 0:
                state /= s
            beliefs[i, t, :] = state

    return beliefs


def _probe_within_sector(
    activations: np.ndarray,
    beliefs: np.ndarray,
    sector_idx: np.ndarray,
    min_sector_mass: float = 0.01,
) -> float:
    """Compute R² for within-sector belief μ_S = beliefs[sector] / sector_mass.

    Filters out samples where sector mass is below min_sector_mass to avoid
    division by near-zero.
    """
    raw = beliefs[:, sector_idx]  # (N, d_sector)
    sector_mass = raw.sum(axis=1)  # (N,)

    # Filter samples with sufficient sector mass
    valid = sector_mass >= min_sector_mass
    if valid.sum() < 2:
        return 0.0

    act_valid = activations[valid]
    raw_valid = raw[valid]
    mass_valid = sector_mass[valid, None]  # (N_valid, 1)

    # Normalize to within-sector distribution
    mu = raw_valid / mass_valid  # (N_valid, d_sector)

    return linear_regression_r2(act_valid, mu)


def linear_regression_r2(X: np.ndarray, Y: np.ndarray) -> float:
    """Compute R^2 from linear regression X -> Y.

    X: (N, d_in), Y: (N, d_out)
    Returns mean R^2 across output dimensions.
    """
    if Y.ndim == 1:
        Y = Y[:, None]

    N = X.shape[0]
    if N < 2:
        return 0.0

    # Add bias term
    X_bias = np.hstack([X, np.ones((N, 1))])

    # Solve via least squares
    try:
        W, residuals, rank, sv = np.linalg.lstsq(X_bias, Y, rcond=None)
    except np.linalg.LinAlgError:
        return 0.0

    Y_pred = X_bias @ W
    ss_res = ((Y - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y - Y.mean(axis=0)) ** 2).sum(axis=0)

    # Per-dimension R^2, clipped to [0, 1]
    r2_per_dim = np.where(ss_tot > 1e-10, 1 - ss_res / ss_tot, 0.0)
    r2_per_dim = np.clip(r2_per_dim, 0.0, 1.0)

    return float(r2_per_dim.mean())
