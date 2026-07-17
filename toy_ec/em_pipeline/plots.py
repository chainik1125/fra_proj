"""SVG plot generation for AFP steering pipeline metrics.

All plot functions save crisp vector SVGs to the given output directory.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import jax
import jax.numpy as jnp

from simplexity.generative_processes.torch_generator import generate_data_batch

from em_pipeline.config import (
    AnalysisResult,
    PretrainResult,
    FinetuneResult,
    SectorFinetuneResult,
    ProcessResult,
    CorrectionSweepResult,
    CorrectionMixResult,
    DiffingResult,
    DecompositionResult,
)

# Global style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "svg.fonttype": "none",  # embed text as text, not paths
})


# =============================================================================
# STAGE 2: ANALYSIS PLOTS
# =============================================================================

def plot_analysis(result: AnalysisResult, output_dir: Path,
                  process: ProcessResult | None = None) -> list[Path]:
    """Generate all Stage 2 analysis plots. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_pi_a_histogram(result, output_dir))
    paths.append(_plot_completion_entropy(result, output_dir))
    paths.append(_plot_polarization_curve(result, output_dir))
    paths.append(_plot_distinguishability(result, output_dir))

    if result.sector_polarization is not None:
        paths.append(_plot_final_pi_a_histogram(result, output_dir))

    if process is not None:
        paths.extend(plot_polarization_trajectories(process, output_dir))

    return paths


def _plot_pi_a_histogram(result: AnalysisResult, output_dir: Path) -> Path:
    """Histogram of post-prompt sector A mass across all prompts."""
    pi_a_values = list(result.prompt_to_pi_a.values())

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(pi_a_values, bins=30, color="#4C72B0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$\pi_A$ (sector A mass after prompt)")
    ax.set_ylabel("Count")
    ax.set_title("Post-prompt belief distribution")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    fig.tight_layout()
    path = output_dir / "pi_a_histogram.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_completion_entropy(result: AnalysisResult, output_dir: Path) -> Path:
    """Completion Shannon entropy vs starting pi_A."""
    pi_a_vals = []
    entropies = []
    for pi_a_str, stats in result.completion_diversity.items():
        pi_a_vals.append(float(pi_a_str))
        entropies.append(stats["entropy"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(pi_a_vals, entropies, "o-", color="#DD8452", markersize=6, linewidth=1.5)
    ax.set_xlabel(r"Starting $\pi_A$")
    ax.set_ylabel("Shannon entropy (nats)")
    ax.set_title("Completion diversity vs sector bias")

    fig.tight_layout()
    path = output_dir / "completion_entropy.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_polarization_curve(result: AnalysisResult, output_dir: Path) -> Path:
    """P(collapse to sector A) vs starting pi_A."""
    pi_a_vals = []
    collapse_fracs = []
    for pi_a_str, frac in sorted(result.polarization_curve.items()):
        pi_a_vals.append(float(pi_a_str))
        collapse_fracs.append(frac)

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(pi_a_vals, collapse_fracs, "s-", color="#55A868", markersize=6, linewidth=1.5)
    ax.set_xlabel(r"Starting $\pi_A$")
    ax.set_ylabel(r"$P(\text{collapse to A})$")
    ax.set_title("Completion polarization curve")
    ax.set_ylim(-0.05, 1.05)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

    fig.tight_layout()
    path = output_dir / "polarization_curve.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_distinguishability(result: AnalysisResult, output_dir: Path) -> Path:
    """Analytical vs empirical D_KL by completion length."""
    lengths = list(range(1, len(result.analytical_kl_by_length) + 1))

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(lengths, result.analytical_kl_by_length, "o--", color="#4C72B0",
            markersize=5, linewidth=1.5, label="Analytical (tag-based)")
    ax.plot(lengths, result.empirical_kl_by_length, "s-", color="#C44E52",
            markersize=5, linewidth=1.5, label="Empirical (full)")
    ax.set_xlabel("Completion length $L$")
    ax.set_ylabel(r"$D_{\mathrm{KL}}(A \| B)$ (nats)")
    ax.set_title("Sector distinguishability by length")
    ax.legend()

    fig.tight_layout()
    path = output_dir / "distinguishability.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_final_pi_a_histogram(result: AnalysisResult, output_dir: Path) -> Path:
    """Histogram of final π_A after completion across all (prompt, completion) pairs."""
    sp = result.sector_polarization
    pi_a_values = np.array(sp["final_pi_a_all"])

    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist(pi_a_values, bins=50, color="#C44E52", edgecolor="white", linewidth=0.5)
    ax.set_xlabel(r"$\pi_A$ (sector A mass after completion)")
    ax.set_ylabel("Count")
    ax.set_title("Post-completion belief distribution")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.7)

    # Annotate with summary stats
    textstr = (
        f"Unique completions: {sp['n_unique_completions']}\n"
        f"Frac polarized: {sp['frac_polarized_09']:.3f}"
    )
    ax.text(
        0.97, 0.95, textstr, transform=ax.transAxes,
        fontsize=9, verticalalignment="top", horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
    )

    fig.tight_layout()
    path = output_dir / "final_pi_a_histogram.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# STAGE 2: POLARIZATION TRAJECTORY PLOTS
# =============================================================================

def plot_polarization_trajectories(
    process: ProcessResult,
    output_dir: Path,
    n_trajectories: int = 20,
    n_fan_samples: int = 1000,
    starting_pi_a_values: list[float] | None = None,
    seed: int = 0,
) -> list[Path]:
    """Plot individual belief trajectories and fan (percentile) diagram.

    Args:
        process: The AFP process (needed for HMMs).
        output_dir: Where to save SVGs.
        n_trajectories: Number of individual traces to plot.
        n_fan_samples: Number of samples for the fan diagram.
        starting_pi_a_values: Starting pi_A values for fan diagram panels.
        seed: Random seed.

    Returns list of saved paths.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if starting_pi_a_values is None:
        starting_pi_a_values = [0.3, 0.5, 0.7]

    info = process.info
    comp_len = info["comp_len"]
    sector_a_idx = np.array(info["sector_a_idx"], dtype=np.int32)

    paths = []
    paths.append(_plot_sample_trajectories(
        process, sector_a_idx, comp_len, n_trajectories, output_dir, seed,
    ))
    paths.append(_plot_fan_diagram(
        process, sector_a_idx, comp_len, n_fan_samples,
        starting_pi_a_values, output_dir, seed,
    ))
    return paths


def _sample_belief_trajectories(
    process: ProcessResult,
    sector_a_idx: np.ndarray,
    comp_len: int,
    n_samples: int,
    starting_pi_a: float,
    seed: int,
) -> np.ndarray:
    """Sample completions and compute pi_A at each step.

    Returns: (n_samples, comp_len + 1) array of pi_A values.
             Column 0 is the starting pi_A, columns 1..comp_len are after each token.
    """
    T_comp = np.array(process.comp_hmm.transition_matrices)  # (V_c, S, S)
    num_states = T_comp.shape[1]
    sector_b_idx = np.array(
        [i for i in range(num_states) if i not in sector_a_idx], dtype=np.int32
    )

    # Construct initial state
    init = np.zeros(num_states)
    init[sector_a_idx] = starting_pi_a / len(sector_a_idx)
    if len(sector_b_idx) > 0:
        init[sector_b_idx] = (1 - starting_pi_a) / len(sector_b_idx)

    # Sample completion sequences using the HMM
    init_batch = jnp.repeat(jnp.array(init)[None, :], n_samples, axis=0)
    key = jax.random.key(seed)
    final_states, comp_inputs, comp_labels = generate_data_batch(
        init_batch, process.comp_hmm, n_samples, comp_len + 1, key,
    )

    # Reconstruct full token sequences
    import torch
    tokens = torch.cat([comp_inputs[:, 0:1], comp_labels], dim=1).numpy()  # (N, comp_len+1)

    # Forward-pass beliefs to get pi_A at each step
    pi_a_trajectories = np.zeros((n_samples, comp_len + 1))
    pi_a_trajectories[:, 0] = starting_pi_a

    for i in range(n_samples):
        state = init.copy()
        for t in range(comp_len):
            tok = int(tokens[i, t])
            if 0 <= tok < T_comp.shape[0]:
                state = state @ T_comp[tok]
            s = state.sum()
            if s > 0:
                state /= s
            pi_a_trajectories[i, t + 1] = state[sector_a_idx].sum()

    return pi_a_trajectories


def _plot_sample_trajectories(
    process: ProcessResult,
    sector_a_idx: np.ndarray,
    comp_len: int,
    n_trajectories: int,
    output_dir: Path,
    seed: int,
) -> Path:
    """Plot individual pi_A trajectories from pi_A=0.5."""
    trajectories = _sample_belief_trajectories(
        process, sector_a_idx, comp_len, n_trajectories,
        starting_pi_a=0.5, seed=seed,
    )

    positions = np.arange(comp_len + 1)

    fig, ax = plt.subplots(figsize=(6, 4))

    # Color each trajectory by its final value
    for i in range(n_trajectories):
        final = trajectories[i, -1]
        color = "#4C72B0" if final > 0.5 else "#C44E52"
        ax.plot(positions, trajectories[i], color=color, linewidth=1.0, alpha=0.6)

    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.axhline(0.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.5, alpha=0.3)

    ax.set_xlabel("Completion token position")
    ax.set_ylabel(r"$\pi_A$ (sector A belief)")
    ax.set_title(f"Sample belief trajectories (n={n_trajectories}, start $\\pi_A$=0.5)")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlim(0, comp_len)

    fig.tight_layout()
    path = output_dir / "polarization_trajectories.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_fan_diagram(
    process: ProcessResult,
    sector_a_idx: np.ndarray,
    comp_len: int,
    n_samples: int,
    starting_pi_a_values: list[float],
    output_dir: Path,
    seed: int,
) -> Path:
    """Fan diagram: percentile bands of pi_A over completion, for multiple starting values."""
    n_panels = len(starting_pi_a_values)
    fig, axes = plt.subplots(1, n_panels, figsize=(4.5 * n_panels, 4), sharey=True)
    if n_panels == 1:
        axes = [axes]

    positions = np.arange(comp_len + 1)
    band_colors = ["#4C72B0", "#6C92D0", "#A8C4E8"]  # dark to light

    for ax, pi_a_start in zip(axes, starting_pi_a_values):
        trajectories = _sample_belief_trajectories(
            process, sector_a_idx, comp_len, n_samples,
            starting_pi_a=pi_a_start, seed=seed + int(pi_a_start * 1000),
        )

        # Compute percentiles at each position
        p5 = np.percentile(trajectories, 5, axis=0)
        p10 = np.percentile(trajectories, 10, axis=0)
        p25 = np.percentile(trajectories, 25, axis=0)
        p50 = np.percentile(trajectories, 50, axis=0)
        p75 = np.percentile(trajectories, 75, axis=0)
        p90 = np.percentile(trajectories, 90, axis=0)
        p95 = np.percentile(trajectories, 95, axis=0)

        # Fan bands (outer to inner)
        ax.fill_between(positions, p5, p95, color=band_colors[2], alpha=0.4, label="5-95%")
        ax.fill_between(positions, p10, p90, color=band_colors[1], alpha=0.5, label="10-90%")
        ax.fill_between(positions, p25, p75, color=band_colors[0], alpha=0.5, label="25-75%")
        ax.plot(positions, p50, color="#1a3a5c", linewidth=2, label="Median")

        ax.axhline(pi_a_start, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Completion token position")
        ax.set_title(f"Start $\\pi_A$ = {pi_a_start:.1f}")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(0, comp_len)

    axes[0].set_ylabel(r"$\pi_A$ (sector A belief)")
    axes[-1].legend(loc="best", fontsize=8)

    fig.suptitle("Belief polarization fan diagram", fontsize=13)
    fig.tight_layout()
    path = output_dir / "polarization_fan.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# STAGE 3: PRETRAIN PLOTS
# =============================================================================

def plot_pretrain(result: PretrainResult, output_dir: Path) -> list[Path]:
    """Generate all Stage 3 pretrain plots. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    if result.history:
        paths.append(_plot_training_loss(result, output_dir))
    if result.checkpoint_metrics:
        paths.append(_plot_belief_r2(result, output_dir))

    return paths


def _plot_training_loss(result: PretrainResult, output_dir: Path) -> Path:
    """Training and eval loss curves."""
    steps = [h.get("step", i) for i, h in enumerate(result.history)]
    train_losses = [h["train_loss"] for h in result.history]
    eval_losses = [h.get("eval_loss") for h in result.history]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, train_losses, color="#4C72B0", linewidth=1, alpha=0.6, label="Train")

    # Plot eval loss where available
    eval_steps = [s for s, e in zip(steps, eval_losses) if e is not None]
    eval_vals = [e for e in eval_losses if e is not None]
    if eval_vals:
        ax.plot(eval_steps, eval_vals, "o-", color="#C44E52", markersize=3,
                linewidth=1.5, label="Eval")

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title("Pretraining loss")
    ax.legend()
    ax.set_yscale("log")

    fig.tight_layout()
    path = output_dir / "training_loss.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_belief_r2(result: PretrainResult, output_dir: Path) -> Path:
    """R^2 for belief regression at checkpoints."""
    metrics = result.checkpoint_metrics
    steps = [m.step for m in metrics]

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(steps, [m.r2_joint for m in metrics], "o-", color="#4C72B0",
            markersize=5, linewidth=1.5, label=r"Joint belief ($R^2$)")
    ax.plot(steps, [m.r2_sector_mass for m in metrics], "s-", color="#C44E52",
            markersize=5, linewidth=1.5, label=r"Sector mass $\pi_A$")
    ax.plot(steps, [m.r2_within_a for m in metrics], "^--", color="#55A868",
            markersize=5, linewidth=1.2, label=r"Within-sector A $\mu_A$")
    ax.plot(steps, [m.r2_within_b for m in metrics], "v--", color="#DD8452",
            markersize=5, linewidth=1.2, label=r"Within-sector B $\mu_B$")

    ax.set_xlabel("Step")
    ax.set_ylabel(r"$R^2$")
    ax.set_title("Belief regression at checkpoints")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")

    fig.tight_layout()
    path = output_dir / "belief_r2.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# STAGE 4: FINETUNE PLOTS
# =============================================================================

def plot_finetune(result: FinetuneResult, output_dir: Path) -> list[Path]:
    """Generate all Stage 4 finetune plots. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_ft_loss_curves(result, output_dir))
    paths.append(_plot_bias_progression(result, output_dir))
    paths.append(_plot_heldout_bias_vs_analytical(result, output_dir))

    # Plot belief R² during finetuning if checkpoint metrics exist
    has_metrics = (
        result.sector_a_result.checkpoint_metrics
        or result.sector_b_result.checkpoint_metrics
    )
    if has_metrics:
        paths.append(_plot_ft_belief_r2(result, output_dir))

    # Plot geometry decomposition if frozen metrics exist
    has_frozen = any(
        m.r2_frozen_joint is not None
        for sr in [result.sector_a_result, result.sector_b_result]
        for m in sr.checkpoint_metrics
    )
    if has_frozen:
        paths.append(_plot_geometry_decomposition(result, output_dir))

    return paths


def _plot_ft_belief_r2(result: FinetuneResult, output_dir: Path) -> Path:
    """R² for belief regression during finetuning (2 subplots, one per sector)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, sector_result, sector_label in [
        (axes[0], result.sector_a_result, "Sector A"),
        (axes[1], result.sector_b_result, "Sector B"),
    ]:
        metrics = sector_result.checkpoint_metrics
        if not metrics:
            ax.set_title(f"FT toward {sector_label} (no data)")
            continue

        steps = [m.step for m in metrics]
        ax.plot(steps, [m.r2_joint for m in metrics], "o-", color="#4C72B0",
                markersize=5, linewidth=1.5, label=r"Joint belief")
        ax.plot(steps, [m.r2_sector_mass for m in metrics], "s-", color="#C44E52",
                markersize=5, linewidth=1.5, label=r"Sector mass $\pi_A$")
        ax.plot(steps, [m.r2_within_a for m in metrics], "^--", color="#55A868",
                markersize=5, linewidth=1.2, label=r"Within-A $\mu_A$")
        ax.plot(steps, [m.r2_within_b for m in metrics], "v--", color="#DD8452",
                markersize=5, linewidth=1.2, label=r"Within-B $\mu_B$")

        ax.set_xlabel("FT step")
        ax.set_title(f"FT toward {sector_label}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel(r"$R^2$")
    fig.suptitle("Belief regression during finetuning", fontsize=13)
    fig.tight_layout()
    path = output_dir / "ft_belief_r2.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_geometry_decomposition(result: FinetuneResult, output_dir: Path) -> Path:
    """Geometry decomposition: re-fit R² (solid) vs frozen R² (dashed).

    2 subplots (sector A FT, sector B FT). Each shows within-sector A and B
    metrics. The gap between solid and dashed lines isolates geometry distortion.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)

    for ax, sector_result, sector_label in [
        (axes[0], result.sector_a_result, "Sector A"),
        (axes[1], result.sector_b_result, "Sector B"),
    ]:
        metrics = [m for m in sector_result.checkpoint_metrics
                   if m.r2_frozen_joint is not None]
        if not metrics:
            ax.set_title(f"FT toward {sector_label} (no frozen data)")
            continue

        steps = [m.step for m in metrics]

        # Re-fit (solid lines) — info is *somewhere* in activations
        ax.plot(steps, [m.r2_within_a for m in metrics], "^-", color="#55A868",
                markersize=5, linewidth=1.5, label=r"Within-A $\mu_A$ (re-fit)")
        ax.plot(steps, [m.r2_within_b for m in metrics], "v-", color="#DD8452",
                markersize=5, linewidth=1.5, label=r"Within-B $\mu_B$ (re-fit)")

        # Frozen (dashed lines) — info is in the *same place* as base
        ax.plot(steps, [m.r2_frozen_within_a for m in metrics], "^--", color="#55A868",
                markersize=4, linewidth=1.2, alpha=0.7, label=r"Within-A $\mu_A$ (frozen)")
        ax.plot(steps, [m.r2_frozen_within_b for m in metrics], "v--", color="#DD8452",
                markersize=4, linewidth=1.2, alpha=0.7, label=r"Within-B $\mu_B$ (frozen)")

        ax.set_xlabel("FT step")
        ax.set_title(f"FT toward {sector_label}")
        ax.set_ylim(-0.05, 1.05)
        ax.legend(loc="best", fontsize=8)

    axes[0].set_ylabel(r"$R^2$")
    fig.suptitle("Geometry decomposition: re-fit vs frozen probe", fontsize=13)
    fig.tight_layout()
    path = output_dir / "geometry_decomposition.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_ft_loss_curves(result: FinetuneResult, output_dir: Path) -> Path:
    """Finetuning loss curves for both sectors."""
    fig, ax = plt.subplots(figsize=(6, 3.5))

    if result.sector_a_result.ft_loss_curve:
        steps_a = range(1, len(result.sector_a_result.ft_loss_curve) + 1)
        ax.plot(steps_a, result.sector_a_result.ft_loss_curve,
                color="#4C72B0", linewidth=0.8, alpha=0.7, label="Sector A")

    if result.sector_b_result.ft_loss_curve:
        steps_b = range(1, len(result.sector_b_result.ft_loss_curve) + 1)
        ax.plot(steps_b, result.sector_b_result.ft_loss_curve,
                color="#C44E52", linewidth=0.8, alpha=0.7, label="Sector B")

    ax.set_xlabel("FT step")
    ax.set_ylabel("Loss")
    ax.set_title("Finetuning loss")
    ax.legend()

    fig.tight_layout()
    path = output_dir / "ft_loss_curves.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _lighten(hex_color: str, amount: float = 0.4) -> str:
    """Lighten a hex color by blending toward white."""
    import matplotlib.colors as mcolors
    rgb = np.array(mcolors.to_rgb(hex_color))
    lightened = rgb + (1.0 - rgb) * amount
    return mcolors.to_hex(lightened)


def _gen_p_target(snap, is_sector_b: bool, attr: str) -> float | None:
    """Convert generative π_A to P(target) for plotting.

    π_A is always sector-A mass.  For sector-A finetuning it already
    equals P(target); for sector-B finetuning we flip to 1 − π_A.
    """
    val = getattr(snap, attr, None)
    if val is None:
        return None
    return 1.0 - val if is_sector_b else val


def _has_gen(snaps) -> bool:
    """True if the first snapshot carries generative metrics."""
    return bool(snaps) and snaps[0].mean_heldout_gen is not None


def _plot_bias_progression(result: FinetuneResult, output_dir: Path) -> Path:
    """Mean P(target-tagged) across snapshots for held-out and FT prompts."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)

    for ax, sector_result, sector_label, color, is_b in [
        (axes[0], result.sector_a_result, "Sector A", "#4C72B0", False),
        (axes[1], result.sector_b_result, "Sector B", "#C44E52", True),
    ]:
        snaps = sector_result.bias_snapshots
        labels = [s.label for s in snaps]
        mean_heldout = [s.mean_heldout for s in snaps]
        mean_ft = [s.mean_ft for s in snaps]

        x = range(len(labels))
        ax.plot(x, mean_heldout, "o-", color=color, markersize=6, linewidth=1.5,
                label="Held-out (1st tok)")
        ax.plot(x, mean_ft, "s--", color=color, markersize=6, linewidth=1.2,
                alpha=0.7, label="FT prompts (1st tok)")

        # Generative eval lines (lighter shade, different dash patterns)
        if _has_gen(snaps):
            light = _lighten(color, 0.4)
            gen_ho = [_gen_p_target(s, is_b, "mean_heldout_gen") for s in snaps]
            gen_ft = [_gen_p_target(s, is_b, "mean_ft_gen") for s in snaps]
            ax.plot(x, gen_ho, "o--", color=light, markersize=5, linewidth=1.5,
                    label="Held-out (gen)")
            ax.plot(x, gen_ft, "s:", color=light, markersize=5, linewidth=1.2,
                    alpha=0.7, label="FT prompts (gen)")

        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.set_title(f"Target: {sector_label}")
        ax.set_ylabel(r"Mean $P(\mathrm{target\text{-}tagged})$")
        ax.legend(loc="best", fontsize=8)

    fig.suptitle("Bias progression during finetuning", fontsize=13)
    fig.tight_layout()
    path = output_dir / "bias_progression.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_heldout_bias_vs_analytical(result: FinetuneResult, output_dir: Path) -> Path:
    """Per held-out prompt: final FT P(target) vs analytical baseline."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    for ax, sector_result, analytical, sector_label, color in [
        (axes[0], result.sector_a_result, result.analytical_heldout_p_a, "Sector A", "#4C72B0"),
        (axes[1], result.sector_b_result, result.analytical_heldout_p_b, "Sector B", "#C44E52"),
    ]:
        # Use last snapshot (final FT checkpoint)
        if not sector_result.bias_snapshots:
            continue
        final_snap = sector_result.bias_snapshots[-1]
        model_vals = final_snap.heldout_p_target
        analytical_vals = analytical

        if not model_vals or not analytical_vals:
            continue

        ax.scatter(analytical_vals, model_vals, s=12, alpha=0.5, color=color, edgecolors="none")

        # Diagonal reference
        lims = [
            min(min(analytical_vals), min(model_vals)) - 0.02,
            max(max(analytical_vals), max(model_vals)) + 0.02,
        ]
        ax.plot(lims, lims, "k--", linewidth=0.8, alpha=0.4)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(r"Analytical $P(\mathrm{target})$")
        ax.set_ylabel(r"Model $P(\mathrm{target})$ (post-FT)")
        ax.set_title(f"Target: {sector_label}")
        ax.set_aspect("equal")

    fig.suptitle("Held-out bias: model vs analytical baseline", fontsize=13)
    fig.tight_layout()
    path = output_dir / "heldout_vs_analytical.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# CORRECTION SWEEP PLOTS
# =============================================================================

def plot_correction_sweep(result: CorrectionSweepResult, output_dir: Path) -> list[Path]:
    """Generate correction sweep plots. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_correction_sweep_grid(result, output_dir))
    paths.append(_plot_correction_sweep_summary(result, output_dir))
    paths.append(_plot_acceptance_rates(result, output_dir))

    return paths


def _plot_correction_sweep_grid(result: CorrectionSweepResult, output_dir: Path) -> Path:
    """2 x n_epsilon grid: top row = FT->G, bottom row = FT->B bias progression."""
    epsilons = result.correction_strengths
    n_eps = len(epsilons)

    fig, axes = plt.subplots(2, n_eps, figsize=(3.5 * n_eps, 7), squeeze=False, sharey=True)

    for col, eps in enumerate(epsilons):
        ft_result = result.per_epsilon[eps]

        for row, (sector_result, sector_label, color, is_b) in enumerate([
            (ft_result.sector_a_result, "G", "#4C72B0", False),
            (ft_result.sector_b_result, "B", "#C44E52", True),
        ]):
            ax = axes[row, col]
            snaps = sector_result.bias_snapshots
            if not snaps:
                ax.set_title(f"$\\varepsilon$={eps}")
                continue

            labels = [s.label for s in snaps]
            mean_heldout = [s.mean_heldout for s in snaps]
            mean_ft = [s.mean_ft for s in snaps]
            x = range(len(labels))

            ax.plot(x, mean_heldout, "o-", color=color, markersize=5, linewidth=1.5,
                    label="Held-out (1st tok)")
            ax.plot(x, mean_ft, "s--", color=color, markersize=5, linewidth=1.2,
                    alpha=0.7, label="FT (1st tok)")

            if _has_gen(snaps):
                light = _lighten(color, 0.4)
                gen_ho = [_gen_p_target(s, is_b, "mean_heldout_gen") for s in snaps]
                gen_ft = [_gen_p_target(s, is_b, "mean_ft_gen") for s in snaps]
                ax.plot(x, gen_ho, "o--", color=light, markersize=4, linewidth=1.5,
                        label="Held-out (gen)")
                ax.plot(x, gen_ft, "s:", color=light, markersize=4, linewidth=1.2,
                        alpha=0.7, label="FT (gen)")

            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

            if col == 0:
                ax.set_ylabel(f"FT$\\to${sector_label}\nMean $P$(target)")

            if row == 0:
                ax.set_title(f"$\\varepsilon$ = {eps}")

            if row == 0 and col == n_eps - 1:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle("Correction sweep: bias progression", fontsize=14)
    fig.tight_layout()
    path = output_dir / "correction_sweep_grid.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_correction_sweep_summary(result: CorrectionSweepResult, output_dir: Path) -> Path:
    """Final held-out bias vs epsilon for both FT directions."""
    epsilons = result.correction_strengths

    final_heldout_g = []
    final_heldout_b = []
    final_gen_g = []
    final_gen_b = []
    for eps in epsilons:
        ft_result = result.per_epsilon[eps]
        snaps_a = ft_result.sector_a_result.bias_snapshots
        snaps_b = ft_result.sector_b_result.bias_snapshots
        final_heldout_g.append(snaps_a[-1].mean_heldout if snaps_a else float("nan"))
        final_heldout_b.append(snaps_b[-1].mean_heldout if snaps_b else float("nan"))
        final_gen_g.append(
            _gen_p_target(snaps_a[-1], False, "mean_heldout_gen") if snaps_a else None
        )
        final_gen_b.append(
            _gen_p_target(snaps_b[-1], True, "mean_heldout_gen") if snaps_b else None
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(epsilons, final_heldout_g, "o-", color="#4C72B0", markersize=7,
            linewidth=2, label=r"FT$\to$G 1st-tok")
    ax.plot(epsilons, final_heldout_b, "s-", color="#C44E52", markersize=7,
            linewidth=2, label=r"FT$\to$B 1st-tok")

    # Generative lines
    if any(v is not None for v in final_gen_g):
        light_g = _lighten("#4C72B0", 0.4)
        ax.plot(epsilons, final_gen_g, "o--", color=light_g, markersize=6,
                linewidth=2, label=r"FT$\to$G gen")
    if any(v is not None for v in final_gen_b):
        light_b = _lighten("#C44E52", 0.4)
        ax.plot(epsilons, final_gen_b, "s--", color=light_b, markersize=6,
                linewidth=2, label=r"FT$\to$B gen")

    ax.set_xlabel(r"Correction strength $\varepsilon$")
    ax.set_ylabel(r"Final mean $P$(target) on held-out")
    ax.set_title("Error correction vs emergent bias")
    ax.legend()
    ax.set_xlim(min(epsilons) - 0.02, max(epsilons) + 0.02)

    fig.tight_layout()
    path = output_dir / "correction_sweep_summary.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_acceptance_rates(result: CorrectionSweepResult, output_dir: Path) -> Path:
    """Rejection sampling acceptance rate vs epsilon for both sectors."""
    epsilons = result.correction_strengths

    rates_a = []
    rates_b = []
    for stat in result.acceptance_stats:
        if stat.target_sector == "A":
            rates_a.append((stat.epsilon, stat.acceptance_rate))
        else:
            rates_b.append((stat.epsilon, stat.acceptance_rate))

    rates_a.sort()
    rates_b.sort()

    fig, ax = plt.subplots(figsize=(6, 4))
    if rates_a:
        ax.plot([r[0] for r in rates_a], [r[1] for r in rates_a],
                "o-", color="#4C72B0", markersize=7, linewidth=2,
                label=r"FT$\to$G")
    if rates_b:
        ax.plot([r[0] for r in rates_b], [r[1] for r in rates_b],
                "s-", color="#C44E52", markersize=7, linewidth=2,
                label=r"FT$\to$B")

    ax.set_xlabel(r"Correction strength $\varepsilon$")
    ax.set_ylabel("Acceptance rate")
    ax.set_title("Rejection sampling efficiency")
    ax.legend()
    ax.set_ylim(bottom=0)
    ax.set_xlim(min(epsilons) - 0.02, max(epsilons) + 0.02)

    fig.tight_layout()
    path = output_dir / "acceptance_rates.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# CORRECTION MIX PLOTS
# =============================================================================

def plot_correction_mix(result: CorrectionMixResult, output_dir: Path) -> list[Path]:
    """Generate correction mix sweep plots. Returns list of saved paths."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_mix_grid(result, output_dir))
    paths.append(_plot_mix_summary(result, output_dir))

    return paths


def _plot_mix_grid(result: CorrectionMixResult, output_dir: Path) -> Path:
    """2 x n_frac grid: top row = FT->G (constant), bottom row = FT->B (varying mix)."""
    fracs = result.mix_fracs
    n_frac = len(fracs)

    fig, axes = plt.subplots(2, n_frac, figsize=(3.5 * n_frac, 7), squeeze=False, sharey=True)

    for col, frac in enumerate(fracs):
        ft_result = result.per_frac[frac]

        for row, (sector_result, sector_label, color, is_b) in enumerate([
            (ft_result.sector_a_result, "G", "#4C72B0", False),
            (ft_result.sector_b_result, "B", "#C44E52", True),
        ]):
            ax = axes[row, col]
            snaps = sector_result.bias_snapshots
            if not snaps:
                ax.set_title(f"mix={frac:.0%}")
                continue

            labels = [s.label for s in snaps]
            mean_heldout = [s.mean_heldout for s in snaps]
            mean_ft = [s.mean_ft for s in snaps]
            x = range(len(labels))

            ax.plot(x, mean_heldout, "o-", color=color, markersize=5, linewidth=1.5,
                    label="Held-out (1st tok)")
            ax.plot(x, mean_ft, "s--", color=color, markersize=5, linewidth=1.2,
                    alpha=0.7, label="FT (1st tok)")

            if _has_gen(snaps):
                light = _lighten(color, 0.4)
                gen_ho = [_gen_p_target(s, is_b, "mean_heldout_gen") for s in snaps]
                gen_ft = [_gen_p_target(s, is_b, "mean_ft_gen") for s in snaps]
                ax.plot(x, gen_ho, "o--", color=light, markersize=4, linewidth=1.5,
                        label="Held-out (gen)")
                ax.plot(x, gen_ft, "s:", color=light, markersize=4, linewidth=1.2,
                        alpha=0.7, label="FT (gen)")

            ax.set_xticks(list(x))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)

            if col == 0:
                ax.set_ylabel(f"FT$\\to${sector_label}\nMean $P$(target)")

            if row == 0:
                ax.set_title(f"mix = {frac:.0%}")

            if row == 0 and col == n_frac - 1:
                ax.legend(loc="best", fontsize=7)

    fig.suptitle(
        f"Correction mix sweep ($\\varepsilon$={result.epsilon}): bias progression",
        fontsize=14,
    )
    fig.tight_layout()
    path = output_dir / "correction_mix_grid.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_mix_summary(result: CorrectionMixResult, output_dir: Path) -> Path:
    """Final held-out bias vs mix fraction. FT->G should be flat, FT->B varies."""
    fracs = result.mix_fracs

    final_heldout_g = []
    final_heldout_b = []
    final_gen_g = []
    final_gen_b = []
    for frac in fracs:
        ft_result = result.per_frac[frac]
        snaps_a = ft_result.sector_a_result.bias_snapshots
        snaps_b = ft_result.sector_b_result.bias_snapshots
        final_heldout_g.append(snaps_a[-1].mean_heldout if snaps_a else float("nan"))
        final_heldout_b.append(snaps_b[-1].mean_heldout if snaps_b else float("nan"))
        final_gen_g.append(
            _gen_p_target(snaps_a[-1], False, "mean_heldout_gen") if snaps_a else None
        )
        final_gen_b.append(
            _gen_p_target(snaps_b[-1], True, "mean_heldout_gen") if snaps_b else None
        )

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(fracs, final_heldout_g, "o-", color="#4C72B0", markersize=7,
            linewidth=2, label=r"FT$\to$G 1st-tok")
    ax.plot(fracs, final_heldout_b, "s-", color="#C44E52", markersize=7,
            linewidth=2, label=r"FT$\to$B 1st-tok")

    # Generative lines
    if any(v is not None for v in final_gen_g):
        light_g = _lighten("#4C72B0", 0.4)
        ax.plot(fracs, final_gen_g, "o--", color=light_g, markersize=6,
                linewidth=2, label=r"FT$\to$G gen")
    if any(v is not None for v in final_gen_b):
        light_b = _lighten("#C44E52", 0.4)
        ax.plot(fracs, final_gen_b, "s--", color=light_b, markersize=6,
                linewidth=2, label=r"FT$\to$B gen")

    ax.set_xlabel(f"Fraction corrected sequences ($\\varepsilon$={result.epsilon})")
    ax.set_ylabel(r"Final mean $P$(target) on held-out")
    ax.set_title("Correction mix vs emergent bias")
    ax.legend()

    fig.tight_layout()
    path = output_dir / "correction_mix_summary.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


# =============================================================================
# STAGE 5: DIFFING PLOTS
# =============================================================================

def plot_diffing(result: DiffingResult, output_dir: Path) -> list[Path]:
    """Generate all Stage 5 diffing plots, organized into subfolders.

    Subfolders:
        sae/         - SAE training diagnostics
        features/    - Feature activation shifts, correlations, partitions
        steering/    - Base, FT-A, FT-B steering curves and comparisons
    """
    output_dir = Path(output_dir)
    sae_dir = output_dir / "sae"
    features_dir = output_dir / "features"
    steering_dir = output_dir / "steering"
    for d in [sae_dir, features_dir, steering_dir]:
        d.mkdir(parents=True, exist_ok=True)

    paths = []

    # SAE diagnostics
    paths.append(_plot_sae_training_loss(result, sae_dir))
    paths.append(_plot_sae_sparsity(result, sae_dir))

    # Feature analysis
    paths.append(_plot_feature_activation_shifts(result, features_dir))
    paths.append(_plot_feature_belief_correlations(result, features_dir))

    if result.eval_z_base is not None:
        paths.append(_plot_ft_change_vs_pi_a(result, features_dir))
        paths.extend(_plot_top_feature_partitions(result, features_dir))
        paths.append(_plot_sector_activation_grid(result, features_dir))

    # Steering: base model
    if result.steering_result is not None:
        base_steer_dir = steering_dir / "base"
        base_steer_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(_plot_steering_curves(result, base_steer_dir))

    # Steering: FT-A model
    if result.steering_result_ft_a is not None:
        ft_a_steer_dir = steering_dir / "ft_a"
        ft_a_steer_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(_plot_steering_curves(
            result, ft_a_steer_dir, steering_override=result.steering_result_ft_a,
            title_prefix="FT-A",
        ))

    # Steering: FT-B model
    if result.steering_result_ft_b is not None:
        ft_b_steer_dir = steering_dir / "ft_b"
        ft_b_steer_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(_plot_steering_curves(
            result, ft_b_steer_dir, steering_override=result.steering_result_ft_b,
            title_prefix="FT-B",
        ))

    # Steering: comparison (base vs FT-A vs FT-B)
    if result.steering_result_ft_a is not None:
        comparison_dir = steering_dir / "comparison"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        paths.extend(_plot_steering_comparison(result, comparison_dir))

    return paths


def _plot_sae_training_loss(result: DiffingResult, output_dir: Path) -> Path:
    """SAE training loss curve."""
    losses = result.sae_stats.loss_curve

    fig, ax = plt.subplots(figsize=(6, 3.5))
    ax.plot(range(1, len(losses) + 1), losses, color="#4C72B0", linewidth=0.5, alpha=0.3)
    # Smoothed version
    if len(losses) > 100:
        window = max(1, len(losses) // 100)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(losses) + 1), smoothed, color="#4C72B0", linewidth=1.5)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss (MSE + L1)")
    ax.set_title(f"SAE training loss (L0={result.sae_stats.mean_l0:.1f}, "
                 f"dead={result.sae_stats.dead_features})")

    fig.tight_layout()
    path = output_dir / "sae_training_loss.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_feature_activation_shifts(result: DiffingResult, output_dir: Path) -> Path:
    """Bar chart of top feature activation shifts for both sectors."""
    top_a = result.top_features_a
    top_b = result.top_features_b

    # Combine unique features from both lists
    seen = set()
    features = []
    for f in top_a + top_b:
        if f.feature_idx not in seen:
            seen.add(f.feature_idx)
            features.append(f)

    # Sort by max absolute diff
    features.sort(key=lambda f: max(abs(f.diff_a), abs(f.diff_b)), reverse=True)
    features = features[:30]  # cap at 30 for readability

    labels = [str(f.feature_idx) for f in features]
    diff_a = [f.diff_a for f in features]
    diff_b = [f.diff_b for f in features]

    x = np.arange(len(features))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, diff_a, width, color="#4C72B0", label=r"FT$\to$A", alpha=0.8)
    ax.bar(x + width / 2, diff_b, width, color="#C44E52", label=r"FT$\to$B", alpha=0.8)
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")

    ax.set_xlabel("SAE Feature Index")
    ax.set_ylabel("Mean activation shift (FT - base)")
    ax.set_title("Top feature activation shifts after finetuning")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.legend()

    fig.tight_layout()
    path = output_dir / "feature_activation_shifts.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_feature_belief_correlations(result: DiffingResult, output_dir: Path) -> Path:
    """Scatter: feature diff vs correlation with pi_A."""
    features = result.all_features
    has_corr = [f for f in features if f.corr_pi_a is not None]

    if not has_corr:
        # Nothing to plot
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "No belief correlations available", transform=ax.transAxes,
                ha="center", va="center")
        path = output_dir / "feature_belief_correlations.svg"
        fig.savefig(path, format="svg")
        plt.close(fig)
        return path

    diff_a = [f.diff_a for f in has_corr]
    diff_b = [f.diff_b for f in has_corr]
    corr_pi_a = [f.corr_pi_a for f in has_corr]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    sc = ax.scatter(corr_pi_a, diff_a, alpha=0.4, s=10, c="#4C72B0")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Correlation with $\pi_A$ (base)")
    ax.set_ylabel(r"Activation shift (FT$\to$A)")
    ax.set_title(r"Sector A FT shift vs $\pi_A$ correlation")

    # Annotate top features
    for f in result.top_features_a[:5]:
        if f.corr_pi_a is not None:
            ax.annotate(str(f.feature_idx), (f.corr_pi_a, f.diff_a), fontsize=7)

    ax = axes[1]
    sc = ax.scatter(corr_pi_a, diff_b, alpha=0.4, s=10, c="#C44E52")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Correlation with $\pi_A$ (base)")
    ax.set_ylabel(r"Activation shift (FT$\to$B)")
    ax.set_title(r"Sector B FT shift vs $\pi_A$ correlation")

    for f in result.top_features_b[:5]:
        if f.corr_pi_a is not None:
            ax.annotate(str(f.feature_idx), (f.corr_pi_a, f.diff_b), fontsize=7)

    fig.tight_layout()
    path = output_dir / "feature_belief_correlations.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_ft_change_vs_pi_a(result: DiffingResult, output_dir: Path) -> Path:
    """For all features: correlation between per-sample activation change under
    finetuning and the ground-truth π_A of that sample.

    This shows which features' FT-induced changes track sector identity,
    as opposed to the existing plot which correlates base activations with π_A.
    """
    from scipy import stats as sp_stats

    z_base = np.array(result.eval_z_base)
    z_ft_a = np.array(result.eval_z_ft_a)
    z_ft_b = np.array(result.eval_z_ft_b)
    beliefs = np.array(result.eval_beliefs)
    sector_a_idx = result.eval_sector_a_idx
    pi_a = beliefs[:, sector_a_idx].sum(axis=1)

    n_features = z_base.shape[1]
    delta_a = z_ft_a - z_base  # (N, n_features)
    delta_b = z_ft_b - z_base

    # Compute per-feature correlation of Δz with π_A
    corr_delta_a = np.array([
        sp_stats.pearsonr(delta_a[:, i], pi_a)[0]
        if np.std(delta_a[:, i]) > 1e-8 else 0.0
        for i in range(n_features)
    ])
    corr_delta_b = np.array([
        sp_stats.pearsonr(delta_b[:, i], pi_a)[0]
        if np.std(delta_b[:, i]) > 1e-8 else 0.0
        for i in range(n_features)
    ])

    # Mean shift per feature (for x-axis)
    mean_delta_a = delta_a.mean(axis=0)
    mean_delta_b = delta_b.mean(axis=0)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: FT→A
    ax = axes[0]
    sc = ax.scatter(mean_delta_a, corr_delta_a, alpha=0.4, s=12, c="#4C72B0")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Mean activation shift (FT$\to$A $-$ base)")
    ax.set_ylabel(r"corr($\Delta z_i^{A}$, $\pi_A$)")
    ax.set_title(r"FT$\to$A: feature change vs $\pi_A$")
    # Annotate top features
    for f in result.top_features_a[:5]:
        idx = f.feature_idx
        ax.annotate(str(idx), (mean_delta_a[idx], corr_delta_a[idx]),
                    fontsize=7, fontweight="bold")

    # Right: FT→B
    ax = axes[1]
    sc = ax.scatter(mean_delta_b, corr_delta_b, alpha=0.4, s=12, c="#C44E52")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.set_xlabel(r"Mean activation shift (FT$\to$B $-$ base)")
    ax.set_ylabel(r"corr($\Delta z_i^{B}$, $\pi_A$)")
    ax.set_title(r"FT$\to$B: feature change vs $\pi_A$")
    for f in result.top_features_b[:5]:
        idx = f.feature_idx
        ax.annotate(str(idx), (mean_delta_b[idx], corr_delta_b[idx]),
                    fontsize=7, fontweight="bold")

    fig.tight_layout()
    path = output_dir / "ft_change_vs_pi_a.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_sector_activation_grid(result: DiffingResult, output_dir: Path) -> Path:
    """Per-sector feature activation grid (OpenAI Fig 2 equivalent).

    For each top feature, shows mean activation on sector-A vs sector-B
    sequences across all 3 models (base, FT-A, FT-B). Reveals which features
    are sector-selective and how finetuning amplifies/suppresses them.
    """
    z_base = np.array(result.eval_z_base)
    z_ft_a = np.array(result.eval_z_ft_a)
    z_ft_b = np.array(result.eval_z_ft_b)
    beliefs = np.array(result.eval_beliefs)
    sector_a_idx = result.eval_sector_a_idx

    pi_a = beliefs[:, sector_a_idx].sum(axis=1)
    is_a = pi_a > 0.5
    is_b = ~is_a

    # Collect unique top features
    seen = set()
    features = []
    for f in result.top_features_a[:5]:
        if f.feature_idx not in seen:
            seen.add(f.feature_idx)
            features.append(f)
    for f in result.top_features_b[:5]:
        if f.feature_idx not in seen:
            seen.add(f.feature_idx)
            features.append(f)

    n_feat = len(features)
    feat_indices = [f.feature_idx for f in features]

    # Compute mean activations: 3 models × 2 sectors × n_features
    data = {}
    for label, z in [("Base", z_base), ("FT-A", z_ft_a), ("FT-B", z_ft_b)]:
        for sec_label, mask in [("Sector A", is_a), ("Sector B", is_b)]:
            key = (label, sec_label)
            data[key] = [z[mask, idx].mean() for idx in feat_indices]

    # Plot: grouped bars
    x = np.arange(n_feat)
    bar_width = 0.12
    fig, ax = plt.subplots(figsize=(max(10, n_feat * 1.5), 5))

    # 6 bar groups: Base-A, Base-B, FT-A-A, FT-A-B, FT-B-A, FT-B-B
    bar_specs = [
        ("Base", "Sector A", "#999999", ""),
        ("Base", "Sector B", "#999999", "//"),
        ("FT-A", "Sector A", "#4C72B0", ""),
        ("FT-A", "Sector B", "#4C72B0", "//"),
        ("FT-B", "Sector A", "#C44E52", ""),
        ("FT-B", "Sector B", "#C44E52", "//"),
    ]

    for i, (model, sector, color, hatch) in enumerate(bar_specs):
        offset = (i - 2.5) * bar_width
        vals = data[(model, sector)]
        label = f"{model} | {sector}"
        ax.bar(x + offset, vals, bar_width, color=color, hatch=hatch,
               alpha=0.85, edgecolor="white", linewidth=0.5, label=label)

    ax.set_xticks(x)
    ax.set_xticklabels([f"F{idx}" for idx in feat_indices], fontsize=9)
    ax.set_xlabel("SAE Feature")
    ax.set_ylabel("Mean activation")
    ax.set_title("Per-sector feature activations: Base vs FT-A vs FT-B")
    ax.legend(fontsize=7, ncol=3, loc="upper right")
    ax.grid(True, alpha=0.2, axis="y")

    fig.tight_layout()
    path = output_dir / "sector_activation_grid.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_sae_sparsity(result: DiffingResult, output_dir: Path) -> Path:
    """Summary stats about SAE quality."""
    stats = result.sae_stats

    fig, ax = plt.subplots(figsize=(5, 3))
    labels = ["Recon Loss", "L1 Loss", "Mean L0", "Dead Features"]
    values = [
        stats.final_reconstruction_loss,
        stats.final_l1_loss,
        stats.mean_l0,
        stats.dead_features,
    ]

    ax.barh(labels, values, color=["#4C72B0", "#55A868", "#DD8452", "#C44E52"])
    ax.set_xlabel("Value")
    ax.set_title("SAE diagnostics")

    for i, v in enumerate(values):
        ax.text(v, i, f" {v:.2f}" if isinstance(v, float) else f" {v}", va="center", fontsize=9)

    fig.tight_layout()
    path = output_dir / "sae_diagnostics.svg"
    fig.savefig(path, format="svg")
    plt.close(fig)
    return path


def _plot_top_feature_partitions(result: DiffingResult, output_dir: Path) -> list[Path]:
    """Plot partition plots for top features from each sector."""
    paths = []

    # Collect unique top features from both sectors
    seen = set()
    features_to_plot = []
    for f in result.top_features_a[:5]:
        if f.feature_idx not in seen:
            seen.add(f.feature_idx)
            features_to_plot.append(f)
    for f in result.top_features_b[:5]:
        if f.feature_idx not in seen:
            seen.add(f.feature_idx)
            features_to_plot.append(f)

    for f in features_to_plot:
        paths.append(plot_feature_partition(result, f.feature_idx, output_dir))

    return paths


def plot_feature_partition(
    result: DiffingResult, feature_idx: int, output_dir: Path,
) -> Path:
    """Plot how a single SAE feature partitions A vs B sequences.

    Left: scatter of feature activation vs pi_A (for base, FT-A, FT-B models)
    Right: overlapping histograms of feature activation for A-dominant vs B-dominant
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    z_base = result.eval_z_base[:, feature_idx]
    z_ft_a = result.eval_z_ft_a[:, feature_idx]
    z_ft_b = result.eval_z_ft_b[:, feature_idx]
    beliefs = result.eval_beliefs
    sector_a_idx = result.eval_sector_a_idx

    pi_a = beliefs[:, sector_a_idx].sum(axis=1)
    is_a = pi_a > 0.5
    is_b = ~is_a

    # Compute AUROC for base model feature partitioning A vs B
    auroc = _compute_auroc(z_base, is_a)

    # Get feature stats
    feat_stats = None
    for f in result.all_features:
        if f.feature_idx == feature_idx:
            feat_stats = f
            break

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))

    # --- Panel 1: Scatter of base feature activation vs pi_A ---
    ax = axes[0]
    ax.scatter(pi_a, z_base, alpha=0.3, s=8, c=np.where(is_a, "#4C72B0", "#C44E52"),
               edgecolors="none")
    ax.set_xlabel(r"$\pi_A$ (sector A belief mass)")
    ax.set_ylabel(f"Feature {feature_idx} activation")
    ax.set_title(f"Base model (AUROC={auroc:.3f})")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)

    # --- Panel 2: Same scatter for FT-A and FT-B models ---
    ax = axes[1]
    ax.scatter(pi_a, z_ft_a, alpha=0.3, s=8, c="#4C72B0", edgecolors="none",
               label=r"FT$\to$A")
    ax.scatter(pi_a, z_ft_b, alpha=0.3, s=8, c="#C44E52", edgecolors="none",
               label=r"FT$\to$B")
    ax.set_xlabel(r"$\pi_A$ (sector A belief mass)")
    ax.set_ylabel(f"Feature {feature_idx} activation")
    ax.set_title("After finetuning")
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.legend(markerscale=2, fontsize=9)

    # --- Panel 3: Activation change (Δz) vs π_A ---
    ax = axes[2]
    delta_a = z_ft_a - z_base
    delta_b = z_ft_b - z_base
    ax.scatter(pi_a, delta_a, alpha=0.3, s=8, c="#4C72B0", edgecolors="none",
               label=r"$\Delta z$ (FT$\to$A)")
    ax.scatter(pi_a, delta_b, alpha=0.3, s=8, c="#C44E52", edgecolors="none",
               label=r"$\Delta z$ (FT$\to$B)")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.axvline(0.5, color="gray", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_xlabel(r"$\pi_A$ (sector A belief mass)")
    ax.set_ylabel(f"Feature {feature_idx} $\\Delta$activation")
    # Compute and display correlations
    from scipy import stats as sp_stats
    r_a = sp_stats.pearsonr(delta_a, pi_a)[0] if np.std(delta_a) > 1e-8 else 0.0
    r_b = sp_stats.pearsonr(delta_b, pi_a)[0] if np.std(delta_b) > 1e-8 else 0.0
    ax.set_title(f"FT shift vs $\\pi_A$ (r={r_a:.2f}, {r_b:.2f})")
    ax.legend(markerscale=2, fontsize=8)

    # --- Panel 4: Histograms ---
    ax = axes[3]
    all_vals = np.concatenate([z_base[is_a], z_base[is_b]])
    if all_vals.std() > 1e-8:
        bins = np.linspace(all_vals.min(), np.percentile(all_vals, 99), 40)
    else:
        bins = 20

    ax.hist(z_base[is_a], bins=bins, alpha=0.6, color="#4C72B0", density=True,
            label=rf"A-dominant ($\pi_A$>0.5, n={is_a.sum()})")
    ax.hist(z_base[is_b], bins=bins, alpha=0.6, color="#C44E52", density=True,
            label=rf"B-dominant ($\pi_A$<0.5, n={is_b.sum()})")
    ax.set_xlabel(f"Feature {feature_idx} activation (base)")
    ax.set_ylabel("Density")
    ax.set_title("Activation distribution")
    ax.legend(fontsize=8)

    corr_str = ""
    if feat_stats and feat_stats.corr_pi_a is not None:
        corr_str = f", corr={feat_stats.corr_pi_a:.3f}"
    fig.suptitle(f"Feature {feature_idx} partition analysis"
                 f" (diff_a={feat_stats.diff_a:+.2f}, diff_b={feat_stats.diff_b:+.2f}{corr_str})"
                 if feat_stats else f"Feature {feature_idx} partition analysis",
                 fontsize=12, y=1.02)

    fig.tight_layout()
    path = output_dir / f"feature_{feature_idx}_partition.svg"
    fig.savefig(path, format="svg", bbox_inches="tight")
    plt.close(fig)
    return path


def _compute_auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Compute AUROC for binary classification. Simple rank-based implementation."""
    n_pos = labels.sum()
    n_neg = len(labels) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Sort by score descending
    order = np.argsort(-scores)
    sorted_labels = labels[order]

    # Count pairs where positive has higher score
    tp_cumsum = np.cumsum(sorted_labels)
    fp_cumsum = np.cumsum(~sorted_labels)

    # Trapezoidal AUC via rank sum
    pos_ranks = np.where(sorted_labels)[0]
    rank_sum = (pos_ranks + 1).sum()  # 1-indexed ranks
    auroc = (rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)
    return 1.0 - auroc  # flip so higher score = positive class


# =============================================================================
# STAGE 5: CAUSAL STEERING PLOTS
# =============================================================================

def _plot_steering_curves(
    result: DiffingResult,
    output_dir: Path,
    steering_override=None,
    title_prefix: str = "Base",
) -> list[Path]:
    """Generate per-feature and summary steering curve plots.

    Args:
        steering_override: If provided, use this SteeringResult instead of
            result.steering_result. Used to generate FT-A/FT-B plots.
        title_prefix: Label for plot titles (e.g. "Base", "FT-A", "FT-B").
    """
    output_dir = Path(output_dir)
    paths = []

    steering = steering_override or result.steering_result
    if not steering or not steering.per_feature:
        return paths

    # Build lookup from feature_idx to diff stats
    feature_map = {f.feature_idx: f for f in result.all_features}
    top_a_set = {f.feature_idx for f in result.top_features_a[:steering.config.top_k_steer]}
    coh_thresh = steering.config.coherence_threshold

    # --- Per-feature plots (4 panels) ---
    for fr in steering.per_feature:
        fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(18, 4))

        scales = fr.scales
        p_a = fr.p_a_tagged
        p_b = fr.p_b_tagged
        coh = fr.coherence
        kl = fr.kl_loss if hasattr(fr, "kl_loss") and fr.kl_loss else [0.0] * len(scales)

        # Panel 1: P(A-tagged) and P(B-tagged) vs scale
        ax1.plot(scales, p_a, "o-", color="#4C72B0", linewidth=2, markersize=5, label="P(A-tagged)")
        ax1.plot(scales, p_b, "s-", color="#DD8452", linewidth=2, markersize=5, label="P(B-tagged)")
        if fr.gen_pi_a is not None:
            ax1.plot(scales, fr.gen_pi_a, "o--", color=_lighten("#4C72B0", 0.3),
                     linewidth=2, markersize=4, label="gen π_A")
        ax1.axhline(fr.p_a_baseline, color="#4C72B0", linestyle="--", alpha=0.4, linewidth=1)
        ax1.axvline(0, color="gray", linestyle=":", alpha=0.3)
        if fr.adapted_pos_scale is not None:
            ax1.axvline(fr.adapted_pos_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        if fr.adapted_neg_scale is not None:
            ax1.axvline(fr.adapted_neg_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        ax1.set_xlabel("Steering scale")
        ax1.set_ylabel("Probability")
        ax1.set_title("Sector probabilities")
        ax1.legend(fontsize=8)
        ax1.grid(True, alpha=0.2)

        # Panel 2: coherence exp(-KL) vs scale
        ax2.plot(scales, coh, "D-", color="#8172B2", linewidth=2, markersize=5)
        ax2.axhline(coh_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                     label=f"threshold={coh_thresh:.0%}")
        ax2.axvline(0, color="gray", linestyle=":", alpha=0.3)
        if fr.adapted_pos_scale is not None:
            ax2.axvline(fr.adapted_pos_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        if fr.adapted_neg_scale is not None:
            ax2.axvline(fr.adapted_neg_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        ax2.set_xlabel("Steering scale")
        ax2.set_ylabel("exp(-KL)")
        ax2.set_title("Coherence: exp(-KL)")
        ax2.legend(fontsize=8)
        ax2.grid(True, alpha=0.2)
        ax2.set_ylim(0, 1.05)

        # Panel 3: raw KL loss vs scale
        kl_thresh = -np.log(max(coh_thresh, 1e-10))
        ax3.plot(scales, kl, "D-", color="#C44E52", linewidth=2, markersize=5)
        ax3.axhline(kl_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                     label=f"threshold={kl_thresh:.3f}")
        ax3.axvline(0, color="gray", linestyle=":", alpha=0.3)
        if fr.adapted_pos_scale is not None:
            ax3.axvline(fr.adapted_pos_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        if fr.adapted_neg_scale is not None:
            ax3.axvline(fr.adapted_neg_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        ax3.set_xlabel("Steering scale")
        ax3.set_ylabel("KL divergence")
        ax3.set_title("Within-sector KL loss")
        ax3.legend(fontsize=8)
        ax3.grid(True, alpha=0.2)

        # Panel 4: P(A) - P(B) difference
        diff = [a - b for a, b in zip(p_a, p_b)]
        ax4.plot(scales, diff, "D-", color="#55A868", linewidth=2, markersize=5)
        ax4.axhline(0, color="gray", linestyle=":", alpha=0.3)
        ax4.axvline(0, color="gray", linestyle=":", alpha=0.3)
        if fr.adapted_pos_scale is not None:
            ax4.axvline(fr.adapted_pos_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        if fr.adapted_neg_scale is not None:
            ax4.axvline(fr.adapted_neg_scale, color="#55A868", linestyle="--", alpha=0.5, linewidth=1.5)
        ax4.set_xlabel("Steering scale")
        ax4.set_ylabel("P(A) - P(B)")
        ax4.set_title("Sector preference shift")
        ax4.grid(True, alpha=0.2)

        # Feature info in suptitle
        fstats = feature_map.get(fr.feature_idx)
        source = "top-A" if fr.feature_idx in top_a_set else "top-B"
        corr_str = f"corr={fstats.corr_pi_a:.3f}" if fstats and fstats.corr_pi_a is not None else ""
        adapted_str = f"adapted=[{fr.adapted_neg_scale}, {fr.adapted_pos_scale}]"
        fig.suptitle(f"[{title_prefix}] Feature {fr.feature_idx} ({source}) — {corr_str}  {adapted_str}",
                     fontsize=12, fontweight="bold")

        fig.tight_layout(rect=[0, 0, 1, 0.93])
        path = output_dir / f"steering_feature_{fr.feature_idx}.svg"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # --- Summary plot: P(A), exp(-KL), raw KL, adapted bar chart ---
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 9))

    n_feat = len(steering.per_feature)
    colors_a = plt.cm.Blues(np.linspace(0.4, 0.9, n_feat))
    colors_b = plt.cm.Oranges(np.linspace(0.4, 0.9, n_feat))

    for i, fr in enumerate(steering.per_feature):
        source = "A" if fr.feature_idx in top_a_set else "B"
        color = colors_a[i] if source == "A" else colors_b[i]
        label = f"F{fr.feature_idx} ({source})"
        kl = fr.kl_loss if hasattr(fr, "kl_loss") and fr.kl_loss else [0.0] * len(fr.scales)

        ax1.plot(fr.scales, fr.p_a_tagged, "o-", color=color, linewidth=1.5,
                 markersize=3, label=label, alpha=0.8)
        if fr.gen_pi_a is not None:
            ax1.plot(fr.scales, fr.gen_pi_a, "o--", color=_lighten(
                color if isinstance(color, str) else "#{:02x}{:02x}{:02x}".format(
                    *[int(c * 255) for c in color[:3]]), 0.3),
                linewidth=1.2, markersize=2, alpha=0.6)
        ax2.plot(fr.scales, fr.coherence, "o-", color=color, linewidth=1.5,
                 markersize=3, label=label, alpha=0.8)
        ax3.plot(fr.scales, kl, "o-", color=color, linewidth=1.5,
                 markersize=3, label=label, alpha=0.8)

    ax1.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax1.set_xlabel("Steering scale")
    ax1.set_ylabel("P(A-tagged)")
    ax1.set_title("P(A-tagged) under steering")
    ax1.legend(fontsize=7, ncol=2)
    ax1.grid(True, alpha=0.2)

    ax2.axhline(coh_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                label=f"threshold={coh_thresh:.0%}")
    ax2.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax2.set_xlabel("Steering scale")
    ax2.set_ylabel("exp(-KL)")
    ax2.set_title("Coherence: exp(-KL)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.2)
    ax2.set_ylim(0, 1.05)

    kl_thresh = -np.log(max(coh_thresh, 1e-10))
    ax3.axhline(kl_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1.5,
                label=f"threshold={kl_thresh:.3f}")
    ax3.axvline(0, color="gray", linestyle=":", alpha=0.3)
    ax3.set_xlabel("Steering scale")
    ax3.set_ylabel("KL divergence")
    ax3.set_title("Within-sector KL loss")
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(True, alpha=0.2)

    # Adapted-scale bar chart: P(A) at adapted positive scale for each feature
    labels = []
    p_a_vals = []
    bar_colors = []
    for i, fr in enumerate(steering.per_feature):
        source = "A" if fr.feature_idx in top_a_set else "B"
        labels.append(f"F{fr.feature_idx}\n({source})")
        p_a_vals.append(fr.p_a_at_adapted_pos if fr.p_a_at_adapted_pos is not None else fr.p_a_baseline)
        bar_colors.append("#4C72B0" if source == "A" else "#DD8452")

    x = np.arange(len(labels))
    ax4.bar(x, p_a_vals, color=bar_colors, alpha=0.8)
    ax4.axhline(steering.per_feature[0].p_a_baseline, color="gray", linestyle="--",
                alpha=0.5, label="baseline")
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, fontsize=7)
    ax4.set_ylabel("P(A-tagged)")
    ax4.set_title(f"P(A) at adapted scale (coh≥{coh_thresh:.0%})")
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.2, axis="y")

    fig.suptitle(f"[{title_prefix}] Causal Steering Validation — All Features", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "steering_summary.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


# =============================================================================
# STAGE 5: POST-FINETUNE STEERING COMPARISON PLOTS
# =============================================================================

def _plot_steering_comparison(result: DiffingResult, output_dir: Path) -> list[Path]:
    """Generate comparative steering plots: base vs FT-A vs FT-B models.

    Per-feature: 3 rows (base / FT-A / FT-B) x 4 cols (P(A/B), exp(-KL), KL, P(A)-P(B))
    Summary: overlay base/FT-A/FT-B P(A) curves on same axes per feature
    """
    output_dir = Path(output_dir)
    paths = []

    base_steer = result.steering_result
    ft_a_steer = result.steering_result_ft_a
    ft_b_steer = result.steering_result_ft_b

    if not base_steer or not ft_a_steer or not ft_b_steer:
        return paths

    # Build lookup from feature_idx to steering result for each model
    base_map = {fr.feature_idx: fr for fr in base_steer.per_feature}
    ft_a_map = {fr.feature_idx: fr for fr in ft_a_steer.per_feature}
    ft_b_map = {fr.feature_idx: fr for fr in ft_b_steer.per_feature}

    feature_map = {f.feature_idx: f for f in result.all_features}
    top_a_set = {f.feature_idx for f in result.top_features_a[:base_steer.config.top_k_steer]}
    coh_thresh = base_steer.config.coherence_threshold

    # --- Per-feature comparison plots (3 rows x 4 cols) ---
    for feat_idx in base_map:
        if feat_idx not in ft_a_map or feat_idx not in ft_b_map:
            continue

        fr_base = base_map[feat_idx]
        fr_ft_a = ft_a_map[feat_idx]
        fr_ft_b = ft_b_map[feat_idx]

        fig, axes = plt.subplots(3, 4, figsize=(20, 12), sharex=True)
        row_data = [
            ("Base", fr_base, "#333333"),
            ("FT-A", fr_ft_a, "#4C72B0"),
            ("FT-B", fr_ft_b, "#C44E52"),
        ]

        for row, (model_label, fr, color) in enumerate(row_data):
            scales = fr.scales
            p_a = fr.p_a_tagged
            p_b = fr.p_b_tagged
            coh = fr.coherence
            kl = fr.kl_loss if fr.kl_loss else [0.0] * len(scales)

            # Col 0: P(A-tagged) and P(B-tagged)
            ax = axes[row, 0]
            ax.plot(scales, p_a, "o-", color="#4C72B0", linewidth=2, markersize=4, label="P(A)")
            ax.plot(scales, p_b, "s-", color="#DD8452", linewidth=2, markersize=4, label="P(B)")
            ax.axhline(fr.p_a_baseline, color="#4C72B0", linestyle="--", alpha=0.4)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
            ax.set_ylabel(f"{model_label}\nProbability")
            if row == 0:
                ax.set_title("P(A-tagged) / P(B-tagged)")
                ax.legend(fontsize=7)
            if row == 2:
                ax.set_xlabel("Steering scale")
            ax.grid(True, alpha=0.2)

            # Col 1: exp(-KL) coherence
            ax = axes[row, 1]
            ax.plot(scales, coh, "D-", color="#8172B2", linewidth=2, markersize=4)
            ax.axhline(coh_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
            ax.set_ylim(0, 1.05)
            if row == 0:
                ax.set_title("Coherence: exp(-KL)")
            if row == 2:
                ax.set_xlabel("Steering scale")
            ax.grid(True, alpha=0.2)

            # Col 2: raw KL
            ax = axes[row, 2]
            kl_thresh = -np.log(max(coh_thresh, 1e-10))
            ax.plot(scales, kl, "D-", color="#C44E52", linewidth=2, markersize=4)
            ax.axhline(kl_thresh, color="red", linestyle="--", alpha=0.6, linewidth=1)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
            if row == 0:
                ax.set_title("Within-sector KL loss")
            if row == 2:
                ax.set_xlabel("Steering scale")
            ax.grid(True, alpha=0.2)

            # Col 3: P(A) - P(B)
            ax = axes[row, 3]
            diff = [a - b for a, b in zip(p_a, p_b)]
            ax.plot(scales, diff, "D-", color="#55A868", linewidth=2, markersize=4)
            ax.axhline(0, color="gray", linestyle=":", alpha=0.3)
            ax.axvline(0, color="gray", linestyle=":", alpha=0.3)
            if row == 0:
                ax.set_title("P(A) - P(B)")
            if row == 2:
                ax.set_xlabel("Steering scale")
            ax.grid(True, alpha=0.2)

        source = "top-A" if feat_idx in top_a_set else "top-B"
        fstats = feature_map.get(feat_idx)
        corr_str = f", corr={fstats.corr_pi_a:.3f}" if fstats and fstats.corr_pi_a is not None else ""
        fig.suptitle(
            f"Feature {feat_idx} ({source}{corr_str}) — Base vs FT-A vs FT-B steering",
            fontsize=13, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        path = output_dir / f"steering_comparison_feature_{feat_idx}.svg"
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    # --- Summary overlay: P(A) curves for base/FT-A/FT-B on same axes ---
    n_feat = len(base_map)
    if n_feat == 0:
        return paths

    n_cols = min(n_feat, 4)
    n_rows = (n_feat + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), squeeze=False)

    sorted_feats = sorted(base_map.keys())
    for i, feat_idx in enumerate(sorted_feats):
        row_i = i // n_cols
        col_i = i % n_cols
        ax = axes[row_i, col_i]

        fr_base = base_map[feat_idx]
        scales = fr_base.scales

        ax.plot(scales, fr_base.p_a_tagged, "o-", color="#333333", linewidth=2,
                markersize=4, label="Base")
        if fr_base.gen_pi_a is not None:
            ax.plot(scales, fr_base.gen_pi_a, "o--", color=_lighten("#333333", 0.4),
                    linewidth=1.5, markersize=3, label="Base gen")

        if feat_idx in ft_a_map:
            ax.plot(scales, ft_a_map[feat_idx].p_a_tagged, "s--", color="#4C72B0",
                    linewidth=2, markersize=4, label="FT-A")
            if ft_a_map[feat_idx].gen_pi_a is not None:
                ax.plot(scales, ft_a_map[feat_idx].gen_pi_a, "s:", color=_lighten("#4C72B0", 0.3),
                        linewidth=1.5, markersize=3, label="FT-A gen")
        if feat_idx in ft_b_map:
            ax.plot(scales, ft_b_map[feat_idx].p_a_tagged, "^:", color="#C44E52",
                    linewidth=2, markersize=4, label="FT-B")
            if ft_b_map[feat_idx].gen_pi_a is not None:
                ax.plot(scales, ft_b_map[feat_idx].gen_pi_a, "^-.", color=_lighten("#C44E52", 0.3),
                        linewidth=1.5, markersize=3, label="FT-B gen")

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3)
        ax.axvline(0, color="gray", linestyle=":", alpha=0.3)

        source = "A" if feat_idx in top_a_set else "B"
        ax.set_title(f"F{feat_idx} ({source})", fontsize=10)
        ax.set_ylabel("P(A-tagged)")
        ax.set_xlabel("Scale")
        ax.grid(True, alpha=0.2)

        if i == 0:
            ax.legend(fontsize=8)

    # Hide unused axes
    for i in range(n_feat, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].set_visible(False)

    fig.suptitle("Steering P(A) overlay: Base vs FT-A vs FT-B",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    path = output_dir / "steering_comparison_summary.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    paths.append(path)

    return paths


# =============================================================================
# STAGE 6: DECOMPOSITION COMPARISON PLOTS
# =============================================================================

def plot_decomposition(result: DecompositionResult, output_dir: Path) -> list[Path]:
    """Generate all Stage 6 decomposition comparison plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    paths.append(_plot_scree(result, output_dir))
    paths.append(_plot_r2_comparison(result, output_dir))
    paths.append(_plot_projection_scatter(result, output_dir))

    print(f"  Saved {len(paths)} decomposition plots to {output_dir}")
    return paths


def _plot_scree(result: DecompositionResult, output_dir: Path) -> Path:
    """PCA scree plot: variance explained per component + cumulative line."""
    ratios = np.array(result.scree_variance_ratio)
    k = len(ratios)
    idx = np.arange(1, k + 1)
    cumulative = np.cumsum(ratios)

    fig, ax1 = plt.subplots(figsize=(8, 4))
    ax1.bar(idx, ratios, color="steelblue", alpha=0.7, label="Per-component")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained Ratio")
    ax1.set_xticks(idx)

    ax2 = ax1.twinx()
    ax2.plot(idx, cumulative, "o-", color="firebrick", markersize=4, label="Cumulative")
    ax2.set_ylabel("Cumulative Variance")
    ax2.set_ylim(0, 1.05)

    # Mark the n_components used for comparison
    nc = result.n_components
    if nc <= k:
        ax1.axvline(nc + 0.5, color="gray", linestyle="--", alpha=0.5)
        ax1.text(nc + 0.7, ax1.get_ylim()[1] * 0.9,
                 f"k={nc}", fontsize=9, color="gray")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right")
    ax1.set_title("PCA Scree Plot")

    fig.tight_layout()
    path = output_dir / "scree_plot.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_r2_comparison(result: DecompositionResult, output_dir: Path) -> Path:
    """Grouped bar chart comparing R-squared across PCA, CCA, ICA."""
    methods = ["PCA", "CCA", "ICA"]
    metrics = [result.pca, result.cca, result.ica]

    r2_joint = [m.r2_joint for m in metrics]
    r2_sector = [m.r2_sector_mass for m in metrics]

    x = np.arange(len(methods))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4))
    bars1 = ax.bar(x - width / 2, r2_joint, width, label="Joint belief R²",
                   color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width / 2, r2_sector, width, label="Sector mass R²",
                   color="coral", alpha=0.8)

    ax.set_ylabel("R²")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.set_title(f"Linear Probe R² from {result.n_components} Components")

    # Value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2, h + 0.02,
                    f"{h:.2f}", ha="center", va="bottom", fontsize=9)

    fig.tight_layout()
    path = output_dir / "r2_comparison.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_projection_scatter(result: DecompositionResult, output_dir: Path) -> Path:
    """3-panel scatter plot: top-2 components per method, colored by pi_A."""
    projs = [
        ("PCA", result.pca_proj_2d),
        ("CCA", result.cca_proj_2d),
        ("ICA", result.ica_proj_2d),
    ]
    pi_a = np.array(result.pi_a_values)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    for ax, (name, proj) in zip(axes, projs):
        if proj is None:
            ax.set_title(f"{name} (no data)")
            continue
        proj = np.array(proj)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=pi_a, cmap="coolwarm",
                        s=8, alpha=0.6, vmin=0, vmax=1)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")
        ax.set_title(name)

    fig.colorbar(sc, ax=axes, label="$\\pi_A$ (sector mass)", shrink=0.8)
    fig.suptitle("2D Projections Colored by Sector Mass", fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 0.92, 0.95])

    path = output_dir / "projection_scatter.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path
