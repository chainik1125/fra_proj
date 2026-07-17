"""Completion-length / polarization sweep.

Runs the full AFP steering pipeline for multiple completion lengths,
then generates summary plots showing how polarization, partition quality,
and steering effectiveness evolve with completion length.

Usage:
    uv run python analysis/em_pipeline/comp_len_sweep.py
"""

import os
import sys
from pathlib import Path

# Ensure repo root and analysis/ are on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "training"), str(_REPO_ROOT / "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from em_pipeline.config import (
    DiffingResult,
    PipelineConfig,
    load_pickle,
)
from em_pipeline.main import run_pipeline
from em_pipeline.plots import _compute_auroc

# ── Configuration ──────────────────────────────────────────────

COMP_LENS = [5, 8, 11, 14, 17, 20]
BASE_CONFIG = Path(__file__).parent / "leaky_reset_config.yaml"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
SWEEP_DIR = OUTPUTS_DIR / "comp_len_sweep"

# Map comp_lens to existing run names (already completed)
EXISTING_RUNS = {5: "run_20260303_0652", 20: "leaky_reset_cl20"}

# ft_steps scales linearly: cl=5 → 200, cl=20 → 2000
FT_STEPS_MIN, FT_STEPS_MAX = 200, 2000
CL_MIN, CL_MAX = 5, 20

# Plot style
plt.rcParams.update({
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 150,
    "svg.fonttype": "none",
})


def scale_ft_steps(cl: int) -> int:
    """Linearly interpolate ft_steps based on completion length."""
    frac = (cl - CL_MIN) / (CL_MAX - CL_MIN)
    return int(FT_STEPS_MIN + frac * (FT_STEPS_MAX - FT_STEPS_MIN))


def run_name_for_cl(cl: int) -> str:
    return EXISTING_RUNS.get(cl, f"leaky_reset_cl{cl}")


def result_path_for_cl(cl: int) -> Path:
    return OUTPUTS_DIR / run_name_for_cl(cl) / "stage5" / "diffing_result.pkl"


# ── Pipeline runner ────────────────────────────────────────────

def run_missing_pipelines():
    """Run the full pipeline for each missing comp_len."""
    for cl in COMP_LENS:
        rpath = result_path_for_cl(cl)
        if rpath.exists():
            print(f"[cl={cl}] Already complete at {rpath}, skipping.")
            continue

        print(f"\n{'='*60}")
        print(f"Running pipeline for comp_len={cl}")
        print(f"{'='*60}")

        cfg = PipelineConfig.from_yaml(str(BASE_CONFIG))
        cfg.sequence.comp_len = cl
        cfg.run_name = run_name_for_cl(cl)
        cfg.finetune.ft_steps = scale_ft_steps(cl)

        run_pipeline(cfg)


# ── Data loading ───────────────────────────────────────────────

def load_all_results() -> dict[int, DiffingResult]:
    """Load DiffingResults for all completed comp_lens."""
    results = {}
    for cl in COMP_LENS:
        rpath = result_path_for_cl(cl)
        if rpath.exists():
            results[cl] = load_pickle(rpath)
            print(f"[cl={cl}] Loaded from {rpath}")
        else:
            print(f"[cl={cl}] Missing — skipping")
    return results


def compute_pi_a(result: DiffingResult) -> np.ndarray:
    """Compute π_A (sector A belief mass) per sample."""
    beliefs = np.asarray(result.eval_beliefs)
    sector_a_idx = np.asarray(result.eval_sector_a_idx)
    return beliefs[:, sector_a_idx].sum(axis=1)


# ── Plot A: Polarization vs Completion Length ──────────────────

def plot_polarization(results: dict[int, DiffingResult]):
    """2-panel: π_A std dev and fraction fully polarized vs comp_len."""
    cls = sorted(results.keys())
    stds = []
    frac_polarized = []

    for cl in cls:
        pi_a = compute_pi_a(results[cl])
        stds.append(np.std(pi_a))
        frac_polarized.append(np.mean((pi_a > 0.9) | (pi_a < 0.1)))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(cls, stds, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax.set_xlabel("Completion length")
    ax.set_ylabel(r"$\sigma(\pi_A)$")
    ax.set_title(r"$\pi_A$ standard deviation")
    ax.set_xticks(cls)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(cls, frac_polarized, "s-", color="#C44E52", linewidth=2, markersize=6)
    ax.set_xlabel("Completion length")
    ax.set_ylabel(r"Fraction $\pi_A > 0.9$ or $< 0.1$")
    ax.set_title("Fraction fully polarized")
    ax.set_xticks(cls)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Polarization vs Completion Length", fontsize=14, y=1.02)
    fig.tight_layout()
    path = SWEEP_DIR / "polarization_vs_comp_len.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Plot B: Feature Quality Metrics vs Completion Length ───────

def _get_top_features_by_swing(result: DiffingResult, k: int = 5):
    """Select top-k features by base-model steering swing (max P(A) - min P(A)).

    Returns list of (feature_idx, swing) tuples.
    """
    if result.steering_result is None:
        return []
    features = []
    for fr in result.steering_result.per_feature:
        swing = max(fr.p_a_tagged) - min(fr.p_a_tagged)
        features.append((fr.feature_idx, swing))
    features.sort(key=lambda x: -x[1])
    return features[:k]


def _feature_auroc(result: DiffingResult, feat_idx: int) -> float:
    """Compute AUROC for a single feature partitioning A vs B."""
    pi_a = compute_pi_a(result)
    is_a = pi_a > 0.5
    z = np.asarray(result.eval_z_base)[:, feat_idx]
    return _compute_auroc(z, is_a)


def _feature_corr_pi_a(result: DiffingResult, feat_idx: int) -> float | None:
    """Get |corr(z, π_A)| for a feature from the stored stats."""
    for f in result.all_features:
        if f.feature_idx == feat_idx:
            if f.corr_pi_a is not None:
                return abs(f.corr_pi_a)
            return None
    return None


def _feature_gen_swing(result: DiffingResult, feat_idx: int) -> float:
    """Get generative steering swing for a feature: max(gen_pi_a) - min(gen_pi_a)."""
    if result.steering_result is None:
        return np.nan
    for fr in result.steering_result.per_feature:
        if fr.feature_idx == feat_idx and fr.gen_pi_a is not None:
            return max(fr.gen_pi_a) - min(fr.gen_pi_a)
    return np.nan


def _feature_consistent_gen_range(
    result: DiffingResult, feat_idx: int, commitment_threshold: float = 0.5,
) -> float:
    """Gen π_A range using only scales where mean_sector_commitment >= threshold."""
    if result.steering_result is None:
        return np.nan
    for fr in result.steering_result.per_feature:
        if fr.feature_idx != feat_idx:
            continue
        if fr.gen_pi_a is None or fr.gen_belief_consistency is None:
            return np.nan
        consistent_pi_a = [
            pi_a for pi_a, bc in zip(fr.gen_pi_a, fr.gen_belief_consistency)
            if bc["mean_sector_commitment"] >= commitment_threshold
        ]
        if len(consistent_pi_a) < 2:
            return 0.0
        return max(consistent_pi_a) - min(consistent_pi_a)
    return np.nan


def plot_feature_quality(results: dict[int, DiffingResult]):
    """5-panel: AUROC, steering swing, |corr(z, π_A)|, gen swing, consistent gen range vs comp_len."""
    cls = sorted(results.keys())

    # Per comp_len: collect metrics for top-5 features
    all_aurocs = {cl: [] for cl in cls}
    all_swings = {cl: [] for cl in cls}
    all_corrs = {cl: [] for cl in cls}
    all_gen_swings = {cl: [] for cl in cls}
    all_consistent_gen = {cl: [] for cl in cls}
    best_auroc = []
    best_swing = []
    best_corr = []
    best_gen_swing = []
    best_consistent_gen = []

    for cl in cls:
        result = results[cl]
        top_feats = _get_top_features_by_swing(result, k=5)
        if not top_feats:
            best_auroc.append(np.nan)
            best_swing.append(np.nan)
            best_corr.append(np.nan)
            best_gen_swing.append(np.nan)
            best_consistent_gen.append(np.nan)
            continue

        aurocs = []
        swings = []
        corrs = []
        gen_swings = []
        consistent_gens = []
        for feat_idx, swing in top_feats:
            aurocs.append(_feature_auroc(result, feat_idx))
            swings.append(swing)
            c = _feature_corr_pi_a(result, feat_idx)
            corrs.append(c if c is not None else np.nan)
            gen_swings.append(_feature_gen_swing(result, feat_idx))
            consistent_gens.append(_feature_consistent_gen_range(result, feat_idx))

        all_aurocs[cl] = aurocs
        all_swings[cl] = swings
        all_corrs[cl] = corrs
        all_gen_swings[cl] = gen_swings
        all_consistent_gen[cl] = consistent_gens
        best_auroc.append(max(aurocs))
        best_swing.append(max(swings))
        best_corr.append(np.nanmax(corrs) if corrs else np.nan)
        best_gen_swing.append(np.nanmax(gen_swings) if gen_swings else np.nan)
        best_consistent_gen.append(np.nanmax(consistent_gens) if consistent_gens else np.nan)

    has_gen = any(not np.isnan(v) for v in best_gen_swing)
    has_consistent = any(not np.isnan(v) for v in best_consistent_gen)
    n_panels = 3 + (1 if has_gen else 0) + (1 if has_consistent else 0)
    fig, axes = plt.subplots(1, n_panels, figsize=(5 * n_panels, 4.5))

    # Panel 1: AUROC
    ax = axes[0]
    for cl in cls:
        for v in all_aurocs[cl]:
            ax.scatter(cl, v, color="#4C72B0", alpha=0.4, s=30, zorder=2)
    ax.plot(cls, best_auroc, "o-", color="#4C72B0", linewidth=2.5, markersize=7,
            zorder=3, label="Best feature")
    ax.set_xlabel("Completion length")
    ax.set_ylabel("AUROC")
    ax.set_title("Partition quality (AUROC)")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Steering swing
    ax = axes[1]
    for cl in cls:
        for v in all_swings[cl]:
            ax.scatter(cl, v, color="#DD8452", alpha=0.4, s=30, zorder=2)
    ax.plot(cls, best_swing, "s-", color="#DD8452", linewidth=2.5, markersize=7,
            zorder=3, label="Best feature")
    ax.set_xlabel("Completion length")
    ax.set_ylabel("max P(A) - min P(A)")
    ax.set_title("Steering swing")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 3: |corr(z, π_A)|
    ax = axes[2]
    for cl in cls:
        for v in all_corrs[cl]:
            if not np.isnan(v):
                ax.scatter(cl, v, color="#8172B2", alpha=0.4, s=30, zorder=2)
    ax.plot(cls, best_corr, "D-", color="#8172B2", linewidth=2.5, markersize=7,
            zorder=3, label="Best feature")
    ax.set_xlabel("Completion length")
    ax.set_ylabel(r"$|\mathrm{corr}(z, \pi_A)|$")
    ax.set_title(r"Feature–belief correlation")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 4: Generative steering swing (if available)
    panel_idx = 3
    if has_gen:
        ax = axes[panel_idx]
        panel_idx += 1
        for cl in cls:
            for v in all_gen_swings[cl]:
                if not np.isnan(v):
                    ax.scatter(cl, v, color="#55A868", alpha=0.4, s=30, zorder=2)
        ax.plot(cls, best_gen_swing, "^-", color="#55A868", linewidth=2.5, markersize=7,
                zorder=3, label="Best feature")
        ax.set_xlabel("Completion length")
        ax.set_ylabel("max gen π_A - min gen π_A")
        ax.set_title("Generative steering swing")
        ax.set_xticks(cls)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    # Panel 5: Consistent generative range (commitment-gated)
    if has_consistent:
        ax = axes[panel_idx]
        panel_idx += 1
        for cl in cls:
            for v in all_consistent_gen[cl]:
                if not np.isnan(v):
                    ax.scatter(cl, v, color="#CCB974", alpha=0.4, s=30, zorder=2)
        ax.plot(cls, best_consistent_gen, "p-", color="#CCB974", linewidth=2.5, markersize=7,
                zorder=3, label="Best feature")
        ax.set_xlabel("Completion length")
        ax.set_ylabel("Consistent gen range")
        ax.set_title("Belief-consistent gen swing")
        ax.set_xticks(cls)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2)

    fig.suptitle("Top-5 Feature Metrics vs Completion Length", fontsize=14, y=1.02)
    fig.tight_layout()
    path = SWEEP_DIR / "feature_quality_vs_comp_len.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Plot C: FT-A Bias and Reversal vs Completion Length ────────

def plot_ft_bias(results: dict[int, DiffingResult]):
    """FT-A baseline P(A), steering reversal, and generative variants vs comp_len."""
    cls = sorted(results.keys())
    baselines = []
    reversals = []
    gen_baselines = []
    gen_reversals = []

    for cl in cls:
        result = results[cl]
        ft_a_steer = result.steering_result_ft_a
        if ft_a_steer is None or not ft_a_steer.per_feature:
            baselines.append(np.nan)
            reversals.append(np.nan)
            gen_baselines.append(np.nan)
            gen_reversals.append(np.nan)
            continue

        # FT-A baseline P(A): use the feature with highest swing
        top_feats = _get_top_features_by_swing(result, k=5)
        feat_indices = {idx for idx, _ in top_feats}

        ft_a_baselines = []
        ft_a_reversals = []
        ft_a_gen_baselines = []
        ft_a_gen_reversals = []
        for fr in ft_a_steer.per_feature:
            if fr.feature_idx not in feat_indices:
                continue
            ft_a_baselines.append(fr.p_a_baseline)
            min_p_a = min(fr.p_a_tagged)
            ft_a_reversals.append(fr.p_a_baseline - min_p_a)
            # Generative metrics
            if fr.gen_pi_a is not None:
                # gen baseline = gen_pi_a at scale=0
                scales = fr.scales
                zero_idx = scales.index(0.0) if 0.0 in scales else len(scales) // 2
                ft_a_gen_baselines.append(fr.gen_pi_a[zero_idx])
                ft_a_gen_reversals.append(fr.gen_pi_a[zero_idx] - min(fr.gen_pi_a))

        baselines.append(np.mean(ft_a_baselines) if ft_a_baselines else np.nan)
        reversals.append(max(ft_a_reversals) if ft_a_reversals else np.nan)
        gen_baselines.append(np.mean(ft_a_gen_baselines) if ft_a_gen_baselines else np.nan)
        gen_reversals.append(max(ft_a_gen_reversals) if ft_a_gen_reversals else np.nan)

    has_gen = any(not np.isnan(v) for v in gen_baselines)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Panel 1: FT-A sector bias
    ax = axes[0]
    ax.plot(cls, baselines, "o-", color="#4C72B0", linewidth=2, markersize=6, label="First-token P(A)")
    if has_gen:
        ax.plot(cls, gen_baselines, "o--", color="#55A868", linewidth=2, markersize=6, label="Gen π_A")
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Completion length")
    ax.set_ylabel("FT-A baseline P(A)")
    ax.set_title("FT-A sector bias")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    # Panel 2: Best steering reversal
    ax = axes[1]
    ax.plot(cls, reversals, "s-", color="#C44E52", linewidth=2, markersize=6, label="First-token")
    if has_gen:
        ax.plot(cls, gen_reversals, "s--", color="#55A868", linewidth=2, markersize=6, label="Generative")
    ax.set_xlabel("Completion length")
    ax.set_ylabel("P(A) reversal (baseline - min)")
    ax.set_title("Best FT-A steering reversal")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("FT-A Bias and Reversal vs Completion Length", fontsize=14, y=1.02)
    fig.tight_layout()
    path = SWEEP_DIR / "ft_bias_vs_comp_len.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Main ───────────────────────────────────────────────────────

def main():
    # Step 1: Run missing pipelines
    run_missing_pipelines()

    # Step 2: Load all results
    print(f"\n{'='*60}")
    print("Loading results for summary plots")
    print(f"{'='*60}")
    results = load_all_results()

    if len(results) < 2:
        print("Need at least 2 completed runs for summary plots. Exiting.")
        return

    # Step 3: Generate summary plots
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nGenerating summary plots in {SWEEP_DIR}")

    plot_polarization(results)
    plot_feature_quality(results)
    plot_ft_bias(results)

    print(f"\nDone. {len(results)} comp_lens plotted.")


if __name__ == "__main__":
    main()
