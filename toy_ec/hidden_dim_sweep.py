"""Hidden-dimension co-scaling sweep.

Tests whether the hidden state space is too small relative to longer sequences
by co-scaling d_g, d_b, and content_symbols alongside comp_len.

Unlike comp_len_sweep (which holds d_g=d_b=content_symbols=5 fixed), this sweep
sets d_g = d_b = content_symbols = comp_len at each point.

Usage:
    uv run python analysis/em_pipeline/hidden_dim_sweep.py
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

COMP_LENS = [8, 14, 20]
BASE_CONFIG = Path(__file__).parent / "leaky_reset_config.yaml"
OUTPUTS_DIR = Path(__file__).parent / "outputs"
SWEEP_DIR = OUTPUTS_DIR / "hidden_dim_sweep"

EXISTING_RUNS = {}

# ft_steps scales linearly: cl=5 → 200, cl=20 → 2000
FT_STEPS_MIN, FT_STEPS_MAX = 200, 2000
# Fixed pretrain steps for speed; we warn if loss hasn't stabilized
PRETRAIN_STEPS = 5000
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


def _linear_scale(cl: int, lo: int, hi: int) -> int:
    """Linearly interpolate between lo (cl=5) and hi (cl=20)."""
    frac = (cl - CL_MIN) / (CL_MAX - CL_MIN)
    return int(lo + frac * (hi - lo))


def scale_ft_steps(cl: int) -> int:
    return _linear_scale(cl, FT_STEPS_MIN, FT_STEPS_MAX)


def check_loss_stabilized(run_dir: Path, cl: int) -> None:
    """Warn if pretrain loss hasn't stabilized (last 20% still declining significantly)."""
    pretrain_path = run_dir / "stage3" / "pretrain_result.pkl"
    if not pretrain_path.exists():
        return
    pretrain_result = load_pickle(pretrain_path)
    history = pretrain_result.history
    if not history or len(history) < 20:
        return
    losses = [h["loss"] for h in history if "loss" in h]
    if len(losses) < 20:
        return
    # Compare mean of last 20% vs previous 20%
    n = len(losses)
    last_20 = np.mean(losses[int(n * 0.8):])
    prev_20 = np.mean(losses[int(n * 0.6):int(n * 0.8)])
    rel_drop = (prev_20 - last_20) / prev_20 if prev_20 > 0 else 0
    if rel_drop > 0.02:
        print(f"  WARNING [cl={cl}]: Loss still declining (last 20% mean={last_20:.4f}, "
              f"prev 20% mean={prev_20:.4f}, rel drop={rel_drop:.1%}). "
              f"Consider increasing pretrain steps.")


def run_name_for_cl(cl: int) -> str:
    return EXISTING_RUNS.get(cl, f"leaky_reset_hd_cl{cl}")


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
        print(f"Running pipeline for comp_len={cl} (d_g=d_b=content_symbols={cl})")
        print(f"{'='*60}")

        cfg = PipelineConfig.from_yaml(str(BASE_CONFIG))
        cfg.sequence.comp_len = cl
        cfg.process.d_g = cl
        cfg.process.d_b = cl
        cfg.process.content_symbols = cl
        cfg.run_name = run_name_for_cl(cl)
        cfg.finetune.ft_steps = scale_ft_steps(cl)

        cfg.pretrain.num_steps = PRETRAIN_STEPS
        cfg.pretrain.checkpoint_steps = [1000, 2500, 5000]

        run_pipeline(cfg)

        # Check if loss stabilized with fixed step budget
        run_dir = OUTPUTS_DIR / run_name_for_cl(cl)
        check_loss_stabilized(run_dir, cl)


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
    ax.set_xlabel("Completion length (= hidden dim)")
    ax.set_ylabel(r"$\sigma(\pi_A)$")
    ax.set_title(r"$\pi_A$ standard deviation")
    ax.set_xticks(cls)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(cls, frac_polarized, "s-", color="#C44E52", linewidth=2, markersize=6)
    ax.set_xlabel("Completion length (= hidden dim)")
    ax.set_ylabel(r"Fraction $\pi_A > 0.9$ or $< 0.1$")
    ax.set_title("Fraction fully polarized")
    ax.set_xticks(cls)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Polarization vs Completion Length (co-scaled hidden dim)", fontsize=14, y=1.02)
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


def plot_feature_quality(results: dict[int, DiffingResult]):
    """3-panel: AUROC, steering swing, |corr(z, π_A)| vs comp_len."""
    cls = sorted(results.keys())

    # Per comp_len: collect metrics for top-5 features
    all_aurocs = {cl: [] for cl in cls}
    all_swings = {cl: [] for cl in cls}
    all_corrs = {cl: [] for cl in cls}
    best_auroc = []
    best_swing = []
    best_corr = []

    for cl in cls:
        result = results[cl]
        top_feats = _get_top_features_by_swing(result, k=5)
        if not top_feats:
            best_auroc.append(np.nan)
            best_swing.append(np.nan)
            best_corr.append(np.nan)
            continue

        aurocs = []
        swings = []
        corrs = []
        for feat_idx, swing in top_feats:
            aurocs.append(_feature_auroc(result, feat_idx))
            swings.append(swing)
            c = _feature_corr_pi_a(result, feat_idx)
            corrs.append(c if c is not None else np.nan)

        all_aurocs[cl] = aurocs
        all_swings[cl] = swings
        all_corrs[cl] = corrs
        best_auroc.append(max(aurocs))
        best_swing.append(max(swings))
        best_corr.append(np.nanmax(corrs) if corrs else np.nan)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    # Panel 1: AUROC
    ax = axes[0]
    for cl in cls:
        for v in all_aurocs[cl]:
            ax.scatter(cl, v, color="#4C72B0", alpha=0.4, s=30, zorder=2)
    ax.plot(cls, best_auroc, "o-", color="#4C72B0", linewidth=2.5, markersize=7,
            zorder=3, label="Best feature")
    ax.set_xlabel("Completion length (= hidden dim)")
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
    ax.set_xlabel("Completion length (= hidden dim)")
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
    ax.set_xlabel("Completion length (= hidden dim)")
    ax.set_ylabel(r"$|\mathrm{corr}(z, \pi_A)|$")
    ax.set_title(r"Feature–belief correlation")
    ax.set_xticks(cls)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2)

    fig.suptitle("Top-5 Feature Metrics vs Completion Length (co-scaled hidden dim)", fontsize=14, y=1.02)
    fig.tight_layout()
    path = SWEEP_DIR / "feature_quality_vs_comp_len.svg"
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {path}")
    return path


# ── Plot C: FT-A Bias and Reversal vs Completion Length ────────

def plot_ft_bias(results: dict[int, DiffingResult]):
    """2-panel: FT-A baseline P(A) and best steering reversal vs comp_len."""
    cls = sorted(results.keys())
    baselines = []
    reversals = []

    for cl in cls:
        result = results[cl]
        ft_a_steer = result.steering_result_ft_a
        if ft_a_steer is None or not ft_a_steer.per_feature:
            baselines.append(np.nan)
            reversals.append(np.nan)
            continue

        # FT-A baseline P(A): use the feature with highest swing
        top_feats = _get_top_features_by_swing(result, k=5)
        feat_indices = {idx for idx, _ in top_feats}

        ft_a_baselines = []
        ft_a_reversals = []
        for fr in ft_a_steer.per_feature:
            if fr.feature_idx not in feat_indices:
                continue
            ft_a_baselines.append(fr.p_a_baseline)
            # Best reversal: how much we can push P(A) down from the FT-A bias
            min_p_a = min(fr.p_a_tagged)
            ft_a_reversals.append(fr.p_a_baseline - min_p_a)

        baselines.append(np.mean(ft_a_baselines) if ft_a_baselines else np.nan)
        reversals.append(max(ft_a_reversals) if ft_a_reversals else np.nan)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    ax = axes[0]
    ax.plot(cls, baselines, "o-", color="#4C72B0", linewidth=2, markersize=6)
    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Completion length (= hidden dim)")
    ax.set_ylabel("FT-A baseline P(A)")
    ax.set_title("FT-A sector bias")
    ax.set_xticks(cls)
    ax.grid(True, alpha=0.2)

    ax = axes[1]
    ax.plot(cls, reversals, "s-", color="#C44E52", linewidth=2, markersize=6)
    ax.set_xlabel("Completion length (= hidden dim)")
    ax.set_ylabel("P(A) reversal (baseline - min)")
    ax.set_title("Best FT-A steering reversal")
    ax.set_xticks(cls)
    ax.grid(True, alpha=0.2)

    fig.suptitle("FT-A Bias and Reversal vs Completion Length (co-scaled hidden dim)", fontsize=14, y=1.02)
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
