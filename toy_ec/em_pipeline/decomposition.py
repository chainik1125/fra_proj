"""Stage 6: Compare PCA, CCA, and ICA decompositions of activations vs belief states.

Runs after pretrain (stage 3). For each method, projects activations to k components,
then measures how well those components linearly predict belief states (R-squared).
"""

import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.cross_decomposition import CCA
from sklearn.decomposition import PCA, FastICA

from em_pipeline.config import (
    DecompositionConfig,
    DecompositionResult,
    ComponentMetrics,
    PipelineConfig,
    ProcessResult,
    PretrainResult,
    save_pickle,
    load_pickle,
    load_json,
    to_np_idx,
)
from em_pipeline.pretrain import collect_probe_data, linear_regression_r2


def run(
    process: ProcessResult,
    pretrain: PretrainResult,
    cfg: DecompositionConfig,
    pipeline_cfg: PipelineConfig,
) -> DecompositionResult:
    """Run PCA, CCA, and ICA on activations, compare belief-prediction R-squared."""

    device = pipeline_cfg.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load pretrained model
    model = _load_base_model(pretrain, process.info["total_vocab"], device)

    # Collect activations and beliefs
    print(f"  Collecting {cfg.n_samples} samples...")
    activations, beliefs_at_last, info_dict = collect_probe_data(
        model, process, cfg.n_samples, seed=pipeline_cfg.seed + 600, device=device,
    )
    sector_a_idx = info_dict["sector_a_idx"]
    pi_a = beliefs_at_last[:, sector_a_idx].sum(axis=1)  # (N,)

    n_components = min(cfg.n_components, beliefs_at_last.shape[1], activations.shape[1])
    print(f"  Using {n_components} components (d_model={activations.shape[1]}, "
          f"num_states={beliefs_at_last.shape[1]})")

    # --- PCA ---
    print("  Running PCA...")
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(activations)
    pca_metrics = _compute_metrics(
        "pca", X_pca, beliefs_at_last, pi_a,
        variance_explained=pca.explained_variance_ratio_.tolist(),
        cumulative_variance=np.cumsum(pca.explained_variance_ratio_).tolist(),
    )

    # Extended scree
    scree_k = min(cfg.max_scree_components, activations.shape[1])
    pca_scree = PCA(n_components=scree_k)
    pca_scree.fit(activations)
    scree_variance_ratio = pca_scree.explained_variance_ratio_.tolist()

    # --- CCA ---
    print("  Running CCA...")
    cca_k = min(n_components, beliefs_at_last.shape[1])
    cca = CCA(n_components=cca_k, max_iter=1000)
    cca.fit(activations, beliefs_at_last)
    X_cca, Y_cca = cca.transform(activations, beliefs_at_last)

    # Canonical correlations: per-column correlation between X_cca and Y_cca
    canon_corrs = []
    for i in range(cca_k):
        r = np.corrcoef(X_cca[:, i], Y_cca[:, i])[0, 1]
        canon_corrs.append(float(abs(r)))

    cca_metrics = _compute_metrics(
        "cca", X_cca, beliefs_at_last, pi_a,
        canonical_correlations=canon_corrs,
    )

    # --- ICA ---
    print("  Running ICA...")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ica = FastICA(n_components=n_components, max_iter=500, tol=1e-4, random_state=pipeline_cfg.seed)
        X_ica = ica.fit_transform(activations)

    ica_metrics = _compute_metrics("ica", X_ica, beliefs_at_last, pi_a)

    print(f"  R² (joint belief):  PCA={pca_metrics.r2_joint:.3f}  "
          f"CCA={cca_metrics.r2_joint:.3f}  ICA={ica_metrics.r2_joint:.3f}")
    print(f"  R² (sector mass):   PCA={pca_metrics.r2_sector_mass:.3f}  "
          f"CCA={cca_metrics.r2_sector_mass:.3f}  ICA={ica_metrics.r2_sector_mass:.3f}")

    return DecompositionResult(
        pca=pca_metrics,
        cca=cca_metrics,
        ica=ica_metrics,
        n_samples=activations.shape[0],
        n_components=n_components,
        scree_variance_ratio=scree_variance_ratio,
        pca_proj_2d=X_pca[:, :2],
        cca_proj_2d=X_cca[:, :2],
        ica_proj_2d=X_ica[:, :2],
        pi_a_values=pi_a,
    )


def save(result: DecompositionResult, path: Path) -> None:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    save_pickle(result, path / "decomposition_result.pkl")


def load(path: Path) -> DecompositionResult:
    return load_pickle(Path(path) / "decomposition_result.pkl")


# =============================================================================
# HELPERS
# =============================================================================

def _compute_metrics(
    method: str,
    X_proj: np.ndarray,
    beliefs: np.ndarray,
    pi_a: np.ndarray,
    variance_explained: list[float] | None = None,
    cumulative_variance: list[float] | None = None,
    canonical_correlations: list[float] | None = None,
) -> ComponentMetrics:
    """Compute per-component correlations and linear probe R-squared."""
    k = X_proj.shape[1]

    # Per-component |correlation| with pi_A
    corr_pi_a = []
    for i in range(k):
        r = np.corrcoef(X_proj[:, i], pi_a)[0, 1]
        corr_pi_a.append(float(abs(r)) if np.isfinite(r) else 0.0)

    # Linear probe R²: projected components -> belief targets
    r2_joint = linear_regression_r2(X_proj, beliefs)
    r2_sector_mass = linear_regression_r2(X_proj, pi_a[:, None])

    return ComponentMetrics(
        method=method,
        n_components=k,
        corr_pi_a=corr_pi_a,
        r2_joint=r2_joint,
        r2_sector_mass=r2_sector_mass,
        variance_explained=variance_explained,
        cumulative_variance=cumulative_variance,
        canonical_correlations=canonical_correlations,
    )


def _load_base_model(pretrain: PretrainResult, total_vocab: int, device: str):
    """Load the pretrained base model."""
    from training.run_minimal import Config as TrainConfig, build_model

    stage_dir = Path(pretrain.run_dir)
    model_config = load_json(stage_dir / "model_config.json")
    tc = model_config["train_config"]

    cfg = TrainConfig(
        d_model=tc["d_model"],
        d_head=tc["d_head"],
        n_heads=tc["n_heads"],
        n_layers=tc["n_layers"],
        d_mlp=tc["d_mlp"],
        n_ctx=tc["n_ctx"],
        device=device,
    )
    model = build_model(cfg, total_vocab)
    model.load_state_dict(torch.load(stage_dir / "model.pt", weights_only=True, map_location=device))
    return model
