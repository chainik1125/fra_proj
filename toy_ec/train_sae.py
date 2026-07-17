"""Standalone SAE training CLI.

Train an SAE on a previous pipeline run's base-model activations without
running the full diffing pipeline. Useful for iterating on SAE hyperparameters.

Usage:
    uv run python analysis/em_pipeline/train_sae.py \
        --run-dir analysis/em_pipeline/outputs/run_20260303_0652 \
        --activation batch_topk --topk-k 20 --num-steps 5000
"""

import argparse
import os
import sys
from pathlib import Path

# Ensure repo root and analysis/ are on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
for _p in [str(_REPO_ROOT), str(_REPO_ROOT / "training"), str(_REPO_ROOT / "analysis")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

from em_pipeline.config import SAEConfig
from em_pipeline.diffing import train_sae_standalone


def main():
    parser = argparse.ArgumentParser(description="Train SAE standalone on a previous run")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Path to pipeline run directory")
    parser.add_argument("--activation", type=str, default=None,
                        help="SAE activation: batch_topk, topk, relu, jumprelu")
    parser.add_argument("--topk-k", type=int, default=None,
                        help="k for topk/batch_topk activations")
    parser.add_argument("--dict-size-multiplier", type=int, default=None,
                        help="SAE dict size = multiplier * d_model")
    parser.add_argument("--l1-coefficient", type=float, default=None,
                        help="L1 sparsity penalty (relu only)")
    parser.add_argument("--aux-coefficient", type=float, default=None,
                        help="Dead-feature aux loss weight (topk/batch_topk)")
    parser.add_argument("--dead-window", type=int, default=None,
                        help="Steps of inactivity before a feature is 'dead' (default: 200)")
    parser.add_argument("--jumprelu-bandwidth", type=float, default=None,
                        help="STE bandwidth for JumpReLU threshold learning (default: 0.001)")
    parser.add_argument("--jumprelu-init-threshold", type=float, default=None,
                        help="Initial threshold for JumpReLU features (default: 0.01)")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="SAE learning rate")
    parser.add_argument("--num-steps", type=int, default=None,
                        help="Number of SAE training steps")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Batch size for SAE training")
    parser.add_argument("--n-train-samples", type=int, default=None,
                        help="Number of sequences to generate for training data")
    parser.add_argument("--device", type=str, default="auto",
                        help="Torch device (auto, cpu, cuda)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--save-dir", type=str, default=None,
                        help="Directory to save SAE weights and activations")

    args = parser.parse_args()

    # Build SAEConfig, overriding only provided arguments
    sae_cfg = SAEConfig()
    if args.activation is not None:
        sae_cfg.activation = args.activation
    if args.topk_k is not None:
        sae_cfg.topk_k = args.topk_k
    if args.dict_size_multiplier is not None:
        sae_cfg.dict_size_multiplier = args.dict_size_multiplier
    if args.l1_coefficient is not None:
        sae_cfg.l1_coefficient = args.l1_coefficient
    if args.aux_coefficient is not None:
        sae_cfg.aux_coefficient = args.aux_coefficient
    if args.dead_window is not None:
        sae_cfg.dead_window = args.dead_window
    if args.jumprelu_bandwidth is not None:
        sae_cfg.jumprelu_bandwidth = args.jumprelu_bandwidth
    if args.jumprelu_init_threshold is not None:
        sae_cfg.jumprelu_init_threshold = args.jumprelu_init_threshold
    if args.learning_rate is not None:
        sae_cfg.learning_rate = args.learning_rate
    if args.num_steps is not None:
        sae_cfg.num_steps = args.num_steps
    if args.batch_size is not None:
        sae_cfg.batch_size = args.batch_size
    if args.n_train_samples is not None:
        sae_cfg.n_train_samples = args.n_train_samples

    print(f"SAE config: activation={sae_cfg.activation}, k={sae_cfg.topk_k}, "
          f"dict_mult={sae_cfg.dict_size_multiplier}, steps={sae_cfg.num_steps}, "
          f"n_train={sae_cfg.n_train_samples}")

    sae, stats, activations = train_sae_standalone(
        run_dir=args.run_dir,
        sae_cfg=sae_cfg,
        device=args.device,
        seed=args.seed,
        save_dir=args.save_dir,
    )

    print(f"\nDone. L0={stats.mean_l0:.1f}, dead={stats.dead_features}/{sae.dict_size}")


if __name__ == "__main__":
    main()
