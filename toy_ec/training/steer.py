"""
Minimal script for loading and analyzing trained runs.

Usage:
    python steer.py <run_name>
    python steer.py test_saver
    python steer.py --list  # List available runs (fast - no heavy imports)
"""

import argparse
import sys
from pathlib import Path

# Add parent to path for shared_tools
sys.path.insert(0, str(Path(__file__).parent.parent))
from shared_tools import LocalRunLoader


def build_model(cfg: dict, vocab_size: int):
    """Rebuild model architecture from config. Imports transformer_lens lazily."""
    from transformer_lens import HookedTransformer, HookedTransformerConfig

    model_cfg = HookedTransformerConfig(
        d_model=cfg["d_model"],
        d_head=cfg["d_head"],
        n_heads=cfg["n_heads"],
        n_layers=cfg["n_layers"],
        n_ctx=cfg["n_ctx"],
        d_mlp=cfg["d_mlp"],
        d_vocab=vocab_size,
        act_fn="gelu",
        normalization_type="LN",
        device=cfg.get("device", "cpu"),
        seed=cfg.get("seed", 42),
    )
    return HookedTransformer(model_cfg)


def load_run(run_id: str):
    """
    Load a training run.

    Args:
        run_id: Name of the run folder

    Returns:
        (model, hmm, config, results)
    """
    loader = LocalRunLoader()
    return loader.load(run_id, model_builder=build_model)


def main():
    parser = argparse.ArgumentParser(description="Load and analyze training runs")
    parser.add_argument("run_id", nargs="?", help="Run name to load")
    parser.add_argument("--list", action="store_true", help="List available runs")
    args = parser.parse_args()

    loader = LocalRunLoader()

    if args.list:
        runs = loader.list_runs()
        if runs:
            print("Available runs:")
            for run in runs:
                print(f"  {run}")
        else:
            print("No runs found.")
        return

    if not args.run_id:
        parser.print_help()
        return

    # Load the run (this triggers heavy imports)
    print(f"Loading run: {args.run_id}")
    model, hmm, cfg, results = load_run(args.run_id)

    print(f"Model: {cfg['n_layers']}L, {cfg['d_model']}d, {cfg['n_heads']}h")
    print(f"HMM: vocab_size={hmm.vocab_size}, states={hmm.num_states}")
    print(f"Training: {cfg['num_steps']} steps")
    print(f"Final train loss: {results['history'][-1]['train_loss']:.4f}")

    # Model and HMM are now ready for analysis
    # Add your steering/analysis code below
    # -----------------------------------------




if __name__ == "__main__":
    main()
