"""AFP Steering Pipeline.

A modular 4-stage pipeline for AFP (Almost-Factored Process) steering experiments.

Stages:
    1. Define: Build the AFP generative process (prompt + completion HMMs)
    2. Analyze: Compute process properties (diversity, polarization, entropy)
    3. Pretrain: Train transformer, probe belief representations at checkpoints
    4. Finetune: Finetune on polarized completions, measure bias generalization
"""

import os
import sys
from pathlib import Path

# Force JAX to CPU (must be before any JAX import)
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Add project paths so we can import from training/, shared_tools/, and analysis/
_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
_TRAINING_DIR = _REPO_ROOT / "training"
_ANALYSIS_DIR = _REPO_ROOT / "analysis"

for _p in [str(_REPO_ROOT), str(_TRAINING_DIR), str(_ANALYSIS_DIR)]:
    if _p not in sys.path:
        sys.path.insert(0, _p)
