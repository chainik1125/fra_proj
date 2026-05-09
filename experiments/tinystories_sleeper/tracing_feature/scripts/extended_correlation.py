"""Extended Spearman correlation with the 18 new ablated features +
originals. Compares QK-side aggregations against measured |Δlogp|."""

import json
import sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

# Load QK concentration results + new ablation results
# Need raw per-μ arrays; re-run part of qk_concentration to get them.

# Shortcut: load qk_concentration.json to recover the top-20 μ's with their concentration stats.
# Actually we need per-μ scores for ALL μ's to compute ranks. Use qk_concentration.py's
# raw output to rebuild the same tensors.
with open(HERE.parent / "results" / "qk_concentration.json") as f:
    qk_out = json.load(f)

# Load extended ablation results (new)
with open(HERE.parent / "qk_vs_ov" / "results" / "ln1_feature_ablation.json") as f:
    abl_new = json.load(f)

# Load original ablation results
with open(HERE.parent / "results" / "ln1_feature_ablation.json") as f:
    abl_orig = json.load(f)

measured = {}
for src in [abl_orig, abl_new]:
    for r in src["results"]:
        if r["alpha"] == 4.0 and isinstance(r["ln1_feature"], int):
            measured[r["ln1_feature"]] = r["delta_logp"]

# To compute per-μ scores for ALL μ's we need to re-run the QK concentration.
# But we have the top-20 scores in qk_concentration.json. Use those + missing are NaN.
# Better: re-load per-μ scores from a recomputation. Let's run the concentration
# and dump the raw scores.

print("=== Measured features ===")
for f, dl in sorted(measured.items(), key=lambda x: -abs(x[1]))[:25]:
    print(f"  f={f:4d}  Δlogp={dl:+.2f}")

print(f"\nn_measured = {len(measured)}")
