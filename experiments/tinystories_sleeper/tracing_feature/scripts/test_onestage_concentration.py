"""Apply concentration heuristic to the one-stage OV per-(h, λ) attribution.

Tests whether `max over h` or `top-K L1 over h` of one-stage OV attribution
predicts causal ablation impact better than the raw sum over h.
"""

import json
from pathlib import Path
import torch
import numpy as np

HERE = Path(__file__).resolve().parent
ov = torch.load(HERE.parent / "results" / "ov_path_per_pair.pt", weights_only=False)
C = ov["per_pair_dep_contrib"]  # (n_heads, d_sae_ln1): one-stage OV per-(h, lambda)
C_diff = C - ov["per_pair_cln_contrib"]

methods = {
    "one-stage raw SUM |Σ_h|":      C.sum(dim=0).abs(),
    "one-stage L1   Σ_h|C|":       C.abs().sum(dim=0),
    "one-stage MAX  max_h|C|":     C.abs().max(dim=0).values,
    "one-stage TOP-3 L1":          C.abs().topk(3, dim=0).values.sum(dim=0),
    "one-stage TOP-5 L1":          C.abs().topk(5, dim=0).values.sum(dim=0),
    "dep-cln raw  |Σ_h ΔC|":       C_diff.sum(dim=0).abs(),
    "dep-cln L1   Σ_h|ΔC|":        C_diff.abs().sum(dim=0),
    "dep-cln MAX  max_h|ΔC|":      C_diff.abs().max(dim=0).values,
    "dep-cln TOP-3 L1":            C_diff.abs().topk(3, dim=0).values.sum(dim=0),
}

with open(HERE.parent / "results" / "ln1_feature_ablation.json") as f:
    abl = json.load(f)

measured = {}
for r in abl["results"]:
    if r["alpha"] == 4.0 and isinstance(r["ln1_feature"], int):
        measured[r["ln1_feature"]] = r["delta_logp"]

feats = sorted(measured.keys())
dlogp = np.array([abs(measured[f]) for f in feats])

from scipy.stats import spearmanr

print("Spearman rho between method and |delta_logp| at alpha=4 (10 tested features):")
for name, scores in methods.items():
    vec = np.array([scores[f].item() for f in feats])
    r = spearmanr(vec, dlogp).correlation
    print(f"  {name:28s}: rho = {r:+.3f}")

print("\nPer-feature ranks under each method:")
header = f"{'feat':>6} {'|d_logp|':>9}"
for name in methods:
    header += f" {name[:12]:>13}"
print(header)
for f in sorted(feats, key=lambda x: -abs(measured[x])):
    line = f"{f:>6d} {abs(measured[f]):>9.2f}"
    for name, scores in methods.items():
        rank = (scores >= scores[f]).sum().item()
        line += f" {rank:>13d}"
    print(line)
