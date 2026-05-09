"""Test hypothesis: does "concentration" of (h, a) attributions predict causal
importance better than raw sum (which ≈ one-stage OV)?

For each ln1 feature λ, compute:
    raw_sum[λ]  = Σ_{h, a} β[h, λ] · γ[a, λ] · q_dep[h, a]              (signed)
    abs_sum[λ] = Σ_{h, a} |β[h, λ] · γ[a, λ] · q_dep[h, a]|              (L1)
    top_K[λ]   = sum over top-K (h, a) pairs of |β · γ · q_dep|
    max_abs[λ] = max over (h, a) of |β · γ · q_dep|

Then correlate each ranking with the observed ablation |Δlogp| at α=4 for
the features we've tested (1412, 870, 1388, 1220, 221, 500, 800, 1000, 300,
1205, 1114, 337, 1191, 865, 157).
"""

import json
import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder

# Load data
cache = torch.load(HERE.parent / "results" / "layer0_cache.pt", weights_only=False)
ov = torch.load(HERE.parent / "results" / "ov_path_per_pair.pt", weights_only=False)
pre = torch.load(HERE.parent / "results" / "pre_attn_path_per_pair.pt", weights_only=False)

meta = cache["meta"]
sae_pre_path = EXP_DIR / meta["sae_configs"]["pre"]["path"]
sae_ln1_path = EXP_DIR / meta["sae_configs"]["ln1"]["path"]

sae_pre, _ = load_crosscoder(sae_pre_path, device="cpu")
sae_ln1, _ = load_crosscoder(sae_ln1_path, device="cpu")

beta_T = ov["beta_T"]                   # (n_heads, d_sae_ln1)
n_heads, d_sae_ln1 = beta_T.shape
d_sae_pre = pre["dep_contrib_pair"].shape[1]

# psi and q_dep
psi = pre["psi"]                        # (n_heads, d_sae_pre)
dep_contrib_pair = pre["dep_contrib_pair"]
safe_psi = psi.clone()
safe_psi[safe_psi.abs() < 1e-10] = 1e-10
q_dep = dep_contrib_pair / safe_psi     # (n_heads, d_sae_pre)

# gamma[a, λ] = centered(SAE_pre.W_dec[a]) · SAE_ln1.W_enc[:, λ]
G = sae_pre.W_dec.float()               # (d_sae_pre, d_model)
G_centered = G - G.mean(dim=1, keepdim=True)
E_ln1 = sae_ln1.W_enc.float()           # (d_model, d_sae_ln1)
gamma = G_centered @ E_ln1              # (d_sae_pre, d_sae_ln1)
print(f"gamma shape={gamma.shape}")

# For each λ, compute 4 summary statistics.
# Full (h, a) contribution tensor per λ: shape (n_heads, d_sae_pre)
# contrib[h, a, λ] = β[h, λ] · γ[a, λ] · q_dep[h, a]
# Compute per-λ vector by iterating λ (to avoid materializing full 3D tensor).

raw_sum = torch.zeros(d_sae_ln1)
abs_sum = torch.zeros(d_sae_ln1)
max_abs = torch.zeros(d_sae_ln1)
top5_abs_sum = torch.zeros(d_sae_ln1)
top20_abs_sum = torch.zeros(d_sae_ln1)

# One-stage OV (for sanity)
per_pair_dep_contrib_ln1 = ov["per_pair_dep_contrib"]   # (n_heads, d_sae_ln1)
one_stage = per_pair_dep_contrib_ln1.sum(dim=0)         # (d_sae_ln1,)

print("Computing per-λ (h, a) attribution stats...")
for lam in range(d_sae_ln1):
    # contrib[h, a] = β[h, λ] · γ[a, λ] · q_dep[h, a]
    # = (β[:, λ].unsqueeze(1) * gamma[:, λ].unsqueeze(0)) * q_dep
    # shape (n_heads, d_sae_pre)
    contrib = (beta_T[:, lam].unsqueeze(1) * gamma[:, lam].unsqueeze(0)) * q_dep
    c_abs = contrib.abs()
    raw_sum[lam] = contrib.sum()
    abs_sum[lam] = c_abs.sum()
    max_abs[lam] = c_abs.max()
    flat = c_abs.flatten()
    top5_abs_sum[lam] = torch.topk(flat, 5).values.sum()
    top20_abs_sum[lam] = torch.topk(flat, 20).values.sum()

# Measured Δlogp at α=4 from earlier ablations (load from both files)
with open(HERE.parent / "results" / "ln1_feature_ablation.json") as f:
    abl = json.load(f)

measured_delta_logp = {}
for r in abl["results"]:
    if r["alpha"] != 4.0:
        continue
    f = r["ln1_feature"]
    if isinstance(f, int):
        measured_delta_logp[f] = r["delta_logp"]

print(f"\nMeasured features: {sorted(measured_delta_logp.keys())}")

# For each method, show the ranking rank of the measured features
methods = {
    "one_stage (Σ_h β·M)": one_stage.abs(),
    "two_stage raw_sum (all (h,a))": raw_sum.abs(),
    "two_stage abs_sum (L1)": abs_sum,
    "two_stage max over (h,a)": max_abs,
    "two_stage top-5 L1": top5_abs_sum,
    "two_stage top-20 L1": top20_abs_sum,
}

# Compute ranks for each measured feature under each method
print(f"\n{'feature':>8}  {'|Δlogp|':>8}  " + "  ".join(f"{n:>28}" for n in methods.keys()))
for f in sorted(measured_delta_logp.keys(), key=lambda x: -abs(measured_delta_logp[x])):
    dl = measured_delta_logp[f]
    parts = [f"{f:>8d}  {abs(dl):>8.2f}"]
    for name, scores in methods.items():
        # Rank: position in descending-sorted |score|
        rank = (scores >= scores[f]).sum().item()
        parts.append(f"  rank={rank:4d} v={scores[f]:+.3e}")
    print("  ".join(parts))

# Rank-correlation: Spearman between method score and |Δlogp|
import numpy as np
features_measured = list(measured_delta_logp.keys())
dlogp_vec = np.array([abs(measured_delta_logp[f]) for f in features_measured])
print(f"\nSpearman correlation with |Δlogp|:")
def _spearman(x, y):
    from scipy.stats import spearmanr
    return spearmanr(x, y).correlation
try:
    from scipy.stats import spearmanr
    for name, scores in methods.items():
        method_vec = np.array([scores[f].item() for f in features_measured])
        r = spearmanr(method_vec, dlogp_vec).correlation
        print(f"  {name:>32s}: ρ = {r:+.3f}")
except ImportError:
    # Fallback: Spearman manual
    def rankdata(x):
        order = np.argsort(-x)  # descending
        ranks = np.empty_like(order)
        ranks[order] = np.arange(len(x))
        return ranks
    for name, scores in methods.items():
        method_vec = np.array([scores[f].item() for f in features_measured])
        # Pearson on ranks
        rx = rankdata(-method_vec)  # descending by method
        ry = rankdata(-dlogp_vec)   # descending by dlogp
        mx, my = rx.mean(), ry.mean()
        num = ((rx - mx) * (ry - my)).sum()
        den = np.sqrt(((rx - mx) ** 2).sum() * ((ry - my) ** 2).sum())
        r = num / den if den > 0 else 0
        print(f"  {name:>32s}: ρ = {r:+.3f}")
