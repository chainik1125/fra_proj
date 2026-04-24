"""Final unified QK-side analysis: recompute per-μ scores, merge with all
measured ablations, Spearman correlations."""

import json, math, sys
from pathlib import Path
import torch
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent
sys.path.insert(0, str(EXP_DIR))

from run_ablation_sweep import load_crosscoder

print("[final] loading cache + SAEs + model...")
cache = torch.load(HERE.parent / "results" / "layer0_cache.pt", weights_only=False)
meta = cache["meta"]
ov = torch.load(HERE.parent / "results" / "ov_path_per_pair.pt", weights_only=False)

hooks = cache["hooks"]
z_ln1 = cache["encodings"]["z_ln1"]
A = hooks["attn_pattern"].float()
is_dep = cache["is_deployment"]
marker = cache["story_marker_pos"]
N, T, d_sae_ln1 = z_ln1.shape
n_heads = meta["n_heads"]
d_head = meta["d_head"]

device = "cuda" if torch.cuda.is_available() else "cpu"
from sleeper_utils import load_sleeper_model
model = load_sleeper_model(device=device)
W_Q = model.W_Q[0].detach().to(device).float()
W_K = model.W_K[0].detach().to(device).float()
sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device=device)
F = sae_ln1.W_dec.detach().to(device).float()

F_Q = torch.stack([F @ W_Q[h] for h in range(n_heads)])
F_K = torch.stack([F @ W_K[h] for h in range(n_heads)])
scale = 1.0 / math.sqrt(d_head)

beta = ov["beta"].to(device)
z_ln1_dev = z_ln1.to(device).float()
g = torch.einsum("bjv,vh->bhj", z_ln1_dev, beta)
g_bar = torch.einsum("bhqj,bhj->bhq", A.to(device), g)
tilde_g = A.to(device) * (g.unsqueeze(2) - g_bar.unsqueeze(-1))

idx = torch.arange(T).unsqueeze(0)
prompt_mask = idx <= marker.unsqueeze(1)
dep_mask = is_dep.unsqueeze(1) & prompt_mask
dep_bq = dep_mask.nonzero(as_tuple=False)
M = dep_bq.shape[0]
b_idx = dep_bq[:, 0]; q_idx = dep_bq[:, 1]

# Compute per-μ score at each dep prompt position — see qk_concentration.py.
print(f"[final] computing r + T_aggr...")
r = torch.einsum("bjv,hvd->bhjd", z_ln1_dev, F_K)
T_aggr = torch.einsum("bhqj,bhjd->bhqd", tilde_g, r)
del r
if device == "cuda":
    torch.cuda.empty_cache()

T_aggr_sel = T_aggr[b_idx, :, q_idx, :]  # (M, n_heads, d_head)
S = torch.einsum("hud,mhd->mu", F_Q, T_aggr_sel) * scale

U_dep = z_ln1_dev[b_idx, q_idx, :]
contribution = (U_dep * S).cpu()  # (M, d_sae_ln1)

signed_mean = contribution.mean(dim=0).abs()
abs_mean = contribution.abs().mean(dim=0)
max_abs = contribution.abs().max(dim=0).values
topK_sum = contribution.abs().topk(min(50, M), dim=0).values.sum(dim=0)

# Load measured Δlogp from both ablation files
with open(HERE.parent / "qk_vs_ov" / "results" / "ln1_feature_ablation.json") as f:
    abl_new = json.load(f)
with open(HERE.parent / "results" / "ln1_feature_ablation.json") as f:
    abl_orig = json.load(f)

measured = {}
for src in [abl_orig, abl_new]:
    for r in src["results"]:
        if r["alpha"] == 4.0 and isinstance(r["ln1_feature"], int):
            measured[r["ln1_feature"]] = r["delta_logp"]

feats = sorted(measured.keys())
dlogp = np.array([abs(measured[f]) for f in feats])
print(f"\n[final] n_measured = {len(feats)} features")

methods = {
    "QK signed_mean": signed_mean,
    "QK L1_mean": abs_mean,
    "QK max_over_(b,q)": max_abs,
    "QK top-50 L1": topK_sum,
}

print("\n=== Spearman ρ vs |Δlogp| at α=4 ===")
for name, scores in methods.items():
    vec = np.array([scores[f].item() for f in feats])
    rho = spearmanr(vec, dlogp).correlation
    print(f"  {name:>28s}: ρ = {rho:+.3f}")

# Per-feature table
print("\n=== Per-feature ranks and raw Δlogp ===")
header = f"{'feat':>6} {'|dlogp|':>9}"
for name in methods:
    header += f" {name[:14]:>16}"
print(header)
for f in sorted(feats, key=lambda x: -abs(measured[x])):
    line = f"{f:>6d} {abs(measured[f]):>9.2f}"
    for name, scores in methods.items():
        rank = (scores >= scores[f]).sum().item()
        line += f" {rank:>16d}"
    print(line)

# Save per-μ scores for later use
out = {
    "methods": {name: {"top30": [int(scores.topk(30).indices[i].item()) for i in range(30)]} for name, scores in methods.items()},
    "per_feature": {str(f): {
        "delta_logp": measured[f],
        "abs_delta_logp": abs(measured[f]),
        **{name.replace(" ", "_"): {
            "rank": int((scores >= scores[f]).sum().item()),
            "value": float(scores[f].item()),
        } for name, scores in methods.items()}
    } for f in feats},
    "spearman": {name: float(spearmanr(np.array([scores[f].item() for f in feats]), dlogp).correlation) for name, scores in methods.items()},
    "n_tested": len(feats),
}
import json as j
(HERE.parent / "qk_vs_ov" / "results" / "final_qk_analysis.json").write_text(j.dumps(out, indent=2))
print("\n[final] wrote final_qk_analysis.json")
