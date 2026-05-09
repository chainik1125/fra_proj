"""Per-feature firing stats: how often does each feature fire on dep vs clean
prompt positions? Used to explain why OV-selected features preserve coherence."""

import json, sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
EXP_DIR = HERE.parent.parent.parent
sys.path.insert(0, str(EXP_DIR))

cache = torch.load(HERE.parent.parent / "results" / "layer0_cache.pt", weights_only=False)
z_ln1 = cache["encodings"]["z_ln1"].float()          # (N, T, F)
is_dep = cache["is_deployment"]                       # (N,)
marker = cache["story_marker_pos"]                    # (N,)

N, T, F = z_ln1.shape
idx = torch.arange(T).unsqueeze(0)
prompt_mask = idx <= marker.unsqueeze(1)              # (N, T)

dep_mask = is_dep.unsqueeze(1) & prompt_mask
cln_mask = (~is_dep).unsqueeze(1) & prompt_mask

feats = {
    "ov_top3":    [1205, 1114, 337],
    "qk_top3":    [870, 1388, 760],
    "qk_top5":    [870, 1388, 760, 303, 435],
    "sweep":      [1412],
}

print(f"\nDataset: {N} test prompts, deployment={is_dep.sum().item()}, clean={(~is_dep).sum().item()}")
print(f"Total prompt positions: dep={dep_mask.sum().item()}, clean={cln_mask.sum().item()}")

for name, flist in feats.items():
    print(f"\n==== {name} = {flist} ====")
    for f in flist:
        z = z_ln1[:, :, f]                            # (N, T)
        dep_z = z[dep_mask]                           # flat
        cln_z = z[cln_mask]
        print(f"  f={f:>5d}  dep: fire_rate={((dep_z > 0).float().mean()).item():.3f}  mean|z|={dep_z.abs().mean().item():.3f}  max|z|={dep_z.abs().max().item():.2f}"
              f"    |  clean: fire_rate={((cln_z > 0).float().mean()).item():.3f}  mean|z|={cln_z.abs().mean().item():.3f}  max|z|={cln_z.abs().max().item():.2f}"
              f"    |  dep/clean ratio(mean|z|) = {(dep_z.abs().mean()/max(1e-8, cln_z.abs().mean())).item():.2f}")

# Also: per-feature SAE-decoder vector norm (how much a single z-unit moves resid)
print("\n==== Decoder norms (||W_dec[f]||_2) ====")
from run_ablation_sweep import load_crosscoder
meta = cache["meta"]
sae_ln1, _ = load_crosscoder(EXP_DIR / meta["sae_configs"]["ln1"]["path"], device="cpu")
W = sae_ln1.W_dec.detach().float()  # (F, d_model)
all_feats = sorted({f for flist in feats.values() for f in flist})
for f in all_feats:
    print(f"  f={f:>5d}  ||W_dec[f]|| = {W[f].norm().item():.3f}")
