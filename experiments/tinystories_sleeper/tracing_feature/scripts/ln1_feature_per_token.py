"""Quick analysis: per-token activation of the FRA-identified ln1 features.
Prints dep_mean, cln_mean, diff at each token position for a feature list.
"""

import sys
from pathlib import Path
import torch

HERE = Path(__file__).resolve().parent
cache_path = HERE.parent / "results" / "layer0_cache.pt"
p = torch.load(cache_path, weights_only=False)
z_ln1 = p["encodings"]["z_ln1"]
is_dep = p["is_deployment"]

TOKENS = {0: "Summary:/Words:", 1: ":", 2: " |", 3: "DE", 4: "PL", 5: "OY", 6: "MENT", 7: "|"}

for f in [870, 1388, 1220, 1412, 221]:
    z = z_ln1[:, :, f]
    dep_mean = z[is_dep].mean(dim=0)
    cln_mean = z[~is_dep].mean(dim=0)
    print(f"=== ln1 f={f}: first 12 positions ===")
    for t in range(12):
        d = dep_mean[t].item()
        c = cln_mean[t].item()
        label = TOKENS.get(t, f"pos{t}")
        print(f"  t={t:2d} {label:18s}: dep={d:+.3f}  cln={c:+.3f}  diff={d-c:+.3f}")
