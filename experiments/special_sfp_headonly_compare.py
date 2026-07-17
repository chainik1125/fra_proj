"""Compare head-only vs full fine-tuning broad transfer at matched narrow transfer."""

from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(
    "/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research/"
    "experiment_folders/em_afp_simpler_codex_auto"
)
CONDS = {
    "product": "0p025_0p025_0p475_0p475",
    "corr-strong": "0p040_0p010_0p460_0p490",
}
X_TARGETS = (0.05, 0.10, 0.15, 0.20, 0.25)


def interp(x, y, xt):
    order = np.argsort(x)
    xs, ys = np.asarray(x)[order], np.asarray(y)[order]
    if xt < xs.min() or xt > xs.max():
        return float("nan")
    return float(np.interp(xt, xs, ys))


rows = []
for name, tag in CONDS.items():
    for variant, suffix in (("full", ""), ("head-only", "_headonly")):
        csv = BASE / f"results_probe_factorization_pi0_{tag}{suffix}" / "combined_probe_metrics.csv"
        df = pd.read_csv(csv)
        df = df[df["layer"] == 2]
        for seed, g in df.groupby("seed"):
            g = g.sort_values("step")
            x = g["D_rollout_MD"].to_numpy()
            y = g["O_rollout_MO"].to_numpy()
            row = {"prior": name, "variant": variant, "seed": seed, "max_narrow": float(x.max())}
            for xt in X_TARGETS:
                row[f"O_MO@{xt}"] = interp(x, y, xt)
            rows.append(row)

out = pd.DataFrame(rows)
print(out.round(4).to_string(index=False))
print()
summary = out.groupby(["prior", "variant"])[[f"O_MO@{xt}" for xt in X_TARGETS] + ["max_narrow"]].mean()
print("means over seeds:")
print(summary.round(4).to_string())
