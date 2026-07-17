"""
Prompt-level bootstrap CI on the cross-finetune misalignment-correlation ρ̄ (§5 rigor).
The §5 headline "EM errors are correlated across finetunes (ρ̄=0.38)" rests on 16 prompts; this
quantifies its uncertainty by resampling the 16 prompts with replacement and recomputing the mean
pairwise correlation of the per-prompt misalignment-rate profiles (results/real_em_xadapter.pt).
Pure CPU analysis of existing data (no model calls). Prints the 95% CI and P(ρ̄>0).
"""
import os, sys
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RES = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))


def mean_pair_corr(mat, pairs):
    cs = [np.corrcoef(mat[i], mat[j])[0, 1] for i, j in pairs]
    return float(np.mean(cs)), cs


def main():
    d = torch.load(os.path.join(RES, "real_em_xadapter.pt"), weights_only=False)
    p = np.asarray(d["p_mat"]); ad = d["adapters"]
    pairs = [(0, 1), (0, 2), (1, 2)]
    names = [f"{ad[i]}-{ad[j]}" for i, j in pairs]
    obs_mean, obs_cs = mean_pair_corr(p, pairs)
    print(f"adapters={ad}  p_mat={p.shape}")
    print("observed per-pair:", dict(zip(names, [round(c, 3) for c in obs_cs])))
    print(f"observed mean ρ̄ = {obs_mean:.4f}")
    rng = np.random.default_rng(0); B = 10000; n = p.shape[1]
    boot = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        with np.errstate(invalid="ignore"):
            m = mean_pair_corr(p[:, idx], pairs)[0]
        if not np.isnan(m):
            boot.append(m)
    boot = np.array(boot)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"bootstrap 95% CI on ρ̄: [{lo:.3f}, {hi:.3f}]  (n_boot={len(boot)}, median {np.median(boot):.3f})")
    print(f"P(ρ̄>0) = {(boot > 0).mean():.3f}   P(ρ̄>0.2) = {(boot > 0.2).mean():.3f}")


if __name__ == "__main__":
    main()
