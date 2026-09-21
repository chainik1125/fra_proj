"""Experiment 1 — RAW Setting-A SAE recovery sweep.

Do trained TopK SAEs recover the GT dictionary on i.i.d. Bernoulli-Gaussian
activations, and where does chi ~ I break FIRST (MCC vs severability)?

Configs span the superposition axis (rho_mm) at fixed sparsity; for each we train
L in {N, 2N} TopK SAEs with K matched to the true L0. No correlations, no hierarchy.
"""
import json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sae_common import CorrHierGen, train_sae, recovery_report, rho_mm

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "sae_exp1_raw.json")

CONFIGS = [
    # (name, N, d, orthogonalize, p)
    ("N16_d32_orth",  16, 32, True,  0.06),   # undercomplete, rho~0  -> expect chi~I
    ("N24_d24_orth",  24, 24, True,  0.06),   # complete square, rho~0
    ("N32_d24",       32, 24, False, 0.06),   # mild overcomplete
    ("N48_d24",       48, 24, False, 0.06),   # substantial superposition -> expect break
]
NSAMP = 120_000
STEPS = 3000


def run():
    results = {}
    t0 = time.time()
    for name, N, d, orth, p in CONFIGS:
        gen = CorrHierGen(N, d, seed=0, p=p, mu=1.0, sigma=0.5, orthogonalize=orth)
        acts, c = gen.sample_a(NSAMP)
        z_gt = (c > 0).astype(float)
        trueL0 = float(z_gt.sum(1).mean())
        K = max(1, int(round(trueL0)))
        rmm = rho_mm(gen.D)
        cfg = dict(N=N, d=d, rho_mm=rmm, trueL0=trueL0, K=K, p=p)
        cfg["sae"] = {}
        for Lmult in (1, 2):
            L = Lmult * N
            sae = train_sae(acts, L, K, steps=STEPS, batch=4096, seed=0)
            rep = recovery_report(sae, gen.D, acts, z_gt)
            rep = {k: v for k, v in rep.items() if not k.startswith("_")}
            cfg["sae"][f"L{Lmult}N"] = rep
            print(f"[{name}] L={L} K={K} rho_mm={rmm:.3f}  "
                  f"MCC={rep['mcc']:.3f} uniq={rep['uniqueness']:.3f} "
                  f"dead={rep['dead']} F1={rep['f1']:.3f} "
                  f"chi_rank={rep['chi_rank']}/{N} unsever={rep['n_unsever']} "
                  f"sev_max={rep['sev_max']:.3f}  ({time.time()-t0:.0f}s)")
        results[name] = cfg
    with open(OUT, "w") as f:
        json.dump(results, f, indent=2)
    print("wrote", OUT, f"total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    run()
