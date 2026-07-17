"""Per-token likelihood trace for continuation B under two sectors."""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research")
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from experiments.special_sfp_probe_factorization import headline_lowprior_cfg, domain_prompt

np.set_printoptions(precision=5, suppress=True)
cfg = headline_lowprior_cfg((0.025, 0.025, 0.475, 0.475))
tid = special_sfp.token_id
eta = 1e-3

ops = special_sfp.token_operators(cfg)
t_marg = ops.sum(axis=0)

prompt = np.array(domain_prompt("O"), dtype=np.int64)  # [0,3,3,3,0]
cont_B = np.array([tid(0, 0), tid(2, 2), tid(0, 0), tid(2, 2)], dtype=np.int64)  # flip
obs = np.concatenate([prompt, cont_B])
n_ctx = len(prompt)

for z, zname in [(0, "MD (best)"), (1, "MO (contradicted by prompt)")]:
    belief = np.zeros(cfg.n_states)
    belief[z * cfg.states_per_leaf] = 1.0
    print(f"\n=== sector {zname}, continuation B ===")
    print(" t  tok  raw_lik    smooth_lik   -log(smooth)  phase")
    total = 0.0
    for t in range(len(obs)):
        raw_post = belief @ ops[obs[t]]
        smooth_post = (1 - eta) * raw_post + (eta / 16) * (belief @ t_marg)
        raw_lik = raw_post.sum()
        lik = smooth_post.sum()
        phase = "prompt" if t < n_ctx else "CONT"
        nll = -np.log(max(lik, 1e-300))
        if t >= n_ctx:
            total += nll
        print(f" {t}  {obs[t]:2d}  {raw_lik:.5f}   {lik:.5f}     {nll:7.3f}      {phase}")
        belief = smooth_post / max(lik, 1e-300)
    print(f"  sum -log over continuation = {total:.3f}  -> logl = {-total:.3f}")
