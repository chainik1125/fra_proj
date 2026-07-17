"""Acceptance test for bag_moments/coherence.py.

1. Ideal hedging learner (w_MO in {0.0013, 0.4}) sampling its own continuations:
   xe should be ~0 / slightly negative, corner-jsd should show the hedging floor,
   mixjsd should be ~0 in BOTH regimes (the fix).
2. Invariant: mixjsd <= corner jsd (grid includes corners).
3. Worked-example continuations A/B/C: xe logliks/dispositions must match the old
   sector_logliks implementation exactly.
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path("/Users/dmitrymanning-coe/Documents/Research/Simplex/simplex-research")
sys.path.insert(0, str(ROOT))

from bag_moments import special_sfp
from bag_moments.coherence import coherence_metrics_all, sector_scan, _thick_step, _thick_predictive
from experiments.special_sfp_probe_factorization import (
    headline_lowprior_cfg,
    domain_prompt,
    sector_logliks,
)

cfg = headline_lowprior_cfg((0.025, 0.025, 0.475, 0.475))
ops = special_sfp.token_operators(cfg)
t_marg = ops.sum(axis=0)
eta = 1e-3
V = cfg.vocab_size
INITS = [z * cfg.states_per_leaf for z in range(4)]
NAMES = special_sfp.LEAVES
prompt = np.array(domain_prompt("O"), dtype=np.int64)
N, T = 1024, 32


def sample_ideal(w_mo, seed):
    """Ideal learner with post-prompt weights (w_mo MO, 1-w_mo AO) samples itself."""
    rng = np.random.default_rng(seed)
    corners = []
    for init in INITS:
        b = np.zeros((1, cfg.n_states))
        b[:, init] = 1.0
        for tok in prompt:
            b, _, _ = _thick_step(b, ops[np.array([tok])], t_marg, eta, V)
        corners.append(b[0])
    bmix = w_mo * corners[1] + (1 - w_mo) * corners[3]
    bmix = np.repeat((bmix / bmix.sum())[None, :], N, axis=0)
    cont = np.empty((N, T), dtype=np.int64)
    logp = np.empty((N, T))
    probs = np.empty((N, T, V))
    idx = np.arange(N)
    for t in range(T):
        pm = _thick_predictive(bmix, ops, eta, V)
        toks = np.clip((rng.random(N)[:, None] > np.cumsum(pm, axis=1)).sum(axis=1), 0, V - 1)
        cont[:, t] = toks
        probs[:, t] = pm
        logp[:, t] = np.log(pm[idx, toks])
        bmix, _, _ = _thick_step(bmix, ops[toks], t_marg, eta, V)
    return cont, logp, probs


print("=== 1+2: ideal-learner floors through coherence_metrics_all ===")
ctx = np.repeat(prompt[None, :], N, axis=0)
for w in (0.0013, 0.4):
    cont, logp, probs = sample_ideal(w, seed=0)
    m = coherence_metrics_all(ops, INITS, NAMES, ctx, cont, logp, "O", eta, model_probs=probs)
    print(f"w_MO={w}: xe={m['O_coh_deficit']:+.4f}  jsd={m['O_jsd_deficit']:.4f}  "
          f"mixjsd={m['O_mixjsd_deficit']:.4f}  mixw_MO={m['O_mixw_MO']:.3f}  mixw_AO={m['O_mixw_AO']:.3f}")
    assert m["O_mixjsd_deficit"] <= m["O_jsd_deficit"] + 1e-12, "invariant violated"

print("\n=== 3: A/B/C continuations, new xe path vs old sector_logliks ===")
tid = special_sfp.token_id
p5 = np.array(domain_prompt("O"), dtype=np.int64)[None, :]
for name, cont in [
    ("A", [[tid(0, 0), tid(2, 3), tid(0, 0), tid(2, 3)]]),
    ("B", [[tid(0, 0), tid(2, 2), tid(0, 0), tid(2, 2)]]),
    ("C", [[tid(0, 0), tid(3, 3), tid(0, 0), tid(2, 3)]]),
]:
    cont = np.asarray(cont, dtype=np.int64)
    old_logl, old_imp = sector_logliks(cfg, p5, cont)
    scan = sector_scan(ops, INITS, p5, cont, eta)
    assert np.allclose(old_logl, scan["logl"]), f"{name}: logl mismatch"
    assert np.allclose(old_imp, scan["impossible_rate"]), f"{name}: imp mismatch"
    print(f"{name}: logl match ({np.round(scan['logl'][0], 3).tolist()}), imp={scan['impossible_rate'][0]:.2f}")

print("\nall checks passed")
