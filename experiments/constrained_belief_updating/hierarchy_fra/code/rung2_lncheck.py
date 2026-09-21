"""Rung 2 corroboration of WORLD B: is the trained 'position-only' model beating the
lag-only frontier because LayerNorm's content-dependent normalization smuggles in an
approximate product (the parent->child gate)?

Retrain gated_occ THREE ways and co-evaluate child MSE on ONE common eval batch
(same batch used for the analytic frontiers), so trained numbers + frontiers are
directly comparable (no seed offset):
  - full     : full content QK (the optimum).
  - pos_LN   : position-only QK (Wq=Wk=0, frozen), WITH LayerNorm  [rung1's pos-only].
  - pos_linLN: position-only QK, LayerNorm REPLACED by a FIXED per-dim affine
               (content-independent -> linear). Removes the per-token normalization
               nonlinearity while keeping activations well-scaled.
Prediction (world B): pos_LN sits near the Bayes floor (captures the gate via LN);
pos_linLN rises toward the affine lag-only frontier (~0.25) once the nonlinearity is gone.

Writes out/rung2_lncheck.json.
"""
import sys, os, json, numpy as np, torch, torch.nn as nn
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess, relu_moments
from hier_model import MLAttn, train_ml
import rung2

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = rung2.PARAMS
T = 24


class FixedAffine(nn.Module):
    """Content-INDEPENDENT (linear) stand-in for LayerNorm: (x-mean)*inv_std, both fixed."""
    def __init__(self, mean, inv_std):
        super().__init__()
        self.register_buffer("mean", torch.tensor(mean, dtype=torch.float32))
        self.register_buffer("inv_std", torch.tensor(inv_std, dtype=torch.float32))

    def forward(self, x):
        return (x - self.mean) * self.inv_std


def make_model(pos_only, lin_ln, gen, seed=0, drop_ln=False):
    m = MLAttn(32, n_layers=2, n_ctx=24, seed=seed)
    if pos_only:
        for blk in m.blocks:
            with torch.no_grad():
                blk.Wq.zero_(); blk.Wk.zero_()
            blk.Wq.requires_grad_(False); blk.Wk.requires_grad_(False)
    if drop_ln:                                    # user directive 2026-07-17: theory-clean = NO LN
        for blk in m.blocks:
            blk.ln = nn.Identity()
    elif lin_ln:                                   # linearized-LN stand-in (kept for cross-check)
        X, _, _ = gen.sample_seq(4000, T)
        mean = X.reshape(-1, 32).mean(0); std = X.reshape(-1, 32).std(0) + 1e-6
        for blk in m.blocks:
            blk.ln = FixedAffine(mean, 1.0 / std)
    return m


def child_mse_on(m, Xt, y, dC):
    with torch.no_grad():
        pred = m(Xt).numpy()
    return float(((pred @ dC - y) ** 2).mean())


def frontiers_on(gen, occ, B):
    """Analytic frontiers (rotation-invariant) on a batch drawn from `gen` (its dict)."""
    X, Y, _ = gen.sample_seq(B, T)
    dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP); bC = float(gen.b @ dC)
    cP = X @ dP - bP; cC = X @ dC - bC
    y = Y @ dC; m1 = gen.m1_C
    from scipy.stats import norm
    qP = float(norm.cdf(-PARAMS["mu_P"] / PARAMS["sig_P"]))
    pi = rung2.bayes_child_pi(cP, cC, True, occ, gen.q_C, qP)
    mse_bayes = float(((y - (m1 * pi + bC)) ** 2).mean())
    pi_pb = rung2.bayes_child_pi(cP, cC, True, occ, gen.q_C, qP, parent_blind=True)
    mse_pblind = float(((y - (m1 * pi_pb + bC)) ** 2).mean())
    mse_lag_rel = rung2.lag_rel_mse(cP, cC, y, Lmax=T - 1)
    return dict(bayes=mse_bayes, parent_blind_bayes=mse_pblind,
                affine_lag_only=mse_lag_rel, gating_value=mse_pblind - mse_bayes)


def main():
    gated, occ = True, 0.5
    B_eval = 60000
    # Frontiers are rotation-invariant; compute them on the SEED-0 dictionary (the one the
    # models train on) so the whole table lives on one geometry. (Cross-checked vs the
    # seed-1234 rung2_frontiers.json: they match within MC error.)
    gen_fr = HierProcess(d=32, gated=gated, child_occlude=occ, seed=0, **PARAMS)
    fr = frontiers_on(gen_fr, occ, 120000)
    R = {"cell": "gated_occ", "eval_B": B_eval, "T": T, "frontiers": fr}
    print(f"frontiers(seed0 dict): bayes={fr['bayes']:.5f} pblind={fr['parent_blind_bayes']:.5f} "
          f"affine_lag={fr['affine_lag_only']:.5f}  gating_value={fr['gating_value']:+.5f}", flush=True)

    configs = [("full", False, False), ("pos_LN", True, False), ("pos_linLN", True, True)]
    R["trained"] = {}
    for nm, pos_only, lin_ln in configs:
        # Train AND eval on the SAME seed-0 dictionary (rung1's protocol). Eval batch is
        # drawn from gen_tr AFTER training (rng advanced -> fresh unseen samples, same dict).
        gen_tr = HierProcess(d=32, gated=gated, child_occlude=occ, seed=0, **PARAMS)
        m = make_model(pos_only, lin_ln, gen_tr, seed=0)
        train_ml(m, gen_tr, T=T, steps=9000)
        Xe, Ye, _ = gen_tr.sample_seq(B_eval, T)
        dC = gen_tr.D[1]
        mse = child_mse_on(m, torch.tensor(Xe, dtype=torch.float32), Ye @ dC, dC)
        R["trained"][nm] = mse
        json.dump(R, open(os.path.join(OUT, "rung2_lncheck.json"), "w"), indent=2)
        print(f"[{nm:9s}] child_mse={mse:.5f}  (bayes={fr['bayes']:.5f}, "
              f"affine_lag={fr['affine_lag_only']:.5f}, pblind={fr['parent_blind_bayes']:.5f})", flush=True)
    print("wrote out/rung2_lncheck.json", flush=True)


if __name__ == "__main__":
    main()
