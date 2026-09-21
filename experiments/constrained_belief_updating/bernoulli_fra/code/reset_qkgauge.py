"""Check 2b: FRA-QK content sector is gauge (justifies position-only rate model).

At a pinned noise level, train K full-attention seeds on reset data:
 (a) full-model loss == position-only loss  -> content-QK carries no loss (gauge).
 (b) content x content FRA-QK block Q is seed-INcoherent (cross-seed corr ~ 0)
     while the positional lag-kernel rate is seed-COHERENT and loss is seed-stable.
This is check1_seednoise (Setting A) transferred to Setting B's attention sector.
Writes out/qkgauge.json.
"""
import json, itertools, numpy as np, torch
from reset import ResetProcess, geometric_rate_fit
from model import OneLayerAttn, train, eval_loss

D, T, N, SIGMA = 32, 28, 6, 0.8
SEEDS = [0, 1, 2]
STEPS = 3000


def pair_corr(mats):
    cs = [np.corrcoef(mats[i].ravel(), mats[j].ravel())[0, 1]
          for i, j in itertools.combinations(range(len(mats)), 2)]
    return float(np.mean(cs)), float(np.std(cs))


def main():
    g0 = ResetProcess(N, D, lam=0.7, p=0.3, mu=1.0, sigma=SIGMA, seed=0, orthogonalize=True)
    losses_full, losses_pos, Qs, rates = [], [], [], []
    for s in SEEDS:
        g = ResetProcess(N, D, lam=0.7, p=0.3, mu=1.0, sigma=SIGMA, seed=s,
                         orthogonalize=True, D=g0.D)
        # full model
        mf = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=s, pos_key=True, n_ctx=T)
        train(mf, g, T=T, steps=STEPS)
        losses_full.append(eval_loss(mf, g, T=T, B=8000))
        Q, _ = mf.fra_QO(g.D)                        # N x N content x content score
        Qc = Q - Q.mean(0, keepdims=True) - Q.mean(1, keepdims=True) + Q.mean()
        Qs.append(Qc[~np.eye(N, dtype=bool)])        # centered off-diag content block
        X, _, _ = g.sample_seq(3000, T)
        al = mf.mean_lag(torch.tensor(X, dtype=torch.float32), t_lo=12)
        rates.append(geometric_rate_fit(al, 1, 5))
        # position-only model (same seed)
        mp = OneLayerAttn(D, d_h=D, bias=True, init=0.02, seed=s, pos_key=True, n_ctx=T)
        with torch.no_grad():
            mp.Wq.zero_(); mp.Wk.zero_()
        mp.Wq.requires_grad_(False); mp.Wk.requires_grad_(False)
        train(mp, g, T=T, steps=STEPS)
        losses_pos.append(eval_loss(mp, g, T=T, B=8000))

    Qcorr = pair_corr(Qs)
    res = dict(setup=dict(D=D, T=T, N=N, SIGMA=SIGMA, seeds=SEEDS, steps=STEPS),
               loss_full_mean=float(np.mean(losses_full)), loss_full_std=float(np.std(losses_full)),
               loss_posonly_mean=float(np.mean(losses_pos)), loss_posonly_std=float(np.std(losses_pos)),
               loss_full_minus_posonly=float(np.mean(losses_full) - np.mean(losses_pos)),
               contentQ_seed_corr_mean=Qcorr[0], contentQ_seed_corr_std=Qcorr[1],
               rate_mean=float(np.mean(rates)), rate_std=float(np.std(rates)),
               eta_theory=float(g0.const['eta'][0]))
    json.dump(res, open("../out/qkgauge.json", "w"), indent=2)
    print(json.dumps(res, indent=2))
    print("\n=> content-QK gauge if: full~posonly loss AND contentQ_seed_corr~0 "
          "AND rate seed-coherent")


if __name__ == "__main__":
    main()
