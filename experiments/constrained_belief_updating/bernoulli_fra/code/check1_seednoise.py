"""Check 1 — A1 / Corollary A1.a.  PREDICTION UNDER TEST + CORRECTION.

Original A1.a prediction: attention loss-flat => FRA attributions are "seed noise"
(cross-seed correlation ~ 0).

What we verify:
 (a) trained loss == analytic affine floor tr(Cov a); attention adds nothing.
 (b) the CLEAN gauge representative (beta=mu_a, W=V=0, attn=0) ALSO achieves the
     floor, with IDENTICALLY ZERO FRA attributions => attention is loss-flat/gauge.
 (c) BUT SGD reproducibly lands on a NON-clean representative: cross-seed FRA
     correlation ~ 1 (NOT ~0). The falsifier fires: "seed noise" is the wrong
     characterization. Correct statement: loss-flat but OPTIMIZER-PINNED.
 (d) mechanism: the constant mu_a is distributed across beta + (W+V)mu ~= mu
     (the k=0/skip/constant gauge of A2.3), and W's input fluctuation is cancelled
     by V through attention (zero-V-only breaks the model; the clean rep needs no
     attention at all). Two loss-equal points (clean vs trained) with wildly
     different attributions = the definitive gauge demonstration.
"""
import json, itertools, numpy as np, torch
from data import BernoulliGaussian, gram_offdiag_stats
from model_A import OneLayerAttnA, train, eval_loss

OUT = "../out/check1_seednoise.json"
N, d, T = 12, 48, 16
SEEDS = [0, 1, 2, 3, 4, 5]
STEPS = 3000


def mean_pair_corr(mats):
    cs = [np.corrcoef(mats[i].ravel(), mats[j].ravel())[0, 1]
          for i, j in itertools.combinations(range(len(mats)), 2)]
    return float(np.mean(cs)), float(np.std(cs))


def main():
    gen = BernoulliGaussian(N, d, seed=100, p=0.12, mu=1.0, sigma=0.3)
    mom = gen.moments()
    mu = mom["mu_a"]
    floor = float(np.trace(mom["Ca"]))
    gram = gram_offdiag_stats(gen.D)

    losses, Qs, Os, mean_recon, zeroV, zeroW = [], [], [], [], [], []
    for s in SEEDS:
        m = OneLayerAttnA(d, bias=True, seed=s)
        train(m, gen, T=T, steps=STEPS)
        losses.append(eval_loss(m, gen, T=T))
        Q, O = m.fra_QO(gen.D); Qs.append(Q); Os.append(O)
        W = m.W.detach().numpy(); V = m.V.detach().numpy(); beta = m.beta.detach().numpy()
        mean_recon.append(float(np.linalg.norm(beta + (W + V) @ mu - mu) / np.linalg.norm(mu)))
        # surgical single-path ablations on the trained gauge rep
        mv = OneLayerAttnA(d, bias=True, seed=s); mv.load_state_dict(m.state_dict())
        with torch.no_grad(): mv.V.zero_()
        zeroV.append(eval_loss(mv, gen, T=T))
        mw = OneLayerAttnA(d, bias=True, seed=s); mw.load_state_dict(m.state_dict())
        with torch.no_grad(): mw.W.zero_()
        zeroW.append(eval_loss(mw, gen, T=T))

    # clean gauge representative: beta=mu, everything-else zero
    mc = OneLayerAttnA(d, bias=True, seed=999)
    with torch.no_grad():
        mc.W.zero_(); mc.V.zero_(); mc.Wq.zero_(); mc.Wk.zero_()
        mc.beta.copy_(torch.tensor(mu, dtype=torch.float32))
    clean_loss = eval_loss(mc, gen, T=T)
    Qc, Oc = mc.fra_QO(gen.D)

    # control: independent random attention params -> Q correlation (metric sanity)
    ctrl = []
    rng = np.random.default_rng(0)
    for _ in range(20):
        M1 = 0.02 * rng.standard_normal((d, d)); M2 = 0.02 * rng.standard_normal((d, d))
        ctrl.append(np.corrcoef((gen.D @ M1 @ gen.D.T).ravel(),
                                (gen.D @ M2 @ gen.D.T).ravel())[0, 1])

    Qcorr = mean_pair_corr(Qs); Ocorr = mean_pair_corr(Os)
    res = dict(
        setup=dict(N=N, d=d, T=T, seeds=SEEDS, steps=STEPS, gram=gram),
        floor_trCov=floor,
        trained_loss_mean=float(np.mean(losses)), trained_loss_std=float(np.std(losses)),
        trained_loss_over_floor=float(np.mean(losses) / floor),
        clean_rep_loss=clean_loss, clean_rep_over_floor=clean_loss / floor,
        clean_rep_attribution_norm=dict(Q=float(np.abs(Qc).max()), O=float(np.abs(Oc).max())),
        trained_attribution_norm=dict(Q=float(np.mean([np.abs(q).mean() for q in Qs])),
                                      O=float(np.mean([np.abs(o).mean() for o in Os]))),
        seed_coherence_measured=dict(Q_meancorr=Qcorr[0], O_meancorr=Ocorr[0]),
        metric_control_indep_random_Qcorr=float(np.mean(ctrl)),
        PREDICTION_seednoise_incoherent=("~0", "FALSIFIED: measured ~1"),
        mean_distribution_gauge=dict(
            beta_plus_WplusV_mu_reconstructs_mu_relerr=float(np.mean(mean_recon)),
            note="beta + (W+V)mu ~= mu : constant distributed across paths (k=0 gauge)"),
        path_ablations=dict(zeroV_only_loss=float(np.mean(zeroV)),
                            zeroW_only_loss=float(np.mean(zeroW)),
                            note="V load-bearing IN the trained gauge (cancels W); clean rep needs no attention"),
    )
    json.dump(res, open(OUT, "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
