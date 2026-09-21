"""Check 4 — A3 dilution scalings on a FIXED trained model.
Split GT feature into m latents sharing its direction, each carrying coeff/m.
Predict: per-pair QK coeff ∝ 1/m^2, per-latent OV ∝ 1/m, group-cut effect
invariant in m to float precision.  (Pure FRA bookkeeping; model weights fixed.)
"""
import json, numpy as np, torch
from data import BernoulliGaussian
from model_A import OneLayerAttnA, train

MS = [1, 2, 4, 8, 16]


def main():
    gen = BernoulliGaussian(12, 48, seed=100, p=0.12, mu=1.0, sigma=0.3)
    m = OneLayerAttnA(48, bias=True, seed=0)
    train(m, gen, T=16, steps=800)      # fixed model; scalings are weight-independent
    D = gen.D
    V = m.V.detach().numpy()
    Wqk = (m.Wq @ m.Wk.T / (m.d_h ** 0.5)).detach().numpy()

    # pick a sequence + indices where the relevant coefficients actually fire
    d0, s0, i, j = 10, 4, 3, 7
    for _try in range(2000):
        X, Y, c = gen.sample_seq(1, 16)
        c0 = c[0]                                  # T x N
        if c0[s0, i] > 0 and c0[d0, i] > 0 and c0[s0, j] > 0:
            break
    A = m.pattern(torch.tensor(X, dtype=torch.float32)).detach().numpy()[0]   # T x T
    assert c0[s0, i] > 0 and c0[d0, i] > 0 and c0[s0, j] > 0, "no active sample found"

    o_i = V @ D[i]                                # OV image of feature i
    Q_ij = float(D[i] @ Wqk @ D[j])

    rows = []
    for mm in MS:
        # OV: per-latent contribution of one split-latent of feature i at (d0<-s0)
        per_latent_OV = A[d0, s0] * (c0[s0, i] / mm) * o_i
        group_OV = sum(A[d0, s0] * (c0[s0, i] / mm) * o_i for _ in range(mm))
        # QK: per-pair score coeff for split pair (i^a at d0, j^b at s0)
        per_pair_QK = (c0[d0, i] / mm) * (c0[s0, j] / mm) * Q_ij
        group_QK = sum((c0[d0, i] / mm) * (c0[s0, j] / mm) * Q_ij
                       for _ in range(mm * mm))
        rows.append(dict(m=mm,
                         per_latent_OV_norm=float(np.linalg.norm(per_latent_OV)),
                         group_OV_norm=float(np.linalg.norm(group_OV)),
                         per_pair_QK=float(per_pair_QK),
                         group_QK=float(group_QK)))

    base_OV = rows[0]["per_latent_OV_norm"]; base_QK = rows[0]["per_pair_QK"]
    grp_OV0 = rows[0]["group_OV_norm"]; grp_QK0 = rows[0]["group_QK"]
    for r in rows:
        r["OV_ratio_to_1overm"] = (r["per_latent_OV_norm"] / base_OV) * r["m"]      # ->1
        r["QK_ratio_to_1overm2"] = (r["per_pair_QK"] / base_QK) * r["m"] ** 2       # ->1
        r["groupOV_invariance_relerr"] = abs(r["group_OV_norm"] - grp_OV0) / grp_OV0
        r["groupQK_invariance_relerr"] = abs(r["group_QK"] - grp_QK0) / abs(grp_QK0)

    res = dict(setup=dict(d0=d0, s0=s0, i=i, j=j, Q_ij=Q_ij),
               rows=rows,
               verdict=dict(
                   OV_per_latent_scales_1_over_m=all(abs(r["OV_ratio_to_1overm"] - 1) < 1e-6 for r in rows),
                   QK_per_pair_scales_1_over_m2=all(abs(r["QK_ratio_to_1overm2"] - 1) < 1e-6 for r in rows),
                   group_cuts_invariant=max(max(r["groupOV_invariance_relerr"], r["groupQK_invariance_relerr"]) for r in rows)))
    json.dump(res, open("../out/check4_dilution.json", "w"), indent=2)
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
