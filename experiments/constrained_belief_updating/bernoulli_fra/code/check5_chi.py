"""Check 5 — A4: chi-realizability and MCC insufficiency.
Two codes for the SAME model over N=3 GT features:
  (A) complete: L=3 latents = GT features (chi=I).  Sever any feature exactly.
  (B) absence : L=2 near-identity contrast latents; feature 1 encoded by (weak)
      absence.  MCC ~ 1 (Hungarian over min(L,N)=2), but e_1 (indeed every pure
      e_i) is NOT in rowspace(chi) -> "sever feature i" is unrealizable.
Decisive numbers: rowspace residual of e_i (0 => realizable) and MCC of each code.
"""
import json, numpy as np
from scipy.optimize import linear_sum_assignment
from data import make_dictionary


def mcc(chi, D):
    """decoder cols w_l = sum_i chi[l,i] d_i ; Hungarian match to GT dirs."""
    W = chi @ D                                  # L x d  (unnormalised decode dirs)
    Wn = W / np.linalg.norm(W, axis=1, keepdims=True)
    S = np.abs(Wn @ D.T)                          # L x N cosine
    r, c = linear_sum_assignment(-S)
    return float(S[r, c].mean()), list(zip(r.tolist(), c.tolist()))


def sever_residual(chi, i):
    """min_xi || xi^T chi - e_i ||  ; 0 => feature i severable as a latent-set op."""
    N = chi.shape[1]
    e = np.zeros(N); e[i] = 1.0
    xi, *_ = np.linalg.lstsq(chi.T, e, rcond=None)   # chi^T xi = e
    achieved = chi.T @ xi
    return float(np.linalg.norm(achieved - e)), achieved.tolist(), xi.tolist()


def main():
    N, d = 3, 40
    D = make_dictionary(N, d, seed=3, orthogonalize=True)   # orthonormal GT

    chi_complete = np.eye(3)
    eps = 0.1
    chi_absence = np.array([[1.0, -eps, 0.0],
                            [0.0, -eps, 1.0]])              # feat 1 by weak absence

    out = {}
    for name, chi in [("complete_L3", chi_complete), ("absence_L2", chi_absence)]:
        m, match = mcc(chi, D)
        sev = {f"feat{i}": dict(residual=sever_residual(chi, i)[0],
                                achieved=sever_residual(chi, i)[1])
               for i in range(N)}
        out[name] = dict(chi=chi.tolist(), L=chi.shape[0], MCC=m,
                         matched_pairs=match,
                         rowspace_rank=int(np.linalg.matrix_rank(chi)),
                         sever=sev,
                         severable_features=[i for i in range(N)
                                             if sever_residual(chi, i)[0] < 1e-9])
    out["verdict"] = dict(
        complete_MCC=out["complete_L3"]["MCC"],
        absence_MCC=out["absence_L2"]["MCC"],
        complete_severable=out["complete_L3"]["severable_features"],
        absence_severable=out["absence_L2"]["severable_features"],
        headline="both high MCC, but absence code severs NO pure feature (rowspace=contrast subspace); MCC does not certify severability",
    )
    json.dump(out, open("../out/check5_chi.json", "w"), indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
