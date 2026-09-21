"""Diagnosis rung 1: redesigned LOSS GATE (full-QK vs position-only-QK, 2L) per cell
+ H2.2 predicted Delta_gate at the new params. Splits: (i) gated does NOT beat
pos-only ~= predicted -> gate never learned (training/capacity); (ii) it DOES ->
gate learned, the raw cut fails to isolate -> rung 2/3. Writes out/rung1.json.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess
from hier_model import MLAttn, train_ml
OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = dict(lam_P=0.5, p_P=0.3, lam_C=0.9, p_C=0.5, mu_P=1.5, sig_P=0.3, mu_C=1.5, sig_C=0.3)


def predicted_delta_gate(rho, qP, lamC, pC, kmax=20):
    tot = 0.0
    for k in range(1, kmax + 1):
        PEk = (1 - rho) * rho ** (k - 1)
        sk = qP ** k
        tot += rho * PEk * sk * (1 - sk) * (lamC ** k * (1 - pC)) ** 2
    return tot


def child_mse(m, gen, T=24, B_=12000):
    X, Y, _ = gen.sample_seq(B_, T)
    with torch.no_grad():
        pred = m(torch.tensor(X, dtype=torch.float32)).numpy()
    return float((((pred - Y) @ gen.D[1]) ** 2).mean())


def train_cell(gated, occ, pos_only, steps=9000, seed=0):
    g = HierProcess(d=32, gated=gated, child_occlude=occ, seed=seed, **PARAMS)
    m = MLAttn(32, n_layers=2, n_ctx=24, seed=seed)
    if pos_only:
        for blk in m.blocks:
            with torch.no_grad():
                blk.Wq.zero_(); blk.Wk.zero_()
            blk.Wq.requires_grad_(False); blk.Wk.requires_grad_(False)
    train_ml(m, g, T=24, steps=steps)
    return child_mse(m, g)


def main():
    qP = PARAMS["lam_P"] + (1 - PARAMS["lam_P"]) * PARAMS["p_P"]
    pred = predicted_delta_gate(0.5, qP, PARAMS["lam_C"], PARAMS["p_C"])
    R = {"config": PARAMS, "occlusion": 0.5, "depth": "2L", "qP": qP,
         "predicted_delta_gate_MSE": pred}
    print(f"H2.2 predicted Delta_gate^MSE at new params (qP={qP:.2f}, lamC=0.9, occ=0.5) = {pred:.5f}",
          flush=True)
    for gated in [True, False]:
        for occ in [0.5, 0.0]:
            nm = f"{'gated' if gated else 'null'}_{'occ' if occ else 'clean'}"
            full = train_cell(gated, occ, False)
            pos = train_cell(gated, occ, True)
            R[nm] = dict(child_mse_full=full, child_mse_posonly=pos, delta_gate_empirical=pos - full)
            json.dump(R, open(os.path.join(OUT, "rung1.json"), "w"), indent=2)
            print(f"{nm:12s}: full={full:.4f} pos={pos:.4f} Delta_gate={pos-full:+.5f}"
                  f"  (predicted {pred:.4f})", flush=True)
    print("wrote out/rung1.json", flush=True)


if __name__ == "__main__":
    main()
