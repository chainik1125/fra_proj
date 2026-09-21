"""Decoupled-bias control: the b || d_P shared-RNG artifact PERSISTS across seeds (structural),
so the mean-ablation convention is always load-bearing. Check it isn't driving the FRA win:
give the process a GENERIC bias (decoupled from d_P), retrain, and run the frontier under BOTH
mean-ablation and zero-ablation. If FRA still beats DoM/SAE under both -> the win is real.
Saves out/decouple_b_check.json.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
from hier_model import train_ml
import centerpiece as cp
import control_frontier as cf

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = cp.PARAMS; T = 24
gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def crossover(xs, ys, extrap=False):
    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1] - 1e-9:
            break
        if ys[i - 1] < 1 <= ys[i]:
            t = (1 - ys[i - 1]) / (ys[i] - ys[i - 1]); return xs[i - 1] + t * (xs[i] - xs[i - 1])
    if extrap and ys[-1] > 0.5:
        return xs[-2] + (1 - ys[-2]) / (ys[-1] - ys[-2]) * (xs[-1] - xs[-2])
    return None


def main():
    gen = HierProcess(d=32, gated=True, child_occlude=0.5, seed=0, **PARAMS)
    gen.b = 0.5 * np.random.default_rng(9999).standard_normal(32)   # GENERIC bias, decoupled from d_P
    dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP); bC = float(gen.b @ dC)
    print(f"decoupled bias: b.dP={bP:+.3f}  b.dC={bC:+.3f}  (generic, |.|~O(0.4))", flush=True)

    ck = os.path.join(OUT, "ck_decoupledb.pt")
    m = make_model(False, False, gen, seed=0, drop_ln=True)
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck))
    else:
        train_ml(m, gen, T=T, steps=cp.STEPS); torch.save(m.state_dict(), ck)
    m.eval()

    X, Y, _ = gen.sample_seq(15000, T)
    fr = cp.frontiers(gen, X, Y, True)
    base_c, base_p = cf.mse(m, X, Y, dP, dC)
    span = fr["parent_blind"] - base_c
    dPt = torch.tensor(dP, dtype=torch.float32)
    Vsae = (dPt / dPt.norm())[:, None]; V3 = cf.fit_probe(X, dP, bP, 3)
    methods = {"FRA_OV_child": lambda g, ma: dict(ovchild_gain=g, mean_ablate=ma),
               "DoM_rank3": lambda g, ma: dict(resV=V3, res_gain=g, mean_ablate=ma),
               "SAE_planted": lambda g, ma: dict(resV=Vsae, res_gain=g, mean_ablate=ma)}
    R = {"decoupled_bias": {"b_dot_dP": bP, "b_dot_dC": bC}, "anchors":
         {"base_child": base_c, "parent_blind": fr["parent_blind"], "span": span, "base_parent": base_p},
         "crossover": {}}
    for conv, ma in [("mean_ablate", True), ("zero_ablate", False)]:
        R["crossover"][conv] = {}
        for name, mk in methods.items():
            xs, ys = [], []
            for g in gains:
                c, p = cf.mse(m, X, Y, dP, dC, **mk(g, ma))
                xs.append(p - base_p); ys.append((c - base_c) / span)
            R["crossover"][conv][name] = crossover(xs, ys, extrap=(name == "FRA_OV_child"))
        cr = R["crossover"][conv]
        fc, d3, sa = cr["FRA_OV_child"], cr["DoM_rank3"], cr["SAE_planted"]
        print(f"[{conv}] crossover collat @rem=1: FRA_child={fc:.5f} DoM3={d3:.5f} SAE={sa:.5f} | "
              f"FRA<DoM3={fc < d3} DoM3/FRA={d3/fc:.0f}x SAE/FRA={sa/fc:.0f}x", flush=True)
        json.dump(R, open(os.path.join(OUT, "decouple_b_check.json"), "w"), indent=2)
    print("wrote out/decouple_b_check.json", flush=True)


if __name__ == "__main__":
    main()
