"""3-seed replication of the CONTROL FRONTIER headline (+ the b || d_P artifact check).
Reuses control_frontier.forward / fit_probe / mse. For each cached seed model, sweep every
method and interpolate the parent collateral at full gate-use removal (removal=1). Report the
FRA_OV_child vs DoM_rank3 (and vs SAE_planted) ordering per seed + mean/spread. Also logs
b.dP / b.dC per seed (does the shared-RNG bias artifact persist across geometries?).
Saves out/seed_frontier.json.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
import centerpiece as cp
import control_frontier as cf

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = cp.PARAMS; T = 24; SEEDS = [0, 1, 2]
gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]


def crossover(xs, ys):
    """collateral at which removal first reaches 1 (monotone-increasing prefix)."""
    for i in range(1, len(ys)):
        if ys[i] < ys[i - 1] - 1e-9:
            break
        if ys[i - 1] < 1 <= ys[i]:
            t = (1 - ys[i - 1]) / (ys[i] - ys[i - 1]); return xs[i - 1] + t * (xs[i] - xs[i - 1])
    return None


def main():
    R = {"seeds": SEEDS, "per_seed": {}}
    for s in SEEDS:
        gen = HierProcess(d=32, gated=True, child_occlude=0.5, seed=s, **PARAMS)
        m = make_model(False, False, gen, seed=s, drop_ln=True)
        m.load_state_dict(torch.load(os.path.join(OUT, f"ck_gated_occ_full_noLN_s{s}.pt"))); m.eval()
        X, Y, _ = gen.sample_seq(15000, T); dP, dC = gen.D[0], gen.D[1]
        bP = float(gen.b @ dP); bC = float(gen.b @ dC)
        fr = cp.frontiers(gen, X, Y, True)
        base_c, base_p = cf.mse(m, X, Y, dP, dC)
        span = fr["parent_blind"] - base_c
        dPt = torch.tensor(dP, dtype=torch.float32)
        Vsae = (dPt / dPt.norm())[:, None]
        V1 = cf.fit_probe(X, dP, bP, 1); V3 = cf.fit_probe(X, dP, bP, 3)
        methods = {"FRA_OV_child": lambda g: dict(ovchild_gain=g),
                   "FRA_OV_full": lambda g: dict(valproj_gain=g),
                   "DoM_rank1": lambda g: dict(resV=V1, res_gain=g),
                   "DoM_rank3": lambda g: dict(resV=V3, res_gain=g),
                   "SAE_planted": lambda g: dict(resV=Vsae, res_gain=g)}
        cross = {}
        for name, mk in methods.items():
            xs, ys = [], []
            for g in gains:
                c, p = cf.mse(m, X, Y, dP, dC, **mk(g))
                xs.append(p - base_p); ys.append((c - base_c) / span)
            cx = crossover(xs, ys)
            # FRA_OV_child may not quite cross 1 in range; extrapolate from last 2 monotone points
            if cx is None and name == "FRA_OV_child" and ys[-1] > 0.5:
                cx = xs[-2] + (1 - ys[-2]) / (ys[-1] - ys[-2]) * (xs[-1] - xs[-2])
            cross[name] = cx
        R["per_seed"][s] = dict(base_child=base_c, parent_blind=fr["parent_blind"], span=span,
                                base_parent=base_p, gating_value=fr["gating_value"],
                                b_dot_dP=bP, b_dot_dC=bC, crossover=cross)
        fchild = cross["FRA_OV_child"]; dom3 = cross["DoM_rank3"]; sae = cross["SAE_planted"]
        print(f"[seed {s}] b.dP={bP:+.3f} b.dC={bC:+.1e} | crossover collat @rem=1: "
              f"FRA_child={fchild:.5f} DoM3={dom3:.5f} SAE={sae:.5f} | "
              f"FRA<DoM3: {fchild < dom3}, ratio DoM3/FRA={dom3/fchild:.0f}x SAE/FRA={sae/fchild:.0f}x",
              flush=True)
        json.dump(R, open(os.path.join(OUT, "seed_frontier.json"), "w"), indent=2)

    # summary
    def col(name): return [R["per_seed"][s]["crossover"][name] for s in SEEDS]
    fc, d3, sa = col("FRA_OV_child"), col("DoM_rank3"), col("SAE_planted")
    ratios_d3 = [d3[i] / fc[i] for i in range(3)]; ratios_sae = [sa[i] / fc[i] for i in range(3)]
    R["summary"] = dict(
        FRA_child_mean=float(np.mean(fc)), FRA_child_spread=[float(min(fc)), float(max(fc))],
        DoM3_mean=float(np.mean(d3)), SAE_mean=float(np.mean(sa)),
        ratio_DoM3_over_FRA=[float(min(ratios_d3)), float(max(ratios_d3))],
        ratio_SAE_over_FRA=[float(min(ratios_sae)), float(max(ratios_sae))],
        FRA_beats_DoM3_all_seeds=bool(all(fc[i] < d3[i] for i in range(3))),
        FRA_beats_SAE_all_seeds=bool(all(fc[i] < sa[i] for i in range(3))),
        b_perp_dC_all_seeds=bool(all(abs(R["per_seed"][s]["b_dot_dC"]) < 1e-6 for s in SEEDS)),
        b_along_dP_all_seeds=bool(all(abs(R["per_seed"][s]["b_dot_dP"]) > 0.5 for s in SEEDS)))
    json.dump(R, open(os.path.join(OUT, "seed_frontier.json"), "w"), indent=2)
    print(f"SUMMARY: FRA_child collat={np.mean(fc):.5f} (spread {min(fc):.5f}-{max(fc):.5f}); "
          f"DoM3/FRA in [{min(ratios_d3):.0f},{max(ratios_d3):.0f}]x, SAE/FRA in "
          f"[{min(ratios_sae):.0f},{max(ratios_sae):.0f}]x; FRA beats DoM3 all: "
          f"{R['summary']['FRA_beats_DoM3_all_seeds']}, beats SAE all: {R['summary']['FRA_beats_SAE_all_seeds']}; "
          f"b_perp_dC_all={R['summary']['b_perp_dC_all_seeds']} b_along_dP_all={R['summary']['b_along_dP_all_seeds']}",
          flush=True)
    print("wrote out/seed_frontier.json", flush=True)


if __name__ == "__main__":
    main()
