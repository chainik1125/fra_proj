"""Seed replication of the centerpiece headline (the pre-registration reversal):
the parent->child gate is carried by OV, not QK. Train gated_occ full (no-LN) at 3 seeds
(0 cached), measure the QK-cut vs OV-cut child effect + G1 gauge-flatness on each, and report
the QK<<OV asymmetry across seeds. Saves out/seed_replicate.json.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
from hier_model import train_ml
import centerpiece as cp

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = cp.PARAMS; T = 24; SEEDS = [0, 1, 2]
gammas = [-3.0, 0.0, 3.0]; LS = [0, 1]


def get(seed):
    gen = HierProcess(d=32, gated=True, child_occlude=0.5, seed=seed, **PARAMS)
    m = make_model(False, False, gen, seed=seed, drop_ln=True)
    ck = os.path.join(OUT, f"ck_gated_occ_full_noLN_s{seed}.pt")
    legacy = os.path.join(OUT, "ck_gated_occ_full_noLN.pt")
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck))
    elif seed == 0 and os.path.exists(legacy):
        m.load_state_dict(torch.load(legacy)); torch.save(m.state_dict(), ck)
    else:
        train_ml(m, gen, T=T, steps=cp.STEPS); torch.save(m.state_dict(), ck)
    m.eval(); return m, gen


def carrier(m, gen):
    X, Y, _ = gen.sample_seq(20000, T); dP, dC = gen.D[0], gen.D[1]
    vg = dP.copy()
    fr = cp.frontiers(gen, X, Y, True)
    base_c, base_p = cp.mse_cd(m, X, Y, dP, dC)
    qk = [cp.mse_cd(m, X, Y, dP, dC, keyproj=(1.0, LS), gamma=g, vgauge=vg)[0] - base_c for g in gammas]
    ov = [cp.mse_cd(m, X, Y, dP, dC, valproj=(1.0, LS), gamma=g, vgauge=vg)[0] - base_c for g in gammas]
    return dict(base_child=base_c, gating_value=fr["gating_value"],
                qk_dchild=float(np.mean(qk)), ov_dchild=float(np.mean(ov)),
                ov_over_qk=float(np.mean(ov) / max(np.mean(qk), 1e-9)),
                qk_gauge_std=float(np.std(qk)), ov_gauge_std=float(np.std(ov)))


def main():
    R = {"platform": "no-LN gated_occ full", "seeds": SEEDS, "per_seed": {}}
    for s in SEEDS:
        m, gen = get(s)
        r = carrier(m, gen); R["per_seed"][s] = r
        print(f"[seed {s}] QK dChild={r['qk_dchild']:+.5f} OV dChild={r['ov_dchild']:+.5f} "
              f"OV/QK={r['ov_over_qk']:.0f}x  gauge_std(QK,OV)=({r['qk_gauge_std']:.1e},{r['ov_gauge_std']:.1e})",
              flush=True)
        json.dump(R, open(os.path.join(OUT, "seed_replicate.json"), "w"), indent=2)
    qk = [R["per_seed"][s]["qk_dchild"] for s in SEEDS]
    ov = [R["per_seed"][s]["ov_dchild"] for s in SEEDS]
    ratio = [R["per_seed"][s]["ov_over_qk"] for s in SEEDS]
    R["summary"] = dict(qk_mean=float(np.mean(qk)), qk_std=float(np.std(qk)),
                        ov_mean=float(np.mean(ov)), ov_std=float(np.std(ov)),
                        ratio_min=float(np.min(ratio)), ratio_max=float(np.max(ratio)),
                        gauge_flat_all=bool(max(R["per_seed"][s]["qk_gauge_std"] for s in SEEDS) < 1e-6
                                            and max(R["per_seed"][s]["ov_gauge_std"] for s in SEEDS) < 1e-6))
    json.dump(R, open(os.path.join(OUT, "seed_replicate.json"), "w"), indent=2)
    print(f"SUMMARY: QK={np.mean(qk):.5f}±{np.std(qk):.5f}  OV={np.mean(ov):.5f}±{np.std(ov):.5f}  "
          f"OV/QK in [{min(ratio):.0f},{max(ratio):.0f}]x  gauge_flat_all={R['summary']['gauge_flat_all']}",
          flush=True)
    print("wrote out/seed_replicate.json", flush=True)


if __name__ == "__main__":
    main()
