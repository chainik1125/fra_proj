"""Centerpiece FINAL — the carrier decomposition of the hierarchy gate (theory-clean, no-LN).

The pre-registered expectation was that the targeted FRA-QK (child-query x parent-key) cut
would carry the gate (O(1), gauge-robust, ~Delta_gate). The carrier diagnostic overturned it:
the gate is carried by OV (values), not QK (keys). Here we make the quantitative decomposition:
gain sweep of the QK cut (project dP out of KEYS) and the OV cut (project dP out of VALUES),
child + parent effect, with the G1-dial band, on gated & null. Saves out/centerpiece_carrier.json
+ out/handle_final.png (the money figure).
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
import centerpiece as cp

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
PARAMS = cp.PARAMS; T = 24
gains = [0.0, 0.25, 0.5, 0.75, 1.0]
gammas = [-3.0, 0.0, 3.0]
LS = [0, 1]


def load(gated, cellname):
    gen = HierProcess(d=32, gated=gated, child_occlude=0.5, seed=0, **PARAMS)
    m = make_model(False, False, gen, seed=0, drop_ln=True)
    m.load_state_dict(torch.load(os.path.join(OUT, f"ck_{cellname}_full_noLN.pt"))); m.eval()
    return m, gen


def sweep(m, X, Y, dP, dC, kind, base_c, base_p, vg):
    out = {}
    for g in gains:
        cv, pv = [], []
        for gm in gammas:
            kw = {kind: (g, LS), "gamma": gm, "vgauge": vg}
            cc, pp = cp.mse_cd(m, X, Y, dP, dC, **kw)
            cv.append(cc - base_c); pv.append(pp - base_p)
        out[g] = dict(dchild_mean=float(np.mean(cv)), dchild_min=float(np.min(cv)),
                      dchild_max=float(np.max(cv)), dparent_mean=float(np.mean(pv)),
                      dchild_gaugestd=float(np.std(cv)))
    return out


def main():
    R = {"platform": "theory-clean: attention-only, NO LayerNorm (user directive 2026-07-17)",
         "finding": "the parent->child gate is carried by OV (values), NOT QK (keys); the targeted "
                    "FRA-QK cut is a gauge-robust NULL -> falsifies H2.3/H3.2 for the trained model",
         "cells": {}}
    for gated, cellname in [(True, "gated_occ"), (False, "null_occ")]:
        m, gen = load(gated, cellname)
        X, Y, _ = gen.sample_seq(20000, T); dP, dC = gen.D[0], gen.D[1]
        vg = dP.copy()
        fr = cp.frontiers(gen, X, Y, gated)
        base_c, base_p = cp.mse_cd(m, X, Y, dP, dC)
        qk = sweep(m, X, Y, dP, dC, "keyproj", base_c, base_p, vg)
        ov = sweep(m, X, Y, dP, dC, "valproj", base_c, base_p, vg)
        R["cells"][cellname] = dict(base_child=base_c, base_parent=base_p, frontiers=fr,
                                    qk_cut=qk, ov_cut=ov,
                                    qk_at1_dchild=qk[1.0]["dchild_mean"], ov_at1_dchild=ov[1.0]["dchild_mean"],
                                    qk_at1_dparent=qk[1.0]["dparent_mean"], ov_at1_dparent=ov[1.0]["dparent_mean"],
                                    qk_gauge_std=qk[1.0]["dchild_gaugestd"], ov_gauge_std=ov[1.0]["dchild_gaugestd"])
        print(f"[{cellname}] base_child={base_c:.5f} gating_value={fr['gating_value']:.4f} | "
              f"QK cut@1 dChild={qk[1.0]['dchild_mean']:+.5f}  OV cut@1 dChild={ov[1.0]['dchild_mean']:+.5f} "
              f"(OV/QK={ov[1.0]['dchild_mean']/max(qk[1.0]['dchild_mean'],1e-9):.0f}x)", flush=True)
        json.dump(R, open(os.path.join(OUT, "centerpiece_carrier.json"), "w"), indent=2)
    _figure(R)
    print("wrote out/centerpiece_carrier.json + out/handle_final.png", flush=True)


def _figure(R):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.7), sharey=True)
    for j, (cellname, C) in enumerate(R["cells"].items()):
        a = ax[j]
        gs = gains
        for kind, col, lab in [("qk_cut", "#8b949e", "FRA-QK cut (parent→keys)"),
                               ("ov_cut", "#1f6feb", "FRA-OV cut (parent→values)")]:
            sw = C[kind]
            mean = [sw[g]["dchild_mean"] for g in gs]
            lo = [sw[g]["dchild_min"] for g in gs]; hi = [sw[g]["dchild_max"] for g in gs]
            a.fill_between(gs, lo, hi, color=col, alpha=0.22)
            a.plot(gs, mean, "-o", color=col, lw=2.2, label=lab + " · Δchild")
        # OV parent collateral
        ovp = [C["ov_cut"][g]["dparent_mean"] for g in gs]
        a.plot(gs, ovp, "--s", color="#d29922", lw=1.4, ms=4, label="FRA-OV cut · Δparent (collateral)")
        gv = C["frontiers"]["gating_value"]
        a.axhline(gv, ls=":", color="#cf222e", lw=1.6, label=f"analytic gating value = {gv:.4f}")
        a.axhline(0, color="k", lw=0.6)
        a.set_title(cellname, fontsize=11)
        a.set_xlabel("cut gain  c")
        if j == 0:
            a.set_ylabel("Δ MSE  (cut − base)")
        a.legend(fontsize=7.6, loc="upper left")
        a.grid(alpha=0.25)
    fig.suptitle("H4 centerpiece (theory-clean, no-LN): the hierarchy gate is carried by OV, NOT QK.\n"
                 "Parent→child screening flows through VALUES (Δchild→gating value); the targeted "
                 "FRA-QK (parent-key) cut is a gauge-robust NULL. Falsifies H2.3/H3.2 for the trained model.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "handle_final.png"), dpi=130)


if __name__ == "__main__":
    main()
