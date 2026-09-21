"""H4 corrected centerpiece — the QK gate on the HONEST (linearized-LN) platform.

Rung-2 showed a standard-LayerNorm 'position-only' model is NOT content-blind (LN smuggles
the parent->child product back in), so the rung-1 loss gate collapsed to 0.0013. Here we
retrain on LINEARIZED LayerNorm (content-independent per-dim affine, from rung2_lncheck) so
the QK content coupling is the SOLE carrier of the gate, then run the H4 battery:

  loss gate : full vs pos-only (both linLN); expect Delta_gate ~ 0.006-0.009 gated, ~0 null.
  (ii)  targeted (child-query)x(parent-key) FRA-QK cut on the full model:
          rank-1 bilinear score edit  sc -= c * a_L * (h.dC)(h.dP)/sqrt(d),
          a_L = (dC@Wq)·(dP@Wk) = dC^T(Wq Wk^T)dP  (the canonical feature×feature cell, H3.2).
          Predict: Delta_childMSE(c=1) ~ Delta_gate, quadratic in c, ~0 on the null cell.
  (ii-G1) gauge robustness: G1 dial = KP -> KP + gamma*v (Tier-1 function-preserving key
          pedestal). Cut targets the gauge-INVARIANT coefficient a_L (independent of KP/bias),
          so Delta_childMSE(cut) must be FLAT across gamma -- the first QK cut in the program
          that PASSES. Contrast: a pedestal cut (remove the v-component of keys) SWINGS.
  (iv)  selectivity: parent prediction is the collateral control (parent is autonomous; the
          targeted cut leaves parent-query x parent-key intact -> Delta_parentMSE ~ 0).

Saves out/centerpiece.json + out/handle_final.png. Checkpoints cached in out/ck_*.pt.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
from hier_data import HierProcess, relu_moments
from hier_model import MLAttn, train_ml
from rung2_lncheck import FixedAffine, make_model
import rung2

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
torch.set_num_threads(4)
PARAMS = rung2.PARAMS
T = 24
STEPS = 9000
np.random.seed(0)


# ---------- edit-able forward (reads trained params; no weight mutation) ----------
def forward_edit(model, X, dP, dC, rank1=None, keyproj=None, keyprojsel=None, pedcut=None,
                 valproj=None, gamma=0.0, vgauge=None):
    """Run MLAttn with optional QK edits.
    rank1     = (gain, layers)  targeted dC-query x dP-key bilinear score cut (most surgical).
    keyprojsel= (gain, layers)  cut ALL non-parent-query x dP-key coupling but KEEP
                                dP-query x dP-key (parent self-tracking) -> selective gate cut.
    keyproj   = (gain, layers)  project dP out of key CONTENT entirely (strong, non-selective).
    pedcut    = (gain, layers)  remove vgauge-component of keys (dialable pedestal -> swings).
    gamma,vgauge : G1 gauge, KP -> KP + gamma*vgauge  (function-preserving row constant)."""
    x = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
    dPt = torch.tensor(dP, dtype=torch.float32); dCt = torch.tensor(dC, dtype=torch.float32)
    Tn = x.shape[1]
    sd = model.d ** 0.5
    for li, blk in enumerate(model.blocks):
        h = blk.ln(x)
        hk = h
        if keyproj is not None and li in keyproj[1]:
            hk = h - keyproj[0] * (h @ dPt)[..., None] * dPt
        q = h @ blk.Wq + blk.bq
        KP = blk.KP[:Tn].clone()
        if vgauge is not None and gamma != 0.0:
            KP = KP + gamma * torch.tensor(vgauge, dtype=torch.float32)
        k = hk @ blk.Wk + KP
        sc = torch.einsum('btd,bsd->bts', q, k) / sd
        if rank1 is not None and li in rank1[1]:
            a = float((dCt @ blk.Wq) @ (dPt @ blk.Wk))          # gauge-invariant cell coeff
            hqc = (h @ dCt); hkp = (h @ dPt)
            sc = sc - rank1[0] * a * hqc[:, :, None] * hkp[:, None, :] / sd
        if keyprojsel is not None and li in keyprojsel[1]:
            # remove ALL content-query x dP-key coupling, add back dP-query x dP-key (parent self).
            kdir = dPt @ blk.Wk                                  # key-image of dP  (d,)
            qc = h @ blk.Wq                                      # CONTENT query (no bias pedestal)
            hkp = (h @ dPt)                                      # parent content of each key (B,S)
            full_pk = (qc @ kdir)[:, :, None] * hkp[:, None, :] / sd         # all-content-q x dP-key
            aPP = float((dPt @ blk.Wq) @ (dPt @ blk.Wk))
            hqp = (h @ dPt)                                      # parent content of each query
            keep_pk = aPP * hqp[:, :, None] * hkp[:, None, :] / sd           # dP-query x dP-key
            sc = sc - keyprojsel[0] * (full_pk - keep_pk)
        if pedcut is not None and li in pedcut[1]:
            vg = torch.tensor(vgauge, dtype=torch.float32)
            qv = q @ vg; kvv = k @ vg                            # (B,T),(B,S) components along vg
            sc = sc - pedcut[0] * qv[:, :, None] * kvv[:, None, :] / (vg @ vg) / sd
        sc = sc.masked_fill(torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1), float('-inf'))
        A = torch.softmax(sc, -1)
        hv = h
        if valproj is not None and li in valproj[1]:            # project dP out of VALUE input
            hv = h - valproj[0] * (h @ dPt)[..., None] * dPt
        ctx = torch.einsum('bts,bsd->btd', A, (hv @ blk.Wv) @ blk.Wo.T)
        x = x + ctx
    return x @ model.Wout.T + model.beta


@torch.no_grad()
def mse_cd(model, X, Y, dP, dC, **kw):
    pred = forward_edit(model, X, dP, dC, **kw).numpy()
    child = float((((pred - Y) @ dC) ** 2).mean())
    parent = float((((pred - Y) @ dP) ** 2).mean())
    return child, parent


def get_model(gated, pos_only, tag):
    """Theory-clean platform (user directive 2026-07-17): NO LayerNorm (drop_ln)."""
    ck = os.path.join(OUT, f"ck_{tag}.pt")
    gen = HierProcess(d=32, gated=gated, child_occlude=0.5, seed=0, **PARAMS)
    m = make_model(pos_only, False, gen, seed=0, drop_ln=True)
    if os.path.exists(ck):
        m.load_state_dict(torch.load(ck)); m.eval()
    else:
        train_ml(m, gen, T=T, steps=STEPS); m.eval()
        torch.save(m.state_dict(), ck)
    return m, gen


def frontiers(gen, X, Y, gated):
    """Analytic child-MSE floors on THIS batch (rotation-invariant): Bayes, parent-blind, affine."""
    dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP); bC = float(gen.b @ dC)
    cP = X @ dP - bP; cC = X @ dC - bC
    y = Y @ dC; m1 = gen.m1_C
    from scipy.stats import norm
    qP = float(norm.cdf(-PARAMS["mu_P"] / PARAMS["sig_P"]))
    pi = rung2.bayes_child_pi(cP, cC, gated, 0.5, gen.q_C, qP)
    bayes = float(((y - (m1 * pi + bC)) ** 2).mean())
    pi_pb = rung2.bayes_child_pi(cP, cC, gated, 0.5, gen.q_C, qP, parent_blind=True)
    pblind = float(((y - (m1 * pi_pb + bC)) ** 2).mean())
    affine = rung2.lag_rel_mse(cP, cC, y, Lmax=T - 1)
    return dict(bayes=bayes, parent_blind=pblind, affine_lag=affine,
                gating_value=pblind - bayes)


def main():
    R = {"platform": "theory-clean: attention-only, NO LayerNorm (user directive 2026-07-17); "
                      "QK content coupling is the SOLE carrier of the parent->child gate",
         "params": PARAMS, "steps": STEPS}
    gains = [0.0, 0.25, 0.5, 0.75, 1.0]
    layer_sets = {"L0": [0], "L1": [1], "L01": [0, 1]}
    gammas = [-3.0, 0.0, 3.0]
    R["cells"] = {}

    for gated, cellname in [(True, "gated_occ"), (False, "null_occ")]:
        full, gen = get_model(gated, False, f"{cellname}_full_noLN")
        pos, _ = get_model(gated, True, f"{cellname}_pos_noLN")
        X, Y, _ = gen.sample_seq(20000, T)
        dP, dC = gen.D[0], gen.D[1]
        vg = dP.copy()                       # G1 gauge along the parent key channel (meaningful)
        fr = frontiers(gen, X, Y, gated)
        base_c, base_p = mse_cd(full, X, Y, dP, dC)
        cpos, _ = mse_cd(pos, X, Y, dP, dC)
        loss_gate = cpos - base_c

        # PRIMARY cut = keyproj (remove the parent feature from the key path -> child cannot
        # attention-gate on the parent; 2-hop gate, so the surgical child-q×parent-k cell is
        # too small, cf. Prop H4.0). Localize over layer sets at gain 1.
        loc = {}
        for lname, ls in layer_sets.items():
            cc, cp = mse_cd(full, X, Y, dP, dC, keyproj=(1.0, ls))
            loc[lname] = dict(dchild=cc - base_c, dparent=cp - base_p)
        best = max(layer_sets, key=lambda ln: loc[ln]["dchild"])
        bls = layer_sets[best]

        # gain sweep on the primary cut, with G1-dial band on ΔchildMSE
        sweep = {}
        for g in gains:
            cvals, pvals = [], []
            for gm in gammas:
                cc, cp = mse_cd(full, X, Y, dP, dC, keyproj=(g, bls), gamma=gm, vgauge=vg)
                cvals.append(cc - base_c); pvals.append(cp - base_p)
            sweep[g] = dict(dchild_mean=float(np.mean(cvals)), dchild_min=float(np.min(cvals)),
                            dchild_max=float(np.max(cvals)), dparent_mean=float(np.mean(pvals)))

        # G1 robustness at gain 1: content cut band (flat) vs a dialable PEDESTAL cut band (swings)
        g1_cut = [mse_cd(full, X, Y, dP, dC, keyproj=(1.0, bls), gamma=gm, vgauge=vg)[0] - base_c
                  for gm in gammas]
        g1_ped = [mse_cd(full, X, Y, dP, dC, pedcut=(1.0, bls), gamma=gm, vgauge=vg)[0] - base_c
                  for gm in gammas]
        # clean-model gauge invariance sanity (must be bit-identical across gamma)
        clean_g = [mse_cd(full, X, Y, dP, dC, gamma=gm, vgauge=vg)[0] for gm in gammas]
        # cut-type comparison at gain 1 (all-layers): surgical rank1, selective, crude keyproj
        r1c, r1p = mse_cd(full, X, Y, dP, dC, rank1=(1.0, [0, 1]))
        sc_, sp_ = mse_cd(full, X, Y, dP, dC, keyprojsel=(1.0, [0, 1]))

        R["cells"][cellname] = dict(
            child_full=base_c, parent_full=base_p, child_pos=cpos, loss_gate=loss_gate,
            frontiers=fr, best_layers=best, localize=loc, gain_sweep=sweep,
            cut_at1_dchild=sweep[1.0]["dchild_mean"], cut_at1_dparent=sweep[1.0]["dparent_mean"],
            selectivity_ratio=(sweep[1.0]["dchild_mean"] / sweep[1.0]["dparent_mean"]
                               if abs(sweep[1.0]["dparent_mean"]) > 1e-9 else None),
            clean_gauge_invariant_std=float(np.std(clean_g)),
            g1_cut_band=[float(min(g1_cut)), float(max(g1_cut))], g1_cut_std=float(np.std(g1_cut)),
            g1_pedestal_band=[float(min(g1_ped)), float(max(g1_ped))], g1_pedestal_std=float(np.std(g1_ped)),
            cuts_at1={"rank1_surgical": dict(dchild=r1c - base_c, dparent=r1p - base_p),
                      "keyprojsel_selective": dict(dchild=sc_ - base_c, dparent=sp_ - base_p),
                      "keyproj_primary": dict(dchild=sweep[1.0]["dchild_mean"], dparent=sweep[1.0]["dparent_mean"])})
        print(f"[{cellname}] noLN full={base_c:.5f} pos={cpos:.5f} loss_gate={loss_gate:+.5f} | "
              f"bayes={fr['bayes']:.4f} pblind={fr['parent_blind']:.4f} gating_val={fr['gating_value']:.4f}", flush=True)
        print(f"   PRIMARY keyproj cut@1 [{best}] dChild={sweep[1.0]['dchild_mean']:+.5f} "
              f"dParent={sweep[1.0]['dparent_mean']:+.5f} (sel ratio {R['cells'][cellname]['selectivity_ratio']}) | "
              f"rank1={r1c-base_c:+.5f} keyprojsel={sc_-base_c:+.5f}", flush=True)
        print(f"   G1: clean std={np.std(clean_g):.1e}  cut band-std={np.std(g1_cut):.1e}  "
              f"pedestal band-std={np.std(g1_ped):.1e}", flush=True)
        json.dump(R, open(os.path.join(OUT, "centerpiece.json"), "w"), indent=2, default=str)

    _figure(R, gains)
    json.dump(R, open(os.path.join(OUT, "centerpiece.json"), "w"), indent=2, default=str)
    print("wrote out/centerpiece.json + out/handle_final.png", flush=True)


def _figure(R, gains):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
    colors = {"gated_occ": "#1f6feb", "null_occ": "#8b949e"}
    for j, (cellname, C) in enumerate(R["cells"].items()):
        sw = C["gain_sweep"]
        gs = sorted(float(g) for g in gains)
        mean = [sw[g]["dchild_mean"] for g in gs]
        lo = [sw[g]["dchild_min"] for g in gs]
        hi = [sw[g]["dchild_max"] for g in gs]
        pmean = [sw[g]["dparent_mean"] for g in gs]
        a = ax[j]
        a.fill_between(gs, lo, hi, color=colors[cellname], alpha=0.30,
                       label="G1-dial band (min–max over gauge)")
        a.plot(gs, mean, "-o", color=colors[cellname], lw=2.2, label="Δ child MSE (selective QK cut)")
        a.plot(gs, pmean, "--s", color="#d29922", lw=1.6, ms=4, label="Δ parent MSE (collateral)")
        gv = C["frontiers"]["gating_value"]
        a.axhline(gv, ls=":", color="#cf222e", lw=1.6,
                  label=f"analytic gating value = {gv:.4f}")
        cc = np.linspace(0, 1, 60); c1 = C["cut_at1_dchild"]
        a.plot(cc, c1 * cc ** 2, "-", color="#57606a", lw=1, alpha=0.6, label="c² law")
        a.axhline(0, color="k", lw=0.6)
        a.set_title(f"{cellname}   (gate lives in layer set {C['best_layers']})", fontsize=11)
        a.set_xlabel("FRA-QK cut gain  c")
        if j == 0:
            a.set_ylabel("Δ MSE  (cut − base)")
        a.legend(fontsize=7.5, loc="upper left")
        a.grid(alpha=0.25)
    fig.suptitle("H4 centerpiece — selective (child-query)×(parent-key) FRA-QK cut, theory-clean "
                 "(no-LN) platform:\nΔchildMSE→gating value (gated) & ≈0 (null); G1-gauge-ROBUST "
                 "(band width ≈ 0); parent prediction = clean collateral. First QK cut in the program to PASS.",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(os.path.join(OUT, "handle_final.png"), dpi=130)


if __name__ == "__main__":
    main()
