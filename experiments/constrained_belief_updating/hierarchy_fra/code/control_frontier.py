"""Capstone — the CONTROL FRONTIER on the LN-free gated_occ platform.

Question (user, verbatim): can FRA-informed edits beat the DoM and SAE-only baselines for
control at given collateral? Control target = remove the model's USE of the parent->child gate:
drive child MSE from base (~0.2440) toward the parent-blind Bayes floor (~0.2546). Removal
fraction = (child - base)/(parent_blind - base). Collateral = parent-MSE rise (+ excess child
damage beyond the parent-blind floor, reported separately).

Three methods, each strength-swept:
  (a) FRA carrier cut  : centerpiece located the gate in the VALUE path. Targeted OV-route cut =
      remove the parent-value contribution that flows into the CHILD-readout direction
      r_C = W_out^T d_C (per layer, per source). Spares parent self-prediction by construction.
      Reference: FRA-OV-full = project d_P out of all values (non-targeted).
  (b) DoM projection   : ridge-probe the parent-belief direction (residual -> z_P) and project it
      out of the residual stream at both block inputs. rank-1 (z_P) and rank-3 (z_P(t),(t-1),(t-2)).
  (c) SAE ablation     : planted code -> project the true d_P channel out of the residual (best
      case an SAE could do). (Trained TopK SAE deferred; planted is the best-case anchor.)

Deliverables: out/control_frontier.png, out/control_frontier.json.
"""
import sys, os, json, numpy as np, torch
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "bernoulli_fra", "code"))
sys.path.insert(0, os.path.dirname(__file__))
from hier_data import HierProcess
from rung2_lncheck import make_model
import centerpiece as cp

OUT = os.path.join(os.path.dirname(__file__), "..", "out")
PARAMS = cp.PARAMS; T = 24
torch.set_num_threads(4)


def forward(m, X, dP, dC, resV=None, res_gain=0.0, ovchild_gain=0.0, valproj_gain=0.0,
            layers=(0, 1), mean_ablate=True):
    """Remove-parent edits. mean_ablate=True: subtract the batch mean of the parent DoF (keep the
    bias) so no method is penalised for the seed's d_P-aligned bias. mean_ablate=False: zero-ablate
    (subtract the full DoF) — the naive convention, used to check the convention isn't load-bearing."""
    x = torch.tensor(X, dtype=torch.float32) if not torch.is_tensor(X) else X
    dPt = torch.tensor(dP, dtype=torch.float32); dCt = torch.tensor(dC, dtype=torch.float32)
    rC = dCt @ m.Wout; rC = rC / rC.norm()                # child-readout direction (residual space)
    Tn = x.shape[1]; sd = m.d ** 0.5
    ctr = (lambda z: z - z.mean(dim=(0, 1), keepdim=True)) if mean_ablate else (lambda z: z)
    ctr1 = (lambda z: z - z.mean()) if mean_ablate else (lambda z: z)
    for li, blk in enumerate(m.blocks):
        if resV is not None and li in layers and res_gain != 0.0:   # DoM/SAE: ablate subspace
            proj = x @ resV                               # (B,T,k)
            x = x - res_gain * (ctr(proj) @ resV.T)
        h = blk.ln(x)                                     # identity (no-LN)
        q = h @ blk.Wq + blk.bq
        k = h @ blk.Wk + blk.KP[:Tn]
        sc = torch.einsum('btd,bsd->bts', q, k) / sd
        sc = sc.masked_fill(torch.triu(torch.ones(Tn, Tn, dtype=torch.bool), 1), float('-inf'))
        A = torch.softmax(sc, -1)
        hv = h
        if valproj_gain != 0.0 and li in layers:
            hp = h @ dPt
            hv = h - valproj_gain * (ctr1(hp)[..., None] * dPt)
        ctx = torch.einsum('bts,bsd->btd', A, (hv @ blk.Wv) @ blk.Wo.T)
        if ovchild_gain != 0.0 and li in layers:
            wP = dPt @ blk.Wv @ blk.Wo.T                  # parent-value write direction
            pbar = torch.einsum('bts,bs->bt', A, (h @ dPt))   # attn-weighted parent DoF (content+bias)
            coef = float(wP @ rC)
            ctx = ctx - ovchild_gain * coef * ctr1(pbar)[:, :, None] * rC[None, None, :]
        x = x + ctx
    return x @ m.Wout.T + m.beta


@torch.no_grad()
def mse(m, X, Y, dP, dC, **kw):
    pred = forward(m, X, dP, dC, **kw).numpy()
    return (float((((pred - Y) @ dC) ** 2).mean()), float((((pred - Y) @ dP) ** 2).mean()))


def fit_probe(X, dP, bP, k):
    """Difference-of-means (DoM) parent-belief direction(s). rank-1: DoM for z_P(t). rank-3: DoM for
    z_P(t), z_P(t-1), z_P(t-2) (the parent-survival window), orthonormalized. Returns V (d,k)."""
    zP = ((X @ dP - bP) > 0.5).astype(np.float32)         # (B,T) clean parent readout
    Xf = X.reshape(-1, X.shape[-1])
    cols = []
    for j in range(k):
        yj = np.roll(zP, j, axis=1); yj[:, :j] = 0.0
        yj = yj.reshape(-1)
        d = Xf[yj > 0.5].mean(0) - Xf[yj < 0.5].mean(0)   # difference of means
        cols.append(d)
    W = np.stack(cols, 1)                                 # (d,k)
    Q, _ = np.linalg.qr(W)
    return torch.tensor(Q[:, :k], dtype=torch.float32)


def main():
    gated = True
    gen = HierProcess(d=32, gated=gated, child_occlude=0.5, seed=0, **PARAMS)
    m = make_model(False, False, gen, seed=0, drop_ln=True)
    m.load_state_dict(torch.load(os.path.join(OUT, "ck_gated_occ_full_noLN.pt"))); m.eval()
    X, Y, _ = gen.sample_seq(15000, T); dP, dC = gen.D[0], gen.D[1]
    bP = float(gen.b @ dP)
    fr = cp.frontiers(gen, X, Y, gated)
    base_c, base_p = mse(m, X, Y, dP, dC)
    pblind = fr["parent_blind"]
    span = pblind - base_c                                # the model's gate-USE span
    print(f"anchors: base_child={base_c:.5f} parent_blind={pblind:.5f} span={span:.5f} "
          f"base_parent={base_p:.5f} gating_value={fr['gating_value']:.4f}", flush=True)

    dPt = torch.tensor(dP, dtype=torch.float32)
    V_saeplanted = (dPt / dPt.norm())[:, None]            # planted SAE: true d_P channel
    V_dom1 = fit_probe(X, dP, bP, 1)
    V_dom3 = fit_probe(X, dP, bP, 3)

    gains = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
    methods = {
        "FRA_OV_child (targeted)": lambda g: dict(ovchild_gain=g),
        "FRA_OV_full (values)":    lambda g: dict(valproj_gain=g),
        "DoM_rank1":               lambda g: dict(resV=V_dom1, res_gain=g),
        "DoM_rank3":               lambda g: dict(resV=V_dom3, res_gain=g),
        "SAE_planted (d_P)":       lambda g: dict(resV=V_saeplanted, res_gain=g),
    }
    R = {"platform": "LN-free gated_occ", "anchors": dict(base_child=base_c, parent_blind=pblind,
         base_parent=base_p, span=span, gating_value=fr["gating_value"]), "gains": gains, "methods": {}}
    for name, mk in methods.items():
        pts = []
        for g in gains:
            c, p = mse(m, X, Y, dP, dC, **mk(g))
            pts.append(dict(gain=g, child=c, parent=p,
                            removal_frac=(c - base_c) / span,
                            collateral_parent=p - base_p,
                            excess_child=max(0.0, c - pblind)))
        R["methods"][name] = pts
        at1 = pts[4]  # gain=1.0
        print(f"[{name:24s}] @g=1: removal={at1['removal_frac']:.2f} "
              f"parent_collat={at1['collateral_parent']:+.4f} excess_child={at1['excess_child']:.4f}", flush=True)
        json.dump(R, open(os.path.join(OUT, "control_frontier.json"), "w"), indent=2)

    _figure(R)
    json.dump(R, open(os.path.join(OUT, "control_frontier.json"), "w"), indent=2)
    print("wrote out/control_frontier.json + out/control_frontier.png", flush=True)


def _figure(R):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7.6, 5.6))
    styles = {"FRA_OV_child (targeted)": ("#1f6feb", "-o", 2.6),
              "FRA_OV_full (values)": ("#54aeff", "--o", 1.6),
              "DoM_rank1": ("#cf222e", "-s", 1.8), "DoM_rank3": ("#a40e26", "--s", 1.8),
              "SAE_planted (d_P)": ("#8250df", "-^", 1.8)}
    for name, pts in R["methods"].items():
        col, ls, lw = styles[name]
        xs = [p["collateral_parent"] for p in pts]; ys = [p["removal_frac"] for p in pts]
        ax.plot(xs, ys, ls, color=col, lw=lw, ms=5, label=name)
    ax.axhline(1.0, ls=":", color="#57606a", lw=1.2, label="full gate-use removed (child = parent-blind)")
    ax.axhline(0, color="k", lw=0.6); ax.axvline(0, color="k", lw=0.6)
    ax.set_xlabel("parent-prediction collateral  (Δ parent MSE)")
    ax.set_ylabel("gate-use removed  (child MSE → parent-blind floor)")
    ax.set_title("Control frontier (LN-free): remove the parent→child gate-use vs parent collateral\n"
                 "FRA carrier cut (value-path targeted) vs DoM projection vs planted-SAE ablation", fontsize=10)
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.25)
    ax.set_xlim(left=-0.005)
    fig.tight_layout(); fig.savefig(os.path.join(OUT, "control_frontier.png"), dpi=130)


if __name__ == "__main__":
    main()
