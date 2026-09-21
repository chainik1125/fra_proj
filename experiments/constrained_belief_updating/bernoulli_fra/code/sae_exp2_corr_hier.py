"""Experiment 2 — correlated (copula) and hierarchical variants.

Literature prediction (hedging / absorption, SS + Chanin): the learned chi picks up
SPECIFIC off-diagonal patterns and severability fails for the affected features
BEFORE MCC degrades much. Dictionary kept ORTHOGONAL (rho_mm~0) so any off-diagonal
chi is attributable to correlation/hierarchy, not superposition.

Two configs:
  (A) copula correlation, moderate scale, rank-4 factor.
  (B) 2-level hierarchy: 6 parent->child pairs gated, rest independent.
"""
import json, time, sys, os
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
from sae_common import CorrHierGen, train_sae, recovery_report, chi_matrix, mcc_hungarian

OUT = os.path.join(os.path.dirname(__file__), "..", "out", "sae_exp2_corr_hier.json")
STEPS = 4000
NSAMP = 150_000


def run_corr():
    N, d = 24, 24
    gen = CorrHierGen(N, d, seed=1, p=0.08, mu=1.0, sigma=0.5, orthogonalize=True,
                      corr_rank=4, corr_scale=0.7)
    acts, c = gen.sample_a(NSAMP)
    z = (c > 0).astype(float)
    # empirical firing correlation among features (the thing hedging should track)
    Zc = z - z.mean(0)
    corr = (Zc.T @ Zc) / len(z)
    dsd = np.sqrt(np.clip(np.diag(corr), 1e-9, None))
    corr = corr / np.outer(dsd, dsd)
    K = max(1, int(round(z.sum(1).mean())))
    sae = train_sae(acts, N, K, steps=STEPS, batch=4096, seed=0)
    rep = recovery_report(sae, gen.D, acts, z)
    chi, _ = chi_matrix(sae.W_dec.detach().numpy(), gen.D)
    li, gi = rep["_li"], rep["_gi"]
    # For each matched latent -> gt i, off-diagonal chi mass and whether it lands on
    # the most-correlated partner j of i.
    hedges = []
    for lat, i in zip(li, gi):
        row = chi[lat].copy()
        row_i = row[i]
        row[i] = 0
        j = int(np.argmax(np.abs(row)))                 # biggest off-diag partner
        jcorr = float(corr[i, j])                       # its GT firing correlation
        # partner that i is MOST correlated with (excluding self)
        cc = corr[i].copy(); cc[i] = 0
        jmax = int(np.argmax(np.abs(cc)))
        hedges.append(dict(gt=int(i), diag=float(row_i), max_off=float(row[j]),
                           off_partner=j, off_partner_corr=jcorr,
                           corr_partner=jmax, corr_partner_val=float(cc[jmax]),
                           chi_on_corr_partner=float(chi[lat, jmax])))
    # aggregate: correlation between |off-diag chi| and GT firing correlation
    offmass = np.array([abs(h["max_off"]) for h in hedges])
    partcorr = np.array([abs(h["off_partner_corr"]) for h in hedges])
    align = float(np.mean([h["off_partner"] == h["corr_partner"] for h in hedges]))
    rep_clean = {k: v for k, v in rep.items() if not k.startswith("_")}
    return dict(cfg=dict(N=N, d=d, K=K, corr_rank=4, corr_scale=0.7),
                report=rep_clean, hedges=hedges,
                hedge_lands_on_corr_partner_frac=align,
                mean_offdiag_mass=float(offmass.mean()),
                corr_offmass_vs_partcorr=float(np.corrcoef(offmass, partcorr)[0, 1]))


def run_hier():
    N, d = 24, 24
    # features 0..5 parents, 6..11 children (child c=6+k gated by parent k),
    # 12..23 independent. Parents/children higher p so children are learnable.
    parents = np.full(N, -1)
    for k in range(6):
        parents[6 + k] = k
    p = np.full(N, 0.08)
    p[:6] = 0.25          # parents fire often
    p[6:12] = 0.5         # child base (effective ~0.125 after gating)
    gen = CorrHierGen(N, d, seed=2, p=p, mu=1.0, sigma=0.5, orthogonalize=True,
                      parents=parents)
    acts, c = gen.sample_a(NSAMP)
    z = (c > 0).astype(float)
    K = max(1, int(round(z.sum(1).mean())))
    sae = train_sae(acts, N, K, steps=STEPS, batch=4096, seed=0)
    rep = recovery_report(sae, gen.D, acts, z)
    chi, _ = chi_matrix(sae.W_dec.detach().numpy(), gen.D)
    sev = rep["_sev"]
    li, gi = rep["_li"], rep["_gi"]
    lat_of_gt = {int(g): int(l) for l, g in zip(li, gi)}
    # absorption: child latent's chi mass on its PARENT feature
    rows = []
    for k in range(6):
        child = 6 + k
        par = k
        lat = lat_of_gt.get(child, None)
        if lat is None:
            continue
        rows.append(dict(child=child, parent=par,
                         chi_child_on_parent=float(chi[lat, par]),
                         chi_child_on_child=float(chi[lat, child]),
                         sev_child=float(sev[child]), sev_parent=float(sev[par])))
    # group severability: children vs parents vs independents
    child_idx = list(range(6, 12)); par_idx = list(range(6)); ind_idx = list(range(12, 24))
    rep_clean = {k: v for k, v in rep.items() if not k.startswith("_")}
    return dict(cfg=dict(N=N, d=d, K=K, hierarchy="6 parent->child pairs"),
                report=rep_clean, absorption=rows,
                sev_children_mean=float(sev[child_idx].mean()),
                sev_parents_mean=float(sev[par_idx].mean()),
                sev_independent_mean=float(sev[ind_idx].mean()),
                mean_child_on_parent=float(np.mean([r["chi_child_on_parent"] for r in rows])))


def run():
    t0 = time.time()
    out = {}
    print("== correlation ==")
    out["corr"] = run_corr()
    c = out["corr"]
    print(f"  MCC={c['report']['mcc']:.3f} F1={c['report']['f1']:.3f} "
          f"hedge_on_corr_partner={c['hedge_lands_on_corr_partner_frac']:.2f} "
          f"mean_offdiag={c['mean_offdiag_mass']:.3f} "
          f"corr(offmass,partnercorr)={c['corr_offmass_vs_partcorr']:.2f}  ({time.time()-t0:.0f}s)")
    print("== hierarchy ==")
    out["hier"] = run_hier()
    h = out["hier"]
    print(f"  MCC={h['report']['mcc']:.3f} F1={h['report']['f1']:.3f} "
          f"child_on_parent={h['mean_child_on_parent']:.3f}  "
          f"sev child/par/ind = {h['sev_children_mean']:.3f}/"
          f"{h['sev_parents_mean']:.3f}/{h['sev_independent_mean']:.3f}  ({time.time()-t0:.0f}s)")
    with open(OUT, "w") as f:
        json.dump(out, f, indent=2)
    print("wrote", OUT)


if __name__ == "__main__":
    run()
