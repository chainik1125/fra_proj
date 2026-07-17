"""
GHMM factored-representation experiment (drift regime, leakage-hardened).

Process: a hidden alignment state c in {A,M} switches via (eps,gamma) and sets the DRIFT
DIRECTION of a 3-state nonunifilar content cycle (aligned cycles +1, misaligned cycles -1)
with MATCHED stickiness `stay` and MATCHED emission concentration `a`. So the within-step
belief sharpness is ~alignment-symmetric (the two latents are only weakly correlated), and
alignment is carried by the cycling DIRECTION of the symbol stream. The exact 6-state filter
yields two latents: alignment log-odds z=logit P(M|obs) and the Mess3 content belief simplex.

We ask whether a TinyGPT trained on next-symbol prediction represents BOTH and how separable
the two codes are -- the "persona (x) capability" question -- with every metric checked
against the right null and all probe splits grouped BY SEQUENCE (no temporal leakage).

Tests
  T0  next-symbol KL to the exact 6-state Bayes filter.
  C   latent-latent correlation corr(z, belief sharpness) in the DATA (reported up front;
      the regime is "near-factored", not perfectly independent).
  T1  linear decodability of z and the Mess3 belief (seq-grouped held-out R²) vs last-symbol
      / sliding-count controls.
  T2  subspace separability: |proj(d_z onto Mess3-plane)| vs a random-direction null floor;
      Mess3 R² after removing a multi-dim z-subspace (q-bin means); z R² after removing the
      Mess3-plane.
  T3  causal factorisation: steer +/- the alignment diff-of-means direction (matched-norm) and
      read (i) ALIGNMENT = predicted drift score s_up-prob - s_down-prob (flips sign) and
      (ii) CAPABILITY = Mess3 belief decoded FROM the steered residual stays at the true
      (unsteered) belief (R² preserved, small simplex displacement). Control = a matched-norm
      direction orthogonal to d_align (random-in-orthocomplement).
  T4  cross-alignment transfer of the Mess3 belief probe across DISJOINT sequences
      (aligned-context tokens from one sequence pool -> misaligned-context tokens from another).

Outputs results/ghmm_factored.pt
"""
import os, sys, time
import numpy as np
import torch
ROOT = os.environ.get("BAG_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
from bag_moments import ghmm, probe, train
from bag_moments.model import GPTConfig, TinyGPT

OUT = os.environ.get("BAG_OUT", os.path.join(ROOT, "results"))
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(int(os.environ.get("BAG_THREADS", "4")))

L = int(os.environ.get("GHMM_L", "128"))
STEPS = int(os.environ.get("GHMM_STEPS", "3000"))
BATCH = int(os.environ.get("GHMM_BATCH", "96"))
LR = 1.5e-3
REGIME = os.environ.get("GHMM_REGIME", "drift")
if REGIME == "drift":
    # near-factored: alignment sets content-cycle DIRECTION; matched sharpness (corr~-0.13)
    EPS, GAMMA, STAY, A = 0.02, 0.06, 0.30, float(os.environ.get("GHMM_A", "0.80"))
    P = ghmm.joint_params_drift(EPS, GAMMA, STAY, A)
    REGIME_PARAMS = {"eps": EPS, "gamma": GAMMA, "stay": STAY, "a": A}
elif REGIME == "entangled":
    # sharpness-modulated CONTRAST: alignment sets transition rate (sticky vs erratic) so belief
    # sharpness depends strongly on alignment (corr~-0.8); matched longer dwell for a comparable
    # z signal. This is the control regime where the two latents are generatively entangled.
    EPS, GAMMA, XA, XM, AA, AM = 0.02, 0.06, 0.02, 0.55, 0.60, 0.60
    P = ghmm.joint_params(EPS, GAMMA, XA, XM, AA, AM)
    REGIME_PARAMS = {"eps": EPS, "gamma": GAMMA, "xA": XA, "xM": XM, "aA": AA, "aM": AM}
else:
    raise ValueError(f"unknown GHMM_REGIME={REGIME}")


def kl_cat(po, pm):
    e = 1e-7; po = np.clip(po, e, 1); pm = np.clip(pm, e, 1)
    return (po * np.log(po / pm)).sum(-1)


def grouped_ridge(R, y, seq_ids, alpha=1.0, train_frac=0.7, seed=0):
    """Ridge with a split that groups BY SEQUENCE (no within-sequence leakage)."""
    R = np.asarray(R, np.float64); y = np.asarray(y, np.float64).ravel()
    uniq = np.unique(seq_ids); rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq); k = int(len(uniq) * train_frac)
    tr_seq = set(perm[:k].tolist())
    tr = np.array([sid in tr_seq for sid in seq_ids]); te = ~tr
    mu = R[tr].mean(0); sd = R[tr].std(0) + 1e-8
    Xtr = (R[tr]-mu)/sd; Xte = (R[te]-mu)/sd; ym = y[tr].mean()
    Amat = Xtr.T@Xtr + alpha*np.eye(Xtr.shape[1])
    w = np.linalg.solve(Amat, Xtr.T@(y[tr]-ym))
    pred = Xte@w + ym
    ss = ((y[te]-pred)**2).sum(); tot = ((y[te]-y[te].mean())**2).sum()+1e-12
    direction = w/sd; intercept = float(ym - mu@direction)
    return {"r2": float(1-ss/tot), "direction": direction, "intercept": intercept, "tr": tr, "te": te}


def run(seed=0, device="cpu", verbose=True):
    torch.manual_seed(seed); rng = np.random.default_rng(seed)
    erng = np.random.default_rng(777)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(int(os.environ.get("GHMM_NEV","3072")), L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    obs_ev_t = torch.tensor(obs_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1-1e-4) / np.clip(1-q_ev, 1e-4, 1-1e-4))
    mxy_ev = ghmm.simplex_xy(mbel_ev)

    def eval_fn(model):
        model.eval()
        with torch.no_grad():
            p = torch.softmax(model(obs_ev_t[:, :-1]), -1).cpu().numpy()
        return {"kl": float(kl_cat(nextp_ev[:, :-1, :], p)[:, 20:].mean())}

    def batch_fn():
        o, _, _ = ghmm.gen_ghmm(BATCH, L, P, rng)
        t = torch.tensor(o, device=device); return t[:, :-1], t[:, 1:]

    cfg = GPTConfig(vocab_size=3, n_ctx=L, d_model=128, n_heads=4, n_layers=3, d_mlp=512)
    model = TinyGPT(cfg)
    t0 = time.time()
    out = train.train(model, batch_fn, steps=STEPS, lr=LR, device=device,
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=max(STEPS//3, 1))
    model.eval()
    kl = out["snapshots"][STEPS]["metrics"]["kl"]
    if verbose:
        print(f"[seed {seed}] next-symbol KL to 6-state Bayes filter = {kl:.5f} "
              f"({(time.time()-t0)/60:.1f} min)", flush=True)

    pos = list(range(20, L-1, 4)); npos = len(pos)
    B = obs_ev.shape[0]
    seq_ids = np.repeat(np.arange(B), npos)  # sequence id per flattened row
    res = {"seed": seed, "kl": kl,
           "config": {"L": L, "steps": STEPS, "batch": BATCH, "regime": REGIME, **REGIME_PARAMS}}

    # ---- C: latent-latent correlation in the data ----
    qf = q_ev[:, pos].reshape(-1); maxc = mbel_ev[:, pos, :].max(-1).reshape(-1)
    bent = (-(np.clip(mbel_ev, 1e-9, 1)*np.log(np.clip(mbel_ev, 1e-9, 1))).sum(-1))[:, pos].reshape(-1)
    res["latent_corr"] = {"corr_q_maxcoord": float(np.corrcoef(qf, maxc)[0, 1]),
                          "corr_q_entropy": float(np.corrcoef(qf, bent)[0, 1])}
    if verbose:
        print(f"  [C] data corr(q, belief maxcoord)={res['latent_corr']['corr_q_maxcoord']:+.3f} "
              f"(near-factored; |corr|<<0.8 of the sharpness-modulated regime)", flush=True)

    zt = z_ev[:, pos].reshape(-1)
    mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    # ---- T1: decodability (seq-grouped) ----
    decode = {}
    for layer in range(cfg.n_layers):
        R = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        pz = grouped_ridge(R, zt, seq_ids); pmx = grouped_ridge(R, mx, seq_ids); pmy = grouped_ridge(R, my, seq_ids)
        decode[layer] = {"z_r2": pz["r2"], "m_r2": 0.5*(pmx["r2"]+pmy["r2"]),
                         "mx_r2": pmx["r2"], "my_r2": pmy["r2"],
                         "_pz": pz, "_pmx": pmx, "_pmy": pmy, "_R": R}
    bz = max(decode, key=lambda l: decode[l]["z_r2"]); bm = max(decode, key=lambda l: decode[l]["m_r2"])
    res["decode"] = {l: {k: v for k, v in d.items() if not k.startswith("_")} for l, d in decode.items()}
    res["best_z_layer"] = bz; res["best_m_layer"] = bm
    if verbose:
        for l in range(cfg.n_layers):
            d = decode[l]; print(f"  layer {l}: z R²={d['z_r2']:.3f}  Mess3 R²={d['m_r2']:.3f}", flush=True)

    # controls (seq-grouped)
    W = 12; obs_in = obs_ev[:, :-1]
    last_oh = np.eye(3)[obs_in[:, pos].reshape(-1)]
    cnts = np.zeros((B, npos, 3))
    for k in range(3):
        cs = np.cumsum((obs_in == k).astype(float), axis=1)
        for j, pp in enumerate(pos):
            lo = max(0, pp-W); cnts[:, j, k] = cs[:, pp] - (cs[:, lo-1] if lo > 0 else 0)
    cnts = cnts.reshape(-1, 3)
    ctrl = {"z_lastsym": grouped_ridge(last_oh, zt, seq_ids)["r2"],
            "z_count": grouped_ridge(cnts, zt, seq_ids)["r2"],
            "m_lastsym": 0.5*(grouped_ridge(last_oh, mx, seq_ids)["r2"]+grouped_ridge(last_oh, my, seq_ids)["r2"]),
            "m_count": 0.5*(grouped_ridge(cnts, mx, seq_ids)["r2"]+grouped_ridge(cnts, my, seq_ids)["r2"])}
    res["controls"] = ctrl
    if verbose:
        print(f"  controls: z(last={ctrl['z_lastsym']:.3f},count={ctrl['z_count']:.3f}) "
              f"M(last={ctrl['m_lastsym']:.3f},count={ctrl['m_count']:.3f})", flush=True)

    # ---- T2: separability with proper nulls ----
    layer = bm; R = decode[layer]["_R"]
    dz = decode[layer]["_pz"]["direction"]; dz = dz/(np.linalg.norm(dz)+1e-9)
    dmx = decode[layer]["_pmx"]["direction"]; dmy = decode[layer]["_pmy"]["direction"]
    Qm, _ = np.linalg.qr(np.stack([dmx, dmy], axis=1))
    cos_in_plane = float(np.linalg.norm(Qm @ (Qm.T @ dz)))
    # random null floor for a unit vector's projection onto a 2-plane in R^d
    rg = np.random.default_rng(7); floors = []
    for _ in range(200):
        v = rg.standard_normal(dz.shape); v /= np.linalg.norm(v)
        floors.append(np.linalg.norm(Qm @ (Qm.T @ v)))
    null_floor = float(np.mean(floors))
    # multi-dim z-subspace from q-bin means (<=4 dims), remove it, test Mess3 survival
    qb = np.clip((qf*5).astype(int), 0, 4)  # 5 bins
    means = np.stack([R[qb == b].mean(0) for b in range(5) if (qb == b).sum() > 50], axis=0)
    means = means - R.mean(0)
    Uz, _, _ = np.linalg.svd(means.T, full_matrices=False)  # (d, k)
    Zsub = Uz[:, :min(3, Uz.shape[1])]
    R_noZ = R - (R @ Zsub) @ Zsub.T
    m_after = 0.5*(grouped_ridge(R_noZ, mx, seq_ids)["r2"] + grouped_ridge(R_noZ, my, seq_ids)["r2"])
    R_noM = R - (R @ Qm) @ Qm.T
    z_after = grouped_ridge(R_noM, zt, seq_ids)["r2"]
    res["separability"] = {"layer": layer, "cos_dz_in_Mplane": cos_in_plane, "null_floor": null_floor,
                           "m_r2_after_remove_zsub": m_after, "z_r2_after_remove_Mplane": z_after,
                           "zsub_dim": int(Zsub.shape[1]),
                           "m_r2_full": decode[layer]["m_r2"], "z_r2_full": decode[layer]["z_r2"]}
    if verbose:
        print(f"  [T2] layer {layer}: |proj(d_z on Mess3-plane)|={cos_in_plane:.3f} vs random null {null_floor:.3f}; "
              f"Mess3 R² after removing {Zsub.shape[1]}-dim z-subspace: {m_after:.3f} (full {decode[layer]['m_r2']:.3f}); "
              f"z R² after removing Mess3-plane: {z_after:.3f} (full {decode[layer]['z_r2']:.3f})", flush=True)

    # ---- T3: causal factorisation via steering ----
    # Steer at an EARLY layer with good z signal so there is downstream compute. We READ the
    # residual AT the steer layer (same layer the probes were fitted on -- so cap_r2 at alpha=0
    # equals the in-sample probe fit, a sane baseline). Compare steering the ALIGNMENT direction
    # (d_align) vs a matched-norm CAPABILITY-subspace direction (d_cap, a unit vector in the
    # Mess3 belief plane with the d_align component removed). Factorisation predicts: steering
    # d_align moves the alignment readout (z) but PRESERVES the capability readout (Mess3 R²),
    # while steering d_cap destroys the capability readout but barely moves z.
    cand = [l for l in range(cfg.n_layers - 1) if decode[l]["z_r2"] > 0.5]
    slayer = cand[0] if cand else max(0, bz - 1)
    Rs = probe.extract_resid(model, obs_ev_t[:, :-1], layer=slayer, pos=pos)
    qpos = q_ev[:, pos].reshape(-1)
    hi = qpos > np.quantile(qpos, 0.8); lo = qpos < np.quantile(qpos, 0.2)
    d_align = Rs[hi].mean(0) - Rs[lo].mean(0); d_align /= (np.linalg.norm(d_align)+1e-9)

    shat = np.argmax(mbel_ev[:, pos, :], -1).reshape(-1)
    up = (shat+1) % 3; dn = (shat-1) % 3
    # probes at the steer layer: capability (mx,my) and alignment (z)
    pmx_s = grouped_ridge(Rs, mx, seq_ids); pmy_s = grouped_ridge(Rs, my, seq_ids)
    pz_s = grouped_ridge(Rs, zt, seq_ids)
    wmx, bx = pmx_s["direction"], pmx_s["intercept"]
    wmy, by = pmy_s["direction"], pmy_s["intercept"]
    wz, bz_ = pz_s["direction"], pz_s["intercept"]
    # capability-subspace control direction: a unit vector IN the Mess3 belief plane at this
    # layer (orthonormal basis from the mx,my probe directions), with the d_align component
    # removed so it is a "pure capability" steer matched in norm to d_align.
    Qcap, _ = np.linalg.qr(np.stack([pmx_s["direction"], pmy_s["direction"]], axis=1))
    d_cap = Qcap[:, 0].copy()
    d_cap = d_cap - (d_cap @ d_align)*d_align; d_cap /= (np.linalg.norm(d_cap)+1e-9)
    # second control: a RANDOM direction orthogonal to d_align (matched norm). This tests whether
    # alignment steering is *special* or whether any equal-norm perturbation preserves capability.
    rgc = np.random.default_rng(99); d_rand = rgc.standard_normal(d_align.shape)
    d_rand = d_rand - (d_rand @ d_align)*d_align; d_rand /= (np.linalg.norm(d_rand)+1e-9)
    typ = float(np.linalg.norm(Rs, axis=1).mean()); scale = typ/6.0

    def steer_all(direction, alpha):
        d = torch.tensor(direction/(np.linalg.norm(direction)+1e-9), dtype=torch.float32, device=device)
        with torch.no_grad():
            logits = model.forward_with_steer(obs_ev_t[:, :-1], layer=slayer, direction=d,
                                              alpha=float(alpha), positions=pos)
            p = torch.softmax(logits[:, pos, :], -1).cpu().numpy().reshape(-1, 3)
            # capture the residual AT the steer layer (right after the steering edit), so the
            # layer-`slayer` probes are applied to layer-`slayer` activations (no layer mismatch).
            x = model._embed(obs_ev_t[:, :-1])
            Rst = None
            for i, blk in enumerate(model.blocks):
                x = blk(x)
                if i == slayer:
                    mask = torch.zeros(x.shape[1], device=device); mask[torch.as_tensor(pos)] = 1.0
                    x = x + float(alpha)*d*mask[None, :, None]
                    Rst = x[:, pos, :].cpu().numpy().reshape(-1, x.shape[-1])
                    break
        drift = float((p[np.arange(len(up)), up] - p[np.arange(len(dn)), dn]).mean())
        z_read = float((Rst @ wz + bz_).mean())  # alignment readout from steered residual
        def r2(pred, tgt):
            ss = ((tgt-pred)**2).sum(); tot = ((tgt-tgt.mean())**2).sum()+1e-12; return float(1-ss/tot)
        cap_r2 = 0.5*(r2(Rst @ wmx + bx, mx) + r2(Rst @ wmy + by, my))
        return drift, z_read, cap_r2

    alphas = [-6, -4, -2, 0, 2, 4, 6]
    steer = {"alphas": alphas, "scale": scale, "slayer": slayer, "align": {}, "ctrl": {}, "rand": {},
             "ctrl_kind": "capability-subspace (Mess3 plane)", "rand_kind": "random orthogonal to d_align"}
    for a in alphas:
        for nm, dirv in [("align", d_align), ("ctrl", d_cap), ("rand", d_rand)]:
            drift, z_read, cap_r2 = steer_all(dirv, a*scale)
            steer[nm][a] = {"drift_score": drift, "z_read": z_read, "cap_r2": cap_r2}
    res["steer"] = steer
    if verbose:
        print(f"  [T3] steering at layer {slayer}; controls = capability-subspace & random-⊥ "
              f"(z_read / drift / capR²):", flush=True)
        for a in alphas:
            al = steer["align"][a]; cl = steer["ctrl"][a]; rl = steer["rand"][a]
            print(f"      a={a:+d}: align z={al['z_read']:+.2f} drift={al['drift_score']:+.3f} capR²={al['cap_r2']:7.3f}"
                  f" | capCtrl capR²={cl['cap_r2']:9.3f} | rand⊥ capR²={rl['cap_r2']:7.3f}", flush=True)

    # ---- T4: cross-alignment transfer across DISJOINT sequences ----
    layer = bm; R = decode[layer]["_R"]
    cflat = c_ev[:, pos].reshape(-1)
    rgt = np.random.default_rng(5); seqperm = rgt.permutation(B)
    poolA = set(seqperm[:B//2].tolist())  # sequences whose aligned tokens are train
    in_poolA = np.array([sid in poolA for sid in seq_ids])
    tr_mask = in_poolA & (cflat == 0)        # aligned-context tokens from pool A
    te_mask = (~in_poolA) & (cflat == 1)     # misaligned-context tokens from pool B (disjoint seqs)
    def fit_eval(Xtr, ytr, Xte, yte):
        mu = Xtr.mean(0); sd = Xtr.std(0)+1e-8
        Xtr2 = (Xtr-mu)/sd; Xte2 = (Xte-mu)/sd
        Amat = Xtr2.T@Xtr2 + 1.0*np.eye(Xtr2.shape[1]); ym = ytr.mean()
        w = np.linalg.solve(Amat, Xtr2.T@(ytr-ym)); pred = Xte2@w+ym
        ss = ((yte-pred)**2).sum(); tot = ((yte-yte.mean())**2).sum()+1e-12; return float(1-ss/tot)
    transfer = {nm: fit_eval(R[tr_mask], yv[tr_mask], R[te_mask], yv[te_mask])
                for nm, yv in [("mx", mx), ("my", my)]}
    res["transfer"] = {"aligned_seqs->misaligned_seqs_mx": transfer["mx"],
                       "aligned_seqs->misaligned_seqs_my": transfer["my"],
                       "in_context_m_r2": decode[bm]["m_r2"]}
    if verbose:
        print(f"  [T4] Mess3 probe transfer (disjoint seqs) aligned→misaligned: "
              f"mx={transfer['mx']:.3f}, my={transfer['my']:.3f} (in-context {decode[bm]['m_r2']:.3f})", flush=True)
    # free big arrays
    for l in decode:
        decode[l].pop("_R", None); decode[l].pop("_pz", None); decode[l].pop("_pmx", None); decode[l].pop("_pmy", None)
    return res, model


def main():
    nseed = int(os.environ.get("GHMM_NSEED", "1"))
    all_res = [run(seed=sd)[0] for sd in range(nseed)]
    res = all_res[0]
    if nseed > 1:
        def col(fn): return np.array([fn(r) for r in all_res], float)
        agg = {}
        agg["kl"] = (float(col(lambda r: r["kl"]).mean()), float(col(lambda r: r["kl"]).std()))
        agg["z_r2"] = (float(col(lambda r: r["decode"][r["best_z_layer"]]["z_r2"]).mean()),
                       float(col(lambda r: r["decode"][r["best_z_layer"]]["z_r2"]).std()))
        agg["m_r2"] = (float(col(lambda r: r["decode"][r["best_m_layer"]]["m_r2"]).mean()),
                       float(col(lambda r: r["decode"][r["best_m_layer"]]["m_r2"]).std()))
        agg["cos_dz_in_Mplane"] = (float(col(lambda r: r["separability"]["cos_dz_in_Mplane"]).mean()),
                                   float(col(lambda r: r["separability"]["cos_dz_in_Mplane"]).std()))
        agg["null_floor"] = (float(col(lambda r: r["separability"]["null_floor"]).mean()), 0.0)
        agg["m_r2_after_remove_zsub"] = (float(col(lambda r: r["separability"]["m_r2_after_remove_zsub"]).mean()),
                                         float(col(lambda r: r["separability"]["m_r2_after_remove_zsub"]).std()))
        agg["transfer_mx"] = (float(col(lambda r: r["transfer"]["aligned_seqs->misaligned_seqs_mx"]).mean()),
                              float(col(lambda r: r["transfer"]["aligned_seqs->misaligned_seqs_mx"]).std()))
        agg["corr_q_maxcoord"] = (float(col(lambda r: r["latent_corr"]["corr_q_maxcoord"]).mean()), 0.0)
        res["multiseed"] = {"seeds": list(range(nseed)), **agg}
        res["all_seeds_res"] = all_res
        print("\n=== GHMM multi-seed aggregate (mean±sd) ===")
        for k, v in agg.items():
            print(f"  {k}: {v[0]:.4f} ± {v[1]:.4f}")
    tag = os.environ.get("GHMM_TAG", "")
    fname = ("ghmm_factored.pt" if REGIME == "drift" else f"ghmm_factored_{REGIME}.pt") if not tag else f"ghmm_factored_{tag}.pt"
    torch.save(res, f"{OUT}/{fname}")
    print(f"saved {OUT}/{fname}")


if __name__ == "__main__":
    main()
