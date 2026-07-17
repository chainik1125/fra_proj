"""
GHMM coding-mechanism control: does causal capability cross-talk track the GENERATIVE COUPLING,
or the GEOMETRIC FOOTPRINT of the learned alignment direction on the capability code?

The λ-sweep (exp_ghmm_sweep.py) showed that steering alignment causes ≈0 capability cross-talk even
at strong statistical coupling (corr(q,sharpness)→−0.88), because the model keeps the alignment
direction nearly ⊥ to the capability probe (footprint≈0). This experiment isolates WHY, with a clean
toggle that holds the generative coupling FIXED and changes only the alignment CODING MECHANISM:

  Both regimes use the SAME emission-sharpness asymmetry (aligned aA=0.92 sharp, misaligned aM=0.48
  diffuse) and the SAME (eps,gamma) — so corr(q, belief-sharpness) is ~identical (≈−0.85) in both.
    * DIR   (drift+sharpness): TrA=drift(+1), TrM=drift(-1).  Alignment is ALSO carried by the cycling
      DIRECTION, a code ~orthogonal to within-step sharpness. (= the λ=1 point of the sweep.)
    * NODIR (sharpness-only): TrA=TrM=symmetric mess3_trans (no drift). Alignment is carried ONLY by
      emission sharpness, so the model is FORCED to read it off the same axis as the capability belief.

Prediction (the factorization is a property of the CODE, not the coupling): DIR keeps footprint≈0 and
cross-talk≈0; NODIR has high footprint and high cross-talk — at the SAME generative coupling. Plus a
per-direction footprint→cross-talk scatter (d_align, randoms, and d_align→capability interpolations)
to show cross-talk is a monotonic function of footprint, with each regime's LEARNED d_align sitting at
opposite ends. Everything anchored to the exact 6-state filter (validated per regime).

Outputs results/ghmm_coding.pt
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

L = int(os.environ.get("CD_L", "96"))
STEPS = int(os.environ.get("CD_STEPS", "2000"))
BATCH = int(os.environ.get("CD_BATCH", "96"))
NEV = int(os.environ.get("CD_NEV", "2048"))
LR = 1.5e-3
EPS, GAMMA, STAY, AA, AM = 0.02, 0.06, 0.30, 0.92, 0.48
XSYM = (1.0 - STAY) / 2.0  # symmetric transition matching P(stay)=STAY: mess3_trans(x) has P(stay)=1-2x
SLAYER = int(os.environ.get("CD_SLAYER", "1"))
NSEED = int(os.environ.get("CD_NSEED", "3"))


def params(regime):
    EA = ghmm.mess3_emit(AA); EM = ghmm.mess3_emit(AM)
    if regime == "DIR":
        TrA = ghmm.mess3_trans_drift(STAY, +1); TrM = ghmm.mess3_trans_drift(STAY, -1)
    elif regime == "NODIR":
        Tr = ghmm.mess3_trans(XSYM); TrA = Tr; TrM = Tr
    else:
        raise ValueError(regime)
    return ghmm.joint_params_from(EPS, GAMMA, TrA, TrM, EA, EM,
                                  meta={"regime": regime, "aA": AA, "aM": AM})


def kl_cat(po, pm):
    e = 1e-7; po = np.clip(po, e, 1); pm = np.clip(pm, e, 1)
    return (po * np.log(po / pm)).sum(-1)


def grouped_ridge(R, y, seq_ids, alpha=1.0, train_frac=0.7, seed=0):
    R = np.asarray(R, np.float64); y = np.asarray(y, np.float64).ravel()
    uniq = np.unique(seq_ids); rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq); k = int(len(uniq) * train_frac)
    tr_seq = set(perm[:k].tolist())
    tr = np.array([sid in tr_seq for sid in seq_ids]); te = ~tr
    mu = R[tr].mean(0); sd = R[tr].std(0) + 1e-8
    Xtr = (R[tr] - mu) / sd; Xte = (R[te] - mu) / sd; ym = y[tr].mean()
    Amat = Xtr.T @ Xtr + alpha * np.eye(Xtr.shape[1])
    w = np.linalg.solve(Amat, Xtr.T @ (y[tr] - ym))
    pred = Xte @ w + ym
    ss = ((y[te] - pred) ** 2).sum(); tot = ((y[te] - y[te].mean()) ** 2).sum() + 1e-12
    return {"r2": float(1 - ss / tot), "direction": w / sd, "intercept": float(ym - mu @ (w / sd))}


def run(regime, seed=0, want_scatter=False, device="cpu"):
    torch.manual_seed(seed); rng = np.random.default_rng(seed); erng = np.random.default_rng(777)
    P = params(regime)
    obs_ev, c_ev, s_ev = ghmm.gen_ghmm(NEV, L, P, erng)
    belief_ev, q_ev, mbel_ev, nextp_ev = ghmm.forward_filter_ghmm(obs_ev, P)
    obs_ev_t = torch.tensor(obs_ev, device=device)
    z_ev = np.log(np.clip(q_ev, 1e-4, 1 - 1e-4) / np.clip(1 - q_ev, 1e-4, 1 - 1e-4))
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
                      snapshot_steps=[STEPS], eval_fn=eval_fn, log_every=STEPS)
    model.eval()
    kl = out["snapshots"][STEPS]["metrics"]["kl"]

    pos = list(range(20, L - 1, 4)); npos = len(pos); B = obs_ev.shape[0]
    seq_ids = np.repeat(np.arange(B), npos)
    qf = q_ev[:, pos].reshape(-1); maxc = mbel_ev[:, pos, :].max(-1).reshape(-1)
    corr = float(np.corrcoef(qf, maxc)[0, 1])
    zt = z_ev[:, pos].reshape(-1); mx = mxy_ev[:, pos, 0].reshape(-1); my = mxy_ev[:, pos, 1].reshape(-1)

    # z decodability (best layer) just to confirm alignment is decodable in both regimes
    zr2 = {}
    for layer in range(cfg.n_layers):
        Rl = probe.extract_resid(model, obs_ev_t[:, :-1], layer=layer, pos=pos)
        zr2[layer] = grouped_ridge(Rl, zt, seq_ids)["r2"]
    bz = max(zr2, key=lambda l: zr2[l]); z_r2 = zr2[bz]

    # steer-layer probes + d_align
    Rs = probe.extract_resid(model, obs_ev_t[:, :-1], layer=SLAYER, pos=pos)
    hi = qf > np.quantile(qf, 0.8); lo = qf < np.quantile(qf, 0.2)
    d_align = Rs[hi].mean(0) - Rs[lo].mean(0); d_align /= (np.linalg.norm(d_align) + 1e-9)
    pmx = grouped_ridge(Rs, mx, seq_ids); pmy = grouped_ridge(Rs, my, seq_ids)
    wmx, bx = pmx["direction"], pmx["intercept"]; wmy, by = pmy["direction"], pmy["intercept"]
    m_r2 = 0.5 * (pmx["r2"] + pmy["r2"])
    typ = float(np.linalg.norm(Rs, axis=1).mean()); ALPHA = 0.5 * typ

    def footprint(d):
        return 0.5 * (abs(d @ wmx) / (np.linalg.norm(wmx) + 1e-9) + abs(d @ wmy) / (np.linalg.norm(wmy) + 1e-9))

    def steer_resid(direction, alpha):
        d = torch.tensor(direction / (np.linalg.norm(direction) + 1e-9), dtype=torch.float32, device=device)
        with torch.no_grad():
            x = model._embed(obs_ev_t[:, :-1]); Rst = None
            for i, blk in enumerate(model.blocks):
                x = blk(x)
                if i == SLAYER:
                    mask = torch.zeros(x.shape[1], device=device); mask[torch.as_tensor(pos)] = 1.0
                    x = x + float(alpha) * d * mask[None, :, None]
                    Rst = x[:, pos, :].cpu().numpy().reshape(-1, x.shape[-1]); break
        return Rst

    def cap_clip(Rst):
        def r2(pred, tgt):
            ss = ((tgt - pred) ** 2).sum(); tot = ((tgt - tgt.mean()) ** 2).sum() + 1e-12
            return max(0.0, float(1 - ss / tot))
        return 0.5 * (r2(Rst @ wmx + bx, mx) + r2(Rst @ wmy + by, my))

    cap0 = cap_clip(steer_resid(d_align, 0.0))
    fp_align = float(footprint(d_align))
    drop_align = cap0 - cap_clip(steer_resid(d_align, ALPHA))
    # random null
    rgc = np.random.default_rng(99 + seed); rdrop = []; rfp = []
    for _ in range(8):
        dr = rgc.standard_normal(d_align.shape); dr = dr - (dr @ d_align) * d_align
        dr /= (np.linalg.norm(dr) + 1e-9)
        rdrop.append(cap0 - cap_clip(steer_resid(dr, ALPHA))); rfp.append(float(footprint(dr)))

    res = {"regime": regime, "seed": seed, "kl": kl, "corr_q_sharp": corr,
           "z_r2": z_r2, "m_r2": m_r2, "best_z_layer": bz, "steer_layer": SLAYER,
           "footprint_align": fp_align, "cap0": cap0, "cap_drop_align": drop_align,
           "cap_drop_rand_mean": float(np.mean(rdrop)), "cap_drop_rand_sd": float(np.std(rdrop)),
           "footprint_rand_mean": float(np.mean(rfp)), "train_min": (time.time() - t0) / 60}

    if want_scatter:
        # per-direction (footprint, cap_drop): d_align, randoms, and d_align->capability interpolations
        ucap = wmx / (np.linalg.norm(wmx) + 1e-9)  # a capability-probe unit direction (high footprint)
        pts = [("align", d_align), ("cap", ucap)]
        rg2 = np.random.default_rng(7 + seed)
        for _ in range(30):
            u = rg2.standard_normal(d_align.shape); u /= np.linalg.norm(u); pts.append(("rand", u))
        for t in np.linspace(0.05, 0.95, 12):
            di = (1 - t) * d_align + t * ucap; di /= (np.linalg.norm(di) + 1e-9); pts.append(("interp", di))
        scat = []
        for kind, d in pts:
            scat.append({"kind": kind, "footprint": float(footprint(d)),
                         "cap_drop": float(cap0 - cap_clip(steer_resid(d, ALPHA)))})
        res["scatter"] = scat

    print(f"  [{regime} s{seed}] KL={kl:.4f} corr(q,sharp)={corr:+.3f} | zR²={z_r2:.3f} mR²={m_r2:.3f} | "
          f"footprint d_align={fp_align:.4f} (rand {res['footprint_rand_mean']:.3f}) | "
          f"capDrop align={drop_align:+.3f} rand={res['cap_drop_rand_mean']:+.3f}±{res['cap_drop_rand_sd']:.3f} "
          f"({res['train_min']:.1f}m)", flush=True)
    return res


def main():
    print(f"=== GHMM coding-mechanism control: DIR vs NODIR x {NSEED} seeds "
          f"(same emission asymmetry aA={AA}/aM={AM}; L={L},steps={STEPS}) ===", flush=True)
    all_res = []
    for regime in ["DIR", "NODIR"]:
        for s in range(NSEED):
            all_res.append(run(regime, seed=s, want_scatter=(s == 0)))
    # aggregate
    agg = {}
    for regime in ["DIR", "NODIR"]:
        rs = [r for r in all_res if r["regime"] == regime]
        def col(k): return np.array([r[k] for r in rs], float)
        agg[regime] = {k: (float(col(k).mean()), float(col(k).std()))
                       for k in ["corr_q_sharp", "z_r2", "m_r2", "footprint_align",
                                 "cap_drop_align", "cap_drop_rand_mean", "footprint_rand_mean"]}
    torch.save({"rows": all_res, "agg": agg,
                "config": {"L": L, "STEPS": STEPS, "NEV": NEV, "nseed": NSEED, "eps": EPS, "gamma": GAMMA,
                           "stay": STAY, "aA": AA, "aM": AM, "slayer": SLAYER}}, f"{OUT}/ghmm_coding.pt")
    print(f"saved {OUT}/ghmm_coding.pt", flush=True)
    print("\n=== aggregate (mean±sd) — SAME generative coupling, opposite cross-talk ===")
    for regime in ["DIR", "NODIR"]:
        a = agg[regime]
        print(f"  {regime:5s}: corr(q,sharp)={a['corr_q_sharp'][0]:+.3f}±{a['corr_q_sharp'][1]:.3f}  "
              f"zR²={a['z_r2'][0]:.3f}  mR²={a['m_r2'][0]:.3f}  "
              f"footprint(d_align)={a['footprint_align'][0]:.4f}±{a['footprint_align'][1]:.4f}  "
              f"capDrop_align={a['cap_drop_align'][0]:+.3f}±{a['cap_drop_align'][1]:.3f}  "
              f"(rand {a['cap_drop_rand_mean'][0]:+.3f})")


if __name__ == "__main__":
    main()
